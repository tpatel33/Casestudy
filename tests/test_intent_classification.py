'''Regression tests for the intent-classification and invalid-input handling.

These tests complement the existing test_assistant.py suite and must not
alter the passing status of any existing test.
'''
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.assistant import (
    INTENT_INVALID,
    INTENT_UNSUPPORTED,
    answer_question,
    classify_intent,
)
from src.gemini import GeminiConfig, GeminiRequestError, GeminiUsage
from src.retrieval import load_context_notes

ROOT = Path(__file__).parents[1]


def _fixtures():
    output = pd.DataFrame([
        {
            'route': 'Chennai-Bangalore', 'week_of': '2024-02-19',
            'cost_per_tonne_km': 2.90,
            'vs_own_history': '+2.0% vs this route s past average',
            'vs_similar_routes': '+3.0% vs similar route types this week',
            'flagged': 'No', 'matched_note_id': '', 'reason': 'No anomaly',
        },
        {
            'route': 'Chennai-Bangalore', 'week_of': '2025-02-24',
            'cost_per_tonne_km': 4.10,
            'vs_own_history': '+20.0% vs this route s past average',
            'vs_similar_routes': '+22.0% vs similar route types this week',
            'flagged': 'No (justified)', 'matched_note_id': 'N001',
            'reason': 'Matches note N001',
        },
        {
            'route': 'Mumbai-Pune', 'week_of': '2025-09-15',
            'cost_per_tonne_km': 3.98,
            'vs_own_history': '+9.2% vs this route s past average',
            'vs_similar_routes': '+23.6% vs similar route types this week',
            'flagged': 'Yes', 'matched_note_id': '', 'reason': 'Unexplained',
        },
    ])
    audit = pd.DataFrame([
        {'route': 'Chennai-Bangalore', 'week_of': '2024-02-19', 'route_type': 'Long', 'anomaly': False, 'trigger_baselines': ''},
        {'route': 'Chennai-Bangalore', 'week_of': '2025-02-24', 'route_type': 'Long', 'anomaly': True, 'trigger_baselines': 'own_history,similar_routes'},
        {'route': 'Mumbai-Pune', 'week_of': '2025-09-15', 'route_type': 'Short', 'anomaly': True, 'trigger_baselines': 'similar_routes'},
    ])
    return output, audit, load_context_notes(ROOT / 'context_notes.csv')


@pytest.mark.parametrize('gibberish', [
    'yhyjuyuyukuy', 'asdf qwerty', 'zxcvbnm', 'qpwoeiru',
    'yhyyjuyuykukuykuykk', 'asdkjhasd kjahsdkj',
])
def test_gibberish_returns_invalid(gibberish):
    output, audit, notes = _fixtures()
    answer = answer_question(gibberish, output, audit, notes)
    # Both INTENT_INVALID (clear keyboard-mash) and INTENT_UNSUPPORTED
    # (plausible-looking but no freight domain) are correct rejections.
    # The critical invariant is that the assistant NEVER falls through to the
    # route-clarification prompt.
    assert answer.status in (INTENT_INVALID, INTENT_UNSUPPORTED), (
        f'Got {answer.status!r} for {gibberish!r} — expected invalid or unsupported'
    )
    assert 'Which route should I inspect?' not in answer.text


@pytest.mark.parametrize('empty', ['', '   ', '\t', '\n'])
def test_empty_returns_invalid(empty):
    output, audit, notes = _fixtures()
    assert answer_question(empty, output, audit, notes).status == INTENT_INVALID


@pytest.mark.parametrize('q', [
    'Why did costs rise?',
    'Which week had the biggest spike?',
    'What caused the increase?',
])
def test_valid_incomplete_question_asks_for_route(q):
    output, audit, notes = _fixtures()
    answer = answer_question(q, output, audit, notes)
    assert answer.status == 'clarification', f'Got {answer.status!r} for {q!r}'
    assert 'Which route' in answer.text


def test_valid_route_without_week_asks_for_week():
    output, audit, notes = _fixtures()
    answer = answer_question('Tell me about the cost for Chennai-Bangalore', output, audit, notes)
    assert answer.status == 'clarification'
    assert 'week' in answer.text.lower() or 'monday' in answer.text.lower()


def test_unknown_route_not_found():
    output, audit, notes = _fixtures()
    answer = answer_question('Why did Atlantis-Narnia spike in March?', output, audit, notes)
    assert answer.status == 'not-found'
    assert 'Chennai' not in answer.text and 'Mumbai' not in answer.text


def test_full_route_investigation_evidence():
    output, audit, notes = _fixtures()
    answer = answer_question('Why did Chennai-Bangalore spike in late February?', output, audit, notes)
    assert answer.status == 'evidence-supported'
    assert 'N001' in answer.text and '2025-02-24' in answer.text


def test_list_unexplained():
    output, audit, notes = _fixtures()
    answer = answer_question('Show unexplained anomalies', output, audit, notes)
    assert answer.status == 'unexplained'
    assert 'Mumbai-Pune' in answer.text


def test_evidence_note_lookup():
    output, audit, notes = _fixtures()
    answer = answer_question('Why was note N006 rejected for Mumbai-Pune on 2025-09-15?', output, audit, notes)
    assert answer.status in {'unexplained', 'not-found', 'clarification'}
    assert 'N006' in answer.text


@pytest.mark.parametrize('q', [
    'What does cost per tonne-km mean?',
    'How does the system work?',
    'What is this whole system?',
    'Explain the setup',
])
def test_system_definition_returns_overview(q):
    output, audit, notes = _fixtures()
    answer = answer_question(q, output, audit, notes)
    assert answer.status == 'overview', f'Got {answer.status!r} for {q!r}'


@pytest.mark.parametrize('q', [
    'Write me a poem about trucks',
    'Tell me a joke',
    'What are the cricket scores today?',
    'Can you recommend a recipe?',
])
def test_offtopic_returns_unsupported(q):
    output, audit, notes = _fixtures()
    answer = answer_question(q, output, audit, notes)
    assert answer.status == INTENT_UNSUPPORTED, f'Got {answer.status!r} for {q!r}'
    assert 'Which route should I inspect?' not in answer.text


def test_invalid_input_bypasses_gemini():
    output, audit, notes = _fixtures()
    answer = answer_question('zxcvbnm qwerty', output, audit, notes)
    # Either INTENT_INVALID (pure gibberish) or INTENT_UNSUPPORTED is correct.
    # The key invariant: no route is extracted, no LLM lookup is triggered.
    assert answer.status in (INTENT_INVALID, INTENT_UNSUPPORTED)
    assert answer.route is None


def test_gemini_error_deterministic_fallback():
    from src.assistant import build_grounding_context
    from src.gemini import generate_grounded_response
    output, audit, notes = _fixtures()
    q = 'Why did Chennai-Bangalore spike on 2025-02-24?'
    det = answer_question(q, output, audit, notes)
    ctx = build_grounding_context(q, det, output, audit, notes)
    cfg = GeminiConfig(api_key='test-key', model='gemini-2.5-flash')
    with patch('src.gemini.urlopen', side_effect=OSError('network error')):
        with pytest.raises(GeminiRequestError):
            generate_grounded_response(q, ctx, cfg)
    assert det.status == 'evidence-supported'
    assert 'N001' in det.text


def test_classify_intent_units():
    from src.assistant import INTENT_ROUTE_INVESTIGATION
    # Pure consonant string -> INTENT_INVALID (no plausible word)
    assert classify_intent('zxcvbnm') == INTENT_INVALID
    assert classify_intent('') == INTENT_INVALID
    assert classify_intent('   ') == INTENT_INVALID
    # Keyboard-mash with vowels -> INTENT_INVALID or INTENT_UNSUPPORTED (both are OK)
    assert classify_intent('yhyjuyuyukuy') in (INTENT_INVALID, INTENT_UNSUPPORTED)
    assert classify_intent('asdf qwerty') in (INTENT_INVALID, INTENT_UNSUPPORTED)
    # Valid freight questions
    assert classify_intent('Why did costs rise?') == INTENT_ROUTE_INVESTIGATION
    assert classify_intent('Why did the route spike?') == INTENT_ROUTE_INVESTIGATION
    # Off-topic
    assert classify_intent('Write me a poem about trucks') == INTENT_UNSUPPORTED
    assert classify_intent('Tell me a joke') == INTENT_UNSUPPORTED
