from pathlib import Path

import pandas as pd

from src.assistant import answer_question
from src.retrieval import load_context_notes


ROOT = Path(__file__).parents[1]


def _fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = pd.DataFrame(
        [
            {
                "route": "Chennai-Bangalore",
                "week_of": "2024-02-19",
                "cost_per_tonne_km": 2.90,
                "vs_own_history": "+2.0% vs this route's past average",
                "vs_similar_routes": "+3.0% vs similar route types this week",
                "flagged": "No",
                "matched_note_id": "",
                "reason": "No anomaly",
            },
            {
                "route": "Chennai-Bangalore",
                "week_of": "2025-02-24",
                "cost_per_tonne_km": 4.10,
                "vs_own_history": "+20.0% vs this route's past average",
                "vs_similar_routes": "+22.0% vs similar route types this week",
                "flagged": "No (justified)",
                "matched_note_id": "N001",
                "reason": "Matches note N001",
            },
            {
                "route": "Mumbai-Pune",
                "week_of": "2025-09-15",
                "cost_per_tonne_km": 3.98,
                "vs_own_history": "+9.2% vs this route's past average",
                "vs_similar_routes": "+23.6% vs similar route types this week",
                "flagged": "Yes",
                "matched_note_id": "",
                "reason": "Unexplained",
            },
        ]
    )
    audit = pd.DataFrame(
        [
            {"route": "Chennai-Bangalore", "week_of": "2024-02-19", "route_type": "Long", "anomaly": False, "trigger_baselines": ""},
            {"route": "Chennai-Bangalore", "week_of": "2025-02-24", "route_type": "Long", "anomaly": True, "trigger_baselines": "own_history,similar_routes"},
            {"route": "Mumbai-Pune", "week_of": "2025-09-15", "route_type": "Short", "anomaly": True, "trigger_baselines": "similar_routes"},
        ]
    )
    return output, audit, load_context_notes(ROOT / "context_notes.csv")


def test_valid_question_cites_exact_note() -> None:
    output, audit, notes = _fixtures()

    answer = answer_question("Why did Chennai-Bangalore spike in late February?", output, audit, notes)

    assert answer.status == "evidence-supported"
    assert "N001" in answer.text
    assert "2025-02-24" in answer.text


def test_unexplained_question_does_not_invent_cause() -> None:
    output, audit, notes = _fixtures()

    answer = answer_question("Why did Mumbai-Pune get pricier in 2025-09-15?", output, audit, notes)

    assert answer.status == "unexplained"
    assert "cannot confirm a cause" in answer.text
    assert answer.note_ids == ()


def test_unknown_route_is_rejected() -> None:
    output, audit, notes = _fixtures()

    answer = answer_question("Why did Atlantis-Moon spike in March?", output, audit, notes)

    assert answer.status == "not-found"
    assert "Atlantis-Moon" in answer.text


def test_unknown_week_is_not_invented() -> None:
    output, audit, notes = _fixtures()

    answer = answer_question("Why did Chennai-Bangalore spike in 2030-01-07?", output, audit, notes)

    assert answer.status == "clarification"
    assert "no computed route-week" in answer.text


def test_ambiguous_route_question_asks_for_clarification() -> None:
    output, audit, notes = _fixtures()

    answer = answer_question("Why did the route get more expensive?", output, audit, notes)

    assert answer.status == "clarification"
    assert "Which route" in answer.text


def test_note_rejection_explanation_uses_retrieval_rules() -> None:
    output, audit, notes = _fixtures()

    answer = answer_question("Why was note N006 rejected for Mumbai-Pune on 2025-09-15?", output, audit, notes)

    assert answer.status == "unexplained"
    assert "N006" in answer.text
    assert "effective date window" in answer.text


def test_global_lists_use_verified_rows() -> None:
    output, audit, notes = _fixtures()

    supported = answer_question("Which anomalies have supporting evidence?", output, audit, notes)
    unexplained = answer_question("Which anomalies remain unexplained?", output, audit, notes)

    assert "N001" in supported.text
    assert "Mumbai-Pune" in unexplained.text
