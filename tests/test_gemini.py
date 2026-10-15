from src.assistant import answer_question, build_grounding_context
from src.gemini import GeminiConfig, GeminiRequestError, GeminiUsage, validate_grounded_response

from test_assistant import _fixtures


def test_system_overview_explains_the_deterministic_setup() -> None:
    output, audit, notes = _fixtures()

    answer = answer_question("What is this whole system about?", output, audit, notes)

    assert answer.status == "overview"
    assert "cost per tonne-km" in answer.text
    assert "previous eight" in answer.text
    assert "same `route_type`" in answer.text


def test_grounding_context_contains_only_selected_verified_note() -> None:
    output, audit, notes = _fixtures()
    answer = answer_question("Why did Chennai-Bangalore spike on 2025-02-24?", output, audit, notes)

    context = build_grounding_context("Why did it spike?", answer, output, audit, notes)

    assert context["allowed_note_ids"] == ["N001"]
    assert context["verified_route_week"]["flagged"] == "No (justified)"
    assert [note["note_id"] for note in context["verified_notes"]] == ["N001"]


def test_validator_rejects_missing_or_invented_provenance() -> None:
    output, audit, notes = _fixtures()
    answer = answer_question("Why did Chennai-Bangalore spike on 2025-02-24?", output, audit, notes)
    context = build_grounding_context("Why did it spike?", answer, output, audit, notes)

    valid = "Chennai-Bangalore was unusual in week 2025-02-24. Evidence note N001 supports the increase."
    invented_note = "Chennai-Bangalore was unusual in week 2025-02-24. Evidence note N999 supports it."

    assert validate_grounded_response(valid, answer, context)
    assert not validate_grounded_response(invented_note, answer, context)


def test_gemini_config_does_not_require_a_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    config = GeminiConfig.from_sources({})

    assert not config.configured
    assert config.model


def test_placeholder_secret_does_not_enable_gemini() -> None:
    config = GeminiConfig.from_sources({"GEMINI_API_KEY": "PASTE_YOUR_GEMINI_KEY_HERE"})

    assert not config.configured


def test_understand_question_parses_valid_json() -> None:
    import json
    from unittest.mock import MagicMock, patch
    from src.gemini import understand_question

    config = GeminiConfig(api_key="mock-key", model="gemini-3.5-flash-lite")
    fake_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "intent": "route_investigation",
                                "route": "Chennai-Bangalore",
                                "week_iso": "2025-02-24",
                                "week_hint": "late February",
                                "valid": True,
                            })
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 20, "totalTokenCount": 70},
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(fake_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("src.gemini.urlopen", return_value=mock_resp):
        understanding = understand_question(
            "Why did Chennai to Bangalore spike in late February?",
            ["Chennai-Bangalore", "Mumbai-Pune"],
            config,
        )

    assert understanding.intent == "route_investigation"
    assert understanding.route == "Chennai-Bangalore"
    assert understanding.week_iso == "2025-02-24"
    assert understanding.valid is True
    assert understanding.usage is not None
    assert understanding.usage.input_tokens == 50


def test_answer_question_with_gemini_understanding() -> None:
    import json
    from unittest.mock import MagicMock, patch

    output, audit, notes = _fixtures()
    config = GeminiConfig(api_key="mock-key", model="gemini-3.5-flash-lite")

    fake_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "intent": "invalid_input",
                                "route": None,
                                "week_iso": None,
                                "week_hint": None,
                                "valid": False,
                            })
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 15, "totalTokenCount": 55},
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(fake_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("src.gemini.urlopen", return_value=mock_resp):
        ans = answer_question("gibberish-string", output, audit, notes, gemini_config=config)

    assert ans.status == "invalid_input"
    assert "Which route" not in ans.text
    assert ans.usage is not None


def test_gemini_interpretation_failure_uses_deterministic_fallback() -> None:
    from unittest.mock import patch

    output, audit, notes = _fixtures()
    config = GeminiConfig(api_key="mock-key", model="gemini-3.5-flash-lite")
    usage = GeminiUsage(provider="gemini", model=config.model)

    with patch(
        "src.gemini.understand_question",
        side_effect=GeminiRequestError("temporary network failure", usage),
    ):
        answer = answer_question(
            "Why did Chennai-Bengaluru spike in late February?",
            output,
            audit,
            notes,
            gemini_config=config,
        )

    assert answer.status == "evidence-supported"
    assert "N001" in answer.text
