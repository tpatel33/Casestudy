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


def _range_fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output, audit, notes = _fixtures()
    extra_output = pd.DataFrame(
        [
            {
                "route": "Chennai-Bangalore",
                "week_of": week,
                "cost_per_tonne_km": cost,
                "vs_own_history": "+2.0% vs this route's past average",
                "vs_similar_routes": "+3.0% vs similar route types this week",
                "flagged": "No",
                "matched_note_id": "",
                "reason": "No anomaly",
            }
            for week, cost in (
                ("2024-03-04", 3.00),
                ("2024-03-11", 3.10),
                ("2025-03-03", 3.20),
                ("2025-03-10", 3.30),
                ("2025-03-31", 3.40),
                ("2025-04-07", 3.50),
            )
        ]
    )
    extra_audit = pd.DataFrame(
        [
            {
                "route": "Chennai-Bangalore",
                "week_of": week,
                "route_type": "Long",
                "anomaly": False,
                "trigger_baselines": "",
            }
            for week, _ in (
                ("2024-03-04", 3.00),
                ("2024-03-11", 3.10),
                ("2025-03-03", 3.20),
                ("2025-03-10", 3.30),
                ("2025-03-31", 3.40),
                ("2025-04-07", 3.50),
            )
        ]
    )
    return (
        pd.concat([output, extra_output], ignore_index=True),
        pd.concat([audit, extra_audit], ignore_index=True),
        notes,
    )


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


def test_route_alias_and_first_monday_are_resolved() -> None:
    output, audit, notes = _range_fixtures()

    answer = answer_question(
        "What was the cost on the first Monday of March 2025 for Chennai-Bengaluru?",
        output,
        audit,
        notes,
    )

    assert answer.status == "not-anomaly"
    assert answer.route == "Chennai-Bangalore"
    assert answer.week_of == pd.Timestamp("2025-03-03")
    assert "3.20" in answer.text


def test_last_monday_and_month_wide_question_return_verified_rows() -> None:
    output, audit, notes = _range_fixtures()

    last = answer_question(
        "What was the cost on the last Monday of March 2025 for Chennai-Bangalore?",
        output,
        audit,
        notes,
    )
    month = answer_question(
        "What were the costs for Chennai-Bangalore in March 2025?",
        output,
        audit,
        notes,
    )

    assert last.week_of == pd.Timestamp("2025-03-31")
    assert month.status == "summary"
    assert "2025-03-03" in month.text
    assert "2025-03-31" in month.text


def test_ambiguous_month_year_asks_for_conversational_clarification() -> None:
    output, audit, notes = _range_fixtures()

    answer = answer_question("What were the costs for Chennai-Bangalore in March?", output, audit, notes)

    assert answer.status == "clarification"
    assert "Which year" in answer.text
    assert "2024" in answer.text and "2025" in answer.text


def test_month_date_range_returns_all_matching_weeks() -> None:
    output, audit, notes = _range_fixtures()

    answer = answer_question(
        "What happened for Chennai-Bangalore from March 2025 to April 2025?",
        output,
        audit,
        notes,
    )

    assert answer.status == "summary"
    assert answer.weeks == tuple(
        pd.to_datetime(value)
        for value in ("2025-03-03", "2025-03-10", "2025-03-31", "2025-04-07")
    )


def test_follow_up_uses_previous_route_and_week_context() -> None:
    output, audit, notes = _range_fixtures()
    first_question = "Why did Chennai-Bangalore spike on 2025-02-24?"
    first = answer_question(first_question, output, audit, notes)

    follow_up = answer_question(
        "What about the next week?",
        output,
        audit,
        notes,
        conversation_context=[
            {"role": "user", "content": first_question},
            {"role": "assistant", "content": first.text},
        ],
    )

    assert follow_up.route == "Chennai-Bangalore"
    assert follow_up.week_of == pd.Timestamp("2025-03-03")
