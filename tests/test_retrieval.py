from pathlib import Path

import pandas as pd

from src.retrieval import find_supporting_note, load_context_notes


NOTES = Path(__file__).parents[1] / "context_notes.csv"


def test_exact_route_note_supports_route_and_peer_in_matching_week() -> None:
    notes = load_context_notes(NOTES)

    match = find_supporting_note(
        notes,
        route="Ahmedabad-Mumbai",
        week_of=pd.Timestamp("2025-01-20"),
        trigger_baselines="own_history,similar_routes",
    )

    assert match is not None
    assert match.note_id == "N002"
    assert match.supports_peer_relative


def test_explicit_note_date_window_applies_to_overlapping_week() -> None:
    notes = load_context_notes(NOTES)

    match = find_supporting_note(
        notes,
        route="Chennai-Bangalore",
        week_of=pd.Timestamp("2025-03-03"),
        trigger_baselines="own_history",
    )

    assert match is not None
    assert match.note_id == "N001"


def test_out_of_window_note_is_rejected() -> None:
    notes = load_context_notes(NOTES)

    match = find_supporting_note(
        notes,
        route="Mumbai-Pune",
        week_of=pd.Timestamp("2025-09-15"),
        trigger_baselines="similar_routes",
    )

    assert match is None


def test_unaffected_note_is_rejected() -> None:
    notes = load_context_notes(NOTES)

    match = find_supporting_note(
        notes,
        route="Mumbai-Delhi",
        week_of=pd.Timestamp("2024-07-29"),
        trigger_baselines="own_history",
    )

    assert match is None


def test_all_routes_note_supports_history_but_not_peer_relative_increase() -> None:
    notes = load_context_notes(NOTES)

    history_match = find_supporting_note(
        notes,
        route="Mumbai-Pune",
        week_of=pd.Timestamp("2025-05-05"),
        trigger_baselines="own_history",
    )
    peer_match = find_supporting_note(
        notes,
        route="Mumbai-Pune",
        week_of=pd.Timestamp("2025-05-05"),
        trigger_baselines="similar_routes",
    )

    assert history_match is not None and history_match.note_id == "N003"
    assert peer_match is None
