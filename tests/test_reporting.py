from pathlib import Path

import pandas as pd

from src.reporting import (
    OUTPUT_COLUMNS,
    build_reports,
    inspect_output_template,
    validate_output_template,
    write_reports,
)
from src.retrieval import load_context_notes


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "origin": "Ahmedabad",
                "destination": "Mumbai",
                "route_type": "Medium",
                "route": "Ahmedabad-Mumbai",
                "week_of": pd.Timestamp("2025-01-20"),
                "cost_per_tonne_km": 3.29,
                "history_baseline": 2.54,
                "peer_baseline": 2.68,
                "vs_own_history": 29.5,
                "vs_similar_routes": 22.5,
                "anomaly": True,
                "trigger_baselines": "own_history,similar_routes",
                "rule": "test rule",
            }
        ]
    )


def test_template_columns_are_exact() -> None:
    columns = validate_output_template(Path(__file__).parents[1] / "sample_output_format_v2.csv")

    assert columns == list(OUTPUT_COLUMNS)
    assert inspect_output_template(Path(__file__).parents[1] / "sample_output_format_v2.csv")[
        "malformed_sample_rows"
    ] == [4]


def test_justified_output_has_note_id_and_grounded_reason(tmp_path: Path) -> None:
    output, audit = build_reports(
        _metrics(), load_context_notes(Path(__file__).parents[1] / "context_notes.csv")
    )
    output_path = tmp_path / "flags.csv"
    audit_path = tmp_path / "audit.csv"
    write_reports(output, audit, output_path, audit_path)

    written = pd.read_csv(output_path, dtype="string")
    assert written.columns.tolist() == list(OUTPUT_COLUMNS)
    assert written.loc[0, "flagged"] == "No (justified)"
    assert written.loc[0, "matched_note_id"] == "N002"
    assert "N002" in written.loc[0, "reason"]
    assert "surcharge" in written.loc[0, "reason"].casefold()
