"""Output contract and grounded explanation generation."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .retrieval import CATEGORY_BY_NOTE, EvidenceMatch, find_supporting_note

OUTPUT_COLUMNS = (
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
)


def validate_output_template(path: str | Path) -> list[str]:
    """Return the exact template columns or raise on contract drift."""

    diagnostics = inspect_output_template(path)
    columns = diagnostics["columns"]
    if columns != list(OUTPUT_COLUMNS):
        raise ValueError(
            "Output template columns do not match the required contract: "
            + ", ".join(columns)
        )
    return columns


def inspect_output_template(path: str | Path) -> dict[str, object]:
    """Inspect the header and report malformed sample rows without changing it."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError("Output template is empty")
    columns = rows[0]
    malformed_rows = [index for index, row in enumerate(rows[1:], start=2) if len(row) != len(columns)]
    return {
        "columns": columns,
        "sample_row_count": len(rows) - 1,
        "malformed_sample_rows": malformed_rows,
    }


def _comparison_text(value: object, suffix: str) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):+0.1f}% vs {suffix}"


def _reason_for_non_anomaly(row: pd.Series) -> str:
    if pd.isna(row["history_baseline"]) and pd.isna(row["peer_baseline"]):
        return "No available historical or same-week peer baseline; no anomaly verdict was assigned."
    return "No anomaly: the current cost did not meet the configured increase threshold against an available baseline."


def _build_reason(row: pd.Series, evidence: EvidenceMatch | None) -> str:
    if not bool(row["anomaly"]):
        return _reason_for_non_anomaly(row)
    if evidence is None:
        return (
            "No valid supporting note found for this route, week, and triggered baseline(s). "
            "Cost rise looks unexplained and is worth a human review."
        )
    return (
        f"Matches note {evidence.note_id} dated {evidence.date:%Y-%m-%d}: {evidence.note} "
        "The cost rise has a clear explanation."
    )


def build_reports(
    weekly_metrics: pd.DataFrame,
    notes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the exact graded output and an internal audit frame."""

    required = {
        "route",
        "week_of",
        "cost_per_tonne_km",
        "vs_own_history",
        "vs_similar_routes",
        "anomaly",
        "trigger_baselines",
        "rule",
    }
    missing = sorted(required.difference(weekly_metrics.columns))
    if missing:
        raise ValueError("Weekly metrics are missing reporting columns: " + ", ".join(missing))

    output_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    ordered = weekly_metrics.sort_values(
        ["week_of", "origin", "destination", "route_type"], kind="stable"
    )
    for _, row in ordered.iterrows():
        evidence = None
        if bool(row["anomaly"]):
            evidence = find_supporting_note(
                notes,
                route=str(row["route"]),
                week_of=pd.Timestamp(row["week_of"]),
                trigger_baselines=str(row["trigger_baselines"]),
            )
        output_rows.append(
            {
                "route": row["route"],
                "week_of": pd.Timestamp(row["week_of"]).strftime("%Y-%m-%d"),
                "cost_per_tonne_km": round(float(row["cost_per_tonne_km"]), 2),
                "vs_own_history": _comparison_text(row["vs_own_history"], "this route's past average"),
                "vs_similar_routes": _comparison_text(row["vs_similar_routes"], "similar route types this week"),
                "flagged": "No (justified)" if evidence else ("Yes" if row["anomaly"] else "No"),
                "matched_note_id": evidence.note_id if evidence else "",
                "reason": _build_reason(row, evidence),
            }
        )
        audit_rows.append(
            {
                "route": row["route"],
                "route_type": row["route_type"],
                "week_of": pd.Timestamp(row["week_of"]).strftime("%Y-%m-%d"),
                "anomaly": bool(row["anomaly"]),
                "trigger_baselines": row["trigger_baselines"],
                "rule": row["rule"],
                "history_baseline": row["history_baseline"],
                "peer_baseline": row["peer_baseline"],
                "matched_note_id": evidence.note_id if evidence else "",
                "evidence_validated": evidence is not None,
                "evidence_category": CATEGORY_BY_NOTE.get(evidence.note_id, "—") if evidence else "—",
                "evidence_scope": "route-specific" if evidence and evidence.supports_peer_relative else ("all-routes-history" if evidence else ""),
            }
        )
    return pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS), pd.DataFrame(audit_rows)


def write_reports(
    output_frame: pd.DataFrame,
    audit_frame: pd.DataFrame,
    output_path: str | Path,
    audit_path: str | Path,
) -> None:
    """Write deterministic graded and internal audit CSVs."""

    output_target = Path(output_path)
    audit_target = Path(audit_path)
    output_target.parent.mkdir(parents=True, exist_ok=True)
    audit_target.parent.mkdir(parents=True, exist_ok=True)
    output_frame.loc[:, list(OUTPUT_COLUMNS)].to_csv(
        output_target, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
    )
    audit_frame.to_csv(audit_target, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
