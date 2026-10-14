"""Run the deterministic freight cost intelligence pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from src.analytics import calculate_weekly_metrics
from src.config import (
    ANOMALY_THRESHOLD_PERCENT,
    CONTEXT_NOTES_PATH,
    HISTORY_WINDOW_WEEKS,
    OUTPUTS_DIR,
    OUTPUT_TEMPLATE_PATH,
    SHIPMENTS_PATH,
)
from src.detection import detect_anomalies
from src.ingestion import load_shipments
from src.reporting import (
    build_reports,
    inspect_output_template,
    validate_output_template,
    write_reports,
)
from src.retrieval import load_context_notes


def run_pipeline() -> dict[str, object]:
    """Run all deterministic stages and write reports to ``outputs``."""

    validate_output_template(OUTPUT_TEMPLATE_PATH)
    template_diagnostics = inspect_output_template(OUTPUT_TEMPLATE_PATH)
    shipments = load_shipments(SHIPMENTS_PATH)
    weekly = calculate_weekly_metrics(shipments.records, history_window=HISTORY_WINDOW_WEEKS)
    detected = detect_anomalies(weekly, threshold_percent=ANOMALY_THRESHOLD_PERCENT)
    notes = load_context_notes(CONTEXT_NOTES_PATH)
    output_frame, audit_frame = build_reports(detected, notes)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    write_reports(
        output_frame,
        audit_frame,
        OUTPUTS_DIR / "flags.csv",
        OUTPUTS_DIR / "audit.csv",
    )
    run_log = {
        "pipeline": "Freight Cost Intelligence",
        "inputs": {
            "shipments_path": str(SHIPMENTS_PATH.name),
            "context_notes_path": str(CONTEXT_NOTES_PATH.name),
            "shipment_input_records": shipments.report.input_records,
            "shipment_accepted_records": shipments.report.accepted_records,
            "shipment_rejected_records": shipments.report.rejected_records,
            "shipment_rejection_reasons": shipments.report.reasons,
            "context_note_records": len(notes),
        },
        "configuration": {
            "history_window_weeks": HISTORY_WINDOW_WEEKS,
            "anomaly_threshold_percent": ANOMALY_THRESHOLD_PERCENT,
        },
        "template": template_diagnostics,
        "results": {
            "weekly_route_records": len(detected),
            "anomalies": int(detected["anomaly"].sum()),
            "justified_anomalies": int((output_frame["flagged"] == "No (justified)").sum()),
            "unexplained_anomalies": int((output_frame["flagged"] == "Yes").sum()),
            "output_rows": len(output_frame),
        },
        "llm_accounting": {
            "llm_calls": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "llm_api_cost": 0.0,
            "note": "The core pipeline uses deterministic local computation and no LLM or paid API.",
        },
        "artifacts": ["outputs/flags.csv", "outputs/audit.csv", "outputs/run_log.json"],
    }
    (OUTPUTS_DIR / "run_log.json").write_text(
        json.dumps(run_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_log


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result["results"], sort_keys=True))

