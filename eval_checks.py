"""Run small, manually selected checks against the actual supplied data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline import run_pipeline
from src.config import OUTPUTS_DIR


EVALUATION_CASES = (
    {"route": "Delhi-Jaipur", "week_of": "2024-11-11", "expected_flagged": "Yes", "expected_note": ""},
    {"route": "Ahmedabad-Mumbai", "week_of": "2025-01-20", "expected_flagged": "No (justified)", "expected_note": "N002"},
    {"route": "Mumbai-Pune", "week_of": "2025-09-15", "expected_flagged": "Yes", "expected_note": ""},
)


def main() -> None:
    run_pipeline()
    output = pd.read_csv(OUTPUTS_DIR / "flags.csv", dtype="string", keep_default_na=False)
    checks: list[dict[str, object]] = []
    for case in EVALUATION_CASES:
        match = output[
            output["route"].eq(case["route"]) & output["week_of"].eq(case["week_of"])
        ]
        passed = (
            len(match) == 1
            and match.iloc[0]["flagged"] == case["expected_flagged"]
            and match.iloc[0]["matched_note_id"] == case["expected_note"]
        )
        checks.append({"case": case, "passed": passed})
    result = {"passed": sum(item["passed"] for item in checks), "total": len(checks), "checks": checks}
    (OUTPUTS_DIR / "eval_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["passed"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

