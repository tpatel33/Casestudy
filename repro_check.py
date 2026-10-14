"""Run the full pipeline three times and compare deterministic artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline import run_pipeline
from src.config import OUTPUTS_DIR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    artifacts = ["flags.csv", "audit.csv"]
    runs: list[dict[str, object]] = []
    for run_number in range(1, 4):
        result = run_pipeline()
        runs.append(
            {
                "run": run_number,
                "results": result["results"],
                "sha256": {name: _sha256(OUTPUTS_DIR / name) for name in artifacts},
            }
        )

    first = runs[0]
    matches = all(
        run["results"] == first["results"] and run["sha256"] == first["sha256"]
        for run in runs[1:]
    )
    report = {
        "runs_completed": len(runs),
        "matched": matches,
        "comparison_scope": [
            "numeric and text output fields via deterministic CSV hashes",
            "anomaly and evidence counts",
            "stable row ordering via deterministic CSV hashes",
        ],
        "runs": runs,
    }
    report_path = OUTPUTS_DIR / "reproducibility_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

