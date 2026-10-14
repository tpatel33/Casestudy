"""Validated ingestion for the supplied shipment CSV."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import SHIPMENT_COLUMNS


class DataValidationError(ValueError):
    """Raised when the input structure cannot support the pipeline."""


@dataclass
class IngestionReport:
    """Counts and reasons produced while validating shipment records."""

    input_records: int
    accepted_records: int = 0
    rejected_records: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def add_reason(self, reason: str, count: int) -> None:
        if count:
            self.reasons[reason] = self.reasons.get(reason, 0) + int(count)


@dataclass
class IngestionResult:
    """Validated records and their non-silent validation report."""

    records: pd.DataFrame
    report: IngestionReport


def _normalise_text_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = frame[column].astype("string").str.strip()


def load_shipments(path: str | Path, *, strict: bool = False) -> IngestionResult:
    """Load and validate shipments without silently discarding bad rows.

    Rows with missing fields, invalid dates, invalid numeric values, duplicate
    shipment IDs, or duplicate complete records are excluded from the returned
    frame and counted in the report. Structural problems raise immediately.
    """

    source = Path(path)
    if not source.is_file():
        raise DataValidationError(f"Shipment file does not exist: {source}")

    frame = pd.read_csv(source, dtype="string", keep_default_na=False)
    missing_columns = [column for column in SHIPMENT_COLUMNS if column not in frame]
    if missing_columns:
        raise DataValidationError(
            "Shipment CSV is missing required columns: " + ", ".join(missing_columns)
        )

    frame = frame.loc[:, list(SHIPMENT_COLUMNS)].copy()
    report = IngestionReport(input_records=len(frame))
    _normalise_text_columns(frame, SHIPMENT_COLUMNS)
    invalid = pd.Series(False, index=frame.index)

    for column in SHIPMENT_COLUMNS:
        missing = frame[column].isna() | frame[column].eq("")
        invalid |= missing
        report.add_reason(f"missing_{column}", int(missing.sum()))

    parsed_dates = pd.to_datetime(
        frame["shipment_date"], format="%Y-%m-%d", errors="coerce"
    )
    invalid_dates = parsed_dates.isna() & frame["shipment_date"].ne("")
    invalid |= invalid_dates
    report.add_reason("invalid_shipment_date", int(invalid_dates.sum()))
    frame["shipment_date"] = parsed_dates

    for column in ("quantity_tonnes", "distance_km", "freight_cost_inr"):
        parsed = pd.to_numeric(frame[column], errors="coerce")
        invalid_numeric = parsed.isna() & frame[column].ne("")
        non_positive = parsed.notna() & parsed.le(0)
        invalid |= invalid_numeric | non_positive
        report.add_reason(f"invalid_{column}", int(invalid_numeric.sum()))
        report.add_reason(f"non_positive_{column}", int(non_positive.sum()))
        frame[column] = parsed

    duplicate_ids = frame["shipment_id"].duplicated(keep=False)
    duplicate_rows = frame.duplicated(keep=False)
    invalid |= duplicate_ids | duplicate_rows
    report.add_reason("duplicate_shipment_id", int(duplicate_ids.sum()))
    report.add_reason("duplicate_record", int(duplicate_rows.sum()))

    validated = frame.loc[~invalid].reset_index(drop=True)
    report.rejected_records = int(invalid.sum())
    report.accepted_records = len(validated)

    if strict and report.rejected_records:
        details = ", ".join(
            f"{reason}={count}" for reason, count in sorted(report.reasons.items())
        )
        raise DataValidationError(
            f"Shipment validation rejected {report.rejected_records} records ({details})"
        )
    if validated.empty:
        raise DataValidationError("Shipment validation produced no usable records")

    return IngestionResult(records=validated, report=report)

