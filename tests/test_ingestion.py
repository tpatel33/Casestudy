from pathlib import Path

import pandas as pd
import pytest

from src.ingestion import DataValidationError, load_shipments


def test_supplied_shipments_load_with_no_rejections() -> None:
    result = load_shipments(Path(__file__).parents[1] / "shipment_records.csv")

    assert len(result.records) == 2940
    assert result.report.rejected_records == 0
    assert pd.api.types.is_datetime64_ns_dtype(result.records["shipment_date"])


def test_invalid_rows_are_reported_and_not_silently_kept(tmp_path: Path) -> None:
    path = tmp_path / "shipments.csv"
    path.write_text(
        "shipment_id,origin,destination,route_type,material,quantity_tonnes,distance_km,freight_cost_inr,shipment_date,transporter\n"
        "S1,A,B,Short,Goods,10,100,1000,2024-01-01,T\n"
        "S2,A,B,Short,Goods,0,100,1000,2024-01-02,T\n"
        "S3,A,B,Short,Goods,10,100,1000,not-a-date,T\n",
        encoding="utf-8",
    )

    result = load_shipments(path)

    assert len(result.records) == 1
    assert result.report.rejected_records == 2
    assert result.report.reasons["non_positive_quantity_tonnes"] == 1
    assert result.report.reasons["invalid_shipment_date"] == 1


def test_missing_required_column_is_structural_error(tmp_path: Path) -> None:
    path = tmp_path / "shipments.csv"
    path.write_text("shipment_id,origin\nS1,A\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="missing required columns"):
        load_shipments(path)

