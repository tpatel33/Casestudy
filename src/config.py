"""Shared paths and deterministic pipeline configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHIPMENTS_PATH = PROJECT_ROOT / "shipment_records.csv"
CONTEXT_NOTES_PATH = PROJECT_ROOT / "context_notes.csv"
OUTPUT_TEMPLATE_PATH = PROJECT_ROOT / "sample_output_format_v2.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ANOMALY_THRESHOLD_PERCENT = 15.0
HISTORY_WINDOW_WEEKS = 8

SHIPMENT_COLUMNS = (
    "shipment_id",
    "origin",
    "destination",
    "route_type",
    "material",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
    "transporter",
)

ROUTE_KEYS = ("origin", "destination", "route_type")

