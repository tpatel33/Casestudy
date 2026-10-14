"""Streamlit dashboard for the freight cost intelligence pipeline."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from pipeline import run_pipeline
from src.config import CONTEXT_NOTES_PATH, OUTPUTS_DIR
from src.retrieval import load_context_notes


st.set_page_config(page_title="Freight Cost Intelligence", page_icon="🚚", layout="wide")


@st.cache_data(show_spinner="Running deterministic freight analysis...")
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_pipeline()
    output = pd.read_csv(OUTPUTS_DIR / "flags.csv", dtype="string", keep_default_na=False)
    audit = pd.read_csv(OUTPUTS_DIR / "audit.csv", keep_default_na=False)
    notes = load_context_notes(CONTEXT_NOTES_PATH)
    return output, audit, notes


output, audit, notes = load_dashboard_data()
output["cost_per_tonne_km"] = pd.to_numeric(output["cost_per_tonne_km"])
output["week_of"] = pd.to_datetime(output["week_of"])
anomaly_output = output[output["flagged"].isin(["Yes", "No (justified)"])]

st.title("Freight Cost Intelligence")
st.caption("Evidence-grounded weekly route monitoring from the supplied shipment records")

metric_columns = st.columns(5)
metric_columns[0].metric("Route-weeks", f"{len(output):,}")
metric_columns[1].metric("Routes", f"{output['route'].nunique():,}")
metric_columns[2].metric("Anomalies", f"{len(anomaly_output):,}")
metric_columns[3].metric("Justified", f"{(output['flagged'] == 'No (justified)').sum():,}")
metric_columns[4].metric("Unexplained", f"{(output['flagged'] == 'Yes').sum():,}")

st.subheader("Route monitoring")
route_options = sorted(output["route"].unique())
selected_route = st.selectbox("Route", route_options)
route_rows = output[output["route"].eq(selected_route)].sort_values("week_of")
chart_data = route_rows.set_index("week_of")[["cost_per_tonne_km"]]
st.line_chart(chart_data, y="cost_per_tonne_km")

comparison_columns = [
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
]
st.dataframe(route_rows[comparison_columns], use_container_width=True, hide_index=True)

st.subheader("Anomaly review")
status_filter = st.multiselect(
    "Show statuses",
    ["Yes", "No (justified)"],
    default=["Yes", "No (justified)"],
)
review_rows = anomaly_output[anomaly_output["flagged"].isin(status_filter)]
st.dataframe(
    review_rows.sort_values(["week_of", "route"]),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Evidence details")
evidence_ids = sorted(review_rows["matched_note_id"].loc[lambda values: values.ne("")].unique())
if evidence_ids:
    selected_note_id = st.selectbox("Matched note", evidence_ids)
    note = notes[notes["note_id"].eq(selected_note_id)].iloc[0]
    st.write(
        {
            "note_id": str(note["note_id"]),
            "date": pd.Timestamp(note["date"]).strftime("%Y-%m-%d"),
            "applies_to": str(note["applies_to"]),
            "note": str(note["note"]),
        }
    )
else:
    st.info("No justified anomaly is selected by the current filters.")

st.subheader("Download")
st.download_button(
    "Download required flags.csv",
    data=(OUTPUTS_DIR / "flags.csv").read_bytes(),
    file_name="flags.csv",
    mime="text/csv",
)

