"""Streamlit dashboard for the freight cost intelligence pipeline."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.assistant import answer_question, build_grounding_context
from pipeline import run_pipeline
from src.config import CONTEXT_NOTES_PATH, OUTPUTS_DIR
from src.gemini import (
    GeminiConfig,
    GeminiRequestError,
    generate_grounded_response,
    validate_grounded_response,
)
from src.retrieval import CATEGORY_BY_NOTE, load_context_notes


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
output["evidence_category"] = output["matched_note_id"].map(CATEGORY_BY_NOTE).fillna("—")
anomaly_output = output[output["flagged"].isin(["Yes", "No (justified)"])]

st.title("Freight Cost Intelligence")
st.caption("Evidence-grounded weekly route monitoring from the supplied shipment records")

with st.expander("How to read this dashboard", expanded=True):
    st.markdown(
        "**Purpose:** Find route-weeks where shipping cost is unusually high. "
        "Every chart point represents one route during one Monday-to-Sunday week."
    )
    st.markdown(
        "**Cost per tonne-km:** total freight cost divided by the tonnes carried and kilometres travelled. "
        "This makes different shipment sizes and distances comparable."
    )
    st.markdown(
        "**Own-route comparison:** compares the week with up to the previous 8 weeks for the same route. "
        "**Similar-route comparison:** compares it with other routes of the same supplied `route_type` during that week."
    )
    st.markdown(
        "**Evidence-supported:** a context note matches the route, dates, and cost impact. "
        "**Unexplained:** the numbers look unusual, but the supplied notes do not prove a cause."
    )

metric_columns = st.columns(5)
metric_columns[0].metric(
    "Weekly route records",
    f"{len(output):,}",
    help="One row for a route during one Monday-to-Sunday week.",
)
metric_columns[1].metric(
    "Routes monitored",
    f"{output['route'].nunique():,}",
    help="Distinct directional routes in the shipment records.",
)
metric_columns[2].metric(
    "Unusual weeks",
    f"{len(anomaly_output):,}",
    help="Weeks exceeding the configured 15% threshold against the route history or peer baseline.",
)
metric_columns[3].metric(
    "Evidence-supported",
    f"{(output['flagged'] == 'No (justified)').sum():,}",
    help="Unusual weeks with a validated supporting context note.",
)
metric_columns[4].metric(
    "Needs explanation",
    f"{(output['flagged'] == 'Yes').sum():,}",
    help="Unusual weeks where no supplied context note proves the cause.",
)

st.subheader("Route monitoring")
st.caption("Select a route to see how its normalized shipping cost changes week by week.")
route_options = sorted(output["route"].unique())
selected_route = st.selectbox("Route", route_options)
route_rows = output[output["route"].eq(selected_route)].sort_values("week_of")
chart_data = route_rows.set_index("week_of")[["cost_per_tonne_km"]].rename(
    columns={"cost_per_tonne_km": "Cost per tonne-km (₹)"}
)
st.caption("Chart: cost per tonne-km (₹). Higher values mean this route was more expensive after adjusting for load and distance.")
st.line_chart(chart_data)

comparison_columns = [
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "evidence_category",
]
route_display = route_rows[comparison_columns].rename(
    columns={
        "week_of": "Week starting",
        "cost_per_tonne_km": "Cost (₹/tonne-km)",
        "vs_own_history": "Compared with this route's history",
        "vs_similar_routes": "Compared with similar routes",
        "flagged": "System status",
        "matched_note_id": "Evidence note",
        "evidence_category": "Evidence category",
    }
)
route_display["System status"] = route_display["System status"].replace(
    {"Yes": "Unexplained anomaly", "No (justified)": "Evidence-supported anomaly", "No": "No anomaly"}
)
st.dataframe(route_display, use_container_width=True, hide_index=True)

st.subheader("Anomaly review")
status_filter = st.multiselect(
    "Show statuses",
    ["Yes", "No (justified)"],
    default=["Yes", "No (justified)"],
)
review_rows = anomaly_output[anomaly_output["flagged"].isin(status_filter)]
review_display = review_rows.sort_values(["week_of", "route"]).copy()
review_display["status_label"] = review_display["flagged"].replace(
    {"Yes": "Unexplained anomaly", "No (justified)": "Evidence-supported anomaly"}
)
review_columns = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "status_label",
    "evidence_category",
    "matched_note_id",
    "reason",
]
review_display = review_display[review_columns].rename(
    columns={
        "route": "Route",
        "week_of": "Week starting",
        "cost_per_tonne_km": "Cost (₹/tonne-km)",
        "vs_own_history": "Compared with route history",
        "vs_similar_routes": "Compared with similar routes",
        "status_label": "Status",
        "evidence_category": "Evidence category",
        "matched_note_id": "Evidence note",
        "reason": "Why this status",
    }
)
st.dataframe(
    review_display,
    use_container_width=True,
    hide_index=True,
)
category_counts = review_rows.loc[
    review_rows["matched_note_id"].ne(""), "evidence_category"
].value_counts()
if category_counts.empty:
    st.info("No evidence categories are assigned because the selected anomalies have no validated notes.")
else:
    st.caption("Validated evidence categories only; unexplained anomalies are not assigned a cause category.")
    st.bar_chart(category_counts)

st.subheader("Evidence details")
evidence_ids = sorted(review_rows["matched_note_id"].loc[lambda values: values.ne("")].unique())
if evidence_ids:
    selected_note_id = st.selectbox("Matched note", evidence_ids)
    note = notes[notes["note_id"].eq(selected_note_id)].iloc[0]
    evidence_columns = st.columns(4)
    evidence_columns[0].metric("Note ID", str(note["note_id"]))
    evidence_columns[1].metric("Note date", pd.Timestamp(note["date"]).strftime("%Y-%m-%d"))
    evidence_columns[2].metric("Applies to", str(note["applies_to"]))
    evidence_columns[3].metric("Category", CATEGORY_BY_NOTE.get(selected_note_id, "—"))
    st.markdown("**Validated evidence text**")
    st.info(str(note["note"]))
else:
    st.info("No justified anomaly is selected by the current filters.")

st.subheader("Ask the analyst assistant")
st.caption(
    "Answers use only computed route-week results and validated context notes. "
    "Try: Why did Chennai-Bangalore spike in late February?"
)


def _streamlit_secrets() -> dict[str, object]:
    try:
        return dict(st.secrets)
    except (FileNotFoundError, RuntimeError):
        return {}


gemini_config = GeminiConfig.from_sources(_streamlit_secrets())
use_gemini = st.toggle(
    "Use optional Gemini wording",
    value=gemini_config.configured,
    disabled=not gemini_config.configured,
    help="Deterministic analytics and evidence retrieval remain authoritative. Gemini only rewrites verified context.",
)
if gemini_config.configured:
    st.caption(f"Gemini wording enabled with model `{gemini_config.model}`; verified facts remain authoritative.")
else:
    st.caption("Gemini is optional and not configured. The deterministic assistant remains fully available.")
if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []
if "llm_usage_log" not in st.session_state:
    st.session_state.llm_usage_log = []
for message in st.session_state.assistant_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about a route, week, anomaly, or evidence note")
if question:
    st.session_state.assistant_messages.append({"role": "user", "content": question})
    deterministic_answer = answer_question(
        question,
        output,
        audit,
        notes,
        gemini_config=gemini_config if use_gemini else None,
        conversation_context=st.session_state.assistant_messages,
    )
    if deterministic_answer.usage:
        st.session_state.llm_usage_log.append(
            {
                "provider": deterministic_answer.usage.provider,
                "model": deterministic_answer.usage.model,
                "input_tokens": deterministic_answer.usage.input_tokens,
                "output_tokens": deterministic_answer.usage.output_tokens,
                "estimated_cost_usd": deterministic_answer.usage.estimated_cost_usd,
                "status": "question_understanding",
            }
        )
    rendered_answer = deterministic_answer.text
    if use_gemini and deterministic_answer.status not in {
        "clarification",
        "not-found",
        "no-results",
        "invalid_input",
        "unsupported_request",
    }:
        context = build_grounding_context(question, deterministic_answer, output, audit, notes)
        try:
            gemini_response = generate_grounded_response(question, context, gemini_config)
            response_is_valid = validate_grounded_response(
                gemini_response.text, deterministic_answer, context
            )
            st.session_state.llm_usage_log.append(
                {
                    "provider": gemini_response.usage.provider,
                    "model": gemini_response.usage.model,
                    "input_tokens": gemini_response.usage.input_tokens,
                    "output_tokens": gemini_response.usage.output_tokens,
                    "estimated_cost_usd": gemini_response.usage.estimated_cost_usd,
                    "status": "validated" if response_is_valid else "rejected",
                }
            )
            if response_is_valid:
                rendered_answer = (
                    f"{gemini_response.text}\n\n"
                    "---\n**Verified deterministic record**\n"
                    f"{deterministic_answer.text}"
                )
            else:
                rendered_answer = (
                    f"{deterministic_answer.text}\n\n"
                    "_Gemini's wording was rejected because it did not preserve the verified provenance._"
                )
        except GeminiRequestError as error:
            usage = error.usage
            st.session_state.llm_usage_log.append(
                {
                    "provider": usage.provider,
                    "model": usage.model,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "estimated_cost_usd": usage.estimated_cost_usd,
                    "status": f"fallback: {error}",
                }
            )
            rendered_answer = f"{deterministic_answer.text}\n\n_Gemini unavailable; using the deterministic answer._"
    st.session_state.assistant_messages.append({"role": "assistant", "content": rendered_answer})
    st.rerun()

if st.session_state.llm_usage_log:
    st.caption(
        f"Optional LLM calls this session: {len(st.session_state.llm_usage_log)}. "
        "Token counts come from Gemini responses; unavailable values are not estimated."
    )
    st.dataframe(pd.DataFrame(st.session_state.llm_usage_log), use_container_width=True, hide_index=True)
    st.download_button(
        "Download this session's LLM usage log",
        data=json.dumps(st.session_state.llm_usage_log, indent=2),
        file_name="llm_usage_log.json",
        mime="application/json",
    )

st.subheader("Download")
st.download_button(
    "Download required flags.csv",
    data=(OUTPUTS_DIR / "flags.csv").read_bytes(),
    file_name="flags.csv",
    mime="text/csv",
)
