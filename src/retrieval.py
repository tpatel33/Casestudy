"""Evidence-first, deterministic retrieval and validation for context notes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import calendar
import re
from datetime import timedelta

import pandas as pd


NOTE_COLUMNS = ("note_id", "date", "applies_to", "note")
CATEGORY_BY_NOTE = {
    "N001": "weather/flood",
    "N002": "festival surcharge",
    "N003": "fuel",
}
UNSUPPORTED_PATTERNS = (
    "not significantly affected",
    "no major disruptions",
    "remained normal",
    "without a rate change",
    "not part of this dataset",
    "improved road conditions",
)
IMPACT_TERMS = (
    "higher trip costs",
    "surcharge",
    "pushing up transportation costs",
    "costs",
    "cost",
    "rate increase",
    "rate rose",
)
MONTHS = {name.casefold()[:3]: number for number, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.casefold()[:3]: number for number, name in enumerate(calendar.month_abbr) if name})


class NoteValidationError(ValueError):
    """Raised when the context-note source is structurally unusable."""


@dataclass(frozen=True)
class EvidenceMatch:
    """A validated note and the baseline scope it can support."""

    note_id: str
    date: pd.Timestamp
    applies_to: str
    note: str
    supports_peer_relative: bool


@dataclass(frozen=True)
class NoteAssessment:
    """Auditable applicability checks for one note and one route-week query."""

    note_id: str
    applies_to: str
    route_applicable: bool
    date_applicable: bool
    content_supports_cost: bool
    baseline_scope_supported: bool
    accepted: bool
    reason: str


def load_context_notes(path: str | Path) -> pd.DataFrame:
    """Load notes with explicit schema, date, ID, and text validation."""

    source = Path(path)
    if not source.is_file():
        raise NoteValidationError(f"Context-note file does not exist: {source}")
    notes = pd.read_csv(source, dtype="string", keep_default_na=False)
    missing = [column for column in NOTE_COLUMNS if column not in notes]
    if missing:
        raise NoteValidationError("Context notes are missing columns: " + ", ".join(missing))
    notes = notes.loc[:, list(NOTE_COLUMNS)].copy()
    for column in NOTE_COLUMNS:
        notes[column] = notes[column].astype("string").str.strip()
    blank = notes["note_id"].eq("") | notes["date"].eq("") | notes["applies_to"].eq("") | notes[
        "note"
    ].eq("")
    if blank.any():
        raise NoteValidationError(f"Context notes contain {int(blank.sum())} incomplete records")
    if notes["note_id"].duplicated().any():
        raise NoteValidationError("Context notes contain duplicate note IDs")
    parsed_dates = pd.to_datetime(notes["date"], format="%Y-%m-%d", errors="coerce")
    if parsed_dates.isna().any():
        raise NoteValidationError("Context notes contain invalid dates")
    notes["date"] = parsed_dates
    windows = notes.apply(
        lambda row: _effective_window(pd.Timestamp(row["date"]), str(row["note"])), axis=1
    )
    notes["effective_start"] = [window[0] for window in windows]
    notes["effective_end"] = [window[1] for window in windows]
    return notes


def _route_applicable(applies_to: str, route: str) -> bool:
    return applies_to.casefold() in {route.casefold(), "all routes"}


def _effective_window(note_date: pd.Timestamp, note_text: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    match = re.search(
        r"from\s+(?P<start_month>[A-Za-z]+)\s+(?P<start_day>\d{1,2})\s+to\s+"
        r"(?P<end_month>[A-Za-z]+)\s+(?P<end_day>\d{1,2})",
        note_text,
        flags=re.IGNORECASE,
    )
    if not match:
        normalized = note_date.normalize()
        return normalized, normalized
    try:
        start = pd.Timestamp(
            year=note_date.year,
            month=MONTHS[match.group("start_month").casefold()[:3]],
            day=int(match.group("start_day")),
        )
        end_year = note_date.year
        end_month = MONTHS[match.group("end_month").casefold()[:3]]
        if end_month < start.month:
            end_year += 1
        end = pd.Timestamp(
            year=end_year,
            month=end_month,
            day=int(match.group("end_day")),
        )
    except (KeyError, ValueError):
        normalized = note_date.normalize()
        return normalized, normalized
    return start, end


def _supports_cost_impact(note: str) -> bool:
    lowered = note.casefold()
    if any(pattern in lowered for pattern in UNSUPPORTED_PATTERNS):
        return False
    return any(term in lowered for term in IMPACT_TERMS)


def assess_notes(
    notes: pd.DataFrame,
    *,
    route: str,
    week_of: pd.Timestamp,
    trigger_baselines: str = "own_history",
) -> list[NoteAssessment]:
    """Explain why each note is or is not valid for a route-week query."""

    target_week = pd.Timestamp(week_of)
    target_end = target_week + timedelta(days=6)
    assessments: list[NoteAssessment] = []
    needs_peer_explanation = "similar_routes" in trigger_baselines.split(",")
    for _, note in notes.iterrows():
        route_applicable = _route_applicable(str(note["applies_to"]), route)
        date_applicable = bool(
            note["effective_start"] <= target_end and note["effective_end"] >= target_week
        )
        content_supports_cost = _supports_cost_impact(str(note["note"]))
        exact_route = str(note["applies_to"]).casefold() == route.casefold()
        baseline_scope_supported = not needs_peer_explanation or exact_route
        if not route_applicable:
            reason = "The note applies to a different route or geography."
        elif not date_applicable:
            reason = "The note's effective date window does not overlap this route-week."
        elif not content_supports_cost:
            reason = "The note does not support an increased cost and may state that conditions were normal or unaffected."
        elif not baseline_scope_supported:
            reason = "A nationwide note may support general history, but cannot explain a peer-relative increase."
        else:
            reason = "The note matches the route, date window, cost-impact content, and requested baseline scope."
        assessments.append(
            NoteAssessment(
                note_id=str(note["note_id"]),
                applies_to=str(note["applies_to"]),
                route_applicable=route_applicable,
                date_applicable=date_applicable,
                content_supports_cost=content_supports_cost,
                baseline_scope_supported=baseline_scope_supported,
                accepted=route_applicable
                and date_applicable
                and content_supports_cost
                and baseline_scope_supported,
                reason=reason,
            )
        )
    return assessments


def find_supporting_note(
    notes: pd.DataFrame,
    *,
    route: str,
    week_of: pd.Timestamp,
    trigger_baselines: str,
) -> EvidenceMatch | None:
    """Return the highest-ranked valid note for one anomalous route-week.

    Exact route notes can support both historical and peer-relative increases.
    ``All Routes`` notes can support a general historical rise, but cannot
    explain why one route is more expensive than its same-week peers.
    """

    if not trigger_baselines:
        return None
    target_week = pd.Timestamp(week_of)
    target_end = target_week + timedelta(days=6)
    candidates = notes[
        notes["effective_start"].le(target_end)
        & notes["effective_end"].ge(target_week)
        & notes["applies_to"].map(lambda value: _route_applicable(value, route))
        & notes["note"].map(_supports_cost_impact)
    ].copy()
    if candidates.empty:
        return None

    needs_peer_explanation = "similar_routes" in trigger_baselines.split(",")
    candidates["exact_route"] = candidates["applies_to"].str.casefold().eq(route.casefold())
    if needs_peer_explanation:
        candidates = candidates[candidates["exact_route"]]
    if candidates.empty:
        return None

    candidates["impact_score"] = candidates["note"].str.casefold().map(
        lambda text: sum(text.count(term) for term in IMPACT_TERMS)
    )
    candidates = candidates.sort_values(
        ["exact_route", "impact_score", "note_id"], ascending=[False, False, True], kind="stable"
    )
    match = candidates.iloc[0]
    return EvidenceMatch(
        note_id=str(match["note_id"]),
        date=pd.Timestamp(match["date"]),
        applies_to=str(match["applies_to"]),
        note=str(match["note"]),
        supports_peer_relative=bool(match["exact_route"]),
    )
