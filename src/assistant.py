"""Deterministic, evidence-grounded analyst question answering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import re
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .retrieval import NoteAssessment, assess_notes


MONTHS = {
    name.casefold(): number
    for number, name in enumerate(
        ("", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
    )
    if name
}
MONTH_ALIASES = {name[:3].casefold(): number for name, number in MONTHS.items()}
PLACE_ALIASES = {
    "bombay": "mumbai",
    "madras": "chennai",
    "bengaluru": "bangalore",
}

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

# Phrases that clearly signal a freight/cost analytics question.
# IMPORTANT: do NOT add generic question words (what, how, when, show, list,
# which, report, data, record) — they match off-topic questions and break the
# intent classifier's Gate-A domain-signal check.
_FREIGHT_SIGNALS = (
    "route", "cost", "spike", "anomal", "freight", "tonne", "week", "note",
    "evidence", "explain", "why", "increase", "pric", "expensive", "rise",
    "flag", "justified", "unexplained", "baseline", "system",
    "shipment", "transport", "distance", "metric", "trend",
    "compare", "reject", "support", "analys", "calcul",
)

# Patterns that identify clearly off-topic requests.
_OFFTOPIC_PATTERNS = (
    r"\b(poem|song|recipe|joke|story|essay|news|cricket|football|movie|music"
    r"|write a|summarize this article|tell me about history|politics"
    r"|weather forecast|stock price|share price)\b",
)

# Tokens that are real English/domain words (heuristic: 3+ chars, found in a
# simple word-shape check).  Used to gate the gibberish detector.
_MIN_REAL_TOKEN_LEN = 3


def _looks_like_gibberish(text: str) -> bool:
    """Return True when the message has no plausible natural-language tokens.

    Strategy
    --------
    A token is considered *implausible* (keyboard-mash) if it meets any of:
    1. All characters are identical.
    2. Vowel ratio (treating y as consonant) is outside [0.15, 0.60].
    3. Any single character accounts for > 35% of the token (repetitive mash
       like 'yhyjuyuyukuy' has 'y'=4/12=33%, 'u'=4/12=33% — flags both).
    4. The token contains a run of 3+ consecutive vowels (aeiou only).
    5. The token length is < 3.

    We return True (gibberish) only when *no* token in the input passes.
    Short common words like 'why', 'how', 'what' are pre-cleared via
    _COMMON_QUESTION_WORDS before this function is called.
    """
    tokens = re.findall(r"[A-Za-z]{3,}", text)
    if not tokens:
        return True

    VOWELS = set("aeiou")  # y is NOT counted as a vowel

    def _plausible_word(token: str) -> bool:
        lower = token.lower()
        n = len(lower)
        # Rule 1: all same char
        if len(set(lower)) == 1:
            return False
        # Rule 2: vowel ratio
        vowel_count = sum(1 for c in lower if c in VOWELS)
        ratio = vowel_count / n
        if ratio < 0.15 or ratio > 0.60:
            return False
        # Rule 3: any single character dominates (> 35%)
        from collections import Counter
        freq = Counter(lower)
        if freq.most_common(1)[0][1] / n > 0.35:
            return False
        # Rule 4: triple-vowel run
        if re.search(r"[aeiou]{3,}", lower):
            return False
        return True

    return not any(_plausible_word(t) for t in tokens)


# Supported intents returned by classify_intent.
INTENT_INVALID = "invalid_input"
INTENT_UNSUPPORTED = "unsupported_request"
INTENT_SYSTEM_OVERVIEW = "system_overview"
INTENT_LIST_SUPPORTED = "list_anomalies"
INTENT_LIST_UNEXPLAINED = "unexplained_anomalies"
INTENT_ROUTE_INVESTIGATION = "route_investigation"
INTENT_EVIDENCE_LOOKUP = "evidence_lookup"
INTENT_GENERAL = "general_supported_question"
def classify_intent(question: str) -> str:
    """Classify a user question into a supported intent without an LLM call.

    Runs purely on regex/heuristics so it is fast, free, and fully
    deterministic.  Returns one of the INTENT_* constants.

    Two-gate design
    ---------------
    Gate A — does the input look like a meaningful natural-language message?
      • Empty/whitespace → INTENT_INVALID immediately.
      • No domain signals AND _looks_like_gibberish → INTENT_INVALID.
      • No domain signals AND not gibberish AND off-topic → INTENT_UNSUPPORTED.
    Gate B — what freight-analytics intent does the message express?
      • Dispatches to system-overview, list, evidence, or route-investigation.
    """
    stripped = question.strip()
    if not stripped:
        return INTENT_INVALID

    lowered = stripped.casefold()

    # --- Gate A: domain-signal presence ---------------------------------
    # Check first so we don't misclassify real-freight questions as gibberish.
    has_domain_signal = any(signal in lowered for signal in _FREIGHT_SIGNALS) or bool(
        re.search(r"\b[A-Za-z]+(?:-[A-Za-z]+|\s+to\s+[A-Za-z]+)\b", stripped)
    )

    if not has_domain_signal:
        # No freight-related words at all.
        if _looks_like_gibberish(stripped):
            return INTENT_INVALID
        # Real words, but nothing to do with freight → off-topic or unsupported.
        combined_offtopic = "|".join(_OFFTOPIC_PATTERNS)
        if re.search(combined_offtopic, lowered):
            return INTENT_UNSUPPORTED
        # Plausible English, no freight words, no specific off-topic marker.
        return INTENT_UNSUPPORTED

    # --- Gate B: domain-intent dispatch ---------------------------------

    # System/definition questions
    if any(
        marker in lowered
        for marker in (
            "what is this system", "whole system", "what does this system",
            "explain the system", "explain the setup", "whole setup",
            "what is the assistant", "cost per tonne", "tonne-km",
            "how does this work", "what is this", "how does the system",
        )
    ):
        return INTENT_SYSTEM_OVERVIEW

    # List intents
    if (("which" in lowered or "list" in lowered or "show" in lowered) and any(
        w in lowered for w in ("supported", "evidence", "justified")
    )):
        return INTENT_LIST_SUPPORTED
    if ("which" in lowered or "list" in lowered or "show" in lowered) and "unexplained" in lowered:
        return INTENT_LIST_UNEXPLAINED

    # Evidence / note lookup
    if re.search(r"\bN\d+\b", stripped.upper()) or any(
        w in lowered for w in ("reject", "why was note", "note id", "evidence note")
    ):
        return INTENT_EVIDENCE_LOOKUP

    # Everything else with a domain signal is a route investigation
    return INTENT_ROUTE_INVESTIGATION

SYSTEM_DESCRIPTION = {
    "purpose": "Monitor weekly freight cost per tonne-km and investigate unusual increases.",
    "route_grouping": "Shipment records are grouped by directional route and the supplied route_type.",
    "week_definition": "Weeks run Monday through Sunday and week_of is the Monday date.",
    "metric": "cost_per_tonne_km = total freight cost / (total tonnes carried * distance in km).",
    "own_history_baseline": "The mean of up to the eight prior observed weeks for the same route, excluding the current week.",
    "peer_baseline": "The same-week mean of other routes with the same supplied route_type, excluding the current route.",
    "verdict_rule": "The configured deterministic detector flags a rising value that exceeds the configured 15% comparison threshold.",
    "evidence_rule": "A context note must match route/geography, overlap the week, support increased costs, and satisfy the triggered baseline scope.",
    "answer_labels": "Evidence-supported, possible explanation, and unexplained are kept distinct.",
}


@dataclass(frozen=True)
class AssistantAnswer:
    """A response plus machine-readable grounding metadata."""

    text: str
    status: str
    route: str | None = None
    week_of: pd.Timestamp | None = None
    note_ids: tuple[str, ...] = ()
    usage: Any | None = None
    weeks: tuple[pd.Timestamp, ...] = ()


@dataclass(frozen=True)
class TimeSelection:
    weeks: tuple[pd.Timestamp, ...] = ()
    error: str | None = None



def _normalise_results(output: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    results = output.copy()
    results["week_of"] = pd.to_datetime(results["week_of"], errors="coerce")
    if "week_of" in audit and "route" in audit:
        audit_copy = audit.copy()
        audit_copy["week_of"] = pd.to_datetime(audit_copy["week_of"], errors="coerce")
        audit_columns = [
            column
            for column in (
                "route",
                "week_of",
                "route_type",
                "anomaly",
                "trigger_baselines",
                "history_baseline",
                "peer_baseline",
                "evidence_scope",
                "evidence_category",
            )
            if column in audit_copy
        ]
        results = results.merge(
            audit_copy[audit_columns],
            on=["route", "week_of"],
            how="left",
            suffixes=("", "_audit"),
        )
    results["cost_per_tonne_km"] = pd.to_numeric(results["cost_per_tonne_km"], errors="coerce")
    return results


def _known_routes(results: pd.DataFrame) -> list[str]:
    return sorted(str(route) for route in results["route"].dropna().unique())


def _find_route(question: str, routes: Iterable[str]) -> tuple[str | None, str | None]:
    lowered = question.casefold().replace("–", "-").replace("—", "-")
    for source, replacement in sorted(PLACE_ALIASES.items(), key=lambda item: -len(item[0])):
        lowered = re.sub(rf"\b{source}\b", replacement, lowered)
    known = list(routes)
    matches = [route for route in known if route.casefold() in lowered]
    if matches:
        return max(matches, key=len), None

    for route in known:
        origin, destination = (part.casefold() for part in route.split("-", 1))
        if re.search(
            rf"\b{re.escape(origin)}\b\s*(?:-|to|towards|/|,)\s*\b{re.escape(destination)}\b",
            lowered,
        ):
            return route, None

    place_matches = [
        route
        for route in known
        if any(re.search(rf"\b{re.escape(place)}\b", lowered) for place in route.casefold().split("-"))
    ]
    if len(place_matches) == 1:
        return place_matches[0], None
    if len(place_matches) > 1:
        choices = ", ".join(f"`{route}`" for route in place_matches[:5])
        return None, f"I found multiple routes ({choices}). Which exact route should I inspect?"
    tokens = re.findall(r"\b[A-Za-z]+-[A-Za-z]+\b", question)
    if tokens:
        return None, f"I could not find route `{tokens[0]}` in the computed results."
    return None, None


def _month_mentions(question: str) -> list[tuple[int, int, int]]:
    names = sorted({*MONTHS, *MONTH_ALIASES}, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(name) for name in names) + r")\b"
    return [
        (MONTHS.get(match.group(1), MONTH_ALIASES.get(match.group(1))), match.start(), match.end())
        for match in re.finditer(pattern, question.casefold())
    ]


def _available_weeks(route_rows: pd.DataFrame) -> tuple[pd.Timestamp, ...]:
    return tuple(sorted(pd.Timestamp(value) for value in route_rows["week_of"].dropna().unique()))


def _year_options(route_rows: pd.DataFrame, month_number: int) -> list[int]:
    weeks = pd.to_datetime(route_rows["week_of"], errors="coerce")
    return sorted(weeks[weeks.dt.month.eq(month_number)].dt.year.dropna().astype(int).unique())


def _year_clarification(month_number: int, years: Sequence[int]) -> str:
    month_name = pd.Timestamp(year=2000, month=month_number, day=1).strftime("%B")
    joined = ", ".join(str(year) for year in years)
    return f"I found {month_name} in {joined}. Which year should I inspect?"


def _normalise_week(value: object) -> pd.Timestamp | None:
    target = pd.to_datetime(value, errors="coerce")
    if pd.isna(target):
        return None
    return pd.Timestamp(target) - timedelta(days=int(pd.Timestamp(target).dayofweek))


def _month_weeks(route_rows: pd.DataFrame, month_number: int, year: int | None) -> tuple[pd.Timestamp, ...]:
    weeks = pd.to_datetime(route_rows["week_of"], errors="coerce")
    mask = weeks.dt.month.eq(month_number)
    if year is not None:
        mask &= weeks.dt.year.eq(year)
    return tuple(sorted(pd.Timestamp(value) for value in weeks[mask].unique()))


def _selection_from_weeks(
    weeks: Iterable[pd.Timestamp],
    route_rows: pd.DataFrame,
    description: str,
) -> TimeSelection:
    available = set(_available_weeks(route_rows))
    selected = tuple(sorted({pd.Timestamp(week) for week in weeks if pd.Timestamp(week) in available}))
    if selected:
        return TimeSelection(selected)
    return TimeSelection(error=f"I found no computed route-week record for {description}.")


def _find_time_selection(
    question: str,
    route_rows: pd.DataFrame,
    *,
    previous_week: pd.Timestamp | None = None,
) -> TimeSelection:
    """Resolve a natural-language time expression to verified route weeks."""

    lowered = question.casefold()
    available = _available_weeks(route_rows)
    if not available:
        return TimeSelection(error="There are no computed weeks for this route.")

    if re.search(r"\b(next|following)\s+week\b", lowered) or re.search(r"\bweek\s+after\b", lowered):
        if previous_week is None:
            return TimeSelection(error="Which earlier week should I use as the reference for the next week?")
        return _selection_from_weeks(
            [previous_week + timedelta(days=7)], route_rows, "the next week in the conversation"
        )
    if re.search(r"\b(previous|prior)\s+week\b", lowered) or re.search(r"\bweek\s+before\b", lowered):
        if previous_week is None:
            return TimeSelection(error="Which earlier week should I use as the reference for the previous week?")
        return _selection_from_weeks(
            [previous_week - timedelta(days=7)], route_rows, "the previous week in the conversation"
        )

    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", question)
    if iso_dates:
        parsed = [_normalise_week(value) for value in iso_dates]
        if any(value is None for value in parsed):
            return TimeSelection(error="That date is not valid. Please provide a real date.")
        if len(parsed) == 1:
            return _selection_from_weeks(parsed, route_rows, f"week `{parsed[0]:%Y-%m-%d}`")
        start, end = sorted(parsed[:2])
        return _selection_from_weeks(
            [week for week in available if start <= week <= end],
            route_rows,
            f"the range `{start:%Y-%m-%d}` to `{end:%Y-%m-%d}`",
        )

    relative_match = re.search(r"\b(\d+)\s+months?\s+ago\b", lowered)
    if relative_match:
        target = pd.Timestamp.today().normalize() - pd.DateOffset(months=int(relative_match.group(1)))
        target = _normalise_week(target)
        return _selection_from_weeks([target], route_rows, f"week `{target:%Y-%m-%d}`")
    if re.search(r"\byesterday\b|\btoday\b|\bthis week\b|\blast week\b", lowered):
        target = previous_week - timedelta(days=7) if previous_week is not None and "last week" in lowered else pd.Timestamp.today().normalize()
        target = _normalise_week(target)
        return _selection_from_weeks([target], route_rows, f"week `{target:%Y-%m-%d}`")

    mentions = _month_mentions(lowered)
    years = [int(value) for value in re.findall(r"\b(20\d{2})\b", lowered)]
    if len(mentions) >= 2 and re.search(r"\b(from|between|through|to|until|and)\b", lowered):
        first_month = mentions[0][0]
        second_month = mentions[1][0]
        if len(set(years)) > 1:
            first_year, second_year = years[0], years[-1]
        elif years:
            first_year = second_year = years[0]
        else:
            first_options = _year_options(route_rows, first_month)
            second_options = _year_options(route_rows, second_month)
            common_years = sorted(set(first_options).intersection(second_options))
            if len(common_years) != 1:
                return TimeSelection(error=_year_clarification(first_month, common_years or first_options or second_options))
            first_year = second_year = common_years[0]
        start_weeks = _month_weeks(route_rows, first_month, first_year)
        end_weeks = _month_weeks(route_rows, second_month, second_year)
        if not start_weeks or not end_weeks:
            return TimeSelection(error="I found no computed route-week records for that date range.")
        start, end = min(start_weeks), max(end_weeks)
        return _selection_from_weeks(
            [week for week in available if start <= week <= end],
            route_rows,
            f"the range `{start:%Y-%m-%d}` to `{end:%Y-%m-%d}`",
        )

    if not mentions:
        return TimeSelection()

    month_number = mentions[0][0]
    month_year_options = _year_options(route_rows, month_number)
    if years:
        year = years[0]
    elif len(month_year_options) == 1:
        year = month_year_options[0]
    elif len(month_year_options) > 1:
        return TimeSelection(error=_year_clarification(month_number, month_year_options))
    else:
        year = None

    candidates = list(_month_weeks(route_rows, month_number, year))
    if not candidates:
        return TimeSelection(error=f"I found no computed route-week records in {pd.Timestamp(year=2000, month=month_number, day=1):%B}.")

    selector = re.search(r"\b(first|last)\s+(?:monday|week)\b", lowered)
    if selector:
        return TimeSelection((candidates[0] if selector.group(1) == "first" else candidates[-1],))
    if re.search(r"\bearly\b", lowered):
        candidates = [week for week in candidates if week.day <= 14]
    elif re.search(r"\blate\b", lowered):
        candidates = [week for week in candidates if week.day >= 15]
    if not candidates:
        return TimeSelection(error="I found no computed route-week records in that part of the month.")
    return TimeSelection(tuple(candidates))


def _find_week(question: str, route_rows: pd.DataFrame) -> tuple[pd.Timestamp | None, str | None]:
    selection = _find_time_selection(question, route_rows)
    if selection.error:
        return None, selection.error
    if len(selection.weeks) == 1:
        return selection.weeks[0], None
    if len(selection.weeks) > 1:
        weeks = ", ".join(week.strftime("%Y-%m-%d") for week in selection.weeks[:5])
        return None, f"I found multiple matching weeks ({weeks}). Which Monday week should I inspect?"
    return None, None


def _format_row_answer(row: pd.Series, notes: pd.DataFrame) -> AssistantAnswer:
    route = str(row["route"])
    week = pd.Timestamp(row["week_of"])
    status = str(row["flagged"])
    observed = (
        f"Observed result for `{route}` during week `{week:%Y-%m-%d}`: "
        f"cost per tonne-km was `{float(row['cost_per_tonne_km']):.2f}`. "
        f"The route comparison was `{row['vs_own_history']}` and the peer comparison was `"
        f"{row['vs_similar_routes']}`."
    )
    note_id = str(row.get("matched_note_id", ""))
    if status == "No (justified)" and note_id:
        note_rows = notes[notes["note_id"].eq(note_id)]
        if len(note_rows) == 1:
            note = note_rows.iloc[0]
            start = pd.Timestamp(note["effective_start"]).strftime("%Y-%m-%d")
            end = pd.Timestamp(note["effective_end"]).strftime("%Y-%m-%d")
            text = (
                f"{observed}\n\n**Evidence-supported.** Matched note `{note_id}` "
                f"(applies to `{note['applies_to']}`, effective `{start}` to `{end}`): "
                f"{note['note']}"
            )
            return AssistantAnswer(text, "evidence-supported", route, week, (note_id,), weeks=(week,))
    if status == "Yes":
        return AssistantAnswer(
            f"{observed}\n\n**Unexplained.** No validated context note supports this increase for the route, week, and triggered baseline. I cannot confirm a cause from the supplied evidence.",
            "unexplained",
            route,
            week,
            weeks=(week,),
        )
    return AssistantAnswer(
        f"{observed}\n\n**No anomaly verdict.** This record is not flagged by the configured deterministic rule.",
        "not-anomaly",
        route,
        week,
        weeks=(week,),
    )


def _format_rows_answer(rows: pd.DataFrame, notes: pd.DataFrame) -> AssistantAnswer:
    """Summarise a verified month or date range without inventing causes."""

    ordered = rows.sort_values("week_of")
    route = str(ordered.iloc[0]["route"])
    weeks = tuple(pd.Timestamp(value) for value in ordered["week_of"])
    parts = [f"**Verified route-week results for `{route}` ({len(ordered)} weeks)**"]
    note_ids: list[str] = []
    statuses: set[str] = set()
    for _, row in ordered.iterrows():
        answer = _format_row_answer(row, notes)
        parts.append(answer.text)
        statuses.add(answer.status)
        for note_id in answer.note_ids:
            if note_id not in note_ids:
                note_ids.append(note_id)
    if "unexplained" in statuses:
        status = "unexplained"
    elif "evidence-supported" in statuses:
        status = "evidence-supported"
    else:
        status = "summary"
    return AssistantAnswer(
        "\n\n".join(parts),
        status,
        route,
        weeks[0] if len(weeks) == 1 else None,
        tuple(note_ids),
        weeks=weeks,
    )


def _list_answers(results: pd.DataFrame, supported: bool) -> AssistantAnswer:
    if supported:
        rows = results[results["flagged"].eq("No (justified)")]
        if rows.empty:
            return AssistantAnswer("No anomalies currently have validated supporting notes.", "no-results")
        lines = [
            f"- `{row.route}`, week `{pd.Timestamp(row.week_of):%Y-%m-%d}`, note `{row.matched_note_id}`"
            for row in rows.itertuples()
        ]
        return AssistantAnswer("**Evidence-supported anomalies:**\n" + "\n".join(lines), "evidence-supported", note_ids=tuple(rows["matched_note_id"]))
    rows = results[results["flagged"].eq("Yes")]
    if rows.empty:
        return AssistantAnswer("No unexplained anomalies are currently present.", "no-results")
    lines = [
        f"- `{row.route}`, week `{pd.Timestamp(row.week_of):%Y-%m-%d}`"
        for row in rows.itertuples()
    ]
    return AssistantAnswer(
        f"**Unexplained anomalies ({len(rows)}):**\n" + "\n".join(lines),
        "unexplained",
    )


def _system_overview_answer() -> AssistantAnswer:
    return AssistantAnswer(
        "**What this system does**\n\n"
        "It monitors shipping costs week by week and highlights route-weeks that look unusually expensive. "
        "It groups shipments by directional route and the supplied `route_type`, then calculates "
        "`total freight cost / (tonnes carried × distance in km)` as cost per tonne-km.\n\n"
        "For each route-week, it compares the result with that route's previous eight available weeks "
        "and with other routes of the same `route_type` during the same week. The deterministic detector "
        "makes the anomaly decision; the evidence retriever separately checks whether a context note "
        "actually supports the increase.\n\n"
        "The assistant can explain observed numbers, cite validated note IDs, explain rejected notes, "
        "and say when the supplied evidence is insufficient. An optional Gemini layer can improve wording, "
        "but it cannot change metrics, verdicts, or evidence.",
        "overview",
    )


def _rejection_answer(question: str, route: str, week: pd.Timestamp, notes: pd.DataFrame) -> AssistantAnswer:
    note_ids = re.findall(r"\bN\d+\b", question.upper())
    if not note_ids:
        available_ids = sorted(notes["note_id"].dropna().unique())
        example_id = available_ids[0] if available_ids else "N001"
        return AssistantAnswer(
            f"Which note ID should I inspect? Available note IDs include: {', '.join(f'`{n}`' for n in available_ids[:6])}.",
            "clarification",
            route,
            week,
            weeks=(week,),
        )
    note_id = note_ids[0]
    assessments = assess_notes(notes, route=route, week_of=week)
    matching = [assessment for assessment in assessments if assessment.note_id == note_id]
    if not matching:
        return AssistantAnswer(
            f"I could not find note `{note_id}` in `context_notes.csv`.",
            "not-found",
            route,
            week,
            weeks=(week,),
        )
    assessment = matching[0]
    verdict = "accepted" if assessment.accepted else "rejected"
    return AssistantAnswer(
        f"For `{route}` in week `{week:%Y-%m-%d}`, note `{note_id}` is **{verdict}**. {assessment.reason}",
        "evidence-supported" if assessment.accepted else "unexplained",
        route,
        week,
        (note_id,),
        weeks=(week,),
    )


def build_grounding_context(
    question: str,
    answer: AssistantAnswer,
    output: pd.DataFrame,
    audit: pd.DataFrame,
    notes: pd.DataFrame,
) -> dict[str, Any]:
    """Build the small verified context that an optional LLM may see."""

    results = _normalise_results(output, audit)
    context: dict[str, Any] = {
        "system_rules": SYSTEM_DESCRIPTION,
        "question": question,
        "deterministic_status": answer.status,
        "deterministic_answer": answer.text,
        "allowed_note_ids": list(answer.note_ids),
    }
    selected_weeks = answer.weeks or ((answer.week_of,) if answer.week_of is not None else ())
    if answer.route and selected_weeks:
        row = results[
            results["route"].eq(answer.route)
            & results["week_of"].isin([pd.Timestamp(week) for week in selected_weeks])
        ].sort_values("week_of")
        if not row.empty:
            fields = (
                "route",
                "week_of",
                "route_type",
                "cost_per_tonne_km",
                "vs_own_history",
                "vs_similar_routes",
                "flagged",
                "matched_note_id",
                "reason",
                "trigger_baselines",
                "evidence_category",
            )
            verified_rows = []
            for _, selected in row.iterrows():
                record = {}
                for field in fields:
                    if field not in selected or pd.isna(selected[field]):
                        continue
                    value = selected[field]
                    record[field] = (
                        pd.Timestamp(value).strftime("%Y-%m-%d")
                        if field == "week_of"
                        else str(value)
                    )
                verified_rows.append(record)
            context["verified_route_weeks"] = verified_rows
            if len(verified_rows) == 1:
                context["verified_route_week"] = verified_rows[0]
    if answer.note_ids:
        selected_notes = notes[notes["note_id"].isin(answer.note_ids)]
        context["verified_notes"] = [
            {
                "note_id": str(row["note_id"]),
                "applies_to": str(row["applies_to"]),
                "effective_start": pd.Timestamp(row["effective_start"]).strftime("%Y-%m-%d"),
                "effective_end": pd.Timestamp(row["effective_end"]).strftime("%Y-%m-%d"),
                "note": str(row["note"]),
            }
            for _, row in selected_notes.iterrows()
        ]
    return context


def _conversation_reference(
    conversation_context: Sequence[Mapping[str, str]] | None,
    results: pd.DataFrame,
    routes: list[str],
) -> tuple[str | None, pd.Timestamp | None]:
    """Recover only the last verified route/week needed for a follow-up."""

    if not conversation_context:
        return None, None
    prior_questions = [
        str(message.get("content", ""))
        for message in conversation_context
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ]
    if not prior_questions:
        return None, None
    prior_question = prior_questions[-1]
    prior_route, _ = _find_route(prior_question, routes)
    if prior_route is None:
        return None, None
    prior_rows = results[results["route"].eq(prior_route)]
    prior_week, _ = _find_week(prior_question, prior_rows)
    return prior_route, prior_week


def _is_contextual_follow_up(question: str) -> bool:
    lowered = question.casefold().strip()
    return bool(
        re.search(r"^(and|also|then)\b", lowered)
        or re.search(r"\bwhat about\b|\bhow about\b", lowered)
        or re.search(r"\b(same route|same week|that week|that route|it|there)\b", lowered)
    )


def _with_usage(answer: AssistantAnswer, usage: Any | None) -> AssistantAnswer:
    if usage is None:
        return answer
    return AssistantAnswer(
        answer.text,
        answer.status,
        answer.route,
        answer.week_of,
        answer.note_ids,
        usage=usage,
        weeks=answer.weeks,
    )


def answer_question(
    question: str,
    output: pd.DataFrame,
    audit: pd.DataFrame,
    notes: pd.DataFrame,
    gemini_config: Any | None = None,
    conversation_context: Sequence[Mapping[str, str]] | None = None,
) -> AssistantAnswer:
    """Answer supported analyst questions using only verified local results.

    When ``gemini_config`` is provided and configured, natural question
    understanding (intent classification and entity extraction) is performed
    via Gemini, with automatic fallback to the deterministic regex/rules
    classifier if Gemini is unavailable, unconfigured, or encounters an error.
    """

    question = question.strip()
    if not question:
        return AssistantAnswer(
            "Please ask a question about a route, week, anomaly, or evidence note.",
            INTENT_INVALID,
        )

    results = _normalise_results(output, audit)
    routes = _known_routes(results)
    lowered = question.casefold()
    prior_route, prior_week = _conversation_reference(conversation_context, results, routes)

    gemini_understanding = None
    if gemini_config is not None and getattr(gemini_config, "configured", False):
        try:
            from .gemini import understand_question

            gemini_understanding = understand_question(
                question,
                routes,
                gemini_config,
                conversation_context=conversation_context,
            )
        except Exception:
            gemini_understanding = None

    if gemini_understanding is not None:
        intent = gemini_understanding.intent
        if not gemini_understanding.valid:
            if intent not in (INTENT_INVALID, INTENT_UNSUPPORTED):
                intent = INTENT_INVALID
    else:
        intent = classify_intent(question)

    if (
        prior_route
        and _is_contextual_follow_up(question)
        and intent in {INTENT_INVALID, INTENT_UNSUPPORTED}
    ):
        intent = INTENT_ROUTE_INVESTIGATION

    understanding_usage = getattr(gemini_understanding, "usage", None)

    # ------------------------------------------------------------------
    # Hard-stop intents: no route/week lookup should ever happen here.
    # ------------------------------------------------------------------
    if intent == INTENT_INVALID:
        return AssistantAnswer(
            "I couldn't understand that question. "
            "Try asking about a route, a cost increase, an anomaly, or an evidence note. "
            'For example: "Why did Chennai-Bangalore spike in late February?"',
            INTENT_INVALID,
            usage=understanding_usage,
        )

    if intent == INTENT_UNSUPPORTED:
        return AssistantAnswer(
            "I'm designed for freight-cost analysis and can only answer questions about "
            "routes, cost anomalies, weekly metrics, and evidence notes. "
            "I can't help with that request.",
            INTENT_UNSUPPORTED,
            usage=understanding_usage,
        )

    # ------------------------------------------------------------------
    # System overview
    # ------------------------------------------------------------------
    if intent == INTENT_SYSTEM_OVERVIEW:
        ans = _system_overview_answer()
        if understanding_usage:
            return _with_usage(ans, understanding_usage)
        return ans

    # ------------------------------------------------------------------
    # List intents — also honour the legacy marker phrases directly
    # ------------------------------------------------------------------
    if intent == INTENT_LIST_SUPPORTED or (
        ("which" in lowered or "list" in lowered or "show" in lowered)
        and any(word in lowered for word in ("supported", "evidence", "justified"))
    ):
        ans = _list_answers(results, supported=True)
        if understanding_usage:
            return _with_usage(ans, understanding_usage)
        return ans

    if intent == INTENT_LIST_UNEXPLAINED or (
        ("which" in lowered or "list" in lowered or "show" in lowered)
        and "unexplained" in lowered
    ):
        ans = _list_answers(results, supported=False)
        if understanding_usage:
            return _with_usage(ans, understanding_usage)
        return ans

    # ------------------------------------------------------------------
    # Route / evidence investigation — now perform route + week lookup
    # ------------------------------------------------------------------
    route = None
    if gemini_understanding and gemini_understanding.route in routes:
        route = gemini_understanding.route
    else:
        route, route_error = _find_route(question, routes)
        if route_error:
            return AssistantAnswer(route_error, "not-found", usage=understanding_usage)
        if route is None and prior_route:
            route = prior_route

    if route is None:
        # The question had freight-domain intent but no identifiable route.
        # Ask for a route rather than rejecting as invalid.
        route_list = ", ".join(f"`{r}`" for r in routes[:5])
        return AssistantAnswer(
            f"Which route should I inspect? Known routes include: {route_list}. Please name one.",
            "clarification",
            usage=understanding_usage,
        )

    route_rows = results[results["route"].eq(route)].sort_values("week_of")
    explanation_question = any(
        word in lowered for word in ("why", "spike", "pricier", "expensive", "increase", "rise", "anomal")
    )
    resolution_rows = route_rows
    has_exact_week_expression = bool(
        re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
        or re.search(r"\b\d+\s+months?\s+ago\b", lowered)
        or re.search(r"\b(today|yesterday|last week|this week|next week|previous week)\b", lowered)
    )
    if explanation_question and not has_exact_week_expression:
        anomaly_rows = route_rows[route_rows["flagged"].isin(["Yes", "No (justified)"])]
        if not anomaly_rows.empty:
            resolution_rows = anomaly_rows

    selection = _find_time_selection(question, resolution_rows, previous_week=prior_week)
    if not selection.weeks and not selection.error and gemini_understanding:
        time_hint = getattr(gemini_understanding, "time_hint", None) or getattr(
            gemini_understanding, "week_hint", None
        )
        if time_hint:
            selection = _find_time_selection(time_hint, resolution_rows, previous_week=prior_week)
        if not selection.weeks and not selection.error:
            start = getattr(gemini_understanding, "date_start_iso", None)
            end = getattr(gemini_understanding, "date_end_iso", None)
            if start and end:
                selection = _find_time_selection(f"{start} to {end}", resolution_rows, previous_week=prior_week)
            elif getattr(gemini_understanding, "week_iso", None):
                selection = _find_time_selection(gemini_understanding.week_iso, resolution_rows)

    if selection.error:
        return AssistantAnswer(selection.error, "clarification", route, usage=understanding_usage)

    if not selection.weeks:
        return AssistantAnswer(
            "Which Monday week should I inspect? You can provide a date such as `2025-02-24`.",
            "clarification",
            route,
            usage=understanding_usage,
        )

    selected_rows = route_rows[route_rows["week_of"].isin(selection.weeks)]
    if len(selected_rows) > 1:
        if "reject" in lowered or ("why was" in lowered and "note" in lowered):
            return AssistantAnswer(
                "Which single Monday week should I use for the note rejection check?",
                "clarification",
                route,
                usage=understanding_usage,
            )
        return _with_usage(_format_rows_answer(selected_rows, notes), understanding_usage)

    week = selection.weeks[0]
    row = selected_rows
    if row.empty:
        return AssistantAnswer(
            f"I found no computed result for `{route}` in week `{week:%Y-%m-%d}`.",
            "not-found",
            route,
            week,
            usage=understanding_usage,
            weeks=(week,),
        )

    if "reject" in lowered or ("why was" in lowered and "note" in lowered):
        ans = _rejection_answer(question, route, week, notes)
        if understanding_usage:
            return _with_usage(ans, understanding_usage)
        return ans

    ans = _format_row_answer(row.iloc[0], notes)
    return _with_usage(ans, understanding_usage)
