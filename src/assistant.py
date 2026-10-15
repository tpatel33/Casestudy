"""Deterministic, evidence-grounded analyst question answering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import re
from typing import Any, Iterable

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
    has_domain_signal = any(signal in lowered for signal in _FREIGHT_SIGNALS)

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
    lowered = question.casefold()
    for source, replacement in PLACE_ALIASES.items():
        lowered = re.sub(rf"\b{source}\b", replacement, lowered)
    known = list(routes)
    matches = [route for route in known if route.casefold() in lowered]
    if matches:
        return max(matches, key=len), None
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


def _find_week(question: str, route_rows: pd.DataFrame) -> tuple[pd.Timestamp | None, str | None]:
    iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", question)
    if iso_dates:
        target = pd.to_datetime(iso_dates[0], format="%Y-%m-%d", errors="coerce")
        if pd.isna(target):
            return None, "That date is not valid. Please provide a Monday week date such as `2025-02-24`."
        target = target - timedelta(days=int(target.dayofweek))
        if target in set(route_rows["week_of"]):
            return target, None
        return None, f"I found no computed route-week record for week `{target:%Y-%m-%d}`."

    lowered = question.casefold()
    relative_match = re.search(r"\b(\d+)\s+months?\s+ago\b", lowered)
    if relative_match:
        target = pd.Timestamp.today().normalize() - pd.DateOffset(months=int(relative_match.group(1)))
        target = target - timedelta(days=int(target.dayofweek))
        if target in set(route_rows["week_of"]):
            return target, None
        return None, f"I found no computed route-week record for week `{target:%Y-%m-%d}`."
    if re.search(r"\byesterday\b", lowered) or re.search(r"\btoday\b", lowered):
        target = pd.Timestamp.today().normalize()
        target = target - timedelta(days=int(target.dayofweek))
        if target in set(route_rows["week_of"]):
            return target, None
        return None, f"I found no computed route-week record for this week's Monday `{target:%Y-%m-%d}`."
    month_number = next(
        (number for name, number in {**MONTHS, **MONTH_ALIASES}.items() if re.search(rf"\b{name}\b", lowered)),
        None,
    )
    if month_number is None:
        return None, None
    candidates = route_rows[route_rows["week_of"].dt.month.eq(month_number)]
    if "late" in lowered:
        candidates = candidates[candidates["week_of"].dt.day.ge(15)]
    if len(candidates) == 1:
        return candidates.iloc[0]["week_of"], None
    if len(candidates) > 1:
        weeks = ", ".join(pd.Timestamp(value).strftime("%Y-%m-%d") for value in candidates["week_of"][:5])
        return None, f"I found multiple matching weeks ({weeks}). Which Monday week should I inspect?"
    return None, f"I found no computed route-week record in {month_number:02d}."


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
            return AssistantAnswer(text, "evidence-supported", route, week, (note_id,))
    if status == "Yes":
        return AssistantAnswer(
            f"{observed}\n\n**Unexplained.** No validated context note supports this increase for the route, week, and triggered baseline. I cannot confirm a cause from the supplied evidence.",
            "unexplained",
            route,
            week,
        )
    return AssistantAnswer(
        f"{observed}\n\n**No anomaly verdict.** This record is not flagged by the configured deterministic rule.",
        "not-anomaly",
        route,
        week,
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
        )
    note_id = note_ids[0]
    assessments = assess_notes(notes, route=route, week_of=week)
    matching = [assessment for assessment in assessments if assessment.note_id == note_id]
    if not matching:
        return AssistantAnswer(f"I could not find note `{note_id}` in `context_notes.csv`.", "not-found", route, week)
    assessment = matching[0]
    verdict = "accepted" if assessment.accepted else "rejected"
    return AssistantAnswer(
        f"For `{route}` in week `{week:%Y-%m-%d}`, note `{note_id}` is **{verdict}**. {assessment.reason}",
        "evidence-supported" if assessment.accepted else "unexplained",
        route,
        week,
        (note_id,),
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
    if answer.route and answer.week_of is not None:
        row = results[
            results["route"].eq(answer.route)
            & results["week_of"].eq(pd.Timestamp(answer.week_of))
        ]
        if not row.empty:
            selected = row.iloc[0]
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
            context["verified_route_week"] = {
                field: str(selected[field])
                for field in fields
                if field in selected and pd.notna(selected[field])
            }
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


def answer_question(
    question: str,
    output: pd.DataFrame,
    audit: pd.DataFrame,
    notes: pd.DataFrame,
    gemini_config: Any | None = None,
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

    gemini_understanding = None
    if gemini_config is not None and getattr(gemini_config, "configured", False):
        try:
            from .gemini import understand_question

            gemini_understanding = understand_question(question, routes, gemini_config)
        except Exception:
            gemini_understanding = None

    if gemini_understanding is not None:
        intent = gemini_understanding.intent
        if not gemini_understanding.valid:
            if intent not in (INTENT_INVALID, INTENT_UNSUPPORTED):
                intent = INTENT_INVALID
    else:
        intent = classify_intent(question)

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
            return AssistantAnswer(
                ans.text, ans.status, ans.route, ans.week_of, ans.note_ids, usage=understanding_usage
            )
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
            return AssistantAnswer(
                ans.text, ans.status, ans.route, ans.week_of, ans.note_ids, usage=understanding_usage
            )
        return ans

    if intent == INTENT_LIST_UNEXPLAINED or (
        ("which" in lowered or "list" in lowered or "show" in lowered)
        and "unexplained" in lowered
    ):
        ans = _list_answers(results, supported=False)
        if understanding_usage:
            return AssistantAnswer(
                ans.text, ans.status, ans.route, ans.week_of, ans.note_ids, usage=understanding_usage
            )
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
        word in lowered for word in ("why", "spike", "pricier", "expensive", "increase", "cost")
    )
    resolution_rows = route_rows
    has_precise_time = bool(
        re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
        or re.search(r"\b\d+\s+months?\s+ago\b", lowered)
        or re.search(r"\b(today|yesterday|last week|this week)\b", lowered)
    )
    if explanation_question and not has_precise_time:
        anomaly_rows = route_rows[route_rows["flagged"].isin(["Yes", "No (justified)"])]
        if not anomaly_rows.empty:
            resolution_rows = anomaly_rows

    week = None
    if gemini_understanding and gemini_understanding.week_iso:
        cand_date = pd.to_datetime(gemini_understanding.week_iso, errors="coerce")
        if pd.notna(cand_date):
            cand_monday = cand_date - timedelta(days=int(cand_date.dayofweek))
            if cand_monday in set(route_rows["week_of"]):
                week = cand_monday

    if week is None:
        time_query = (
            gemini_understanding.week_hint
            if gemini_understanding and gemini_understanding.week_hint
            else question
        )
        week, week_error = _find_week(time_query, resolution_rows)
        if week is None and time_query != question:
            week, week_error = _find_week(question, resolution_rows)
        if week_error:
            return AssistantAnswer(week_error, "clarification", route, usage=understanding_usage)

    if week is None:
        return AssistantAnswer(
            "Which Monday week should I inspect? You can provide a date such as `2025-02-24`.",
            "clarification",
            route,
            usage=understanding_usage,
        )

    row = route_rows[route_rows["week_of"].eq(week)]
    if row.empty:
        return AssistantAnswer(
            f"I found no computed result for `{route}` in week `{week:%Y-%m-%d}`.",
            "not-found",
            route,
            week,
            usage=understanding_usage,
        )

    if "reject" in lowered or ("why was" in lowered and "note" in lowered):
        ans = _rejection_answer(question, route, week, notes)
        if understanding_usage:
            return AssistantAnswer(
                ans.text, ans.status, ans.route, ans.week_of, ans.note_ids, usage=understanding_usage
            )
        return ans

    ans = _format_row_answer(row.iloc[0], notes)
    if understanding_usage:
        return AssistantAnswer(
            ans.text, ans.status, ans.route, ans.week_of, ans.note_ids, usage=understanding_usage
        )
    return ans

