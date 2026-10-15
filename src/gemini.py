"""Optional Gemini layer for the freight-cost analyst assistant.

Two responsibilities:
1. ``understand_question`` — uses Gemini to naturally parse intent and extract
   entities (route, week) from a free-form user question. This replaces the
   regex keyword classifier when a Gemini API key is configured.
2. ``generate_grounded_response`` — rewrites a verified deterministic answer
   into more readable prose. The deterministic engine remains authoritative;
   Gemini cannot change metrics, verdicts, note IDs, or routes.

Both functions are optional and fall back gracefully when the API key is
absent or a network error occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .assistant import AssistantAnswer


@dataclass(frozen=True)
class GeminiConfig:
    """Runtime configuration loaded from environment variables or secrets."""

    api_key: str | None
    model: str = "gemini-3.5-flash-lite"
    timeout_seconds: int = 20

    @classmethod
    def from_sources(cls, secrets: Mapping[str, Any] | None = None) -> "GeminiConfig":
        secrets = secrets or {}
        api_key = os.getenv("GEMINI_API_KEY") or secrets.get("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL") or secrets.get("GEMINI_MODEL") or cls.model
        timeout_value = os.getenv("GEMINI_TIMEOUT_SECONDS") or secrets.get(
            "GEMINI_TIMEOUT_SECONDS"
        )
        try:
            timeout_seconds = max(5, int(timeout_value or cls.timeout_seconds))
        except (TypeError, ValueError):
            timeout_seconds = cls.timeout_seconds
        return cls(
            api_key=str(api_key).strip() if api_key else None,
            model=str(model).strip(),
            timeout_seconds=timeout_seconds,
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key and "PASTE_YOUR_GEMINI_KEY_HERE" not in self.api_key)


@dataclass(frozen=True)
class GeminiUsage:
    """Usage returned by Gemini, without inventing missing token or cost data."""

    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None


@dataclass(frozen=True)
class GeminiResponse:
    text: str
    usage: GeminiUsage


class GeminiRequestError(RuntimeError):
    """Raised when an optional Gemini request cannot produce an answer."""

    def __init__(self, message: str, usage: GeminiUsage) -> None:
        super().__init__(message)
        self.usage = usage


def _usage_from_metadata(model: str, metadata: Mapping[str, Any] | None) -> GeminiUsage:
    metadata = metadata or {}
    input_tokens = metadata.get("promptTokenCount")
    output_tokens = metadata.get("candidatesTokenCount")
    total_tokens = metadata.get("totalTokenCount")
    return GeminiUsage(
        provider="gemini",
        model=model,
        input_tokens=int(input_tokens) if input_tokens is not None else None,
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        total_tokens=int(total_tokens) if total_tokens is not None else None,
        estimated_cost_usd=None,
    )


def _error_usage(config: GeminiConfig) -> GeminiUsage:
    return GeminiUsage(provider="gemini", model=config.model)


def _prompt(question: str, context: Mapping[str, Any]) -> str:
    serialized_context = json.dumps(context, ensure_ascii=False, indent=2, default=str)
    return f"""You are the communication layer for a freight-cost analyst assistant.

Answer the user's question using ONLY the VERIFIED CONTEXT below. The deterministic
analytics and evidence validator are authoritative. Do not recalculate, alter, or
override any metric, route, date, anomaly status, or note ID. Do not use outside
knowledge. If the context is insufficient, say that clearly. Do not infer that a
political, economic, weather, or other external event caused a change unless the
verified context explicitly supports it.

For evidence-supported claims, cite the exact note ID and its effective date window.
Distinguish observed metrics, evidence-supported explanations, possible hypotheses,
and unexplained causes. Keep the answer concise and ask one clarifying question if
the verified context says the request is ambiguous.

User question:
{question}

VERIFIED CONTEXT:
{serialized_context}
"""


def generate_grounded_response(
    question: str,
    context: Mapping[str, Any],
    config: GeminiConfig,
) -> GeminiResponse:
    """Generate wording from verified context through Gemini's REST API."""

    if not config.configured:
        raise GeminiRequestError("Gemini is not configured.", _error_usage(config))

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.model}:generateContent"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": _prompt(question, context)}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 600,
        },
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": str(config.api_key),
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except OSError:
            detail = str(error)
        raise GeminiRequestError(
            f"Gemini request failed with HTTP {error.code}: {detail[:300]}",
            _error_usage(config),
        ) from error
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise GeminiRequestError(
            f"Gemini request could not be completed: {error}", _error_usage(config)
        ) from error

    candidates = response_payload.get("candidates") or []
    if not candidates:
        raise GeminiRequestError(
            "Gemini returned no candidate answer.",
            _usage_from_metadata(config.model, response_payload.get("usageMetadata")),
        )
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    if not text:
        raise GeminiRequestError(
            "Gemini returned an empty answer.",
            _usage_from_metadata(config.model, response_payload.get("usageMetadata")),
        )
    return GeminiResponse(
        text=text,
        usage=_usage_from_metadata(config.model, response_payload.get("usageMetadata")),
    )


def validate_grounded_response(text: str, answer: AssistantAnswer, context: Mapping[str, Any]) -> bool:
    """Reject responses that lose required route, date, verdict, or provenance."""

    normalized = text.casefold()
    if not text.strip():
        return False
    if answer.route and answer.route.casefold() not in normalized:
        return False
    if answer.week_of is not None and answer.week_of.strftime("%Y-%m-%d") not in text:
        return False
    expected_note_ids = set(answer.note_ids)
    mentioned_note_ids = set(re.findall(r"\bN\d+\b", text.upper()))
    if not expected_note_ids.issubset(mentioned_note_ids):
        return False
    allowed_note_ids = set(context.get("allowed_note_ids", []))
    if mentioned_note_ids - allowed_note_ids:
        return False
    if answer.status == "unexplained" and not any(
        phrase in normalized
        for phrase in ("cannot confirm", "insufficient", "no validated", "unexplained", "no evidence")
    ):
        return False
    if answer.status == "overview" and not all(
        phrase in normalized for phrase in ("tonne-km", "route_type", "evidence")
    ):
        return False
    return True


# ---------------------------------------------------------------------------
# Natural-language question understanding via Gemini
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuestionUnderstanding:
    """Structured understanding of a user question, produced by Gemini.

    Attributes
    ----------
    intent:
        One of the INTENT_* constants from ``assistant.py``.
    route:
        The exact route name as it appears in the data, or None.
    week_iso:
        An ISO-8601 date string (YYYY-MM-DD) for the Monday of the target
        week if Gemini could resolve it, otherwise None.
    week_hint:
        The raw time phrase from the user (e.g. "late February") that the
        deterministic ``_find_week`` function can use as a fallback, or None.
    valid:
        False when Gemini returned something unparseable or signalled that the
        question is not about freight analytics at all.
    usage:
        Optional GeminiUsage metrics for tracking API consumption.
    """

    intent: str
    route: str | None = None
    week_iso: str | None = None
    week_hint: str | None = None
    time_hint: str | None = None
    date_start_iso: str | None = None
    date_end_iso: str | None = None
    valid: bool = True
    usage: GeminiUsage | None = None


def _understanding_prompt(
    question: str,
    known_routes: list[str],
    conversation_context: Sequence[Mapping[str, str]] | None = None,
) -> str:
    routes_str = ", ".join(f'"{r}"' for r in known_routes)
    recent_context = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in (conversation_context or [])[-6:]
        if message.get("content")
    )
    return f"""You are the natural language understanding layer for a freight-cost intelligence assistant.

Your task is to analyze the user's question and output a single JSON object:
{{
  "intent": "<one of the supported intents below>",
  "route": "<exact route name from the known list, or null>",
  "week_iso": "<Monday date YYYY-MM-DD if specified or inferrable, else null>",
  "week_hint": "<the raw time phrase from the question, e.g. 'late February', or null>",
  "time_hint": "<the complete raw date/range phrase, or null>",
  "date_start_iso": "<start date YYYY-MM-DD for a range, or null>",
  "date_end_iso": "<end date YYYY-MM-DD for a range, or null>",
  "valid": <true if this is a coherent freight question, false if gibberish, random characters, or complete nonsense>
}}

Supported intents:
- "invalid_input"         : Gibberish, keyboard mash (e.g. 'yhyjuyuyukuy', 'asdfghjkl'), random characters, or meaningless input.
- "unsupported_request"   : Off-topic questions (e.g. poems, songs, cricket, weather, recipes, politics, general chat).
- "system_overview"       : Questions asking what this system/assistant is, how metrics/baselines work, how cost is calculated.
- "list_anomalies"        : Asking to list or show anomalies with evidence / justified anomalies.
- "unexplained_anomalies" : Asking to list or show unexplained anomalies / anomalies without evidence.
- "evidence_lookup"       : Inquiries about specific evidence note IDs (e.g. N001, N002) or why a note was rejected.
- "route_investigation"   : Questions about cost, spikes, price increases, or anomalies on a route or week.

Known routes in system: {routes_str}

Rules:
- If the question is gibberish like 'yhyjuyuyukuy', set "intent": "invalid_input", "valid": false, "route": null.
- If off-topic (e.g. 'write a poem', 'who won the match'), set "intent": "unsupported_request", "valid": false, "route": null.
- If asking about a route, match city names/aliases to known routes (e.g., 'Bombay' -> Mumbai, 'Madras' -> Chennai).
- If the route is not in known routes but is a valid shipping question, keep intent="route_investigation" and set route to null.
- Preserve the user's time wording in "time_hint". Include a date range only when the user explicitly asks for a range.
- Do not guess a missing year when more than one year could match; leave dates null so the application can ask a clarification.
- Resolve follow-up questions from the recent conversation only when the referenced route or week is clear.
- Return ONLY the JSON object, no explanation.

Recent conversation:
{recent_context or "(none)"}

User question: "{question}"
"""


def understand_question(
    question: str,
    known_routes: list[str],
    config: "GeminiConfig",
    conversation_context: Sequence[Mapping[str, str]] | None = None,
) -> QuestionUnderstanding:
    """Use Gemini to naturally understand a user question and extract entities.

    Returns a ``QuestionUnderstanding`` with intent, route, and optional week
    information. Raises ``GeminiRequestError`` on API failure so the caller
    can fall back to the regex classifier.
    """
    if not config.configured:
        raise GeminiRequestError("Gemini is not configured.", _error_usage(config))

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.model}:generateContent"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": _understanding_prompt(question, known_routes, conversation_context)}],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 200,
            "responseMimeType": "application/json",
        },
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": str(config.api_key),
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace")
        except OSError:
            detail = str(error)
        raise GeminiRequestError(
            f"Gemini understanding request failed with HTTP {error.code}: {detail[:200]}",
            _error_usage(config),
        ) from error
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise GeminiRequestError(
            f"Gemini understanding request could not complete: {error}",
            _error_usage(config),
        ) from error

    candidates = response_payload.get("candidates") or []
    if not candidates:
        raise GeminiRequestError(
            "Gemini returned no understanding candidate.",
            _usage_from_metadata(config.model, response_payload.get("usageMetadata")),
        )
    parts = candidates[0].get("content", {}).get("parts", [])
    raw_text = "".join(str(part.get("text", "")) for part in parts).strip()

    # Parse the JSON response from Gemini
    try:
        # Gemini sometimes wraps JSON in ```json ... ```
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        data = json.loads(json_match.group() if json_match else raw_text)
    except (json.JSONDecodeError, AttributeError) as exc:
        raise GeminiRequestError(
            f"Gemini returned unparseable understanding JSON: {raw_text[:100]}",
            _usage_from_metadata(config.model, response_payload.get("usageMetadata")),
        ) from exc

    return QuestionUnderstanding(
        intent=str(data.get("intent", "route_investigation")),
        route=data.get("route") or None,
        week_iso=data.get("week_iso") or None,
        week_hint=data.get("week_hint") or None,
        time_hint=data.get("time_hint") or None,
        date_start_iso=data.get("date_start_iso") or None,
        date_end_iso=data.get("date_end_iso") or None,
        valid=bool(data.get("valid", True)),
        usage=_usage_from_metadata(config.model, response_payload.get("usageMetadata")),
    )
