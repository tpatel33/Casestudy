# Freight Cost Intelligence

This project implements the FreightTiger case study as a deterministic Python pipeline plus a Streamlit dashboard. It uses only the supplied shipment records and context notes; the core pipeline makes no LLM or paid API calls.

## Architecture

- `src/ingestion.py` validates the shipment schema, dates, numeric fields, duplicates, and rejected-row reasons.
- `src/analytics.py` assigns Monday weeks and calculates weighted route-week cost per tonne-km, an eight-observation causal history baseline, and a same-week same-route-type peer baseline.
- `src/detection.py` applies the configurable 15% rising-cost anomaly rule without consulting evidence.
- `src/retrieval.py` validates exact route/date applicability and rejects notes that contradict a cost increase.
- `src/assistant.py` resolves analyst questions against computed route-week rows and validated notes without changing verdicts.
- `src/gemini.py` optionally rewrites verified answers through Gemini; it never calculates metrics, selects evidence, or changes verdicts.
- `src/reporting.py` writes the exact required CSV and a separate audit CSV.
- `pipeline.py` runs all stages; `eval_checks.py` runs selected labeled cases; `repro_check.py` performs three-run reproducibility checks.
- `app.py` provides route trends, comparisons, anomaly review, evidence details, and the required CSV download.

The supplied inputs remain at the project root because duplicating or relocating them would add no value. The PDF was present, but this environment had no PDF text extractor; the attached written brief and CSV contents were used as the implementation requirements.

## Run locally

```text
python -m pip install -r requirements.txt -r requirements-dev.txt
python pipeline.py
python -m pytest tests -q
python eval_checks.py
python repro_check.py
python -m streamlit run app.py
```

Generated artifacts are written to `outputs/`:

- `flags.csv`: the graded eight-column output.
- `audit.csv`: internal rule, baseline, and evidence audit fields.
- `run_log.json`: input counts, configuration, result counts, template diagnostics, and zero LLM accounting.
- `eval_results.json`: selected evaluation outcomes.
- `reproducibility_report.json`: three-run SHA-256 comparison.

## Metric definitions and behavior

Route identity is directional and consists of `origin`, `destination`, and `route_type`; reverse routes are not merged. A week runs Monday through Sunday, and `week_of` is its Monday date.

For each route-week:

```text
cost_per_tonne_km = sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)
```

The historical baseline is the mean of up to the previous eight observed route-week values, excluding the current and all future weeks. The peer baseline is the mean of other routes with the same `route_type` in the same week. Missing or zero baselines remain unavailable; they are never treated as zero.

The initial anomaly threshold is 15% for either available baseline. This is a case-study design assumption, not an industry standard. A flagged increase is not a claim of fraud or deliberate overcharging.

Evidence must match the route or be explicitly nationwide, overlap the target week, and contain a defensible cost-impact statement. Exact route notes can explain route and peer-relative increases. `All Routes` notes can support a general historical cost rise when the observed rise is within the note's stated scope, but cannot explain why one route exceeded same-week peers because peers face the same nationwide factor. Explicit prose windows such as Feb 24–Mar 8 are honored. Notes saying costs were unaffected, traffic remained normal, or that a change did not affect this dataset are rejected. Unexplained anomalies have a blank `matched_note_id`.

The template header is valid and preserved. Its third sample row is malformed CSV because a comma in the unquoted reason creates a ninth field; this is recorded in `run_log.json`, and generated output uses correct CSV quoting.

## Conversational analyst assistant

The dashboard includes an **Ask the analyst assistant** section. It supports questions about a specific route and week, evidence-supported anomalies, unexplained anomalies, matched notes, and why a note was rejected. Route names and week dates are resolved against the computed results; explanation questions prioritize anomalous route-weeks, while ambiguous requests receive a clarification instead of a guess.

Answers are first resolved deterministically using only `flags.csv`, `audit.csv`, and `context_notes.csv`. Evidence-supported answers cite the exact note ID and effective date window. Unexplained increases explicitly say that the supplied evidence cannot confirm a cause.

The dashboard has an optional Gemini wording toggle. When enabled, only a small structured context is sent to Gemini: the user question, system rules, selected route-week metrics, deterministic verdict, and validated note details. Gemini is instructed not to recalculate or use outside knowledge. Its response must preserve the verified route, week, verdict, and note IDs; otherwise the app falls back to the deterministic answer. No API key is required for the core dashboard.

Gemini configuration can be supplied through environment variables or Streamlit Community Cloud secrets:

```text
GEMINI_API_KEY=your-key-in-a-secret-store
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT_SECONDS=20
```

The app records chat-call provider, model, actual input/output token counts when Gemini returns them, and estimated cost only when available. Missing usage or cost values remain unavailable; they are never recorded as fabricated zeroes. The key must not be placed in `.env.example`, source files, or Git history.

## Actual supplied-data results

The supplied data contains 2,940 shipment rows from 2024-01-01 through 2025-12-28, seven directional routes, and ten context notes. All 2,940 shipment rows passed validation. The current pipeline produces 728 route-week rows, 27 anomalies, three justified anomalies, and 24 unexplained anomalies. The justified cases are Ahmedabad-Mumbai with `N002`, plus two Chennai-Bangalore weeks covered by the Feb 24–Mar 8 window in `N001`.

The selected evaluation cases pass: Delhi-Jaipur on 2024-11-11 is unexplained, Ahmedabad-Mumbai on 2025-01-20 matches `N002`, and Mumbai-Pune on 2025-09-15 remains unexplained because `N006` is outside the target week and states stable demand.

Three complete pipeline runs matched. The reproducibility report stores SHA-256 hashes for the graded and audit CSVs. The run log records zero LLM calls, zero input/output tokens, and zero LLM API cost.

## Deployment

For Streamlit Community Cloud, create or select a GitHub repository, choose `app.py` as the main file, and select Python 3.12 in Advanced settings. The supplied CSV files must remain alongside `app.py`, `pipeline.py`, and `src/`. Community Cloud runs from the repository root, so the project-relative paths in `src/config.py` work on Linux. Runtime dependencies are pinned in `requirements.txt`; test-only dependencies are in `requirements-dev.txt`. No secrets or external services are required for deterministic mode. To enable Gemini, add `GEMINI_API_KEY`, `GEMINI_MODEL`, and optionally `GEMINI_TIMEOUT_SECONDS` under the app's Community Cloud Secrets settings; never commit them.

Commit the application source, `src/`, `tests/`, `requirements.txt`, `requirements-dev.txt`, `README.md`, the three input CSV files, and optionally the supplied case-study PDF. Do not commit `.venv/`, caches, logs, generated `outputs/` artifacts, local Streamlit secrets, `.env` files, credentials, or private keys. After deployment, verify the dashboard loads, the metrics match the local run, the evidence table shows only validated note IDs, and the `flags.csv` download has the exact eight-column schema.

## Limitations and future improvements

The explanation ability is bounded by the evidence base, by design. The current validator uses exact route matching, Monday-week overlap, and deterministic note-ID categories (`N001` flood/weather, `N002` festival surcharge, `N003` fuel). Adding a valid route/date-specific note or connecting a verified fuel, toll, weather, or festival feed can expand explanations without changing the anomaly calculation. The pipeline will never compensate for missing evidence with an invented cause. Future work could add a reviewed note ontology, explicit event end dates, richer manual labels, configurable threshold sensitivity analysis, and a database-backed history for larger production volumes.
