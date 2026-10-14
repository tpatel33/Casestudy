"""Deterministic anomaly rules for weekly route metrics."""

from __future__ import annotations

import pandas as pd

from .config import ANOMALY_THRESHOLD_PERCENT


def detect_anomalies(
    weekly_metrics: pd.DataFrame,
    *,
    threshold_percent: float = ANOMALY_THRESHOLD_PERCENT,
) -> pd.DataFrame:
    """Flag rising costs meeting the threshold against an available baseline.

    The threshold is a configurable design assumption for this case study. A
    missing baseline is never treated as zero, and evidence is intentionally not
    consulted here.
    """

    if threshold_percent < 0:
        raise ValueError("threshold_percent must be non-negative")
    required = {
        "cost_per_tonne_km",
        "history_baseline",
        "peer_baseline",
        "vs_own_history",
        "vs_similar_routes",
    }
    missing = sorted(required.difference(weekly_metrics.columns))
    if missing:
        raise ValueError("Weekly metrics are missing detection columns: " + ", ".join(missing))

    result = weekly_metrics.copy()
    history_trigger = result["vs_own_history"].ge(threshold_percent) & result[
        "history_baseline"
    ].notna()
    peer_trigger = result["vs_similar_routes"].ge(threshold_percent) & result[
        "peer_baseline"
    ].notna()
    result["history_trigger"] = history_trigger
    result["peer_trigger"] = peer_trigger
    result["anomaly"] = history_trigger | peer_trigger
    result["trigger_baselines"] = ""
    result.loc[history_trigger & ~peer_trigger, "trigger_baselines"] = "own_history"
    result.loc[~history_trigger & peer_trigger, "trigger_baselines"] = "similar_routes"
    result.loc[history_trigger & peer_trigger, "trigger_baselines"] = "own_history,similar_routes"
    result["rule"] = result["anomaly"].map(
        lambda flagged: (
            f"current cost increase >= {threshold_percent:g}% against at least one available baseline"
            if flagged
            else f"no current cost increase >= {threshold_percent:g}% against an available baseline"
        )
    )
    return result

