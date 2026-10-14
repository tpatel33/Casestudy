import pandas as pd

from src.detection import detect_anomalies


def _metrics(history_change, peer_change, history=100, peer=100):
    return pd.DataFrame(
        [
            {
                "cost_per_tonne_km": 100,
                "history_baseline": history,
                "peer_baseline": peer,
                "vs_own_history": history_change,
                "vs_similar_routes": peer_change,
            }
        ]
    )


def test_increase_below_threshold_is_not_anomaly() -> None:
    result = detect_anomalies(_metrics(14.99, float("nan")))

    assert not result.loc[0, "anomaly"]
    assert result.loc[0, "trigger_baselines"] == ""


def test_increase_at_threshold_flags_against_available_baseline() -> None:
    result = detect_anomalies(_metrics(15, float("nan")))

    assert result.loc[0, "anomaly"]
    assert result.loc[0, "trigger_baselines"] == "own_history"


def test_missing_baselines_are_not_treated_as_zero() -> None:
    result = detect_anomalies(_metrics(float("nan"), float("nan"), history=float("nan"), peer=float("nan")))

    assert not result.loc[0, "anomaly"]

