from datetime import timedelta

import pandas as pd

from src.analytics import calculate_weekly_metrics, monday_week


def _shipments(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_monday_week_assignment() -> None:
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-07", "2024-01-08"]))

    result = monday_week(dates)

    assert result.dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-01",
        "2024-01-01",
        "2024-01-08",
    ]


def test_weekly_cost_is_weighted_by_total_tonne_km() -> None:
    data = _shipments(
        [
            {"origin": "A", "destination": "B", "route_type": "Short", "quantity_tonnes": 1, "distance_km": 10, "freight_cost_inr": 100, "shipment_date": "2024-01-01"},
            {"origin": "A", "destination": "B", "route_type": "Short", "quantity_tonnes": 9, "distance_km": 10, "freight_cost_inr": 900, "shipment_date": "2024-01-02"},
        ]
    )

    result = calculate_weekly_metrics(data)

    assert result.loc[0, "cost_per_tonne_km"] == 10
    assert result.loc[0, "tonne_km"] == 100


def test_history_excludes_current_week_and_is_limited_to_eight_observations() -> None:
    rows = []
    for week_number in range(10):
        rows.append(
            {
                "origin": "A",
                "destination": "B",
                "route_type": "Short",
                "quantity_tonnes": 1,
                "distance_km": 1,
                "freight_cost_inr": week_number + 1,
                "shipment_date": (pd.Timestamp("2024-01-01") + timedelta(days=7 * week_number)).strftime("%Y-%m-%d"),
            }
        )

    result = calculate_weekly_metrics(_shipments(rows))

    assert pd.isna(result.loc[0, "history_baseline"])
    assert result.loc[1, "history_baseline"] == 1
    assert result.loc[9, "history_observations"] == 8
    assert result.loc[9, "history_baseline"] == 5.5


def test_peer_baseline_excludes_current_route_and_missing_peer_is_null() -> None:
    data = _shipments(
        [
            {"origin": "A", "destination": "B", "route_type": "Short", "quantity_tonnes": 1, "distance_km": 1, "freight_cost_inr": 20, "shipment_date": "2024-01-01"},
            {"origin": "C", "destination": "D", "route_type": "Short", "quantity_tonnes": 1, "distance_km": 1, "freight_cost_inr": 10, "shipment_date": "2024-01-01"},
            {"origin": "E", "destination": "F", "route_type": "Long", "quantity_tonnes": 1, "distance_km": 1, "freight_cost_inr": 30, "shipment_date": "2024-01-01"},
        ]
    )

    result = calculate_weekly_metrics(data)

    assert result.loc[result["route"] == "A-B", "peer_baseline"].item() == 10
    assert pd.isna(result.loc[result["route"] == "E-F", "peer_baseline"].item())
