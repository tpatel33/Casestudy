"""Deterministic weekly freight cost analytics."""

from __future__ import annotations

import pandas as pd

from .config import HISTORY_WINDOW_WEEKS, ROUTE_KEYS


def monday_week(date_series: pd.Series) -> pd.Series:
    """Return the Monday starting each Monday-through-Sunday week."""

    dates = pd.to_datetime(date_series, errors="raise")
    return dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")


def _route_label(frame: pd.DataFrame) -> pd.Series:
    return frame["origin"].astype(str) + "-" + frame["destination"].astype(str)


def calculate_weekly_metrics(
    shipments: pd.DataFrame, *, history_window: int = HISTORY_WINDOW_WEEKS
) -> pd.DataFrame:
    """Calculate weighted route-week costs and causal comparison baselines.

    The cost metric is total freight cost divided by total tonne-kilometres.
    Historical values are prior observed weeks only; peer values are the
    unweighted mean of other route-week values of the same route type.
    """

    required = {
        "origin",
        "destination",
        "route_type",
        "quantity_tonnes",
        "distance_km",
        "freight_cost_inr",
        "shipment_date",
    }
    missing = sorted(required.difference(shipments.columns))
    if missing:
        raise ValueError("Shipments are missing analytics columns: " + ", ".join(missing))
    if history_window < 1:
        raise ValueError("history_window must be at least one")

    frame = shipments.copy()
    frame["shipment_date"] = pd.to_datetime(frame["shipment_date"], errors="raise")
    frame["week_of"] = monday_week(frame["shipment_date"])
    frame["tonne_km"] = frame["quantity_tonnes"] * frame["distance_km"]
    if frame["tonne_km"].le(0).any():
        raise ValueError("Analytics requires positive quantity-distance denominators")

    grouping = [*ROUTE_KEYS, "week_of"]
    weekly = (
        frame.groupby(grouping, as_index=False, sort=True)
        .agg(
            freight_cost_inr=("freight_cost_inr", "sum"),
            tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_date", "size"),
        )
        .sort_values(grouping, kind="stable")
        .reset_index(drop=True)
    )
    weekly["cost_per_tonne_km"] = weekly["freight_cost_inr"] / weekly["tonne_km"]
    weekly["route"] = _route_label(weekly)

    weekly["history_baseline"] = float("nan")
    weekly["history_observations"] = 0
    for route_identity, route_rows in weekly.groupby(list(ROUTE_KEYS), sort=False):
        route_indices = route_rows.index.tolist()
        for position, row_index in enumerate(route_indices):
            prior_indices = route_indices[max(0, position - history_window) : position]
            if prior_indices:
                values = weekly.loc[prior_indices, "cost_per_tonne_km"]
                weekly.loc[row_index, "history_baseline"] = values.mean()
                weekly.loc[row_index, "history_observations"] = len(values)

    peer_grouping = ["route_type", "week_of"]
    peer_summary = weekly.groupby(peer_grouping)["cost_per_tonne_km"].agg(
        peer_sum="sum", peer_count="count"
    )
    peer_keys = pd.MultiIndex.from_frame(weekly[peer_grouping])
    peer_positions = peer_summary.index.get_indexer(peer_keys)
    peer_sums = peer_summary["peer_sum"].to_numpy()[peer_positions]
    peer_counts = peer_summary["peer_count"].to_numpy()[peer_positions]
    weekly["peer_baseline"] = float("nan")
    has_peer = peer_counts > 1
    weekly.loc[has_peer, "peer_baseline"] = (
        peer_sums[has_peer] - weekly.loc[has_peer, "cost_per_tonne_km"].to_numpy()
    ) / (peer_counts[has_peer] - 1)

    for baseline, output in (
        ("history_baseline", "vs_own_history"),
        ("peer_baseline", "vs_similar_routes"),
    ):
        weekly[output] = float("nan")
        valid = weekly[baseline].notna() & weekly[baseline].ne(0)
        weekly.loc[valid, output] = (
            (weekly.loc[valid, "cost_per_tonne_km"] / weekly.loc[valid, baseline]) - 1
        ) * 100

    columns = [
        *ROUTE_KEYS,
        "route",
        "week_of",
        "freight_cost_inr",
        "tonne_km",
        "shipment_count",
        "cost_per_tonne_km",
        "history_baseline",
        "history_observations",
        "vs_own_history",
        "peer_baseline",
        "vs_similar_routes",
    ]
    return weekly.loc[:, columns]
