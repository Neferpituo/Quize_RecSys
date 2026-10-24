"""Weighted popularity baseline."""

import numpy as np
import pandas as pd


def weighted_popularity_scores(
    hist_items,
    candidate_set,
    reference_time,
    window_week,
    half_life_days,
):
    """Return candidate show scores using time-decayed unique-user popularity."""
    frame = hist_items.loc[:, ["user_id", "tv_show_id", "last_view_time"]].copy()
    frame = frame.loc[frame["tv_show_id"].isin(candidate_set)]
    if frame.empty:
        return {}
    reference_time = pd.Timestamp(reference_time)
    age_days = (
        reference_time - pd.to_datetime(frame["last_view_time"])
    ).dt.total_seconds() / 86400.0
    frame = frame.loc[(age_days >= 0) & (age_days <= window_week * 7)]
    if frame.empty:
        return {}
    age_days = age_days.loc[frame.index]
    if half_life_days is None or not np.isfinite(half_life_days):
        weights = np.ones(len(frame), dtype=np.float64)
    else:
        weights = np.power(0.5, age_days.to_numpy() / float(half_life_days))
    frame = frame.assign(weight=weights)
    scores = frame.groupby("tv_show_id", sort=False)["weight"].sum()
    scores = scores.sort_values(ascending=False)
    return {int(show_id): float(score) for show_id, score in scores.items()}


def weighted_popularity(
    hist_items,
    candidate_set,
    reference_time,
    window_week,
    half_life_days,
    k=5,
):
    """Return the top-k time-decayed popular shows inside the candidate pool."""
    scores = weighted_popularity_scores(
        hist_items,
        candidate_set,
        reference_time,
        window_week,
        half_life_days,
    )
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top = [show_id for show_id, _ in ranked[:k]]
    if len(top) < k:
        seen = set(top)
        top.extend(sorted(int(value) for value in candidate_set if int(value) not in seen)[: k - len(top)])
    return top
