"""Baseline recommenders and ranking metrics for the sweet.tv project."""

import numpy as np
import pandas as pd


def ap_at_k(recommended, relevant, k=5):
    """Return AP@k for one user without side effects."""
    relevant = set(relevant)
    if not relevant:
        return np.nan
    hits = 0
    score = 0.0
    for rank, item in enumerate(list(recommended)[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def map_at_k(recommendations, ground_truth, k=5):
    """Return MAP@k and the number of users with non-empty ground truth."""
    scores = []
    n_users_eval = 0
    for user, relevant in ground_truth.items():
        if not relevant:
            continue
        n_users_eval += 1
        scores.append(ap_at_k(recommendations.get(user, []), relevant, k=k))
    mean_score = float(np.mean(scores)) if scores else float("nan")
    return mean_score, n_users_eval


def fill_with_fallback(recommendations, fallback, k=5):
    """Fill recommendations to exactly k unique ids using the fallback list."""
    result = []
    seen = set()
    for item in list(recommendations) + list(fallback):
        item = int(item)
        if item in seen:
            continue
        result.append(item)
        seen.add(item)
        if len(result) == k:
            break
    return result


def build_fallback(history_items, candidates, k=5):
    """Build a global popularity fallback restricted to candidate show ids."""
    history_pool = history_items[history_items["tv_show_id"].isin(candidates)]
    popularity = history_pool.groupby("tv_show_id")["user_id"].nunique().sort_values(
        ascending=False
    )
    fallback = [int(value) for value in popularity.head(k).index]
    if len(fallback) < k:
        seen = set(fallback)
        extras = [int(value) for value in sorted(candidates) if int(value) not in seen]
        fallback.extend(extras[: k - len(fallback)])
    return fallback


def recommend_global_pop(hist_items, candidates, k=5):
    """Return the global popularity top-k for every user present in history."""
    top_ids = build_fallback(hist_items, candidates, k=k)
    users = np.unique(hist_items["user_id"].to_numpy())
    return {int(user): list(top_ids) for user in users}


def recommend_recent_pop(recent_items, candidates, k=5):
    """Return the recent popularity top-k for every user present in history."""
    top_ids = build_fallback(recent_items, candidates, k=k)
    users = np.unique(recent_items["user_id"].to_numpy())
    return {int(user): list(top_ids) for user in users}


def recommend_personal(hist_items, candidates, fallback, users, k=5):
    """Return personal total-watched top-k recommendations filled by fallback."""
    users = [int(user) for user in users]
    history_pool = hist_items[hist_items["tv_show_id"].isin(candidates)]
    history_sorted = history_pool.sort_values(
        ["user_id", "total_watched_s", "tv_show_id"],
        ascending=[True, False, True],
    )
    raw = {
        int(user): [int(show) for show in group["tv_show_id"].head(k)]
        for user, group in history_sorted.groupby("user_id", sort=False)
    }
    result = {}
    for user in users:
        recommendations = fill_with_fallback(raw.get(user, []), fallback, k=k)
        if len(recommendations) < k:
            recommendations = fill_with_fallback(
                recommendations, sorted(int(value) for value in candidates), k=k
            )
        result[user] = recommendations
    return result
