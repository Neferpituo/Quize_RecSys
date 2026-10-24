"""Evaluation helpers for the sweet.tv recommender project."""

import numpy as np
import pandas as pd

from src.data import items_for_weeks_with_ratio

MODE_RATIOS = {"T03": 0.3, "T05": 0.5, "T08": 0.8}


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


def build_ground_truth(pair_frame, views_frame, weeks, candidate_set, mode):
    """Build a user -> relevant show set for a target definition mode."""
    if mode not in set(MODE_RATIOS) | {"T_top5"}:
        raise ValueError(f"Unknown target mode: {mode}")
    ratio_threshold = 0.0 if mode == "T_top5" else MODE_RATIOS[mode]
    items = items_for_weeks_with_ratio(
        pair_frame, views_frame, weeks, ratio_threshold
    )
    items = items.loc[items["tv_show_id"].isin(candidate_set)].copy()
    if mode == "T_top5":
        items = items.sort_values(
            ["user_id", "total_watched_s", "tv_show_id"],
            ascending=[True, False, True],
        )
        items = items.groupby("user_id", sort=False).head(5)
    return {
        int(user): set(int(show) for show in group["tv_show_id"])
        for user, group in items.groupby("user_id", sort=False)
    }


def build_unified_fallback(history_items, candidates, k=5):
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


def _self_test():
    """Run a small MAP@5 sanity check."""
    toy_score = ap_at_k([101, 102, 103, 104, 105], {101, 103, 106}, k=5)
    toy_expected = (1 / 1 + 2 / 3) / 3
    assert np.isclose(toy_score, toy_expected)
    mean_score, n_users = map_at_k(
        {1: [101, 102, 103, 104, 105], 2: [201, 202, 203, 204, 205]},
        {1: {101}, 2: set()},
        k=5,
    )
    assert n_users == 1
    assert np.isclose(mean_score, 1.0)
    print("eval.py self-tests passed")


if __name__ == "__main__":
    _self_test()
