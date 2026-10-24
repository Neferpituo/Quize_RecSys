"""Item-item co-occurrence graph with time decay."""

from collections import defaultdict

import numpy as np
import pandas as pd


def build_item_graph(hist_items, half_life_days, min_user_weeks):
    """Build a time-decayed item-item co-occurrence graph from user-week sets."""
    frame = hist_items.loc[
        :, ["user_id", "tv_show_id", "iso_week", "total_watched_s"]
    ].drop_duplicates(["user_id", "iso_week", "tv_show_id"])
    if frame.empty:
        return pd.DataFrame(
            columns=["item_a", "item_b", "count_user_weeks", "decay_weight"]
        )
    reference_week = int(frame["iso_week"].max())
    age_weeks = reference_week - frame["iso_week"].astype("int64")
    frame = frame.assign(
        decay_weight=np.power(
            0.5, age_weeks.to_numpy(dtype=np.float64) / (float(half_life_days) / 7.0)
        )
    )
    pairs = frame.loc[:, ["user_id", "iso_week", "tv_show_id", "decay_weight"]].merge(
        frame.loc[:, ["user_id", "iso_week", "tv_show_id", "decay_weight"]],
        on=["user_id", "iso_week"],
        suffixes=("_a", "_b"),
    )
    pairs = pairs.loc[pairs["tv_show_id_a"] < pairs["tv_show_id_b"]]
    if pairs.empty:
        return pd.DataFrame(
            columns=["item_a", "item_b", "count_user_weeks", "decay_weight"]
        )
    graph = pairs.groupby(["tv_show_id_a", "tv_show_id_b"], sort=False).agg(
        count_user_weeks=("user_id", "size"),
        decay_weight=("decay_weight_a", "sum"),
    ).reset_index()
    graph.columns = ["item_a", "item_b", "count_user_weeks", "decay_weight"]
    graph = graph.loc[graph["count_user_weeks"] >= int(min_user_weeks)]
    graph = graph.sort_values(
        ["decay_weight", "count_user_weeks", "item_a", "item_b"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return graph


def build_graph_adjacency(item_graph):
    """Convert a graph DataFrame into an undirected adjacency dictionary."""
    adjacency = defaultdict(dict)
    for row in item_graph.itertuples(index=False):
        item_a = int(row.item_a)
        item_b = int(row.item_b)
        weight = float(row.decay_weight)
        adjacency[item_a][item_b] = adjacency[item_a].get(item_b, 0.0) + weight
        adjacency[item_b][item_a] = adjacency[item_b].get(item_a, 0.0) + weight
    return dict(adjacency)


def graph_item_scores(item_graph, user_history_show_ids, candidate_set, n_seeds=20):
    """Return candidate-filtered one-hop graph scores for a user."""
    if not user_history_show_ids:
        return {}
    adjacency = item_graph if isinstance(item_graph, dict) else build_graph_adjacency(item_graph)
    seeds = [int(value) for value in list(user_history_show_ids)[:n_seeds]]
    seed_set = set(seeds)
    scores = defaultdict(float)
    for seed in seeds:
        for neighbor, weight in adjacency.get(seed, {}).items():
            if neighbor in seed_set or neighbor not in candidate_set:
                continue
            scores[neighbor] += float(weight)
    return dict(scores)


def recommend_graph(item_graph, user_history_show_ids, candidate_set, k=5, n_seeds=20):
    """Return top-k one-hop graph recommendations for a user."""
    scores = graph_item_scores(
        item_graph, user_history_show_ids, candidate_set, n_seeds=n_seeds
    )
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [show_id for show_id, _ in ranked[:k]]
