"""Implicit ALS model helpers."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


def build_user_show_matrix(hist_items, reference_time, half_life_days):
    """Build a confidence CSR matrix with log-watched and time decay."""
    frame = hist_items.loc[
        :, ["user_id", "tv_show_id", "total_watched_s", "last_view_time"]
    ].copy()
    frame = frame.loc[frame["total_watched_s"] > 0]
    reference_time = pd.Timestamp(reference_time)
    age_days = (
        reference_time - pd.to_datetime(frame["last_view_time"])
    ).dt.total_seconds() / 86400.0
    decay = np.power(0.5, age_days.to_numpy() / float(half_life_days))
    confidence = np.log1p(frame["total_watched_s"].to_numpy(dtype=np.float64)) * decay
    user_index = np.sort(frame["user_id"].unique())
    item_index = np.sort(frame["tv_show_id"].unique())
    user_lookup = pd.Index(user_index).get_indexer(frame["user_id"].to_numpy())
    item_lookup = pd.Index(item_index).get_indexer(frame["tv_show_id"].to_numpy())
    matrix = csr_matrix(
        (confidence.astype(np.float32), (user_lookup, item_lookup)),
        shape=(len(user_index), len(item_index)),
        dtype=np.float32,
    )
    return matrix, user_index, item_index


def train_als(matrix, factors, regularization, iterations, alpha, seed=42):
    """Fit an implicit ALS model and return it."""
    model = AlternatingLeastSquares(
        factors=int(factors),
        regularization=float(regularization),
        iterations=int(iterations),
        alpha=float(alpha),
        random_state=int(seed),
        use_gpu=False,
        num_threads=1,
    )
    model.fit(matrix, show_progress=False)
    return model


def als_scores(model, user_idx, items_index, candidate_set, n_candidates=200):
    """Return candidate-filtered model scores for one user row."""
    if user_idx is None or user_idx < 0:
        return {}
    empty = csr_matrix((1, model.item_factors.shape[0]), dtype=np.float32)
    ids, values = model.recommend(
        int(user_idx),
        empty,
        N=int(n_candidates),
        filter_already_liked_items=False,
    )
    result = {}
    for item_idx, score in zip(ids, values):
        show_id = int(items_index[int(item_idx)])
        if show_id in candidate_set:
            result[show_id] = float(score)
    return result


def recommend_als(model, user_idx, items_index, candidate_set, k=5, n_candidates=200):
    """Return the top-k ALS recommendations inside the candidate pool."""
    scores = als_scores(
        model, user_idx, items_index, candidate_set, n_candidates=n_candidates
    )
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [show_id for show_id, _ in ranked[:k]]
