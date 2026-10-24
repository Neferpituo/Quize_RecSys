"""Hybrid score blending for popularity and recommender models."""

import numpy as np


def _min_max(scores):
    if not scores:
        return {}
    values = np.asarray(list(scores.values()), dtype=np.float64)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return {key: 0.0 for key in scores}
    return {key: (float(value) - low) / (high - low) for key, value in scores.items()}


def hybrid_scores(popularity_scores, model_scores, alpha):
    """Blend popularity and model scores after min-max normalization."""
    popularity_norm = _min_max(popularity_scores)
    model_norm = _min_max(model_scores)
    items = set(popularity_norm) | set(model_norm)
    return {
        item: float(alpha) * popularity_norm.get(item, 0.0)
        + (1.0 - float(alpha)) * model_norm.get(item, 0.0)
        for item in items
    }


def hybrid_score(popularity_scores, model_scores, alpha, k=5):
    """Return the top-k blended recommendation list."""
    scores = hybrid_scores(popularity_scores, model_scores, alpha)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [item for item, _ in ranked[:k]]


def weighted_sum_scores(score_dicts, weights):
    """Blend several score dictionaries with fixed weights."""
    normalized = [_min_max(scores) for scores in score_dicts]
    items = set()
    for scores in normalized:
        items.update(scores)
    return {
        item: float(
            sum(weight * scores.get(item, 0.0) for weight, scores in zip(weights, normalized))
        )
        for item in items
    }


def weighted_sum_recommendations(score_dicts, weights, k=5):
    """Return the top-k items from a weighted score blend."""
    scores = weighted_sum_scores(score_dicts, weights)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [item for item, _ in ranked[:k]]
