"""Shared data and evaluation pipeline for the sweet.tv recommender."""

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd()
for candidate in [Path.cwd(), *Path.cwd().parents]:
    if (candidate / "src" / "data.py").exists():
        PROJECT_ROOT = candidate
        break
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_aggregated_items  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data_par" / "raw"
DERIVED_DIR = PROJECT_ROOT / "data_par" / "derived"
SOLUTION_DIR = PROJECT_ROOT / "outputs" / "solution"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
SUBMISSION_PATH = PROJECT_ROOT / "data_par" / "submission.csv"

PUBLIC_SCORES = {
    "recent_pop": 0.29553,
    "global_pop": 0.22583,
    "personal": 0.21959,
}
PRIVATE_SCORES = {
    "recent_pop": 0.29554,
    "global_pop": 0.23147,
    "personal": 0.21044,
}
TARGET_WEEKS = tuple(range(21, 31))
HISTORY_WEEKS = tuple(range(11, 21))


def load_views():
    """Load the cached merged view log."""
    return pd.read_parquet(DERIVED_DIR / "views_merged.parquet")


def load_pair():
    """Load the cached interval pair table."""
    return pd.read_parquet(DERIVED_DIR / "pair_11_30.parquet")


def load_schedule(path, columns=None):
    """Load and normalize one schedule file."""
    frame = pd.read_parquet(path, columns=columns)
    frame["channel_id"] = frame["channel_id"].astype("int64")
    frame["tv_show_id"] = pd.to_numeric(
        frame["tv_show_id"], errors="coerce"
    ).fillna(0).astype("int64")
    frame["start_time"] = pd.to_datetime(
        frame["start_time"], format="%d.%m.%Y %H:%M:%S", errors="coerce"
    ).astype("datetime64[ns]")
    frame["duration"] = pd.to_numeric(
        frame["duration"], errors="coerce"
    ).fillna(0).astype("int64")
    return frame


def load_validation_context():
    """Return views, pair, and the validation candidate pool."""
    views = load_views()
    pair = load_pair()
    schedule_2 = load_schedule(
        RAW_DIR / "export_arh_21-30-final.parquet",
        columns=["channel_id", "start_time", "duration", "tv_show_id"],
    )
    candidate_set = set(
        int(value)
        for value in schedule_2.loc[
            schedule_2["tv_show_id"] != 0, "tv_show_id"
        ].unique()
    )
    return views, pair, candidate_set


def load_test_candidate_set():
    """Return the candidate pool from the test schedule catalogue."""
    frame = pd.read_parquet(
        RAW_DIR / "export_arh_31-42-final.parquet", columns=["tv_show_id"]
    )
    values = pd.to_numeric(frame["tv_show_id"], errors="coerce").fillna(0).astype("int64")
    return set(int(value) for value in values.unique() if int(value) != 0)


def load_test_users():
    """Return test user ids in template order."""
    frame = pd.read_csv(SUBMISSION_PATH, dtype=str)
    return [int(value) for value in frame["user_id"]]


def build_window_items(
    pair_frame,
    views_frame,
    weeks,
    ratio_threshold,
    candidate_set=None,
    with_week=False,
):
    """Build aggregated user-show items for a set of ISO weeks."""
    weeks = tuple(int(week) for week in weeks)
    subset = pair_frame.loc[pair_frame["iso_week"].isin(weeks)]
    if with_week:
        parts = []
        for week in weeks:
            week_subset = subset.loc[subset["iso_week"] == week]
            if week_subset.empty:
                continue
            _, _, items = build_aggregated_items(
                week_subset, ratio_threshold=ratio_threshold
            )
            if items.empty:
                continue
            items = items.copy()
            items["iso_week"] = week
            last_time = week_subset.groupby(
                ["user_id", "tv_show_id"], sort=False
            )["view_start_time"].max().rename("last_view_time").reset_index()
            items = items.merge(last_time, on=["user_id", "tv_show_id"], how="left")
            parts.append(items)
        items = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    else:
        _, _, items = build_aggregated_items(
            subset, ratio_threshold=ratio_threshold
        )
        if not items.empty:
            last_time = subset.groupby(
                ["user_id", "tv_show_id"], sort=False
            )["view_start_time"].max().rename("last_view_time").reset_index()
            items = items.merge(last_time, on=["user_id", "tv_show_id"], how="left")
    if candidate_set is not None and len(items):
        items = items.loc[items["tv_show_id"].isin(candidate_set)].reset_index(drop=True)
    return items


def load_primary_mode():
    """Return the selected local target mode and its decision details."""
    path = EVAL_DIR / "eval_top5.json"
    with open(path, encoding="utf-8") as file:
        rows = json.load(file)
    by_key = {(row["mode"], row["method"]): float(row["map_at_5"]) for row in rows}
    recent_local = by_key[("T_top5", "recent_pop")]
    relative_gap = abs(recent_local - PUBLIC_SCORES["recent_pop"]) / PUBLIC_SCORES["recent_pop"]
    order = sorted(
        ["recent_pop", "global_pop", "personal"],
        key=lambda method: by_key[("T_top5", method)],
        reverse=True,
    )
    order_matches = order == ["recent_pop", "global_pop", "personal"]
    mode = "T_top5" if relative_gap <= 0.30 and order_matches else "T05"
    return {
        "mode": mode,
        "relative_gap": float(relative_gap),
        "order_matches": bool(order_matches),
        "order": order,
        "recent_local": float(recent_local),
        "recent_public": PUBLIC_SCORES["recent_pop"],
    }


def load_primary_ground_truth(pair_frame, views_frame, candidate_set):
    """Build ground truth for the selected local target mode."""
    from src.eval import build_ground_truth

    decision = load_primary_mode()
    ground_truth = build_ground_truth(
        pair_frame,
        views_frame,
        TARGET_WEEKS,
        candidate_set,
        decision["mode"],
    )
    return ground_truth, decision


def save_json(path, payload):
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_json(path):
    """Read a JSON document."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_pickle(path, payload, compress=False):
    """Write a pickle, optionally using gzip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compress else open
    with opener(path, "wb") as file:
        pickle.dump(payload, file)


def load_pickle(path, compress=False):
    """Read a pickle, optionally using gzip."""
    opener = gzip.open if compress else open
    with opener(path, "rb") as file:
        return pickle.load(file)


def model_scores_to_dict(show_ids, scores, candidate_set):
    """Convert parallel show ids and scores into a candidate-filtered dict."""
    result = {}
    for show_id, score in zip(show_ids, scores):
        show_id = int(show_id)
        if show_id in candidate_set:
            result[show_id] = float(score)
    return result


def evaluate_recommendations(recommendations, ground_truth):
    """Return MAP@5 and evaluated user count."""
    from src.eval import map_at_k

    return map_at_k(recommendations, ground_truth, k=5)
