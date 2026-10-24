"""Build three calibration baseline submissions for the sweet.tv competition."""

import gc
import json
import sys
from datetime import datetime, timezone
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

from src.baselines import (
    build_fallback,
    recommend_global_pop,
    recommend_personal,
    recommend_recent_pop,
)
from src.data import (
    RATIO_THRESHOLD,
    build_overlap_pairs,
    items_for_weeks_with_ratio,
    merge_overlapping_views,
)

SEED = 42
BASELINE_HIST_THRESHOLD = 0.5
RAW_DIR = PROJECT_ROOT / "data_par" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "submissions"
LOG_DIR = PROJECT_ROOT / "logs"
SUBMISSION_TEMPLATE = PROJECT_ROOT / "data_par" / "submission.csv"
EVAL_MATRIX_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "eval_matrix.json"


def load_views(path):
    """Load and normalize the view log."""
    frame = pd.read_parquet(path).rename(columns={"duraton": "duration"})
    frame["user_id"] = frame["user_id"].astype("uint64")
    frame["vsetv_id"] = frame["vsetv_id"].astype("int64")
    frame["start_time"] = pd.to_datetime(
        frame["start_time"], errors="coerce"
    ).astype("datetime64[ns]")
    frame["stop_time"] = pd.to_datetime(
        frame["stop_time"], errors="coerce"
    ).astype("datetime64[ns]")
    frame["duration"] = pd.to_numeric(
        frame["duration"], errors="coerce"
    ).fillna(0).astype("int64")
    return frame


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


def most_frequent_label(label_frame, show_id, column):
    """Return the most frequent label for one tv_show_id."""
    examples = label_frame.loc[label_frame["tv_show_id"] == show_id, column]
    if len(examples) == 0:
        return ""
    return str(examples.value_counts().index[0])


def write_submission(path, users, recommendations, fallback):
    """Write a submission with one integer user_id and five candidate ids."""
    with open(path, "w", encoding="utf-8") as file:
        file.write("user_id,tv_show_id\n")
        for user in users:
            values = recommendations.get(int(user), list(fallback))[:5]
            if len(values) != 5 or len(set(values)) != 5:
                raise ValueError(f"Invalid recommendation for user {user}: {values}")
            file.write(f"{int(user)},{' '.join(str(int(value)) for value in values)}\n")


started = datetime.now(timezone.utc)
views = load_views(RAW_DIR / "dataset11-30week1.parquet")
views_merged = merge_overlapping_views(views)
views_merged["iso_week"] = views_merged["start_time"].dt.isocalendar().week.astype("int16")
del views
gc.collect()

train_columns = ["channel_id", "start_time", "duration", "tv_show_id"]
train_1 = load_schedule(RAW_DIR / "export_arh_11-20-final.parquet", train_columns)
train_2 = load_schedule(RAW_DIR / "export_arh_21-30-final.parquet", train_columns)
schedule_11_30 = pd.concat([train_1, train_2], ignore_index=True).drop_duplicates(
    subset=["channel_id", "start_time"], keep="first"
).reset_index(drop=True)
del train_1, train_2
gc.collect()

arrays = build_overlap_pairs(views_merged, schedule_11_30)
pair = pd.DataFrame({
    "user_id": views_merged["user_id"].to_numpy()[arrays["view_idx"]],
    "tv_show_id": schedule_11_30["tv_show_id"].to_numpy()[arrays["sched_idx"]],
    "overlap_s": arrays["overlap_s"],
    "view_idx": arrays["view_idx"],
    "sched_idx": arrays["sched_idx"],
    "view_duration": views_merged["duration"].to_numpy()[arrays["view_idx"]],
    "show_duration": schedule_11_30["duration"].to_numpy()[arrays["sched_idx"]],
})
del arrays, schedule_11_30
gc.collect()

hist_all = items_for_weeks_with_ratio(
    pair, views_merged, range(11, 31), BASELINE_HIST_THRESHOLD
)
hist_recent = items_for_weeks_with_ratio(
    pair, views_merged, range(27, 31), BASELINE_HIST_THRESHOLD
)
del pair
gc.collect()

test_schedule = pd.read_parquet(
    RAW_DIR / "export_arh_31-42-final.parquet",
    columns=["tv_show_id", "tv_show_title", "tv_show_category"],
)
test_schedule["tv_show_id"] = pd.to_numeric(
    test_schedule["tv_show_id"], errors="coerce"
).fillna(0).astype("int64")
candidates_test = set(
    int(value) for value in test_schedule.loc[
        test_schedule["tv_show_id"] != 0, "tv_show_id"
    ].unique()
)

template = pd.read_csv(SUBMISSION_TEMPLATE, dtype=str)
users = [int(value) for value in template["user_id"]]
fallback_test = build_fallback(hist_all, candidates_test, k=5)
global_top5 = build_fallback(hist_all, candidates_test, k=5)
recent_top5 = build_fallback(hist_recent, candidates_test, k=5)

global_user_recs = recommend_global_pop(hist_all, candidates_test, k=5)
recent_user_recs = recommend_recent_pop(hist_recent, candidates_test, k=5)
global_recs = {
    user: global_user_recs.get(user, list(global_top5)) for user in users
}
recent_recs = {
    user: recent_user_recs.get(user, list(recent_top5)) for user in users
}
personal_recs = recommend_personal(
    hist_all, candidates_test, fallback_test, users, k=5
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
files = {
    "global_pop": OUT_DIR / "cal_global_pop.csv",
    "recent_pop": OUT_DIR / "cal_recent_pop.csv",
    "personal": OUT_DIR / "cal_personal.csv",
}
write_submission(files["global_pop"], users, global_recs, fallback_test)
write_submission(files["recent_pop"], users, recent_recs, recent_top5)
write_submission(files["personal"], users, personal_recs, fallback_test)

for method, path in files.items():
    frame = pd.read_csv(path, dtype=str)
    unique_ids = set()
    for value in frame["tv_show_id"]:
        unique_ids.update(int(token) for token in value.split())
    print(f"{method}: file={path.relative_to(PROJECT_ROOT)} | unique_ids={len(unique_ids)}")

personal_history = hist_all[hist_all["tv_show_id"].isin(candidates_test)]
personal_counts = personal_history.groupby("user_id").size().reindex(users, fill_value=0)
fallback_users = int((personal_counts < 5).sum())
print("Personal users needing fallback:", fallback_users, f"({fallback_users / len(users):.2%})")
print("Template users with zero history:", int((personal_counts == 0).sum()))
print("Candidate pool size:", len(candidates_test))
print("Global top-5:")
for show_id in global_top5:
    print(" ", show_id, most_frequent_label(test_schedule, show_id, "tv_show_title"),
          "|", most_frequent_label(test_schedule, show_id, "tv_show_category"))
print("Recent top-5:")
for show_id in recent_top5:
    print(" ", show_id, most_frequent_label(test_schedule, show_id, "tv_show_title"),
          "|", most_frequent_label(test_schedule, show_id, "tv_show_category"))

with open(EVAL_MATRIX_PATH, encoding="utf-8") as file:
    matrix_rows = json.load(file)
matrix = pd.DataFrame(matrix_rows)
local = matrix[(matrix["hist_threshold"] == BASELINE_HIST_THRESHOLD)]
for group_name in ["all", "test"]:
    pivot = local[local["group"] == group_name].pivot(
        index="method", columns="target_threshold", values="MAP@5"
    )
    print(f"Local MAP@5 pivot, H=0.5, group={group_name}:")
    print(pivot.to_string())

created_at = datetime.now(timezone.utc).isoformat()
log_rows = []
for method, path in files.items():
    method_rows = local[
        (local["method"] == method) & (local["group"] == "all")
    ].set_index("target_threshold")["MAP@5"]
    method_test = local[
        (local["method"] == method) & (local["group"] == "test")
    ].set_index("target_threshold")["MAP@5"]
    log_rows.append({
        "version": f"cal_{method}",
        "file": str(path.relative_to(PROJECT_ROOT)),
        "description": f"cal_{method} H=0.5",
        "hist_threshold": BASELINE_HIST_THRESHOLD,
        "local_MAP_all_T03": float(method_rows.loc[0.3]),
        "local_MAP_all_T05": float(method_rows.loc[0.5]),
        "local_MAP_all_T08": float(method_rows.loc[0.8]),
        "local_MAP_test_T03": float(method_test.loc[0.3]),
        "local_MAP_test_T05": float(method_test.loc[0.5]),
        "local_MAP_test_T08": float(method_test.loc[0.8]),
        "public_score": "",
        "created_at": created_at,
    })
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_frame = pd.DataFrame(log_rows)
log_frame.to_csv(LOG_DIR / "submission_log.csv", index=False)
print("Saved:", (LOG_DIR / "submission_log.csv").relative_to(PROJECT_ROOT))

for method in ["global_pop", "recent_pop", "personal"]:
    print(f"Description for Kaggle: cal_{method} H=0.5")
print("Interpretation rule:")
print("- If public(personal) > public(recent_pop), the target is the mean MAP over T03 and T05.")
print("- If public(personal) < public(recent_pop), the target is T08.")
print("- If the relative difference is below 10%, the target is the mean MAP over T03, T05, and T08.")
print("Elapsed seconds:", round((datetime.now(timezone.utc) - started).total_seconds(), 2))
