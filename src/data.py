"""Reusable data helpers for the sweet.tv recommender project."""

import numpy as np
import pandas as pd

RATIO_THRESHOLD = 0.8

try:
    import pytest
except ImportError:
    pytest = None

def _mark_skip_without_pytest(func):
    """Mark a pytest-style test as skipped when pytest is unavailable."""
    func.__test__ = False
    func.__unittest_skip__ = True
    func.__unittest_skip_why__ = "pytest is not installed"
    return func


_requires_pytest = (
    _mark_skip_without_pytest if pytest is None else (lambda func: func)
)


def to_epoch_seconds(series):
    """Convert a datetime-like Series to integer Unix seconds."""
    return series.astype("datetime64[ns]").astype("int64") // 10**9


def merge_overlapping_views(views_df):
    """Merge overlapping view intervals for each (user_id, vsetv_id) pair.

    Intervals are sorted by start time and merged when the next start is
    strictly earlier than the running maximum stop. Intervals that only touch
    at an endpoint are kept separate. The returned duration is the merged
    interval length in seconds.
    """
    columns = ["user_id", "vsetv_id", "start_time", "stop_time", "duration"]
    missing = [column for column in columns if column not in views_df.columns]
    if missing:
        raise ValueError(f"Missing required view columns: {missing}")

    if views_df.shape[0] == 0:
        empty = views_df.loc[:, columns].copy()
        empty["user_id"] = empty["user_id"].astype("uint64")
        empty["vsetv_id"] = empty["vsetv_id"].astype("int64")
        empty["start_time"] = empty["start_time"].astype("datetime64[ns]")
        empty["stop_time"] = empty["stop_time"].astype("datetime64[ns]")
        empty["duration"] = empty["duration"].astype("int64")
        return empty.reset_index(drop=True)

    frame = views_df.loc[:, ["user_id", "vsetv_id", "start_time", "stop_time"]].copy()
    frame["start_s"] = to_epoch_seconds(frame["start_time"])
    frame["stop_s"] = to_epoch_seconds(frame["stop_time"])
    frame = frame.sort_values(
        ["user_id", "vsetv_id", "start_s", "stop_s"], kind="mergesort"
    ).reset_index(drop=True)

    keys = ["user_id", "vsetv_id"]
    previous_max_stop = frame.groupby(keys, sort=False)["stop_s"].cummax().shift(1)
    group_change = (
        frame["user_id"].ne(frame["user_id"].shift())
        | frame["vsetv_id"].ne(frame["vsetv_id"].shift())
    )
    new_segment = group_change | frame["start_s"].ge(previous_max_stop)
    frame["segment_id"] = new_segment.cumsum()

    merged = frame.groupby("segment_id", sort=False).agg(
        user_id=("user_id", "first"),
        vsetv_id=("vsetv_id", "first"),
        start_s=("start_s", "min"),
        stop_s=("stop_s", "max"),
    ).reset_index(drop=True)

    merged["start_time"] = pd.to_datetime(merged["start_s"], unit="s").astype(
        "datetime64[ns]"
    )
    merged["stop_time"] = pd.to_datetime(merged["stop_s"], unit="s").astype(
        "datetime64[ns]"
    )
    merged["duration"] = (merged["stop_s"] - merged["start_s"]).astype("int64")
    return merged.loc[:, columns].reset_index(drop=True)


def build_overlap_pairs(views_frame, schedule_frame):
    """Build interval-overlap pairs between view rows and schedule programs.

    Returns a dictionary with positional view and schedule indices, overlap
    seconds, the number of candidate interval pairs, and a monotonicity flag
    for schedule end times within each channel.
    """
    view = pd.DataFrame({
        "view_idx": np.arange(len(views_frame), dtype=np.int64),
        "user_id": views_frame["user_id"].to_numpy(),
        "channel_id": views_frame["vsetv_id"].to_numpy().astype(np.int64),
        "vstart": to_epoch_seconds(views_frame["start_time"]),
        "view_duration": views_frame["duration"].to_numpy().astype(np.int64),
    })
    view["vend"] = view["vstart"] + view["view_duration"]

    program = pd.DataFrame({
        "sched_idx": np.arange(len(schedule_frame), dtype=np.int64),
        "channel_id": schedule_frame["channel_id"].to_numpy().astype(np.int64),
        "tv_show_id": schedule_frame["tv_show_id"].to_numpy().astype(np.int64),
        "pstart": to_epoch_seconds(schedule_frame["start_time"]),
        "show_duration": schedule_frame["duration"].to_numpy().astype(np.int64),
    })
    program["pend"] = program["pstart"] + program["show_duration"]

    view = view.sort_values(["channel_id", "vstart"], kind="mergesort").reset_index(drop=True)
    program = program.sort_values(["channel_id", "pstart"], kind="mergesort").reset_index(drop=True)

    end_monotonic = (
        program.groupby("channel_id", sort=False)["pend"].diff().fillna(0).ge(0).all()
    )

    view_channel = view["channel_id"].to_numpy()
    view_start = view["vstart"].to_numpy()
    view_end = view["vend"].to_numpy()
    program_channel = program["channel_id"].to_numpy()
    program_start = program["pstart"].to_numpy()
    program_end = program["pend"].to_numpy()
    program_show = program["tv_show_id"].to_numpy()
    view_index = view["view_idx"].to_numpy()
    program_index = program["sched_idx"].to_numpy()

    view_group_start = np.flatnonzero(np.r_[True, view_channel[1:] != view_channel[:-1]])
    view_group_end = np.r_[view_group_start[1:], len(view_channel)]
    program_group_start = np.flatnonzero(np.r_[True, program_channel[1:] != program_channel[:-1]])
    program_group_end = np.r_[program_group_start[1:], len(program_channel)]
    program_group_channel = program_channel[program_group_start]

    view_parts = []
    program_parts = []
    overlap_parts = []
    total_candidates = 0

    for view_begin, view_stop in zip(view_group_start, view_group_end):
        channel = view_channel[view_begin]
        program_position = np.searchsorted(program_group_channel, channel)
        if program_position >= len(program_group_channel):
            continue
        if program_group_channel[program_position] != channel:
            continue

        program_begin = program_group_start[program_position]
        program_stop = program_group_end[program_position]
        channel_program_start = program_start[program_begin:program_stop]
        channel_program_end = program_end[program_begin:program_stop]

        low = np.searchsorted(channel_program_end, view_start[view_begin:view_stop], side="right")
        high = np.searchsorted(channel_program_start, view_end[view_begin:view_stop], side="left")
        counts = np.maximum(high - low, 0)
        total_candidates += int(counts.sum())
        if counts.sum() == 0:
            continue

        repeated_view = np.repeat(np.arange(view_stop - view_begin, dtype=np.int64), counts)
        pair_offsets = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        program_offset = low[repeated_view] + pair_offsets

        overlap_seconds = (
            np.minimum(view_end[view_begin:view_stop][repeated_view], channel_program_end[program_offset])
            - np.maximum(view_start[view_begin:view_stop][repeated_view], channel_program_start[program_offset])
        )
        keep = (overlap_seconds > 0) & (program_show[program_begin + program_offset] != 0)
        if not keep.any():
            continue

        view_parts.append(
            view_index[view_begin:view_stop][repeated_view[keep]].astype(np.int32)
        )
        program_parts.append(
            program_index[program_begin + program_offset[keep]].astype(np.int32)
        )
        overlap_parts.append(overlap_seconds[keep].astype(np.int32))

    if view_parts:
        pair_view_index = np.concatenate(view_parts)
        pair_program_index = np.concatenate(program_parts)
        pair_overlap = np.concatenate(overlap_parts)
    else:
        pair_view_index = np.array([], dtype=np.int32)
        pair_program_index = np.array([], dtype=np.int32)
        pair_overlap = np.array([], dtype=np.int32)

    return {
        "view_idx": pair_view_index,
        "sched_idx": pair_program_index,
        "overlap_s": pair_overlap,
        "n_candidates": total_candidates,
        "end_monotonic": bool(end_monotonic),
    }


def build_airing_table(pair_frame, ratio_threshold=RATIO_THRESHOLD):
    """Aggregate interval pairs into user-show-airing rows with a ratio flag."""
    airing = pair_frame.groupby(
        ["user_id", "tv_show_id", "sched_idx"], sort=False
    ).agg(
        watched_s=("overlap_s", "sum"),
        show_duration=("show_duration", "first"),
    ).reset_index()
    airing["ratio"] = (
        np.minimum(airing["watched_s"], airing["show_duration"])
        / airing["show_duration"].replace(0, np.nan)
    )
    airing["qualified"] = airing["ratio"] >= ratio_threshold
    return airing


def airings_exceeding_duration(airing_frame):
    """Return the number of airing rows where watched_s exceeds show_duration."""
    return int((airing_frame["watched_s"] > airing_frame["show_duration"]).sum())


def build_aggregated_items(pair_frame, ratio_threshold=RATIO_THRESHOLD):
    """Build aggregated user-show items and return airing, user-show, and items."""
    airing = build_airing_table(pair_frame, ratio_threshold=ratio_threshold)
    user_show = airing.groupby(
        ["user_id", "tv_show_id"], sort=False
    ).agg(
        total_watched_s=("watched_s", "sum"),
        total_show_duration_s=("show_duration", "sum"),
        qualified_airings=("qualified", "sum"),
    ).reset_index()
    user_show["aggregate_ratio"] = (
        np.minimum(user_show["total_watched_s"], user_show["total_show_duration_s"])
        / user_show["total_show_duration_s"].replace(0, np.nan)
    )
    user_show["aggregated_qualified"] = user_show["aggregate_ratio"] >= ratio_threshold
    items = user_show[
        user_show["aggregated_qualified"] & (user_show["qualified_airings"] >= 1)
    ].copy()
    return airing, user_show, items


def items_for_weeks(pair_frame, views_frame, week_values, ratio_threshold=RATIO_THRESHOLD):
    """Filter interval pairs to selected ISO weeks and return aggregated items."""
    week_values = np.asarray(list(week_values))
    view_week = views_frame["iso_week"].to_numpy()
    pair_week = view_week[pair_frame["view_idx"].to_numpy()]
    mask = np.isin(pair_week, week_values)
    subset = pair_frame.loc[mask]
    _, _, items = build_aggregated_items(subset, ratio_threshold=ratio_threshold)
    return items


def items_for_weeks_with_ratio(pair_frame, views_frame, week_values, ratio_threshold):
    """Return aggregated items for weeks using an explicit ratio threshold."""
    return items_for_weeks(
        pair_frame, views_frame, week_values, ratio_threshold=ratio_threshold
    )


def items_for_week_with_ratio(pair_frame, views_frame, week_values, ratio_threshold):
    """Alias for items_for_weeks_with_ratio kept for the requested API."""
    return items_for_weeks_with_ratio(
        pair_frame, views_frame, week_values, ratio_threshold=ratio_threshold
    )


def _views_from_intervals(intervals, user_id=1, vsetv_id=10):
    """Build a typed view frame from (start, stop) second intervals."""
    rows = []
    for start, stop in intervals:
        rows.append({
            "user_id": user_id,
            "vsetv_id": vsetv_id,
            "start_time": pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=start),
            "stop_time": pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=stop),
            "duration": stop - start,
        })
    frame = pd.DataFrame(rows)
    frame["user_id"] = frame["user_id"].astype("uint64")
    frame["vsetv_id"] = frame["vsetv_id"].astype("int64")
    frame["start_time"] = frame["start_time"].astype("datetime64[ns]")
    frame["stop_time"] = frame["stop_time"].astype("datetime64[ns]")
    frame["duration"] = frame["duration"].astype("int64")
    return frame


@_requires_pytest
def test_four_identical_intervals_collapse_to_one():
    """Four copies of one interval must collapse to a single row."""
    frame = _views_from_intervals([(0, 60)] * 4)
    merged = merge_overlapping_views(frame)
    assert merged.shape[0] == 1
    assert int(merged.loc[0, "duration"]) == 60


@_requires_pytest
def test_two_partially_overlapping_intervals_collapse_to_one():
    """Two partially overlapping intervals must collapse to one span."""
    frame = _views_from_intervals([(0, 60), (30, 90)])
    merged = merge_overlapping_views(frame)
    assert merged.shape[0] == 1
    assert int(merged.loc[0, "duration"]) == 90


@_requires_pytest
def test_two_touching_intervals_stay_separate():
    """Intervals that only touch at an endpoint must stay separate."""
    frame = _views_from_intervals([(0, 60), (60, 120)])
    merged = merge_overlapping_views(frame)
    assert merged.shape[0] == 2


@_requires_pytest
def test_three_disjoint_intervals_stay_separate():
    """Three disjoint intervals must stay separate."""
    frame = _views_from_intervals([(0, 10), (20, 30), (40, 50)])
    merged = merge_overlapping_views(frame)
    assert merged.shape[0] == 3


@_requires_pytest
def test_overlap_pairs_match_bruteforce():
    """Overlap pairs must match a brute-force scan on small random data."""
    rng = np.random.default_rng(42)
    base = pd.Timestamp("2020-03-09")
    channels = [1, 2, 3]
    schedule_rows = []
    show_id = 1000
    for channel in channels:
        t = 0
        for _ in range(14):
            duration = int(rng.integers(60, 240))
            schedule_rows.append({
                "channel_id": channel,
                "start_time": base + pd.Timedelta(seconds=t),
                "duration": duration,
                "tv_show_id": 0 if rng.random() < 0.15 else show_id,
            })
            t += duration
            show_id += 1
    schedule = pd.DataFrame(schedule_rows)

    view_rows = []
    for _ in range(50):
        channel = int(rng.choice(channels))
        channel_end = int(
            (schedule.loc[schedule["channel_id"] == channel, "duration"].sum())
        )
        start = int(rng.integers(0, channel_end - 1))
        duration = int(rng.integers(10, 300))
        view_rows.append({
            "user_id": np.uint64(1),
            "vsetv_id": channel,
            "start_time": base + pd.Timedelta(seconds=start),
            "stop_time": base + pd.Timedelta(seconds=start + duration),
            "duration": duration,
        })
    first = schedule.iloc[0]
    touch_index = len(view_rows)
    view_rows.append({
        "user_id": np.uint64(1),
        "vsetv_id": int(first["channel_id"]),
        "start_time": first["start_time"] - pd.Timedelta(seconds=100),
        "stop_time": first["start_time"],
        "duration": 100,
    })
    views = pd.DataFrame(view_rows)
    views["user_id"] = views["user_id"].astype("uint64")
    views["vsetv_id"] = views["vsetv_id"].astype("int64")
    views["duration"] = views["duration"].astype("int64")

    arrays = build_overlap_pairs(views, schedule)
    actual = set(zip(
        arrays["view_idx"].tolist(),
        arrays["sched_idx"].tolist(),
        arrays["overlap_s"].tolist(),
    ))

    expected = set()
    view_start = to_epoch_seconds(views["start_time"]).to_numpy()
    view_end = view_start + views["duration"].to_numpy()
    sched_start = to_epoch_seconds(schedule["start_time"]).to_numpy()
    sched_end = sched_start + schedule["duration"].to_numpy()
    sched_channel = schedule["channel_id"].to_numpy()
    sched_show = schedule["tv_show_id"].to_numpy()
    view_channel = views["vsetv_id"].to_numpy()
    for view_index in range(len(views)):
        for sched_index in range(len(schedule)):
            if view_channel[view_index] != sched_channel[sched_index]:
                continue
            overlap = int(
                min(view_end[view_index], sched_end[sched_index])
                - max(view_start[view_index], sched_start[sched_index])
            )
            if overlap > 0 and sched_show[sched_index] != 0:
                expected.add((view_index, sched_index, overlap))
    assert actual == expected
    assert all(view != touch_index for view, _, _ in actual)


@_requires_pytest
def test_merged_views_have_no_airings_exceeding_duration():
    """Merged duplicate views must not create watched_s above show_duration."""
    rng = np.random.default_rng(42)
    base = pd.Timestamp("2020-03-09")
    schedule_rows = []
    for channel in [10, 20]:
        t = 0
        for _ in range(5):
            duration = int(rng.integers(100, 180))
            schedule_rows.append({
                "channel_id": channel,
                "start_time": base + pd.Timedelta(seconds=t),
                "duration": duration,
                "tv_show_id": 2000 + channel + t,
            })
            t += duration
    schedule = pd.DataFrame(schedule_rows)

    user_one = _views_from_intervals(
        [(0, 60), (30, 90), (200, 260), (0, 60), (0, 60), (200, 260), (30, 90)],
        user_id=1,
        vsetv_id=10,
    )
    user_two = _views_from_intervals(
        [(100, 160), (100, 160), (150, 300), (400, 470), (400, 470)],
        user_id=2,
        vsetv_id=20,
    )
    views = pd.concat([user_one, user_two], ignore_index=True)
    merged = merge_overlapping_views(views)
    arrays = build_overlap_pairs(merged, schedule)
    pair = pd.DataFrame({
        "user_id": merged["user_id"].to_numpy()[arrays["view_idx"]],
        "tv_show_id": schedule["tv_show_id"].to_numpy()[arrays["sched_idx"]],
        "overlap_s": arrays["overlap_s"],
        "view_idx": arrays["view_idx"],
        "sched_idx": arrays["sched_idx"],
        "view_duration": merged["duration"].to_numpy()[arrays["view_idx"]],
        "show_duration": schedule["duration"].to_numpy()[arrays["sched_idx"]],
    })
    airing = build_airing_table(pair, ratio_threshold=RATIO_THRESHOLD)
    assert airings_exceeding_duration(airing) == 0


if __name__ == "__main__":
    test_four_identical_intervals_collapse_to_one()
    test_two_partially_overlapping_intervals_collapse_to_one()
    test_two_touching_intervals_stay_separate()
    test_three_disjoint_intervals_stay_separate()
    test_overlap_pairs_match_bruteforce()
    test_merged_views_have_no_airings_exceeding_duration()
    print("src/data.py self-tests passed")
