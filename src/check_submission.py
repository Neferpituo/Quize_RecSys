"""Validate a baseline submission file against the competition template."""

import argparse
from pathlib import Path

import pandas as pd


def load_candidates(test_schedule_path):
    """Return the set of non-zero tv_show_id values in a test schedule."""
    frame = pd.read_parquet(test_schedule_path)
    values = pd.to_numeric(frame["tv_show_id"], errors="coerce").fillna(0).astype("int64")
    return set(int(value) for value in values.unique() if int(value) != 0)


def check_submission(path, template_path, candidates):
    """Return a list of validation errors for one submission file."""
    errors = []
    template = pd.read_csv(template_path, dtype=str)
    template.columns = [column.strip() for column in template.columns]
    frame = pd.read_csv(path, dtype=str)
    frame.columns = [column.strip() for column in frame.columns]

    if list(frame.columns) != ["user_id", "tv_show_id"]:
        errors.append(f"header must be user_id,tv_show_id, got {list(frame.columns)}")
        return errors
    if list(template.columns) != ["user_id", "tv_show_id"]:
        errors.append("template header is not user_id,tv_show_id")
        return errors
    if frame.shape[0] != 1260:
        errors.append(f"expected 1260 rows, got {frame.shape[0]}")
    if template.shape[0] != 1260:
        errors.append(f"template has {template.shape[0]} rows, expected 1260")
    if frame.shape[0] == template.shape[0]:
        if frame["user_id"].tolist() != template["user_id"].tolist():
            errors.append("user_id order differs from the template")

    candidate_set = set(int(value) for value in candidates)
    for row_index, row in frame.iterrows():
        user_text = str(row["user_id"]).strip()
        if not user_text.isdigit():
            errors.append(f"row {row_index}: user_id is not a positive integer")
        tokens = str(row["tv_show_id"]).replace(",", " ").split()
        if len(tokens) != 5:
            errors.append(f"row {row_index}: expected 5 tv_show_id values, got {len(tokens)}")
            continue
        if any(not token.isdigit() for token in tokens):
            errors.append(f"row {row_index}: non-integer tv_show_id token")
            continue
        values = [int(token) for token in tokens]
        if len(set(values)) != 5:
            errors.append(f"row {row_index}: tv_show_id values are not distinct")
        if any(value == 0 for value in values):
            errors.append(f"row {row_index}: zero tv_show_id found")
        missing = [value for value in values if value not in candidate_set]
        if missing:
            errors.append(f"row {row_index}: ids outside candidate pool {missing}")
    return errors


def main():
    """Run the command-line submission check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--template", type=Path, default=Path("data_par/submission.csv"))
    parser.add_argument(
        "--test-schedule",
        type=Path,
        default=Path("data_par/raw/export_arh_31-42-final.parquet"),
    )
    args = parser.parse_args()
    candidates = load_candidates(args.test_schedule)
    errors = check_submission(args.submission, args.template, candidates)
    if errors:
        print(f"FAILED: {args.submission}")
        for error in errors[:20]:
            print(" -", error)
        raise SystemExit(1)
    print(f"PASSED: {args.submission} | rows=1260 | candidate_pool={len(candidates)}")


if __name__ == "__main__":
    main()
