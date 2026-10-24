"""Calibrate local target definitions against Kaggle baseline scores."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path.cwd()
for candidate in [Path.cwd(), *Path.cwd().parents]:
    if (candidate / "src" / "data.py").exists():
        PROJECT_ROOT = candidate
        break
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import recommend_personal
from src.eval import build_ground_truth, build_unified_fallback, fill_with_fallback, map_at_k
from src.pipeline import (
    EVAL_DIR,
    HISTORY_WEEKS,
    PRIVATE_SCORES,
    PUBLIC_SCORES,
    SOLUTION_DIR,
    build_window_items,
    load_primary_mode,
    load_validation_context,
    save_json,
)

started = time.time()
views, pair, candidate_set = load_validation_context()
hist_items = build_window_items(
    pair, views, HISTORY_WEEKS, 0.8, candidate_set=candidate_set
)
recent_items = build_window_items(
    pair, views, range(17, 21), 0.8, candidate_set=candidate_set
)
global_top5 = build_unified_fallback(hist_items, candidate_set, k=5)
recent_top5 = build_unified_fallback(recent_items, candidate_set, k=5)

rows = []
for mode in ["T03", "T05", "T08", "T_top5"]:
    ground_truth = build_ground_truth(
        pair, views, range(21, 31), candidate_set, mode
    )
    users = list(ground_truth)
    recommendations = {
        "global_pop": {user: list(global_top5) for user in users},
        "recent_pop": {user: list(recent_top5) for user in users},
        "personal": recommend_personal(
            hist_items, candidate_set, global_top5, users, k=5
        ),
    }
    for method, recs in recommendations.items():
        filled = {
            user: fill_with_fallback(recs.get(user, []), global_top5, k=5)
            for user in users
        }
        score, n_users = map_at_k(filled, ground_truth, k=5)
        rows.append({
            "mode": mode,
            "method": method,
            "map_at_5": float(score),
            "n_users_eval": int(n_users),
        })

save_json(EVAL_DIR / "eval_top5.json", rows)
decision = load_primary_mode()

lines = [
    "# Calibration",
    "",
    "Local MAP@5 on validation weeks 21-30 with candidate pool from train_schedule_2.",
    "",
    "| mode | global_pop | recent_pop | personal |",
    "|---|---:|---:|---:|",
]
for mode in ["T03", "T05", "T08", "T_top5"]:
    values = {
        row["method"]: row["map_at_5"] for row in rows if row["mode"] == mode
    }
    lines.append(
        f"| {mode} | {values['global_pop']:.6f} | {values['recent_pop']:.6f} | {values['personal']:.6f} |"
    )
lines.extend([
    "",
    "Kaggle baseline scores:",
    "",
    "| method | public | private |",
    "|---|---:|---:|",
])
for method in ["recent_pop", "global_pop", "personal"]:
    lines.append(
        f"| {method} | {PUBLIC_SCORES[method]:.5f} | {PRIVATE_SCORES[method]:.5f} |"
    )
lines.extend([
    "",
    "## Primary target decision",
    "",
])
if decision["mode"] == "T_top5":
    lines.append(
        "T_top5 is accepted as the primary local target. "
        f"The local recent_pop MAP@5 on T_top5 is {decision['recent_local']:.6f}, "
        f"the public recent_pop score is {decision['recent_public']:.5f}, and the relative gap is "
        f"{decision['relative_gap']:.2%}, within the 30% limit. "
        f"The T_top5 baseline order is {decision['order']}, which matches the public order "
        "recent_pop > global_pop > personal. This is an explicit departure from the earlier "
        "T08 rule because the Kaggle target appears softer and friendlier to repeat consumption."
    )
else:
    lines.append(
        "T_top5 is not accepted as the primary local target. "
        f"The local recent_pop MAP@5 on T_top5 is {decision['recent_local']:.6f}, "
        f"the public recent_pop score is {decision['recent_public']:.5f}, and the relative gap is "
        f"{decision['relative_gap']:.2%}; order_matches={decision['order_matches']}. "
        "The primary local target remains T05 and the divergence from Kaggle is documented."
    )
lines.extend([
    "",
    "## Model status",
    "",
    "No model public score is available at calibration time. No model is confirmed to beat "
    "recent_pop on public; personalization has not yet demonstrated a gain over fresh popularity.",
])
(SOLUTION_DIR / "calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Saved:", EVAL_DIR / "eval_top5.json")
print("Saved:", SOLUTION_DIR / "calibration.md")
print("Primary mode:", decision["mode"])
print("Recent T_top5 local/public:", decision["recent_local"], decision["recent_public"])
print("Relative gap:", round(decision["relative_gap"], 6))
print("Order matches:", decision["order_matches"])
print("Elapsed seconds:", round(time.time() - started, 2))
