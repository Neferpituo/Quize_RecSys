# Calibration

Local MAP@5 on validation weeks 21-30 with candidate pool from train_schedule_2.

| mode | global_pop | recent_pop | personal |
|---|---:|---:|---:|
| T03 | 0.079160 | 0.216015 | 0.082499 |
| T05 | 0.068645 | 0.155546 | 0.069826 |
| T08 | 0.054378 | 0.087168 | 0.043858 |
| T_top5 | 0.002170 | 0.023439 | 0.019748 |

Kaggle baseline scores:

| method | public | private |
|---|---:|---:|
| recent_pop | 0.29553 | 0.29554 |
| global_pop | 0.22583 | 0.23147 |
| personal | 0.21959 | 0.21044 |

## Primary target decision

T_top5 is not accepted as the primary local target. The local recent_pop MAP@5 on T_top5 is 0.023439, the public recent_pop score is 0.29553, and the relative gap is 92.07%; order_matches=False. The primary local target remains T05 and the divergence from Kaggle is documented.

## Model status

No model public score is available at calibration time. No model is confirmed to beat recent_pop on public; personalization has not yet demonstrated a gain over fresh popularity.
