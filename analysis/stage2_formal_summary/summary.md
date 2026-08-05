# Stage-2 Experiment Summary

Reference baseline: B0 -> Official default baseline

| Experiment | Precision | Recall | mAP50 | mAP50-95 | Scratch AP50-95 | Inference (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Exp_A | 0.8474 | 0.6459 | 0.7537 | 0.4136 | 0.3714 | 1.6793 |
| Exp_B | 0.4261 | 0.5818 | 0.4495 | 0.2442 | 0.0999 | 2.3865 |
| Exp_C | 0.6977 | 0.5293 | 0.5791 | 0.3137 | 0.3335 | 2.7234 |

## Overfitting Check

- Exp_A: best_epoch=95, best_curve_mAP50-95=0.4159, last_epoch=100, last_curve_mAP50-95=0.4125, overfit_flag=False. No severe post-50 decline detected.
- Exp_B: best_epoch=94, best_curve_mAP50-95=0.2450, last_epoch=100, last_curve_mAP50-95=0.2374, overfit_flag=False. No severe post-50 decline detected.
- Exp_C: best_epoch=88, best_curve_mAP50-95=0.3135, last_epoch=100, last_curve_mAP50-95=0.3047, overfit_flag=False. No severe post-50 decline detected.

## Synergy Check

- Scratch AP50-95 super-additive: False
- Overall mAP50-95 super-additive: False
- Interpretation: At least one single-factor branch is negative versus the reference baseline, so a strict 1+1>2 claim is not scientifically supported.
- Scratch deltas vs reference: Exp_A=0.0947, Exp_B=-0.1768, Exp_C=0.0569
- mAP50-95 deltas vs reference: Exp_A=0.0050, Exp_B=-0.1644, Exp_C=-0.0949
