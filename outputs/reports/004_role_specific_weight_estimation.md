# Experiment 004 — Role-Specific Weight Estimation & Prototype Score Formula

Generated: 2026-06-29T09:59:12.835035+00:00

This is a prototype scoring layer only. It is not a final production score.

## Objective
Estimate role-specific metric and dimension weights using data evidence from Experiments 002 and 003.

## Dataset used
`/home/platform/DataPlatform/tmp/master_data_warehouse` plus Experiment 002/003 tables.

## Eligible populations per role
- GK: players=12, metrics=5, dimensions=3
- CB: players=25, metrics=9, dimensions=9
- FB: players=23, metrics=8, dimensions=7
- MID: players=34, metrics=7, dimensions=6
- WINGER: players=25, metrics=7, dimensions=6
- CF: players=22, metrics=7, dimensions=4

## Metrics excluded and why
Manual-review direction metrics and non-candidate-ready metrics are excluded from prototype scoring; all remain visible in `004_metric_direction_registry.csv`.

## Direction registry
Directions are transparent rule-based classifications. Unclear contextual metrics are marked manual_review_required and excluded.

## Metric weighting methodology
Equal, PCA loading, variance contribution, stability-adjusted, entropy, bootstrap stability, and shrinkage ensemble with caps.

## Dimension weighting methodology
Blend explained variance, reliable metric count/stability, bootstrap consistency, PCA contribution, redundancy penalty, and small-sample penalty.

## Prototype score formula
Metric scores are normalized and direction-oriented, combined into 0-100 dimension scores, then combined by dimension weights into 0-100 prototype role scores.

## Sensitivity analysis
Rank correlations across equal/PCA/stability/entropy/ensemble methods and rank-instability players are exported.

## Main findings
See JSON report and role summary tables.

## Limitations
Local sample only; not production-final; manual review required for some metric directions; Experiment 005 validation required.

## Why this is not production-final yet
No cross-season/cross-league validation, no calibration, limited local population, and football review still required.

## What Experiment 005 should validate next
Robustness, calibration, rank stability, confidence intervals, league/season fairness, and football interpretability.
