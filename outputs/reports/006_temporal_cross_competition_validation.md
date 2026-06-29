# Experiment 006 — Temporal, Match-Level & Cross-Competition Validation

## 1. Objective
Materialize match-level and season-split histories and validate temporal/cross-context stability without declaring production coefficients.

## 2. Dataset used
`/home/platform/DataPlatform/tmp/master_data_warehouse` plus Experiment 002–005 outputs.

## 3. Match-level score materialization
Created `006_match_level_score_history.csv` and metric mapping status table.

## 4. Season-split methodology
Full, first-half, second-half, and rolling 3/5-match windows are produced where possible; match order is used when dates are limited.

## 5. Temporal stability results
See `006_temporal_stability.csv` and role summary in JSON report.

## 6. Leave-one-season-out validation
See `006_leave_one_season_out.csv`; insufficient groups are explicitly marked.

## 7. Leave-one-competition-out validation
See `006_leave_one_competition_out.csv`; insufficient groups are explicitly marked.

## 8. Population drift analysis
KS, Wasserstein, shift metrics, and summary statuses are exported.

## 9. Team context sensitivity
Team-context correlations and adjustment candidates are diagnosed only; no correction is applied.

## 10. Minutes threshold sensitivity
Threshold recommendations are exported without changing official thresholds.

## 11. Calibration curves
Score bands are compared against confidence, minutes reliability, temporal stability, and review flags.

## 12. Football expert review workflow
Review workflow table is generated with empty reviewer decision/comment columns.

## 13. Production-candidate gate
All roles remain research/validation status on this local sample.

## 14. Main findings
The validation layer is materialized, but evidence is insufficient for production.

## 15. Limitations
Local sample scope, limited seasons/competitions, partial match-level mappings, pending expert review.

## 16. Why production is still not declared
No full-population multi-season/multi-competition validation and no completed expert review.

## 17. Recommended Experiment 007
Full-population validation with completed expert workflow and production-candidate calibration gate.
