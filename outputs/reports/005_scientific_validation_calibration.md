# Experiment 005 — Scientific Validation & Calibration of the Football Score Engine

## 1. Objective

Validate whether Experiment 004 prototype role scores are scientifically defensible. No new weights or production coefficients are declared.

## 2. Dataset used

Data root: `/home/platform/DataPlatform/tmp/master_data_warehouse`. Inputs are Experiment 001–004 artefacts, especially prototype role scores, feature matrix, dimension scores, weights, normalization decisions, and quality flags.

## 3. Eligible populations per role

- GK: 12 players validated; average confidence 70.79; readiness Prototype
- CB: 25 players validated; average confidence 69.32; readiness Prototype
- FB: 23 players validated; average confidence 69.17; readiness Prototype
- MID: 34 players validated; average confidence 73.14; readiness Prototype
- WINGER: 25 players validated; average confidence 66.84; readiness Prototype
- CF: 22 players validated; average confidence 67.88; readiness Prototype

## 4. Metrics used per role

Metrics are inherited from Experiment 004 prototype score inputs. No new metrics are introduced. Manual-review metrics remain excluded from prototype scoring.

## 5. Metrics excluded and why

Non-ready metrics from Experiment 002 and manual-direction metrics from Experiment 004 are excluded. Exclusion rationale is preserved in `004_metric_direction_registry.csv` and validation flags in `quality_flags.csv`.

## 6. Direction registry

No direction changes are made in Experiment 005. Direction decisions are validated indirectly through sensitivity, confidence, and football-review candidate outputs.

## 7. Metric weighting methodology

Weights are not re-estimated. Experiment 005 validates existing prototype weights using bootstrap, weight confidence intervals, sensitivity, and readiness checks.

## 8. Dimension weighting methodology

Dimension weights from Experiment 004 are validated through dimension-score bootstrap, rank stability, score independence, and redundancy analysis.

## 9. Prototype score formula

The Experiment 004 prototype score formula is treated as fixed: normalized oriented metric scores -> weighted dimension scores -> weighted prototype role score.

## 10. Sensitivity analysis

Sensitivity rows: 2205. Tests include metric removal proxy, dimension removal, perturbation, and normalization proxy.

## 11. Main findings

All roles remain Prototype. Confidence indices are computed for every scored player. Score redundancy above the configured threshold was not found in this local sample.

## 12. Limitations

The local sample is not the full multi-competition/two-season population. Match-level bootstrap is approximated from score/dimension/metric artefacts because match-grain score histories are not materialized. Football expert review, cross-season validation, and cross-league validation remain pending.

## 13. Why this is not production-final yet

Production deployment requires full-population rerun, real match-level temporal validation, football expert review, and cross-season/cross-league robustness. No production coefficients are declared here.

## 14. What Experiment 006 should validate next

Experiment 006 should materialize match-level or season-split score histories and run temporal/cross-competition validation before any production deployment decision.

## Output tables

- bootstrap_statistics: 141 rows
- score_confidence: 141 rows
- reliability_summary: 6 rows
- score_correlations: 595 rows
- score_mutual_information: 595 rows
- score_redundancy: 0 rows
- rank_stability: 141 rows
- weight_confidence_intervals: 43 rows
- production_readiness: 48 rows
- football_review_candidates: 285 rows
- score_sensitivity_analysis: 2205 rows
- rank_instability_players: 157 rows
- explainability_contributions: 846 rows
- dimension_independence_pca: 10 rows
