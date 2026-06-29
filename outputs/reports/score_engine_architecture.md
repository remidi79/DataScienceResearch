# Score Engine Architecture

Pipeline:

Player -> Raw Metrics -> Normalization -> Direction Correction -> Feature Selection -> Metric Weights -> Dimension Scores -> Role Calibration -> Percentile -> Confidence -> Explainability -> Production Score Object.

Reusable classes are implemented in `src/football_score_engine_research/production_engine.py`:

- MetricContribution
- DimensionScore
- ScoreConfidence
- ScoreExplanation
- RoleCalibration
- PlayerScore
- ScoreEngine

The implementation is configuration-driven. Metrics, dimensions, directions, weights, confidence thresholds, percentile rules, and eligibility rules are loaded from generated configuration. No business logic should hardcode score metrics.
