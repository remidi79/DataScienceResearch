# Explainability Framework

Every score must explain metric-level, dimension-level, and overall-role contributions.

Metric contribution = oriented_normalized_metric_value * metric_weight * dimension_weight.

Positive contributors are the highest positive deviations from the player's role score baseline. Negative contributors are the lowest contribution deltas.

The API must return: metric, raw value, normalized value, direction, metric weight, dimension, dimension weight, contribution, contribution percentage, and flags.

The framework is football-reviewable but does not automatically judge whether a player is good or bad. Review flags route anomalies to experts.
