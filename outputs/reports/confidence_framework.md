# Confidence Framework

Confidence is configuration-driven and does not alter metric coefficients.

Formula:

`confidence = 0.25*bootstrap_stability + 0.20*weight_stability + 0.15*minutes_reliability + 0.15*population_reliability + 0.10*metric_coverage + 0.10*data_quality + 0.05*validation_status_score`

Inputs:
- Bootstrap stability from Experiment 005/006.
- Weight stability from Experiment 004/005.
- Minutes reliability from eligibility and score confidence artefacts.
- Population reliability from role sample size, seasons, and competitions.
- Metric coverage from match/season metric availability.
- Data quality from quality flags.
- Validation status from production candidate gates.

Bands:
- Excellent: >=85
- Good: >=75
- Adequate: >=65
- Research Only: <65
