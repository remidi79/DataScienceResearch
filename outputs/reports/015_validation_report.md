# Experiment 015 — Event-Derived Research Scouting Score Prototype

## Objective
Create a research-only player scouting score prototype from Experiment 014 event-derived outputs.

## Why Experiment 015 was needed
Experiment 014 showed a partially feasible event-derived fallback, but no role reaches production-grade sample size. This prototype tests transparent role-specific scoring mechanics without production claims.

## Inputs from Experiment 014
- Player-match rows: 350
- Player-season rows: 295
- Roles observed: CB, CF, FB, GK, MID, WINGER

## Metric selection
Only safely derivable and partially derivable metrics with usable player-season columns were included. Provider-direct, tracking, manual video, not derivable, and missing-column metrics were excluded.

## Role dimensions
Role-specific dimensions were created for GK, CB, FB, MID, WINGER, and CF. Weak dimensions are marked partial/low-confidence/sample-too-small.

## Normalization
Metrics are normalized role-by-role using role percentiles, z-scores, min-max 0-100, and confidence-adjusted values. GK is never compared with CF.

## Research-only scoring formula
`research_scouting_score = confidence-adjusted average of available dimension scores`.
No final coefficients were learned from this small sample.

## Similarity prototype
Similarity is same-role only, using normalized event-derived dimension vectors with cosine similarity and Euclidean distance fallback.

## Explainability
Every score has strongest dimensions, weakest dimensions, missing metrics, confidence reducers, derivation warnings, and a research-only explanation.

## Validation results
Validation result: PASS.

## Limitations
Only 11 local matches, 1 competition, 1 season, low role samples, event-derived formulas only, no licensed provider backfill, and no production validation.

## Why this is not production
This is research-only, event-derived, not provider-direct, not a final scouting score, not a production score, and not a production bundle. It does not replace licensed StatsBomb provider-direct scouting scores.

## Recommended next step
Review methodology and either improve the prototype with more validated data or resume licensed provider-direct ingestion before production score design.


See `outputs/tables/015_validation_results.csv` for check-level validation.
