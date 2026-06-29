# Experiment 014 — Event-Derived / Open-Data Fallback Feasibility

## Objective
Assess whether local event and lineup data can support a clearly labelled research fallback layer.

## Why Experiment 014 was needed
Licensed provider backfill remains blocked by credentials/provider access, so we audited what can be safely derived from local events without fabricating provider-direct stats.

## Current credential blocker
Provider access remains blocked. This experiment does not call provider APIs and does not use credentials.

## Available local event data
- Matches: 11
- Competitions: 1
- Seasons: 1
- Players: 295
- Teams: 18
- Events: 35082

## Event field coverage
See `outputs/tables/014_event_field_coverage.csv` and `outputs/figures/014_event_field_coverage.png`.

## Derivable metric catalog
Safely derivable: 15; partially derivable: 16; non-derivable/blocked: 2.

## Role-dimension support matrix
See `outputs/tables/014_role_dimension_support_matrix.csv`.

## Player-match event-derived metrics
Rows: 350. These rows are labelled `event_derived_research_fallback_not_provider_direct`.

## Team-match event-derived metrics
Rows: 22. These rows are labelled `event_derived_research_fallback_not_provider_direct`.

## Player-season event-derived metrics
Rows: 295. Output name is `event_derived_player_season_metrics` and is not provider-direct.

## Comparison with provider stats where possible
See `outputs/tables/014_event_vs_provider_metric_comparison.csv`. Comparisons are diagnostic only and do not certify equivalence.

## Research fallback score feasibility
See `outputs/tables/014_research_fallback_score_feasibility.csv`. No final fallback scores are computed.

## Limitations
- Local sample is limited.
- Event-derived definitions differ from StatsBomb provider-direct aggregate endpoints.
- Tracking/360/video-dependent metrics remain blocked.
- Role minutes depend on lineup interval quality.

## Why this is not production
This does not use licensed provider backfill, does not reproduce provider-direct metrics, does not change coefficients, does not generate a production bundle, and does not mark any role production-ready.

## Recommended Experiment 015
Only after review: design a separate research-only event-derived score prototype with explicit lineage and no production claims, or return to licensed provider backfill once credentials are active.
