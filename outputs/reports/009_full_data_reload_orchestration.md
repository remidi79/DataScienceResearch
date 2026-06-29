# Experiment 009 — Full DataPlatform Reload & End-to-End Recalibration Orchestration

## 1. Objective
Create the reload and rerun orchestration layer required before production-candidate generation.

## 2. Why Experiment 009 was needed
Experiment 008 found the local root below production target.

## 3. Current data coverage
```json
{
  "competitions": 0,
  "seasons": 0,
  "matches": 0,
  "teams": 0,
  "players": 0,
  "player_match_rows": 0,
  "player_season_rows": 0,
  "events": 0,
  "lineups": 0
}
```

## 4. Expected production data contract
See `009_full_population_data_contract.md`.

## 5. Missing data gaps
See `009_missing_data_gap_analysis.csv`.

## 6. Reload plan
See `009_full_data_reload_plan.md` and `009_reload_tasks.csv`.

## 7. Data readiness gate result
FAIL.

## 8. Whether rerun pipeline was allowed
False.

## 9. Rerun pipeline status if executed
Executed: False.

## 10. Blockers
- missing_or_insufficient_competitions
- missing_or_insufficient_seasons
- missing_or_insufficient_matches
- missing_or_insufficient_teams
- missing_or_insufficient_player_match_rows
- missing_or_insufficient_player_season_rows
- missing_or_insufficient_events
- missing_or_insufficient_lineups
- missing_dataset:player_match_stats_direct
- missing_dataset:team_match_stats_direct
- missing_dataset:player_season_stats_direct
- missing_dataset:team_season_stats_direct
- missing_dataset:silver_events
- missing_dataset:silver_lineups
- missing_dataset:silver_matches
- missing_dataset:competition_metadata
- missing_dataset:season_metadata
- missing_dataset:team_metadata
- missing_dataset:player_metadata

## 11. Next action required
Reload DataPlatform to meet the production data contract, then run validate_loaded_root and rerun_research_pipeline.

## 12. Why production is still not declared
No full-population gate pass and no rerun of Experiments 001–008.

## 13. Recommended Experiment 010
Execute or integrate the actual DataPlatform reload workflow.
