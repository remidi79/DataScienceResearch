# Experiment 010 — Experiment 009 Validation Result

Command:

`uv run python experiments/009_full_data_reload_orchestration.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode validate_loaded_root`

Return code: 0

## Stdout

```
{
  "experiment_id": "009",
  "title": "Full DataPlatform Reload & End-to-End Recalibration Orchestration",
  "generated_at": "2026-06-29T13:54:43.096613+00:00",
  "data_root": "/home/platform/DataPlatform/tmp/master_data_warehouse_full",
  "target_root": "",
  "run_mode": "validate_loaded_root",
  "coverage_summary": {
    "competitions": 0,
    "seasons": 0,
    "matches": 0,
    "teams": 0,
    "players": 0,
    "player_match_rows": 0,
    "player_season_rows": 0,
    "events": 0,
    "lineups": 0
  },
  "data_readiness_result": "FAIL",
  "rerun_pipeline_allowed": false,
  "rerun_pipeline_executed": false,
  "production_coefficients_declared": false,
  "production_candidate_bundle_generated": false,
  "reports": [
    "009_full_population_data_contract.json",
    "009_full_population_data_contract.md",
    "009_full_data_reload_plan.md",
    "009_rerun_pipeline_manifest.json",
    "009_pipeline_failure_reason.md"
  ],
  "tables": [
    "009_data_root_discovery.csv",
    "009_missing_data_gap_analysis.csv",
    "009_data_quality_issues.csv",
    "009_id_consistency_audit.csv",
    "009_reload_tasks.csv",
    "009_data_readiness_gate.csv",
    "009_rerun_pipeline_status.csv"
  ],
  "figures_generated": 7,
  "figure_paths": [
    "outputs/figures/009_data_coverage_summary.png",
    "outputs/figures/009_competition_season_coverage.png",
    "outputs/figures/009_dataset_row_counts.png",
    "outputs/figures/009_missing_data_gap_summary.png",
    "outputs/figures/009_data_readiness_gate.png",
    "outputs/figures/009_reload_task_priority.png",
    "outputs/figures/009_rerun_pipeline_status.png"
  ],
  "blockers": [
    "missing_or_insufficient_competitions",
    "missing_or_insufficient_seasons",
    "missing_or_insufficient_matches",
    "missing_or_insufficient_teams",
    "missing_or_insufficient_player_match_rows",
    "missing_or_insufficient_player_season_rows",
    "missing_or_insufficient_events",
    "missing_or_insufficient_lineups",
    "missing_dataset:player_match_stats_direct",
    "missing_dataset:team_match_stats_direct",
    "missing_dataset:player_season_stats_direct",
    "missing_dataset:team_season_stats_direct",
    "missing_dataset:silver_events",
    "missing_dataset:silver_lineups",
    "missing_dataset:silver_matches",
    "missing_dataset:competition_metadata",
    "missing_dataset:season_metadata",
    "missing_dataset:team_metadata",
    "missing_dataset:player_metadata"
  ]
}

```

## Stderr

```

```
