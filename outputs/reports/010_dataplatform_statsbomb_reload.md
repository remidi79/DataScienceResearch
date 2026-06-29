# Experiment 010 — DataPlatform StatsBomb Coverage Expansion & Reload Execution

## 1. Objective
Execute/prepare the missing DataPlatform reload workflow.

## 2. Why Experiment 010 was needed
Experiment 009 showed the current root failed production coverage.

## 3. Current root limitations
Insufficient competitions, seasons, matches, player-match rows, events, and lineups.

## 4. Source discovery results
Discovered 3035 candidate source files.

## 5. Source-to-target mapping
Mapped 1 datasets; blocked 10.

## 6. Dry-run reload plan
See `010_dry_run_reload_plan.csv` and `010_dry_run_reload_summary.md`.

## 7. Reload execution result if executed
Run mode: validate_target.

## 8. Target root coverage
See `010_target_coverage_summary.csv`.

## 9. ID consistency validation
See `010_id_consistency_validation.csv`.

## 10. Data quality issues
See loaded dataset quality and ID/orphan/duplicate tables.

## 11. Experiment 009 compatibility result
PASS.

## 12. Whether the target root is ready
FAIL.

## 13. Exact next command
`uv run python experiments/009_full_data_reload_orchestration.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode validate_loaded_root`

## 14. Why production is still not declared
This experiment loads/validates data only; it does not score, change coefficients, create bundles, or deploy.

## 15. Recommended Experiment 011
Provider/API ingestion or upstream DataPlatform load for missing competitions and seasons if local sources cannot meet the target.
