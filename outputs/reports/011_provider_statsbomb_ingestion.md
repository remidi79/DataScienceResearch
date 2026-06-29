# Experiment 011 — Provider/API-backed StatsBomb Ingestion & Coverage Expansion

## 1. Objective
Create provider/API-backed StatsBomb ingestion workflow for `/home/platform/DataPlatform/tmp/master_data_warehouse_full`.

## 2. Why Experiment 011 was needed
Experiment 010 showed local sources were insufficient.

## 3. Provider access discovery
Rows: 10.

## 4. Provider credentials status without exposing secrets
missing. No secret values are printed.

## 5. Available competitions/seasons
Competitions: 3; seasons: 4.

## 6. Coverage selection plan
Selected competitions: 3; selected seasons: 1.

## 7. Dry-run ingestion results
See `011_dry_run_ingestion_summary.md`.

## 8. Execution result if executed
Execute ingestion run: False.

## 9. Direct stats materialization status
See `011_direct_stats_materialization_status.csv`.

## 10. Schema normalization
Schema files written under `outputs/schemas/`.

## 11. Target coverage summary
See `011_target_coverage_summary.csv`.

## 12. ID consistency validation
See `011_id_consistency_validation.csv`.

## 13. Experiment 010 validation result
PASS.

## 14. Experiment 009 validation result
PASS.

## 15. Whether the target root is now ready
FAIL.

## 16. Exact next command
`uv run python experiments/011_provider_statsbomb_ingestion.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode execute_ingestion`

## 17. Why production is still not declared
This experiment is ingestion only. No score coefficients, production bundles, or API integration were changed.

## 18. Recommended Experiment 012
Execute licensed provider ingestion/backfill once credentials and coverage are confirmed, then rerun 010 and 009 validation.
