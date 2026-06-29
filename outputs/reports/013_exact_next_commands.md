# Experiment 013 — Exact Next Commands

## Section A — Before credentials are available

The pipeline remains blocked. Do not run ingestion, scoring, production bundle generation, or API integration.

Run only:

```bash
cd /home/platform/DataScienceResearch
python scripts/check_statsbomb_credentials.py --json
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode credentials_preflight
```

## Section B — After credentials are available

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode credentials_preflight
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode provider_access_preflight
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode dry_run_backfill
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode execute_backfill --resume
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode validate_backfill
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode validate_all_gates
```

If all gates pass:

```bash
uv run python experiments/009_full_data_reload_orchestration.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode rerun_research_pipeline
```
