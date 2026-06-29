# Experiment 013 — Credentials Activation Runbook

No real secret values belong in git, reports, notebooks, or shell history shared with others.

## 1. Required credential variables
Use the variables required by the existing DataPlatform client. Username/password is the current supported path:

```bash
export STATSBOMB_USERNAME="<licensed_username>"
export STATSBOMB_PASSWORD="<licensed_password>"
```

If the provider account uses token-based auth, set only the approved variables:

```bash
export STATSBOMB_API_TOKEN="<licensed_api_token>"
```

## 2. Current shell session only
Set credentials in the active shell, a systemd EnvironmentFile, or CI secret store. Do not commit them.

## 3. Check detection without printing values

```bash
python3 scripts/check_statsbomb_credentials.py --json
```

If `python3` is unavailable, use:

```bash
uv run python scripts/check_statsbomb_credentials.py --json
```

## 4. Run Experiment 012 credentials preflight

```bash
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode credentials_preflight
```

## 5. Provider access preflight

```bash
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode provider_access_preflight
```

## 6. Dry-run backfill

```bash
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode dry_run_backfill
```

## 7. Execute only after dry run passes

```bash
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode execute_backfill --resume
```

## 8. Validate all gates

```bash
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode validate_all_gates
```

## 9. If credentials fail
Check variable names, shell scope, systemd environment files, and DataPlatform-approved config path. Do not paste secrets into reports.

## 10. If provider coverage is insufficient
Do not run scoring. Expand licensed provider coverage or update selected competitions/seasons, then rerun dry-run and gate validation.
