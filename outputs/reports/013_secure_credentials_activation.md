# Experiment 013 — Secure Credentials Activation & Licensed Backfill Runbook

## 1. Objective
Prepare secure credential activation and licensed backfill execution workflow.

## 2. Why Experiment 013 was needed
Experiment 012 is implemented but blocked by missing runtime credentials.

## 3. Current blocker
licensed StatsBomb credential variables are not detected in the current process/config scan.

## 4. Secret handling audit
See `outputs/tables/013_secret_handling_audit.csv`.

## 5. Safe env template
See `configs/013_statsbomb_credentials.env.example`.

## 6. Credential activation runbook
See `outputs/reports/013_credentials_activation_runbook.md`.

## 7. Safe execution script
See `scripts/run_licensed_backfill_safe.sh`.

## 8. Credential helper result
Credentials status: missing.

## 9. Current blocked-state validation
Experiment 012 credentials preflight return code: 0.

## 10. Exact next commands
See `outputs/reports/013_exact_next_commands.md`.

## 11. Why production is still not declared
No scoring, coefficient changes, production bundle, API integration, or fake data generation occurred.

## 12. Recommended next step
Activate licensed credentials securely and run the safe backfill script.
