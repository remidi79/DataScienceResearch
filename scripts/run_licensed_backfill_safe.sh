#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="/home/platform/DataPlatform"
TARGET_ROOT="/home/platform/DataPlatform/tmp/master_data_warehouse_full"
EXECUTE=0
RESUME=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      SOURCE_ROOT="$2"; shift 2 ;;
    --target-root)
      TARGET_ROOT="$2"; shift 2 ;;
    --execute)
      EXECUTE=1; shift ;;
    --resume)
      RESUME=1; shift ;;
    --force)
      FORCE=1; shift ;;
    -h|--help)
      cat <<'HELP'
Usage: scripts/run_licensed_backfill_safe.sh [--source-root PATH] [--target-root PATH] [--execute] [--resume] [--force]

Default mode is dry-run only. The script never prints credential values.
Use --execute only after credentials_preflight, provider_access_preflight, and dry_run_backfill pass.
HELP
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=(python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=(python)
elif command -v uv >/dev/null 2>&1; then
  PYTHON_BIN=(uv run python)
else
  echo "No supported Python runtime found. Install python3 or uv." >&2
  exit 127
fi

echo "Selected Python runtime: ${PYTHON_BIN[*]}"

"${PYTHON_BIN[@]}" scripts/check_statsbomb_credentials.py --json >/tmp/hermes_statsbomb_credential_check.json || {
  echo "StatsBomb credentials are not detected. No provider access attempted." >&2
  rm -f /tmp/hermes_statsbomb_credential_check.json
  exit 2
}
rm -f /tmp/hermes_statsbomb_credential_check.json

COMMON=(uv run python experiments/012_licensed_provider_backfill.py --source-root "$SOURCE_ROOT" --target-root "$TARGET_ROOT")

"${COMMON[@]}" --run-mode credentials_preflight
"${COMMON[@]}" --run-mode provider_access_preflight
"${COMMON[@]}" --run-mode plan_backfill
"${COMMON[@]}" --run-mode dry_run_backfill

if [[ "$EXECUTE" -ne 1 ]]; then
  echo "Dry-run path completed. execute_backfill was not run. Re-run with --execute after confirming gates."
else
  EXTRA=()
  [[ "$RESUME" -eq 1 ]] && EXTRA+=(--resume)
  [[ "$FORCE" -eq 1 ]] && EXTRA+=(--force)
  if [[ "$FORCE" -eq 1 ]]; then
    echo "Force mode requested. Existing valid outputs may be overwritten by Experiment 012 logic."
  fi
  "${COMMON[@]}" --run-mode execute_backfill "${EXTRA[@]}"
fi

"${COMMON[@]}" --run-mode validate_backfill
"${COMMON[@]}" --run-mode validate_all_gates

REPORT="outputs/reports/012_licensed_provider_backfill.json"
if [[ -f "$REPORT" ]] && grep -q '"target_readiness_gate_result": "PASS"' "$REPORT"; then
  echo "All gates appear ready. Next command:"
  echo "cd /home/platform/DataScienceResearch && uv run python experiments/009_full_data_reload_orchestration.py --data-root $TARGET_ROOT --run-mode rerun_research_pipeline"
else
  echo "Backfill gates are not passing yet. Do not run scoring or production pipeline."
fi
