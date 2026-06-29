# Football Score Engine Research

Reproducible research framework for building role-specific, interpretable football composite scores from StatsBomb events and provider aggregate stats.

The framework is designed to consume DataPlatform warehouse artifacts without recalculating provider-truth StatsBomb aggregate endpoints. Initial local source root:

`/home/platform/DataPlatform/tmp/master_data_warehouse`

## Principles

- No universal player scores. Every score is modelled by positional family.
- No manually assigned final weights without empirical validation.
- Start from football semantics, then statistics, then interpretable ML, then complex models only when justified.
- Keep provider lineage and metric provenance visible.
- Percentiles are computed within positional family by default.
- Low-minute samples require explicit minimums and shrinkage research before production use.

## Structure

```text
notebooks/          Executable research notebooks
experiments/        Reproducible experiment entrypoints
outputs/figures/    Generated figures
outputs/tables/     Generated tables
outputs/models/     Saved models and fitted artefacts
outputs/reports/    Experiment JSON/Markdown reports
src/                Reusable score-engine research code
methodology.md      Chronological research journal
```

## Reproducible experiments

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/001_data_contract_inventory.py \
  --data-root /home/platform/DataPlatform/tmp/master_data_warehouse
uv run python experiments/002_role_eligibility_stable_metric_population.py \
  --data-root /home/platform/DataPlatform/tmp/master_data_warehouse
uv run python experiments/003_feature_engineering_normalization.py \
  --data-root /home/platform/DataPlatform/tmp/master_data_warehouse
uv run python experiments/004_role_specific_weight_estimation.py \
  --data-root /home/platform/DataPlatform/tmp/master_data_warehouse
```

Experiment 001 inventories the StatsBomb data contract and metric universe. Experiment 002 defines role eligibility, UNKNOWN/MULTI_ROLE handling, and stable metric populations for later score-engine modelling. Experiment 003 builds the scientific feature layer: role-specific distribution analysis, normalization decisions, benchmarks, redundancy checks, latent dimensions, and weight-preparation signals without fitting final weights. Each experiment generates a notebook, tables, figures, report, and appends to `methodology.md`.

## Production target

For each score and role family, the final engine must emit raw score, normalized score, role percentile, competition percentile, global percentile, confidence interval, selected features, coefficients/importances, validation metrics, and explainability artefacts.


## Experiment 004

Role-specific weight estimation and prototype score formulas. Outputs metric/dimension weights, prototype dimension scores, prototype role scores, sensitivity analysis, and quality flags. These are prototype research scores only and must not be treated as final production scores.


## Experiment 005

Scientific validation and calibration of Experiment 004 prototype scores. Run:

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/005_scientific_validation_calibration.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse
```

Outputs bootstrap statistics, confidence index, reliability, sensitivity, score independence, football-review candidates, explainability, and production-readiness tables. It does not declare production coefficients.


## Experiment 006

Temporal, match-level, and cross-competition validation for prototype score engines. Run:

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/006_temporal_cross_competition_validation.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse
```

Outputs match-level histories, season splits, rolling windows, temporal stability, leave-one-season/competition-out validation, population drift, team-context sensitivity, threshold sensitivity, calibration curves, football expert review workflow, and a production-candidate gate. Scores remain research/validation scores; no final production coefficients are declared.


## Experiment 007

Production score-engine architecture, explainability contract, confidence framework, versioning strategy, configuration strategy, readiness dashboard, research gap analysis, and future full-population pipeline design. Run:

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/007_production_score_engine_framework.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse
```

This experiment does not change production coefficients. It makes the system ready for full-population recalibration once the complete multi-season StatsBomb dataset is available.


## Experiment 008

Full-population recalibration and production-candidate gate. Run:

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/008_full_population_recalibration.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse
```

The experiment first audits whether the full StatsBomb/DataPlatform population is available. If readiness fails, it stops before production-candidate bundle creation and writes `outputs/reports/008_no_production_candidate_reason.md`. A signed bundle under `outputs/production_candidate_bundle/score_engine_v0.9.0/` is created only when at least one role reaches Production Candidate or Production Ready. Scores that remain blocked are not production scores.


## Experiment 009

Full DataPlatform reload and end-to-end recalibration orchestration. Run modes:

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/009_full_data_reload_orchestration.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse --run-mode audit_only
uv run python experiments/009_full_data_reload_orchestration.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse --run-mode validate_loaded_root
uv run python experiments/009_full_data_reload_orchestration.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse --run-mode rerun_research_pipeline
```

The script audits the data root, writes the production data contract, produces missing-data gaps and reload tasks, applies a strict readiness gate, and only allows Experiments 001–008 to rerun when the gate passes. API integration and production deployment must wait until full-population recalibration passes.


## Experiment 010

DataPlatform StatsBomb coverage expansion and reload execution workflow. Run modes:

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/010_dataplatform_statsbomb_reload.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode discover_sources
uv run python experiments/010_dataplatform_statsbomb_reload.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode dry_run_reload
uv run python experiments/010_dataplatform_statsbomb_reload.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode execute_reload
uv run python experiments/010_dataplatform_statsbomb_reload.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode validate_target
```

This is a data reload workflow, not scoring production. It does not change coefficients, create production bundles, or start API integration. After validation, run Experiment 009 validate_loaded_root against the target root.


## Experiment 011

Provider/API-backed StatsBomb ingestion and coverage expansion workflow.

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/011_provider_statsbomb_ingestion.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode discover_provider_access
uv run python experiments/011_provider_statsbomb_ingestion.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode discover_provider_metadata
uv run python experiments/011_provider_statsbomb_ingestion.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode plan_coverage
uv run python experiments/011_provider_statsbomb_ingestion.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode dry_run_ingestion
uv run python experiments/011_provider_statsbomb_ingestion.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode execute_ingestion
uv run python experiments/011_provider_statsbomb_ingestion.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode validate_ingestion
```

Config example: `configs/011_provider_ingestion_config.example.json`.
Target root: `/home/platform/DataPlatform/tmp/master_data_warehouse_full`.

This is provider ingestion, not production scoring. It does not change score coefficients, generate production bundles, or start API integration.


## Experiment 012

Licensed provider backfill execution and gate validation. This is not production scoring.

Run modes: credentials_preflight, provider_access_preflight, plan_backfill, dry_run_backfill, execute_backfill, validate_backfill, validate_all_gates.

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode credentials_preflight
uv run python experiments/012_licensed_provider_backfill.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode validate_all_gates
```

Credentials are read from approved environment variables or approved DataPlatform config files; values are never printed. Target root: `/home/platform/DataPlatform/tmp/master_data_warehouse_full`.


## Experiment 013

Secure credentials activation and licensed backfill runbook.

Safety rules:
- never commit real credentials
- never print secrets
- never run production scoring before warehouse gates pass
- never generate production bundles from a blocked target root

Runbook: `outputs/reports/013_credentials_activation_runbook.md`
Helper scripts:
- `scripts/check_statsbomb_credentials.py`
- `scripts/run_licensed_backfill_safe.sh`

Credential helper runtime on this host:

```bash
python3 scripts/check_statsbomb_credentials.py --json
```

If `python3` is unavailable, use:

```bash
uv run python scripts/check_statsbomb_credentials.py --json
```

After credentials are available:

```bash
cd /home/platform/DataScienceResearch
scripts/run_licensed_backfill_safe.sh --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --execute --resume
```

If all gates pass, run the Experiment 009 rerun command documented in `outputs/reports/013_exact_next_commands.md`.

## Experiment 014

Event-derived / open-data fallback feasibility. This is a research-only fallback study and does not replace licensed StatsBomb provider-direct stats.

Run:

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/014_event_derived_fallback_feasibility.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse
```

Generated outputs include `outputs/tables/014_*`, `outputs/reports/014_event_derived_fallback_feasibility.*`, `outputs/reports/014_event_derived_data_contract.*`, `notebooks/014_event_derived_fallback_feasibility.ipynb`, and `outputs/figures/014_*.png`.

Warning: event-derived metrics are formula-based research artefacts. They are not provider-direct metrics, not licensed provider backfill, not production scoring, and not a production bundle.

## Experiment 015

Event-derived research scouting score prototype. This is research/demo only, event-derived, not provider-direct, not production-ready, and does not replace licensed StatsBomb provider-direct scouting scores.

Run:

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/015_event_derived_research_scouting_score.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse
```

Generated outputs include `outputs/tables/015_*`, `outputs/reports/015_event_derived_research_scouting_score.*`, `notebooks/015_event_derived_research_scouting_score.ipynb`, and `outputs/figures/015_*.png`.

Warning: the score is a research-only event-derived prototype. It is not a final scouting score, not a production score, not provider-direct, and not a production bundle.

