# Player Scouting Outputs Index

This index closes the player scouting / score engine research workstream. It is an artefact index only; it does not create Experiment 016 and does not expose Experiment 015 as production.

| Experiment | Status | Production ready | Research only | Script | Notebook | Limitations |
|---|---|---:|---:|---|---|---|
| 001 — StatsBomb Data Contract and Metric Universe Inventory | completed_research_inventory | false | true | `001_data_contract_inventory.py` | `001_data_contract_inventory.ipynb` | Local sample only; inventory does not validate production scoring. |
| 002 — Role Eligibility & Stable Metric Population | completed_research_role_eligibility | false | true | `002_role_eligibility_stable_metric_population.py` | `002_role_eligibility_stable_metric_population.ipynb` | Role eligibility constrained by local lineup/minutes coverage and sample size. |
| 003 — Scientific Feature Engineering & Normalization | completed_research_normalization | false | true | `003_feature_engineering_normalization.py` | `003_feature_engineering_normalization.ipynb` | Normalization decisions are research decisions on a limited sample. |
| 004 — Role-Specific Weight Estimation & Prototype Score Formula | completed_research_prototype_only | false | true | `004_role_specific_weight_estimation.py` | `004_role_specific_weight_estimation.ipynb` | Prototype weights only; small-sample shrinkage and instability flags remain. |
| 005 — Scientific Validation & Calibration | completed_research_validation_not_production | false | true | `005_scientific_validation_calibration.py` | `005_scientific_validation_calibration.ipynb` | Validation remains research-only; expert review and full-population validation pending. |
| 006 — Temporal, Match-Level & Cross-Competition Validation | completed_validation_gates_blocked_by_sample | false | true | `006_temporal_cross_competition_validation.py` | `006_temporal_cross_competition_validation.ipynb` | Only one competition/season in local root; temporal/cross-competition gates cannot be satisfied. |
| 007 — Production Score Engine Framework | framework_ready_for_full_population_recalibration_not_deployment | false | true | `007_production_score_engine_framework.py` | `007_production_score_engine_framework.ipynb` | Architecture/config framework exists but deployment requires full-population recalibration and gates. |
| 008 — Full-Population Recalibration Gate | blocked_no_full_population_no_bundle | false | true | `008_full_population_recalibration.py` | `008_full_population_recalibration.ipynb` | Full population unavailable; no production candidate bundle generated. |
| 009 — Full DataPlatform Reload Orchestration | blocked_target_full_root_not_ready | false | false | `009_full_data_reload_orchestration.py` | `009_full_data_reload_orchestration.ipynb` | Full target warehouse incomplete/empty; rerun pipeline blocked. |
| 010 — DataPlatform StatsBomb Coverage Reload | blocked_incomplete_target_source_mapping | false | false | `010_dataplatform_statsbomb_reload.py` | `010_dataplatform_statsbomb_reload.ipynb` | DataPlatform source mapping insufficient for full target root. |
| 011 — Provider/API-backed StatsBomb Ingestion | blocked_missing_credentials_no_ingestion_execute | false | false | `011_provider_statsbomb_ingestion.py` | `011_provider_statsbomb_ingestion.ipynb` | Licensed provider credentials missing; execute ingestion not run. |
| 012 — Licensed Provider Backfill | blocked_missing_credentials_no_backfill_execute | false | false | `012_licensed_provider_backfill.py` | `012_licensed_provider_backfill.ipynb` | Licensed provider credentials/provider access missing or blocked; execute backfill not run. |
| 013 — Secure Credentials Activation | blocked_credentials_not_detected | false | false | `scripts/check_statsbomb_credentials.py; scripts/run_licensed_backfill_safe.sh` | `013_secure_credentials_activation.ipynb` | Credential activation still blocked in runtime; helper reports missing supported credential method. |
| 014 — Event-Derived Fallback Feasibility | completed_research_only_event_fallback_feasibility | false | true | `014_event_derived_fallback_feasibility.py` | `014_event_derived_fallback_feasibility.ipynb` | Research fallback only; event-derived metrics cannot replace provider-direct stats; 11-match sample. |
| 015 — Event-Derived Research Scouting Score Prototype | completed_research_only_event_derived_prototype | false | true | `015_event_derived_research_scouting_score.py` | `015_event_derived_research_scouting_score.ipynb` | Research-only event-derived prototype; not provider-direct; not production-ready; low-sample flagged; no production score claim. |

## Output locations

- Reports: `outputs/reports/`
- Tables: `outputs/tables/`
- Figures: `outputs/figures/`
- Notebooks: `notebooks/`
