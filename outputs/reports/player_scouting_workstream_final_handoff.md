# Player Scouting Workstream Final Handoff

Generated: 2026-06-29T18:54:47Z

## 1. Executive summary

Experiments 001–015 are completed for the player scouting / score engine research workstream. The workstream is closed as research, not production. The current engine artefacts provide a research-validated framework, score contracts, validation scaffolding, and event-derived research fallback prototype, but production scoring remains blocked.

Key status:
- Experiment 015 is research-only.
- Experiment 015 is event-derived only.
- Experiment 015 is not provider-direct.
- Experiment 015 is not production-ready.
- Experiment 015 must not replace licensed provider-direct scouting scores.
- Current local sample has only 11 matches, 1 competition, and 1 season.
- No role reaches production-grade sample size.
- Similarity is same-role only and low-sample flagged.
- Production scoring requires licensed provider backfill and full-population validation.

## 2. Timeline of Experiments 001–015

| Experiment | Title | Status |
|---|---|---|
| 001 | StatsBomb Data Contract and Metric Universe Inventory | completed_research_inventory |
| 002 | Role Eligibility & Stable Metric Population | completed_research_role_eligibility |
| 003 | Scientific Feature Engineering & Normalization | completed_research_normalization |
| 004 | Role-Specific Weight Estimation & Prototype Score Formula | completed_research_prototype_only |
| 005 | Scientific Validation & Calibration | completed_research_validation_not_production |
| 006 | Temporal, Match-Level & Cross-Competition Validation | completed_validation_gates_blocked_by_sample |
| 007 | Production Score Engine Framework | framework_ready_for_full_population_recalibration_not_deployment |
| 008 | Full-Population Recalibration Gate | blocked_no_full_population_no_bundle |
| 009 | Full DataPlatform Reload Orchestration | blocked_target_full_root_not_ready |
| 010 | DataPlatform StatsBomb Coverage Reload | blocked_incomplete_target_source_mapping |
| 011 | Provider/API-backed StatsBomb Ingestion | blocked_missing_credentials_no_ingestion_execute |
| 012 | Licensed Provider Backfill | blocked_missing_credentials_no_backfill_execute |
| 013 | Secure Credentials Activation | blocked_credentials_not_detected |
| 014 | Event-Derived Fallback Feasibility | completed_research_only_event_fallback_feasibility |
| 015 | Event-Derived Research Scouting Score Prototype | completed_research_only_event_derived_prototype |

## 3. What each experiment achieved

### Experiment 001 — StatsBomb Data Contract and Metric Universe Inventory
Inventoried the available metric universe, row counts, event types, raw event fields, role-family map, missingness, correlations, near-zero variance, and PCA baseline. Local sample included 11 matches, 35,082 events, 350 player-match direct rows, and 489 player-season direct rows.
Status: `completed_research_inventory`.
Limitation: Local sample only; inventory does not validate production scoring.

### Experiment 002 — Role Eligibility & Stable Metric Population
Resolved role eligibility with lineup/minutes evidence, role thresholds, candidate metric stability, inclusion/exclusion reasons, and visible UNKNOWN/MULTI_ROLE data-quality classes. No score coefficients were fitted.
Status: `completed_research_role_eligibility`.
Limitation: Role eligibility constrained by local lineup/minutes coverage and sample size.

### Experiment 003 — Scientific Feature Engineering & Normalization
Built feature engineering and normalization research: role-specific distributions, normalization decisions, benchmark bands, redundancy checks, clustering, PCA/latent-dimension evidence, and weight-preparation signals. No final weights were created.
Status: `completed_research_normalization`.
Limitation: Normalization decisions are research decisions on a limited sample.

### Experiment 004 — Role-Specific Weight Estimation & Prototype Score Formula
Estimated prototype-only role/dimension/metric weights with transparent shrinkage and sensitivity analysis. Scores were explicitly research artefacts, not production-final coefficients.
Status: `completed_research_prototype_only`.
Limitation: Prototype weights only; small-sample shrinkage and instability flags remain.

### Experiment 005 — Scientific Validation & Calibration
Validated prototype scores with bootstrap, confidence intervals, sensitivity, rank stability, explainability, independence/redundancy checks, football review queues, and production-readiness diagnostics. Production readiness remained false.
Status: `completed_research_validation_not_production`.
Limitation: Validation remains research-only; expert review and full-population validation pending.

### Experiment 006 — Temporal, Match-Level & Cross-Competition Validation
Performed temporal/match-level/cross-competition validation scaffolding and gates. The local data could not satisfy full temporal/cross-season/cross-competition evidence requirements; production candidates were blocked.
Status: `completed_validation_gates_blocked_by_sample`.
Limitation: Only one competition/season in local root; temporal/cross-competition gates cannot be satisfied.

### Experiment 007 — Production Score Engine Framework
Converted research artefacts into a config-driven score-engine framework, schemas, explainability/confidence/versioning documents, and readiness dashboard. Status: ready for full-population recalibration, not production deployment.
Status: `framework_ready_for_full_population_recalibration_not_deployment`.
Limitation: Architecture/config framework exists but deployment requires full-population recalibration and gates.

### Experiment 008 — Full-Population Recalibration Gate
Attempted full-population recalibration gate and production-candidate bundle logic. Full population was unavailable, so no production candidate bundle was generated.
Status: `blocked_no_full_population_no_bundle`.
Limitation: Full population unavailable; no production candidate bundle generated.

### Experiment 009 — Full DataPlatform Reload Orchestration
Built full DataPlatform reload orchestration and data contract/gap analysis. Target full root validation failed and rerun pipeline was not allowed/executed.
Status: `blocked_target_full_root_not_ready`.
Limitation: Full target warehouse incomplete/empty; rerun pipeline blocked.

### Experiment 010 — DataPlatform StatsBomb Coverage Reload
Mapped DataPlatform StatsBomb source coverage into the target full warehouse. Only one dataset mapped and ten target datasets remained blocked; target readiness failed.
Status: `blocked_incomplete_target_source_mapping`.
Limitation: DataPlatform source mapping insufficient for full target root.

### Experiment 011 — Provider/API-backed StatsBomb Ingestion
Built provider/API-backed ingestion planning and validation gates. Credentials were missing; ingestion execution did not run; target readiness remained failed and no fake data was created.
Status: `blocked_missing_credentials_no_ingestion_execute`.
Limitation: Licensed provider credentials missing; execute ingestion not run.

### Experiment 012 — Licensed Provider Backfill
Built licensed-provider backfill gates and safe execution modes. Credentials/provider access remained missing or blocked; licensed backfill did not execute.
Status: `blocked_missing_credentials_no_backfill_execute`.
Limitation: Licensed provider credentials/provider access missing or blocked; execute backfill not run.

### Experiment 013 — Secure Credentials Activation
Built secure credential activation runbook/helper/wrapper and documented exact next commands. Credentials were not detected and provider access remained BLOCKED_OR_FAIL; no backfill or production bundle occurred.
Status: `blocked_credentials_not_detected`.
Limitation: Credential activation still blocked in runtime; helper reports missing supported credential method.

### Experiment 014 — Event-Derived Fallback Feasibility
Assessed event-derived fallback feasibility from local events/lineups only. It produced research-only event-derived player-match/team-match/player-season metrics and a data contract. Coverage was limited to 11 matches, one competition, one season.
Status: `completed_research_only_event_fallback_feasibility`.
Limitation: Research fallback only; event-derived metrics cannot replace provider-direct stats; 11-match sample.

### Experiment 015 — Event-Derived Research Scouting Score Prototype
Built research-only event-derived scouting score and same-role similarity prototype from Experiment 014 outputs. Outputs are not provider-direct, not production-ready, low-sample flagged, and cannot replace licensed provider-direct scouting scores.
Status: `completed_research_only_event_derived_prototype`.
Limitation: Research-only event-derived prototype; not provider-direct; not production-ready; low-sample flagged; no production score claim.

## 4. Current repository state

- Experiment scripts 001–012 and 014–015 are present under `experiments/`.
- Experiment 013 is operational/runbook-oriented and is represented by credential helper/wrapper scripts plus reports.
- Notebooks 001–015 are present under `notebooks/`.
- Reports and tables for 001–015 are present under `outputs/reports/` and `outputs/tables/`.
- Final closure outputs are this handoff, an output index, a production-readiness checklist, and a research/demo usage note.

## 5. Current engine status

The score engine is a research framework and configuration architecture. Experiment 007 made it ready for full-population recalibration, not production deployment. Experiments 008–013 show the full-population/licensed-provider path is still blocked.

## 6. Current event-derived fallback status

Experiment 014 found partial research feasibility from local events: 11 matches, 1 competition, 1 season, 35082 events, 350 player-match rows, and 295 player-season rows.

Experiment 015 converted those outputs into a research-only player scouting score prototype and same-role similarity prototype. It remains low-sample flagged and non-production.

## 7. Production readiness status

Production readiness: `NOT_READY`.

Primary reasons:
- Licensed StatsBomb credentials/provider access are missing or blocked.
- Full target warehouse is incomplete.
- Experiment 012 backfill has not executed.
- Full-population rerun has not occurred.
- Experiment 008 production-candidate gate has not passed.
- Expert review is not complete.
- No production bundle has been generated.

## 8. Research-only outputs

- Experiments 001–006: scientific research inventory, eligibility, normalization, prototype weights, validation, temporal/cross-competition validation scaffolding.
- Experiment 007: framework/contracts ready for full-population recalibration.
- Experiment 014: event-derived fallback feasibility and data contract.
- Experiment 015: event-derived research scouting score and same-role similarity prototype.

## 9. Production-blocked outputs

- Experiment 008 production-candidate bundle: blocked; not generated.
- Experiment 009 full research rerun on full warehouse: blocked.
- Experiment 010 full target reload: blocked by incomplete source mapping.
- Experiment 011 provider ingestion execution: blocked by missing credentials/access.
- Experiment 012 licensed backfill execution: blocked by missing credentials/access.
- Experiment 015 scores: blocked from production by sample size, event-derived-only lineage, and missing provider-direct validation.

## 10. Data requirements still missing

- Full target warehouse with competitions, seasons, matches, teams, players, lineups, events, player-match stats, player-season stats, team-match stats, and team-season stats.
- Sufficient multi-season and multi-competition population for every role.
- Provider-direct player/team match and season stats for licensed StatsBomb coverage.
- Full-population role/minutes eligibility evidence.
- Expert-reviewed validation examples and benchmark populations.

## 11. Credentials requirements still missing

- Credentials status: `missing`.
- Provider access status: `BLOCKED_OR_FAIL`.
- Licensed provider backfill executed: `false`.
- Credential values must remain outside git/reports and must only be verified through redacted preflight helpers.

## 12. Full-population rerun requirements

Required order:
1. Activate StatsBomb credentials in an approved shell/config path.
2. Run credential helper and Experiment 012 credentials preflight.
3. Execute licensed backfill only after access/coverage gates pass.
4. Validate Experiment 011 ingestion.
5. Validate Experiment 010 target root.
6. Validate Experiment 009 loaded root.
7. Rerun Experiments 001–008 on the full warehouse under a timestamped run folder.
8. Allow Experiment 008 to generate a production-candidate bundle only if strict gates pass.

## 13. Safety boundaries

- Do not create Experiment 016.
- Do not continue scoring experiments.
- Do not start API integration.
- Do not expose Experiment 015 as production.
- Do not call event-derived metrics provider-direct.
- Do not create fake data.
- Do not generate production bundles before gates pass.

## 14. What must not be exposed as production

- Experiment 004 prototype role scores.
- Experiment 005 validation/calibration scores.
- Experiment 006 temporal validation artefacts.
- Experiment 014 event-derived metrics.
- Experiment 015 event-derived research scouting scores.
- Experiment 015 same-role similarity rows.
- Any score from the 11-match local sample.

## 15. Recommended next actions

Production path:
1. Activate licensed StatsBomb credentials.
2. Run `python3 scripts/check_statsbomb_credentials.py --json`.
3. Run Experiment 012 `credentials_preflight`.
4. Run `scripts/run_licensed_backfill_safe.sh --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --execute --resume` only after preflight passes.
5. Validate Experiments 011, 010, and 009.
6. Rerun Experiments 001–008 on the full warehouse and require Experiment 008 production-candidate gate PASS before any API integration.

Research/demo path:
1. Use Experiment 015 only for internal methodology review, analyst feedback, and clearly-badged UI prototypes.
2. Keep all Experiment 015 outputs labelled research-only and production_ready=false.
3. Do not use Experiment 015 for recruitment ranking or automated decisions.

