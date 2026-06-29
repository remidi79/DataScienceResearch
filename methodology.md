# Football Score Engine Research Journal

This file is append-only. Each completed experiment records objective, hypothesis, dataset, methods, results, limitations, decision, and next steps.

## Experiment 001 — StatsBomb Data Contract and Metric Universe Inventory

Date: 2026-06-29T09:04:44.402757+00:00

### Objective

Inventory the local StatsBomb event and direct provider-stat data contracts before building any score engine. Establish available grains, metric counts, event coverage, role-family feasibility, redundancy signals, and first PCA variance signals.

### Football Hypothesis

Provider aggregate stats and raw events expose different football dimensions. Direct StatsBomb player/team stats should be treated as provider-truth aggregate facts; event data should provide contextual sequence, possession, pressure, OBV, and action-level evidence. Role-specific score engines are feasible only if positional family coverage and metric availability are explicit.

### Dataset

Source root: `/home/platform/DataPlatform/tmp/master_data_warehouse`

Rows audited:

| Dataset | Rows |
|---|---:|
| player_match_stats_direct | 350 |
| team_match_stats_direct | 20 |
| player_season_stats_direct | 489 |
| team_season_stats_direct | 16 |
| silver_events | 35082 |
| silver_lineups | 450 |
| silver_matches | 11 |

### Normalization Tested

No score normalization was selected in this inventory experiment. The experiment profiles candidates needed for later tests: per-90 provider fields, raw counts, role-family percentiles, competition percentiles, robust scaling, and empirical CDF percentiles.

### Feature Selection

No production feature selection decision was made. Initial screening computed missingness, near-zero variance, high-correlation pairs, and PCA explained variance on player-season provider metrics.

### Algorithms

- Schema and metric coverage profiling
- Event type frequency analysis
- Role-family inference from lineup positions
- Correlation screening at absolute correlation >= 0.90
- PCA on robust-scaled player-season numeric metrics

### Evaluation

Evaluation criteria were data availability, lineage clarity, role-family viability, redundancy risk, and suitability for interpretable downstream modelling.

### Results

- Metric universe: 608 unique direct provider metric names across player/team match/season grains.
- Event rows: 35082.
- Player-season rows: 489.
- High-correlation player-season metric pairs: 58.
- Near-zero variance player-season metrics: 4.
- PCA components needed for >=80% variance: 2.

### Figures

- `outputs/figures/001_row_counts.png`
- `outputs/figures/001_event_type_counts.png`
- `outputs/figures/001_metric_counts_by_grain.png`
- `outputs/figures/001_player_season_pca_variance.png`
- `outputs/figures/001_role_family_counts.png`

### Discussion

The local sample contains enough StatsBomb direct aggregate metric breadth to start score-engine research, but the current audited root is Botola-focused and not the full multi-competition/two-season universe described in the target objective. The inventory confirms that direct provider stats, raw events, lineups, and matches must be joined through explicit provider IDs and that score research should start from player-season metrics for stability, then validate with player-match and event-derived context.

### Limitations

- This experiment audits the current local DataPlatform root, not a freshly fetched full StatsBomb population.
- Role inference uses lineup position observations and does not yet apply minutes-weighted tactical role assignment.
- PCA and correlation screening are exploratory only; they do not define production score weights.
- No supervised target, expert labels, or cross-league validation was used.

### Decision

Proceed to role-specific score-family experiments. The first production-oriented score research should use player-season metrics with minimum minutes, role-family percentiles, robust scaling, redundancy pruning, and interpretable linear/PCA/factor baselines before tree models.

### Production Recommendation

Build a reusable score-engine pipeline with separate stages for data contract validation, role assignment, normalization comparison, redundancy removal, interpretable factor discovery, model fitting, percentile calibration, explainability, and model-card export.

### Next Steps

1. Experiment 002: role-family and minutes-threshold validation.
2. Experiment 003: normalization comparison for player-season metrics.
3. Experiment 004: possession/progression score family for midfielders/full backs/center backs.
4. Experiment 005: attacking score family for CF/wingers/AM.
5. Experiment 006: goalkeeper score family using GK direct stats and event evidence.

## Experiment 002 — Role Eligibility & Stable Metric Population

Date: 2026-06-29T09:21:33.526569+00:00

### Objective

Define the role-specific eligible player populations and stable metric universe that will gate every future score engine.

### Football Hypothesis

Composite scores become unstable and misleading when low-minute players, hybrid role players, or sparse/noisy metrics are allowed to define coefficients. Role families require different minimum-minute thresholds and different metric eligibility screens.

### Dataset

Source root: `/home/platform/DataPlatform/tmp/master_data_warehouse`

Input rows:

| Dataset | Rows |
|---|---:|
| player season stats direct | 489 |
| player match stats direct | 350 |
| silver lineups | 450 |

### Normalization Tested

No final normalization was selected. This experiment prepares Experiment 003 by defining eligible role populations and stable raw metric candidates. Later normalization tests should compare per-90, robust z-score, rank, quantile, empirical CDF, role percentile, and Bayesian shrinkage on this filtered universe.

### Feature Selection

Initial exclusion gates:

- metric coverage < 40% within eligible role population
- near-zero variance
- membership in a high-correlation pair with abs(correlation) >= 0.90
- unreasonable coefficient of variation > 3.0
- split-half reliability < 0.6 when measurable

### Algorithms

- Minutes-weighted role assignment from lineup position intervals
- 70% dominant-role rule; otherwise MULTI_ROLE
- Role-specific threshold eligibility: GK 600, CB/FB/MID 900, WINGER/CF 750 minutes
- Metric coverage/variance/CV screening
- Bootstrap mean confidence intervals
- Split-half reliability from player-match metric splits
- Correlation-exclusion screen using Experiment 001 high-correlation table when available, otherwise recalculated

### Evaluation

| Role | Threshold minutes | Assigned players | Eligible players | Stable metrics |
|---|---:|---:|---:|---:|
| GK | 600.0 | 19 | 12 | 79 |
| CB | 900.0 | 44 | 25 | 101 |
| FB | 900.0 | 45 | 23 | 101 |
| MID | 900.0 | 77 | 34 | 97 |
| WINGER | 750.0 | 57 | 25 | 99 |
| CF | 750.0 | 37 | 22 | 97 |

### Results

- Assigned role players: 279
- MULTI_ROLE players: 15
- UNKNOWN players: 97
- Candidate metrics requested: 72
- Candidate metrics ready: 44
- Candidate metrics blocked: 25
- Candidate metrics missing from provider stats: 3

### Figures

- `outputs/figures/002_role_assignment_counts.png`
- `outputs/figures/002_role_eligible_players.png`
- `outputs/figures/002_stable_metrics_by_role.png`
- `outputs/figures/002_candidate_metric_status.png`

### Discussion

The experiment implements the requested population-first research order. It does not search final coefficients. It explicitly excludes MULTI_ROLE players from initial coefficient modelling and records which requested football metrics are currently available, stable, blocked, or missing under the local DataPlatform root.

### Limitations

- Current local data root is not yet the full multi-competition/two-season StatsBomb universe.
- Season-to-season correlation cannot be measured from the current single-season local sample.
- Split-half reliability is limited by small player-match counts in the local root.
- Role-minute resolution uses lineup intervals and player-match minutes where available; provider lineup interval quality should be rechecked on the full dataset.

### Decision

Use this role eligibility and metric stability output as a mandatory gate for Experiment 003 normalization research. Do not fit production score coefficients until normalization and stability are validated on the full intended population.

### Production Recommendation

Encode role assignment, minimum-minute thresholds, UNKNOWN/MULTI_ROLE exclusion, metric coverage gates, high-correlation pruning, and stability diagnostics into the production score-engine preflight step.

### Next Steps

1. Experiment 003: normalization comparison on the stable role-specific metric universe.
2. Experiment 004: Defensive Contribution and Ball Progression score-family baselines.
3. Experiment 005: Chance Creation and Finishing score-family baselines.
4. Re-run Experiment 002 once the full two-season/all-competition StatsBomb dataset is loaded.

## Experiment 003 — Scientific Feature Engineering & Normalization

Date: 2026-06-29T09:35:35.894733+00:00

### Objective

Create the scientific role-specific feature layer that all future score engines will use. This experiment normalizes READY metrics from Experiment 002, profiles distributions, builds benchmark cutoffs, detects redundancy, and discovers latent dimensions. It does not compute final score weights.

### Football Hypothesis

A defensible score engine needs stable role-specific feature transformations before modelling. The same raw metric can require a different normalization method depending on role population and distribution shape.

### Dataset

Source root: `/home/platform/DataPlatform/tmp/master_data_warehouse`

Inputs:

- `marts_v2/mart_statsbomb_player_season_stats_direct_v1.jsonl`
- `outputs/tables/002_role_resolution.csv`
- `outputs/tables/002_candidate_metric_status.csv`

### Normalization Tested

- Z-score
- Robust Z-score
- Percentile Rank
- Min-Max
- Log transform
- Quantile transform
- Winsorized Z-score

Selection criterion: transformed distribution stability score = abs(skewness) + 0.5 * abs(excess kurtosis) + 0.05 * outlier percentage.

### Feature Selection

Only `candidate_ready` metrics from Experiment 002 are used. Additional redundancy pairs are identified from Pearson/Spearman correlations >= 0.9; removals are recommendations for Experiment 004, not destructive changes to source data.

### Algorithms

- Distribution profiling: mean, median, std, MAD, IQR, CV, percentiles, missing %, outlier %, skewness, kurtosis, Shapiro test
- Normalization comparison across seven methods
- Pearson and Spearman correlation
- Hierarchical clustering on absolute Spearman distance
- PCA cumulative variance and loadings
- FactorAnalysis as a secondary latent-dimension check
- Mutual information against unsupervised PC1 for weight-preparation candidates

### Evaluation

| Role | Eligible players | READY metrics | Normalization decisions | Redundant pairs | Latent dimensions |
|---|---:|---:|---:|---:|---:|
| GK | 12 | 5 | 5 | 0 | 3 |
| CB | 25 | 9 | 9 | 0 | 9 |
| FB | 23 | 8 | 8 | 0 | 7 |
| MID | 34 | 8 | 8 | 0 | 7 |
| WINGER | 25 | 7 | 7 | 0 | 6 |
| CF | 22 | 7 | 7 | 0 | 4 |

### Results

- Total metric statistics rows: 44
- Normalization method evaluations: 308
- Normalization decisions: 44
- Benchmark rows: 220
- Redundancy pairs: 0
- Latent dimension rows: 44
- Weight-preparation rows: 44
- Figures generated: 42

### Figures

Per-role figures are written under `outputs/figures/003_<ROLE>_*`: distributions, QQ plots, correlation heatmaps, PCA variance, cluster dendrograms, boxplots, and benchmark distributions.

### Discussion

The experiment produces a reproducible feature-engineering layer without hardcoded football weights. Dimension labels remain data-driven clusters, with metric membership explained by hierarchical clustering and PCA/factor evidence. Football interpretation should happen in Experiment 004 after reviewing these empirical clusters.

### Limitations

- Current local root remains a limited sample rather than the full multi-competition/two-season target population.
- Shapiro, multimodality, and split-dimensional diagnostics are sample-size sensitive.
- Quantile normalization can over-stabilize very small samples; decisions must be rerun on the full population.
- Mutual information is unsupervised against PC1, not a target-based importance score.

### Decision

Use `003_normalization_decisions.csv`, `003_role_benchmarks.csv`, `003_feature_redundancy.csv`, and `003_latent_dimensions.csv` as the feature-layer contract for Experiment 004. Do not compute final score weights yet.

### Production Recommendation

Future production score engines should load role-specific normalization decisions and benchmark cutoffs from versioned artifacts, then recompute them whenever population coverage changes materially.

### Next Steps

1. Review latent clusters by role and assign football-readable labels only where the data supports them.
2. Experiment 004: build first interpretable score-family baselines using the Experiment 003 feature layer.
3. Re-run Experiment 003 on the full StatsBomb multi-competition/two-season dataset before production coefficients.

## Experiment 004 — Role-Specific Weight Estimation & Prototype Score Formula

Date: 2026-06-29T09:57:13.186613+00:00

### Objective

Estimate role-specific metric and dimension weights in a reproducible, explainable way and produce prototype scoring artefacts for Experiment 005 validation. The scores are not final production scores.

### Football Hypothesis

Role-specific score prototypes should combine data-driven metric evidence within latent dimensions and dimension evidence within each role, while shrinking unstable estimates toward equal weights under small samples or method disagreement.

### Dataset

Source root: `/home/platform/DataPlatform/tmp/master_data_warehouse` plus Experiment 002 eligibility and Experiment 003 feature-layer tables.

### Normalization Used

Experiment 003 selected normalization methods were applied per role/metric. Lower-is-better metrics are inverted only when the direction registry classifies them safely; manual-review metrics are excluded from scoring.

### Feature Selection

Only metrics that were `candidate_ready`, had Experiment 003 normalization decisions, and belonged to an Experiment 003 latent dimension were eligible. MULTI_ROLE and UNKNOWN players were excluded from coefficient fitting.

### Algorithms

Equal weights, PCA loading weights, variance contribution weights, stability-adjusted weights, entropy weights, bootstrap stability weights, shrinkage-to-equal ensemble weights, metric caps, dimension evidence blending, bootstrap score uncertainty, and rank-sensitivity analysis.

### Evaluation

| Role | Players scored | Metrics used | Dimensions | Unstable weights |
|---|---:|---:|---:|---:|
| GK | 12 | 5 | 3 | 5 |
| CB | 25 | 9 | 9 | 9 |
| FB | 23 | 8 | 7 | 8 |
| MID | 34 | 7 | 6 | 7 |
| WINGER | 25 | 7 | 6 | 7 |
| CF | 22 | 7 | 4 | 7 |

### Results

- Normalized feature rows: 1094
- Metric weight decisions: 43
- Dimension weight decisions: 35
- Prototype dimension scores: 864
- Prototype role scores: 141
- Quality flags: 833
- Figures generated: 54

### Figures

Per-role figures are written under `outputs/figures/004_<ROLE>_*`: metric weights, dimension weights, score distributions, ranking sensitivity, bootstrap uncertainty, top-20 table, score-vs-minutes, and method rank-correlation heatmap.

### Discussion

The output is a transparent prototype scoring layer with traceable data evidence, shrinkage, caps, and warning flags. It intentionally avoids final football claims.

### Limitations

The local data root is a limited sample, not the full multi-competition/two-season target population. Several weights are unstable because of small samples, single-metric dimensions, or method disagreement. Manual-review metric directions were excluded from prototype scores.

### Decision

Use Experiment 004 artefacts only as prototype inputs for Experiment 005 validation, calibration, and comparison. Do not use these scores in production.

### Production Recommendation

Production score deployment requires Experiment 005 validation and a rerun on the full intended StatsBomb population, with football review of direction registry and latent dimension names.

### Next Steps

Experiment 005 should validate prototype score stability, calibration, cross-role/cross-league robustness, sensitivity, and benchmark fairness before any production score engine is declared.

## Experiment 005 — Scientific Validation & Calibration of the Football Score Engine

Date: 2026-06-29T10:21:11.082854+00:00

### Objective
Validate whether prototype scores from Experiment 004 are scientifically reliable, statistically robust, football-reviewable, and suitable for later production candidacy. No new production coefficients are declared.

### Scientific Questions
Can scores be trusted, are rankings stable, are dimensions independent, are weights robust, are confidence intervals acceptable, and what requires football expert review?

### Validation Methods
Bootstrap score simulation, rank stability, sensitivity analysis, split-half reliability, ICC approximation, Cronbach alpha, jackknife/Monte Carlo proxies, score independence, mutual information, PCA, clustering, confidence index, and readiness classification.

### Reliability Results
Reliability rows: 6.

### Bootstrap Results
Every prototype role score has bootstrap mean, standard deviation, variance, and confidence interval. Bootstrap rows: 141.

### Sensitivity Results
Sensitivity rows: 2205.

### Score Independence
Correlation rows: 595; redundancy rows: 0.

### Football Review
Football review candidates and explainability contribution rows are generated for expert review, not automatic player judgement.

### Confidence Index
Every score has confidence index, confidence interval, weight stability, bootstrap stability, sample quality, minutes reliability, population size, and data quality fields.

### Production Readiness
All roles remain Prototype because football validation, cross-season validation, and cross-league validation are pending on the full intended dataset.

### Limitations
Local sample only; match-level bootstrap is approximated from available score/dimension/metric artefacts because match-grain score histories are not yet materialized; production deployment is not justified.

### Recommendations
Run full-population validation, materialize match-level historical scores, complete football expert review, then revisit production candidacy.

### Next Experiment
Experiment 006 should focus on full-population rerun or match-level temporal validation before any production deployment decision.

## Experiment 006 — Temporal, Match-Level & Cross-Competition Validation

Date: 2026-06-29T10:47:38.971612+00:00

### Objective
Materialize match-level and season-split score histories and validate temporal, team, season, competition, threshold, calibration, and production-candidate stability. No production coefficients are declared.

### Football Hypothesis
A defensible role-specific score engine must remain stable across match samples, season splits, competitions, teams, and minutes thresholds before production use.

### Dataset
Data root: `/home/platform/DataPlatform/tmp/master_data_warehouse` plus Experiment 002–005 artefacts.

### Normalization Used
Experiment 003 normalization decisions are reused for match-level metric normalization; Experiment 004 weights are treated as fixed prototypes.

### Feature Selection
Only Experiment 004 prototype metrics and dimensions are validated. Missing match-level mappings are explicitly flagged.

### Algorithms
Match-level materialization, rolling windows, split-half temporal comparison, leave-one-season/competition-out, KS/Wasserstein drift, team-context correlations, threshold sensitivity, calibration bands, and readiness gate rules.

### Evaluation
Rows generated: match history 169, season split 423, rolling 0, temporal 147.

### Results
All roles remain research/validation scores. Production Candidate is not declared because full multi-season/multi-competition evidence and expert review are insufficient.

### Figures
Generated 70 figures under `outputs/figures/006_*`.

### Discussion
Experiment 006 adds temporal and context validation around the prototype engine, but the local sample limits cross-season and cross-competition conclusions.

### Limitations
Local dataset scope is limited; many roles lack enough seasons/competitions; match-level metric mapping is partial; team-context correction is only diagnosed, not applied.

### Decision
Keep scores as research/validation prototypes.

### Production Recommendation
Do not deploy final coefficients. Load full target data and complete expert review before production-candidate promotion.

### Next Steps
Experiment 007 should run full-population validation, materialize complete match histories, and integrate expert review decisions.

## Experiment 007 — Production Score Engine, Explainability & Confidence Framework

Date: 2026-06-29T10:59:09.183822+00:00

### Objective
Transform research outputs into a production Score Engine architecture, explainability contract, confidence framework, versioning strategy, configuration framework, readiness dashboard, research-gap roadmap, and full-population recalibration design. No new coefficients are introduced.

### Production Architecture
Implemented reusable production-facing dataclasses and a configuration-driven `ScoreEngine` skeleton in `src/football_score_engine_research/production_engine.py`.

### Explainability Framework
Defined metric, dimension, and overall role explanation model with positive/negative contributors and review flags.

### Confidence Framework
Defined confidence formula combining bootstrap stability, weight stability, minutes reliability, population reliability, metric coverage, data quality, and validation status.

### Score Versioning
Defined 0.1 Prototype, 0.5 Research Validated, 0.9 Full Population Validated, and 1.0 Production Ready stages. Current stage: `0.5.0-research-validated`.

### Configuration Strategy
Generated configuration-driven score definitions from Experiment 003–006 artefacts. Metrics, dimensions, directions, normalization, weights, eligibility, percentiles, and confidence rules are config objects.

### Production Readiness
Readiness dashboard rows: 6. All roles remain research/validation status pending full-population recalibration and expert review.

### Remaining Research Gaps
Research gap rows: 56. Main gaps: additional seasons, more competitions, complete expert review, match-level metric mapping, and full-population recalibration.

### Next Experiment
Experiment 008 should run the full-population recalibration pipeline once the complete StatsBomb dataset is available, then produce candidate coefficient bundles for gate review.

## Experiment 008 — Full-Population Recalibration & Production-Candidate Bundle

Date: 2026-06-29T11:22:05.941574+00:00

### Objective
Audit full-population readiness and run the production-candidate recalibration gate using the Experiment 007 architecture. Stop before candidate bundle creation if the available population is insufficient.

### Football Hypothesis
A production-candidate score engine requires enough seasons, competitions, matches, eligible players, stable metrics, temporal evidence, cross-competition validation, expert review, and traceable coefficients.

### Dataset
Data root: `/home/platform/DataPlatform/tmp/master_data_warehouse`. Full-population available: False.

### Normalization Used
Full normalization recalibration was blocked because the data-readiness gate failed. Previous normalization artefacts were archived into 008 tables with explicit blocked status.

### Feature Selection
Full metric stability recalculation was blocked by population readiness. Candidate/rejection tables document that no production candidates were promoted.

### Algorithms
Data readiness audit, role eligibility summary, production gate checks, blocked-output archival, production-candidate bundle guard, and validation/report generation.

### Evaluation
Inventory rows: 7; data gate rows: 10; production gate rows: 90.

### Results
No production-candidate bundle was generated. All roles remain Research Prototype due to insufficient population coverage.

### Figures
Generated 65 figures under `outputs/figures/008_*`.

### Discussion
Experiment 008 correctly prevents false production promotion when the local DataPlatform root is still limited.

### Limitations
The current root has only the available local sample, not the full multi-season/multi-competition StatsBomb target population.

### Decision
Do not create or deploy a production-candidate coefficient bundle.

### Production Recommendation
Load the full target StatsBomb population, rerun Experiment 008, then generate a signed candidate bundle only for roles that pass strict gates.

### Next Steps
Experiment 009 should be the full data ingestion / reload and rerun orchestration needed before repeating Experiment 008 on complete data.

## Experiment 009 — Full DataPlatform Reload & End-to-End Recalibration Orchestration

Date: 2026-06-29T11:40:03.404364+00:00

### Objective
Create a reproducible DataPlatform reload and end-to-end rerun orchestration layer before any production-candidate bundle can be generated.

### Football Hypothesis
Production score recalibration requires a complete population across seasons, competitions, matches, events, lineups, and role-eligible players before statistical/football validation can be trusted.

### Dataset
Data root: `/home/platform/DataPlatform/tmp/master_data_warehouse`. Current coverage: {'competitions': 2, 'seasons': 2, 'matches': 11, 'teams': 18, 'players': 473, 'player_match_rows': 350, 'player_season_rows': 489, 'events': 35082, 'lineups': 450}.

### Normalization Used
No score normalization is recalculated in Experiment 009. The orchestration only controls whether Experiments 001–008 may rerun.

### Feature Selection
No new feature selection is performed. Required datasets and key fields are specified in the full-population data contract.

### Algorithms
Root discovery, full data contract validation, gap analysis, ID consistency audit, reload planning, readiness gate, safe rerun orchestration, and timestamped run manifest generation.

### Evaluation
Data readiness gate result: FAIL. Rerun pipeline executed: False.

### Results
The current root remains below production target, so the rerun pipeline is blocked and no production coefficients or bundles are produced.

### Figures
Generated 7 orchestration figures.

### Discussion
Experiment 009 prevents premature API integration or production-candidate generation before full DataPlatform reload is complete.

### Limitations
It does not fetch or scrape data. It defines and validates the reload orchestration layer only.

### Decision
Do not rerun Experiments 001–008 until the data readiness gate passes.

### Production Recommendation
Complete DataPlatform reload tasks, validate the loaded root, then run `rerun_research_pipeline`.

### Next Steps
Experiment 010 should execute the actual DataPlatform reload or integration workflow, then return to Experiment 009 validation.

## Experiment 010 — DataPlatform StatsBomb Coverage Expansion & Reload Execution

Date: 2026-06-29T12:05:21.583836+00:00

### Objective
Build and validate the DataPlatform StatsBomb reload execution workflow required to satisfy the Experiment 009 production data contract.

### Football Hypothesis
The score engine cannot become production-candidate until DataPlatform contains enough matches, events, lineups, competitions, seasons, and provider stats to support role-specific validation.

### Dataset
Source root: `/home/platform/DataPlatform`. Target root: `/home/platform/DataPlatform/tmp/master_data_warehouse_full`.

### Normalization Used
None. This is data reload orchestration, not scoring.

### Feature Selection
None. Required source-to-target mapping is defined for DataPlatform datasets.

### Algorithms
Source discovery, source-to-target mapping, dry-run projection, optional copy-based materialization, ID consistency checks, target coverage gate, and Experiment 009 compatibility invocation.

### Evaluation
Target readiness result: FAIL. Experiment 009 validation status: not_run_in_mode.

### Results
Reload workflow artefacts were generated. Production is still not declared.

### Figures
Generated 7 figures.

### Discussion
Experiment 010 prepares data reload execution but does not rerun scoring or produce coefficients.

### Limitations
If no larger raw/bronze source is present, execute/dry-run cannot reach production targets.

### Decision
Do not run API integration or production scoring until target root passes Experiment 009.

### Production Recommendation
Complete missing DataPlatform source ingestion, rerun Experiment 010 execute/validate, then use Experiment 009 rerun pipeline only after gate passes.

### Next Steps
Experiment 011 should perform provider/API ingestion or upstream DataPlatform loading for missing competitions/seasons if sources are unavailable locally.


## Experiment 011 — Provider/API-backed StatsBomb Ingestion & Coverage Expansion

Date: 2026-06-29T12:36:49.254040+00:00

### Objective
Create provider/API-backed StatsBomb ingestion controls for the full DataPlatform target root.

### Football Hypothesis
Production-candidate score validation requires licensed provider coverage across multiple competitions and seasons; local partial Botola data is insufficient.

### Dataset
Source root: `/home/platform/DataPlatform`. Target root: `/home/platform/DataPlatform/tmp/master_data_warehouse_full`.

### Normalization Used
None. This experiment ingests and validates data only.

### Feature Selection
None. Dataset selection is based on provider coverage and production data contract requirements.

### Algorithms
Provider access discovery, credential-presence detection, cached/API metadata discovery, coverage planning, dry-run ingestion, schema validation, direct-stat status classification, target readiness gates, and Experiment 010/009 compatibility checks.

### Evaluation
Target readiness: FAIL. Credentials status: missing.

### Results
Provider ingestion workflow generated required artefacts. No production score coefficients were changed.

### Figures
Generated 9 figures.

### Discussion
Execution remains blocked unless licensed StatsBomb credentials and enough provider coverage are available.

### Limitations
No fake data or unauthorized scraping is used. Direct provider stats remain blocked when credentials/endpoints are unavailable.

### Decision
Do not declare production readiness or rerun scoring pipeline until ingestion gates pass.

### Production Recommendation
Load licensed provider data, validate with Experiments 010 and 009, then explicitly request the full research rerun.

### Next Steps
Experiment 012 should execute licensed provider ingestion/backfill once credentials and coverage are confirmed.


## Experiment 012 — Licensed Provider Backfill Execution & Gate Validation

Date: 2026-06-29T13:42:39.591043+00:00

### Objective
Execute and validate licensed-provider backfill controls for the full target warehouse.

### Football Hypothesis
Production score research remains blocked until licensed StatsBomb coverage satisfies multi-competition and multi-season data gates.

### Dataset
Source root: `/home/platform/DataPlatform`. Target root: `/home/platform/DataPlatform/tmp/master_data_warehouse_full`.

### Normalization Used
None. This is provider data ingestion/backfill validation only.

### Feature Selection
None. Coverage selection uses provider metadata and production contract thresholds.

### Algorithms
Credential preflight, provider access preflight, metadata coverage planning, dry-run projection, guarded execution, schema validation, ID validation, and Experiment 011/010/009 gate chain.

### Evaluation
Target readiness: FAIL.

### Results
Backfill workflow artefacts generated. No score coefficients or production bundles were created.

### Figures
Generated 10 figures.

### Discussion
Backfill execution is guarded by credentials, provider access, and coverage gates.

### Limitations
No fake data is generated; insufficient provider coverage stops the workflow.

### Decision
Do not declare production readiness.

### Production Recommendation
Run full research rerun only after all gates pass.

### Next Steps
Experiment 013 should be the explicit full research rerun after a passing warehouse gate.


## Experiment 013 — Secure Credentials Activation & Licensed Backfill Runbook

Date: 2026-06-29T14:03:20.303436+00:00

### Objective
Prepare secure credential activation and licensed backfill execution workflow.

### Football Hypothesis
No production score research can proceed until licensed provider credentials are activated and warehouse gates pass.

### Dataset
No scoring dataset is created; target remains `/home/platform/DataPlatform/tmp/master_data_warehouse_full`.

### Normalization Used
None.

### Feature Selection
None.

### Algorithms
Secret handling audit, safe env template, credential detection helper, runbook, dry-run-first execution script, and blocked-state validation.

### Evaluation
Credential status: missing. Backfill executed: false.

### Results
Operational artefacts were generated without secrets, fake data, coefficients, production bundles, or API integration.

### Figures
Generated four operational figures.

### Discussion
Credentials must be activated outside git before licensed backfill can run.

### Limitations
No provider fetch is attempted without credentials.

### Decision
Keep production and scoring blocked.

### Production Recommendation
Run safe backfill only after credentials are detected and dry-run passes.

### Next Steps
Experiment 014 should execute the gate-approved research rerun only after the warehouse passes.

## Experiment 014 — Event-Derived / Open-Data Fallback Feasibility

### Objective
Evaluate whether local event and lineup data can support a separate event-derived research fallback.

### Football Hypothesis
Some player/team actions can be measured from events with clear lineage, but provider-direct aggregate metrics and tracking-dependent dimensions cannot be reproduced safely without licensed provider data.

### Dataset
Local approved `/home/platform/DataPlatform/tmp/master_data_warehouse` silver events, lineups, matches, and existing provider-direct marts for comparison only.

### Normalization Used
No score normalization or coefficient fitting. Player-season diagnostic percentiles are role-local research indicators only.

### Feature Selection
Only safely or partially derivable event metrics with explicit required event types and fields.

### Algorithms
Event grouping, formula-based aggregation, coverage auditing, role-dimension support classification, and provider comparison correlations where comparable data exists.

### Evaluation
Field coverage, lineage completeness, sample size, role support, provider-comparison diagnostics, and production-blocking checks.

### Results
Event-derived player-match, team-match, and player-season research tables were generated with explicit non-provider-direct lineage. No production score was computed.

### Figures
See `outputs/figures/014_*.png`.

### Discussion
The fallback layer can support research diagnostics for some roles/dimensions, but it cannot replace licensed StatsBomb provider-direct stats.

### Limitations
Limited sample, field-dependent formulas, provider definition mismatch, no tracking/360/video replacement.

### Decision
Proceed only as research fallback feasibility. Do not use as production data.

### Production Recommendation
Do not declare production readiness. Continue licensed provider backfill path for production-grade warehouse completeness.

### Next Steps
Experiment 015, if requested, should prototype a separate research-only event-derived score layer or return to licensed backfill after credentials are active.

## Experiment 015 — Event-Derived Research Scouting Score Prototype

### Objective
Create a research-only event-derived player scouting score prototype from Experiment 014 outputs.

### Football Hypothesis
A transparent role-specific prototype can summarize available event-derived dimensions for exploratory scouting, but the result is not provider-direct and is not production-ready.

### Dataset
Experiment 014 event-derived player-match, player-season, metric catalog, role support, and feasibility tables plus local warehouse context from `/home/platform/DataPlatform/tmp/master_data_warehouse`.

### Normalization Used
Role-specific percentiles, z-scores, min-max 0-100, and confidence-adjusted values. GK is never normalized against CF or other out-of-role populations.

### Feature Selection
Only safely derivable and partially derivable event metrics with usable player-season columns are included. Provider-direct, tracking, video, and unavailable metrics are excluded.

### Algorithms
Equal-weight dimension scoring, confidence-adjusted dimension averaging, role-local research scouting score, same-role cosine similarity with Euclidean fallback, and row-level explainability.

### Evaluation
Validation checks enforce 0-100 bounds, role-local normalization, research-only labels, production_ready=false, low-sample flags, no fake data, no provider-direct replacement, no production coefficients, no production bundle, and no Experiment 016.

### Results
Research-only scouting score, dimension score, normalized metric, similarity, explanation, validation, report, notebook, and figures were generated.

### Discussion
The prototype is useful for methodology review and product discussion only. It is blocked from production by sample size, missing licensed provider access, and absent production validation.

### Limitations
Only 11 local matches, 1 competition, 1 season, low sample sizes for every role, event-derived formulas only, and no licensed provider-direct replacement.

### Decision
Keep as research-only prototype. Do not expose as production score.

### Production Recommendation
Do not ship. Resume licensed provider backfill and full-population validation before any production score work.

### Next Steps
Review feature/dimension definitions and, only after explicit approval, either improve the research prototype with more data or return to licensed provider-direct ingestion.

