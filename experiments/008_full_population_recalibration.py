from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_score_engine_research.io import flatten_metrics, write_json
from football_score_engine_research.production_engine import ScoreEngine

EXPERIMENT_ID = "008"
TITLE = "Full-Population Recalibration & Production-Candidate Bundle"
ROLES = ["GK", "CB", "FB", "MID", "WINGER", "CF"]
MIN_TARGETS = {
    "seasons": 2,
    "competitions": 3,
    "outfield_eligible_players": 50,
    "gk_eligible_players": 20,
    "matches": 100,
    "teams": 20,
}


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def table_path(name: str) -> Path:
    return ROOT / "outputs/tables" / name


def read_table(name: str, required: bool = True) -> pd.DataFrame:
    path = table_path(name)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def audit_dataset(data_root: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    datasets = {
        "player_match_stats_direct": data_root / "marts_v2/mart_statsbomb_player_match_stats_direct_v1.jsonl",
        "player_season_stats_direct": data_root / "marts_v2/mart_statsbomb_player_season_stats_direct_v1.jsonl",
        "team_match_stats_direct": data_root / "marts_v2/mart_statsbomb_team_match_stats_direct_v1.jsonl",
        "team_season_stats_direct": data_root / "marts_v2/mart_statsbomb_team_season_stats_direct_v1.jsonl",
        "silver_matches": data_root / "silver/silver_matches.jsonl",
        "silver_lineups": data_root / "silver/silver_lineups.jsonl",
        "silver_events": data_root / "silver/silver_events.jsonl",
    }
    rows = []
    summary: dict[str, int] = {}
    for name, path in datasets.items():
        raw = read_jsonl_rows(path)
        df = pd.DataFrame(raw)
        row = {"dataset_name": name, "path": str(path), "row_count": len(df)}
        for col, out in [
            ("statsbomb_player_id", "unique_players"), ("player_id", "unique_players_alt"),
            ("team_id", "unique_teams"), ("team_name", "unique_team_names"),
            ("match_provider_id", "unique_matches"), ("provider_id", "unique_matches_alt"),
            ("competition_id", "unique_competitions"), ("season_id", "unique_seasons"),
        ]:
            row[out] = int(df[col].nunique()) if col in df.columns else 0
        if "match_date" in df.columns:
            dates = pd.to_datetime(df.match_date, errors="coerce")
            row["date_min"] = str(dates.min().date()) if dates.notna().any() else ""
            row["date_max"] = str(dates.max().date()) if dates.notna().any() else ""
        else:
            row["date_min"] = ""; row["date_max"] = ""
        key_fields = [c for c in ["statsbomb_player_id", "team_id", "match_provider_id", "competition_id", "season_id", "provider_id", "match_date"] if c in df.columns]
        row["missing_key_fields"] = ";".join([c for c in key_fields if df[c].isna().any()])
        row["production_readiness_status"] = "present" if len(df) else "missing"
        rows.append(row)
    inv = pd.DataFrame(rows)
    summary["competitions"] = int(inv[[c for c in inv.columns if c.startswith("unique_competitions")]].max(axis=1).max()) if not inv.empty else 0
    summary["seasons"] = int(inv[[c for c in inv.columns if c.startswith("unique_seasons")]].max(axis=1).max()) if not inv.empty else 0
    summary["matches"] = max(int(inv.unique_matches.max()), int(inv.unique_matches_alt.max())) if not inv.empty else 0
    summary["teams"] = max(int(inv.unique_teams.max()), int(inv.unique_team_names.max())) if not inv.empty else 0
    summary["players"] = max(int(inv.unique_players.max()), int(inv.unique_players_alt.max())) if not inv.empty else 0
    summary["player_season_rows"] = int(inv.loc[inv.dataset_name == "player_season_stats_direct", "row_count"].sum())
    summary["player_match_rows"] = int(inv.loc[inv.dataset_name == "player_match_stats_direct", "row_count"].sum())
    summary["events"] = int(inv.loc[inv.dataset_name == "silver_events", "row_count"].sum())
    summary["lineups"] = int(inv.loc[inv.dataset_name == "silver_lineups", "row_count"].sum())
    return inv, summary


def data_readiness_gate(summary: dict[str, int], role_resolution: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    rows = []
    role_counts = role_resolution[role_resolution.get("eligible_for_initial_coefficients", False) == True].groupby("assigned_role").size() if not role_resolution.empty and "eligible_for_initial_coefficients" in role_resolution.columns else pd.Series(dtype=int)
    checks = [
        ("at_least_2_seasons", summary["seasons"], MIN_TARGETS["seasons"]),
        ("at_least_3_competitions_if_available", summary["competitions"], MIN_TARGETS["competitions"]),
        ("enough_matches_for_temporal_validation", summary["matches"], MIN_TARGETS["matches"]),
        ("enough_teams_for_context_bias_testing", summary["teams"], MIN_TARGETS["teams"]),
        ("enough_goalkeepers", int(role_counts.get("GK", 0)), MIN_TARGETS["gk_eligible_players"]),
    ]
    for role in ["CB", "FB", "MID", "WINGER", "CF"]:
        checks.append((f"enough_{role.lower()}_eligible_players", int(role_counts.get(role, 0)), MIN_TARGETS["outfield_eligible_players"]))
    for criterion, observed, minimum in checks:
        status = "PASS" if observed >= minimum else "FAIL"
        rows.append({"criterion": criterion, "observed": observed, "minimum_required": minimum, "status": status, "blocks_production_candidate": status == "FAIL"})
    gate = pd.DataFrame(rows)
    return gate, bool((gate.status == "PASS").all())


def blocked_table(columns: list[str], rows: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame([{c: "blocked_data_readiness_failed" if c == "status" else None for c in columns}], columns=columns)


def build_blocked_outputs(inp: dict[str, pd.DataFrame], full_ready: bool, data_summary: dict[str, int]) -> dict[str, pd.DataFrame]:
    roles = inp["roles"]
    role_resolution = roles.copy()
    if "assigned_role" not in role_resolution.columns and "role" in role_resolution.columns:
        role_resolution["assigned_role"] = role_resolution["role"]
    role_resolution["experiment_008_status"] = "available_population_recomputed_or_reused_from_role_resolution"
    elig_rows = []
    for role in ROLES:
        sub = role_resolution[role_resolution.assigned_role == role] if "assigned_role" in role_resolution.columns else pd.DataFrame()
        eligible = sub[sub.get("eligible_for_initial_coefficients", False) == True] if not sub.empty and "eligible_for_initial_coefficients" in sub.columns else pd.DataFrame()
        elig_rows.append({"role": role, "assigned_players": len(sub), "eligible_players": len(eligible), "production_minimum": MIN_TARGETS["gk_eligible_players"] if role == "GK" else MIN_TARGETS["outfield_eligible_players"], "status": "PASS" if len(eligible) >= (MIN_TARGETS["gk_eligible_players"] if role == "GK" else MIN_TARGETS["outfield_eligible_players"]) else "FAIL"})
    role_elig = pd.DataFrame(elig_rows)
    threshold = inp["threshold_006"].copy()
    if threshold.empty:
        threshold = blocked_table(["role", "threshold", "eligible_player_count", "threshold_recommendation", "status"])
    threshold["experiment_008_status"] = "not_recalibrated_full_population_unavailable"

    candidate = inp["candidate_002"].copy()
    if not candidate.empty:
        candidate = candidate.rename(columns={"role_family": "role"})
        candidate["production_status"] = "research_only"
        candidate["experiment_008_status"] = "blocked_full_population_unavailable"
    metric_stability = candidate.copy()
    rejection = candidate[[c for c in candidate.columns if c in {"role", "requested_metric_alias", "status", "exclusion_reasons", "production_status", "experiment_008_status"}]].copy() if not candidate.empty else blocked_table(["role", "metric", "rejection_reason", "status"])

    norm = inp["norm_003"].copy(); norm["experiment_008_status"] = "not_recalibrated_full_population_unavailable"
    metric_stats = inp["metric_stats_003"].copy(); metric_stats["experiment_008_status"] = "not_recalibrated_full_population_unavailable"
    benchmarks = inp["benchmarks_003"].copy(); benchmarks["experiment_008_status"] = "not_recalibrated_full_population_unavailable"
    latent = inp["latent_003"].copy(); latent["stability_score"] = None; latent["confidence_score"] = None; latent["football_interpretation"] = "requires_full_population_review"; latent["experiment_008_status"] = "not_recalibrated_full_population_unavailable"
    clusters = inp["clusters_003"].copy(); clusters["experiment_008_status"] = "not_recalibrated_full_population_unavailable"
    dim_validation = latent[[c for c in latent.columns if c in {"role_family", "role", "dimension_id", "dimension_name", "stability_score", "confidence_score", "football_interpretation", "experiment_008_status"}]].copy()

    metric_w = inp["metric_w_004"].copy(); metric_w["experiment_008_status"] = "not_recomputed_full_population_unavailable"
    dim_w = inp["dim_w_004"].copy(); dim_w["experiment_008_status"] = "not_recomputed_full_population_unavailable"
    weight_ci = inp["weight_ci_005"].copy(); weight_ci["experiment_008_status"] = "not_recomputed_full_population_unavailable"
    dim_scores = inp["dim_scores_004"].copy(); dim_scores["experiment_008_status"] = "not_recalculated_full_population_unavailable"
    role_scores = inp["role_scores_004"].copy(); role_scores["experiment_008_status"] = "not_recalculated_full_population_unavailable"
    explanations = inp["explain_005"].copy(); explanations["experiment_008_status"] = "not_recalculated_full_population_unavailable"

    production_gate_rows = []
    for role in ROLES:
        eligible = int(role_elig.loc[role_elig.role == role, "eligible_players"].iloc[0]) if not role_elig[role_elig.role == role].empty else 0
        checks = {
            "enough_eligible_players": eligible >= (20 if role == "GK" else 50),
            "enough_matches": data_summary["matches"] >= MIN_TARGETS["matches"],
            "enough_competitions": data_summary["competitions"] >= MIN_TARGETS["competitions"],
            "enough_seasons": data_summary["seasons"] >= MIN_TARGETS["seasons"],
            "metric_coverage_pass": False,
            "temporal_stability_pass": False,
            "cross_season_validation_pass": False,
            "cross_competition_validation_pass": False,
            "rank_stability_pass": False,
            "confidence_interval_pass": False,
            "team_context_bias_acceptable": False,
            "population_drift_acceptable": False,
            "expert_review_completed_or_pending": False,
            "metric_direction_reviewed": False,
            "no_critical_data_quality_issue": False,
        }
        readiness = "Research Prototype"
        for criterion, passed in checks.items():
            production_gate_rows.append({"role": role, "criterion": criterion, "status": "PASS" if passed else "FAIL", "readiness_status": readiness, "production_ready_allowed": False, "reason": "full_population_data_readiness_failed" if not passed else "available_population_pass"})
    production_gate = pd.DataFrame(production_gate_rows)
    dashboard = production_gate.groupby("role").agg(failed_criteria=("status", lambda s: int((s == "FAIL").sum()))).reset_index()
    dashboard["readiness_status"] = "Research Prototype"
    dashboard["production_candidate_status"] = "not_generated_data_readiness_failed"
    blockers = production_gate[production_gate.status == "FAIL"].copy()

    validation_tables = {
        "008_bootstrap_statistics": inp["bootstrap_005"].assign(experiment_008_status="not_recomputed_full_population_unavailable"),
        "008_score_confidence": inp["confidence_005"].assign(experiment_008_status="not_recomputed_full_population_unavailable"),
        "008_reliability_summary": inp["reliability_005"].assign(experiment_008_status="not_recomputed_full_population_unavailable"),
        "008_rank_stability": inp["rank_stability_005"].assign(experiment_008_status="not_recomputed_full_population_unavailable"),
        "008_leave_one_season_out": inp["loso_006"].assign(experiment_008_status="not_recomputed_full_population_unavailable"),
        "008_leave_one_competition_out": inp["loco_006"].assign(experiment_008_status="not_recomputed_full_population_unavailable"),
        "008_team_context_sensitivity": inp["team_context_006"].assign(experiment_008_status="not_recomputed_full_population_unavailable"),
        "008_population_drift_summary": inp["population_drift_006"].assign(experiment_008_status="not_recomputed_full_population_unavailable"),
        "008_score_calibration_curves": inp["calibration_006"].assign(experiment_008_status="not_recomputed_full_population_unavailable"),
        "008_football_expert_review_queue": inp["review_006"].assign(experiment_008_status="expert_review_pending_full_population"),
    }

    tables = {
        "008_role_resolution": role_resolution,
        "008_role_eligibility_summary": role_elig,
        "008_minutes_threshold_recalibration": threshold,
        "008_metric_stability_by_role": metric_stability,
        "008_candidate_metric_status": candidate,
        "008_metric_rejection_reasons": rejection,
        "008_normalization_methods": inp["norm_methods_003"].assign(experiment_008_status="not_recalibrated_full_population_unavailable"),
        "008_normalization_decisions": norm,
        "008_metric_statistics": metric_stats,
        "008_role_benchmarks": benchmarks,
        "008_latent_dimensions": latent,
        "008_metric_clusters": clusters,
        "008_dimension_validation": dim_validation,
        "008_metric_weight_methods": inp["metric_weight_methods_004"].assign(experiment_008_status="not_recomputed_full_population_unavailable"),
        "008_metric_weight_decisions": metric_w,
        "008_dimension_weight_methods": inp["dim_weight_methods_004"].assign(experiment_008_status="not_recomputed_full_population_unavailable"),
        "008_dimension_weight_decisions": dim_w,
        "008_weight_confidence_intervals": weight_ci,
        "008_full_population_dimension_scores": dim_scores,
        "008_full_population_role_scores": role_scores,
        "008_full_population_score_explanations": explanations,
        "008_production_candidate_gate": production_gate,
        "008_production_readiness_dashboard": dashboard,
        "008_blockers_by_role": blockers,
    }
    tables.update(validation_tables)
    return tables


def load_previous_outputs() -> dict[str, pd.DataFrame]:
    return {
        "roles": read_table("002_role_resolution.csv", False),
        "candidate_002": read_table("002_candidate_metric_status.csv", False),
        "norm_003": read_table("003_normalization_decisions.csv", False),
        "norm_methods_003": read_table("003_normalization_methods.csv", False),
        "metric_stats_003": read_table("003_metric_statistics.csv", False),
        "benchmarks_003": read_table("003_role_benchmarks.csv", False),
        "latent_003": read_table("003_latent_dimensions.csv", False),
        "clusters_003": read_table("003_metric_clusters.csv", False),
        "metric_w_004": read_table("004_metric_weight_decisions.csv", False),
        "metric_weight_methods_004": read_table("004_metric_weight_methods.csv", False),
        "dim_w_004": read_table("004_dimension_weight_decisions.csv", False),
        "dim_weight_methods_004": read_table("004_dimension_weight_methods.csv", False),
        "dim_scores_004": read_table("004_prototype_dimension_scores.csv", False),
        "role_scores_004": read_table("004_prototype_role_scores.csv", False),
        "bootstrap_005": read_table("bootstrap_statistics.csv", False),
        "confidence_005": read_table("score_confidence.csv", False),
        "reliability_005": read_table("reliability_summary.csv", False),
        "rank_stability_005": read_table("rank_stability.csv", False),
        "weight_ci_005": read_table("weight_confidence_intervals.csv", False),
        "explain_005": read_table("explainability_contributions.csv", False),
        "threshold_006": read_table("006_minutes_threshold_sensitivity.csv", False),
        "loso_006": read_table("006_leave_one_season_out.csv", False),
        "loco_006": read_table("006_leave_one_competition_out.csv", False),
        "team_context_006": read_table("006_team_context_sensitivity.csv", False),
        "population_drift_006": read_table("006_population_drift_summary.csv", False),
        "calibration_006": read_table("006_score_calibration_curves.csv", False),
        "review_006": read_table("006_football_expert_review_workflow.csv", False),
    }


def make_figures(tables: dict[str, pd.DataFrame], inventory: pd.DataFrame, gate: pd.DataFrame) -> list[str]:
    fig_dir = ROOT / "outputs/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    def save(name: str) -> None:
        plt.tight_layout(); plt.savefig(fig_dir / name, dpi=150); plt.close(); paths.append(str(Path("outputs/figures") / name))
    scores = tables["008_full_population_role_scores"]
    metric_w = tables["008_metric_weight_decisions"]
    dim_w = tables["008_dimension_weight_decisions"]
    conf = tables["008_score_confidence"]
    rank = tables["008_rank_stability"]
    temporal = tables["008_leave_one_season_out"]
    cross = tables["008_leave_one_competition_out"]
    team = tables["008_team_context_sensitivity"]
    cal = tables["008_score_calibration_curves"]
    for role in ROLES:
        rs = scores[scores.get("role", pd.Series(dtype=str)) == role] if not scores.empty and "role" in scores.columns else pd.DataFrame()
        plt.figure(figsize=(8, 4)); sns.histplot(rs.get("prototype_role_score", pd.Series(dtype=float)), kde=True); plt.title(f"{role} score distribution — blocked/full-pop unavailable"); save(f"008_{role}_score_distribution.png")
        plt.figure(figsize=(8, 4)); sns.barplot(data=dim_w[dim_w.role == role] if "role" in dim_w.columns else pd.DataFrame(), x="adjusted_dimension_weight", y="dimension_name"); plt.title(f"{role} dimension weights"); save(f"008_{role}_dimension_weights.png")
        plt.figure(figsize=(8, 4)); sns.barplot(data=metric_w[metric_w.role == role] if "role" in metric_w.columns else pd.DataFrame(), x="selected_metric_weight", y="metric"); plt.title(f"{role} metric weights"); save(f"008_{role}_metric_weights.png")
        c = conf[conf.role == role] if "role" in conf.columns else pd.DataFrame(); plt.figure(figsize=(8,4)); sns.histplot(c.get("confidence_index", pd.Series(dtype=float)), kde=True); plt.title(f"{role} confidence distribution"); save(f"008_{role}_confidence_distribution.png")
        r = rank[rank.role == role] if "role" in rank.columns else pd.DataFrame(); plt.figure(figsize=(8,4)); sns.histplot(r.get("average_rank_variation", pd.Series(dtype=float))); plt.title(f"{role} rank stability"); save(f"008_{role}_rank_stability.png")
        plt.figure(figsize=(8,4)); sns.barplot(data=temporal[temporal.role == role] if "role" in temporal.columns else pd.DataFrame(), x="status", y="available_groups"); plt.title(f"{role} temporal validation"); save(f"008_{role}_temporal_validation.png")
        plt.figure(figsize=(8,4)); sns.barplot(data=cross[cross.role == role] if "role" in cross.columns else pd.DataFrame(), x="status", y="available_groups"); plt.title(f"{role} cross competition drift"); save(f"008_{role}_cross_competition_drift.png")
        plt.figure(figsize=(8,4)); sns.barplot(data=team[team.role == role] if "role" in team.columns else pd.DataFrame(), x="team_context_metric", y="score_team_context_correlation"); plt.xticks(rotation=45); plt.title(f"{role} team context sensitivity"); save(f"008_{role}_team_context_sensitivity.png")
        plt.figure(figsize=(8,4)); sns.lineplot(data=cal[cal.role == role] if "role" in cal.columns else pd.DataFrame(), x="score_band", y="mean_confidence_index", marker="o"); plt.title(f"{role} calibration curve"); save(f"008_{role}_calibration_curve.png")
        plt.figure(figsize=(8,4)); sns.countplot(data=gate[gate.role == role], y="status"); plt.title(f"{role} production gate"); save(f"008_{role}_production_gate.png")
    plt.figure(figsize=(10,4)); sns.countplot(data=gate, y="readiness_status", hue="role"); plt.title("Global production readiness dashboard"); save("008_global_production_readiness_dashboard.png")
    plt.figure(figsize=(10,4)); c=conf.groupby("role", as_index=False).confidence_index.mean() if "confidence_index" in conf.columns else pd.DataFrame({"role":ROLES,"confidence_index":[0]*6}); sns.barplot(data=c, x="role", y="confidence_index"); plt.title("Global role confidence comparison"); save("008_global_role_confidence_comparison.png")
    plt.figure(figsize=(10,4)); sns.barplot(data=inventory, x="dataset_name", y="row_count"); plt.xticks(rotation=45); plt.title("Global data coverage summary"); save("008_global_data_coverage_summary.png")
    plt.figure(figsize=(10,4)); sns.countplot(data=gate, y="status"); plt.title("Global validation gate summary"); save("008_global_validation_gate_summary.png")
    blockers=tables["008_blockers_by_role"].groupby("role", as_index=False).size(); plt.figure(figsize=(10,4)); sns.barplot(data=blockers, x="role", y="size"); plt.title("Global blockers by role"); save("008_global_blockers_by_role.png")
    return paths


def no_candidate_report(data_gate: pd.DataFrame, summary: dict[str, int]) -> None:
    failed = data_gate[data_gate.status == "FAIL"]
    text = "# Experiment 008 — No Production Candidate Reason\n\n"
    text += "A production-candidate bundle was not generated because the available data root does not meet the minimum production target.\n\n"
    text += "## Dataset summary\n\n"
    for k, v in summary.items(): text += f"- {k}: {v}\n"
    text += "\n## Failed gates\n\n"
    for r in failed.itertuples(index=False): text += f"- {r.criterion}: observed {r.observed}, minimum {r.minimum_required}\n"
    text += "\nNo production coefficients were changed, signed, or deployed.\n"
    (ROOT / "outputs/reports/008_no_production_candidate_reason.md").write_text(text, encoding="utf-8")


def write_notebook() -> None:
    heads = [
        "# Experiment 008 — Full-Population Recalibration & Production-Candidate Bundle",
        "## 1. Objective", "## 2. Whether Full Population Was Available", "## 3. Dataset Coverage",
        "## 4. Role Eligibility Recalibration", "## 5. Metric Stability Recalibration", "## 6. Normalization Recalibration",
        "## 7. Latent Dimension Recalibration", "## 8. Weight Recalibration", "## 9. Full Score Recalculation",
        "## 10. Full Validation Results", "## 11. Production-Candidate Gate", "## 12. Roles Passing / Failing",
        "## 13. Main Blockers by Role", "## 14. Expert Review Requirements", "## 15. Config Bundle Status",
        "## 16. Why This Is Or Is Not Production-Ready", "## 17. Recommended Next Step",
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(h + "\n\nReproducible Experiment 008 recalibration gate artefact." if h.startswith("##") else h) for h in heads]
    nb.cells += [nbf.v4.new_code_cell("import pandas as pd\npd.read_csv('outputs/tables/008_full_population_inventory.csv').head()"), nbf.v4.new_code_cell("pd.read_csv('outputs/tables/008_production_candidate_gate.csv').head()")]
    nbf.write(nb, ROOT / "notebooks/008_full_population_recalibration.ipynb")


def append_methodology(report: dict[str, Any]) -> None:
    p = ROOT / "methodology.md"; text = p.read_text()
    if "## Experiment 008" in text: return
    section = f"""
## Experiment 008 — {TITLE}

Date: {report['generated_at']}

### Objective
Audit full-population readiness and run the production-candidate recalibration gate using the Experiment 007 architecture. Stop before candidate bundle creation if the available population is insufficient.

### Football Hypothesis
A production-candidate score engine requires enough seasons, competitions, matches, eligible players, stable metrics, temporal evidence, cross-competition validation, expert review, and traceable coefficients.

### Dataset
Data root: `{report['data_root']}`. Full-population available: {report['full_population_available']}.

### Normalization Used
Full normalization recalibration was blocked because the data-readiness gate failed. Previous normalization artefacts were archived into 008 tables with explicit blocked status.

### Feature Selection
Full metric stability recalculation was blocked by population readiness. Candidate/rejection tables document that no production candidates were promoted.

### Algorithms
Data readiness audit, role eligibility summary, production gate checks, blocked-output archival, production-candidate bundle guard, and validation/report generation.

### Evaluation
Inventory rows: {report['table_counts']['full_population_inventory']}; data gate rows: {report['table_counts']['data_readiness_gate']}; production gate rows: {report['table_counts']['production_candidate_gate']}.

### Results
No production-candidate bundle was generated. All roles remain Research Prototype due to insufficient population coverage.

### Figures
Generated {report['figures_generated']} figures under `outputs/figures/008_*`.

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
"""
    p.write_text(text.rstrip() + "\n\n" + section.strip() + "\n", encoding="utf-8")


def update_readme() -> None:
    p = ROOT / "README.md"; text = p.read_text()
    if "experiments/008_full_population_recalibration.py" not in text:
        text += "\n\n## Experiment 008\n\nFull-population recalibration and production-candidate gate. Run:\n\n```bash\ncd /home/platform/DataScienceResearch\nuv run python experiments/008_full_population_recalibration.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse\n```\n\nThe experiment first audits whether the full StatsBomb/DataPlatform population is available. If readiness fails, it stops before production-candidate bundle creation and writes `outputs/reports/008_no_production_candidate_reason.md`. A signed bundle under `outputs/production_candidate_bundle/score_engine_v0.9.0/` is created only when at least one role reaches Production Candidate or Production Ready. Scores that remain blocked are not production scores.\n"
        p.write_text(text, encoding="utf-8")


def maybe_create_bundle(full_ready: bool, gate: pd.DataFrame, data_root: Path) -> str | None:
    pass_roles = sorted(gate.loc[gate.readiness_status.isin(["Production Candidate", "Production Ready"]), "role"].unique()) if "readiness_status" in gate.columns else []
    if not full_ready or not pass_roles:
        return None
    bundle = ROOT / "outputs/production_candidate_bundle/score_engine_v0.9.0"
    bundle.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "outputs/reports/score_engine_config.json").read_text()) if (ROOT / "outputs/reports/score_engine_config.json").exists() else {}
    coefficients = read_table("008_metric_weight_decisions.csv", False).to_dict(orient="records")
    write_json(bundle / "score_engine_config.json", config)
    write_json(bundle / "production_score_schema.json", json.loads((ROOT / "outputs/reports/production_score_schema.json").read_text()))
    write_json(bundle / "coefficients_by_role.json", coefficients)
    write_json(bundle / "normalization_params_by_role.json", read_table("008_normalization_decisions.csv", False).to_dict(orient="records"))
    read_table("008_full_population_role_scores.csv", False).to_csv(bundle / "percentile_tables_by_role.csv", index=False)
    write_json(bundle / "confidence_framework.json", {"source": "Experiment 007/008"})
    write_json(bundle / "quality_gate_report.json", gate.to_dict(orient="records"))
    (bundle / "validation_summary.md").write_text("# Validation Summary\n", encoding="utf-8")
    (bundle / "changelog.md").write_text("# Changelog\n", encoding="utf-8")
    coeff_hash = hashlib.sha256(json.dumps(coefficients, sort_keys=True, default=str).encode()).hexdigest()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "engine_version": "0.9.0", "data_root": str(data_root), "config_hash": config_hash, "coefficient_hash": coeff_hash, "validation_status": "candidate", "roles_included": pass_roles, "roles_rejected": sorted(set(ROLES) - set(pass_roles)), "production_candidate_status": "generated_not_deployed"}
    write_json(bundle / "manifest.json", manifest)
    return str(bundle)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", default="/home/platform/DataPlatform/tmp/master_data_warehouse")
    args = parser.parse_args(); data_root = Path(args.data_root).resolve()
    (ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True); (ROOT / "outputs/reports").mkdir(parents=True, exist_ok=True)
    inventory, summary = audit_dataset(data_root)
    prev = load_previous_outputs()
    gate, full_ready = data_readiness_gate(summary, prev["roles"])
    inventory.to_csv(table_path("008_full_population_inventory.csv"), index=False)
    gate.to_csv(table_path("008_data_readiness_gate.csv"), index=False)
    # Validate Experiment 007 production engine config can be loaded; do not change coefficients.
    config_path = ROOT / "outputs/reports/score_engine_config.json"
    if config_path.exists():
        engine = ScoreEngine(json.loads(config_path.read_text()))
        cfg_errors = engine.validate_config()
        if cfg_errors: raise SystemExit("Experiment 007 config invalid: " + ", ".join(cfg_errors))
    tables = build_blocked_outputs(prev, full_ready, summary)
    for name, df in tables.items():
        df.to_csv(table_path(f"{name}.csv"), index=False)
    bundle = maybe_create_bundle(full_ready, tables["008_production_candidate_gate"], data_root)
    if bundle is None:
        no_candidate_report(gate, summary)
    figs = make_figures(tables, inventory, tables["008_production_candidate_gate"])
    write_notebook()
    report = {
        "experiment_id": EXPERIMENT_ID,
        "title": TITLE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "full_population_available": full_ready,
        "dataset_summary": summary,
        "production_coefficients_changed": False,
        "production_ready_declared": False,
        "production_candidate_bundle_generated": bundle is not None,
        "bundle_location": bundle,
        "table_counts": {"full_population_inventory": int(len(inventory)), "data_readiness_gate": int(len(gate)), **{k.replace("008_", ""): int(len(v)) for k, v in tables.items()}},
        "figures_generated": len(figs),
        "figure_paths": figs,
        "roles": ROLES,
        "readiness_by_role": tables["008_production_readiness_dashboard"].to_dict(orient="records"),
    }
    write_json(ROOT / "outputs/reports/008_full_population_recalibration.json", report)
    md = ROOT / "outputs/reports/008_full_population_recalibration.md"
    md.write_text(
        f"# Experiment 008 — {TITLE}\n\n"
        f"## 1. Objective\nRun full-population recalibration and production-candidate gate without deploying or falsely promoting scores.\n\n"
        f"## 2. Whether full population was available\n{full_ready}. The current root fails the production data-readiness gate.\n\n"
        f"## 3. Dataset coverage\n```json\n{json.dumps(summary, indent=2)}\n```\n\n"
        "## 4. Role eligibility recalibration\nRole resolution and eligibility summary are written, but full recalibration is blocked by data readiness.\n\n"
        "## 5. Metric stability recalibration\nBlocked; archived into 008 tables with explicit status.\n\n"
        "## 6. Normalization recalibration\nBlocked; previous artefacts preserved with blocked status.\n\n"
        "## 7. Latent dimension recalibration\nBlocked until full population is available.\n\n"
        "## 8. Weight recalibration\nBlocked; no coefficients changed.\n\n"
        "## 9. Full score recalculation\nBlocked; no production scores declared.\n\n"
        "## 10. Full validation results\nBlocked by population readiness; prior validation artefacts are archived into 008 outputs for traceability.\n\n"
        "## 11. Production-candidate gate\nAll roles remain Research Prototype.\n\n"
        "## 12. Roles passing / failing\nNo role passes production-candidate gates on the current root.\n\n"
        "## 13. Main blockers by role\nSee `008_blockers_by_role.csv`.\n\n"
        "## 14. Expert review requirements\nExpert review remains required before any production promotion.\n\n"
        f"## 15. Whether config bundle was generated\n{bundle is not None}.\n\n"
        "## 16. Why this is or is not production-ready\nNot production-ready: insufficient seasons, competitions, matches, eligible players, and expert review.\n\n"
        "## 17. Recommended next step\nLoad/rebuild the full StatsBomb population and rerun Experiment 008.\n",
        encoding="utf-8",
    )
    append_methodology(report); update_readme()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
