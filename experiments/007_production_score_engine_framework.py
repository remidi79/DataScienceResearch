from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_score_engine_research.io import write_json
from football_score_engine_research.production_engine import ScoreEngine

EXPERIMENT_ID = "007"
TITLE = "Production Score Engine, Explainability & Confidence Framework"
ROLES = ["GK", "CB", "FB", "MID", "WINGER", "CF"]
ENGINE_VERSION = "0.5.0-research-validated"


def read_table(name: str) -> pd.DataFrame:
    p = ROOT / "outputs/tables" / name
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p)


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "normalization": read_table("003_normalization_decisions.csv"),
        "metric_directions": read_table("004_metric_direction_registry.csv"),
        "metric_weights": read_table("004_metric_weight_decisions.csv"),
        "dimension_weights": read_table("004_dimension_weight_decisions.csv"),
        "score_confidence": read_table("score_confidence.csv"),
        "production_gate": read_table("006_production_candidate_gate.csv"),
        "review_workflow": read_table("006_football_expert_review_workflow.csv"),
        "mapping_status": read_table("006_match_metric_mapping_status.csv"),
        "temporal_stability": read_table("006_temporal_stability.csv"),
        "threshold_sensitivity": read_table("006_minutes_threshold_sensitivity.csv"),
        "population_drift": read_table("006_population_drift_summary.csv"),
        "team_context": read_table("006_team_context_sensitivity.csv"),
    }


def build_score_config(inp: dict[str, pd.DataFrame]) -> dict[str, Any]:
    metric_w = inp["metric_weights"]
    dim_w = inp["dimension_weights"]
    norm = inp["normalization"]
    directions = inp["metric_directions"]
    mapping = inp["mapping_status"]
    definitions: dict[str, Any] = {}
    for role in ROLES:
        role_metrics = metric_w[metric_w.role == role].copy()
        role_dims = dim_w[dim_w.role == role].copy()
        metrics = []
        for row in role_metrics.itertuples(index=False):
            nrow = norm[(norm.role_family == role) & (norm.metric == row.metric)]
            drow = directions[(directions.role_family == role) & (directions.metric == row.metric)]
            mrow = mapping[(mapping.role == role) & (mapping.metric == row.metric)]
            metrics.append({
                "metric": row.metric,
                "dimension_id": row.latent_dimension,
                "weight": float(row.selected_metric_weight),
                "normalization": None if nrow.empty else str(nrow.selected_normalization.iloc[0]),
                "direction": None if drow.empty else str(drow.direction.iloc[0]),
                "match_level_mapping_status": None if mrow.empty else str(mrow.mapping_status.iloc[0]),
                "include_in_score": False if (not drow.empty and str(drow.direction.iloc[0]) == "manual_review_required") else True,
                "confidence": str(row.weight_confidence),
                "warning_flags": [] if pd.isna(row.warning_flags) else str(row.warning_flags).split(";"),
            })
        dimensions = []
        for row in role_dims.itertuples(index=False):
            dimensions.append({
                "dimension_id": row.dimension_name,
                "weight": float(row.adjusted_dimension_weight),
                "number_of_metrics": int(row.number_of_metrics),
                "confidence_level": str(row.confidence_level),
                "warning_flags": [] if pd.isna(row.warning_flags) else str(row.warning_flags).split(";"),
            })
        definitions[role] = {
            "score_name": f"{role}_prototype_role_score",
            "status": "research_validation_score",
            "role_specific": True,
            "metrics": metrics,
            "dimensions": dimensions,
            "calibration": {
                "percentile_scope": "within_role_family",
                "competition_percentile": "secondary_output",
                "global_percentile": "secondary_output",
                "calibration_status": "requires_full_population_recalibration",
            },
        }
    return {
        "engine_version": ENGINE_VERSION,
        "engine_stage": "research_validated_architecture_not_production_coefficients",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "score_definitions": definitions,
        "eligibility_rules": {
            "role_families": ROLES,
            "unknown_policy": "exclude_from_coefficient_fitting_keep_visible",
            "multi_role_policy": "exclude_from_initial_coefficients",
            "official_threshold_change_policy": "changes_require_full_population_revalidation",
        },
        "percentile_rules": {
            "primary": "within_player_positional_family",
            "competition": "within_role_and_competition_when_population_sufficient",
            "global": "secondary_context_only",
            "minimum_population_for_percentile": 30,
        },
        "confidence_framework": {
            "formula": "0.25*bootstrap_stability + 0.20*weight_stability + 0.15*minutes_reliability + 0.15*population_reliability + 0.10*metric_coverage + 0.10*data_quality + 0.05*validation_status_score",
            "quality_bands": {"excellent": 85, "good": 75, "adequate": 65, "research_only": 0},
        },
        "versioning": {
            "0.1": "Prototype",
            "0.5": "Research Validated",
            "0.9": "Full Population Validated",
            "1.0": "Production Ready",
            "current": ENGINE_VERSION,
        },
    }


def build_schema(config: dict[str, Any]) -> dict[str, Any]:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Football Score Engine PlayerScore",
        "type": "object",
        "required": ["player_id", "role", "score_engine_version", "scores", "percentiles", "confidence", "metric_contributions", "dimension_contributions", "strengths", "weaknesses", "quality_flags", "metadata"],
        "properties": {
            "player_id": {"type": "string"},
            "player_name": {"type": ["string", "null"]},
            "role": {"type": "string", "enum": ROLES},
            "score_engine_version": {"type": "string"},
            "scores": {
                "type": "object",
                "properties": {
                    "overall": {"type": "number", "minimum": 0, "maximum": 100},
                    "dimensions": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0, "maximum": 100}},
                },
            },
            "percentiles": {"type": "object", "properties": {"role": {"type": "number"}, "competition": {"type": ["number", "null"]}, "global": {"type": ["number", "null"]}}},
            "confidence": {"type": "object", "required": ["confidence_index", "confidence_interval", "bootstrap_stability", "weight_stability", "minutes_reliability", "population_reliability", "metric_coverage", "data_quality", "validation_status"]},
            "metric_contributions": {"type": "array", "items": {"type": "object", "required": ["metric", "dimension", "raw_value", "normalized_value", "direction", "weight", "contribution"]}},
            "dimension_contributions": {"type": "array", "items": {"type": "object", "required": ["dimension", "score", "weight", "contribution"]}},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "weaknesses": {"type": "array", "items": {"type": "string"}},
            "quality_flags": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object"},
        },
        "score_engine_roles": {role: config["score_definitions"][role] for role in ROLES},
    }
    return schema


def readiness_dashboard(inp: dict[str, pd.DataFrame]) -> pd.DataFrame:
    gate = inp["production_gate"]
    conf = inp["score_confidence"]
    rows = []
    for role in ROLES:
        rg = gate[gate.role == role]
        c = conf[conf.role == role]
        checks = {k: int((rg[rg.criterion == k].status == "PASS").any()) for k in rg.criterion.unique()}
        status = rg.readiness_status.iloc[0] if not rg.empty else "Research Prototype"
        rows.append({
            "score_engine": f"{role}_role_score",
            "role": role,
            "statistical_validation": "PASS" if checks.get("temporal_stability", 0) else "FAIL",
            "football_validation": "PENDING",
            "confidence": float(c.confidence_index.mean()) if not c.empty else None,
            "stability": "PASS" if checks.get("temporal_stability", 0) else "FAIL",
            "sample_size_status": "PASS" if checks.get("enough_eligible_players", 0) else "FAIL",
            "data_coverage": "PASS" if checks.get("enough_matches", 0) else "FAIL",
            "calibration_status": "requires_full_population_recalibration",
            "production_status": status,
            "ready": "No",
        })
    return pd.DataFrame(rows)


def gap_analysis(inp: dict[str, pd.DataFrame]) -> pd.DataFrame:
    gate = inp["production_gate"]
    mapping = inp["mapping_status"]
    rows = []
    for role in ROLES:
        rg = gate[gate.role == role]
        for row in rg[rg.status != "PASS"].itertuples(index=False):
            rows.append({
                "role": role,
                "gap_type": row.criterion,
                "gap_description": f"Production gate criterion `{row.criterion}` is {row.status}.",
                "severity": "high" if row.status == "FAIL" else "medium",
                "required_action": "Load full StatsBomb population, rerun validation, and complete expert review." if row.criterion in {"enough_seasons", "enough_competitions", "expert_review_completed"} else "Investigate and rerun validation.",
                "blocks_version": "1.0",
            })
        missing = mapping[(mapping.role == role) & (mapping.mapping_status != "available_match_level")]
        if not missing.empty:
            rows.append({"role": role, "gap_type": "match_metric_mapping", "gap_description": f"{len(missing)} score metrics are not directly available at match level.", "severity": "medium", "required_action": "Complete match-level metric mapping or define approved approximations.", "blocks_version": "0.9"})
    return pd.DataFrame(rows)


def write_docs(config: dict[str, Any], dashboard: pd.DataFrame, gaps: pd.DataFrame) -> None:
    reports = ROOT / "outputs/reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "production_score_schema.json", build_schema(config))
    write_json(reports / "score_engine_config.json", config)
    (reports / "confidence_framework.md").write_text(
        "# Confidence Framework\n\n"
        "Confidence is configuration-driven and does not alter metric coefficients.\n\n"
        "Formula:\n\n"
        "`confidence = 0.25*bootstrap_stability + 0.20*weight_stability + 0.15*minutes_reliability + 0.15*population_reliability + 0.10*metric_coverage + 0.10*data_quality + 0.05*validation_status_score`\n\n"
        "Inputs:\n- Bootstrap stability from Experiment 005/006.\n- Weight stability from Experiment 004/005.\n- Minutes reliability from eligibility and score confidence artefacts.\n- Population reliability from role sample size, seasons, and competitions.\n- Metric coverage from match/season metric availability.\n- Data quality from quality flags.\n- Validation status from production candidate gates.\n\n"
        "Bands:\n- Excellent: >=85\n- Good: >=75\n- Adequate: >=65\n- Research Only: <65\n",
        encoding="utf-8",
    )
    (reports / "explainability_framework.md").write_text(
        "# Explainability Framework\n\n"
        "Every score must explain metric-level, dimension-level, and overall-role contributions.\n\n"
        "Metric contribution = oriented_normalized_metric_value * metric_weight * dimension_weight.\n\n"
        "Positive contributors are the highest positive deviations from the player's role score baseline. Negative contributors are the lowest contribution deltas.\n\n"
        "The API must return: metric, raw value, normalized value, direction, metric weight, dimension, dimension weight, contribution, contribution percentage, and flags.\n\n"
        "The framework is football-reviewable but does not automatically judge whether a player is good or bad. Review flags route anomalies to experts.\n",
        encoding="utf-8",
    )
    (reports / "score_engine_architecture.md").write_text(
        "# Score Engine Architecture\n\n"
        "Pipeline:\n\nPlayer -> Raw Metrics -> Normalization -> Direction Correction -> Feature Selection -> Metric Weights -> Dimension Scores -> Role Calibration -> Percentile -> Confidence -> Explainability -> Production Score Object.\n\n"
        "Reusable classes are implemented in `src/football_score_engine_research/production_engine.py`:\n\n"
        "- MetricContribution\n- DimensionScore\n- ScoreConfidence\n- ScoreExplanation\n- RoleCalibration\n- PlayerScore\n- ScoreEngine\n\n"
        "The implementation is configuration-driven. Metrics, dimensions, directions, weights, confidence thresholds, percentile rules, and eligibility rules are loaded from generated configuration. No business logic should hardcode score metrics.\n",
        encoding="utf-8",
    )
    (reports / "score_versioning.md").write_text(
        "# Score Versioning\n\n"
        "- 0.1 Prototype: early research artefacts, no production use.\n"
        "- 0.5 Research Validated: Experiments 001-007 architecture complete, local/sample validation only.\n"
        "- 0.9 Full Population Validated: complete StatsBomb multi-season/multi-competition recalibration and validation passed.\n"
        "- 1.0 Production Ready: expert review complete, production gates passed, immutable coefficient bundle signed and documented.\n\n"
        f"Current version: {ENGINE_VERSION}.\n\n"
        "Every version must store source dataset snapshot, experiment outputs, config hash, coefficient bundle, and validation report.\n",
        encoding="utf-8",
    )
    (reports / "future_full_population_pipeline.md").write_text(
        "# Future Full Population Pipeline\n\n"
        "This design is not executed in Experiment 007.\n\n"
        "1. Reload all eligible StatsBomb competitions and seasons.\n"
        "2. Rebuild data inventory and provider metric contract.\n"
        "3. Recompute role eligibility and minutes thresholds.\n"
        "4. Recompute normalization distributions by role.\n"
        "5. Re-estimate weights with the approved Experiment 004 methodology.\n"
        "6. Recalculate percentiles and confidence intervals.\n"
        "7. Validate bootstrap, temporal, cross-season, and cross-league robustness.\n"
        "8. Generate expert-review work queues and collect reviewer decisions.\n"
        "9. Produce signed coefficient/config bundle.\n"
        "10. Promote or reject each score engine using the predefined production gate.\n\n"
        "Promotion requires enough players, matches, seasons, competitions, low drift, stable ranks, completed expert review, and no critical metric-direction issues.\n",
        encoding="utf-8",
    )


def write_notebook() -> None:
    heads = [
        "# Experiment 007 — Production Score Engine, Explainability & Confidence Framework",
        "## 1. Objective",
        "## 2. Production Architecture",
        "## 3. Explainability Framework",
        "## 4. Confidence Framework",
        "## 5. Score Card Contract",
        "## 6. Explainability API Contract",
        "## 7. Score Versioning",
        "## 8. Configuration Strategy",
        "## 9. Production Readiness Dashboard",
        "## 10. Research Gap Analysis",
        "## 11. Future Full Population Pipeline",
        "## 12. Conclusions",
        "## 13. Next Experiment",
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(h + "\n\nReproducible Experiment 007 architecture artefact." if h.startswith("##") else h) for h in heads]
    nb.cells += [
        nbf.v4.new_code_cell("import json\nfrom pathlib import Path\nschema=json.loads(Path('outputs/reports/production_score_schema.json').read_text())\nschema['title']"),
        nbf.v4.new_code_cell("import pandas as pd\npd.read_csv('outputs/tables/production_readiness_dashboard.csv').head()"),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/research_gap_analysis.csv').head()"),
    ]
    nbf.write(nb, ROOT / "notebooks/007_production_score_engine_framework.ipynb")


def append_methodology(report: dict[str, Any]) -> None:
    p = ROOT / "methodology.md"
    text = p.read_text()
    if "## Experiment 007" in text:
        return
    section = f"""
## Experiment 007 — {TITLE}

Date: {report['generated_at']}

### Objective
Transform research outputs into a production Score Engine architecture, explainability contract, confidence framework, versioning strategy, configuration framework, readiness dashboard, research-gap roadmap, and full-population recalibration design. No new coefficients are introduced.

### Production Architecture
Implemented reusable production-facing dataclasses and a configuration-driven `ScoreEngine` skeleton in `src/football_score_engine_research/production_engine.py`.

### Explainability Framework
Defined metric, dimension, and overall role explanation model with positive/negative contributors and review flags.

### Confidence Framework
Defined confidence formula combining bootstrap stability, weight stability, minutes reliability, population reliability, metric coverage, data quality, and validation status.

### Score Versioning
Defined 0.1 Prototype, 0.5 Research Validated, 0.9 Full Population Validated, and 1.0 Production Ready stages. Current stage: `{ENGINE_VERSION}`.

### Configuration Strategy
Generated configuration-driven score definitions from Experiment 003–006 artefacts. Metrics, dimensions, directions, normalization, weights, eligibility, percentiles, and confidence rules are config objects.

### Production Readiness
Readiness dashboard rows: {report['dashboard_rows']}. All roles remain research/validation status pending full-population recalibration and expert review.

### Remaining Research Gaps
Research gap rows: {report['gap_rows']}. Main gaps: additional seasons, more competitions, complete expert review, match-level metric mapping, and full-population recalibration.

### Next Experiment
Experiment 008 should run the full-population recalibration pipeline once the complete StatsBomb dataset is available, then produce candidate coefficient bundles for gate review.
"""
    p.write_text(text.rstrip() + "\n\n" + section.strip() + "\n", encoding="utf-8")


def update_readme() -> None:
    p = ROOT / "README.md"
    text = p.read_text()
    if "experiments/007_production_score_engine_framework.py" not in text:
        text += "\n\n## Experiment 007\n\nProduction score-engine architecture, explainability contract, confidence framework, versioning strategy, configuration strategy, readiness dashboard, research gap analysis, and future full-population pipeline design. Run:\n\n```bash\ncd /home/platform/DataScienceResearch\nuv run python experiments/007_production_score_engine_framework.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse\n```\n\nThis experiment does not change production coefficients. It makes the system ready for full-population recalibration once the complete multi-season StatsBomb dataset is available.\n"
        p.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/platform/DataPlatform/tmp/master_data_warehouse")
    args = parser.parse_args()
    inp = load_inputs()
    config = build_score_config(inp)
    engine = ScoreEngine(config)
    config_errors = engine.validate_config()
    if config_errors:
        raise SystemExit("Invalid generated config: " + ", ".join(config_errors))
    dashboard = readiness_dashboard(inp)
    gaps = gap_analysis(inp)
    (ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True)
    dashboard.to_csv(ROOT / "outputs/tables/production_readiness_dashboard.csv", index=False)
    gaps.to_csv(ROOT / "outputs/tables/research_gap_analysis.csv", index=False)
    # Versioned copies keep chronological experiment lineage.
    dashboard.to_csv(ROOT / "outputs/tables/007_production_readiness_dashboard.csv", index=False)
    gaps.to_csv(ROOT / "outputs/tables/007_research_gap_analysis.csv", index=False)
    write_docs(config, dashboard, gaps)
    write_notebook()
    report = {
        "experiment_id": EXPERIMENT_ID,
        "title": TITLE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(Path(args.data_root).resolve()),
        "engine_version": ENGINE_VERSION,
        "production_coefficients_changed": False,
        "production_coefficients_declared": False,
        "score_schema_roles": ROLES,
        "dashboard_rows": int(len(dashboard)),
        "gap_rows": int(len(gaps)),
        "config_driven": True,
        "remaining_status": "ready_for_full_population_recalibration_not_production_deployment",
        "generated_outputs": [
            "outputs/reports/production_score_schema.json",
            "outputs/reports/confidence_framework.md",
            "outputs/reports/explainability_framework.md",
            "outputs/reports/score_engine_architecture.md",
            "outputs/reports/score_versioning.md",
            "outputs/tables/production_readiness_dashboard.csv",
            "outputs/tables/research_gap_analysis.csv",
            "outputs/reports/future_full_population_pipeline.md",
        ],
    }
    write_json(ROOT / "outputs/reports/007_production_score_engine_framework.json", report)
    (ROOT / "outputs/reports/007_production_score_engine_framework.md").write_text(
        f"# Experiment 007 — {TITLE}\n\n"
        "No production coefficients are changed or declared.\n\n"
        "## Production Architecture\nSee `score_engine_architecture.md` and `src/football_score_engine_research/production_engine.py`.\n\n"
        "## Explainability Framework\nSee `explainability_framework.md`.\n\n"
        "## Confidence Framework\nSee `confidence_framework.md`.\n\n"
        "## Score Versioning\nSee `score_versioning.md`.\n\n"
        "## Configuration Strategy\nSee `score_engine_config.json` and `production_score_schema.json`.\n\n"
        "## Production Readiness\nSee `production_readiness_dashboard.csv`. All roles remain blocked from production until full-population recalibration and expert review.\n\n"
        "## Remaining Research Gaps\nSee `research_gap_analysis.csv`.\n\n"
        "## Next Experiment\nRun full-population recalibration when complete StatsBomb data is available.\n",
        encoding="utf-8",
    )
    append_methodology(report)
    update_readme()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
