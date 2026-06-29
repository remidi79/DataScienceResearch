from __future__ import annotations

import argparse
import json
import subprocess
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
from football_score_engine_research.io import write_json

EXPERIMENT_ID = "009"
TITLE = "Full DataPlatform Reload & End-to-End Recalibration Orchestration"
MIN_TARGETS = {
    "competitions": 3,
    "seasons": 2,
    "matches": 100,
    "teams": 20,
    "player_match_rows": 3000,
    "player_season_rows": 300,
    "events": 250000,
    "lineups": 3000,
    "outfield_eligible_players": 50,
    "gk_eligible_players": 20,
}
ROLES = ["GK", "CB", "FB", "MID", "WINGER", "CF"]

DATASETS = {
    "player_match_stats_direct": "marts_v2/mart_statsbomb_player_match_stats_direct_v1.jsonl",
    "team_match_stats_direct": "marts_v2/mart_statsbomb_team_match_stats_direct_v1.jsonl",
    "player_season_stats_direct": "marts_v2/mart_statsbomb_player_season_stats_direct_v1.jsonl",
    "team_season_stats_direct": "marts_v2/mart_statsbomb_team_season_stats_direct_v1.jsonl",
    "silver_events": "silver/silver_events.jsonl",
    "silver_lineups": "silver/silver_lineups.jsonl",
    "silver_matches": "silver/silver_matches.jsonl",
}


def read_jsonl_sample(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def output_dirs() -> None:
    for d in ["outputs/tables", "outputs/reports", "outputs/figures", "outputs/runs", "notebooks"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def discover_root(data_root: Path) -> pd.DataFrame:
    rows = []
    extra_patterns = {
        "competition_metadata": ["*competition*.jsonl", "*competitions*.jsonl"],
        "season_metadata": ["*season*.jsonl", "*seasons*.jsonl"],
        "team_metadata": ["*team*.jsonl", "*teams*.jsonl"],
        "player_metadata": ["*player*.jsonl", "*players*.jsonl"],
    }
    paths: dict[str, list[Path]] = {k: [data_root / rel] for k, rel in DATASETS.items()}
    for name, pats in extra_patterns.items():
        found: list[Path] = []
        for pat in pats:
            found.extend(data_root.rglob(pat))
        paths[name] = sorted(set(found))[:10]
    required_keys = ["statsbomb_player_id", "player_id", "team_id", "match_provider_id", "provider_id", "competition_id", "season_id", "match_date"]
    for dataset, files in paths.items():
        real_files = [p for p in files if p.exists()]
        row_count = sum(count_lines(p) for p in real_files if p.suffix == ".jsonl")
        sample = pd.concat([read_jsonl_sample(p, 200) for p in real_files if p.suffix == ".jsonl"], ignore_index=True) if real_files else pd.DataFrame()
        cols = list(sample.columns)
        key_available = [c for c in required_keys if c in cols]
        missing_keys = []
        for key in required_keys:
            if key in cols and sample[key].isna().any():
                missing_keys.append(key)
        date_min = date_max = ""
        if "match_date" in sample.columns:
            dt = pd.to_datetime(sample.match_date, errors="coerce")
            if dt.notna().any():
                date_min, date_max = str(dt.min().date()), str(dt.max().date())
        comp_cov = int(sample.competition_id.nunique()) if "competition_id" in sample.columns else 0
        season_cov = int(sample.season_id.nunique()) if "season_id" in sample.columns else 0
        rows.append({
            "dataset_name": dataset,
            "path": ";".join(str(p) for p in real_files) if real_files else "",
            "file_count": len(real_files),
            "row_count": row_count,
            "columns": ";".join(cols),
            "key_fields_available": ";".join(key_available),
            "missing_key_fields": ";".join(missing_keys),
            "date_range": f"{date_min}..{date_max}" if date_min else "",
            "competition_coverage": comp_cov,
            "season_coverage": season_cov,
            "status": "present" if real_files and row_count else "missing_or_empty",
        })
    return pd.DataFrame(rows)


def build_contract() -> dict[str, Any]:
    return {
        "matches": {"required_fields": ["match_id", "competition_id", "competition_name", "season_id", "season_name", "match_date", "home_team_id", "away_team_id", "home_team_name", "away_team_name"]},
        "lineups": {"required_fields": ["match_id", "player_id", "player_name", "team_id", "team_name", "position", "start_time", "end_time", "minutes", "lineup_role_evidence"]},
        "player_match_stats": {"required_fields": ["player_id", "player_name", "match_id", "team_id", "competition_id", "season_id", "minutes", "metric_columns"]},
        "player_season_stats": {"required_fields": ["player_id", "player_name", "team_id", "competition_id", "season_id", "minutes", "metric_columns"]},
        "events": {"required_fields": ["match_id", "event_id", "event_type", "player_id", "team_id", "possession", "play_pattern", "location", "timestamp", "obv_fields"]},
        "minimum_production_targets": MIN_TARGETS,
        "role_targets": {"GK": 20, "CB": 50, "FB": 50, "MID": 50, "WINGER": 50, "CF": 50},
    }


def current_summary(discovery: pd.DataFrame) -> dict[str, int]:
    def ds(name: str) -> pd.Series:
        x = discovery[discovery.dataset_name == name]
        return x.iloc[0] if not x.empty else pd.Series(dtype=object)
    matches = int(ds("silver_matches").get("row_count", 0) or 0)
    pm_rows = int(ds("player_match_stats_direct").get("row_count", 0) or 0)
    ps_rows = int(ds("player_season_stats_direct").get("row_count", 0) or 0)
    events = int(ds("silver_events").get("row_count", 0) or 0)
    lineups = int(ds("silver_lineups").get("row_count", 0) or 0)
    core = discovery[discovery.dataset_name.isin(DATASETS.keys())]
    comps = int(max(core.competition_coverage.max() if not core.empty else 0, 0))
    seasons = int(max(core.season_coverage.max() if not core.empty else 0, 0))
    teams = 0
    players = 0
    for dataset in ["player_season_stats_direct", "player_match_stats_direct", "silver_events"]:
        pstr = ds(dataset).get("path", "")
        if pstr:
            df = read_jsonl_sample(Path(str(pstr).split(";")[0]), None)
            if "team_id" in df.columns: teams = max(teams, int(df.team_id.nunique()))
            if "team_name" in df.columns: teams = max(teams, int(df.team_name.nunique()))
            if "statsbomb_player_id" in df.columns: players = max(players, int(df.statsbomb_player_id.nunique()))
            if "player_id" in df.columns: players = max(players, int(df.player_id.nunique()))
    return {"competitions": comps, "seasons": seasons, "matches": matches, "teams": teams, "players": players, "player_match_rows": pm_rows, "player_season_rows": ps_rows, "events": events, "lineups": lineups}


def gap_analysis(discovery: pd.DataFrame, contract: dict[str, Any], summary: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gaps = []
    checks = [
        ("competitions", summary["competitions"], MIN_TARGETS["competitions"]),
        ("seasons", summary["seasons"], MIN_TARGETS["seasons"]),
        ("matches", summary["matches"], MIN_TARGETS["matches"]),
        ("teams", summary["teams"], MIN_TARGETS["teams"]),
        ("player_match_rows", summary["player_match_rows"], MIN_TARGETS["player_match_rows"]),
        ("player_season_rows", summary["player_season_rows"], MIN_TARGETS["player_season_rows"]),
        ("events", summary["events"], MIN_TARGETS["events"]),
        ("lineups", summary["lineups"], MIN_TARGETS["lineups"]),
    ]
    for entity, observed, expected in checks:
        gaps.append({"gap_type": f"missing_or_insufficient_{entity}", "observed": observed, "expected_minimum": expected, "gap_size": max(0, expected - observed), "severity": "critical" if observed < expected else "none", "status": "FAIL" if observed < expected else "PASS"})
    for row in discovery.itertuples(index=False):
        if row.status != "present":
            gaps.append({"gap_type": f"missing_dataset:{row.dataset_name}", "observed": 0, "expected_minimum": 1, "gap_size": 1, "severity": "critical", "status": "FAIL"})
        if getattr(row, "missing_key_fields", ""):
            gaps.append({"gap_type": f"missing_key_fields:{row.dataset_name}", "observed": row.missing_key_fields, "expected_minimum": "no missing required keys", "gap_size": None, "severity": "high", "status": "WARNING"})
    quality = []
    for row in discovery.itertuples(index=False):
        quality.append({"dataset_name": row.dataset_name, "issue_type": "missing_or_empty" if row.status != "present" else "none", "severity": "critical" if row.status != "present" else "none", "detail": "dataset missing or empty" if row.status != "present" else "no critical file-level issue detected"})
    id_rows = []
    matches_path = discovery.loc[discovery.dataset_name == "silver_matches", "path"].iloc[0] if (discovery.dataset_name == "silver_matches").any() else ""
    pm_path = discovery.loc[discovery.dataset_name == "player_match_stats_direct", "path"].iloc[0] if (discovery.dataset_name == "player_match_stats_direct").any() else ""
    lineups_path = discovery.loc[discovery.dataset_name == "silver_lineups", "path"].iloc[0] if (discovery.dataset_name == "silver_lineups").any() else ""
    matches = read_jsonl_sample(Path(matches_path), None) if matches_path else pd.DataFrame()
    pm = read_jsonl_sample(Path(pm_path), None) if pm_path else pd.DataFrame()
    lineups = read_jsonl_sample(Path(lineups_path), None) if lineups_path else pd.DataFrame()
    if not matches.empty and not pm.empty and "provider_id" in matches.columns and "match_provider_id" in pm.columns:
        match_ids = set(matches.provider_id.astype(str)); pm_ids = set(pm.match_provider_id.astype(str))
        id_rows.append({"audit": "player_stats_without_match_metadata", "count": len(pm_ids - match_ids), "status": "PASS" if len(pm_ids - match_ids) == 0 else "FAIL"})
        id_rows.append({"audit": "matches_without_player_stats", "count": len(match_ids - pm_ids), "status": "PASS" if len(match_ids - pm_ids) == 0 else "WARNING"})
    if not lineups.empty and not pm.empty and "match_provider_id" in lineups.columns and "match_provider_id" in pm.columns:
        id_rows.append({"audit": "lineups_without_player_stats_match", "count": len(set(lineups.match_provider_id.astype(str)) - set(pm.match_provider_id.astype(str))), "status": "WARNING"})
    if not id_rows:
        id_rows.append({"audit": "id_consistency", "count": 0, "status": "BLOCKED", "detail": "insufficient metadata for full ID audit"})
    return pd.DataFrame(gaps), pd.DataFrame(quality), pd.DataFrame(id_rows)


def reload_plan() -> tuple[pd.DataFrame, str]:
    tasks = [
        ("R001", "competition_metadata", "StatsBomb competitions/seasons endpoints or DataPlatform bronze metadata", "metadata/competitions_seasons", 1, 1, 3, 2, 0),
        ("R002", "silver_matches", "StatsBomb matches endpoint by competition/season", "silver/silver_matches.jsonl", 1, 100, 3, 2, 100),
        ("R003", "silver_lineups", "StatsBomb lineups endpoint by match", "silver/silver_lineups.jsonl", 1, 3000, 3, 2, 100),
        ("R004", "silver_events", "StatsBomb events endpoint by match", "silver/silver_events.jsonl", 1, 250000, 3, 2, 100),
        ("R005", "player_match_stats_direct", "StatsBomb player match stats endpoint", "marts_v2/mart_statsbomb_player_match_stats_direct_v1.jsonl", 1, 3000, 3, 2, 100),
        ("R006", "team_match_stats_direct", "StatsBomb team match stats endpoint", "marts_v2/mart_statsbomb_team_match_stats_direct_v1.jsonl", 1, 200, 3, 2, 100),
        ("R007", "player_season_stats_direct", "StatsBomb player season stats endpoint", "marts_v2/mart_statsbomb_player_season_stats_direct_v1.jsonl", 2, 300, 3, 2, 0),
        ("R008", "team_season_stats_direct", "StatsBomb team season stats endpoint", "marts_v2/mart_statsbomb_team_season_stats_direct_v1.jsonl", 2, 20, 3, 2, 0),
    ]
    rows = []
    for tid, dataset, source, target, priority, min_rows, min_comp, min_season, min_match in tasks:
        rows.append({"task_id": tid, "dataset": dataset, "source": source, "target": target, "priority": priority, "blocking_status": "blocking", "validation_check": "row counts, required keys, ID consistency, coverage gates", "expected_min_rows": min_rows, "expected_min_competitions": min_comp, "expected_min_seasons": min_season, "expected_min_matches": min_match, "owner_status": "pending_dataplatform_reload"})
    md = "# Experiment 009 — Full Data Reload Plan\n\nReload DataPlatform in dependency order:\n\n"
    for r in rows:
        md += f"- {r['task_id']} {r['dataset']}: source={r['source']} -> target={r['target']}; validate {r['validation_check']}.\n"
    md += "\nThe score-engine rerun must not start until all blocking datasets pass the readiness gate.\n"
    return pd.DataFrame(rows), md


def readiness_gate(summary: dict[str, int], gaps: pd.DataFrame, id_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    criteria = [
        ("competitions_ge_3", summary["competitions"], MIN_TARGETS["competitions"]),
        ("seasons_ge_2", summary["seasons"], MIN_TARGETS["seasons"]),
        ("matches_ge_100", summary["matches"], MIN_TARGETS["matches"]),
        ("teams_ge_20", summary["teams"], MIN_TARGETS["teams"]),
        ("player_match_rows_sufficient", summary["player_match_rows"], MIN_TARGETS["player_match_rows"]),
        ("player_season_rows_sufficient", summary["player_season_rows"], MIN_TARGETS["player_season_rows"]),
        ("events_sufficient", summary["events"], MIN_TARGETS["events"]),
        ("lineups_sufficient", summary["lineups"], MIN_TARGETS["lineups"]),
    ]
    for c, obs, req in criteria:
        rows.append({"criterion": c, "observed": obs, "required": req, "status": "PASS" if obs >= req else "FAIL", "failure_reason": "" if obs >= req else f"observed {obs} below required {req}"})
    critical_id = int((id_audit.status == "FAIL").sum()) if "status" in id_audit.columns else 1
    rows.append({"criterion": "no_critical_id_consistency_issue", "observed": critical_id, "required": 0, "status": "PASS" if critical_id == 0 else "FAIL", "failure_reason": "critical ID audit issue" if critical_id else ""})
    # Role eligibility cannot be recomputed production-grade unless upstream lineups/player stats are sufficient.
    role_status = "PASS" if summary["lineups"] >= MIN_TARGETS["lineups"] and summary["player_season_rows"] >= MIN_TARGETS["player_season_rows"] else "FAIL"
    rows.append({"criterion": "role_eligibility_can_be_computed", "observed": min(summary["lineups"], summary["player_season_rows"]), "required": min(MIN_TARGETS["lineups"], MIN_TARGETS["player_season_rows"]), "status": role_status, "failure_reason": "insufficient lineups/player-season population" if role_status == "FAIL" else ""})
    for role in ROLES:
        rows.append({"criterion": f"enough_eligible_players_{role}", "observed": "requires Experiment 002 rerun", "required": 20 if role == "GK" else 50, "status": "BLOCKED" if role_status == "FAIL" else "WARNING", "failure_reason": "blocked until full role eligibility recomputation"})
    return pd.DataFrame(rows)


def run_pipeline_if_allowed(data_root: Path, run_dir: Path, gate: pd.DataFrame, run_mode: str) -> tuple[pd.DataFrame, dict[str, Any], str | None]:
    experiments = [
        "001_data_contract_inventory.py", "002_role_eligibility_stable_metric_population.py", "003_feature_engineering_normalization.py", "004_role_specific_weight_estimation.py", "005_scientific_validation_calibration.py", "006_temporal_cross_competition_validation.py", "007_production_score_engine_framework.py", "008_full_population_recalibration.py",
    ]
    allowed = bool(gate.status.eq("PASS").all())
    if run_mode != "rerun_research_pipeline":
        rows = [{"experiment": e, "status": "not_requested", "return_code": None, "log_path": ""} for e in experiments]
        return pd.DataFrame(rows), {"run_mode": run_mode, "executed": False, "reason": "run mode did not request rerun"}, None
    if not allowed:
        reason = "Data readiness gate failed; Experiments 001–008 were not rerun."
        rows = [{"experiment": e, "status": "blocked_data_readiness_failed", "return_code": None, "log_path": ""} for e in experiments]
        return pd.DataFrame(rows), {"run_mode": run_mode, "executed": False, "reason": reason}, reason
    run_dir.mkdir(parents=True, exist_ok=True)
    status_rows = []
    failure = None
    for exp in experiments:
        cmd = ["uv", "run", "python", f"experiments/{exp}", "--data-root", str(data_root)]
        log = run_dir / f"{exp}.log"
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=1200)
        log.write_text("COMMAND: " + " ".join(cmd) + "\n\nSTDOUT\n" + proc.stdout + "\nSTDERR\n" + proc.stderr, encoding="utf-8")
        status_rows.append({"experiment": exp, "status": "PASS" if proc.returncode == 0 else "FAIL", "return_code": proc.returncode, "log_path": str(log)})
        if proc.returncode != 0:
            failure = f"{exp} failed with return code {proc.returncode}"
            break
    manifest = {"run_mode": run_mode, "executed": True, "run_dir": str(run_dir), "failure": failure, "experiments": status_rows}
    return pd.DataFrame(status_rows), manifest, failure


def write_contract(contract: dict[str, Any]) -> None:
    write_json(ROOT / "outputs/reports/009_full_population_data_contract.json", contract)
    md = "# Experiment 009 — Full Population Data Contract\n\n"
    for section, spec in contract.items():
        md += f"## {section}\n\n"
        if isinstance(spec, dict) and "required_fields" in spec:
            md += "Required fields:\n" + "\n".join(f"- {x}" for x in spec["required_fields"]) + "\n\n"
        else:
            md += "```json\n" + json.dumps(spec, indent=2) + "\n```\n\n"
    (ROOT / "outputs/reports/009_full_population_data_contract.md").write_text(md, encoding="utf-8")


def figures(discovery: pd.DataFrame, gaps: pd.DataFrame, gate: pd.DataFrame, reload_tasks: pd.DataFrame, pipeline_status: pd.DataFrame, summary: dict[str, int]) -> list[str]:
    fig_dir = ROOT / "outputs/figures"; fig_dir.mkdir(parents=True, exist_ok=True); paths=[]
    def save(name: str):
        plt.tight_layout(); plt.savefig(fig_dir/name, dpi=150); plt.close(); paths.append(str(Path("outputs/figures")/name))
    cov = pd.DataFrame([{"metric": k, "observed": v, "target": MIN_TARGETS.get(k, None)} for k, v in summary.items() if k in MIN_TARGETS])
    plt.figure(figsize=(10,5)); sns.barplot(data=cov, x="metric", y="observed"); plt.xticks(rotation=45); plt.title("009 data coverage summary"); save("009_data_coverage_summary.png")
    comp = discovery[["dataset_name", "competition_coverage", "season_coverage"]].melt("dataset_name")
    plt.figure(figsize=(10,5)); sns.barplot(data=comp, x="dataset_name", y="value", hue="variable"); plt.xticks(rotation=45); plt.title("009 competition/season coverage"); save("009_competition_season_coverage.png")
    plt.figure(figsize=(10,5)); sns.barplot(data=discovery, x="dataset_name", y="row_count"); plt.xticks(rotation=45); plt.title("009 dataset row counts"); save("009_dataset_row_counts.png")
    plt.figure(figsize=(10,5)); sns.countplot(data=gaps[gaps.status != "PASS"], y="gap_type"); plt.title("009 missing data gap summary"); save("009_missing_data_gap_summary.png")
    plt.figure(figsize=(9,5)); sns.countplot(data=gate, y="status"); plt.title("009 data readiness gate"); save("009_data_readiness_gate.png")
    plt.figure(figsize=(9,5)); sns.countplot(data=reload_tasks, x="priority", hue="blocking_status"); plt.title("009 reload task priority"); save("009_reload_task_priority.png")
    if not pipeline_status.empty:
        plt.figure(figsize=(9,5)); sns.countplot(data=pipeline_status, y="status"); plt.title("009 rerun pipeline status"); save("009_rerun_pipeline_status.png")
    return paths


def write_notebook() -> None:
    heads = ["# Experiment 009 — Full DataPlatform Reload & End-to-End Recalibration Orchestration", "## 1. Objective", "## 2. Why Experiment 009 Was Needed", "## 3. Current Data Coverage", "## 4. Expected Production Data Contract", "## 5. Missing Data Gaps", "## 6. Reload Plan", "## 7. Data Readiness Gate Result", "## 8. Rerun Pipeline Permission", "## 9. Rerun Pipeline Status", "## 10. Blockers", "## 11. Next Action Required", "## 12. Why Production Is Still Not Declared", "## 13. Recommended Experiment 010"]
    nb = nbf.v4.new_notebook(); nb.cells=[nbf.v4.new_markdown_cell(h+"\n\nReproducible Experiment 009 orchestration artefact." if h.startswith("##") else h) for h in heads]
    nb.cells += [nbf.v4.new_code_cell("import pandas as pd\npd.read_csv('outputs/tables/009_data_root_discovery.csv').head()"), nbf.v4.new_code_cell("pd.read_csv('outputs/tables/009_data_readiness_gate.csv').head()")]
    nbf.write(nb, ROOT/"notebooks/009_full_data_reload_orchestration.ipynb")


def append_methodology(report: dict[str, Any]) -> None:
    p=ROOT/"methodology.md"; text=p.read_text()
    if "## Experiment 009" in text: return
    sec=f"""
## Experiment 009 — {TITLE}

Date: {report['generated_at']}

### Objective
Create a reproducible DataPlatform reload and end-to-end rerun orchestration layer before any production-candidate bundle can be generated.

### Football Hypothesis
Production score recalibration requires a complete population across seasons, competitions, matches, events, lineups, and role-eligible players before statistical/football validation can be trusted.

### Dataset
Data root: `{report['data_root']}`. Current coverage: {report['coverage_summary']}.

### Normalization Used
No score normalization is recalculated in Experiment 009. The orchestration only controls whether Experiments 001–008 may rerun.

### Feature Selection
No new feature selection is performed. Required datasets and key fields are specified in the full-population data contract.

### Algorithms
Root discovery, full data contract validation, gap analysis, ID consistency audit, reload planning, readiness gate, safe rerun orchestration, and timestamped run manifest generation.

### Evaluation
Data readiness gate result: {report['data_readiness_result']}. Rerun pipeline executed: {report['rerun_pipeline_executed']}.

### Results
The current root remains below production target, so the rerun pipeline is blocked and no production coefficients or bundles are produced.

### Figures
Generated {report['figures_generated']} orchestration figures.

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
"""
    p.write_text(text.rstrip()+"\n\n"+sec.strip()+"\n", encoding="utf-8")


def update_readme() -> None:
    p=ROOT/"README.md"; text=p.read_text()
    if "experiments/009_full_data_reload_orchestration.py" not in text:
        text += "\n\n## Experiment 009\n\nFull DataPlatform reload and end-to-end recalibration orchestration. Run modes:\n\n```bash\ncd /home/platform/DataScienceResearch\nuv run python experiments/009_full_data_reload_orchestration.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse --run-mode audit_only\nuv run python experiments/009_full_data_reload_orchestration.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse --run-mode validate_loaded_root\nuv run python experiments/009_full_data_reload_orchestration.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse --run-mode rerun_research_pipeline\n```\n\nThe script audits the data root, writes the production data contract, produces missing-data gaps and reload tasks, applies a strict readiness gate, and only allows Experiments 001–008 to rerun when the gate passes. API integration and production deployment must wait until full-population recalibration passes.\n"
        p.write_text(text, encoding="utf-8")


def main() -> None:
    output_dirs()
    parser=argparse.ArgumentParser(); parser.add_argument("--data-root", default="/home/platform/DataPlatform/tmp/master_data_warehouse"); parser.add_argument("--target-root", default=""); parser.add_argument("--run-mode", choices=["audit_only","validate_loaded_root","rerun_research_pipeline"], default="audit_only")
    args=parser.parse_args(); data_root=Path(args.data_root).resolve()
    timestamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir=ROOT/"outputs/runs"/f"{timestamp}_full_recalibration"
    discovery=discover_root(data_root); discovery.to_csv(ROOT/"outputs/tables/009_data_root_discovery.csv", index=False)
    contract=build_contract(); write_contract(contract)
    summary=current_summary(discovery)
    gaps, quality, id_audit=gap_analysis(discovery, contract, summary)
    gaps.to_csv(ROOT/"outputs/tables/009_missing_data_gap_analysis.csv", index=False)
    quality.to_csv(ROOT/"outputs/tables/009_data_quality_issues.csv", index=False)
    id_audit.to_csv(ROOT/"outputs/tables/009_id_consistency_audit.csv", index=False)
    tasks, plan_md=reload_plan(); tasks.to_csv(ROOT/"outputs/tables/009_reload_tasks.csv", index=False); (ROOT/"outputs/reports/009_full_data_reload_plan.md").write_text(plan_md, encoding="utf-8")
    gate=readiness_gate(summary, gaps, id_audit); gate.to_csv(ROOT/"outputs/tables/009_data_readiness_gate.csv", index=False)
    pipeline_status, manifest, failure=run_pipeline_if_allowed(data_root, run_dir, gate, args.run_mode)
    pipeline_status.to_csv(ROOT/"outputs/tables/009_rerun_pipeline_status.csv", index=False)
    write_json(ROOT/"outputs/reports/009_rerun_pipeline_manifest.json", manifest)
    if failure or not bool(gate.status.eq("PASS").all()):
        reason = failure or "Data readiness gate failed; rerun pipeline blocked."
        (ROOT/"outputs/reports/009_pipeline_failure_reason.md").write_text("# Experiment 009 Pipeline Failure / Block Reason\n\n"+reason+"\n", encoding="utf-8")
    figs=figures(discovery, gaps, gate, tasks, pipeline_status, summary)
    write_notebook()
    readiness_result="PASS" if bool(gate.status.eq("PASS").all()) else "FAIL"
    report={"experiment_id":EXPERIMENT_ID,"title":TITLE,"generated_at":datetime.now(timezone.utc).isoformat(),"data_root":str(data_root),"target_root":args.target_root,"run_mode":args.run_mode,"coverage_summary":summary,"data_readiness_result":readiness_result,"rerun_pipeline_allowed":readiness_result=="PASS","rerun_pipeline_executed":bool(manifest.get("executed")),"production_coefficients_declared":False,"production_candidate_bundle_generated":False,"reports":["009_full_population_data_contract.json","009_full_population_data_contract.md","009_full_data_reload_plan.md","009_rerun_pipeline_manifest.json","009_pipeline_failure_reason.md"],"tables":["009_data_root_discovery.csv","009_missing_data_gap_analysis.csv","009_data_quality_issues.csv","009_id_consistency_audit.csv","009_reload_tasks.csv","009_data_readiness_gate.csv","009_rerun_pipeline_status.csv"],"figures_generated":len(figs),"figure_paths":figs,"blockers":gaps[gaps.status!="PASS"].gap_type.head(20).tolist()}
    write_json(ROOT/"outputs/reports/009_full_data_reload_orchestration.json", report)
    md=f"# Experiment 009 — {TITLE}\n\n## 1. Objective\nCreate the reload and rerun orchestration layer required before production-candidate generation.\n\n## 2. Why Experiment 009 was needed\nExperiment 008 found the local root below production target.\n\n## 3. Current data coverage\n```json\n{json.dumps(summary, indent=2)}\n```\n\n## 4. Expected production data contract\nSee `009_full_population_data_contract.md`.\n\n## 5. Missing data gaps\nSee `009_missing_data_gap_analysis.csv`.\n\n## 6. Reload plan\nSee `009_full_data_reload_plan.md` and `009_reload_tasks.csv`.\n\n## 7. Data readiness gate result\n{readiness_result}.\n\n## 8. Whether rerun pipeline was allowed\n{readiness_result == 'PASS'}.\n\n## 9. Rerun pipeline status if executed\nExecuted: {manifest.get('executed')}.\n\n## 10. Blockers\n" + "\n".join(f"- {b}" for b in report["blockers"]) + "\n\n## 11. Next action required\nReload DataPlatform to meet the production data contract, then run validate_loaded_root and rerun_research_pipeline.\n\n## 12. Why production is still not declared\nNo full-population gate pass and no rerun of Experiments 001–008.\n\n## 13. Recommended Experiment 010\nExecute or integrate the actual DataPlatform reload workflow.\n"
    (ROOT/"outputs/reports/009_full_data_reload_orchestration.md").write_text(md, encoding="utf-8")
    append_methodology(report); update_readme(); print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
