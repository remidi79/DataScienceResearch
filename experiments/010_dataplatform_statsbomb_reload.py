from __future__ import annotations

import argparse
import json
import shutil
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
EXPERIMENT_ID = "010"
TITLE = "DataPlatform StatsBomb Coverage Expansion & Reload Execution"
TARGETS = {
    "competition_metadata": {"min_rows": 1, "rel": "competition_metadata/competition_metadata.jsonl", "fields": ["competition_id", "competition_name"]},
    "season_metadata": {"min_rows": 2, "rel": "season_metadata/season_metadata.jsonl", "fields": ["season_id", "season_name", "competition_id"]},
    "silver_matches": {"min_rows": 100, "rel": "silver/silver_matches.jsonl", "fields": ["provider_id", "competition_id", "season_id", "match_date"]},
    "silver_lineups": {"min_rows": 3000, "rel": "silver/silver_lineups.jsonl", "fields": ["match_provider_id", "player_id", "player_name", "team_id"]},
    "silver_events": {"min_rows": 250000, "rel": "silver/silver_events.jsonl", "fields": ["match_id", "type", "team"]},
    "player_match_stats_direct": {"min_rows": 3000, "rel": "marts_v2/mart_statsbomb_player_match_stats_direct_v1.jsonl", "fields": ["statsbomb_player_id", "match_provider_id", "team_id", "competition_id", "season_id", "metrics"]},
    "team_match_stats_direct": {"min_rows": 200, "rel": "marts_v2/mart_statsbomb_team_match_stats_direct_v1.jsonl", "fields": ["match_provider_id", "team_id", "competition_id", "season_id", "metrics"]},
    "player_season_stats_direct": {"min_rows": 300, "rel": "marts_v2/mart_statsbomb_player_season_stats_direct_v1.jsonl", "fields": ["statsbomb_player_id", "team_id", "competition_id", "season_id", "metrics"]},
    "team_season_stats_direct": {"min_rows": 20, "rel": "marts_v2/mart_statsbomb_team_season_stats_direct_v1.jsonl", "fields": ["team_id", "competition_id", "season_id", "metrics"]},
    "player_metadata": {"min_rows": 1, "rel": "player_metadata/player_metadata.jsonl", "fields": ["player_id", "player_name"]},
    "team_metadata": {"min_rows": 1, "rel": "team_metadata/team_metadata.jsonl", "fields": ["team_id", "team_name"]},
}
MIN_GATE = {"competitions": 3, "seasons": 2, "matches": 100, "teams": 20, "player_match_rows": 3000, "player_season_rows": 300, "events": 250000, "lineups": 3000}


def ensure_dirs() -> None:
    for d in ["outputs/tables", "outputs/reports", "outputs/figures", "notebooks"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def safe_json(line: str) -> dict[str, Any] | None:
    try:
        return json.loads(line)
    except Exception:
        return None


def count_rows(path: Path, limit: int | None = None) -> tuple[int, list[dict[str, Any]]]:
    if not path.exists() or path.is_dir():
        return 0, []
    rows: list[dict[str, Any]] = []
    count = 0
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        count += 1
                        if limit is None or len(rows) < limit:
                            obj = safe_json(line)
                            if isinstance(obj, dict):
                                rows.append(obj)
                            elif isinstance(obj, list):
                                rows.extend([x for x in obj if isinstance(x, dict)][: max(0, (limit or 200) - len(rows))])
        elif path.suffix.lower() == ".csv":
            count = max(0, sum(1 for _ in path.open("rb")) - 1)
            if limit:
                rows = pd.read_csv(path, nrows=limit).to_dict(orient="records")
        elif path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                rows = [x for x in obj if isinstance(x, dict)]
            elif isinstance(obj, dict):
                rows = [obj]
            else:
                rows = []
            count = len(rows)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path, columns=None)
            count = len(df); rows = df.head(limit or 100).to_dict(orient="records")
    except Exception:
        return 0, []
    return count, rows


def flatten_keys(rows: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for r in rows[:200]:
        keys.update(r.keys())
        if isinstance(r.get("metrics"), dict):
            keys.update(f"metrics.{k}" for k in r["metrics"].keys())
        if isinstance(r.get("identity"), dict):
            keys.update(f"identity.{k}" for k in r["identity"].keys())
    return keys


def guess_dataset(path: Path) -> str:
    s = str(path).lower()
    if "player_match" in s or "player-stats" in s and "match" in s: return "player_match_stats_direct"
    if "team_match" in s or "team-stats" in s and "match" in s: return "team_match_stats_direct"
    if "player_season" in s: return "player_season_stats_direct"
    if "team_season" in s: return "team_season_stats_direct"
    if "lineup" in s: return "silver_lineups"
    if "event" in s: return "silver_events"
    if "match" in s: return "silver_matches"
    if "competition" in s: return "competition_metadata"
    if "season" in s: return "season_metadata"
    if "team" in s: return "team_metadata"
    if "player" in s: return "player_metadata"
    return "unknown"


def source_discovery(source_root: Path) -> pd.DataFrame:
    rows=[]
    suffixes={".jsonl", ".json", ".csv", ".parquet"}
    files=[p for p in source_root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes and ("statsbomb" in str(p).lower() or any(k in str(p).lower() for k in ["match", "event", "lineup", "player", "team", "competition", "season"]))]
    # cap not needed; current repo manageable
    for p in sorted(files):
        count, sample = count_rows(p, 200)
        keys = flatten_keys(sample)
        comp=set(); season=set(); match=set()
        for r in sample:
            for k in ["competition_id", "competition", "competition_name"]:
                if k in r and r[k] is not None: comp.add(str(r[k]))
            for k in ["season_id", "season", "season_name"]:
                if k in r and r[k] is not None: season.add(str(r[k]))
            for k in ["match_provider_id", "provider_id", "match_id", "master_match_id"]:
                if k in r and r[k] is not None: match.add(str(r[k]))
            if isinstance(r.get("identity"), dict):
                if r["identity"].get("match_id") is not None: match.add(str(r["identity"]["match_id"]))
        stat=p.stat()
        rows.append({"source_path": str(p), "dataset_guess": guess_dataset(p), "file_type": p.suffix.lower().lstrip('.'), "file_count": 1, "row_count_if_readable": count, "size_mb": round(stat.st_size/1024/1024, 3), "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(), "competition_count_if_detectable": len(comp), "season_count_if_detectable": len(season), "match_count_if_detectable": len(match), "columns": ";".join(sorted(list(keys))[:120]), "status": "readable" if count or sample else "unreadable_or_empty"})
    return pd.DataFrame(rows)


def map_sources(discovery: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for target, spec in TARGETS.items():
        cand=discovery[discovery.dataset_guess == target].copy()
        if cand.empty:
            rows.append({"target_dataset": target, "source_path": "", "source_format": "", "mapping_status": "missing_source", "required_fields_present": False, "missing_fields": ";".join(spec["fields"]), "transformation_needed": "source discovery required", "blocking_issue": "no candidate source found"})
            continue
        cand=cand.sort_values(["row_count_if_readable", "size_mb"], ascending=False)
        best=cand.iloc[0]
        cols=set(str(best.get("columns", "")).split(";"))
        missing=[]
        for f in spec["fields"]:
            variants={f, f"identity.{f}", f"metrics.{f}"}
            if not (variants & cols):
                missing.append(f)
        status="ready_to_load" if not missing and len(cand)==1 else ("duplicate_source_candidates" if len(cand)>1 and not missing else "missing_required_fields")
        rows.append({"target_dataset": target, "source_path": best.source_path, "source_format": best.file_type, "mapping_status": status, "required_fields_present": not missing, "missing_fields": ";".join(missing), "transformation_needed": "copy_to_compatible_layout" if status in {"ready_to_load", "duplicate_source_candidates"} else "schema_completion_required", "blocking_issue": "" if status in {"ready_to_load", "duplicate_source_candidates"} else "missing required fields"})
    return pd.DataFrame(rows)


def dry_run(mapping: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows=[]
    for r in mapping.itertuples(index=False):
        source=Path(r.source_path) if isinstance(r.source_path, str) and r.source_path else None
        count, sample = count_rows(source, 200) if source else (0, [])
        rows.append({"target_dataset": r.target_dataset, "source_path": r.source_path, "mapping_status": r.mapping_status, "would_load_rows": count, "missing_fields": r.missing_fields, "duplicate_source_candidates": r.mapping_status == "duplicate_source_candidates", "schema_conflict": bool(r.missing_fields), "id_inconsistency_risk": "unknown_until_execute" if count else "blocked_no_source", "ready_for_execute": r.mapping_status in {"ready_to_load", "duplicate_source_candidates"}})
    df=pd.DataFrame(rows)
    projection={
        "competitions": 1,
        "seasons": 1,
        "matches": int(df.loc[df.target_dataset=="silver_matches", "would_load_rows"].max() if (df.target_dataset=="silver_matches").any() else 0),
        "player_match_rows": int(df.loc[df.target_dataset=="player_match_stats_direct", "would_load_rows"].max() if (df.target_dataset=="player_match_stats_direct").any() else 0),
        "events": int(df.loc[df.target_dataset=="silver_events", "would_load_rows"].max() if (df.target_dataset=="silver_events").any() else 0),
        "lineups": int(df.loc[df.target_dataset=="silver_lineups", "would_load_rows"].max() if (df.target_dataset=="silver_lineups").any() else 0),
    }
    md="# Experiment 010 — Dry Run Reload Summary\n\n"
    for k, target in [("competitions",3),("seasons",2),("matches",100),("player_match_rows",3000),("events",250000),("lineups",3000)]:
        md += f"- Can reach {k} target? {'YES' if projection[k] >= target else 'NO'} (projected {projection[k]}, target {target})\n"
    return df, md


def materialize_metadata_from_sources(target_root: Path, loaded: dict[str, Path]) -> None:
    # Lightweight metadata derived from loaded core files for compatibility; no fake metrics.
    comp_rows=[]; season_rows=[]; team_rows=[]; player_rows=[]
    for name, p in loaded.items():
        _, sample = count_rows(p, 5000)
        for r in sample:
            cid=r.get("competition_id") or (r.get("identity") or {}).get("competition_id")
            sid=r.get("season_id") or (r.get("identity") or {}).get("season_id")
            cname=(r.get("identity") or {}).get("competition_name") or r.get("competition_name")
            sname=(r.get("identity") or {}).get("season_name") or r.get("season_name")
            if cid is not None: comp_rows.append({"competition_id": cid, "competition_name": cname})
            if sid is not None: season_rows.append({"season_id": sid, "season_name": sname, "competition_id": cid})
            tid=r.get("team_id") or (r.get("identity") or {}).get("team_id")
            tname=r.get("team_name") or (r.get("identity") or {}).get("team_name")
            if tid is not None: team_rows.append({"team_id": tid, "team_name": tname})
            pid=r.get("statsbomb_player_id") or r.get("player_id") or (r.get("identity") or {}).get("player_id")
            pname=r.get("player_name") or (r.get("identity") or {}).get("player_name")
            if pid is not None: player_rows.append({"player_id": pid, "player_name": pname})
    for rel, rows in [("competition_metadata/competition_metadata.jsonl", comp_rows), ("season_metadata/season_metadata.jsonl", season_rows), ("team_metadata/team_metadata.jsonl", team_rows), ("player_metadata/player_metadata.jsonl", player_rows)]:
        p=target_root/rel; p.parent.mkdir(parents=True, exist_ok=True)
        df=pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()
        with p.open("w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False, default=str)+"\n")


def execute_reload(mapping: pd.DataFrame, target_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_root.mkdir(parents=True, exist_ok=True)
    manifest=[]; quality=[]; loaded={}
    for r in mapping.itertuples(index=False):
        spec=TARGETS[r.target_dataset]
        target=target_root/spec["rel"]
        if r.mapping_status in {"ready_to_load", "duplicate_source_candidates", "missing_required_fields"} and r.source_path:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r.source_path, target)
            loaded[r.target_dataset]=target
            row_count, sample=count_rows(target, 1000)
            keys=flatten_keys(sample)
            dup=0
            if sample:
                df=pd.DataFrame(sample)
                subset=[c for c in ["statsbomb_player_id", "player_id", "team_id", "match_provider_id", "provider_id", "competition_id", "season_id"] if c in df.columns]
                dup=int(df.duplicated(subset=subset).sum()) if subset else 0
            manifest.append({"dataset": r.target_dataset, "target_path": str(target), "source_path": r.source_path, "row_count": row_count, "unique_ids": len(keys), "duplicate_count_sample": dup, "load_status": "loaded_with_schema_warnings" if r.missing_fields else "loaded"})
            quality.append({"dataset": r.target_dataset, "row_count": row_count, "null_counts_json": json.dumps({}), "duplicate_count_sample": dup, "missing_fields": r.missing_fields, "quality_status": "WARNING" if r.missing_fields else "PASS"})
        else:
            manifest.append({"dataset": r.target_dataset, "target_path": str(target), "source_path": r.source_path, "row_count": 0, "unique_ids": 0, "duplicate_count_sample": 0, "load_status": "blocked"})
            quality.append({"dataset": r.target_dataset, "row_count": 0, "null_counts_json": "{}", "duplicate_count_sample": 0, "missing_fields": r.missing_fields, "quality_status": "BLOCKED"})
    materialize_metadata_from_sources(target_root, loaded)
    return pd.DataFrame(manifest), pd.DataFrame(quality)


def dataset_path(target_root: Path, target: str) -> Path:
    return target_root / TARGETS[target]["rel"]


def coverage(target_root: Path) -> pd.DataFrame:
    rows=[]
    values={}
    for target, spec in TARGETS.items():
        p=dataset_path(target_root, target)
        n, sample=count_rows(p, 5000)
        df=pd.DataFrame(sample)
        comp=int(df.competition_id.nunique()) if "competition_id" in df.columns else 0
        season=int(df.season_id.nunique()) if "season_id" in df.columns else 0
        match=0
        for c in ["provider_id", "match_provider_id", "match_id"]:
            if c in df.columns: match=max(match, int(df[c].nunique()))
        team=0
        for c in ["team_id", "team_name"]:
            if c in df.columns: team=max(team, int(df[c].nunique()))
        player=0
        for c in ["statsbomb_player_id", "player_id", "player_name"]:
            if c in df.columns: player=max(player, int(df[c].nunique()))
        date_range=""
        if "match_date" in df.columns:
            d=pd.to_datetime(df.match_date, errors="coerce")
            if d.notna().any(): date_range=f"{d.min().date()}..{d.max().date()}"
        rows.append({"dataset": target, "path": str(p), "row_count": n, "competitions": comp, "seasons": season, "matches": match, "teams": team, "players": player, "date_range": date_range, "events_per_match": n/match if target=="silver_events" and match else None, "lineups_per_match": n/match if target=="silver_lineups" and match else None, "player_match_rows_per_match": n/match if target=="player_match_stats_direct" and match else None})
    return pd.DataFrame(rows)


def id_validation(target_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches=pd.DataFrame(count_rows(dataset_path(target_root,"silver_matches"), None)[1])
    events=pd.DataFrame(count_rows(dataset_path(target_root,"silver_events"), None)[1])
    lineups=pd.DataFrame(count_rows(dataset_path(target_root,"silver_lineups"), None)[1])
    pm=pd.DataFrame(count_rows(dataset_path(target_root,"player_match_stats_direct"), None)[1])
    tm=pd.DataFrame(count_rows(dataset_path(target_root,"team_match_stats_direct"), None)[1])
    rows=[]; orphans=[]; dups=[]
    match_ids=set(matches.get("provider_id", pd.Series(dtype=str)).astype(str)) if not matches.empty else set()
    checks=[("events_match_in_matches", events, "match_id"), ("lineups_match_in_matches", lineups, "match_provider_id"), ("player_stats_match_in_matches", pm, "match_provider_id")]
    for name, df, col in checks:
        if df.empty or col not in df.columns or not match_ids:
            rows.append({"check": name, "status": "BLOCKED", "issue_count": 0, "detail": "missing dataset or key"}); continue
        ids=set(df[col].astype(str)); miss=ids-match_ids
        rows.append({"check": name, "status": "PASS" if not miss else "FAIL", "issue_count": len(miss), "detail": ""})
        for x in list(miss)[:100]: orphans.append({"check": name, "orphan_id": x})
    for name, df, keys in [("duplicate_player_match", pm, ["statsbomb_player_id", "match_provider_id"]), ("duplicate_team_match", tm, ["team_id", "match_provider_id"]), ("duplicate_match", matches, ["provider_id"] )]:
        if df.empty or not all(k in df.columns for k in keys):
            dups.append({"check": name, "duplicate_count": 0, "status": "BLOCKED"})
        else:
            cnt=int(df.duplicated(subset=keys).sum())
            dups.append({"check": name, "duplicate_count": cnt, "status": "PASS" if cnt==0 else "FAIL"})
    return pd.DataFrame(rows), pd.DataFrame(orphans), pd.DataFrame(dups)


def readiness_gate(cov: pd.DataFrame, idv: pd.DataFrame) -> pd.DataFrame:
    summary={
        "competitions": int(cov.competitions.max()) if not cov.empty else 0,
        "seasons": int(cov.seasons.max()) if not cov.empty else 0,
        "matches": int(cov.matches.max()) if not cov.empty else 0,
        "teams": int(cov.teams.max()) if not cov.empty else 0,
        "player_match_rows": int(cov.loc[cov.dataset=="player_match_stats_direct", "row_count"].sum()),
        "player_season_rows": int(cov.loc[cov.dataset=="player_season_stats_direct", "row_count"].sum()),
        "events": int(cov.loc[cov.dataset=="silver_events", "row_count"].sum()),
        "lineups": int(cov.loc[cov.dataset=="silver_lineups", "row_count"].sum()),
    }
    rows=[]
    for k, req in MIN_GATE.items():
        obs=summary[k]; rows.append({"criterion": k, "observed": obs, "required": req, "status": "PASS" if obs>=req else "FAIL", "reason": "" if obs>=req else f"observed {obs} below required {req}"})
    critical=int((idv.status=="FAIL").sum()) if not idv.empty and "status" in idv.columns else 1
    rows.append({"criterion": "no_critical_id_consistency_failure", "observed": critical, "required": 0, "status": "PASS" if critical==0 else "FAIL", "reason": "" if critical==0 else "critical ID failures"})
    return pd.DataFrame(rows)


def run_exp009_validation(target_root: Path) -> tuple[pd.DataFrame, str]:
    cmd=["uv", "run", "python", "experiments/009_full_data_reload_orchestration.py", "--data-root", str(target_root), "--run-mode", "validate_loaded_root"]
    proc=subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=600)
    md="# Experiment 010 — Experiment 009 Validation Result\n\n"
    md += "Command:\n\n`" + " ".join(cmd) + "`\n\n"
    md += f"Return code: {proc.returncode}\n\n"
    md += "## Stdout\n\n```\n" + proc.stdout[-4000:] + "\n```\n\n## Stderr\n\n```\n" + proc.stderr[-4000:] + "\n```\n"
    status="PASS" if proc.returncode==0 else "FAIL"
    result=pd.DataFrame([{"command":" ".join(cmd), "return_code":proc.returncode, "status":status, "next_command_if_passes": f"uv run python experiments/009_full_data_reload_orchestration.py --data-root {target_root} --run-mode rerun_research_pipeline" if proc.returncode==0 else "blocked"}])
    return result, md


def figures(discovery, mapping, dry, manifest, cov, idv, gate):
    fd=ROOT/"outputs/figures"; fd.mkdir(parents=True, exist_ok=True); paths=[]
    def save(name): plt.tight_layout(); plt.savefig(fd/name, dpi=150); plt.close(); paths.append(str(Path("outputs/figures")/name))
    plt.figure(figsize=(10,5)); sns.countplot(data=discovery, y="dataset_guess"); plt.title("010 source discovery summary"); save("010_source_discovery_summary.png")
    plt.figure(figsize=(10,5)); sns.countplot(data=mapping, y="mapping_status"); plt.title("010 source to target mapping status"); save("010_source_to_target_mapping_status.png")
    plt.figure(figsize=(10,5)); sns.barplot(data=dry, x="target_dataset", y="would_load_rows"); plt.xticks(rotation=45); plt.title("010 dry run coverage projection"); save("010_dry_run_coverage_projection.png")
    plt.figure(figsize=(10,5)); sns.barplot(data=manifest, x="dataset", y="row_count"); plt.xticks(rotation=45); plt.title("010 loaded dataset row counts"); save("010_loaded_dataset_row_counts.png")
    comp=cov.melt(id_vars=["dataset"], value_vars=["row_count"]); plt.figure(figsize=(10,5)); sns.barplot(data=comp, x="dataset", y="value"); plt.xticks(rotation=45); plt.title("010 target coverage vs required"); save("010_target_coverage_vs_required.png")
    plt.figure(figsize=(8,4)); sns.countplot(data=idv, y="status"); plt.title("010 ID consistency status"); save("010_id_consistency_status.png")
    plt.figure(figsize=(8,4)); sns.countplot(data=gate, y="status"); plt.title("010 target readiness gate"); save("010_target_readiness_gate.png")
    return paths


def write_notebook():
    heads=["# Experiment 010 — DataPlatform StatsBomb Coverage Expansion & Reload Execution","## 1. Objective","## 2. Why Experiment 010 Was Needed","## 3. Current Root Limitations","## 4. Source Discovery Results","## 5. Source-to-Target Mapping","## 6. Dry-Run Reload Plan","## 7. Reload Execution Result","## 8. Target Root Coverage","## 9. ID Consistency Validation","## 10. Data Quality Issues","## 11. Experiment 009 Compatibility Result","## 12. Whether Target Root Is Ready","## 13. Exact Next Command","## 14. Why Production Is Still Not Declared","## 15. Recommended Experiment 011"]
    nb=nbf.v4.new_notebook(); nb.cells=[nbf.v4.new_markdown_cell(h+"\n\nReproducible Experiment 010 reload artefact." if h.startswith("##") else h) for h in heads]
    nb.cells += [nbf.v4.new_code_cell("import pandas as pd\npd.read_csv('outputs/tables/010_source_discovery.csv').head()"), nbf.v4.new_code_cell("pd.read_csv('outputs/tables/010_target_readiness_gate.csv').head()")]
    nbf.write(nb, ROOT/"notebooks/010_dataplatform_statsbomb_reload.ipynb")


def append_methodology(report):
    p=ROOT/"methodology.md"; txt=p.read_text()
    if "## Experiment 010" in txt: return
    sec=f"""
## Experiment 010 — {TITLE}

Date: {report['generated_at']}

### Objective
Build and validate the DataPlatform StatsBomb reload execution workflow required to satisfy the Experiment 009 production data contract.

### Football Hypothesis
The score engine cannot become production-candidate until DataPlatform contains enough matches, events, lineups, competitions, seasons, and provider stats to support role-specific validation.

### Dataset
Source root: `{report['source_root']}`. Target root: `{report['target_root']}`.

### Normalization Used
None. This is data reload orchestration, not scoring.

### Feature Selection
None. Required source-to-target mapping is defined for DataPlatform datasets.

### Algorithms
Source discovery, source-to-target mapping, dry-run projection, optional copy-based materialization, ID consistency checks, target coverage gate, and Experiment 009 compatibility invocation.

### Evaluation
Target readiness result: {report['target_readiness_result']}. Experiment 009 validation status: {report['experiment_009_validation_status']}.

### Results
Reload workflow artefacts were generated. Production is still not declared.

### Figures
Generated {report['figures_generated']} figures.

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
"""
    p.write_text(txt.rstrip()+"\n\n"+sec.strip()+"\n", encoding="utf-8")


def update_readme():
    p=ROOT/"README.md"; txt=p.read_text()
    if "experiments/010_dataplatform_statsbomb_reload.py" not in txt:
        txt += "\n\n## Experiment 010\n\nDataPlatform StatsBomb coverage expansion and reload execution workflow. Run modes:\n\n```bash\ncd /home/platform/DataScienceResearch\nuv run python experiments/010_dataplatform_statsbomb_reload.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode discover_sources\nuv run python experiments/010_dataplatform_statsbomb_reload.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode dry_run_reload\nuv run python experiments/010_dataplatform_statsbomb_reload.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode execute_reload\nuv run python experiments/010_dataplatform_statsbomb_reload.py --source-root /home/platform/DataPlatform --target-root /home/platform/DataPlatform/tmp/master_data_warehouse_full --run-mode validate_target\n```\n\nThis is a data reload workflow, not scoring production. It does not change coefficients, create production bundles, or start API integration. After validation, run Experiment 009 validate_loaded_root against the target root.\n"
        p.write_text(txt, encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--source-root", default="/home/platform/DataPlatform"); parser.add_argument("--target-root", default="/home/platform/DataPlatform/tmp/master_data_warehouse_full"); parser.add_argument("--run-mode", choices=["discover_sources","dry_run_reload","execute_reload","validate_target"], default="discover_sources")
    args=parser.parse_args(); ensure_dirs(); source_root=Path(args.source_root).resolve(); target_root=Path(args.target_root).resolve()
    discovery=source_discovery(source_root); discovery.to_csv(ROOT/"outputs/tables/010_source_discovery.csv", index=False)
    mapping=map_sources(discovery); mapping.to_csv(ROOT/"outputs/tables/010_source_to_target_mapping.csv", index=False)
    dry, dry_md=dry_run(mapping); dry.to_csv(ROOT/"outputs/tables/010_dry_run_reload_plan.csv", index=False); (ROOT/"outputs/reports/010_dry_run_reload_summary.md").write_text(dry_md, encoding="utf-8")
    manifest=pd.DataFrame(columns=["dataset","target_path","source_path","row_count","unique_ids","duplicate_count_sample","load_status"])
    quality=pd.DataFrame(columns=["dataset","row_count","null_counts_json","duplicate_count_sample","missing_fields","quality_status"])
    if args.run_mode == "execute_reload":
        manifest, quality=execute_reload(mapping, target_root)
    else:
        for r in mapping.itertuples(index=False):
            manifest.loc[len(manifest)]={"dataset": r.target_dataset, "target_path": str(target_root/TARGETS[r.target_dataset]["rel"]), "source_path": r.source_path, "row_count": 0, "unique_ids": 0, "duplicate_count_sample": 0, "load_status": "not_executed"}
            quality.loc[len(quality)]={"dataset": r.target_dataset, "row_count": 0, "null_counts_json": "{}", "duplicate_count_sample": 0, "missing_fields": r.missing_fields, "quality_status": "NOT_EXECUTED"}
    manifest.to_csv(ROOT/"outputs/tables/010_loaded_dataset_manifest.csv", index=False); quality.to_csv(ROOT/"outputs/tables/010_loaded_dataset_quality.csv", index=False)
    cov=coverage(target_root); cov.to_csv(ROOT/"outputs/tables/010_target_coverage_summary.csv", index=False)
    idv, orphans, dups=id_validation(target_root); idv.to_csv(ROOT/"outputs/tables/010_id_consistency_validation.csv", index=False); orphans.to_csv(ROOT/"outputs/tables/010_orphan_records.csv", index=False); dups.to_csv(ROOT/"outputs/tables/010_duplicate_records.csv", index=False)
    gate=readiness_gate(cov, idv); gate.to_csv(ROOT/"outputs/tables/010_target_readiness_gate.csv", index=False)
    exp009_result=pd.DataFrame([{"command": f"uv run python experiments/009_full_data_reload_orchestration.py --data-root {target_root} --run-mode validate_loaded_root", "return_code": None, "status": "not_run_in_mode", "next_command_if_passes": f"uv run python experiments/009_full_data_reload_orchestration.py --data-root {target_root} --run-mode rerun_research_pipeline"}])
    exp009_md="# Experiment 010 — Experiment 009 Validation Result\n\nNot executed in this run mode. Command documented:\n\n`uv run python experiments/009_full_data_reload_orchestration.py --data-root " + str(target_root) + " --run-mode validate_loaded_root`\n"
    if args.run_mode == "validate_target":
        exp009_result, exp009_md=run_exp009_validation(target_root)
    exp009_result.to_csv(ROOT/"outputs/tables/010_experiment_009_gate_result.csv", index=False); (ROOT/"outputs/reports/010_experiment_009_validation_result.md").write_text(exp009_md, encoding="utf-8")
    figs=figures(discovery, mapping, dry, manifest, cov, idv, gate)
    write_notebook()
    ready=bool(gate.status.eq("PASS").all()) if not gate.empty else False
    ok_mapping = mapping.mapping_status.isin(["ready_to_load", "duplicate_source_candidates"]) if not mapping.empty else pd.Series(dtype=bool)
    mapped_count = int(ok_mapping.sum()) if not mapping.empty else 0
    blocked_count = int((~ok_mapping).sum()) if not mapping.empty else 0
    report={"experiment_id":EXPERIMENT_ID,"title":TITLE,"generated_at":datetime.now(timezone.utc).isoformat(),"source_root":str(source_root),"target_root":str(target_root),"run_mode":args.run_mode,"source_discovery_count":int(len(discovery)),"mapped_datasets_count":mapped_count,"blocked_datasets_count":blocked_count,"target_readiness_result":"PASS" if ready else "FAIL","experiment_009_validation_status":str(exp009_result.status.iloc[0]) if not exp009_result.empty else "not_run","production_coefficients_declared":False,"production_candidate_bundle_generated":False,"figures_generated":len(figs),"next_command":f"uv run python experiments/009_full_data_reload_orchestration.py --data-root {target_root} --run-mode validate_loaded_root","blockers":gate[gate.status!="PASS"].criterion.tolist() if not gate.empty else []}
    (ROOT/"outputs/reports/010_dataplatform_statsbomb_reload.md").write_text(f"# Experiment 010 — {TITLE}\n\n## 1. Objective\nExecute/prepare the missing DataPlatform reload workflow.\n\n## 2. Why Experiment 010 was needed\nExperiment 009 showed the current root failed production coverage.\n\n## 3. Current root limitations\nInsufficient competitions, seasons, matches, player-match rows, events, and lineups.\n\n## 4. Source discovery results\nDiscovered {len(discovery)} candidate source files.\n\n## 5. Source-to-target mapping\nMapped {report['mapped_datasets_count']} datasets; blocked {report['blocked_datasets_count']}.\n\n## 6. Dry-run reload plan\nSee `010_dry_run_reload_plan.csv` and `010_dry_run_reload_summary.md`.\n\n## 7. Reload execution result if executed\nRun mode: {args.run_mode}.\n\n## 8. Target root coverage\nSee `010_target_coverage_summary.csv`.\n\n## 9. ID consistency validation\nSee `010_id_consistency_validation.csv`.\n\n## 10. Data quality issues\nSee loaded dataset quality and ID/orphan/duplicate tables.\n\n## 11. Experiment 009 compatibility result\n{report['experiment_009_validation_status']}.\n\n## 12. Whether the target root is ready\n{report['target_readiness_result']}.\n\n## 13. Exact next command\n`{report['next_command']}`\n\n## 14. Why production is still not declared\nThis experiment loads/validates data only; it does not score, change coefficients, create bundles, or deploy.\n\n## 15. Recommended Experiment 011\nProvider/API ingestion or upstream DataPlatform load for missing competitions and seasons if local sources cannot meet the target.\n", encoding="utf-8")
    with (ROOT/"outputs/reports/010_dataplatform_statsbomb_reload.json").open("w", encoding="utf-8") as f: json.dump(report, f, indent=2, ensure_ascii=False)
    append_methodology(report); update_readme(); print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__": main()
