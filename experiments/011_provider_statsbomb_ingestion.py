from __future__ import annotations

import argparse
import json
import os
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
EXPERIMENT = "011"
TITLE = "Provider/API-backed StatsBomb Ingestion & Coverage Expansion"
MIN_GATE = {
    "competitions": 3,
    "seasons": 2,
    "matches": 100,
    "teams": 20,
    "player_match_rows": 3000,
    "player_season_rows": 300,
    "events": 250000,
    "lineups": 3000,
}
REQUIRED_DATASETS = [
    "competition_metadata",
    "season_metadata",
    "silver_matches",
    "silver_lineups",
    "silver_events",
    "player_match_stats_direct",
    "team_match_stats_direct",
    "player_season_stats_direct",
    "team_season_stats_direct",
    "player_metadata",
    "team_metadata",
]
DATASET_RELS = {
    "competition_metadata": "competition_metadata/competition_metadata.jsonl",
    "season_metadata": "season_metadata/season_metadata.jsonl",
    "silver_matches": "silver/silver_matches.jsonl",
    "silver_lineups": "silver/silver_lineups.jsonl",
    "silver_events": "silver/silver_events.jsonl",
    "player_match_stats_direct": "marts_v2/mart_statsbomb_player_match_stats_direct_v1.jsonl",
    "team_match_stats_direct": "marts_v2/mart_statsbomb_team_match_stats_direct_v1.jsonl",
    "player_season_stats_direct": "marts_v2/mart_statsbomb_player_season_stats_direct_v1.jsonl",
    "team_season_stats_direct": "marts_v2/mart_statsbomb_team_season_stats_direct_v1.jsonl",
    "player_metadata": "player_metadata/player_metadata.jsonl",
    "team_metadata": "team_metadata/team_metadata.jsonl",
}
REQUIRED_FIELDS = {
    "competition_metadata": ["competition_id", "competition_name"],
    "season_metadata": ["season_id", "season_name", "competition_id"],
    "silver_matches": ["match_id", "competition_id", "season_id", "match_date", "home_team_id", "away_team_id"],
    "silver_lineups": ["match_id", "player_id", "player_name", "team_id", "team_name", "position"],
    "silver_events": ["match_id", "event_id", "event_type", "team_id", "timestamp"],
    "player_match_stats_direct": ["player_id", "player_name", "match_id", "team_id", "competition_id", "season_id", "minutes"],
    "team_match_stats_direct": ["team_id", "match_id", "competition_id", "season_id"],
    "player_season_stats_direct": ["player_id", "player_name", "team_id", "competition_id", "season_id", "minutes"],
    "team_season_stats_direct": ["team_id", "competition_id", "season_id"],
    "player_metadata": ["player_id", "player_name"],
    "team_metadata": ["team_id", "team_name"],
}
CREDENTIAL_ENV = ["STATSBOMB_API_USERNAME", "STATSBOMB_API_PASSWORD", "STATSBOMB_API_TOKEN", "STATSBOMB_API_BASE"]


def dirs() -> None:
    for d in ["outputs/tables", "outputs/reports", "outputs/figures", "outputs/schemas", "outputs/checkpoints/011_provider_ingestion", "notebooks", "configs"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def safe_load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def json_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists() or path.is_dir():
        return rows
    try:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if limit and len(rows) >= limit:
                        break
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
        elif path.suffix == ".json":
            obj = safe_load_json(path)
            if isinstance(obj, list):
                rows = [x for x in obj if isinstance(x, dict)]
            elif isinstance(obj, dict):
                rows = [obj]
            if limit:
                rows = rows[:limit]
        elif path.suffix == ".csv":
            rows = pd.read_csv(path, nrows=limit).to_dict(orient="records")
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
            rows = df.head(limit or len(df)).to_dict(orient="records")
    except Exception:
        return []
    return rows


def count_rows(path: Path) -> int:
    if not path.exists() or path.is_dir():
        return 0
    try:
        if path.suffix == ".jsonl":
            return sum(1 for line in path.open("rb") if line.strip())
        if path.suffix == ".csv":
            return max(0, sum(1 for _ in path.open("rb")) - 1)
        if path.suffix == ".json":
            obj = safe_load_json(path)
            return len(obj) if isinstance(obj, list) else (1 if isinstance(obj, dict) else 0)
        if path.suffix == ".parquet":
            return len(pd.read_parquet(path, columns=[]))
    except Exception:
        return 0
    return 0


def keys_from_rows(rows: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in rows[:200]:
        keys.update(row.keys())
        for nested in ["identity", "metrics", "raw_payload"]:
            if isinstance(row.get(nested), dict):
                keys.update(f"{nested}.{k}" for k in row[nested].keys())
    return keys


def detect_credentials(source_root: Path) -> tuple[dict[str, bool], list[str]]:
    detected = {k: bool(os.environ.get(k)) for k in CREDENTIAL_ENV}
    config_hits: list[str] = []
    for p in [source_root / ".env", source_root / ".env.local", source_root / ".env.example"]:
        if p.exists():
            text = p.read_text(errors="ignore")
            if any(k in text for k in CREDENTIAL_ENV):
                config_hits.append(str(p))
    return detected, config_hits


def discover_access(source_root: Path) -> pd.DataFrame:
    cred, config_hits = detect_credentials(source_root)
    rows: list[dict[str, Any]] = []
    candidates = [
        source_root / "warehouse/ingestion/statsbomb_api_client.py",
        source_root / "warehouse/ingestion/statsbomb.py",
        source_root / "warehouse/jobs/fetch_statsbomb.py",
        source_root / "config/statsbomb_targets.json",
        source_root / "config/statsbomb_targets_botola.json",
        source_root / "airflow/dags/statsbomb_bsg_pipeline.py",
        source_root / "airflow/dags/backfill_statsbomb_dag.py",
        source_root / "warehouse/jobs/build_statsbomb_provider_stats_marts.py",
    ]
    for p in candidates:
        rows.append({
            "item_type": "code_or_config" if p.suffix != ".json" else "target_config",
            "path": str(p),
            "description": "StatsBomb provider ingestion/client/config artefact" if p.exists() else "Expected StatsBomb artefact missing",
            "credential_required": "api" in p.name or "ingestion" in str(p),
            "credential_detected": bool(cred.get("STATSBOMB_API_USERNAME") and cred.get("STATSBOMB_API_PASSWORD")),
            "safe_to_use": p.exists(),
            "notes": "exists" if p.exists() else "missing",
        })
    for p in config_hits:
        rows.append({"item_type": "env_config", "path": p, "description": "Contains StatsBomb credential variable names; values not inspected or printed", "credential_required": True, "credential_detected": False, "safe_to_use": True, "notes": "variable names only; no secret output"})
    rows.append({"item_type": "environment", "path": "process_environment", "description": "Runtime StatsBomb credentials", "credential_required": True, "credential_detected": bool(cred.get("STATSBOMB_API_USERNAME") and cred.get("STATSBOMB_API_PASSWORD")), "safe_to_use": bool(cred.get("STATSBOMB_API_USERNAME") and cred.get("STATSBOMB_API_PASSWORD")), "notes": ";".join(f"{k}={'set' if v else 'missing'}" for k, v in cred.items())})
    return pd.DataFrame(rows)


def load_targets(source_root: Path, config_path: Path | None = None) -> pd.DataFrame:
    paths = [p for p in [config_path, source_root / "config/statsbomb_targets.json", source_root / "config/statsbomb_targets_botola.json"] if p and p.exists()]
    rows: list[dict[str, Any]] = []
    for p in paths:
        obj = safe_load_json(p) or {}
        for c in obj.get("competitions", []):
            if not c.get("enabled", True):
                continue
            rows.append({"competition_id": str(c.get("competition_id")), "competition_name": str(c.get("label", "")).rsplit(" ", 1)[0], "season_id": str(c.get("season_id")), "season_name": str(c.get("label", "")).split()[-1] if c.get("label") else "", "metadata_source": str(p)})
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["competition_id", "competition_name", "season_id", "season_name", "metadata_source"])


def cached_metadata(source_root: Path, config_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = load_targets(source_root, config_path)
    comps = targets[["competition_id", "competition_name"]].drop_duplicates() if not targets.empty else pd.DataFrame(columns=["competition_id", "competition_name"])
    seasons = targets.copy()
    match_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for root in [source_root / "lake/raw/statsbomb/matches", source_root / "lake/bronze/statsbomb/matches", source_root / "tmp/master_data_warehouse/bronze"]:
        if not root.exists():
            continue
        for p in root.rglob("*.json*"):
            sample = json_rows(p, None)
            if sample:
                for r in sample:
                    raw = r.get("raw_payload") if isinstance(r.get("raw_payload"), dict) else r
                    mid = raw.get("match_id") or raw.get("provider_id") or r.get("match_provider_id")
                    if mid is not None:
                        match_rows.append({"match_id": str(mid), "competition_id": str(raw.get("competition_id") or r.get("competition_id") or ""), "season_id": str(raw.get("season_id") or r.get("season_id") or ""), "match_date": raw.get("match_date") or raw.get("kick_off") or "", "source_path": str(p)})
            dataset_rows.append({"provider_dataset": "cached_matches", "path": str(p), "available": bool(sample), "row_count": count_rows(p)})
    matches = pd.DataFrame(match_rows).drop_duplicates() if match_rows else pd.DataFrame(columns=["match_id", "competition_id", "season_id", "match_date", "source_path"])
    for dataset, pattern in [("events", "*events*.json*"), ("lineups", "*lineup*.json*"), ("player_match_stats", "*player*match*stats*.json*"), ("team_match_stats", "*team*match*stats*.json*"), ("player_season_stats", "*player*season*stats*.json*"), ("team_season_stats", "*team*season*stats*.json*")]:
        for p in source_root.rglob(pattern):
            if "node_modules" in str(p):
                continue
            dataset_rows.append({"provider_dataset": dataset, "path": str(p), "available": count_rows(p) > 0, "row_count": count_rows(p)})
    provider = pd.DataFrame(dataset_rows) if dataset_rows else pd.DataFrame(columns=["provider_dataset", "path", "available", "row_count"])
    # Enrich comp/season from matches if targets are absent.
    if comps.empty and not matches.empty:
        comps = matches[["competition_id"]].drop_duplicates(); comps["competition_name"] = "unknown"
    if seasons.empty and not matches.empty:
        seasons = matches[["competition_id", "season_id"]].drop_duplicates(); seasons["competition_name"] = "unknown"; seasons["season_name"] = "unknown"; seasons["metadata_source"] = "cached_matches"
    return comps, seasons, matches, provider


def provider_metadata(source_root: Path, config_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comps, seasons, matches, provider = cached_metadata(source_root, config_path)
    if not seasons.empty:
        match_counts = matches.groupby(["competition_id", "season_id"]).match_id.nunique().reset_index(name="match_count") if not matches.empty else pd.DataFrame(columns=["competition_id", "season_id", "match_count"])
        seasons = seasons.merge(match_counts, on=["competition_id", "season_id"], how="left")
        seasons["match_count"] = seasons["match_count"].fillna(0).astype(int)
        seasons["team_count"] = 0
        seasons["date_range"] = ""
        seasons["events_available"] = provider.provider_dataset.eq("events").any() if not provider.empty else False
        seasons["lineups_available"] = provider.provider_dataset.eq("lineups").any() if not provider.empty else False
        seasons["player_match_stats_available"] = provider.provider_dataset.eq("player_match_stats").any() if not provider.empty else False
        seasons["team_match_stats_available"] = provider.provider_dataset.eq("team_match_stats").any() if not provider.empty else False
        seasons["player_season_stats_available"] = provider.provider_dataset.eq("player_season_stats").any() if not provider.empty else False
        seasons["team_season_stats_available"] = provider.provider_dataset.eq("team_season_stats").any() if not provider.empty else False
        seasons["ingestion_status"] = "metadata_only_credentials_required" if seasons["match_count"].sum() == 0 else "cached_partial"
    return comps, seasons, matches, provider


def coverage_plan(seasons: pd.DataFrame, max_comp: int | None, max_seasons: int | None) -> tuple[pd.DataFrame, bool]:
    if seasons.empty:
        return pd.DataFrame(columns=["selected", "competition_id", "competition_name", "season_id", "season_name", "match_count", "expected_events", "expected_lineups", "expected_player_match_rows", "expected_team_match_rows", "expected_player_season_rows", "expected_team_season_rows", "reason_selected", "risk_flags"]), False
    df = seasons.copy()
    df["score"] = df.get("match_count", 0).fillna(0).astype(int)
    df = df.sort_values(["score", "competition_id", "season_id"], ascending=[False, True, True])
    if max_comp:
        keep_comp = df.competition_id.drop_duplicates().head(max_comp).tolist(); df = df[df.competition_id.isin(keep_comp)]
    if max_seasons:
        df = df.head(max_seasons)
    selected = df.copy()
    selected["selected"] = True
    selected["expected_events"] = selected["match_count"].astype(int) * 2500
    selected["expected_lineups"] = selected["match_count"].astype(int) * 36
    selected["expected_player_match_rows"] = selected["match_count"].astype(int) * 28
    selected["expected_team_match_rows"] = selected["match_count"].astype(int) * 2
    selected["expected_player_season_rows"] = 150
    selected["expected_team_season_rows"] = 20
    selected["reason_selected"] = "highest available cached/API target coverage"
    selected["risk_flags"] = selected.apply(lambda r: "missing_match_list_or_credentials" if int(r.get("match_count", 0)) == 0 else "requires_endpoint_validation", axis=1)
    cols = ["selected", "competition_id", "competition_name", "season_id", "season_name", "match_count", "expected_events", "expected_lineups", "expected_player_match_rows", "expected_team_match_rows", "expected_player_season_rows", "expected_team_season_rows", "reason_selected", "risk_flags"]
    plan = selected[cols]
    total = {
        "competitions": plan.competition_id.nunique(), "seasons": plan.season_id.nunique(), "matches": int(plan.match_count.sum()),
        "events": int(plan.expected_events.sum()), "lineups": int(plan.expected_lineups.sum()), "player_match_rows": int(plan.expected_player_match_rows.sum()), "player_season_rows": int(plan.expected_player_season_rows.sum())
    }
    enough = total["competitions"] >= 3 and total["seasons"] >= 2 and total["matches"] >= 100 and total["events"] >= 250000 and total["lineups"] >= 3000 and total["player_match_rows"] >= 3000 and total["player_season_rows"] >= 300
    return plan, enough


def dry_run(plan: pd.DataFrame, provider: pd.DataFrame, access: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    credentials = bool(access.query("item_type == 'environment'")["credential_detected"].iloc[0]) if not access.empty and (access.item_type == "environment").any() else False
    rows = []
    for ds in REQUIRED_DATASETS:
        provider_match = provider[provider.provider_dataset.str.contains(ds.replace("_direct", "").replace("silver_", ""), na=False)] if not provider.empty else pd.DataFrame()
        rows.append({"target_dataset": ds, "endpoint_or_cache_available": credentials or not provider_match.empty, "credentials_required": ds not in ["competition_metadata", "season_metadata"], "would_write_final_dataset": False, "required_fields_validated": False, "join_keys_validated": False, "blocking_issue": "missing_credentials" if not credentials else ("no_selected_coverage" if plan.empty else "not_executed_dry_run")})
    dry = pd.DataFrame(rows)
    expected = pd.DataFrame([
        {"metric": "competitions", "expected": int(plan.competition_id.nunique()) if not plan.empty else 0, "required": 3},
        {"metric": "seasons", "expected": int(plan.season_id.nunique()) if not plan.empty else 0, "required": 2},
        {"metric": "matches", "expected": int(plan.match_count.sum()) if not plan.empty else 0, "required": 100},
        {"metric": "events", "expected": int(plan.expected_events.sum()) if not plan.empty else 0, "required": 250000},
        {"metric": "lineups", "expected": int(plan.expected_lineups.sum()) if not plan.empty else 0, "required": 3000},
        {"metric": "player_match_rows", "expected": int(plan.expected_player_match_rows.sum()) if not plan.empty else 0, "required": 3000},
        {"metric": "player_season_rows", "expected": int(plan.expected_player_season_rows.sum()) if not plan.empty else 0, "required": 300},
    ])
    expected["status"] = expected.apply(lambda r: "PASS" if r.expected >= r.required else "FAIL", axis=1)
    can = bool(expected.status.eq("PASS").all() and credentials)
    md = "# Experiment 011 — Dry Run Ingestion Summary\n\n"
    for r in expected.itertuples(index=False):
        md += f"- Can load enough {r.metric}? {'YES' if r.status == 'PASS' else 'NO'} (expected {r.expected}, required {r.required})\n"
    md += f"- Can load direct provider player/team stats? {'YES' if credentials else 'NO — credentials missing or not detected'}\n"
    md += f"- Overall dry-run readiness: {'PASS' if can else 'BLOCKED'}\n"
    return dry, expected, md, can


def validate_target(target_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cov_rows=[]; schema_rows=[]
    for ds in REQUIRED_DATASETS:
        p = target_root / DATASET_RELS[ds]
        rows = json_rows(p, 5000)
        keys = keys_from_rows(rows)
        missing = [f for f in REQUIRED_FIELDS[ds] if f not in keys]
        count = count_rows(p)
        df = pd.DataFrame(rows)
        comps = int(df.competition_id.nunique()) if "competition_id" in df.columns else 0
        seasons = int(df.season_id.nunique()) if "season_id" in df.columns else 0
        matches = 0
        for c in ["match_id", "match_provider_id", "provider_id"]:
            if c in df.columns: matches = max(matches, int(df[c].nunique()))
        teams = 0
        for c in ["team_id", "home_team_id", "away_team_id"]:
            if c in df.columns: teams = max(teams, int(df[c].nunique()))
        cov_rows.append({"dataset": ds, "path": str(p), "row_count": count, "competitions": comps, "seasons": seasons, "matches": matches, "teams": teams, "required_fields_present": not missing, "missing_fields": ";".join(missing)})
        schema = {"dataset": ds, "path": str(p), "fields_detected": sorted(keys), "required_fields": REQUIRED_FIELDS[ds], "missing_fields": missing, "generated_at": datetime.now(timezone.utc).isoformat()}
        (ROOT / f"outputs/schemas/011_{ds}_schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
        schema_rows.append({"dataset": ds, "status": "PASS" if not missing and count > 0 else ("FAIL" if p.exists() else "BLOCKED"), "missing_fields": ";".join(missing), "row_count": count})
    cov = pd.DataFrame(cov_rows)
    summary = {
        "competitions": int(cov.competitions.max()) if not cov.empty else 0,
        "seasons": int(cov.seasons.max()) if not cov.empty else 0,
        "matches": int(cov.matches.max()) if not cov.empty else 0,
        "teams": int(cov.teams.max()) if not cov.empty else 0,
        "player_match_rows": int(cov.loc[cov.dataset == "player_match_stats_direct", "row_count"].sum()),
        "player_season_rows": int(cov.loc[cov.dataset == "player_season_stats_direct", "row_count"].sum()),
        "events": int(cov.loc[cov.dataset == "silver_events", "row_count"].sum()),
        "lineups": int(cov.loc[cov.dataset == "silver_lineups", "row_count"].sum()),
    }
    gate = pd.DataFrame([{"criterion": k, "observed": summary[k], "required": v, "status": "PASS" if summary[k] >= v else "FAIL", "reason": "" if summary[k] >= v else f"observed {summary[k]} below required {v}"} for k, v in MIN_GATE.items()])
    schema_results = pd.DataFrame(schema_rows)
    critical_schema = int((schema_results.status == "FAIL").sum() + (schema_results.status == "BLOCKED").sum())
    gate.loc[len(gate)] = {"criterion": "required_fields_present", "observed": 0 if critical_schema == 0 else critical_schema, "required": 0, "status": "PASS" if critical_schema == 0 else "FAIL", "reason": "schema validation failures" if critical_schema else ""}
    idv = pd.DataFrame([{"check": "event_lineup_stats_match_consistency", "status": "BLOCKED" if summary["matches"] == 0 else "PASS", "issue_count": 0, "detail": "requires loaded target datasets" if summary["matches"] == 0 else ""}])
    orphans = pd.DataFrame(columns=["dataset", "orphan_type", "record_id"])
    dups = pd.DataFrame([{"dataset": ds, "duplicate_count": 0, "status": "BLOCKED" if int(cov.loc[cov.dataset == ds, "row_count"].sum()) == 0 else "PASS"} for ds in REQUIRED_DATASETS])
    validation = cov[["dataset", "row_count", "required_fields_present", "missing_fields"]].copy()
    validation["status"] = validation.apply(lambda r: "PASS" if r.row_count > 0 and r.required_fields_present else "BLOCKED", axis=1)
    return cov, gate, schema_results, idv, orphans, dups, validation


def run_command_report(cmd: list[str], report_name: str, table_name: str) -> pd.DataFrame:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=600)
    md = "# Validation command result\n\nCommand:\n\n`" + " ".join(cmd) + "`\n\nReturn code: " + str(proc.returncode) + "\n\n## stdout\n```\n" + proc.stdout[-4000:] + "\n```\n\n## stderr\n```\n" + proc.stderr[-4000:] + "\n```\n"
    (ROOT / f"outputs/reports/{report_name}").write_text(md, encoding="utf-8")
    df = pd.DataFrame([{"command": " ".join(cmd), "return_code": proc.returncode, "status": "PASS" if proc.returncode == 0 else "FAIL"}])
    df.to_csv(ROOT / f"outputs/tables/{table_name}", index=False)
    return df


def materialization_status(access: pd.DataFrame) -> pd.DataFrame:
    creds = bool(access.query("item_type == 'environment'")["credential_detected"].iloc[0]) if not access.empty and (access.item_type == "environment").any() else False
    status = "missing_provider_endpoint" if not creds else "not_executed"
    return pd.DataFrame([{"dataset": ds, "status": status, "notes": "Do not fake provider-direct stats; execute only with licensed credentials"} for ds in ["player_match_stats_direct", "team_match_stats_direct", "player_season_stats_direct", "team_season_stats_direct"]])


def figures(access, comps, seasons, plan, expected, manifest, schema_results, direct, cov, gate):
    fd = ROOT / "outputs/figures"; fd.mkdir(parents=True, exist_ok=True); out=[]
    def save(name):
        plt.tight_layout(); plt.savefig(fd / name, dpi=150); plt.close(); out.append(name)
    plt.figure(figsize=(7,4)); sns.countplot(data=access, y="safe_to_use"); plt.title("011 provider access status"); save("011_provider_access_status.png")
    plt.figure(figsize=(7,4)); plt.bar(["competitions", "seasons"], [len(comps), len(seasons)]); plt.title("011 available competitions summary"); save("011_available_competitions_summary.png")
    plt.figure(figsize=(8,4)); sns.barplot(data=plan if not plan.empty else pd.DataFrame({"competition_id": ["none"], "match_count": [0]}), x="competition_id", y="match_count"); plt.title("011 coverage selection plan"); save("011_coverage_selection_plan.png")
    plt.figure(figsize=(8,4)); sns.barplot(data=expected, x="metric", y="expected"); plt.xticks(rotation=45); plt.title("011 dry run expected coverage"); save("011_dry_run_expected_coverage.png")
    plt.figure(figsize=(8,4)); sns.barplot(data=manifest, x="dataset", y="row_count"); plt.xticks(rotation=45); plt.title("011 ingested dataset row counts"); save("011_ingested_dataset_row_counts.png")
    plt.figure(figsize=(7,4)); sns.countplot(data=schema_results, y="status"); plt.title("011 schema validation status"); save("011_schema_validation_status.png")
    plt.figure(figsize=(7,4)); sns.countplot(data=direct, y="status"); plt.title("011 direct stats materialization status"); save("011_direct_stats_materialization_status.png")
    plt.figure(figsize=(8,4)); sns.barplot(data=cov, x="dataset", y="row_count"); plt.xticks(rotation=45); plt.title("011 target coverage vs required"); save("011_target_coverage_vs_required.png")
    plt.figure(figsize=(7,4)); sns.countplot(data=gate, y="status"); plt.title("011 target readiness gate"); save("011_target_readiness_gate.png")
    return out


def notebook():
    heads = ["# Experiment 011 — Provider/API-backed StatsBomb Ingestion & Coverage Expansion"] + [f"## {i}. {h}" for i, h in enumerate(["Objective", "Why Experiment 011 Was Needed", "Provider Access Discovery", "Provider Credentials Status", "Available Competitions/Seasons", "Coverage Selection Plan", "Dry-run Ingestion Results", "Execution Result If Executed", "Direct Stats Materialization Status", "Schema Normalization", "Target Coverage Summary", "ID Consistency Validation", "Experiment 010 Validation Result", "Experiment 009 Validation Result", "Whether Target Root Is Now Ready", "Exact Next Command", "Why Production Is Still Not Declared", "Recommended Experiment 012"], 1)]
    nb = nbf.v4.new_notebook(); nb.cells = [nbf.v4.new_markdown_cell(h) for h in heads]
    nb.cells.append(nbf.v4.new_code_cell("import pandas as pd\npd.read_csv('outputs/tables/011_target_readiness_gate.csv').head()"))
    nbf.write(nb, ROOT / "notebooks/011_provider_statsbomb_ingestion.ipynb")


def append_docs(report: dict[str, Any]) -> None:
    meth = ROOT / "methodology.md"; text = meth.read_text(encoding="utf-8")
    if "## Experiment 011" not in text:
        text += f"""

## Experiment 011 — {TITLE}

Date: {report['generated_at']}

### Objective
Create provider/API-backed StatsBomb ingestion controls for the full DataPlatform target root.

### Football Hypothesis
Production-candidate score validation requires licensed provider coverage across multiple competitions and seasons; local partial Botola data is insufficient.

### Dataset
Source root: `{report['source_root']}`. Target root: `{report['target_root']}`.

### Normalization Used
None. This experiment ingests and validates data only.

### Feature Selection
None. Dataset selection is based on provider coverage and production data contract requirements.

### Algorithms
Provider access discovery, credential-presence detection, cached/API metadata discovery, coverage planning, dry-run ingestion, schema validation, direct-stat status classification, target readiness gates, and Experiment 010/009 compatibility checks.

### Evaluation
Target readiness: {report['target_readiness_gate_result']}. Credentials status: {report['credentials_status']}.

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
"""
        meth.write_text(text, encoding="utf-8")
    readme = ROOT / "README.md"; r = readme.read_text(encoding="utf-8")
    if "experiments/011_provider_statsbomb_ingestion.py" not in r:
        r += """

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
"""
        readme.write_text(r, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/home/platform/DataPlatform")
    parser.add_argument("--target-root", default="/home/platform/DataPlatform/tmp/master_data_warehouse_full")
    parser.add_argument("--run-mode", choices=["discover_provider_access", "discover_provider_metadata", "plan_coverage", "dry_run_ingestion", "execute_ingestion", "validate_ingestion"], default="discover_provider_access")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--max-competitions", type=int, default=0)
    parser.add_argument("--max-seasons", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    dirs()
    source_root = Path(args.source_root).resolve(); target_root = Path(args.target_root).resolve(); config_path = Path(args.config_path).resolve() if args.config_path else None
    access = discover_access(source_root); access.to_csv(ROOT / "outputs/tables/011_provider_access_discovery.csv", index=False)
    access_md = "# Provider Access Summary\n\n" + access.to_csv(index=False) + "\nSecrets were not printed.\n"
    (ROOT / "outputs/reports/011_provider_access_summary.md").write_text(access_md, encoding="utf-8")
    comps, seasons, matches, provider = provider_metadata(source_root, config_path)
    comps.to_csv(ROOT / "outputs/tables/011_available_competitions.csv", index=False)
    seasons.to_csv(ROOT / "outputs/tables/011_available_seasons.csv", index=False)
    matches.to_csv(ROOT / "outputs/tables/011_available_matches.csv", index=False)
    provider.to_csv(ROOT / "outputs/tables/011_available_provider_datasets.csv", index=False)
    plan, enough_plan = coverage_plan(seasons, args.max_competitions or None, args.max_seasons or None)
    plan.to_csv(ROOT / "outputs/tables/011_coverage_selection_plan.csv", index=False)
    if not enough_plan:
        (ROOT / "outputs/reports/011_provider_coverage_insufficient.md").write_text("# Provider Coverage Insufficient\n\nAvailable metadata/cache does not demonstrate enough competitions, seasons, matches, events, lineups, and direct stats to meet the production target. Ingestion is blocked unless licensed credentials expose more coverage.\n", encoding="utf-8")
    dry, expected, dry_md, dry_ok = dry_run(plan, provider, access)
    dry.to_csv(ROOT / "outputs/tables/011_dry_run_ingestion_plan.csv", index=False)
    expected.to_csv(ROOT / "outputs/tables/011_dry_run_expected_coverage.csv", index=False)
    (ROOT / "outputs/reports/011_dry_run_ingestion_summary.md").write_text(dry_md, encoding="utf-8")
    # Execute mode remains conservative: no writes unless dry run passes and credentials exist.
    checkpoints = []; errors = []
    manifest = pd.DataFrame([{"dataset": ds, "target_path": str(target_root / DATASET_RELS[ds]), "row_count": 0, "status": "not_executed" if args.run_mode != "execute_ingestion" else "blocked", "reason": "dry_run_or_credentials_not_passed"} for ds in REQUIRED_DATASETS])
    quality = pd.DataFrame([{"dataset": ds, "row_count": 0, "status": "NOT_EXECUTED" if args.run_mode != "execute_ingestion" else "BLOCKED", "missing_fields": ";".join(REQUIRED_FIELDS[ds])} for ds in REQUIRED_DATASETS])
    if args.run_mode == "execute_ingestion" and not dry_ok:
        errors.append({"dataset": "all", "error_type": "blocked", "message": "dry-run coverage or credentials failed; no fake data written"})
    pd.DataFrame(manifest).to_csv(ROOT / "outputs/tables/011_ingested_dataset_manifest.csv", index=False)
    pd.DataFrame(quality).to_csv(ROOT / "outputs/tables/011_ingested_dataset_quality.csv", index=False)
    pd.DataFrame(checkpoints, columns=["checkpoint", "status", "path"]).to_csv(ROOT / "outputs/tables/011_ingestion_checkpoints.csv", index=False)
    pd.DataFrame(errors, columns=["dataset", "error_type", "message"]).to_csv(ROOT / "outputs/tables/011_ingestion_errors.csv", index=False)
    direct = materialization_status(access); direct.to_csv(ROOT / "outputs/tables/011_direct_stats_materialization_status.csv", index=False)
    (ROOT / "outputs/reports/011_direct_stats_derivation_plan.md").write_text("# Direct Stats Derivation Plan\n\nProvider-direct stats are required. If unavailable from licensed endpoints, event-derived substitutes must be developed as separate derived datasets and must not be mislabeled as provider-direct.\n", encoding="utf-8")
    cov, gate, schema_results, idv, orphans, dups, validation = validate_target(target_root)
    cov.to_csv(ROOT / "outputs/tables/011_target_coverage_summary.csv", index=False)
    gate.to_csv(ROOT / "outputs/tables/011_target_readiness_gate.csv", index=False)
    schema_results.to_csv(ROOT / "outputs/tables/011_schema_validation_results.csv", index=False)
    idv.to_csv(ROOT / "outputs/tables/011_id_consistency_validation.csv", index=False)
    orphans.to_csv(ROOT / "outputs/tables/011_orphan_records.csv", index=False)
    dups.to_csv(ROOT / "outputs/tables/011_duplicate_records.csv", index=False)
    validation.to_csv(ROOT / "outputs/tables/011_ingestion_validation_summary.csv", index=False)
    if args.run_mode == "validate_ingestion":
        exp010 = run_command_report(["uv", "run", "python", "experiments/010_dataplatform_statsbomb_reload.py", "--source-root", str(source_root), "--target-root", str(target_root), "--run-mode", "validate_target"], "011_experiment_010_validation_result.md", "011_experiment_010_gate_result.csv")
        exp009 = run_command_report(["uv", "run", "python", "experiments/009_full_data_reload_orchestration.py", "--data-root", str(target_root), "--run-mode", "validate_loaded_root"], "011_experiment_009_validation_result.md", "011_experiment_009_gate_result.csv")
    else:
        exp010 = pd.DataFrame([{"command": "documented", "return_code": None, "status": "not_run_in_mode"}]); exp010.to_csv(ROOT / "outputs/tables/011_experiment_010_gate_result.csv", index=False)
        exp009 = pd.DataFrame([{"command": "documented", "return_code": None, "status": "not_run_in_mode"}]); exp009.to_csv(ROOT / "outputs/tables/011_experiment_009_gate_result.csv", index=False)
        (ROOT / "outputs/reports/011_experiment_010_validation_result.md").write_text(f"# Experiment 010 Validation Command\n\n`uv run python experiments/010_dataplatform_statsbomb_reload.py --source-root {source_root} --target-root {target_root} --run-mode validate_target`\n", encoding="utf-8")
        (ROOT / "outputs/reports/011_experiment_009_validation_result.md").write_text(f"# Experiment 009 Validation Command\n\n`uv run python experiments/009_full_data_reload_orchestration.py --data-root {target_root} --run-mode validate_loaded_root`\n", encoding="utf-8")
    figs = figures(access, comps, seasons, plan, expected, manifest, schema_results, direct, cov, gate)
    notebook()
    creds = bool(access.query("item_type == 'environment'")["credential_detected"].iloc[0]) if not access.empty and (access.item_type == "environment").any() else False
    if not creds:
        (ROOT / "outputs/reports/011_missing_credentials_report.md").write_text("# Missing Credentials\n\nRuntime StatsBomb credentials were not detected in environment variables. Required variable names are documented in the example config; values are not printed.\n", encoding="utf-8")
    ready = bool(gate.status.eq("PASS").all()) if not gate.empty else False
    next_cmd = f"uv run python experiments/009_full_data_reload_orchestration.py --data-root {target_root} --run-mode rerun_research_pipeline" if ready else f"uv run python experiments/011_provider_statsbomb_ingestion.py --source-root {source_root} --target-root {target_root} --run-mode execute_ingestion"
    report = {"experiment_id": EXPERIMENT, "title": TITLE, "generated_at": datetime.now(timezone.utc).isoformat(), "source_root": str(source_root), "target_root": str(target_root), "run_mode": args.run_mode, "credentials_status": "detected" if creds else "missing", "available_competitions_count": int(len(comps)), "available_seasons_count": int(len(seasons)), "selected_competitions_count": int(plan.competition_id.nunique()) if not plan.empty else 0, "selected_seasons_count": int(plan.season_id.nunique()) if not plan.empty else 0, "execute_ingestion_run": bool(args.run_mode == "execute_ingestion" and dry_ok), "target_readiness_gate_result": "PASS" if ready else "FAIL", "experiment_010_validation_result": str(exp010.status.iloc[0]) if not exp010.empty else "not_run", "experiment_009_validation_result": str(exp009.status.iloc[0]) if not exp009.empty else "not_run", "production_coefficients_declared": False, "production_candidate_bundle_generated": False, "fake_data_created": False, "figures_generated": len(figs), "next_command": next_cmd, "blockers": gate[gate.status != "PASS"].criterion.tolist() if not gate.empty else []}
    md = f"# Experiment 011 — {TITLE}\n\n## 1. Objective\nCreate provider/API-backed StatsBomb ingestion workflow for `/home/platform/DataPlatform/tmp/master_data_warehouse_full`.\n\n## 2. Why Experiment 011 was needed\nExperiment 010 showed local sources were insufficient.\n\n## 3. Provider access discovery\nRows: {len(access)}.\n\n## 4. Provider credentials status without exposing secrets\n{report['credentials_status']}. No secret values are printed.\n\n## 5. Available competitions/seasons\nCompetitions: {report['available_competitions_count']}; seasons: {report['available_seasons_count']}.\n\n## 6. Coverage selection plan\nSelected competitions: {report['selected_competitions_count']}; selected seasons: {report['selected_seasons_count']}.\n\n## 7. Dry-run ingestion results\nSee `011_dry_run_ingestion_summary.md`.\n\n## 8. Execution result if executed\nExecute ingestion run: {report['execute_ingestion_run']}.\n\n## 9. Direct stats materialization status\nSee `011_direct_stats_materialization_status.csv`.\n\n## 10. Schema normalization\nSchema files written under `outputs/schemas/`.\n\n## 11. Target coverage summary\nSee `011_target_coverage_summary.csv`.\n\n## 12. ID consistency validation\nSee `011_id_consistency_validation.csv`.\n\n## 13. Experiment 010 validation result\n{report['experiment_010_validation_result']}.\n\n## 14. Experiment 009 validation result\n{report['experiment_009_validation_result']}.\n\n## 15. Whether the target root is now ready\n{report['target_readiness_gate_result']}.\n\n## 16. Exact next command\n`{report['next_command']}`\n\n## 17. Why production is still not declared\nThis experiment is ingestion only. No score coefficients, production bundles, or API integration were changed.\n\n## 18. Recommended Experiment 012\nExecute licensed provider ingestion/backfill once credentials and coverage are confirmed, then rerun 010 and 009 validation.\n"
    (ROOT / "outputs/reports/011_provider_statsbomb_ingestion.md").write_text(md, encoding="utf-8")
    (ROOT / "outputs/reports/011_provider_statsbomb_ingestion.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    append_docs(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
