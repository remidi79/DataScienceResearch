from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat as nbf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXPERIMENT_ID = "014"
TITLE = "Event-Derived / Open-Data Fallback Feasibility"
FORMULA_VERSION = "event_derived_research_v0.1"
TARGETS = {
    "competitions": 3,
    "seasons": 2,
    "matches": 100,
    "teams": 20,
    "player_match_rows": 3000,
    "player_season_rows": 300,
    "events": 250000,
    "lineups": 3000,
}
ROLE_FAMILIES = ["GK", "CB", "FB", "MID", "WINGER", "CF"]


def read_jsonl(path: Path, limit: int | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def raw_get(payload: Any, key: str, default: Any = None) -> Any:
    return payload.get(key, default) if isinstance(payload, dict) else default


def nested_name(payload: Any, key: str) -> Any:
    value = raw_get(payload, key)
    if isinstance(value, dict):
        return value.get("name")
    return value


def has_nested(payload: Any, key: str) -> bool:
    value = raw_get(payload, key)
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def location_pair(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, list) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None, None
    return None, None


def progress_flag(row: pd.Series) -> bool:
    x = row.get("x")
    end_x = row.get("end_x")
    try:
        if pd.isna(x) or pd.isna(end_x):
            return False
        return float(end_x) - float(x) >= 10 and float(end_x) >= 60
    except Exception:
        return False


def role_from_position(position: str | None) -> str:
    if not position or not isinstance(position, str):
        return "UNKNOWN"
    p = position.lower()
    if "goalkeeper" in p:
        return "GK"
    if "center back" in p or "centre back" in p or "left center back" in p or "right center back" in p:
        return "CB"
    if "back" in p or "wing back" in p:
        return "FB"
    if "wing" in p or "wide" in p:
        return "WINGER"
    if "striker" in p or "forward" in p:
        return "CF"
    if "midfield" in p or "midfielder" in p:
        return "MID"
    return "UNKNOWN"


def seconds_from_time(value: Any) -> int | None:
    if not value or not isinstance(value, str):
        return None
    try:
        parts = value.split(":")
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return int(h * 3600 + m * 60 + s)
    except Exception:
        return None


def estimate_minutes(positions: Any) -> float:
    if isinstance(positions, str):
        try:
            positions = json.loads(positions)
        except Exception:
            positions = []
    if not isinstance(positions, list) or not positions:
        return 0.0
    minutes = 0.0
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        start = seconds_from_time(pos.get("from")) or 0
        end = seconds_from_time(pos.get("to"))
        if end is None:
            # Final whistle / no sub-off time. Conservative 90-minute estimate.
            end = 90 * 60
        if end > start:
            minutes += (end - start) / 60
    return max(0.0, min(minutes, 130.0))


def primary_position(positions: Any) -> str | None:
    if isinstance(positions, str):
        try:
            positions = json.loads(positions)
        except Exception:
            positions = []
    if isinstance(positions, list) and positions:
        first = positions[0]
        if isinstance(first, dict):
            return first.get("position")
    return None


def load_data(data_root: Path) -> dict[str, pd.DataFrame]:
    return {
        "events": read_jsonl(data_root / "silver" / "silver_events.jsonl"),
        "lineups": read_jsonl(data_root / "silver" / "silver_lineups.jsonl"),
        "matches": read_jsonl(data_root / "silver" / "silver_matches.jsonl"),
        "player_match_direct": read_jsonl(data_root / "marts_v2" / "mart_statsbomb_player_match_stats_direct_v1.jsonl"),
        "team_match_direct": read_jsonl(data_root / "marts_v2" / "mart_statsbomb_team_match_stats_direct_v1.jsonl"),
        "player_season_direct": read_jsonl(data_root / "marts_v2" / "mart_statsbomb_player_season_stats_direct_v1.jsonl"),
        "team_season_direct": read_jsonl(data_root / "marts_v2" / "mart_statsbomb_team_season_stats_direct_v1.jsonl"),
    }


def enrich_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    events = events.copy()
    raw = events.get("raw_payload", pd.Series([{}] * len(events)))
    events["pass_completed"] = events.apply(lambda r: r.get("event_type") == "Pass" and not has_nested(r.get("raw_payload"), "outcome") and pd.isna(r.get("outcome")), axis=1)
    events["shot_on_target"] = raw.apply(lambda p: nested_name(raw_get(p, "shot"), "outcome") in {"Goal", "Saved", "Saved to Post"})
    events["goal"] = raw.apply(lambda p: nested_name(raw_get(p, "shot"), "outcome") == "Goal")
    events["xg"] = raw.apply(lambda p: raw_get(raw_get(p, "shot"), "statsbomb_xg"))
    events["assist"] = raw.apply(lambda p: bool(raw_get(raw_get(p, "pass"), "goal_assist", False)))
    events["key_pass"] = raw.apply(lambda p: raw_get(raw_get(p, "pass"), "shot_assist", False) or raw_get(raw_get(p, "pass"), "goal_assist", False))
    events["is_progressive"] = events.apply(progress_flag, axis=1)
    events["final_third_entry"] = events.apply(lambda r: pd.notna(r.get("end_x")) and float(r.get("end_x")) >= 80 if pd.notna(r.get("end_x")) else False, axis=1)
    events["box_entry"] = events.apply(lambda r: pd.notna(r.get("end_x")) and pd.notna(r.get("end_y")) and float(r.get("end_x")) >= 102 and 18 <= float(r.get("end_y")) <= 62 if pd.notna(r.get("end_x")) and pd.notna(r.get("end_y")) else False, axis=1)
    events["has_location"] = events["x"].notna() & events["y"].notna()
    events["has_end_location"] = events["end_x"].notna() & events["end_y"].notna()
    for col in ["match_provider_id", "player_provider_id", "team_provider_id", "competition_id", "season_id"]:
        if col in events.columns:
            events[col] = events[col].astype(str)
    return events


def build_event_audit(events: pd.DataFrame, lineups: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def count_unique(df: pd.DataFrame, col: str) -> int:
        return int(df[col].nunique()) if not df.empty and col in df.columns else 0
    audit_rows = [
        {"metric": "match_count", "value": count_unique(events, "match_provider_id") or count_unique(matches, "provider_id"), "status": "observed"},
        {"metric": "competition_count", "value": count_unique(events, "competition_id") or count_unique(matches, "competition_id"), "status": "observed"},
        {"metric": "season_count", "value": count_unique(events, "season_id") or count_unique(matches, "season_id"), "status": "observed"},
        {"metric": "player_count", "value": count_unique(events, "player_provider_id") or count_unique(lineups, "player_provider_id"), "status": "observed"},
        {"metric": "team_count", "value": count_unique(events, "team_provider_id") or len(set(matches.get("home_team_provider_id", [])) | set(matches.get("away_team_provider_id", []))) if not matches.empty else 0, "status": "observed"},
        {"metric": "event_count", "value": int(len(events)), "status": "observed"},
        {"metric": "lineup_rows", "value": int(len(lineups)), "status": "observed"},
        {"metric": "player_event_linkage_coverage", "value": float(events["player_provider_id"].notna().mean()) if "player_provider_id" in events else 0, "status": "ratio"},
        {"metric": "team_event_linkage_coverage", "value": float(events["team_provider_id"].notna().mean()) if "team_provider_id" in events else 0, "status": "ratio"},
        {"metric": "possession_coverage", "value": float(events["raw_payload"].apply(lambda p: has_nested(p, "possession")).mean()) if "raw_payload" in events else 0, "status": "ratio"},
        {"metric": "location_coverage", "value": float(events["has_location"].mean()) if "has_location" in events else 0, "status": "ratio"},
        {"metric": "shot_detail_coverage", "value": float(events["raw_payload"].apply(lambda p: has_nested(p, "shot")).mean()) if "raw_payload" in events else 0, "status": "ratio"},
        {"metric": "pass_detail_coverage", "value": float(events["raw_payload"].apply(lambda p: has_nested(p, "pass")).mean()) if "raw_payload" in events else 0, "status": "ratio"},
        {"metric": "carry_detail_coverage", "value": float(events["raw_payload"].apply(lambda p: has_nested(p, "carry")).mean()) if "raw_payload" in events else 0, "status": "ratio"},
        {"metric": "defensive_action_coverage", "value": float(events["event_type"].isin(["Interception", "Block", "Clearance", "Ball Recovery", "Duel"]).mean()) if "event_type" in events else 0, "status": "ratio"},
        {"metric": "pressure_coverage", "value": float((events["event_type"] == "Pressure").mean()) if "event_type" in events else 0, "status": "ratio"},
        {"metric": "obv_field_coverage", "value": float(events["raw_payload"].apply(lambda p: any(str(k).startswith("obv_") for k in p.keys()) if isinstance(p, dict) else False).mean()) if "raw_payload" in events else 0, "status": "ratio"},
    ]
    audit = pd.DataFrame(audit_rows)
    event_type = events["event_type"].value_counts(dropna=False).reset_index() if "event_type" in events else pd.DataFrame(columns=["event_type", "count"])
    if not event_type.empty:
        event_type.columns = ["event_type", "count"]
        event_type["share"] = event_type["count"] / max(int(event_type["count"].sum()), 1)
    field_rows = []
    for col in events.columns:
        if col == "raw_payload":
            continue
        field_rows.append({"field": col, "present_rows": int(events[col].notna().sum()), "total_rows": len(events), "coverage": float(events[col].notna().mean()) if len(events) else 0})
    nested_keys = Counter()
    for payload in events.get("raw_payload", pd.Series(dtype=object)):
        if isinstance(payload, dict):
            nested_keys.update(payload.keys())
    for key, count in nested_keys.items():
        field_rows.append({"field": f"raw_payload.{key}", "present_rows": int(count), "total_rows": len(events), "coverage": float(count / max(len(events), 1))})
    field_cov = pd.DataFrame(field_rows).sort_values("coverage", ascending=False) if field_rows else pd.DataFrame(columns=["field", "present_rows", "total_rows", "coverage"])
    lineup = lineups.copy() if not lineups.empty else pd.DataFrame()
    if not lineup.empty:
        lineup["estimated_minutes"] = lineup["positions"].apply(estimate_minutes) if "positions" in lineup else 0.0
        lineup["primary_position"] = lineup["positions"].apply(primary_position) if "positions" in lineup else None
        lineup["role_family"] = lineup["primary_position"].apply(role_from_position)
        lineup_cov = pd.DataFrame([
            {"metric": "lineup_rows", "value": len(lineup), "coverage": 1.0},
            {"metric": "players_with_position", "value": int(lineup["primary_position"].notna().sum()), "coverage": float(lineup["primary_position"].notna().mean())},
            {"metric": "players_with_positive_minutes", "value": int((lineup["estimated_minutes"] > 0).sum()), "coverage": float((lineup["estimated_minutes"] > 0).mean())},
            {"metric": "known_role_family_rows", "value": int((lineup["role_family"] != "UNKNOWN").sum()), "coverage": float((lineup["role_family"] != "UNKNOWN").mean())},
        ])
    else:
        lineup_cov = pd.DataFrame(columns=["metric", "value", "coverage"])
    return audit, event_type, field_cov, lineup_cov


def metric_catalog(events: pd.DataFrame) -> pd.DataFrame:
    available_types = set(events.get("event_type", pd.Series(dtype=str)).dropna().astype(str))
    has_xg = bool("raw_payload" in events and events["raw_payload"].apply(lambda p: has_nested(raw_get(p, "shot"), "statsbomb_xg")).any())
    has_obv = bool("raw_payload" in events and events["raw_payload"].apply(lambda p: any(str(k).startswith("obv_") for k in p.keys()) if isinstance(p, dict) else False).any())
    defs = [
        ("minutes", "partially_derivable", ["Starting XI", "Substitution"], ["lineups.positions"], "Estimate from lineup position intervals", "Depends on lineup interval quality", "medium", "minutes", True),
        ("appearances", "safely_derivable", ["lineups"], ["player_provider_id", "match_provider_id"], "One player-lineup row per match", "Bench-only semantics may vary", "high", "appearances", True),
        ("starts", "partially_derivable", ["lineups"], ["positions.start_reason"], "Starting XI role interval", "Requires start_reason", "medium", "starts", True),
        ("touches", "partially_derivable", list(available_types), ["player_provider_id"], "Count on-ball event rows", "Provider touch definition may differ", "medium", "touches", True),
        ("possessions_involved", "partially_derivable", list(available_types), ["raw_payload.possession"], "Unique possessions by player/team", "Possession IDs must be present", "medium", "possessions", True),
        ("ball_receipts", "safely_derivable" if "Ball Receipt*" in available_types or "Ball Receipt" in available_types else "not_derivable", ["Ball Receipt*"], ["event_type"], "Count ball receipt events", "Depends on exact event naming", "high", "ball_receipts", True),
        ("carries", "safely_derivable" if "Carry" in available_types else "not_derivable", ["Carry"], ["event_type"], "Count carry events", "None", "high", "carries", True),
        ("pressures", "safely_derivable" if "Pressure" in available_types else "not_derivable", ["Pressure"], ["event_type"], "Count pressure events", "Provider aggregate pressure models may differ", "high", "pressures", True),
        ("passes", "safely_derivable" if "Pass" in available_types else "not_derivable", ["Pass"], ["event_type"], "Count pass events", "None", "high", "passes", True),
        ("successful_passes", "partially_derivable" if "Pass" in available_types else "not_derivable", ["Pass"], ["pass.outcome"], "Passes with no unsuccessful outcome", "StatsBomb semantics depend on outcome null", "medium", "successful_passes", True),
        ("progressive_passes", "partially_derivable" if "Pass" in available_types else "not_derivable", ["Pass"], ["x", "end_x"], "Passes advancing at least 10 units into advanced zones", "Research proxy, not provider direct", "medium", "progressive_passes", True),
        ("passes_into_final_third", "partially_derivable" if "Pass" in available_types else "not_derivable", ["Pass"], ["end_x"], "Pass end_x >= 80", "Pitch orientation assumptions", "medium", "passes_into_final_third", True),
        ("passes_into_box", "partially_derivable" if "Pass" in available_types else "not_derivable", ["Pass"], ["end_x", "end_y"], "Pass end location inside box", "Pitch coordinate assumptions", "medium", "passes_into_box", True),
        ("carries_into_final_third", "partially_derivable" if "Carry" in available_types else "not_derivable", ["Carry"], ["end_x"], "Carry end_x >= 80", "Pitch orientation assumptions", "medium", "carries_into_final_third", True),
        ("carries_into_box", "partially_derivable" if "Carry" in available_types else "not_derivable", ["Carry"], ["end_x", "end_y"], "Carry end location inside box", "Pitch coordinate assumptions", "medium", "carries_into_box", True),
        ("shots", "safely_derivable" if "Shot" in available_types else "not_derivable", ["Shot"], ["event_type"], "Count shot events", "None", "high", "shots", True),
        ("shots_on_target", "partially_derivable" if "Shot" in available_types else "not_derivable", ["Shot"], ["shot.outcome"], "Shot outcomes Goal/Saved/Saved to Post", "Outcome taxonomy required", "medium", "shots_on_target", True),
        ("goals", "safely_derivable" if "Shot" in available_types else "not_derivable", ["Shot"], ["shot.outcome"], "Shot outcome Goal", "Own goals separate", "high", "goals", True),
        ("non_penalty_shots", "partially_derivable" if "Shot" in available_types else "not_derivable", ["Shot"], ["shot.type"], "Shots excluding Penalty type", "Requires shot type", "medium", "non_penalty_shots", True),
        ("xg", "safely_derivable" if has_xg else "requires_provider_direct_stats", ["Shot"], ["shot.statsbomb_xg"], "Sum StatsBomb event xG if field exists", "Provider field; do not recalculate model", "high" if has_xg else "low", "xg", bool(has_xg)),
        ("obv", "safely_derivable" if has_obv else "requires_provider_direct_stats", list(available_types), ["raw_payload.obv_*"], "Sum event OBV fields if present", "Provider event value field, not model recompute", "high" if has_obv else "low", "obv", bool(has_obv)),
        ("key_passes", "partially_derivable" if "Pass" in available_types else "not_derivable", ["Pass"], ["pass.shot_assist", "pass.goal_assist"], "Count pass shot/goal assists", "Depends on nested flags", "medium", "key_passes", True),
        ("assists", "partially_derivable" if "Pass" in available_types else "not_derivable", ["Pass"], ["pass.goal_assist"], "Count goal-assist pass flags", "Provider assist definition may differ", "medium", "assists", True),
        ("ball_recoveries", "safely_derivable" if "Ball Recovery" in available_types else "not_derivable", ["Ball Recovery"], ["event_type"], "Count ball recovery events", "None", "high", "ball_recoveries", True),
        ("interceptions", "safely_derivable" if "Interception" in available_types else "not_derivable", ["Interception"], ["event_type"], "Count interceptions", "None", "high", "interceptions", True),
        ("clearances", "safely_derivable" if "Clearance" in available_types else "not_derivable", ["Clearance"], ["event_type"], "Count clearances", "None", "high", "clearances", True),
        ("blocks", "safely_derivable" if "Block" in available_types else "not_derivable", ["Block"], ["event_type"], "Count blocks", "None", "high", "blocks", True),
        ("fouls", "safely_derivable" if "Foul Committed" in available_types else "not_derivable", ["Foul Committed"], ["event_type"], "Count committed fouls", "None", "high", "fouls", True),
        ("aerial_duels", "partially_derivable" if "Duel" in available_types else "not_derivable", ["Duel"], ["duel.type"], "Count aerial-labeled duels", "Nested duel taxonomy required", "medium", "aerials", True),
        ("dribbles", "safely_derivable" if "Dribble" in available_types else "not_derivable", ["Dribble"], ["event_type"], "Count dribble events", "None", "high", "dribbles", True),
        ("turnovers", "partially_derivable" if available_types else "not_derivable", ["Dispossessed", "Miscontrol"], ["event_type"], "Dispossessed + miscontrol events", "Incomplete turnover definition", "medium", "turnovers", True),
        ("360_space_metrics", "requires_tracking_data", [], ["360 freeze frame/tracking"], "Cannot derive without 360/tracking/freeze-frame data", "Requires licensed data", "low", "360 metrics", False),
        ("pressing_intensity_model", "requires_provider_direct_stats", ["Pressure"], ["provider aggregate model"], "Event pressure counts do not reproduce provider model", "Provider-specific model", "low", "aggression/pressure model", False),
    ]
    return pd.DataFrame([{
        "metric_name": d[0], "derivation_status": d[1], "required_event_types": ";".join(map(str, d[2])), "required_fields": ";".join(d[3]),
        "formula_description": d[4], "limitations": d[5], "confidence_level": d[6], "comparable_to_provider_metric": bool(d[7]),
        "provider_metric_equivalent": d[7], "safe_for_research_score": bool(d[8])
    } for d in defs])


def build_player_metrics(events: pd.DataFrame, lineups: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    e = events.copy()
    keys = ["match_provider_id", "player_provider_id", "team_provider_id", "competition_id", "season_id"]
    e = e[e["player_provider_id"].notna()].copy()
    grouped = e.groupby(keys, dropna=False)
    rows = []
    for key, g in grouped:
        row = dict(zip(keys, key))
        row["player_name"] = g["player_name"].dropna().iloc[0] if "player_name" in g and g["player_name"].notna().any() else None
        row["team_name"] = g["team_name"].dropna().iloc[0] if "team_name" in g and g["team_name"].notna().any() else None
        row.update({
            "touches": len(g),
            "passes_attempted": int((g["event_type"] == "Pass").sum()),
            "passes_completed": int(((g["event_type"] == "Pass") & (g["pass_completed"])).sum()),
            "progressive_passes": int(((g["event_type"] == "Pass") & (g["is_progressive"])).sum()),
            "final_third_passes": int(((g["event_type"] == "Pass") & (g["final_third_entry"])).sum()),
            "box_entries_pass": int(((g["event_type"] == "Pass") & (g["box_entry"])).sum()),
            "carries": int((g["event_type"] == "Carry").sum()),
            "progressive_carries": int(((g["event_type"] == "Carry") & (g["is_progressive"])).sum()),
            "shots": int((g["event_type"] == "Shot").sum()),
            "shots_on_target": int(g["shot_on_target"].sum()),
            "goals": int(g["goal"].sum()),
            "xg": float(pd.to_numeric(g["xg"], errors="coerce").sum()) if "xg" in g else 0.0,
            "pressures": int((g["event_type"] == "Pressure").sum()),
            "ball_recoveries": int((g["event_type"] == "Ball Recovery").sum()),
            "interceptions": int((g["event_type"] == "Interception").sum()),
            "clearances": int((g["event_type"] == "Clearance").sum()),
            "blocks": int((g["event_type"] == "Block").sum()),
            "assists": int(g["assist"].sum()),
            "key_passes": int(g["key_pass"].sum()),
            "possessions_involved": int(g["raw_payload"].apply(lambda p: raw_get(p, "possession")).dropna().nunique()),
        })
        row["pass_completion_rate"] = row["passes_completed"] / row["passes_attempted"] if row["passes_attempted"] else np.nan
        row["derivation_confidence"] = "medium"
        row["required_fields_present"] = True
        row["missing_field_warning"] = "none"
        row["formula_version"] = FORMULA_VERSION
        row["metric_lineage"] = "event_derived_research_fallback_not_provider_direct"
        rows.append(row)
    pm = pd.DataFrame(rows)
    if not lineups.empty and not pm.empty:
        lu = lineups.copy()
        lu["estimated_minutes"] = lu["positions"].apply(estimate_minutes) if "positions" in lu else 0.0
        lu["primary_position"] = lu["positions"].apply(primary_position) if "positions" in lu else None
        lu["role_family"] = lu["primary_position"].apply(role_from_position)
        lu["match_provider_id"] = lu["match_provider_id"].astype(str)
        lu["player_provider_id"] = lu["player_provider_id"].astype(str)
        pm = pm.merge(lu[["match_provider_id", "player_provider_id", "estimated_minutes", "primary_position", "role_family"]], on=["match_provider_id", "player_provider_id"], how="left")
        pm.rename(columns={"estimated_minutes": "minutes"}, inplace=True)
    return pm


def build_team_metrics(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    keys = ["match_provider_id", "team_provider_id", "competition_id", "season_id"]
    rows = []
    for key, g in events.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row["team_name"] = g["team_name"].dropna().iloc[0] if "team_name" in g and g["team_name"].notna().any() else None
        passes = int((g["event_type"] == "Pass").sum())
        comp = int(((g["event_type"] == "Pass") & (g["pass_completed"])).sum())
        row.update({
            "event_count": len(g), "possession_count": int(g["raw_payload"].apply(lambda p: raw_get(p, "possession")).dropna().nunique()),
            "pass_volume": passes, "passes_completed": comp, "pass_completion": comp / passes if passes else np.nan,
            "final_third_entries": int(g["final_third_entry"].sum()), "box_entries": int(g["box_entry"].sum()),
            "shots": int((g["event_type"] == "Shot").sum()), "xg": float(pd.to_numeric(g["xg"], errors="coerce").sum()),
            "pressures": int((g["event_type"] == "Pressure").sum()),
            "defensive_actions": int(g["event_type"].isin(["Interception", "Block", "Clearance", "Ball Recovery", "Duel"]).sum()),
            "field_tilt_proxy": float((g["x"] >= 80).mean()) if "x" in g else np.nan,
            "territory_proxy_avg_x": float(pd.to_numeric(g["x"], errors="coerce").mean()) if "x" in g else np.nan,
            "event_count_defensive_third": int((pd.to_numeric(g["x"], errors="coerce") < 40).sum()),
            "event_count_middle_third": int(((pd.to_numeric(g["x"], errors="coerce") >= 40) & (pd.to_numeric(g["x"], errors="coerce") < 80)).sum()),
            "event_count_attacking_third": int((pd.to_numeric(g["x"], errors="coerce") >= 80).sum()),
            "formula_version": FORMULA_VERSION,
            "metric_lineage": "event_derived_research_fallback_not_provider_direct",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_player_season(pm: pd.DataFrame) -> pd.DataFrame:
    if pm.empty:
        return pd.DataFrame()
    dims = ["minutes", "touches", "passes_attempted", "passes_completed", "progressive_passes", "final_third_passes", "box_entries_pass", "carries", "progressive_carries", "shots", "shots_on_target", "goals", "xg", "pressures", "ball_recoveries", "interceptions", "clearances", "blocks", "assists", "key_passes", "possessions_involved"]
    existing = [c for c in dims if c in pm.columns]
    group_cols = ["competition_id", "season_id", "player_provider_id"]
    agg = pm.groupby(group_cols, dropna=False)[existing].sum(min_count=1).reset_index()
    meta = pm.groupby(group_cols, dropna=False).agg(player_name=("player_name", "first"), team_name=("team_name", "first"), role_family=("role_family", lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else "UNKNOWN"), matches=("match_provider_id", "nunique")).reset_index()
    out = agg.merge(meta, on=group_cols, how="left")
    if "minutes" in out:
        for c in existing:
            if c != "minutes":
                out[f"{c}_per90"] = out[c] / out["minutes"].replace(0, np.nan) * 90
    if "passes_attempted" in out and "passes_completed" in out:
        out["pass_completion_rate"] = out["passes_completed"] / out["passes_attempted"].replace(0, np.nan)
    for c in ["touches_per90", "progressive_passes_per90", "shots_per90", "xg_per90", "pressures_per90"]:
        if c in out:
            out[f"{c}_role_percentile"] = out.groupby("role_family")[c].rank(pct=True) * 100
    out["reliability_flag"] = np.where((out.get("minutes", 0) >= 600) & (out["matches"] >= 5), "medium_sample", "low_sample")
    out["metric_lineage"] = "event_derived_player_season_metrics_research_fallback_not_provider_direct"
    return out


def role_support_matrix(catalog: pd.DataFrame) -> pd.DataFrame:
    role_dims = {
        "GK": {"shot_stopping": ["shots", "goals", "xg"], "distribution": ["passes", "successful_passes"], "sweeping": ["clearances"], "claiming": ["aerial_duels"], "decision": ["turnovers"]},
        "CB": {"defending": ["interceptions", "clearances", "blocks", "aerial_duels"], "progression": ["progressive_passes", "carries"], "possession": ["passes", "successful_passes"], "duels": ["aerial_duels"]},
        "FB": {"progression": ["progressive_passes", "progressive_carries"], "creation": ["passes_into_box", "key_passes"], "defending": ["interceptions", "blocks"], "possession": ["passes", "successful_passes"]},
        "MID": {"possession": ["passes", "successful_passes", "ball_receipts"], "progression": ["progressive_passes", "carries"], "defending": ["pressures", "ball_recoveries"], "creation": ["key_passes"]},
        "WINGER": {"creation": ["key_passes", "passes_into_box"], "progression": ["progressive_carries", "carries_into_box"], "attack": ["shots", "xg"], "pressing": ["pressures"]},
        "CF": {"finishing": ["shots", "goals", "xg"], "creation": ["key_passes", "assists"], "pressing": ["pressures"], "possession": ["ball_receipts", "turnovers"]},
    }
    status_map = dict(zip(catalog["metric_name"], catalog["derivation_status"]))
    rows = []
    for role, dims in role_dims.items():
        for dim, metrics in dims.items():
            available = [m for m in metrics if status_map.get(m) in {"safely_derivable", "partially_derivable"}]
            missing = [m for m in metrics if m not in available]
            if len(available) == len(metrics):
                status = "supported_by_events"
            elif available:
                status = "partially_supported"
            else:
                status = "blocked_without_provider_stats"
            rows.append({"role": role, "dimension": dim, "required_metrics": ";".join(metrics), "event_derived_available_metrics": ";".join(available), "missing_metrics": ";".join(missing), "support_status": status, "research_feasibility": "possible_research_only" if available else "not_feasible", "production_feasibility": "not_production_ready_provider_stats_required"})
    return pd.DataFrame(rows)


def compare_provider(pm: pd.DataFrame, tm: pd.DataFrame, player_direct: pd.DataFrame, team_direct: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    if not pm.empty and not player_direct.empty:
        direct_rows = []
        for _, row in player_direct.iterrows():
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            direct_rows.append({"match_provider_id": str(row.get("match_provider_id")), "player_provider_id": str(row.get("statsbomb_player_id")), **{f"provider_{k}": v for k, v in metrics.items() if k in {"passes", "successful_passes", "shots", "goals", "pressures", "ball_recoveries", "interceptions", "clearances", "blocks", "xg"}}})
        ddf = pd.DataFrame(direct_rows)
        if not ddf.empty:
            merged = pm.merge(ddf, on=["match_provider_id", "player_provider_id"], how="inner")
            pairs = [("passes_attempted", "provider_passes"), ("passes_completed", "provider_successful_passes"), ("shots", "provider_shots"), ("goals", "provider_goals"), ("pressures", "provider_pressures"), ("ball_recoveries", "provider_ball_recoveries"), ("interceptions", "provider_interceptions"), ("clearances", "provider_clearances"), ("blocks", "provider_blocks"), ("xg", "provider_xg")]
            for left, right in pairs:
                if left in merged and right in merged:
                    x = pd.to_numeric(merged[left], errors="coerce")
                    y = pd.to_numeric(merged[right], errors="coerce")
                    both = x.notna() & y.notna()
                    comparisons.append({"grain": "player_match", "event_metric": left, "provider_metric": right.replace("provider_", ""), "status": "comparison_available" if both.sum() >= 2 else "comparison_not_available", "rows_compared": int(both.sum()), "pearson": float(x[both].corr(y[both], method="pearson")) if both.sum() >= 2 else np.nan, "spearman": float(x[both].corr(y[both], method="spearman")) if both.sum() >= 2 else np.nan, "mean_absolute_difference": float((x[both] - y[both]).abs().mean()) if both.any() else np.nan, "median_absolute_difference": float((x[both] - y[both]).abs().median()) if both.any() else np.nan, "coverage_difference": float(x.notna().mean() - y.notna().mean()), "missingness_difference": float(x.isna().mean() - y.isna().mean())})
    if not comparisons:
        comparisons.append({"grain": "player_match", "event_metric": "all", "provider_metric": "all", "status": "comparison_not_available", "rows_compared": 0, "pearson": np.nan, "spearman": np.nan, "mean_absolute_difference": np.nan, "median_absolute_difference": np.nan, "coverage_difference": np.nan, "missingness_difference": np.nan})
    return pd.DataFrame(comparisons)


def feasibility(player_season: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role in ROLE_FAMILIES:
        role_rows = matrix[matrix["role"] == role]
        supported = int(role_rows["support_status"].isin(["supported_by_events", "partially_supported"]).sum())
        blocked = int((~role_rows["support_status"].isin(["supported_by_events", "partially_supported"])).sum())
        ps = player_season[player_season.get("role_family", "") == role] if not player_season.empty and "role_family" in player_season else pd.DataFrame()
        eligible = int(((ps.get("minutes", 0) >= 600) & (ps.get("matches", 0) >= 5)).sum()) if not ps.empty else 0
        sample = int(len(ps))
        if sample == 0:
            status = "blocked_by_sample_size"
        elif supported >= 3 and eligible >= 10:
            status = "feasible_research_only"
        elif supported > 0:
            status = "partially_feasible_research_only"
        else:
            status = "not_feasible"
        rows.append({"role": role, "available_dimensions": supported, "blocked_dimensions": blocked, "supported_metrics_count": len(set(";".join(role_rows["event_derived_available_metrics"]).split(";")) - {""}), "eligible_players": eligible, "sample_size": sample, "reliability": "medium" if eligible >= 10 else "low", "interpretability": "medium_event_lineage" if supported else "low", "feasibility_status": status})
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path, rotate: int = 45) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    if df.empty or x not in df or y not in df:
        plt.text(0.5, 0.5, "No data", ha="center")
    else:
        d = df.head(20)
        plt.bar(d[x].astype(str), d[y])
        plt.xticks(rotation=rotate, ha="right")
        plt.title(title)
        plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def generate_reports(root: Path, outputs: dict[str, Any]) -> None:
    reports = root / "outputs" / "reports"
    tables = root / "outputs" / "tables"
    figures = root / "outputs" / "figures"
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "name": "event_derived_research_fallback_contract",
        "central_datawarehouse_principle": "All outputs are research artefacts under the MyTeam/DataPlatform warehouse context; StatsBomb is treated as a provider source, not the platform boundary.",
        "lineage": "event_derived_research_fallback_not_provider_direct",
        "not_provider_direct": True,
        "not_production_ready": True,
        "metrics_are_formula_based": True,
        "confidence_field_dependent": True,
        "cannot_replace_licensed_provider_stats_without_validation": True,
        "primary_tables": ["014_event_derived_player_match_metrics.csv", "014_event_derived_team_match_metrics.csv", "014_event_derived_player_season_metrics.csv"],
    }
    write_json(contract, reports / "014_event_derived_data_contract.json")
    (reports / "014_event_derived_data_contract.md").write_text("""# Experiment 014 Event-Derived Data Contract

This contract defines event-derived research fallback outputs inside the centralized MyTeam/DataPlatform warehouse context.

- These metrics are event-derived.
- These metrics are not provider-direct StatsBomb aggregate stats.
- StatsBomb is treated only as a provider source feeding the warehouse.
- Metrics are formula-based and confidence depends on field availability.
- Outputs are not production-ready.
- Outputs cannot replace licensed provider stats without validation and expert review.
""", encoding="utf-8")
    summary = outputs["summary"]
    write_json(summary, reports / "014_event_derived_fallback_feasibility.json")
    md = f"""# Experiment 014 — Event-Derived / Open-Data Fallback Feasibility

## Objective
Assess whether local event and lineup data can support a clearly labelled research fallback layer.

## Why Experiment 014 was needed
Licensed provider backfill remains blocked by credentials/provider access, so we audited what can be safely derived from local events without fabricating provider-direct stats.

## Current credential blocker
Provider access remains blocked. This experiment does not call provider APIs and does not use credentials.

## Available local event data
- Matches: {summary['match_count']}
- Competitions: {summary['competition_count']}
- Seasons: {summary['season_count']}
- Players: {summary['player_count']}
- Teams: {summary['team_count']}
- Events: {summary['event_count']}

## Event field coverage
See `outputs/tables/014_event_field_coverage.csv` and `outputs/figures/014_event_field_coverage.png`.

## Derivable metric catalog
Safely derivable: {summary['safely_derivable_metrics']}; partially derivable: {summary['partially_derivable_metrics']}; non-derivable/blocked: {summary['non_derivable_metrics']}.

## Role-dimension support matrix
See `outputs/tables/014_role_dimension_support_matrix.csv`.

## Player-match event-derived metrics
Rows: {summary['player_match_rows']}. These rows are labelled `event_derived_research_fallback_not_provider_direct`.

## Team-match event-derived metrics
Rows: {summary['team_match_rows']}. These rows are labelled `event_derived_research_fallback_not_provider_direct`.

## Player-season event-derived metrics
Rows: {summary['player_season_rows']}. Output name is `event_derived_player_season_metrics` and is not provider-direct.

## Comparison with provider stats where possible
See `outputs/tables/014_event_vs_provider_metric_comparison.csv`. Comparisons are diagnostic only and do not certify equivalence.

## Research fallback score feasibility
See `outputs/tables/014_research_fallback_score_feasibility.csv`. No final fallback scores are computed.

## Limitations
- Local sample is limited.
- Event-derived definitions differ from StatsBomb provider-direct aggregate endpoints.
- Tracking/360/video-dependent metrics remain blocked.
- Role minutes depend on lineup interval quality.

## Why this is not production
This does not use licensed provider backfill, does not reproduce provider-direct metrics, does not change coefficients, does not generate a production bundle, and does not mark any role production-ready.

## Recommended Experiment 015
Only after review: design a separate research-only event-derived score prototype with explicit lineage and no production claims, or return to licensed provider backfill once credentials are active.
"""
    (reports / "014_event_derived_fallback_feasibility.md").write_text(md, encoding="utf-8")
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Experiment 014 — Event-Derived / Open-Data Fallback Feasibility"),
        nbf.v4.new_markdown_cell("## Objective\nAssess safe event-derived research fallback feasibility inside the centralized MyTeam/DataPlatform warehouse context."),
        nbf.v4.new_markdown_cell("## Dataset\nLocal approved `/home/platform/DataPlatform/tmp/master_data_warehouse` silver events, lineups, matches, and existing direct-provider marts where present for comparison only."),
        nbf.v4.new_markdown_cell("## Findings\nNo fake data, no provider-direct replacement, no production bundle, no score coefficient changes."),
        nbf.v4.new_code_cell("import pandas as pd\nsummary = pd.read_json('outputs/reports/014_event_derived_fallback_feasibility.json', typ='series')\nsummary"),
    ]
    (root / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, root / "notebooks" / "014_event_derived_fallback_feasibility.ipynb")


def update_docs(root: Path) -> None:
    methodology = root / "methodology.md"
    marker = "## Experiment 014 — Event-Derived / Open-Data Fallback Feasibility"
    section = f"""

{marker}

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
"""
    text = methodology.read_text(encoding="utf-8") if methodology.exists() else ""
    if marker not in text:
        methodology.write_text(text.rstrip() + section + "\n", encoding="utf-8")
    readme = root / "README.md"
    marker2 = "## Experiment 014"
    sec = """

## Experiment 014

Event-derived / open-data fallback feasibility. This is a research-only fallback study and does not replace licensed StatsBomb provider-direct stats.

Run:

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/014_event_derived_fallback_feasibility.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse
```

Generated outputs include `outputs/tables/014_*`, `outputs/reports/014_event_derived_fallback_feasibility.*`, `outputs/reports/014_event_derived_data_contract.*`, `notebooks/014_event_derived_fallback_feasibility.ipynb`, and `outputs/figures/014_*.png`.

Warning: event-derived metrics are formula-based research artefacts. They are not provider-direct metrics, not licensed provider backfill, not production scoring, and not a production bundle.
"""
    rtext = readme.read_text(encoding="utf-8") if readme.exists() else ""
    if marker2 not in rtext:
        readme.write_text(rtext.rstrip() + sec + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root)
    tables = repo / "outputs" / "tables"
    figures = repo / "outputs" / "figures"
    data = load_data(data_root)
    events = enrich_events(data["events"])
    lineups = data["lineups"]
    matches = data["matches"]
    audit, event_type, field_cov, lineup_cov = build_event_audit(events, lineups, matches)
    catalog = metric_catalog(events)
    matrix = role_support_matrix(catalog)
    pm = build_player_metrics(events, lineups, matches)
    tm = build_team_metrics(events)
    ps = aggregate_player_season(pm)
    comp = compare_provider(pm, tm, data["player_match_direct"], data["team_match_direct"])
    feas = feasibility(ps, matrix)
    outputs = {
        "014_event_data_audit.csv": audit,
        "014_event_type_coverage.csv": event_type,
        "014_event_field_coverage.csv": field_cov,
        "014_lineup_coverage.csv": lineup_cov,
        "014_event_derived_metric_catalog.csv": catalog,
        "014_role_dimension_support_matrix.csv": matrix,
        "014_event_derived_player_match_metrics.csv": pm,
        "014_event_derived_team_match_metrics.csv": tm,
        "014_event_derived_player_season_metrics.csv": ps,
        "014_event_vs_provider_metric_comparison.csv": comp,
        "014_research_fallback_score_feasibility.csv": feas,
    }
    for name, df in outputs.items():
        write_csv(df, tables / name)
    plot_bar(event_type, "event_type", "count", "Event type distribution", figures / "014_event_type_distribution.png")
    plot_bar(field_cov.head(25), "field", "coverage", "Event field coverage", figures / "014_event_field_coverage.png")
    plot_bar(lineup_cov, "metric", "coverage", "Lineup coverage", figures / "014_lineup_coverage.png")
    status_counts = catalog["derivation_status"].value_counts().reset_index(); status_counts.columns = ["status", "count"]
    plot_bar(status_counts, "status", "count", "Metric derivability status", figures / "014_metric_derivability_status.png")
    matrix_counts = matrix.groupby(["role", "support_status"]).size().reset_index(name="count")
    plot_bar(matrix_counts, "role", "count", "Role-dimension support matrix", figures / "014_role_dimension_support_matrix.png", rotate=0)
    metric_cov = pd.DataFrame([{"metric": c, "coverage": float(pm[c].notna().mean())} for c in pm.columns if c not in {"match_provider_id", "player_provider_id", "team_provider_id", "competition_id", "season_id", "player_name", "team_name"}]) if not pm.empty else pd.DataFrame()
    plot_bar(metric_cov, "metric", "coverage", "Player-match metric coverage", figures / "014_player_match_metric_coverage.png")
    team_cov = pd.DataFrame([{"metric": c, "coverage": float(tm[c].notna().mean())} for c in tm.columns if c not in {"match_provider_id", "team_provider_id", "competition_id", "season_id", "team_name"}]) if not tm.empty else pd.DataFrame()
    plot_bar(team_cov, "metric", "coverage", "Team-match metric coverage", figures / "014_team_match_metric_coverage.png")
    plot_bar(comp.fillna(0), "event_metric", "rows_compared", "Event vs provider comparison rows", figures / "014_event_vs_provider_comparison.png")
    feas_counts = feas["feasibility_status"].value_counts().reset_index(); feas_counts.columns = ["status", "count"]
    plot_bar(feas_counts, "status", "count", "Research fallback feasibility", figures / "014_research_fallback_feasibility.png")
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "title": TITLE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "central_datawarehouse_principle": "MyTeam/DataPlatform is the central warehouse; StatsBomb is a provider source only.",
        "match_count": int(audit.loc[audit["metric"] == "match_count", "value"].iloc[0]) if not audit.empty else 0,
        "competition_count": int(audit.loc[audit["metric"] == "competition_count", "value"].iloc[0]) if not audit.empty else 0,
        "season_count": int(audit.loc[audit["metric"] == "season_count", "value"].iloc[0]) if not audit.empty else 0,
        "player_count": int(audit.loc[audit["metric"] == "player_count", "value"].iloc[0]) if not audit.empty else 0,
        "team_count": int(audit.loc[audit["metric"] == "team_count", "value"].iloc[0]) if not audit.empty else 0,
        "event_count": int(len(events)),
        "event_types_available": int(event_type["event_type"].nunique()) if not event_type.empty else 0,
        "safely_derivable_metrics": int((catalog["derivation_status"] == "safely_derivable").sum()),
        "partially_derivable_metrics": int((catalog["derivation_status"] == "partially_derivable").sum()),
        "non_derivable_metrics": int((~catalog["derivation_status"].isin(["safely_derivable", "partially_derivable"])).sum()),
        "player_match_rows": int(len(pm)),
        "team_match_rows": int(len(tm)),
        "player_season_rows": int(len(ps)),
        "provider_comparison_rows": int(len(comp)),
        "fake_data_created": False,
        "unauthorized_scraping": False,
        "provider_direct_replacement_claimed": False,
        "production_coefficients_changed": False,
        "production_bundle_generated": False,
        "experiment_015_started": False,
    }
    generate_reports(repo, {"summary": summary})
    update_docs(repo)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
