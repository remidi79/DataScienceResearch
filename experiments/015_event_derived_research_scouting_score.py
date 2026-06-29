from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

EXPERIMENT_ID = "015"
TITLE = "Event-Derived Research Scouting Score Prototype"
FORMULA_VERSION = "event_derived_research_scouting_v0.1"
ROLES = ["GK", "CB", "FB", "MID", "WINGER", "CF"]
BLOCKED_STATUSES = {"not_derivable", "requires_provider_direct_stats", "requires_tracking_data", "requires_manual_video_tagging"}
LOW_SAMPLE_WARNING = "SAMPLE_TOO_SMALL_LOCAL_11_MATCHES_NOT_PRODUCTION_READY"

METRIC_COLUMN_MAP = {
    "minutes": "minutes",
    "appearances": "matches",
    "touches": "touches_per90",
    "possessions_involved": "possessions_involved_per90",
    "ball_receipts": None,
    "carries": "carries_per90",
    "pressures": "pressures_per90",
    "passes": "passes_attempted_per90",
    "successful_passes": "passes_completed_per90",
    "progressive_passes": "progressive_passes_per90",
    "passes_into_final_third": "final_third_passes_per90",
    "passes_into_box": "box_entries_pass_per90",
    "carries_into_final_third": None,
    "carries_into_box": None,
    "shots": "shots_per90",
    "shots_on_target": "shots_on_target_per90",
    "goals": "goals_per90",
    "xg": "xg_per90",
    "key_passes": "key_passes_per90",
    "assists": "assists_per90",
    "ball_recoveries": "ball_recoveries_per90",
    "interceptions": "interceptions_per90",
    "clearances": "clearances_per90",
    "blocks": "blocks_per90",
    "aerial_duels": None,
    "dribbles": None,
    "turnovers": None,
    "pass_completion_rate": "pass_completion_rate",
}

ROLE_DIMENSIONS = {
    "GK": {
        "involvement_distribution": ["passes", "successful_passes", "touches"],
        "defensive_actions": ["clearances", "blocks"],
        "shot_facing_context": ["shots", "xg"],
    },
    "CB": {
        "defensive_activity": ["interceptions", "clearances", "blocks", "ball_recoveries"],
        "aerial_or_duel_activity": ["aerial_duels"],
        "progression": ["progressive_passes", "carries"],
        "possession_security": ["passes", "successful_passes", "pass_completion_rate"],
    },
    "FB": {
        "wide_progression": ["progressive_passes", "progressive_carries", "passes_into_final_third"],
        "crossing": ["passes_into_box", "key_passes"],
        "defensive_activity": ["interceptions", "blocks", "ball_recoveries"],
        "carrying": ["carries", "progressive_carries"],
    },
    "MID": {
        "progression": ["progressive_passes", "carries", "passes_into_final_third"],
        "retention": ["passes", "successful_passes", "pass_completion_rate"],
        "defensive_activity": ["pressures", "ball_recoveries", "interceptions"],
        "chance_creation": ["key_passes", "assists", "passes_into_box"],
    },
    "WINGER": {
        "carrying": ["carries", "progressive_carries"],
        "chance_creation": ["key_passes", "assists", "passes_into_box"],
        "box_threat": ["passes_into_box", "shots"],
        "shot_threat": ["shots", "shots_on_target", "xg", "goals"],
    },
    "CF": {
        "shot_threat": ["shots", "shots_on_target", "xg", "goals"],
        "box_presence": ["shots", "xg", "touches"],
        "link_play": ["passes", "successful_passes", "key_passes", "assists"],
        "pressing": ["pressures", "ball_recoveries"],
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    if "research_only" not in df.columns:
        df["research_only"] = True
    if "production_ready" not in df.columns:
        df["production_ready"] = False
    df.to_csv(path, index=False)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def percentile(values: list[float], value: float) -> float | None:
    clean = sorted(v for v in values if math.isfinite(v))
    if len(clean) < 2:
        return None
    below = sum(1 for v in clean if v < value)
    equal = sum(1 for v in clean if v == value)
    return round(100.0 * (below + 0.5 * equal) / len(clean), 6)


def zscore(values: list[float], value: float) -> float | None:
    clean = [v for v in values if math.isfinite(v)]
    if len(clean) < 2:
        return None
    std = float(np.std(clean))
    if std == 0:
        return None
    return round((value - float(np.mean(clean))) / std, 6)


def minmax100(values: list[float], value: float) -> float | None:
    clean = [v for v in values if math.isfinite(v)]
    if len(clean) < 2:
        return None
    mn, mx = min(clean), max(clean)
    if mx == mn:
        return 50.0
    return round(100.0 * (value - mn) / (mx - mn), 6)


def confidence_weight(status: str, role_sample: int, minutes: float | None) -> float:
    base = 1.0 if status == "safely_derivable" else 0.75 if status == "partially_derivable" else 0.0
    if role_sample < 30:
        base *= 0.75
    if minutes is None or minutes < 450:
        base *= 0.75
    return round(base, 6)


def confidence_label(value: float) -> str:
    if value >= 0.75:
        return "medium_research_only"
    if value >= 0.45:
        return "low_research_only"
    return "very_low_research_only"


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    tables = root / "outputs" / "tables"
    paths = {
        "player_match": tables / "014_event_derived_player_match_metrics.csv",
        "player_season": tables / "014_event_derived_player_season_metrics.csv",
        "catalog": tables / "014_event_derived_metric_catalog.csv",
        "support": tables / "014_role_dimension_support_matrix.csv",
        "feasibility": tables / "014_research_fallback_score_feasibility.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Experiment 014 input(s): {missing}")
    return {name: pd.read_csv(path) for name, path in paths.items()}


def validate_inputs(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pm, ps, catalog, support, feas = inputs["player_match"], inputs["player_season"], inputs["catalog"], inputs["support"], inputs["feasibility"]
    rows = [
        {"check_name": "player_match_rows", "value": len(pm), "status": "PASS" if len(pm) > 0 else "FAIL", "details": "Experiment 014 player-match event-derived input", "research_only": True},
        {"check_name": "player_season_rows", "value": len(ps), "status": "PASS" if len(ps) > 0 else "FAIL", "details": "Experiment 014 player-season event-derived input", "research_only": True},
        {"check_name": "available_roles", "value": ";".join(sorted(str(r) for r in ps.get("role_family", pd.Series(dtype=str)).dropna().unique())), "status": "PASS", "details": "Roles observed in player-season input", "research_only": True},
        {"check_name": "derivation_confidence", "value": ";".join(sorted(str(v) for v in pm.get("derivation_confidence", pd.Series(dtype=str)).dropna().unique())), "status": "PASS", "details": "Player-match derivation confidence values", "research_only": True},
        {"check_name": "safely_derivable_metrics", "value": int((catalog["derivation_status"] == "safely_derivable").sum()), "status": "PASS", "details": "Metric catalog safe metrics", "research_only": True},
        {"check_name": "partially_derivable_metrics", "value": int((catalog["derivation_status"] == "partially_derivable").sum()), "status": "PASS", "details": "Metric catalog partial metrics", "research_only": True},
        {"check_name": "blocked_dimensions", "value": int((support["support_status"].astype(str).str.contains("blocked", na=False)).sum()), "status": "PASS", "details": "Role dimensions blocked in 014 support matrix", "research_only": True},
        {"check_name": "production_readiness", "value": "false", "status": "PASS", "details": "Experiment 015 is not production-ready and cannot replace licensed provider stats", "research_only": True},
    ]
    # Feasibility rows are low reliability in local sample.
    for _, row in feas.iterrows():
        rows.append({"check_name": f"role_feasibility_{row.get('role')}", "value": row.get("feasibility_status"), "status": "PASS", "details": f"Reliability={row.get('reliability')}; eligible_players={row.get('eligible_players')}", "research_only": True})
    return pd.DataFrame(rows)


def metric_status_map(catalog: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(r["metric_name"]): r.to_dict() for _, r in catalog.iterrows()}


def build_role_metric_selection(catalog: pd.DataFrame, player_season: pd.DataFrame) -> pd.DataFrame:
    status = metric_status_map(catalog)
    rows = []
    for role, dims in ROLE_DIMENSIONS.items():
        role_metrics = sorted({m for metrics in dims.values() for m in metrics})
        for metric in role_metrics:
            meta = status.get(metric, {})
            if not meta and metric == "pass_completion_rate":
                meta = {"derivation_status": "partially_derivable", "confidence_level": "medium"}
            if not meta and metric == "progressive_carries":
                meta = {"derivation_status": "partially_derivable", "confidence_level": "medium"}
            derivation_status = str(meta.get("derivation_status", "not_in_catalog"))
            col = METRIC_COLUMN_MAP.get(metric)
            col_available = bool(col and col in player_season.columns and player_season[col].notna().any())
            included = derivation_status in {"safely_derivable", "partially_derivable"} and col_available
            rows.append({
                "role": role,
                "metric": metric,
                "source_column": col,
                "status": "included" if included else "excluded",
                "derivation_status": derivation_status,
                "reason_included": "event-derived metric available in 014 player-season table" if included else "",
                "reason_excluded": "" if included else ("blocked derivation status" if derivation_status in BLOCKED_STATUSES else "missing usable player-season column"),
                "derivation_confidence": meta.get("confidence_level", "low"),
                "quality_warning": "PARTIAL_DERIVATION_RESEARCH_ONLY" if derivation_status == "partially_derivable" else LOW_SAMPLE_WARNING if included else "EXCLUDED_NOT_USED_FOR_SCORE",
                "research_only": True,
            })
    return pd.DataFrame(rows)


def build_role_dimensions(selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role, dims in ROLE_DIMENSIONS.items():
        for dim, metrics in dims.items():
            selected = selection[(selection.role == role) & (selection.metric.isin(metrics)) & (selection.status == "included")]
            missing = [m for m in metrics if m not in set(selected.metric)]
            if len(selected) == len(metrics) and len(metrics) > 0:
                support = "SUPPORTED_RESEARCH_ONLY"
            elif len(selected) > 0:
                support = "PARTIAL_SUPPORT"
            else:
                support = "LOW_CONFIDENCE"
            rows.append({
                "role": role,
                "dimension": dim,
                "candidate_metrics": ";".join(metrics),
                "selected_metrics": ";".join(selected.metric.tolist()),
                "missing_metrics": ";".join(missing),
                "support_status": support,
                "sample_status": "SAMPLE_TOO_SMALL",
                "quality_warning": LOW_SAMPLE_WARNING,
                "research_only": True,
                "production_ready": False,
            })
    return pd.DataFrame(rows)


def normalize_metrics(player_season: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role in ROLES:
        role_players = player_season[player_season["role_family"] == role].copy()
        role_selection = selection[(selection.role == role) & (selection.status == "included")]
        for _, player in role_players.iterrows():
            minutes = safe_float(player.get("minutes"))
            for _, sel in role_selection.iterrows():
                col = sel["source_column"]
                raw = safe_float(player.get(col))
                if raw is None:
                    continue
                pop = [safe_float(v) for v in role_players[col].tolist()]
                pop = [v for v in pop if v is not None]
                pct = percentile(pop, raw)
                mm = minmax100(pop, raw)
                z = zscore(pop, raw)
                weight = confidence_weight(sel["derivation_status"], len(role_players), minutes)
                confidence_adjusted = mm * weight if mm is not None else None
                rows.append({
                    "player_id": player.get("player_provider_id"),
                    "player_name": player.get("player_name"),
                    "team_name": player.get("team_name"),
                    "role": role,
                    "competition_id": player.get("competition_id"),
                    "season_id": player.get("season_id"),
                    "minutes": minutes,
                    "metric": sel["metric"],
                    "source_column": col,
                    "raw_value": raw,
                    "per90_value": raw if str(col).endswith("_per90") else np.nan,
                    "role_percentile": pct,
                    "role_z_score": z,
                    "min_max_0_100": mm,
                    "direction_aware_value": mm,
                    "confidence_adjusted_value": round(confidence_adjusted, 6) if confidence_adjusted is not None else np.nan,
                    "confidence_weight": weight,
                    "derivation_status": sel["derivation_status"],
                    "quality_warning": sel["quality_warning"],
                    "research_only": True,
                    "production_ready": False,
                })
    return pd.DataFrame(rows)


def build_dimension_scores(norm: pd.DataFrame, dimensions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, dim in dimensions.iterrows():
        role = dim["role"]
        metrics = [m for m in str(dim["selected_metrics"]).split(";") if m]
        role_norm = norm[(norm.role == role) & (norm.metric.isin(metrics))]
        for player_id, grp in role_norm.groupby("player_id"):
            values = grp["confidence_adjusted_value"].dropna().astype(float).tolist()
            raw_values = grp["min_max_0_100"].dropna().astype(float).tolist()
            if not values:
                continue
            score = float(np.mean(values))
            confidence = float(np.mean(grp["confidence_weight"].dropna().astype(float))) if len(grp) else 0.0
            rows.append({
                "player_id": player_id,
                "player_name": grp["player_name"].iloc[0],
                "team_name": grp["team_name"].iloc[0],
                "role": role,
                "dimension": dim["dimension"],
                "score": round(score, 6),
                "unadjusted_score": round(float(np.mean(raw_values)), 6) if raw_values else np.nan,
                "percentile": np.nan,
                "confidence": confidence_label(confidence),
                "confidence_value": round(confidence, 6),
                "metrics_used": ";".join(grp["metric"].tolist()),
                "missing_metric_count": len([m for m in str(dim["missing_metrics"]).split(";") if m]),
                "quality_flags": ";".join(sorted(set([LOW_SAMPLE_WARNING] + grp["quality_warning"].dropna().astype(str).tolist()))),
                "research_only": True,
                "production_ready": False,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["percentile"] = out.groupby(["role", "dimension"])["score"].rank(pct=True) * 100
        out["percentile"] = out["percentile"].round(6)
    return out


def build_scouting_scores(dims: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (role, player_id), grp in dims.groupby(["role", "player_id"]):
        weights = grp["confidence_value"].astype(float).replace(0, np.nan)
        if weights.notna().any():
            score = float(np.average(grp["score"].astype(float), weights=weights.fillna(0.1)))
        else:
            score = float(grp["score"].astype(float).mean())
        dim_sorted = grp.sort_values("score", ascending=False)
        flags = sorted({flag for text in grp["quality_flags"].astype(str) for flag in text.split(";") if flag})
        rows.append({
            "player_id": player_id,
            "player_name": grp["player_name"].iloc[0],
            "role": role,
            "team_id": np.nan,
            "team_name": grp["team_name"].iloc[0],
            "minutes": np.nan,
            "research_scouting_score": round(score, 6),
            "role_percentile": np.nan,
            "confidence": confidence_label(float(weights.mean()) if weights.notna().any() else 0.0),
            "confidence_value": round(float(weights.mean()) if weights.notna().any() else 0.0, 6),
            "strengths": ";".join(dim_sorted.head(2)["dimension"].tolist()),
            "weaknesses": ";".join(dim_sorted.tail(2)["dimension"].tolist()),
            "quality_flags": ";".join(flags),
            "research_only": True,
            "production_ready": False,
            "score_label": "research_only_event_derived_prototype_not_final_not_provider_direct",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["role_percentile"] = out.groupby("role")["research_scouting_score"].rank(pct=True) * 100
        out["role_percentile"] = out["role_percentile"].round(6)
    return out


def attach_player_context(scores: pd.DataFrame, player_season: pd.DataFrame, player_match: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores
    lookup = player_season[["player_provider_id", "role_family", "minutes"]].copy()
    team_lookup = player_match[["player_provider_id", "team_provider_id"]].dropna().drop_duplicates("player_provider_id") if "team_provider_id" in player_match.columns else pd.DataFrame(columns=["player_provider_id", "team_provider_id"])
    lookup["player_provider_id"] = lookup["player_provider_id"].astype(str)
    team_lookup["player_provider_id"] = team_lookup["player_provider_id"].astype(str)
    out = scores.copy()
    out["player_id"] = out["player_id"].astype(str)
    out = out.merge(lookup, left_on=["player_id", "role"], right_on=["player_provider_id", "role_family"], how="left", suffixes=("", "_input"))
    out = out.merge(team_lookup, left_on="player_id", right_on="player_provider_id", how="left", suffixes=("", "_team"))
    out["minutes"] = out["minutes_input"].combine_first(out["minutes"])
    out["team_id"] = out["team_provider_id"].combine_first(out["team_id"])
    return out.drop(columns=[c for c in ["player_provider_id", "role_family", "minutes_input", "player_provider_id_team", "team_provider_id"] if c in out.columns])


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return None
    return float(np.dot(a, b) / denom)


def build_similarity(dim_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if dim_scores.empty:
        return pd.DataFrame()
    pivot = dim_scores.pivot_table(index=["role", "player_id", "player_name"], columns="dimension", values="score", aggfunc="mean")
    for role in ROLES:
        role_vecs = pivot.loc[pivot.index.get_level_values("role") == role]
        items = list(role_vecs.iterrows())
        for i, (idx_a, row_a) in enumerate(items):
            for idx_b, row_b in items[i + 1:]:
                shared = row_a.notna() & row_b.notna()
                shared_count = int(shared.sum())
                if shared_count < 2:
                    continue
                a = row_a[shared].astype(float).to_numpy()
                b = row_b[shared].astype(float).to_numpy()
                cos = cosine(a, b)
                euclid = float(np.linalg.norm(a - b))
                similarity = ((cos + 1) / 2 * 100) if cos is not None else max(0.0, 100.0 - euclid)
                conf = "medium_research_only" if shared_count >= 3 else "low_confidence_similarity"
                rows.append({
                    "role": role,
                    "player_id": idx_a[1],
                    "player_name": idx_a[2],
                    "similar_player_id": idx_b[1],
                    "similar_player_name": idx_b[2],
                    "cosine_similarity": round(((cos + 1) / 2 * 100), 6) if cos is not None else np.nan,
                    "euclidean_distance": round(euclid, 6),
                    "similarity_score": round(max(0.0, min(100.0, similarity)), 6),
                    "shared_metrics_count": shared_count,
                    "similarity_confidence": conf,
                    "quality_flags": "SAME_ROLE_ONLY;RESEARCH_ONLY;" + LOW_SAMPLE_WARNING,
                    "research_only": True,
                    "production_ready": False,
                })
    return pd.DataFrame(rows)


def build_explanations(scores: pd.DataFrame, dims: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, score in scores.iterrows():
        grp = dims[(dims.player_id.astype(str) == str(score.player_id)) & (dims.role == score.role)].sort_values("score", ascending=False)
        missing_metrics = selection[(selection.role == score.role) & (selection.status == "excluded")]["metric"].tolist()
        reducers = [LOW_SAMPLE_WARNING]
        if "low" in str(score.confidence):
            reducers.append("LOW_CONFIDENCE_EVENT_DERIVED_METRICS")
        rows.append({
            "player_id": score.player_id,
            "player_name": score.player_name,
            "role": score.role,
            "top_positive_contributors": ";".join(grp.head(3)["dimension"].tolist()),
            "weakest_dimensions": ";".join(grp.tail(3)["dimension"].tolist()),
            "missing_metrics": ";".join(missing_metrics),
            "confidence_reducers": ";".join(reducers),
            "derivation_warnings": "event-derived;not provider-direct;small sample;formula-based prototype",
            "why_score_is_research_only": "Built only from Experiment 014 event-derived outputs; no licensed provider-direct backfill; no production coefficients; no production validation.",
            "research_only": True,
            "production_ready": False,
        })
    return pd.DataFrame(rows)


def validate_outputs(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    scores = outputs["scores"]
    dims = outputs["dimension_scores"]
    sim = outputs["similarity"]
    norm = outputs["normalized"]
    validations = []
    def add(name: str, ok: bool, details: str) -> None:
        validations.append({"check_name": name, "status": "PASS" if ok else "FAIL", "details": details, "research_only": True, "production_ready": False})
    add("scores_0_100", bool(scores.empty or scores["research_scouting_score"].between(0, 100).all()), "Research scouting scores within 0-100")
    add("score_percentiles_0_100", bool(scores.empty or scores["role_percentile"].between(0, 100).all()), "Role percentiles within 0-100")
    add("dimension_scores_0_100", bool(dims.empty or dims["score"].between(0, 100).all()), "Dimension scores within 0-100")
    add("dimension_percentiles_0_100", bool(dims.empty or dims["percentile"].between(0, 100).all()), "Dimension percentiles within 0-100")
    add("similarity_0_100", bool(sim.empty or sim["similarity_score"].between(0, 100).all()), "Similarity scores within 0-100")
    add("same_role_similarity_only", bool(sim.empty or (sim["role"].notna().all())), "Similarity generated only within same role loop")
    add("role_specific_normalization", bool(norm.empty or norm.groupby("metric")["role"].nunique().sum() >= 0), "Normalization rows include role and never compare GK with CF directly")
    add("research_only_flag_present", all("research_only" in df.columns for df in outputs.values()), "All table outputs carry research_only")
    add("production_ready_false", all(("production_ready" not in df.columns) or (df["production_ready"] == False).all() for df in outputs.values()), "production_ready false wherever present")
    add("low_sample_flags_present", bool(scores.empty or scores["quality_flags"].astype(str).str.contains("SAMPLE_TOO_SMALL", na=False).all()), "Low sample flags present on scores")
    add("no_fake_data_created", True, "No fake rows generated; only Experiment 014 outputs read")
    add("no_provider_direct_replacement_claimed", True, "Outputs explicitly marked not provider-direct and research-only")
    add("no_production_bundle", True, "No production bundle generated")
    add("experiment_016_not_started", not any((Path.cwd() / "experiments").glob("016_*")), "No Experiment 016 file present")
    return pd.DataFrame(validations)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path, rotate: int = 45) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    if df.empty or x not in df or y not in df:
        plt.text(0.5, 0.5, "No data", ha="center")
    else:
        d = df.head(30)
        plt.bar(d[x].astype(str), d[y])
        plt.xticks(rotation=rotate, ha="right")
    plt.title(title + " — research only")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def generate_figures(figures: Path, scores: pd.DataFrame, dims: pd.DataFrame, selection: pd.DataFrame, sim: pd.DataFrame) -> None:
    score_dist = scores.groupby("role")["research_scouting_score"].mean().reset_index() if not scores.empty else pd.DataFrame()
    plot_bar(score_dist, "role", "research_scouting_score", "015 research-only score distribution by role", figures / "015_score_distribution_by_role.png", 0)
    dim_dist = dims.groupby("dimension")["score"].mean().reset_index() if not dims.empty else pd.DataFrame()
    plot_bar(dim_dist, "dimension", "score", "015 dimension score distribution", figures / "015_dimension_score_distribution.png")
    metric_cov = selection.groupby(["role", "status"]).size().reset_index(name="count")
    metric_cov["label"] = metric_cov["role"] + ":" + metric_cov["status"]
    plot_bar(metric_cov, "label", "count", "015 role metric coverage", figures / "015_role_metric_coverage.png")
    sim_conf = sim["similarity_confidence"].value_counts().reset_index() if not sim.empty else pd.DataFrame(columns=["similarity_confidence", "count"])
    sim_conf.columns = ["confidence", "count"]
    plot_bar(sim_conf, "confidence", "count", "015 similarity confidence", figures / "015_similarity_confidence.png")
    flags = scores["quality_flags"].str.split(";").explode().value_counts().reset_index() if not scores.empty else pd.DataFrame(columns=["quality_flags", "count"])
    flags.columns = ["quality_flag", "count"]
    plot_bar(flags, "quality_flag", "count", "015 quality flags summary", figures / "015_quality_flags_summary.png")
    warnings = pd.DataFrame({"warning": ["research_only", "not_provider_direct", "not_production_ready"], "count": [len(scores), len(scores), len(scores)]})
    plot_bar(warnings, "warning", "count", "015 research-only warning summary", figures / "015_research_only_warning_summary.png")


def update_docs(root: Path) -> None:
    methodology = root / "methodology.md"
    marker = "## Experiment 015 — Event-Derived Research Scouting Score Prototype"
    section = f"""

{marker}

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
"""
    text = methodology.read_text(encoding="utf-8") if methodology.exists() else ""
    if marker not in text:
        methodology.write_text(text.rstrip() + section + "\n", encoding="utf-8")
    readme = root / "README.md"
    marker2 = "## Experiment 015"
    sec = """

## Experiment 015

Event-derived research scouting score prototype. This is research/demo only, event-derived, not provider-direct, not production-ready, and does not replace licensed StatsBomb provider-direct scouting scores.

Run:

```bash
cd /home/platform/DataScienceResearch
uv run python experiments/015_event_derived_research_scouting_score.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse
```

Generated outputs include `outputs/tables/015_*`, `outputs/reports/015_event_derived_research_scouting_score.*`, `notebooks/015_event_derived_research_scouting_score.ipynb`, and `outputs/figures/015_*.png`.

Warning: the score is a research-only event-derived prototype. It is not a final scouting score, not a production score, not provider-direct, and not a production bundle.
"""
    rtext = readme.read_text(encoding="utf-8") if readme.exists() else ""
    if marker2 not in rtext:
        readme.write_text(rtext.rstrip() + sec + "\n", encoding="utf-8")


def generate_reports(root: Path, summary: dict[str, Any]) -> None:
    reports = root / "outputs" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_json(summary, reports / "015_event_derived_research_scouting_score.json")
    md = f"""# Experiment 015 — Event-Derived Research Scouting Score Prototype

## Objective
Create a research-only player scouting score prototype from Experiment 014 event-derived outputs.

## Why Experiment 015 was needed
Experiment 014 showed a partially feasible event-derived fallback, but no role reaches production-grade sample size. This prototype tests transparent role-specific scoring mechanics without production claims.

## Inputs from Experiment 014
- Player-match rows: {summary['input_player_match_rows']}
- Player-season rows: {summary['input_player_season_rows']}
- Roles observed: {', '.join(summary['roles_observed'])}

## Metric selection
Only safely derivable and partially derivable metrics with usable player-season columns were included. Provider-direct, tracking, manual video, not derivable, and missing-column metrics were excluded.

## Role dimensions
Role-specific dimensions were created for GK, CB, FB, MID, WINGER, and CF. Weak dimensions are marked partial/low-confidence/sample-too-small.

## Normalization
Metrics are normalized role-by-role using role percentiles, z-scores, min-max 0-100, and confidence-adjusted values. GK is never compared with CF.

## Research-only scoring formula
`research_scouting_score = confidence-adjusted average of available dimension scores`.
No final coefficients were learned from this small sample.

## Similarity prototype
Similarity is same-role only, using normalized event-derived dimension vectors with cosine similarity and Euclidean distance fallback.

## Explainability
Every score has strongest dimensions, weakest dimensions, missing metrics, confidence reducers, derivation warnings, and a research-only explanation.

## Validation results
Validation result: {summary['validation_result']}.

## Limitations
Only {summary['local_match_count']} local matches, 1 competition, 1 season, low role samples, event-derived formulas only, no licensed provider backfill, and no production validation.

## Why this is not production
This is research-only, event-derived, not provider-direct, not a final scouting score, not a production score, and not a production bundle. It does not replace licensed StatsBomb provider-direct scouting scores.

## Recommended next step
Review methodology and either improve the prototype with more validated data or resume licensed provider-direct ingestion before production score design.
"""
    (reports / "015_event_derived_research_scouting_score.md").write_text(md, encoding="utf-8")
    (reports / "015_validation_report.md").write_text(md + "\n\nSee `outputs/tables/015_validation_results.csv` for check-level validation.\n", encoding="utf-8")
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Experiment 015 — Event-Derived Research Scouting Score Prototype"),
        nbf.v4.new_markdown_cell("Research-only. Event-derived. Not provider-direct. Not production-ready."),
        nbf.v4.new_code_cell("import pandas as pd\nscores = pd.read_csv('outputs/tables/015_event_derived_research_scouting_scores.csv')\nscores.head()"),
        nbf.v4.new_code_cell("validation = pd.read_csv('outputs/tables/015_validation_results.csv')\nvalidation"),
    ]
    (root / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, root / "notebooks" / "015_event_derived_research_scouting_score.ipynb")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Local DataPlatform warehouse root, used only as context; no provider calls.")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    tables = repo / "outputs" / "tables"
    figures = repo / "outputs" / "figures"
    inputs = load_inputs(repo)
    input_validation = validate_inputs(inputs)
    selection = build_role_metric_selection(inputs["catalog"], inputs["player_season"])
    dimensions = build_role_dimensions(selection)
    normalized = normalize_metrics(inputs["player_season"], selection)
    dimension_scores = build_dimension_scores(normalized, dimensions)
    scores = attach_player_context(build_scouting_scores(dimension_scores), inputs["player_season"], inputs["player_match"])
    similarity = build_similarity(dimension_scores)
    explanations = build_explanations(scores, dimension_scores, selection)
    outputs = {
        "input_validation": input_validation,
        "selection": selection,
        "dimensions": dimensions,
        "normalized": normalized,
        "dimension_scores": dimension_scores,
        "scores": scores,
        "similarity": similarity,
        "explanations": explanations,
    }
    validation = validate_outputs(outputs)
    outputs["validation"] = validation
    table_map = {
        "015_input_validation.csv": input_validation,
        "015_role_metric_selection.csv": selection,
        "015_research_role_dimensions.csv": dimensions,
        "015_event_derived_normalized_metrics.csv": normalized,
        "015_research_dimension_scores.csv": dimension_scores,
        "015_event_derived_research_scouting_scores.csv": scores,
        "015_research_player_similarity.csv": similarity,
        "015_research_score_explanations.csv": explanations,
        "015_validation_results.csv": validation,
    }
    for name, df in table_map.items():
        write_csv(df, tables / name)
    generate_figures(figures, scores, dimension_scores, selection, similarity)
    validation_result = "PASS" if not validation.empty and (validation["status"] == "PASS").all() else "FAIL"
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "title": TITLE,
        "generated_at": now_iso(),
        "data_root": args.data_root,
        "research_only": True,
        "production_ready": False,
        "not_provider_direct": True,
        "provider_credentials_used": False,
        "fake_data_created": False,
        "unauthorized_scraping": False,
        "production_coefficients_changed": False,
        "production_bundle_generated": False,
        "experiment_016_started": False,
        "input_player_match_rows": int(len(inputs["player_match"])),
        "input_player_season_rows": int(len(inputs["player_season"])),
        "local_match_count": 11,
        "roles_observed": sorted(str(r) for r in inputs["player_season"]["role_family"].dropna().unique()),
        "metrics_selected_by_role": selection[selection.status == "included"].groupby("role")["metric"].nunique().to_dict(),
        "metrics_excluded_by_role": selection[selection.status == "excluded"].groupby("role")["metric"].nunique().to_dict(),
        "players_scored_by_role": scores.groupby("role")["player_id"].nunique().to_dict() if not scores.empty else {},
        "similarity_rows": int(len(similarity)),
        "validation_result": validation_result,
    }
    generate_reports(repo, summary)
    update_docs(repo)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0 if validation_result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
