from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_score_engine_research.io import flatten_metrics, write_json

EXPERIMENT_ID = "004"
EXPERIMENT_TITLE = "Role-Specific Weight Estimation & Prototype Score Formula"
ROLES = ["GK", "CB", "FB", "MID", "WINGER", "CF"]
RANDOM_SEED = 42
SMALL_SAMPLE_N = 30

HIGHER_PATTERNS = (
    "xg", "goals_90", "np_shots", "key_pass", "assist", "xa", "ball_recover", "interception",
    "clearance", "block", "pressure", "counterpressure", "aerial", "dribble", "cross", "obv_pass",
    "obv_defensive", "obv_dribble", "op_f3", "xgbuildup", "xgchain", "touches_inside_box",
    "pass_length", "average_x_defensive_action", "passes_into", "box_cross", "through_balls",
)
LOWER_PATTERNS = ("goals_faced",)
MANUAL_PATTERNS = ("psxg_faced", "shots_faced", "ot_shots_faced", "obv_gk", "np_xg_faced")


def finite_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def minmax_0_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = s.dropna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if clean.empty:
        return out
    span = clean.max() - clean.min()
    out.loc[clean.index] = 50.0 if span == 0 or pd.isna(span) else (clean - clean.min()) / span * 100.0
    return out.clip(0, 100)


def normalize_series(series: pd.Series, method: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = s.dropna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if clean.empty:
        return out
    if method == "z_score":
        std = clean.std(ddof=1)
        out.loc[clean.index] = (clean - clean.mean()) / std if std and not np.isnan(std) else 0.0
    elif method == "robust_z_score":
        med = clean.median(); mad = (clean - med).abs().median()
        out.loc[clean.index] = 0.6745 * (clean - med) / mad if mad and not np.isnan(mad) else 0.0
    elif method == "percentile_rank":
        out.loc[clean.index] = clean.rank(pct=True, method="average") * 100.0
        return out
    elif method == "min_max":
        return minmax_0_100(clean.reindex(s.index))
    elif method == "log_transform":
        shifted = clean - clean.min()
        transformed = np.log1p(shifted)
        std = transformed.std(ddof=1)
        out.loc[clean.index] = (transformed - transformed.mean()) / std if std and not np.isnan(std) else 0.0
    elif method == "quantile_transform":
        out.loc[clean.index] = clean.rank(pct=True, method="average") * 100.0
        return out
    elif method == "winsorized_z_score":
        lo, hi = clean.quantile([0.05, 0.95])
        clipped = clean.clip(lo, hi)
        std = clipped.std(ddof=1)
        out.loc[clean.index] = (clipped - clipped.mean()) / std if std and not np.isnan(std) else 0.0
    else:
        raise ValueError(method)
    return minmax_0_100(out)


def benchmark_band(value: float | None, role: str, metric: str, benchmarks: pd.DataFrame) -> str:
    if value is None or pd.isna(value):
        return "Missing"
    rows = benchmarks[(benchmarks.role_family == role) & (benchmarks.metric == metric)]
    for row in rows.itertuples(index=False):
        if value >= row.lower_bound and value <= row.upper_bound:
            return str(row.benchmark_band)
    if rows.empty:
        return "Unbenchmarked"
    if value < rows.lower_bound.min():
        return "Very Poor"
    return "Excellent"


def classify_direction(metric: str, role: str) -> tuple[str, str]:
    m = metric.lower()
    if any(p in m for p in MANUAL_PATTERNS):
        return "manual_review_required", "contextual goalkeeper/opponent-volume metric; needs model context before inversion/use"
    if any(p in m for p in LOWER_PATTERNS):
        return "lower_is_better", "transparent metric-name rule: conceded/faced outcome is lower-is-better"
    if any(p in m for p in HIGHER_PATTERNS):
        return "higher_is_better", "transparent metric-name rule: action/value/output metric is higher-is-better"
    if "ratio" in m or "length" in m or "average_x" in m:
        return "manual_review_required", "contextual ratio/location/length metric; requires football review before scoring direction"
    return "manual_review_required", "no safe automatic direction rule matched"


def cap_and_normalize(weights: pd.Series, cap: float) -> pd.Series:
    w = weights.fillna(0).clip(lower=0).astype(float)
    if w.sum() <= 0:
        w[:] = 1.0 / len(w) if len(w) else 0
        return w
    w = w / w.sum()
    # Iteratively cap and redistribute excess.
    for _ in range(10):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if under.any() and w[under].sum() > 0:
            w[under] += excess * w[under] / w[under].sum()
        else:
            break
    return w / w.sum() if w.sum() > 0 else w


def entropy_weight(values: pd.DataFrame) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float)
    X = values.apply(minmax_0_100).fillna(0) / 100.0
    X = X + 1e-9
    P = X.div(X.sum(axis=0), axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)
    n = max(len(X), 2)
    entropy = -(P * np.log(P + 1e-12)).sum(axis=0) / np.log(n)
    diversity = 1 - entropy
    if diversity.sum() <= 0:
        return pd.Series(1 / len(values.columns), index=values.columns)
    return diversity / diversity.sum()


def bootstrap_metric_stability(values: pd.DataFrame, n_boot: int = 200) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    scores = {}
    for col in values.columns:
        clean = finite_series(values[col])
        if len(clean) < 3:
            scores[col] = 0.0
            continue
        means = np.array([rng.choice(clean, size=len(clean), replace=True).mean() for _ in range(n_boot)])
        width = np.percentile(means, 97.5) - np.percentile(means, 2.5)
        scale = clean.std(ddof=1) or 1.0
        scores[col] = 1.0 / (1.0 + abs(width / scale))
    s = pd.Series(scores)
    return s / s.sum() if s.sum() > 0 else pd.Series(1 / len(s), index=s.index)


def build_input_frames(data_root: Path) -> dict[str, pd.DataFrame]:
    player_season_path = data_root / "marts_v2/mart_statsbomb_player_season_stats_direct_v1.jsonl"
    paths = [
        player_season_path,
        ROOT / "outputs/tables/002_role_resolution.csv",
        ROOT / "outputs/tables/002_candidate_metric_status.csv",
        ROOT / "outputs/tables/003_normalization_decisions.csv",
        ROOT / "outputs/tables/003_metric_statistics.csv",
        ROOT / "outputs/tables/003_latent_dimensions.csv",
        ROOT / "outputs/tables/003_metric_clusters.csv",
        ROOT / "outputs/tables/003_weight_preparation.csv",
        ROOT / "outputs/tables/003_role_benchmarks.csv",
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Experiment 004 inputs: {missing}")
    player = flatten_metrics(player_season_path, ["statsbomb_player_id", "player_name", "team_id", "team_name", "competition_id", "season_id"])
    player["statsbomb_player_id"] = player.statsbomb_player_id.astype(str)
    frames = {
        "player": player,
        "roles": pd.read_csv(ROOT / "outputs/tables/002_role_resolution.csv"),
        "candidates": pd.read_csv(ROOT / "outputs/tables/002_candidate_metric_status.csv"),
        "norm": pd.read_csv(ROOT / "outputs/tables/003_normalization_decisions.csv"),
        "stats": pd.read_csv(ROOT / "outputs/tables/003_metric_statistics.csv"),
        "latent": pd.read_csv(ROOT / "outputs/tables/003_latent_dimensions.csv"),
        "clusters": pd.read_csv(ROOT / "outputs/tables/003_metric_clusters.csv"),
        "weight_prep": pd.read_csv(ROOT / "outputs/tables/003_weight_preparation.csv"),
        "benchmarks": pd.read_csv(ROOT / "outputs/tables/003_role_benchmarks.csv"),
    }
    frames["roles"]["statsbomb_player_id"] = frames["roles"].statsbomb_player_id.astype(str)
    return frames


def prepare_feature_matrix(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player = frames["player"]
    roles = frames["roles"]
    ready = frames["candidates"][frames["candidates"].status == "candidate_ready"].copy()
    norm = frames["norm"]
    latent = frames["latent"]
    benchmarks = frames["benchmarks"]
    eligible = roles[(roles.assigned_role.isin(ROLES)) & (roles.eligible_for_initial_coefficients == True)].copy()  # noqa: E712
    base = player.merge(eligible[["statsbomb_player_id", "assigned_role", "season_minutes"]], on="statsbomb_player_id", how="inner")
    rows = []
    dirs = []
    for r in ready.itertuples(index=False):
        role = r.role_family; metric = r.requested_metric_alias
        if metric not in player.columns:
            continue
        nrow = norm[(norm.role_family == role) & (norm.metric == metric)]
        lrow = latent[(latent.role_family == role) & (latent.metric == metric)]
        if nrow.empty or lrow.empty:
            continue
        direction, reason = classify_direction(metric, role)
        dirs.append({"role_family": role, "metric": metric, "direction": direction, "direction_reason": reason, "included_in_prototype_score": direction in {"higher_is_better", "lower_is_better"}})
        role_df = base[base.assigned_role == role].copy()
        raw = pd.to_numeric(role_df[metric], errors="coerce")
        norm_values = normalize_series(raw, str(nrow.iloc[0].selected_normalization))
        if direction == "lower_is_better":
            oriented = 100.0 - norm_values
        elif direction == "higher_is_better":
            oriented = norm_values
        else:
            oriented = np.nan
        percentiles = raw.rank(pct=True, method="average") * 100
        if direction == "lower_is_better":
            percentiles = 100 - percentiles
        for idx, p in role_df.iterrows():
            rows.append({
                "player_id": p.statsbomb_player_id,
                "player_name": p.get("player_name"),
                "team_id": p.get("team_id"),
                "team_name": p.get("team_name"),
                "competition_id": p.get("competition_id"),
                "season_id": p.get("season_id"),
                "role": role,
                "minutes": p.get("minutes", p.get("season_minutes")),
                "metric": metric,
                "latent_dimension": lrow.iloc[0].dimension_id,
                "raw_metric_value": raw.loc[idx] if idx in raw.index else np.nan,
                "selected_normalization": nrow.iloc[0].selected_normalization,
                "normalized_metric_value": norm_values.loc[idx] if idx in norm_values.index else np.nan,
                "oriented_metric_score": oriented.loc[idx] if hasattr(oriented, "loc") and idx in oriented.index else np.nan,
                "percentile_rank": percentiles.loc[idx] if idx in percentiles.index else np.nan,
                "role_benchmark_band": benchmark_band(raw.loc[idx], role, metric, benchmarks) if idx in raw.index else "Missing",
                "direction": direction,
            })
    feature_matrix = pd.DataFrame(rows)
    direction_registry = pd.DataFrame(dirs).drop_duplicates(["role_family", "metric"])
    return feature_matrix, direction_registry, eligible


def estimate_metric_weights(feature_matrix: pd.DataFrame, direction_registry: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_df = frames["stats"]
    wp = frames["weight_prep"]
    rows = []
    decisions = []
    for (role, dim), group in feature_matrix.groupby(["role", "latent_dimension"]):
        usable_metrics = sorted(set(group.loc[group.direction.isin(["higher_is_better", "lower_is_better"]), "metric"]))
        if not usable_metrics:
            continue
        pivot = group.pivot_table(index="player_id", columns="metric", values="oriented_metric_score", aggfunc="mean")[usable_metrics]
        n_players = len(pivot)
        m = len(usable_metrics)
        cap = 1.0 if m == 1 else (0.60 if m == 2 else 0.40)
        equal = pd.Series(1 / m, index=usable_metrics)
        pca_w = wp[(wp.role_family == role) & (wp.metric.isin(usable_metrics))].set_index("metric").reindex(usable_metrics)["pca_pc1_abs_loading"].fillna(0)
        pca_w = pca_w / pca_w.sum() if pca_w.sum() > 0 else equal.copy()
        var_w = wp[(wp.role_family == role) & (wp.metric.isin(usable_metrics))].set_index("metric").reindex(usable_metrics)["normalized_variance"].fillna(0)
        var_w = var_w / var_w.sum() if var_w.sum() > 0 else equal.copy()
        st = stats_df[(stats_df.role_family == role) & (stats_df.metric.isin(usable_metrics))].set_index("metric").reindex(usable_metrics)
        stability = 1 / (1 + st["cv"].abs().fillna(1) + (st["outlier_pct"].fillna(0) / 100))
        stab_w = stability / stability.sum() if stability.sum() > 0 else equal.copy()
        ent_w = entropy_weight(pivot)
        boot_w = bootstrap_metric_stability(pivot)
        all_methods = pd.DataFrame({"equal": equal, "pca": pca_w, "variance": var_w, "stability": stab_w, "entropy": ent_w, "bootstrap": boot_w}).fillna(equal)
        disagreement = all_methods.std(axis=1).mean()
        shrink = min(0.75, (0.35 if n_players < SMALL_SAMPLE_N else 0.10) + min(0.30, disagreement * 2.0))
        ensemble_raw = all_methods[["pca", "variance", "stability", "entropy", "bootstrap"]].mean(axis=1)
        ensemble = (1 - shrink) * ensemble_raw + shrink * equal
        ensemble = cap_and_normalize(ensemble, cap)
        for metric in usable_metrics:
            ci_width_proxy = 1 - float(boot_w.get(metric, 0))
            flags = []
            if n_players < SMALL_SAMPLE_N: flags.append("small_sample")
            if all_methods.loc[metric].std() > 0.15: flags.append("high_method_disagreement")
            if ci_width_proxy > 0.6: flags.append("wide_confidence_interval")
            if m == 1: flags.append("single_metric_dimension")
            if ensemble.loc[metric] >= cap - 1e-9 and m > 1: flags.append("weight_cap_applied")
            rows.append({
                "role": role, "latent_dimension": dim, "metric": metric,
                "equal_weight": equal.loc[metric], "pca_loading_weight": pca_w.loc[metric],
                "variance_contribution_weight": var_w.loc[metric], "stability_adjusted_weight": stab_w.loc[metric],
                "entropy_weight": ent_w.loc[metric], "bootstrap_stability_weight": boot_w.loc[metric],
                "method_disagreement": float(all_methods.loc[metric].std()),
                "shrinkage_to_equal": shrink, "max_weight_cap": cap,
                "ensemble_candidate_weight": ensemble.loc[metric], "warning_flags": ";".join(flags),
            })
            decisions.append({
                "role": role, "latent_dimension": dim, "metric": metric,
                "selected_metric_weight": ensemble.loc[metric], "weight_confidence": "low" if flags else "medium",
                "is_unstable": bool(flags), "decision_reason": "ensemble of PCA/variance/stability/entropy/bootstrap with shrinkage toward equal weighting",
                "warning_flags": ";".join(flags), "single_metric_dimension": m == 1,
            })
    return pd.DataFrame(rows), pd.DataFrame(decisions)


def estimate_dimension_weights(metric_decisions: pd.DataFrame, frames: dict[str, pd.DataFrame], feature_redundancy_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    latent = frames["latent"]
    wp = frames["weight_prep"]
    stats_df = frames["stats"]
    pca_var = pd.read_csv(ROOT / "outputs/tables/003_pca_variance.csv")
    red = pd.read_csv(feature_redundancy_path) if feature_redundancy_path.exists() else pd.DataFrame()
    rows = []
    decisions = []
    for role, role_dec in metric_decisions.groupby("role"):
        dims = []
        for dim, group in role_dec.groupby("latent_dimension"):
            metrics = list(group.metric)
            n_metrics = len(metrics)
            st = stats_df[(stats_df.role_family == role) & (stats_df.metric.isin(metrics))]
            avg_stability = float((1 / (1 + st.cv.abs().fillna(1))).mean()) if not st.empty else 0.0
            explained = float(pca_var[pca_var.role_family == role].explained_variance_ratio.head(max(1, min(n_metrics, 3))).mean()) if not pca_var[pca_var.role_family == role].empty else 0.0
            boot = float(1 - group.is_unstable.mean())
            pca_contrib = float(wp[(wp.role_family == role) & (wp.metric.isin(metrics))].pca_pc1_abs_loading.mean()) if not wp.empty else 0.0
            red_penalty = 0.0
            if not red.empty and "role_family" in red.columns:
                red_penalty = float(len(red[(red.role_family == role) & (red.metric_a.isin(metrics) | red.metric_b.isin(metrics))])) * 0.05
            small_penalty = 0.20 if len(frames["roles"][(frames["roles"].assigned_role == role) & (frames["roles"].eligible_for_initial_coefficients == True)]) < SMALL_SAMPLE_N else 0.0  # noqa: E712
            raw = max(0.001, (0.30 * avg_stability + 0.25 * explained + 0.25 * pca_contrib + 0.20 * boot) * (1 - red_penalty) * (1 - small_penalty))
            flags = []
            if n_metrics == 1: flags.append("single_metric_dimension")
            if small_penalty: flags.append("small_sample")
            if boot < 0.7: flags.append("unstable_weight")
            dims.append((dim, metrics, n_metrics, avg_stability, explained, boot, pca_contrib, red_penalty, small_penalty, raw, flags))
        total = sum(d[-2] for d in dims) or 1.0
        for dim, metrics, n_metrics, avg_stability, explained, boot, pca_contrib, red_penalty, small_penalty, raw, flags in dims:
            adj = raw / total
            conf = "low" if flags else ("medium" if len(metrics) < 3 else "medium_high")
            row = {"role": role, "dimension_name": dim, "number_of_metrics": n_metrics, "average_stability": avg_stability, "explained_variance": explained, "bootstrap_stability": boot, "pca_contribution": pca_contrib, "redundancy_penalty": red_penalty, "small_sample_penalty": small_penalty, "raw_dimension_weight": raw, "adjusted_dimension_weight": adj, "confidence_level": conf, "warning_flags": ";".join(flags)}
            rows.append(row); decisions.append(row | {"decision_reason": "weighted evidence blend normalized within role; prototype only"})
    return pd.DataFrame(rows), pd.DataFrame(decisions)


def compute_scores(feature_matrix: pd.DataFrame, metric_decisions: pd.DataFrame, dim_decisions: pd.DataFrame, eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = feature_matrix[feature_matrix.direction.isin(["higher_is_better", "lower_is_better"])].copy()
    merged = usable.merge(metric_decisions[["role", "latent_dimension", "metric", "selected_metric_weight", "warning_flags", "single_metric_dimension"]], on=["role", "latent_dimension", "metric"], how="inner")
    dim_rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    boot_role_bounds = {}
    for (role, dim, pid), group in merged.groupby(["role", "latent_dimension", "player_id"]):
        weights = group.selected_metric_weight.fillna(0)
        available = group.oriented_metric_score.notna()
        if available.any() and weights[available].sum() > 0:
            score = float(np.average(group.loc[available, "oriented_metric_score"], weights=weights[available]))
        else:
            score = np.nan
        dim_rows.append({
            "role": role, "latent_dimension": dim, "player_id": pid,
            "player_name": group.player_name.iloc[0], "team_id": group.team_id.iloc[0], "team_name": group.team_name.iloc[0],
            "minutes": group.minutes.iloc[0], "dimension_score": score,
            "number_of_available_metrics": int(available.sum()), "missing_metric_count": int((~available).sum()),
            "confidence_flag": "low" if group.warning_flags.astype(str).str.len().gt(0).any() else "medium",
            "single_metric_dimension_warning": bool(group.single_metric_dimension.any()),
        })
    dim_scores = pd.DataFrame(dim_rows)
    if not dim_scores.empty:
        dim_scores["dimension_percentile"] = dim_scores.groupby(["role", "latent_dimension"])["dimension_score"].rank(pct=True) * 100
        dim_scores["dimension_rank_within_role"] = dim_scores.groupby(["role", "latent_dimension"])["dimension_score"].rank(ascending=False, method="min")
    role_rows = []
    method_score_rows = []
    sensitivity_rows = []
    rank_instability_rows = []
    method_cols = ["equal_weight", "pca_loading_weight", "stability_adjusted_weight", "entropy_weight", "ensemble_candidate_weight"]
    for role, role_dim in dim_scores.groupby("role"):
        dw = dim_decisions[dim_decisions.role == role].set_index("dimension_name")
        player_scores = {}
        for pid, group in role_dim.groupby("player_id"):
            group = group.set_index("latent_dimension")
            common = group.index.intersection(dw.index)
            if len(common) == 0:
                continue
            weights = dw.loc[common, "adjusted_dimension_weight"]
            values = group.loc[common, "dimension_score"]
            ok = values.notna()
            score = float(np.average(values[ok], weights=weights[ok])) if ok.any() and weights[ok].sum() > 0 else np.nan
            flags = set(";".join(dw.loc[common, "warning_flags"].fillna("")).split(";")) - {""}
            if len(role_dim.player_id.unique()) < SMALL_SAMPLE_N: flags.add("small_sample")
            if group.missing_metric_count.sum() > 0: flags.add("missing_metrics")
            if any(group.single_metric_dimension_warning): flags.add("single_metric_dimension")
            flags.add("limited_dataset_scope")
            boot = []
            vals = values[ok].to_numpy(); w = weights[ok].to_numpy()
            if len(vals) > 0:
                for _ in range(250):
                    sample_idx = rng.choice(np.arange(len(vals)), len(vals), replace=True)
                    boot.append(float(np.average(vals[sample_idx], weights=w[sample_idx])))
            lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (np.nan, np.nan)
            unc = hi - lo if pd.notna(hi) and pd.notna(lo) else np.nan
            if pd.notna(unc) and unc > 20: flags.add("wide_confidence_interval")
            player_scores[pid] = score
            role_rows.append({"role": role, "player_id": pid, "player_name": group.player_name.iloc[0], "team_id": group.team_id.iloc[0], "team_name": group.team_name.iloc[0], "minutes": group.minutes.iloc[0], "prototype_role_score": score, "confidence_level": "low" if len(flags) >= 3 else "medium", "score_uncertainty": unc, "bootstrap_lower_bound": lo, "bootstrap_upper_bound": hi, "data_quality_warning": ";".join(sorted(flags)), "single_metric_dimension_warning": "single_metric_dimension" in flags, "small_sample_warning": "small_sample" in flags})
        # Sensitivity role scores by metric weighting method, keeping dimension weights fixed.
        role_merged = usable[usable.role == role].merge(metric_decisions[metric_decisions.role == role], on=["role", "latent_dimension", "metric"], how="inner")
        for method in method_cols:
            temp_dim = []
            method_weight_col = method if method != "ensemble_candidate_weight" else "selected_metric_weight"
            # Map method weights from full method table when needed.
            method_weights = None
            if method != "ensemble_candidate_weight":
                mw = metric_weight_methods_global[(metric_weight_methods_global.role == role)][["latent_dimension", "metric", method]].rename(columns={method: "mw"})
                method_weights = mw
                rm = usable[usable.role == role].merge(mw, on=["latent_dimension", "metric"], how="inner")
            else:
                rm = role_merged.rename(columns={"selected_metric_weight": "mw"})
            for (pid, dim), g in rm.groupby(["player_id", "latent_dimension"]):
                ok = g.oriented_metric_score.notna()
                ds = float(np.average(g.loc[ok, "oriented_metric_score"], weights=g.loc[ok, "mw"])) if ok.any() and g.loc[ok, "mw"].sum() > 0 else np.nan
                temp_dim.append({"player_id": pid, "latent_dimension": dim, "dimension_score": ds})
            td = pd.DataFrame(temp_dim)
            for pid, g in td.groupby("player_id") if not td.empty else []:
                g = g.set_index("latent_dimension"); common = g.index.intersection(dw.index)
                sc = float(np.average(g.loc[common, "dimension_score"], weights=dw.loc[common, "adjusted_dimension_weight"])) if len(common) else np.nan
                method_score_rows.append({"role": role, "player_id": pid, "weighting_method": method.replace("_weight", ""), "prototype_role_score": sc})
        ms = pd.DataFrame([r for r in method_score_rows if r["role"] == role])
        if not ms.empty:
            rank_pivot = ms.pivot_table(index="player_id", columns="weighting_method", values="prototype_role_score")
            rank_pivot = rank_pivot.rank(ascending=False)
            corr = rank_pivot.corr(method="spearman")
            for a in corr.columns:
                for b in corr.columns:
                    sensitivity_rows.append({"role": role, "method_a": a, "method_b": b, "spearman_rank_correlation": corr.loc[a, b]})
            rank_range = rank_pivot.max(axis=1) - rank_pivot.min(axis=1)
            for pid, rr in rank_range.sort_values(ascending=False).items():
                if rr >= max(3, 0.20 * len(rank_pivot)):
                    rank_instability_rows.append({"role": role, "player_id": pid, "rank_range": float(rr), "high_rank_instability": True})
    role_scores = pd.DataFrame(role_rows)
    if not role_scores.empty:
        role_scores["role_rank"] = role_scores.groupby("role")["prototype_role_score"].rank(ascending=False, method="min")
        role_scores["role_percentile"] = role_scores.groupby("role")["prototype_role_score"].rank(pct=True) * 100
    return dim_scores, role_scores, pd.DataFrame(sensitivity_rows), pd.DataFrame(rank_instability_rows), pd.DataFrame(method_score_rows), merged

# global used by compute_scores for sensitivity
metric_weight_methods_global = pd.DataFrame()


def quality_flags(metric_decisions: pd.DataFrame, dim_decisions: pd.DataFrame, role_scores: pd.DataFrame, direction_registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in metric_decisions.itertuples(index=False):
        for flag in str(r.warning_flags).split(";"):
            if flag and flag != "nan": rows.append({"scope": "metric_weight", "role": r.role, "entity": r.metric, "flag": flag})
    for r in dim_decisions.itertuples(index=False):
        for flag in str(r.warning_flags).split(";"):
            if flag and flag != "nan": rows.append({"scope": "dimension_weight", "role": r.role, "entity": r.dimension_name, "flag": flag})
    for r in role_scores.itertuples(index=False):
        for flag in str(r.data_quality_warning).split(";"):
            if flag and flag != "nan": rows.append({"scope": "prototype_role_score", "role": r.role, "entity": r.player_id, "flag": flag})
    for r in direction_registry[direction_registry.direction == "manual_review_required"].itertuples(index=False):
        rows.append({"scope": "metric_direction", "role": r.role_family, "entity": r.metric, "flag": "manual_direction_review_required"})
    return pd.DataFrame(rows)


def save_figures(role: str, metric_decisions: pd.DataFrame, dim_decisions: pd.DataFrame, dim_scores: pd.DataFrame, role_scores: pd.DataFrame, sensitivity: pd.DataFrame, method_scores: pd.DataFrame) -> list[str]:
    paths = []
    fig_dir = ROOT / "outputs/figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    md = metric_decisions[metric_decisions.role == role]
    dd = dim_decisions[dim_decisions.role == role]
    ds = dim_scores[dim_scores.role == role]
    rs = role_scores[role_scores.role == role]
    sens = sensitivity[sensitivity.role == role]
    ms = method_scores[method_scores.role == role]
    def save(path: Path):
        plt.tight_layout(); plt.savefig(path, dpi=160); plt.close(); paths.append(str(path.relative_to(ROOT)))
    plt.figure(figsize=(12, 6)); sns.barplot(data=md, x="selected_metric_weight", y="metric", hue="latent_dimension", dodge=False); plt.title(f"{role} metric weights"); save(fig_dir / f"004_{role}_metric_weights.png")
    plt.figure(figsize=(10, 5)); sns.barplot(data=dd, x="adjusted_dimension_weight", y="dimension_name", color="#285C7D"); plt.title(f"{role} dimension weights"); save(fig_dir / f"004_{role}_dimension_weights.png")
    plt.figure(figsize=(11, 6)); sns.histplot(data=ds, x="dimension_score", hue="latent_dimension", kde=True); plt.title(f"{role} prototype dimension score distributions"); save(fig_dir / f"004_{role}_dimension_score_distributions.png")
    plt.figure(figsize=(9, 5)); sns.histplot(data=rs, x="prototype_role_score", kde=True, color="#285C7D"); plt.title(f"{role} prototype role score distribution"); save(fig_dir / f"004_{role}_role_score_distribution.png")
    if not ms.empty:
        pivot = ms.pivot_table(index="player_id", columns="weighting_method", values="prototype_role_score").rank(ascending=False)
        plt.figure(figsize=(8, max(5, min(12, len(pivot)*0.25)))); sns.heatmap(pivot.head(40), cmap="viridis", cbar_kws={"label": "rank"}); plt.title(f"{role} ranking sensitivity heatmap"); save(fig_dir / f"004_{role}_ranking_sensitivity_heatmap.png")
    else:
        plt.figure(figsize=(6, 4)); plt.text(.1,.5,"No sensitivity data"); save(fig_dir / f"004_{role}_ranking_sensitivity_heatmap.png")
    plt.figure(figsize=(10, 6)); top = rs.sort_values("prototype_role_score", ascending=False).head(25); plt.errorbar(top.prototype_role_score, top.player_name.fillna(top.player_id), xerr=[top.prototype_role_score-top.bootstrap_lower_bound, top.bootstrap_upper_bound-top.prototype_role_score], fmt="o"); plt.title(f"{role} bootstrap uncertainty"); save(fig_dir / f"004_{role}_bootstrap_uncertainty.png")
    fig, ax = plt.subplots(figsize=(12, max(4, min(10, len(rs.head(20))*0.35)))); ax.axis("off"); table_df = rs.sort_values("prototype_role_score", ascending=False)[["player_name","team_name","prototype_role_score","role_rank"]].head(20).round(2); ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center"); ax.set_title(f"{role} top 20 prototype role scores"); save(fig_dir / f"004_{role}_top20_table.png")
    plt.figure(figsize=(9, 5)); sns.scatterplot(data=rs, x="minutes", y="prototype_role_score"); plt.title(f"{role} score vs minutes"); save(fig_dir / f"004_{role}_score_vs_minutes.png")
    if not sens.empty:
        corr = sens.pivot(index="method_a", columns="method_b", values="spearman_rank_correlation"); plt.figure(figsize=(7, 6)); sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap="vlag"); plt.title(f"{role} method rank-correlation heatmap"); save(fig_dir / f"004_{role}_method_comparison_rank_correlation_heatmap.png")
    else:
        plt.figure(figsize=(6,4)); plt.text(.1,.5,"No method comparison data"); save(fig_dir / f"004_{role}_method_comparison_rank_correlation_heatmap.png")
    return paths


def write_notebook(path: Path, data_root: Path) -> None:
    nb = nbf.v4.new_notebook()
    sections = [
        ("# Experiment 004 — Role-Specific Weight Estimation & Prototype Score Formula", ""),
        ("## 1. Objective", "Estimate role-specific prototype metric and dimension weights. This is not a final production score."),
        ("## 2. Football hypothesis", "Different roles require independent weighting logic, and every weight must be traceable to data evidence."),
        ("## 3. Dataset", f"Data root: `{data_root}` plus Experiment 002/003 tables."),
        ("## 4. Feature engineering", "Build normalized feature matrix with selected Experiment 003 normalization methods, direction registry, percentiles, and benchmark bands."),
        ("## 5. Exploratory Data Analysis", "Inspect prototype feature, weight, dimension, and score tables."),
        ("## 6. Statistical Analysis", "Equal, PCA, variance, stability, entropy, bootstrap, and shrunk ensemble candidate weights are compared."),
        ("## 7. Machine Learning", "No predictive model is trained. PCA/MI/bootstrap are used only as unsupervised weighting evidence."),
        ("## 8. Explainability", "Metric and dimension weights include method components, shrinkage, caps, confidence, and warning flags."),
        ("## 9. Validation", "Ad-hoc verification checks tables, reports, figures, sums, score ranges, role exclusions, and no Experiment 005 artefacts."),
        ("## 10. Conclusions", "Prototype scores are research artefacts for Experiment 005 validation only."),
        ("## 11. Next Experiments", "Experiment 005 should validate, calibrate, and compare scores across robustness checks."),
        ("## Reproduce", f"```bash\nuv run python experiments/004_role_specific_weight_estimation.py --data-root {data_root}\n```"),
    ]
    nb.cells = [nbf.v4.new_markdown_cell(h + ("\n\n" + b if b else "")) for h, b in sections]
    nb.cells += [
        nbf.v4.new_code_cell("import pandas as pd\npd.read_csv('outputs/tables/004_metric_weight_decisions.csv').head()"),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/004_dimension_weight_decisions.csv').head()"),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/004_prototype_role_scores.csv').head()"),
    ]
    nbf.write(nb, path)


def append_methodology(report: dict[str, Any]) -> None:
    path = ROOT / "methodology.md"
    existing = path.read_text(encoding="utf-8")
    if "## Experiment 004" in existing:
        return
    role_lines = "\n".join(f"| {r['role']} | {r['players_scored']} | {r['metrics_used']} | {r['dimensions']} | {r['unstable_weights']} |" for r in report["role_summary"])
    text = f"""
## Experiment 004 — {EXPERIMENT_TITLE}

Date: {report['generated_at']}

### Objective

Estimate role-specific metric and dimension weights in a reproducible, explainable way and produce prototype scoring artefacts for Experiment 005 validation. The scores are not final production scores.

### Football Hypothesis

Role-specific score prototypes should combine data-driven metric evidence within latent dimensions and dimension evidence within each role, while shrinking unstable estimates toward equal weights under small samples or method disagreement.

### Dataset

Source root: `{report['data_root']}` plus Experiment 002 eligibility and Experiment 003 feature-layer tables.

### Normalization Used

Experiment 003 selected normalization methods were applied per role/metric. Lower-is-better metrics are inverted only when the direction registry classifies them safely; manual-review metrics are excluded from scoring.

### Feature Selection

Only metrics that were `candidate_ready`, had Experiment 003 normalization decisions, and belonged to an Experiment 003 latent dimension were eligible. MULTI_ROLE and UNKNOWN players were excluded from coefficient fitting.

### Algorithms

Equal weights, PCA loading weights, variance contribution weights, stability-adjusted weights, entropy weights, bootstrap stability weights, shrinkage-to-equal ensemble weights, metric caps, dimension evidence blending, bootstrap score uncertainty, and rank-sensitivity analysis.

### Evaluation

| Role | Players scored | Metrics used | Dimensions | Unstable weights |
|---|---:|---:|---:|---:|
{role_lines}

### Results

- Normalized feature rows: {report['table_counts']['normalized_feature_matrix']}
- Metric weight decisions: {report['table_counts']['metric_weight_decisions']}
- Dimension weight decisions: {report['table_counts']['dimension_weight_decisions']}
- Prototype dimension scores: {report['table_counts']['prototype_dimension_scores']}
- Prototype role scores: {report['table_counts']['prototype_role_scores']}
- Quality flags: {report['table_counts']['quality_flags']}
- Figures generated: {report['figures_generated']}

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
"""
    path.write_text(existing.rstrip() + "\n\n" + text.strip() + "\n", encoding="utf-8")


def update_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    if "experiments/004_role_specific_weight_estimation.py" in text:
        return
    text = text.replace(
        "uv run python experiments/003_feature_engineering_normalization.py \\\n  --data-root /home/platform/DataPlatform/tmp/master_data_warehouse",
        "uv run python experiments/003_feature_engineering_normalization.py \\\n  --data-root /home/platform/DataPlatform/tmp/master_data_warehouse\nuv run python experiments/004_role_specific_weight_estimation.py \\\n  --data-root /home/platform/DataPlatform/tmp/master_data_warehouse",
    )
    text += "\n\n## Experiment 004\n\nRole-specific weight estimation and prototype score formulas. Outputs metric/dimension weights, prototype dimension scores, prototype role scores, sensitivity analysis, and quality flags. These are prototype research scores only and must not be treated as final production scores.\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/platform/DataPlatform/tmp/master_data_warehouse")
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    frames = build_input_frames(data_root)
    feature_matrix, direction_registry, eligible = prepare_feature_matrix(frames)
    global metric_weight_methods_global
    metric_weight_methods, metric_weight_decisions = estimate_metric_weights(feature_matrix, direction_registry, frames)
    metric_weight_methods_global = metric_weight_methods.copy()
    dim_methods, dim_decisions = estimate_dimension_weights(metric_weight_decisions, frames, ROOT / "outputs/tables/003_feature_redundancy.csv")
    dim_scores, role_scores, sensitivity, rank_instability, method_scores, fitted_feature_rows = compute_scores(feature_matrix, metric_weight_decisions, dim_decisions, eligible)
    qflags = quality_flags(metric_weight_decisions, dim_decisions, role_scores, direction_registry)

    tables = {
        "004_normalized_feature_matrix": feature_matrix,
        "004_metric_direction_registry": direction_registry,
        "004_metric_weight_methods": metric_weight_methods,
        "004_metric_weight_decisions": metric_weight_decisions,
        "004_dimension_weight_methods": dim_methods,
        "004_dimension_weight_decisions": dim_decisions,
        "004_prototype_dimension_scores": dim_scores,
        "004_prototype_role_scores": role_scores,
        "004_score_sensitivity_analysis": sensitivity,
        "004_rank_instability_players": rank_instability,
        "004_quality_flags": qflags,
    }
    for name, df in tables.items():
        df.to_csv(ROOT / f"outputs/tables/{name}.csv", index=False)

    figure_paths = []
    for role in ROLES:
        figure_paths.extend(save_figures(role, metric_weight_decisions, dim_decisions, dim_scores, role_scores, sensitivity, method_scores))

    role_summary = []
    for role in ROLES:
        role_summary.append({
            "role": role,
            "players_scored": int(role_scores[role_scores.role == role].player_id.nunique()) if not role_scores.empty else 0,
            "metrics_used": int(metric_weight_decisions[metric_weight_decisions.role == role].metric.nunique()),
            "dimensions": int(dim_decisions[dim_decisions.role == role].dimension_name.nunique()),
            "unstable_weights": int(metric_weight_decisions[(metric_weight_decisions.role == role) & (metric_weight_decisions.is_unstable)].shape[0]),
            "manual_review_metrics": int(direction_registry[(direction_registry.role_family == role) & (direction_registry.direction == "manual_review_required")].metric.nunique()),
            "quality_flags": int(qflags[qflags.role == role].shape[0]) if not qflags.empty else 0,
        })
    report = {
        "experiment_id": EXPERIMENT_ID, "title": EXPERIMENT_TITLE, "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root), "prototype_only": True, "production_final": False,
        "role_summary": role_summary,
        "table_counts": {k.replace("004_", ""): int(len(v)) for k, v in tables.items()},
        "figures_generated": len(figure_paths), "figure_paths": figure_paths,
        "limitations": ["local sample only", "not full multi-competition/two-season population", "manual-review directions excluded", "prototype scores require Experiment 005 validation"],
    }
    write_json(ROOT / "outputs/reports/004_role_specific_weight_estimation.json", report)
    md = ROOT / "outputs/reports/004_role_specific_weight_estimation.md"
    md.write_text(
        f"# Experiment 004 — {EXPERIMENT_TITLE}\n\n"
        f"Generated: {report['generated_at']}\n\n"
        "This is a prototype scoring layer only. It is not a final production score.\n\n"
        "## Objective\nEstimate role-specific metric and dimension weights using data evidence from Experiments 002 and 003.\n\n"
        f"## Dataset used\n`{data_root}` plus Experiment 002/003 tables.\n\n"
        "## Eligible populations per role\n" + "\n".join(f"- {r['role']}: players={r['players_scored']}, metrics={r['metrics_used']}, dimensions={r['dimensions']}" for r in role_summary) + "\n\n"
        "## Metrics excluded and why\nManual-review direction metrics and non-candidate-ready metrics are excluded from prototype scoring; all remain visible in `004_metric_direction_registry.csv`.\n\n"
        "## Direction registry\nDirections are transparent rule-based classifications. Unclear contextual metrics are marked manual_review_required and excluded.\n\n"
        "## Metric weighting methodology\nEqual, PCA loading, variance contribution, stability-adjusted, entropy, bootstrap stability, and shrinkage ensemble with caps.\n\n"
        "## Dimension weighting methodology\nBlend explained variance, reliable metric count/stability, bootstrap consistency, PCA contribution, redundancy penalty, and small-sample penalty.\n\n"
        "## Prototype score formula\nMetric scores are normalized and direction-oriented, combined into 0-100 dimension scores, then combined by dimension weights into 0-100 prototype role scores.\n\n"
        "## Sensitivity analysis\nRank correlations across equal/PCA/stability/entropy/ensemble methods and rank-instability players are exported.\n\n"
        "## Main findings\nSee JSON report and role summary tables.\n\n"
        "## Limitations\nLocal sample only; not production-final; manual review required for some metric directions; Experiment 005 validation required.\n\n"
        "## Why this is not production-final yet\nNo cross-season/cross-league validation, no calibration, limited local population, and football review still required.\n\n"
        "## What Experiment 005 should validate next\nRobustness, calibration, rank stability, confidence intervals, league/season fairness, and football interpretability.\n",
        encoding="utf-8",
    )
    write_notebook(ROOT / "notebooks/004_role_specific_weight_estimation.ipynb", data_root)
    append_methodology(report)
    update_readme()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
