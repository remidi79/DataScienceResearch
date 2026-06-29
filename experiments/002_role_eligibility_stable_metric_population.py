from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_score_engine_research.io import flatten_metrics, read_jsonl, write_json
from football_score_engine_research.roles import ROLE_FAMILIES, infer_role_family

EXPERIMENT_ID = "002"
EXPERIMENT_TITLE = "Role Eligibility & Stable Metric Population"
ROLE_MINUTE_THRESHOLDS = {"GK": 600.0, "CB": 900.0, "FB": 900.0, "MID": 900.0, "WINGER": 750.0, "CF": 750.0}
ROLE_ASSIGNMENT_SHARE = 0.70
MIN_METRIC_COVERAGE = 0.40
MIN_SPLIT_HALF_RELIABILITY = 0.60
MAX_REASONABLE_CV = 3.0
CORRELATION_THRESHOLD = 0.90

# User-requested first-pass football metric families, mapped to actual StatsBomb provider metric aliases.
CANDIDATE_SCORE_METRICS = {
    "GK": {
        "shot_stopping": ["save_ratio", "np_psxg_faced_90", "goals_faced_90", "obv_gk_90", "ot_shots_faced_ratio"],
        "aerial_command": ["claim_success", "aerial_ratio", "aerial_wins_90", "crossing_ratio"],
        "distribution": ["passing_ratio", "obv_pass_90", "pass_length", "p_pass_length"],
        "sweeping": ["average_x_defensive_action"],
    },
    "CB": {
        "defensive_contribution": ["interceptions_90", "ball_recoveries_90", "aerial_ratio", "clearance_90", "blocks_per_shot", "obv_defensive_action_90"],
        "ball_progression": ["op_f3_passes_90", "lbp_pass_ratio", "obv_pass_90", "xgbuildup_90", "op_xgbuildup_90"],
    },
    "FB": {
        "ball_progression": ["obv_dribble_carry_90", "op_f3_passes_90", "lbp_pass_ratio", "obv_pass_90"],
        "wide_creation": ["crossing_ratio", "box_cross_ratio", "dribble_ratio", "key_passes_90", "xa_90"],
        "defensive_work": ["ball_recoveries_90", "pressures_90", "counterpressures_90"],
    },
    "MID": {
        "possession_and_progression": ["op_passes_90", "passing_ratio", "op_f3_passes_90", "obv_dribble_carry_90", "obv_pass_90", "xgbuildup_90"],
        "defensive_work": ["ball_recoveries_90", "pressures_90", "counterpressures_90"],
        "creation": ["key_passes_90", "xa_90", "through_balls_90"],
    },
    "WINGER": {
        "chance_creation": ["obv_dribble_carry_90", "dribble_ratio", "crossing_ratio", "key_passes_90", "xa_90", "touches_inside_box_90", "obv_pass_90"],
        "goal_threat": ["np_xg_90", "np_shots_90", "shot_on_target_ratio", "obv_shot_90"],
    },
    "CF": {
        "finishing": ["np_xg_90", "goals_90", "np_shots_90", "shot_on_target_ratio", "np_xg_per_shot", "obv_shot_90"],
        "box_presence": ["touches_inside_box_90", "aerial_ratio", "aerial_wins_90"],
        "link_play": ["op_xgchain_90", "xgbuildup_90", "op_passes_into_and_touches_inside_box_90"],
    },
}


def parse_time_to_minutes(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    if isinstance(value, (int, float)):
        return float(value) / 60.0 if value > 130 else float(value)
    text = str(value)
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 60.0 + minutes + seconds / 60.0
        if len(parts) == 2:
            return float(parts[0]) + float(parts[1]) / 60.0
    except ValueError:
        return None
    return None


def lineup_positions(row: dict[str, Any]) -> list[dict[str, Any]]:
    positions = row.get("positions") or []
    if isinstance(positions, str):
        try:
            positions = json.loads(positions)
        except Exception:
            return []
    return positions if isinstance(positions, list) else []


def build_player_match_minutes(player_match_df: pd.DataFrame) -> pd.DataFrame:
    if "minutes" not in player_match_df.columns:
        player_match_df["minutes"] = np.nan
    cols = ["statsbomb_player_id", "match_provider_id", "player_name", "team_id", "team_name", "minutes"]
    return player_match_df[[c for c in cols if c in player_match_df.columns]].copy()


def resolve_roles(lineups: list[dict[str, Any]], player_match_minutes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_minutes = {
        (str(row.statsbomb_player_id), str(row.match_provider_id)): float(row.minutes) if pd.notna(row.minutes) else np.nan
        for row in player_match_minutes.itertuples(index=False)
    }
    role_minutes: dict[str, Counter[str]] = defaultdict(Counter)
    role_examples: dict[str, Counter[str]] = defaultdict(Counter)
    names: dict[str, str] = {}
    for row in lineups:
        player_id = str(row.get("player_provider_id"))
        match_id = str(row.get("match_provider_id"))
        if player_id in ("None", ""):
            continue
        names[player_id] = row.get("player_name") or names.get(player_id, "")
        positions = lineup_positions(row)
        total_match_minutes = match_minutes.get((player_id, match_id), np.nan)
        if not positions:
            role_minutes[player_id]["UNKNOWN"] += float(total_match_minutes) if pd.notna(total_match_minutes) else 0.0
            continue
        segment_rows = []
        for pos in positions:
            role = infer_role_family(pos.get("position"))
            start = parse_time_to_minutes(pos.get("from")) or 0.0
            end = parse_time_to_minutes(pos.get("to"))
            if end is None:
                end = float(total_match_minutes) if pd.notna(total_match_minutes) and total_match_minutes > start else 90.0
            minutes = max(0.0, end - start)
            segment_rows.append((role, minutes, pos.get("position") or ""))
        segment_total = sum(minutes for _, minutes, _ in segment_rows)
        if pd.notna(total_match_minutes) and segment_total > 0:
            scale = float(total_match_minutes) / segment_total
        else:
            scale = 1.0
        for role, minutes, position_text in segment_rows:
            role_minutes[player_id][role] += minutes * scale
            role_examples[player_id][position_text] += 1
    rows = []
    detail_rows = []
    for player_id, counter in role_minutes.items():
        total = sum(counter.values())
        if total <= 0:
            assigned = "UNKNOWN"
            dominant_share = 0.0
            dominant_role = "UNKNOWN"
        else:
            dominant_role, dominant_minutes = counter.most_common(1)[0]
            dominant_share = dominant_minutes / total
            if dominant_role == "UNKNOWN":
                assigned = "UNKNOWN"
            elif dominant_share >= ROLE_ASSIGNMENT_SHARE:
                assigned = dominant_role
            else:
                assigned = "MULTI_ROLE"
        rows.append(
            {
                "statsbomb_player_id": player_id,
                "player_name": names.get(player_id, ""),
                "assigned_role": assigned,
                "dominant_role": dominant_role,
                "dominant_role_share": dominant_share,
                "lineup_role_minutes": total,
                "position_examples": "; ".join([p for p, _ in role_examples[player_id].most_common(3)]),
            }
        )
        for role, minutes in counter.items():
            detail_rows.append({"statsbomb_player_id": player_id, "role_family": role, "role_minutes": minutes, "role_minute_share": minutes / total if total else 0.0})
    return pd.DataFrame(rows), pd.DataFrame(detail_rows)


def split_half_reliability(player_match_df: pd.DataFrame, role_map: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    base = player_match_df.merge(role_map[["statsbomb_player_id", "assigned_role"]], on="statsbomb_player_id", how="left")
    base = base[base["assigned_role"].isin(ROLE_MINUTE_THRESHOLDS)]
    rows = []
    sort_cols = [c for c in ["statsbomb_player_id", "match_provider_id"] if c in base.columns]
    base = base.sort_values(sort_cols).copy()
    base["split_index"] = base.groupby("statsbomb_player_id").cumcount() % 2
    for role, group in base.groupby("assigned_role"):
        for metric in metrics:
            if metric not in group.columns:
                continue
            pivot = group.pivot_table(index="statsbomb_player_id", columns="split_index", values=metric, aggfunc="mean")
            if 0 not in pivot.columns or 1 not in pivot.columns:
                reliability = np.nan
                n_players = 0
            else:
                paired = pivot[[0, 1]].dropna()
                n_players = len(paired)
                reliability = paired[0].corr(paired[1]) if n_players >= 3 and paired[0].nunique() > 1 and paired[1].nunique() > 1 else np.nan
            rows.append({"role_family": role, "metric": metric, "split_half_reliability": reliability, "split_half_players": n_players})
    return pd.DataFrame(rows)


def bootstrap_ci(values: pd.Series, n_bootstrap: int = 500, seed: int = 42) -> tuple[float | None, float | None]:
    clean = values.dropna().astype(float).to_numpy()
    if len(clean) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(clean, size=len(clean), replace=True).mean()) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def metric_stability(player_season: pd.DataFrame, player_match: pd.DataFrame, role_map: pd.DataFrame, excluded_corr_metrics: set[str]) -> pd.DataFrame:
    merged = player_season.merge(role_map[["statsbomb_player_id", "assigned_role", "dominant_role_share"]], on="statsbomb_player_id", how="left")
    merged["assigned_role"] = merged["assigned_role"].fillna("UNKNOWN")
    numeric_metrics = [c for c in merged.select_dtypes(include=["number"]).columns if c not in {"competition_id", "season_id", "team_id", "statsbomb_player_id"}]
    if "minutes" in numeric_metrics:
        numeric_metrics.remove("minutes")
    reliability = split_half_reliability(player_match, role_map, numeric_metrics)
    reliability_map = {(r.role_family, r.metric): r for r in reliability.itertuples(index=False)}
    rows = []
    for role, threshold in ROLE_MINUTE_THRESHOLDS.items():
        group = merged[(merged["assigned_role"] == role) & (merged["minutes"].fillna(0) >= threshold)]
        n_players = len(group)
        for metric in numeric_metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            non_null = int(values.notna().sum())
            coverage = non_null / n_players if n_players else 0.0
            variance = float(values.var()) if non_null >= 2 else np.nan
            mean = float(values.mean()) if non_null else np.nan
            std = float(values.std()) if non_null >= 2 else np.nan
            cv = abs(std / mean) if mean and not math.isnan(mean) and std is not None and not math.isnan(std) else np.nan
            ci_low, ci_high = bootstrap_ci(values)
            rel_row = reliability_map.get((role, metric))
            rel = float(rel_row.split_half_reliability) if rel_row is not None and pd.notna(rel_row.split_half_reliability) else np.nan
            split_n = int(rel_row.split_half_players) if rel_row is not None else 0
            exclusion_reasons = []
            if n_players == 0:
                exclusion_reasons.append("no_eligible_players_for_role_threshold")
            if coverage < MIN_METRIC_COVERAGE:
                exclusion_reasons.append("coverage_lt_40pct")
            if pd.isna(variance) or variance <= 1e-12:
                exclusion_reasons.append("near_zero_variance")
            if metric in excluded_corr_metrics:
                exclusion_reasons.append("highly_correlated_ge_0_90")
            if pd.notna(cv) and cv > MAX_REASONABLE_CV:
                exclusion_reasons.append("high_coefficient_of_variation")
            if pd.notna(rel) and rel < MIN_SPLIT_HALF_RELIABILITY:
                exclusion_reasons.append("split_half_reliability_lt_0_60")
            rows.append(
                {
                    "role_family": role,
                    "metric": metric,
                    "eligible_players": n_players,
                    "non_null_players": non_null,
                    "coverage_rate": coverage,
                    "mean": mean,
                    "std": std,
                    "coefficient_of_variation": cv,
                    "variance": variance,
                    "bootstrap_mean_ci_low": ci_low,
                    "bootstrap_mean_ci_high": ci_high,
                    "split_half_reliability": rel,
                    "split_half_players": split_n,
                    "excluded_from_initial_models": bool(exclusion_reasons),
                    "exclusion_reasons": ";".join(exclusion_reasons),
                }
            )
    return pd.DataFrame(rows)


def candidate_metric_status(stability: pd.DataFrame, available_metrics: set[str]) -> pd.DataFrame:
    rows = []
    for role, families in CANDIDATE_SCORE_METRICS.items():
        for score_family, metrics in families.items():
            for metric in metrics:
                status_rows = stability[(stability["role_family"] == role) & (stability["metric"] == metric)]
                if metric not in available_metrics:
                    status = "missing_from_provider_stats"
                    reason = "metric_alias_not_found_in_player_season_stats"
                    eligible = 0
                    coverage = np.nan
                    reliability = np.nan
                elif status_rows.empty:
                    status = "not_evaluated"
                    reason = "not_numeric_or_no_role_population"
                    eligible = 0
                    coverage = np.nan
                    reliability = np.nan
                else:
                    row = status_rows.iloc[0]
                    status = "candidate_ready" if not bool(row["excluded_from_initial_models"]) else "candidate_blocked"
                    reason = row["exclusion_reasons"]
                    eligible = int(row["eligible_players"])
                    coverage = row["coverage_rate"]
                    reliability = row["split_half_reliability"]
                rows.append(
                    {
                        "role_family": role,
                        "score_family": score_family,
                        "requested_metric_alias": metric,
                        "status": status,
                        "eligible_players": eligible,
                        "coverage_rate": coverage,
                        "split_half_reliability": reliability,
                        "reason": reason,
                    }
                )
    return pd.DataFrame(rows)


def save_barplot(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    plt.figure(figsize=(11, 7))
    sns.barplot(data=df, x=x, y=y, color="#285C7D")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def write_notebook(path: Path, data_root: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# Experiment {EXPERIMENT_ID} — {EXPERIMENT_TITLE}"),
        nbf.v4.new_markdown_cell("## 1. Objective\n\nDefine reference populations for all future score engines: role eligibility, minutes thresholds, UNKNOWN/MULTI_ROLE handling, and metric stability screens."),
        nbf.v4.new_markdown_cell("## 2. Football hypothesis\n\nRole-specific football scores are only defensible when populations are stable. Players with insufficient minutes or genuinely mixed roles should not define early model coefficients."),
        nbf.v4.new_markdown_cell("## 3. Dataset\n\nDataPlatform source root: `" + str(data_root) + "`. Uses direct StatsBomb player-season/player-match stats plus lineups for role-minute resolution."),
        nbf.v4.new_code_cell("import pandas as pd\nsummary = pd.read_csv('outputs/tables/002_role_eligibility_summary.csv')\nsummary"),
        nbf.v4.new_markdown_cell("## 4. Feature engineering\n\nLineup position intervals are converted into role minutes. A player is assigned to a role only when one role has at least 70% of observed role minutes; otherwise the player becomes MULTI_ROLE. Eligibility thresholds are GK 600, CB/FB/MID 900, WINGER/CF 750 minutes."),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/002_role_resolution.csv').head(20)"),
        nbf.v4.new_markdown_cell("## 5. Exploratory Data Analysis\n\nInspect role resolution counts, eligible populations, and candidate metric availability."),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/002_candidate_metric_status.csv').head(40)"),
        nbf.v4.new_markdown_cell("## 6. Statistical Analysis\n\nFor each role and metric: coverage, variance, coefficient of variation, bootstrap mean confidence interval, and split-half reliability from player-match data where possible."),
        nbf.v4.new_code_cell("stability = pd.read_csv('outputs/tables/002_metric_stability_by_role.csv')\nstability.head(20)"),
        nbf.v4.new_code_cell("stability[~stability.excluded_from_initial_models].groupby('role_family').metric.nunique()"),
        nbf.v4.new_markdown_cell("## 7. Machine Learning\n\nNo coefficient model is trained in this experiment. The output defines the eligible modelling population and metric screen for later normalization, feature selection, and interpretable model experiments."),
        nbf.v4.new_markdown_cell("## 8. Explainability\n\nEvery metric has an explicit inclusion/exclusion reason: low coverage, near-zero variance, high correlation, high CV, or low split-half reliability."),
        nbf.v4.new_markdown_cell("## 9. Validation\n\nThe script validates required input files, writes deterministic tables/figures/reports, updates methodology append-only, and is checked by the ad-hoc verification command described in the final report."),
        nbf.v4.new_markdown_cell("## 10. Conclusions\n\nUse eligible role populations and stable metric screens as gates before any score weighting. UNKNOWN and MULTI_ROLE players should not train initial coefficients."),
        nbf.v4.new_markdown_cell("## 11. Next Experiments\n\nExperiment 003 should compare normalization pipelines on stable role-specific metric populations before any final coefficient search."),
        nbf.v4.new_markdown_cell("## Reproduce\n\n```bash\nuv run python experiments/002_role_eligibility_stable_metric_population.py --data-root " + str(data_root) + "\n```"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def append_methodology(path: Path, report: dict[str, Any]) -> None:
    marker = f"## Experiment {EXPERIMENT_ID}"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Football Score Engine Research Journal\n"
    if marker in existing:
        return
    role_lines = "\n".join(
        f"| {row['role_family']} | {row['threshold_minutes']} | {row['assigned_players']} | {row['eligible_players']} | {row['stable_metrics']} |"
        for row in report["role_summary"]
    )
    section = f"""
## Experiment {EXPERIMENT_ID} — {EXPERIMENT_TITLE}

Date: {report['generated_at']}

### Objective

Define the role-specific eligible player populations and stable metric universe that will gate every future score engine.

### Football Hypothesis

Composite scores become unstable and misleading when low-minute players, hybrid role players, or sparse/noisy metrics are allowed to define coefficients. Role families require different minimum-minute thresholds and different metric eligibility screens.

### Dataset

Source root: `{report['data_root']}`

Input rows:

| Dataset | Rows |
|---|---:|
| player season stats direct | {report['input_rows']['player_season']} |
| player match stats direct | {report['input_rows']['player_match']} |
| silver lineups | {report['input_rows']['lineups']} |

### Normalization Tested

No final normalization was selected. This experiment prepares Experiment 003 by defining eligible role populations and stable raw metric candidates. Later normalization tests should compare per-90, robust z-score, rank, quantile, empirical CDF, role percentile, and Bayesian shrinkage on this filtered universe.

### Feature Selection

Initial exclusion gates:

- metric coverage < 40% within eligible role population
- near-zero variance
- membership in a high-correlation pair with abs(correlation) >= 0.90
- unreasonable coefficient of variation > {MAX_REASONABLE_CV}
- split-half reliability < {MIN_SPLIT_HALF_RELIABILITY} when measurable

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
{role_lines}

### Results

- Assigned role players: {report['assignment_counts'].get('assigned_role_players', 0)}
- MULTI_ROLE players: {report['assignment_counts'].get('MULTI_ROLE', 0)}
- UNKNOWN players: {report['assignment_counts'].get('UNKNOWN', 0)}
- Candidate metrics requested: {report['candidate_metric_summary']['requested']}
- Candidate metrics ready: {report['candidate_metric_summary']['ready']}
- Candidate metrics blocked: {report['candidate_metric_summary']['blocked']}
- Candidate metrics missing from provider stats: {report['candidate_metric_summary']['missing']}

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
"""
    path.write_text(existing.rstrip() + "\n\n" + section.strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/platform/DataPlatform/tmp/master_data_warehouse")
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    paths = {
        "player_season": data_root / "marts_v2/mart_statsbomb_player_season_stats_direct_v1.jsonl",
        "player_match": data_root / "marts_v2/mart_statsbomb_player_match_stats_direct_v1.jsonl",
        "lineups": data_root / "silver/silver_lineups.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    lineups = read_jsonl(paths["lineups"])
    player_match = flatten_metrics(paths["player_match"], id_fields=["statsbomb_player_id", "match_provider_id", "player_name", "team_id", "team_name", "competition_id", "season_id"])
    player_season = flatten_metrics(paths["player_season"], id_fields=["statsbomb_player_id", "player_name", "team_id", "team_name", "competition_id", "season_id"])
    for df in [player_match, player_season]:
        df["statsbomb_player_id"] = df["statsbomb_player_id"].astype(str)

    role_resolution, role_minutes_detail = resolve_roles(lineups, build_player_match_minutes(player_match))
    role_resolution["statsbomb_player_id"] = role_resolution["statsbomb_player_id"].astype(str)
    player_total_minutes = player_season[["statsbomb_player_id", "minutes"]].copy() if "minutes" in player_season.columns else pd.DataFrame(columns=["statsbomb_player_id", "minutes"])
    role_resolution = role_resolution.merge(player_total_minutes.rename(columns={"minutes": "season_minutes"}), on="statsbomb_player_id", how="left")
    role_resolution["threshold_minutes"] = role_resolution["assigned_role"].map(ROLE_MINUTE_THRESHOLDS)
    role_resolution["eligible_for_initial_coefficients"] = role_resolution.apply(
        lambda row: bool(row["assigned_role"] in ROLE_MINUTE_THRESHOLDS and pd.notna(row["season_minutes"]) and row["season_minutes"] >= row["threshold_minutes"]), axis=1
    )

    role_resolution.to_csv(ROOT / "outputs/tables/002_role_resolution.csv", index=False)
    role_minutes_detail.to_csv(ROOT / "outputs/tables/002_role_minutes_detail.csv", index=False)

    corr_path = ROOT / "outputs/tables/001_player_season_high_correlations.csv"
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path)
    else:
        numeric = player_season.select_dtypes(include=["number"]).drop(columns=["competition_id", "season_id", "team_id"], errors="ignore")
        corr = numeric.corr().abs()
        pairs = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                value = corr.loc[a, b]
                if pd.notna(value) and value >= CORRELATION_THRESHOLD:
                    pairs.append({"metric_a": a, "metric_b": b, "abs_corr": value})
        corr_df = pd.DataFrame(pairs)
    excluded_corr_metrics = set(corr_df.get("metric_b", pd.Series(dtype=str)).dropna().astype(str))

    stability = metric_stability(player_season, player_match, role_resolution, excluded_corr_metrics)
    stability.to_csv(ROOT / "outputs/tables/002_metric_stability_by_role.csv", index=False)
    stable_metric_counts = (
        stability[~stability["excluded_from_initial_models"]]
        .groupby("role_family")["metric"]
        .nunique()
        .rename("stable_metrics")
        .reset_index()
    )

    assigned_counts = role_resolution["assigned_role"].value_counts(dropna=False).rename_axis("assigned_role").reset_index(name="players")
    assigned_counts.to_csv(ROOT / "outputs/tables/002_role_assignment_counts.csv", index=False)
    role_summary_rows = []
    for role, threshold in ROLE_MINUTE_THRESHOLDS.items():
        assigned = role_resolution[role_resolution["assigned_role"] == role]
        eligible = assigned[assigned["eligible_for_initial_coefficients"]]
        stable_count = int(stable_metric_counts.loc[stable_metric_counts["role_family"] == role, "stable_metrics"].iloc[0]) if role in set(stable_metric_counts["role_family"]) else 0
        role_summary_rows.append(
            {
                "role_family": role,
                "threshold_minutes": threshold,
                "assigned_players": int(len(assigned)),
                "eligible_players": int(len(eligible)),
                "median_minutes": float(assigned["season_minutes"].median()) if len(assigned) else np.nan,
                "stable_metrics": stable_count,
            }
        )
    role_summary = pd.DataFrame(role_summary_rows)
    role_summary.to_csv(ROOT / "outputs/tables/002_role_eligibility_summary.csv", index=False)

    available_metrics = set(player_season.select_dtypes(include=["number"]).columns)
    candidate_status = candidate_metric_status(stability, available_metrics)
    candidate_status.to_csv(ROOT / "outputs/tables/002_candidate_metric_status.csv", index=False)

    save_barplot(assigned_counts, "players", "assigned_role", ROOT / "outputs/figures/002_role_assignment_counts.png", "Role assignment counts")
    save_barplot(role_summary, "eligible_players", "role_family", ROOT / "outputs/figures/002_role_eligible_players.png", "Eligible players by role threshold")
    save_barplot(role_summary, "stable_metrics", "role_family", ROOT / "outputs/figures/002_stable_metrics_by_role.png", "Stable metrics by eligible role population")
    status_counts = candidate_status.groupby("status").size().rename("metrics").reset_index().sort_values("metrics", ascending=False)
    save_barplot(status_counts, "metrics", "status", ROOT / "outputs/figures/002_candidate_metric_status.png", "Requested candidate metric status")

    candidate_summary = {
        "requested": int(len(candidate_status)),
        "ready": int((candidate_status["status"] == "candidate_ready").sum()),
        "blocked": int((candidate_status["status"] == "candidate_blocked").sum()),
        "missing": int((candidate_status["status"] == "missing_from_provider_stats").sum()),
    }
    assignment_counts = {
        "assigned_role_players": int(role_resolution[role_resolution["assigned_role"].isin(ROLE_MINUTE_THRESHOLDS)].shape[0]),
        "MULTI_ROLE": int((role_resolution["assigned_role"] == "MULTI_ROLE").sum()),
        "UNKNOWN": int((role_resolution["assigned_role"] == "UNKNOWN").sum()),
    }
    report = {
        "experiment_id": EXPERIMENT_ID,
        "title": EXPERIMENT_TITLE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "input_rows": {"player_season": int(len(player_season)), "player_match": int(len(player_match)), "lineups": int(len(lineups))},
        "role_assignment_share_threshold": ROLE_ASSIGNMENT_SHARE,
        "role_minute_thresholds": ROLE_MINUTE_THRESHOLDS,
        "assignment_counts": assignment_counts,
        "role_summary": role_summary.to_dict(orient="records"),
        "candidate_metric_summary": candidate_summary,
        "outputs": {
            "notebook": f"notebooks/{EXPERIMENT_ID}_role_eligibility_stable_metric_population.ipynb",
            "tables": "outputs/tables/002_*.csv",
            "figures": "outputs/figures/002_*.png",
        },
    }
    write_json(ROOT / "outputs/reports/002_role_eligibility_stable_metric_population.json", report)
    (ROOT / "outputs/reports/002_role_eligibility_stable_metric_population.md").write_text(
        f"# Experiment {EXPERIMENT_ID} — {EXPERIMENT_TITLE}\n\n"
        f"Generated: {report['generated_at']}\n\n"
        f"Data root: `{data_root}`\n\n"
        f"Assigned role players: {assignment_counts['assigned_role_players']}\n\n"
        f"MULTI_ROLE: {assignment_counts['MULTI_ROLE']}\n\n"
        f"UNKNOWN: {assignment_counts['UNKNOWN']}\n\n"
        f"Candidate metrics: {candidate_summary}\n\n"
        "See CSV tables and PNG figures for detailed evidence.\n",
        encoding="utf-8",
    )
    write_notebook(ROOT / f"notebooks/{EXPERIMENT_ID}_role_eligibility_stable_metric_population.ipynb", data_root)
    append_methodology(ROOT / "methodology.md", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
