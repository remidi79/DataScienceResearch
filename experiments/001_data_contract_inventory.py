from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd
import seaborn as sns

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_score_engine_research.analysis import screen_features
from football_score_engine_research.io import flatten_metrics, read_jsonl, write_json
from football_score_engine_research.roles import infer_role_family

EXPERIMENT_ID = "001"
EXPERIMENT_TITLE = "StatsBomb Data Contract and Metric Universe Inventory"


def metric_inventory(rows: list[dict[str, Any]], grain: str) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    numeric_counts: Counter[str] = Counter()
    examples: dict[str, Any] = {}
    for row in rows:
        metrics = row.get("metrics") or {}
        for key, value in metrics.items():
            counts[key] += 1
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_counts[key] += 1
            examples.setdefault(key, value)
    total = len(rows)
    return pd.DataFrame(
        [
            {
                "grain": grain,
                "metric": key,
                "rows_present": counts[key],
                "coverage_rate": counts[key] / total if total else 0,
                "numeric_rows": numeric_counts[key],
                "numeric_coverage_rate": numeric_counts[key] / total if total else 0,
                "example_value": examples.get(key),
            }
            for key in sorted(counts)
        ]
    )


def first_position_text(lineup_row: dict[str, Any]) -> str | None:
    positions = lineup_row.get("positions") or []
    if isinstance(positions, str):
        try:
            positions = json.loads(positions)
        except Exception:
            return positions
    if positions and isinstance(positions[0], dict):
        return positions[0].get("position")
    return None


def build_role_map(lineups: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in lineups:
        position = first_position_text(row)
        rows.append(
            {
                "statsbomb_player_id": row.get("player_provider_id"),
                "player_name": row.get("player_name"),
                "team_id": row.get("team_provider_id"),
                "position_text": position,
                "role_family": infer_role_family(position),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Keep the most frequent role observed per player across matches.
    role = (
        df.groupby(["statsbomb_player_id", "player_name", "role_family", "position_text"], dropna=False)
        .size()
        .reset_index(name="observations")
        .sort_values(["statsbomb_player_id", "observations"], ascending=[True, False])
        .drop_duplicates("statsbomb_player_id")
    )
    return role


def event_inventory(events: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    type_counts = Counter(row.get("event_type") or "UNKNOWN" for row in events)
    event_type_df = pd.DataFrame(
        [{"event_type": key, "events": value, "share": value / len(events)} for key, value in type_counts.items()]
    ).sort_values("events", ascending=False)
    raw_keys: Counter[str] = Counter()
    model_fields = [
        "obv_total_net",
        "obv_for_net",
        "obv_against_net",
        "shot",
        "pass",
        "carry",
        "duel",
        "under_pressure",
        "counterpress",
        "possession",
        "possession_team",
        "play_pattern",
    ]
    for row in events:
        raw = row.get("raw_payload") or {}
        for key in model_fields:
            if key in raw and raw.get(key) is not None:
                raw_keys[key] += 1
    raw_df = pd.DataFrame(
        [{"raw_payload_field": key, "events_present": raw_keys[key], "coverage_rate": raw_keys[key] / len(events)} for key in model_fields]
    ).sort_values("events_present", ascending=False)
    return event_type_df, raw_df


def save_barplot(df: pd.DataFrame, x: str, y: str, path: Path, title: str, top_n: int | None = None) -> None:
    plot_df = df.head(top_n) if top_n else df
    plt.figure(figsize=(12, 7))
    sns.barplot(data=plot_df, x=x, y=y, color="#285C7D")
    plt.title(title)
    plt.xlabel(x.replace("_", " "))
    plt.ylabel(y.replace("_", " "))
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def append_methodology(methodology_path: Path, report: dict[str, Any]) -> None:
    marker = f"## Experiment {EXPERIMENT_ID}"
    existing = methodology_path.read_text(encoding="utf-8") if methodology_path.exists() else "# Football Score Engine Research Journal\n\n"
    if marker in existing:
        return
    section = f"""
## Experiment {EXPERIMENT_ID} — {EXPERIMENT_TITLE}

Date: {report['generated_at']}

### Objective

Inventory the local StatsBomb event and direct provider-stat data contracts before building any score engine. Establish available grains, metric counts, event coverage, role-family feasibility, redundancy signals, and first PCA variance signals.

### Football Hypothesis

Provider aggregate stats and raw events expose different football dimensions. Direct StatsBomb player/team stats should be treated as provider-truth aggregate facts; event data should provide contextual sequence, possession, pressure, OBV, and action-level evidence. Role-specific score engines are feasible only if positional family coverage and metric availability are explicit.

### Dataset

Source root: `{report['data_root']}`

Rows audited:

| Dataset | Rows |
|---|---:|
{chr(10).join(f"| {k} | {v} |" for k, v in report['row_counts'].items())}

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

- Metric universe: {report['metric_universe_total']} unique direct provider metric names across player/team match/season grains.
- Event rows: {report['row_counts'].get('silver_events', 0)}.
- Player-season rows: {report['row_counts'].get('player_season_stats_direct', 0)}.
- High-correlation player-season metric pairs: {report['player_season_screen']['high_correlation_pairs']}.
- Near-zero variance player-season metrics: {report['player_season_screen']['near_zero_variance_count']}.
- PCA components needed for >=80% variance: {report['player_season_screen']['pca_components_for_80pct_variance']}.

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
"""
    methodology_path.write_text(existing.rstrip() + "\n\n" + section.strip() + "\n", encoding="utf-8")


def write_notebook(path: Path, data_root: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# Experiment {EXPERIMENT_ID} — {EXPERIMENT_TITLE}"),
        nbf.v4.new_markdown_cell("## 1. Objective\n\nInventory StatsBomb events, lineups, matches, and direct provider-stat marts before constructing any score. The goal is to establish metric availability, role-family feasibility, redundancy risk, and first latent-variance signals."),
        nbf.v4.new_markdown_cell("## 2. Football hypothesis\n\nProvider aggregate stats represent stable performance summaries, while raw events preserve contextual action evidence. Composite scores must respect both and remain role-specific."),
        nbf.v4.new_markdown_cell("## 3. Dataset\n\nDataPlatform source root: `" + str(data_root) + "`. Audited grains: player-match stats, team-match stats, player-season stats, team-season stats, raw silver events, lineups, and matches."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nDATA_ROOT = Path('" + str(data_root) + "')\nprint(DATA_ROOT)\npd.read_csv('outputs/tables/001_dataset_row_counts.csv')"),
        nbf.v4.new_markdown_cell("## 4. Feature engineering\n\nMetrics are flattened from provider `metrics` maps. Role family is inferred from lineup position text. No production score features are selected in this inventory experiment."),
        nbf.v4.new_code_cell("role_counts = pd.read_csv('outputs/tables/001_role_family_counts.csv')\nrole_counts"),
        nbf.v4.new_markdown_cell("## 5. Exploratory Data Analysis\n\nProfile dataset sizes, event type distribution, direct-provider metric counts by grain, and raw event payload field coverage."),
        nbf.v4.new_code_cell("metric_counts = pd.read_csv('outputs/tables/001_metric_inventory.csv').groupby('grain').metric.nunique().sort_values(ascending=False)\nmetric_counts"),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/001_event_type_counts.csv').head(20)"),
        nbf.v4.new_markdown_cell("## 6. Statistical Analysis\n\nCompute missingness, near-zero variance, high-correlation pairs, and PCA cumulative variance for player-season provider metrics."),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/001_player_season_missingness.csv').head(20)"),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/001_player_season_high_correlations.csv').head(20)"),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/001_player_season_pca_variance.csv').head(10)"),
        nbf.v4.new_markdown_cell("## 7. Machine Learning\n\nOnly an exploratory robust-scaled PCA is used here. No supervised model is trained and no production scoring model is saved in Experiment 001."),
        nbf.v4.new_markdown_cell("## 8. Explainability\n\nExplainability at this stage is contract-level: metric provenance, metric coverage, role-family coverage, and correlation redundancy tables. Later experiments will add coefficients, loadings, permutation importance, and SHAP where applicable."),
        nbf.v4.new_markdown_cell("## 9. Validation\n\nThe experiment validates required input files exist, writes deterministic CSV/PNG/report artefacts, validates notebook structure through nbformat, and appends the methodology journal once."),
        nbf.v4.new_markdown_cell("## 10. Conclusions\n\nSee `outputs/reports/001_data_contract_inventory.md` and the appended Experiment 001 section in `methodology.md`. The local root is suitable for initial score-engine research, but it is not yet the full multi-competition/two-season target population."),
        nbf.v4.new_markdown_cell("## 11. Next Experiments\n\nExperiment 002: role-family and minutes-threshold validation. Experiment 003: normalization comparison. Experiment 004+: score-family research by role."),
        nbf.v4.new_markdown_cell("## Reproduce\n\n```bash\nuv run python experiments/001_data_contract_inventory.py --data-root " + str(data_root) + "\n```"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/platform/DataPlatform/tmp/master_data_warehouse")
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()

    paths = {
        "player_match_stats_direct": data_root / "marts_v2/mart_statsbomb_player_match_stats_direct_v1.jsonl",
        "team_match_stats_direct": data_root / "marts_v2/mart_statsbomb_team_match_stats_direct_v1.jsonl",
        "player_season_stats_direct": data_root / "marts_v2/mart_statsbomb_player_season_stats_direct_v1.jsonl",
        "team_season_stats_direct": data_root / "marts_v2/mart_statsbomb_team_season_stats_direct_v1.jsonl",
        "silver_events": data_root / "silver/silver_events.jsonl",
        "silver_lineups": data_root / "silver/silver_lineups.jsonl",
        "silver_matches": data_root / "silver/silver_matches.jsonl",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required datasets under {data_root}: {missing}")

    rows = {name: read_jsonl(path) for name, path in paths.items()}
    row_counts = {name: len(value) for name, value in rows.items()}
    pd.DataFrame([{"dataset": k, "rows": v} for k, v in row_counts.items()]).to_csv(ROOT / "outputs/tables/001_dataset_row_counts.csv", index=False)

    metric_frames = []
    for name in ["player_match_stats_direct", "team_match_stats_direct", "player_season_stats_direct", "team_season_stats_direct"]:
        metric_frames.append(metric_inventory(rows[name], name))
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(ROOT / "outputs/tables/001_metric_inventory.csv", index=False)

    event_types, raw_field_coverage = event_inventory(rows["silver_events"])
    event_types.to_csv(ROOT / "outputs/tables/001_event_type_counts.csv", index=False)
    raw_field_coverage.to_csv(ROOT / "outputs/tables/001_event_raw_field_coverage.csv", index=False)

    role_map = build_role_map(rows["silver_lineups"])
    role_map.to_csv(ROOT / "outputs/tables/001_role_family_map.csv", index=False)
    role_counts = role_map["role_family"].value_counts(dropna=False).rename_axis("role_family").reset_index(name="players")
    role_counts.to_csv(ROOT / "outputs/tables/001_role_family_counts.csv", index=False)

    player_season = flatten_metrics(paths["player_season_stats_direct"], id_fields=["statsbomb_player_id", "player_name", "team_id", "team_name", "competition_id", "season_id"])
    player_season = player_season.merge(role_map[["statsbomb_player_id", "role_family", "position_text"]], on="statsbomb_player_id", how="left")
    player_season["role_family"] = player_season["role_family"].fillna("UNKNOWN")
    player_season.to_csv(ROOT / "outputs/tables/001_player_season_flat_sample.csv", index=False)

    screen = screen_features(player_season, corr_threshold=0.90)
    screen.missingness.to_csv(ROOT / "outputs/tables/001_player_season_missingness.csv", index=False)
    screen.correlated_pairs.to_csv(ROOT / "outputs/tables/001_player_season_high_correlations.csv", index=False)
    screen.pca_variance.to_csv(ROOT / "outputs/tables/001_player_season_pca_variance.csv", index=False)
    pd.DataFrame({"metric": screen.near_zero_variance}).to_csv(ROOT / "outputs/tables/001_player_season_near_zero_variance.csv", index=False)

    save_barplot(pd.DataFrame([{"dataset": k, "rows": v} for k, v in row_counts.items()]).sort_values("rows", ascending=False), "rows", "dataset", ROOT / "outputs/figures/001_row_counts.png", "Dataset row counts")
    save_barplot(event_types, "events", "event_type", ROOT / "outputs/figures/001_event_type_counts.png", "Top StatsBomb event types", top_n=20)
    metric_counts = metrics.groupby("grain")["metric"].nunique().reset_index(name="metric_count").sort_values("metric_count", ascending=False)
    save_barplot(metric_counts, "metric_count", "grain", ROOT / "outputs/figures/001_metric_counts_by_grain.png", "Direct provider metrics by grain")
    save_barplot(role_counts, "players", "role_family", ROOT / "outputs/figures/001_role_family_counts.png", "Players by inferred role family")
    if not screen.pca_variance.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=screen.pca_variance, x="component", y="cumulative_variance", marker="o")
        plt.axhline(0.80, color="#9B2C2C", linestyle="--", linewidth=1)
        plt.title("Player-season PCA cumulative variance")
        plt.tight_layout()
        plt.savefig(ROOT / "outputs/figures/001_player_season_pca_variance.png", dpi=180)
        plt.close()

    pca_components_80 = None
    if not screen.pca_variance.empty:
        above = screen.pca_variance[screen.pca_variance["cumulative_variance"] >= 0.80]
        pca_components_80 = int(above.index[0] + 1) if not above.empty else None

    report = {
        "experiment_id": EXPERIMENT_ID,
        "title": EXPERIMENT_TITLE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "row_counts": row_counts,
        "metric_universe_total": int(metrics["metric"].nunique()),
        "metrics_by_grain": metric_counts.to_dict(orient="records"),
        "role_family_counts": role_counts.to_dict(orient="records"),
        "player_season_screen": {
            "numeric_columns": len(screen.numeric_columns),
            "near_zero_variance_count": len(screen.near_zero_variance),
            "high_correlation_pairs": int(len(screen.correlated_pairs)),
            "pca_components_for_80pct_variance": pca_components_80,
        },
        "outputs": {
            "notebook": f"notebooks/{EXPERIMENT_ID}_data_contract_inventory.ipynb",
            "tables": "outputs/tables/001_*.csv",
            "figures": "outputs/figures/001_*.png",
        },
    }
    write_json(ROOT / "outputs/reports/001_data_contract_inventory.json", report)

    md_report = ROOT / "outputs/reports/001_data_contract_inventory.md"
    md_report.write_text(
        f"# Experiment {EXPERIMENT_ID} — {EXPERIMENT_TITLE}\n\n"
        f"Generated: {report['generated_at']}\n\n"
        f"Data root: `{data_root}`\n\n"
        f"Metric universe: {report['metric_universe_total']} unique metric names.\n\n"
        f"High-correlation player-season pairs: {report['player_season_screen']['high_correlation_pairs']}\n\n"
        f"PCA components for >=80% variance: {pca_components_80}\n\n"
        "See CSV tables and PNG figures for evidence.\n",
        encoding="utf-8",
    )
    write_notebook(ROOT / f"notebooks/{EXPERIMENT_ID}_data_contract_inventory.ipynb", data_root)
    append_methodology(ROOT / "methodology.md", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
