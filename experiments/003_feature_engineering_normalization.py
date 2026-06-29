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
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy import stats
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import QuantileTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_score_engine_research.io import flatten_metrics, write_json

EXPERIMENT_ID = "003"
EXPERIMENT_TITLE = "Scientific Feature Engineering & Normalization"
ROLES = ["GK", "CB", "FB", "MID", "WINGER", "CF"]
NORMALIZATION_METHODS = [
    "z_score",
    "robust_z_score",
    "percentile_rank",
    "min_max",
    "log_transform",
    "quantile_transform",
    "winsorized_z_score",
]
REDUNDANCY_THRESHOLD = 0.90
RANDOM_SEED = 42


def finite_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def mad(series: pd.Series) -> float:
    values = finite_series(series)
    if values.empty:
        return float("nan")
    med = values.median()
    return float((values - med).abs().median())


def outlier_pct_iqr(series: pd.Series) -> float:
    values = finite_series(series)
    if len(values) < 4:
        return float("nan")
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if not iqr:
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(((values < lower) | (values > upper)).mean() * 100.0)


def modality(values: pd.Series) -> tuple[bool, int, str]:
    clean = finite_series(values)
    if len(clean) < 8 or clean.nunique() < 4:
        return False, 1, "insufficient_unique_values"
    bins = min(12, max(5, int(math.sqrt(len(clean)))))
    counts, _ = np.histogram(clean.to_numpy(), bins=bins)
    peaks = 0
    for idx in range(1, len(counts) - 1):
        if counts[idx] > counts[idx - 1] and counts[idx] > counts[idx + 1] and counts[idx] > 0:
            peaks += 1
    if counts[0] > counts[1] and counts[0] > 0:
        peaks += 1
    if counts[-1] > counts[-2] and counts[-1] > 0:
        peaks += 1
    return peaks >= 2, int(max(peaks, 1)), f"histogram_peaks={peaks};bins={bins}"


def metric_stats(role: str, metric: str, series: pd.Series, total_n: int) -> dict[str, Any]:
    values = finite_series(series)
    n = len(values)
    missing_pct = 100.0 * (1.0 - n / total_n) if total_n else 100.0
    if n == 0:
        base = {k: np.nan for k in ["mean", "median", "std", "mad", "iqr", "cv", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "min", "max", "skewness", "kurtosis", "outlier_pct", "shapiro_stat", "shapiro_p"]}
        base.update({"role_family": role, "metric": metric, "n": 0, "missing_pct": missing_pct, "heavy_tails": False, "multimodal": False, "mode_count": 0, "modality_method": "no_values"})
        return base
    q = values.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    std = float(values.std(ddof=1)) if n >= 2 else np.nan
    mean = float(values.mean())
    cv = abs(std / mean) if mean and not math.isnan(mean) and not math.isnan(std) else np.nan
    skew = float(values.skew()) if n >= 3 else np.nan
    kurt = float(values.kurtosis()) if n >= 4 else np.nan
    heavy_tails = bool(pd.notna(kurt) and kurt > 3.0)
    multimodal, mode_count, modality_method = modality(values)
    shapiro_stat = shapiro_p = np.nan
    if 3 <= n <= 5000 and values.nunique() >= 3:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(values.to_numpy())
        except Exception:
            shapiro_stat = shapiro_p = np.nan
    return {
        "role_family": role,
        "metric": metric,
        "n": n,
        "mean": mean,
        "median": float(values.median()),
        "std": std,
        "mad": mad(values),
        "iqr": float(q.loc[0.75] - q.loc[0.25]),
        "cv": cv,
        "p5": float(q.loc[0.05]),
        "p10": float(q.loc[0.10]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p90": float(q.loc[0.90]),
        "p95": float(q.loc[0.95]),
        "min": float(values.min()),
        "max": float(values.max()),
        "missing_pct": missing_pct,
        "outlier_pct": outlier_pct_iqr(values),
        "skewness": skew,
        "kurtosis": kurt,
        "heavy_tails": heavy_tails,
        "multimodal": bool(multimodal),
        "mode_count": mode_count,
        "modality_method": modality_method,
        "shapiro_stat": float(shapiro_stat) if pd.notna(shapiro_stat) else np.nan,
        "shapiro_p": float(shapiro_p) if pd.notna(shapiro_p) else np.nan,
    }


def normalize_series(series: pd.Series, method: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = values.dropna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if clean.empty:
        return out
    if method == "z_score":
        std = clean.std(ddof=1)
        out.loc[clean.index] = (clean - clean.mean()) / std if std and not np.isnan(std) else 0.0
    elif method == "robust_z_score":
        m = clean.median()
        mdev = (clean - m).abs().median()
        out.loc[clean.index] = 0.6745 * (clean - m) / mdev if mdev and not np.isnan(mdev) else 0.0
    elif method == "percentile_rank":
        out.loc[clean.index] = clean.rank(pct=True, method="average")
    elif method == "min_max":
        rng = clean.max() - clean.min()
        out.loc[clean.index] = (clean - clean.min()) / rng if rng and not np.isnan(rng) else 0.5
    elif method == "log_transform":
        shifted = clean - clean.min()
        transformed = np.log1p(shifted)
        std = transformed.std(ddof=1)
        out.loc[clean.index] = (transformed - transformed.mean()) / std if std and not np.isnan(std) else 0.0
    elif method == "quantile_transform":
        if len(clean) < 3 or clean.nunique() < 2:
            out.loc[clean.index] = 0.0
        else:
            n_quantiles = min(100, len(clean))
            qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state=RANDOM_SEED)
            out.loc[clean.index] = qt.fit_transform(clean.to_numpy().reshape(-1, 1)).ravel()
    elif method == "winsorized_z_score":
        lo, hi = clean.quantile([0.05, 0.95])
        clipped = clean.clip(lo, hi)
        std = clipped.std(ddof=1)
        out.loc[clean.index] = (clipped - clipped.mean()) / std if std and not np.isnan(std) else 0.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return out.replace([np.inf, -np.inf], np.nan)


def normalization_quality(series: pd.Series) -> dict[str, float]:
    values = finite_series(series)
    if len(values) < 3:
        return {"normalized_skewness": np.nan, "normalized_kurtosis": np.nan, "normalized_outlier_pct": np.nan, "normalization_quality_score": np.inf}
    skew = float(values.skew()) if len(values) >= 3 else np.nan
    kurt = float(values.kurtosis()) if len(values) >= 4 else 0.0
    out_pct = outlier_pct_iqr(values)
    # Stable distribution objective: low absolute skew, controlled excess kurtosis, low IQR outlier share.
    score = abs(skew if pd.notna(skew) else 0.0) + 0.5 * abs(kurt if pd.notna(kurt) else 0.0) + 0.05 * (out_pct if pd.notna(out_pct) else 0.0)
    return {
        "normalized_skewness": skew,
        "normalized_kurtosis": kurt,
        "normalized_outlier_pct": out_pct,
        "normalization_quality_score": float(score),
    }


def benchmark_rows(role: str, metric: str, series: pd.Series) -> list[dict[str, Any]]:
    values = finite_series(series)
    if values.empty:
        return []
    cuts = values.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    return [
        {"role_family": role, "metric": metric, "benchmark_band": "Very Poor", "lower_bound": float(values.min()), "upper_bound": float(cuts.loc[0.05]), "percentile_range": "0-5"},
        {"role_family": role, "metric": metric, "benchmark_band": "Poor", "lower_bound": float(cuts.loc[0.05]), "upper_bound": float(cuts.loc[0.25]), "percentile_range": "5-25"},
        {"role_family": role, "metric": metric, "benchmark_band": "Average", "lower_bound": float(cuts.loc[0.25]), "upper_bound": float(cuts.loc[0.75]), "percentile_range": "25-75"},
        {"role_family": role, "metric": metric, "benchmark_band": "Good", "lower_bound": float(cuts.loc[0.75]), "upper_bound": float(cuts.loc[0.95]), "percentile_range": "75-95"},
        {"role_family": role, "metric": metric, "benchmark_band": "Excellent", "lower_bound": float(cuts.loc[0.95]), "upper_bound": float(values.max()), "percentile_range": "95-100"},
    ]


def role_feature_frame(player_season: pd.DataFrame, role_resolution: pd.DataFrame, ready: pd.DataFrame, role: str) -> tuple[pd.DataFrame, list[str]]:
    role_info = role_resolution[["statsbomb_player_id", "assigned_role"]].drop_duplicates("statsbomb_player_id")
    df = player_season.merge(role_info, on="statsbomb_player_id", how="left")
    threshold = role_resolution.loc[role_resolution["assigned_role"] == role, "threshold_minutes"].dropna()
    threshold_value = float(threshold.iloc[0]) if not threshold.empty else 0.0
    df = df[(df["assigned_role"] == role) & (pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0) >= threshold_value)].copy()
    metrics = sorted(set(ready.loc[ready["role_family"] == role, "requested_metric_alias"]).intersection(df.columns))
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df, metrics


def correlation_and_clusters(role: str, df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any | None]:
    if len(metrics) < 2 or len(df) < 3:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    X = df[metrics].replace([np.inf, -np.inf], np.nan)
    usable = [m for m in metrics if X[m].notna().sum() >= 3 and X[m].nunique(dropna=True) >= 2]
    if len(usable) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    X = X[usable].fillna(X[usable].median(numeric_only=True))
    pearson = X.corr(method="pearson")
    spearman = X.corr(method="spearman")
    pairs = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            p = pearson.loc[a, b]
            s = spearman.loc[a, b]
            if abs(p) >= REDUNDANCY_THRESHOLD or abs(s) >= REDUNDANCY_THRESHOLD:
                pairs.append({"role_family": role, "metric_a": a, "metric_b": b, "pearson_corr": float(p), "spearman_corr": float(s), "redundancy_rule": "abs_pearson_or_spearman_ge_0_90"})
    redundancy = pd.DataFrame(pairs)
    distance = (1 - spearman.abs().clip(0, 1)).copy()
    distance_values = distance.to_numpy(copy=True)
    np.fill_diagonal(distance_values, 0.0)
    distance = pd.DataFrame(distance_values, index=distance.index, columns=distance.columns)
    Z = linkage(squareform(distance.values, checks=False), method="average")
    cluster_ids = fcluster(Z, t=0.35, criterion="distance")
    clusters = pd.DataFrame({"role_family": role, "metric": usable, "cluster_id": cluster_ids})
    return pearson, spearman, redundancy, (Z, clusters)


def latent_dimensions(role: str, df: pd.DataFrame, metrics: list[str], clusters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(metrics) < 2 or len(df) < 3:
        return pd.DataFrame(), pd.DataFrame()
    X = df[metrics].replace([np.inf, -np.inf], np.nan)
    usable = [m for m in metrics if X[m].notna().sum() >= 3 and X[m].nunique(dropna=True) >= 2]
    if len(usable) < 2:
        return pd.DataFrame(), pd.DataFrame()
    X = X[usable].fillna(X[usable].median(numeric_only=True))
    Z = StandardScaler().fit_transform(X)
    n_components = min(5, len(usable), max(1, len(df) - 1))
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED).fit(Z)
    variance_rows = [{"role_family": role, "component": f"PC{i+1}", "explained_variance_ratio": float(v), "cumulative_variance": float(pca.explained_variance_ratio_[: i + 1].sum())} for i, v in enumerate(pca.explained_variance_ratio_)]
    # FactorAnalysis is used as a secondary latent check; loadings are not final weights.
    try:
        fa_n = min(3, n_components)
        fa = FactorAnalysis(n_components=fa_n, random_state=RANDOM_SEED).fit(Z)
        fa_loadings = pd.DataFrame(fa.components_.T, index=usable, columns=[f"FA{i+1}" for i in range(fa_n)])
    except Exception:
        fa_loadings = pd.DataFrame(index=usable)
    cluster_map = clusters.set_index("metric")["cluster_id"].to_dict() if not clusters.empty else {m: 1 for m in usable}
    rows = []
    loadings = pd.DataFrame(pca.components_.T, index=usable, columns=[f"PC{i+1}" for i in range(n_components)])
    for metric in usable:
        pc_abs = loadings.loc[metric].abs()
        primary_pc = str(pc_abs.idxmax())
        primary_loading = float(loadings.loc[metric, primary_pc])
        factor_note = ""
        if not fa_loadings.empty:
            fa_abs = fa_loadings.loc[metric].abs()
            factor_note = f"; secondary_factor={fa_abs.idxmax()} loading={fa_loadings.loc[metric, fa_abs.idxmax()]:.3f}"
        rows.append(
            {
                "role_family": role,
                "dimension_id": f"{role}_D{int(cluster_map.get(metric, 1))}",
                "dimension_label": f"Data-driven cluster {int(cluster_map.get(metric, 1))}",
                "metric": metric,
                "cluster_id": int(cluster_map.get(metric, 1)),
                "primary_pca_component": primary_pc,
                "primary_pca_loading": primary_loading,
                "assignment_explanation": f"Assigned by hierarchical feature cluster {int(cluster_map.get(metric, 1))} and strongest PCA association {primary_pc} loading={primary_loading:.3f}{factor_note}.",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(variance_rows)


def weight_preparation(role: str, df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if len(metrics) < 2 or len(df) < 3:
        return pd.DataFrame()
    X = df[metrics].replace([np.inf, -np.inf], np.nan)
    usable = [m for m in metrics if X[m].notna().sum() >= 3 and X[m].nunique(dropna=True) >= 2]
    if len(usable) < 2:
        return pd.DataFrame()
    X = X[usable].fillna(X[usable].median(numeric_only=True))
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(len(usable), len(X)), random_state=RANDOM_SEED).fit(scaled)
    pc1_scores = pca.transform(scaled)[:, 0]
    variances = X.var(ddof=1)
    variance_total = variances.sum() if variances.sum() else np.nan
    try:
        mi = mutual_info_regression(scaled, pc1_scores, random_state=RANDOM_SEED)
    except Exception:
        mi = np.zeros(len(usable))
    rows = []
    for idx, metric in enumerate(usable):
        hist = np.histogram(X[metric], bins=min(10, max(3, int(math.sqrt(len(X))))))[0]
        probs = hist[hist > 0] / hist.sum() if hist.sum() else np.array([])
        entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0
        rows.append(
            {
                "role_family": role,
                "metric": metric,
                "normalized_variance": float(variances[metric] / variance_total) if variance_total and not pd.isna(variance_total) else np.nan,
                "pca_pc1_abs_loading": float(abs(pca.components_[0][idx])),
                "mutual_information_to_pc1": float(mi[idx]),
                "information_gain_candidate_entropy": entropy,
                "variance_contribution_candidate": float(variances[metric]),
                "note": "Unsupervised candidate signal only; not a final score weight.",
            }
        )
    return pd.DataFrame(rows)


def save_role_figures(role: str, df: pd.DataFrame, metrics: list[str], pearson: pd.DataFrame | None, pca_variance: pd.DataFrame, cluster_payload: Any | None, benchmarks: pd.DataFrame) -> list[str]:
    paths: list[str] = []
    if not metrics or df.empty:
        return paths
    plot_metrics = metrics[: min(12, len(metrics))]
    fig_dir = ROOT / "outputs/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Distribution plots.
    ncols = 3
    nrows = math.ceil(len(plot_metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(4, 3 * nrows)))
    axes = np.array(axes).reshape(-1)
    for ax, metric in zip(axes, plot_metrics):
        sns.histplot(finite_series(df[metric]), kde=True, ax=ax, color="#285C7D")
        ax.set_title(metric)
    for ax in axes[len(plot_metrics):]:
        ax.axis("off")
    fig.suptitle(f"{role} ready metric distributions")
    fig.tight_layout()
    path = fig_dir / f"003_{role}_distributions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    # QQ plots.
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(4, 3 * nrows)))
    axes = np.array(axes).reshape(-1)
    for ax, metric in zip(axes, plot_metrics):
        values = finite_series(df[metric])
        if len(values) >= 3:
            stats.probplot(values, dist="norm", plot=ax)
        ax.set_title(metric)
    for ax in axes[len(plot_metrics):]:
        ax.axis("off")
    fig.suptitle(f"{role} QQ plots")
    fig.tight_layout()
    path = fig_dir / f"003_{role}_qq_plots.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(ROOT)))

    # Correlation heatmap.
    if pearson is not None and not pearson.empty:
        plt.figure(figsize=(max(8, len(pearson) * 0.7), max(6, len(pearson) * 0.6)))
        sns.heatmap(pearson, cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.8})
        plt.title(f"{role} Pearson correlation heatmap")
        plt.tight_layout()
        path = fig_dir / f"003_{role}_correlation_heatmap.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path.relative_to(ROOT)))

    # PCA variance.
    if not pca_variance.empty:
        role_var = pca_variance[pca_variance["role_family"] == role]
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=role_var, x="component", y="cumulative_variance", marker="o")
        plt.axhline(0.80, color="#9B2C2C", linestyle="--", linewidth=1)
        plt.title(f"{role} PCA cumulative variance")
        plt.tight_layout()
        path = fig_dir / f"003_{role}_pca_variance.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path.relative_to(ROOT)))

    # Dendrogram.
    if cluster_payload is not None:
        Z, clusters = cluster_payload
        plt.figure(figsize=(max(9, len(clusters) * 0.6), 6))
        dendrogram(Z, labels=list(clusters["metric"]), leaf_rotation=90)
        plt.title(f"{role} hierarchical feature clustering")
        plt.tight_layout()
        path = fig_dir / f"003_{role}_cluster_dendrogram.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path.relative_to(ROOT)))

    # Boxplots.
    box = df[plot_metrics].melt(var_name="metric", value_name="value")
    plt.figure(figsize=(max(10, len(plot_metrics) * 0.8), 6))
    sns.boxplot(data=box, x="metric", y="value", color="#D9A441")
    plt.xticks(rotation=75, ha="right")
    plt.title(f"{role} metric boxplots")
    plt.tight_layout()
    path = fig_dir / f"003_{role}_boxplots.png"
    plt.savefig(path, dpi=160)
    plt.close()
    paths.append(str(path.relative_to(ROOT)))

    # Benchmark distributions.
    role_bench = benchmarks[benchmarks["role_family"] == role]
    if not role_bench.empty:
        bench_counts = role_bench.groupby("benchmark_band").size().rename("metric_bands").reset_index()
        order = ["Very Poor", "Poor", "Average", "Good", "Excellent"]
        plt.figure(figsize=(8, 5))
        sns.barplot(data=bench_counts, x="benchmark_band", y="metric_bands", order=order, color="#285C7D")
        plt.title(f"{role} benchmark band definitions")
        plt.tight_layout()
        path = fig_dir / f"003_{role}_benchmark_distributions.png"
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path.relative_to(ROOT)))
    return paths


def write_notebook(path: Path, data_root: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# Experiment {EXPERIMENT_ID} — {EXPERIMENT_TITLE}"),
        nbf.v4.new_markdown_cell("## 1. Objective\n\nCreate the scientific feature layer for future score engines. No final weights and no supervised score model are built."),
        nbf.v4.new_markdown_cell("## 2. Football hypothesis\n\nRole-specific ready metrics need role-specific normalization, benchmarks, redundancy checks, and latent-dimension discovery before score weighting."),
        nbf.v4.new_markdown_cell("## 3. Dataset\n\nUses Experiment 002 READY metrics only, player-season StatsBomb direct provider stats, and role eligibility from Experiment 002."),
        nbf.v4.new_code_cell("import pandas as pd\nnormalization = pd.read_csv('outputs/tables/003_normalization_decisions.csv')\nnormalization.head()"),
        nbf.v4.new_markdown_cell("## 4. Feature engineering\n\nFor each eligible role, metrics are transformed by z-score, robust z-score, percentile rank, min-max, log transform, quantile transform, and winsorized z-score. The selected method minimizes transformed skew, excess kurtosis, and IQR outlier share."),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/003_metric_statistics.csv').head(20)"),
        nbf.v4.new_markdown_cell("## 5. Exploratory Data Analysis\n\nDistribution, QQ, boxplot, and benchmark figures are produced per role under outputs/figures."),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/003_role_benchmarks.csv').head(20)"),
        nbf.v4.new_markdown_cell("## 6. Statistical Analysis\n\nPearson/Spearman correlations, redundancy pairs, hierarchical feature clustering, PCA variance, FactorAnalysis checks, and unsupervised feature-importance candidates are computed."),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/003_feature_redundancy.csv').head(20)"),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/003_latent_dimensions.csv').head(20)"),
        nbf.v4.new_markdown_cell("## 7. Machine Learning\n\nNo predictive model is trained. PCA, FactorAnalysis, QuantileTransformer, hierarchical clustering, and mutual information are used only for feature engineering research and weight preparation."),
        nbf.v4.new_code_cell("pd.read_csv('outputs/tables/003_weight_preparation.csv').head(20)"),
        nbf.v4.new_markdown_cell("## 8. Explainability\n\nEvery normalization decision has an explicit distribution-quality score. Every latent dimension assignment includes cluster id and PCA/factor evidence."),
        nbf.v4.new_markdown_cell("## 9. Validation\n\nThe script validates required Experiment 002 inputs, regenerates all expected tables/figures/reports, and appends methodology once."),
        nbf.v4.new_markdown_cell("## 10. Conclusions\n\nExperiment 003 prepares the normalized feature layer and benchmark definitions for Experiment 004. Final weights remain intentionally uncomputed."),
        nbf.v4.new_markdown_cell("## 11. Next Experiments\n\nExperiment 004 may use this feature layer to build the first interpretable score-family baselines. Do not proceed to Experiment 004 inside this notebook."),
        nbf.v4.new_markdown_cell("## Reproduce\n\n```bash\nuv run python experiments/003_feature_engineering_normalization.py --data-root " + str(data_root) + "\n```"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def append_methodology(path: Path, report: dict[str, Any]) -> None:
    marker = f"## Experiment {EXPERIMENT_ID}"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Football Score Engine Research Journal\n"
    if marker in existing:
        return
    role_lines = "\n".join(f"| {r['role_family']} | {r['eligible_players']} | {r['ready_metrics']} | {r['normalization_decisions']} | {r['redundant_pairs']} | {r['latent_dimensions']} |" for r in report["role_summary"])
    section = f"""
## Experiment {EXPERIMENT_ID} — {EXPERIMENT_TITLE}

Date: {report['generated_at']}

### Objective

Create the scientific role-specific feature layer that all future score engines will use. This experiment normalizes READY metrics from Experiment 002, profiles distributions, builds benchmark cutoffs, detects redundancy, and discovers latent dimensions. It does not compute final score weights.

### Football Hypothesis

A defensible score engine needs stable role-specific feature transformations before modelling. The same raw metric can require a different normalization method depending on role population and distribution shape.

### Dataset

Source root: `{report['data_root']}`

Inputs:

- `marts_v2/mart_statsbomb_player_season_stats_direct_v1.jsonl`
- `outputs/tables/002_role_resolution.csv`
- `outputs/tables/002_candidate_metric_status.csv`

### Normalization Tested

- Z-score
- Robust Z-score
- Percentile Rank
- Min-Max
- Log transform
- Quantile transform
- Winsorized Z-score

Selection criterion: transformed distribution stability score = abs(skewness) + 0.5 * abs(excess kurtosis) + 0.05 * outlier percentage.

### Feature Selection

Only `candidate_ready` metrics from Experiment 002 are used. Additional redundancy pairs are identified from Pearson/Spearman correlations >= {REDUNDANCY_THRESHOLD}; removals are recommendations for Experiment 004, not destructive changes to source data.

### Algorithms

- Distribution profiling: mean, median, std, MAD, IQR, CV, percentiles, missing %, outlier %, skewness, kurtosis, Shapiro test
- Normalization comparison across seven methods
- Pearson and Spearman correlation
- Hierarchical clustering on absolute Spearman distance
- PCA cumulative variance and loadings
- FactorAnalysis as a secondary latent-dimension check
- Mutual information against unsupervised PC1 for weight-preparation candidates

### Evaluation

| Role | Eligible players | READY metrics | Normalization decisions | Redundant pairs | Latent dimensions |
|---|---:|---:|---:|---:|---:|
{role_lines}

### Results

- Total metric statistics rows: {report['table_counts']['metric_statistics']}
- Normalization method evaluations: {report['table_counts']['normalization_methods']}
- Normalization decisions: {report['table_counts']['normalization_decisions']}
- Benchmark rows: {report['table_counts']['role_benchmarks']}
- Redundancy pairs: {report['table_counts']['feature_redundancy']}
- Latent dimension rows: {report['table_counts']['latent_dimensions']}
- Weight-preparation rows: {report['table_counts']['weight_preparation']}
- Figures generated: {report['figures_generated']}

### Figures

Per-role figures are written under `outputs/figures/003_<ROLE>_*`: distributions, QQ plots, correlation heatmaps, PCA variance, cluster dendrograms, boxplots, and benchmark distributions.

### Discussion

The experiment produces a reproducible feature-engineering layer without hardcoded football weights. Dimension labels remain data-driven clusters, with metric membership explained by hierarchical clustering and PCA/factor evidence. Football interpretation should happen in Experiment 004 after reviewing these empirical clusters.

### Limitations

- Current local root remains a limited sample rather than the full multi-competition/two-season target population.
- Shapiro, multimodality, and split-dimensional diagnostics are sample-size sensitive.
- Quantile normalization can over-stabilize very small samples; decisions must be rerun on the full population.
- Mutual information is unsupervised against PC1, not a target-based importance score.

### Decision

Use `003_normalization_decisions.csv`, `003_role_benchmarks.csv`, `003_feature_redundancy.csv`, and `003_latent_dimensions.csv` as the feature-layer contract for Experiment 004. Do not compute final score weights yet.

### Production Recommendation

Future production score engines should load role-specific normalization decisions and benchmark cutoffs from versioned artifacts, then recompute them whenever population coverage changes materially.

### Next Steps

1. Review latent clusters by role and assign football-readable labels only where the data supports them.
2. Experiment 004: build first interpretable score-family baselines using the Experiment 003 feature layer.
3. Re-run Experiment 003 on the full StatsBomb multi-competition/two-season dataset before production coefficients.
"""
    path.write_text(existing.rstrip() + "\n\n" + section.strip() + "\n", encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/platform/DataPlatform/tmp/master_data_warehouse")
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()

    player_season_path = data_root / "marts_v2/mart_statsbomb_player_season_stats_direct_v1.jsonl"
    role_resolution_path = ROOT / "outputs/tables/002_role_resolution.csv"
    candidate_status_path = ROOT / "outputs/tables/002_candidate_metric_status.csv"
    required = [player_season_path, role_resolution_path, candidate_status_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Experiment 003 inputs: {missing}. Run Experiments 001 and 002 first.")

    player_season = flatten_metrics(player_season_path, id_fields=["statsbomb_player_id", "player_name", "team_id", "team_name", "competition_id", "season_id"])
    player_season["statsbomb_player_id"] = player_season["statsbomb_player_id"].astype(str)
    role_resolution = pd.read_csv(role_resolution_path)
    role_resolution["statsbomb_player_id"] = role_resolution["statsbomb_player_id"].astype(str)
    ready = pd.read_csv(candidate_status_path)
    ready = ready[ready["status"] == "candidate_ready"].copy()

    all_stats: list[pd.DataFrame] = []
    all_norm_methods: list[pd.DataFrame] = []
    all_norm_decisions: list[dict[str, Any]] = []
    all_benchmarks: list[pd.DataFrame] = []
    all_clusters: list[pd.DataFrame] = []
    all_redundancy: list[pd.DataFrame] = []
    all_latent: list[pd.DataFrame] = []
    all_pca_variance: list[pd.DataFrame] = []
    all_weight_prep: list[pd.DataFrame] = []
    role_summary: list[dict[str, Any]] = []
    figure_paths: list[str] = []

    for role in ROLES:
        role_df, metrics = role_feature_frame(player_season, role_resolution, ready, role)
        stat_rows = [metric_stats(role, metric, role_df[metric], len(role_df)) for metric in metrics]
        stats_df = pd.DataFrame(stat_rows)
        if not stats_df.empty:
            all_stats.append(stats_df)
        bench_rows = []
        method_rows = []
        decisions = []
        for metric in metrics:
            bench_rows.extend(benchmark_rows(role, metric, role_df[metric]))
            best_method = None
            best_score = np.inf
            best_quality: dict[str, float] = {}
            for method in NORMALIZATION_METHODS:
                normalized = normalize_series(role_df[metric], method)
                quality = normalization_quality(normalized)
                method_rows.append({"role_family": role, "metric": metric, "normalization_method": method, **quality})
                if quality["normalization_quality_score"] < best_score:
                    best_score = quality["normalization_quality_score"]
                    best_method = method
                    best_quality = quality
            decisions.append({"role_family": role, "metric": metric, "selected_normalization": best_method, **best_quality, "selection_rule": "min_abs_skew_plus_half_abs_kurtosis_plus_outlier_penalty"})
        if bench_rows:
            all_benchmarks.append(pd.DataFrame(bench_rows))
        if method_rows:
            all_norm_methods.append(pd.DataFrame(method_rows))
        all_norm_decisions.extend(decisions)

        pearson, spearman, redundancy, cluster_payload = correlation_and_clusters(role, role_df, metrics)
        clusters_df = pd.DataFrame()
        if cluster_payload is not None:
            _, clusters_df = cluster_payload
            all_clusters.append(clusters_df)
        if not redundancy.empty:
            # Recommend removing the second metric from each redundant pair unless later football review overrides it.
            redundancy = redundancy.copy()
            redundancy["recommended_action"] = "review_remove_metric_b_or_keep_one_with_better_interpretability"
            all_redundancy.append(redundancy)
        latent_df, pca_var = latent_dimensions(role, role_df, metrics, clusters_df)
        if not latent_df.empty:
            all_latent.append(latent_df)
        if not pca_var.empty:
            all_pca_variance.append(pca_var)
        weight_df = weight_preparation(role, role_df, metrics)
        if not weight_df.empty:
            all_weight_prep.append(weight_df)
        role_benchmarks_for_fig = pd.concat(all_benchmarks, ignore_index=True) if all_benchmarks else pd.DataFrame()
        figure_paths.extend(save_role_figures(role, role_df, metrics, pearson if isinstance(pearson, pd.DataFrame) else None, pca_var, cluster_payload, role_benchmarks_for_fig))
        role_summary.append(
            {
                "role_family": role,
                "eligible_players": int(len(role_df)),
                "ready_metrics": int(len(metrics)),
                "normalization_decisions": int(len(decisions)),
                "redundant_pairs": int(len(redundancy)) if isinstance(redundancy, pd.DataFrame) else 0,
                "latent_dimensions": int(latent_df["dimension_id"].nunique()) if not latent_df.empty else 0,
            }
        )

    tables = {
        "metric_statistics": pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame(),
        "normalization_methods": pd.concat(all_norm_methods, ignore_index=True) if all_norm_methods else pd.DataFrame(),
        "normalization_decisions": pd.DataFrame(all_norm_decisions),
        "role_benchmarks": pd.concat(all_benchmarks, ignore_index=True) if all_benchmarks else pd.DataFrame(),
        "metric_clusters": pd.concat(all_clusters, ignore_index=True) if all_clusters else pd.DataFrame(),
        "feature_redundancy": pd.concat(all_redundancy, ignore_index=True) if all_redundancy else pd.DataFrame(columns=["role_family", "metric_a", "metric_b", "pearson_corr", "spearman_corr", "redundancy_rule", "recommended_action"]),
        "latent_dimensions": pd.concat(all_latent, ignore_index=True) if all_latent else pd.DataFrame(),
        "pca_variance": pd.concat(all_pca_variance, ignore_index=True) if all_pca_variance else pd.DataFrame(),
        "weight_preparation": pd.concat(all_weight_prep, ignore_index=True) if all_weight_prep else pd.DataFrame(),
    }
    for name, df in tables.items():
        df.to_csv(ROOT / f"outputs/tables/003_{name}.csv", index=False)
        # Also write the exact table names requested in the experiment brief.
        if name in {
            "normalization_methods",
            "metric_statistics",
            "role_benchmarks",
            "metric_clusters",
            "feature_redundancy",
            "latent_dimensions",
            "normalization_decisions",
        }:
            df.to_csv(ROOT / f"outputs/tables/{name}.csv", index=False)

    report = {
        "experiment_id": EXPERIMENT_ID,
        "title": EXPERIMENT_TITLE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "inputs": {"player_season_rows": int(len(player_season)), "ready_metric_rows": int(len(ready)), "role_resolution_rows": int(len(role_resolution))},
        "normalization_methods": NORMALIZATION_METHODS,
        "role_summary": role_summary,
        "table_counts": {name: int(len(df)) for name, df in tables.items()},
        "figures_generated": len(figure_paths),
        "figure_paths": figure_paths,
        "outputs": {"notebook": f"notebooks/{EXPERIMENT_ID}_feature_engineering_normalization.ipynb", "tables": "outputs/tables/003_*.csv", "figures": "outputs/figures/003_*.png"},
    }
    write_json(ROOT / "outputs/reports/003_feature_engineering_normalization.json", report)
    (ROOT / "outputs/reports/003_feature_engineering_normalization.md").write_text(
        f"# Experiment {EXPERIMENT_ID} — {EXPERIMENT_TITLE}\n\n"
        f"Generated: {report['generated_at']}\n\n"
        f"Data root: `{data_root}`\n\n"
        f"READY metric rows from Experiment 002: {report['inputs']['ready_metric_rows']}\n\n"
        f"Figures generated: {len(figure_paths)}\n\n"
        "## Role summary\n\n"
        + "\n".join(f"- {r['role_family']}: eligible={r['eligible_players']}, ready_metrics={r['ready_metrics']}, redundant_pairs={r['redundant_pairs']}, latent_dimensions={r['latent_dimensions']}" for r in role_summary)
        + "\n\nNo final weights were computed.\n",
        encoding="utf-8",
    )
    write_notebook(ROOT / f"notebooks/{EXPERIMENT_ID}_feature_engineering_normalization.ipynb", data_root)
    append_methodology(ROOT / "methodology.md", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
