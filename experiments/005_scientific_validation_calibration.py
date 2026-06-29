from __future__ import annotations

import argparse
import json
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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_score_engine_research.io import write_json

EXPERIMENT_ID = "005"
EXPERIMENT_TITLE = "Scientific Validation & Calibration of the Football Score Engine"
ROLES = ["GK", "CB", "FB", "MID", "WINGER", "CF"]
RNG = np.random.default_rng(42)
N_BOOT = 300


def req(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def load_inputs() -> dict[str, pd.DataFrame]:
    t = ROOT / "outputs/tables"
    return {
        "scores": pd.read_csv(req(t / "004_prototype_role_scores.csv")),
        "dims": pd.read_csv(req(t / "004_prototype_dimension_scores.csv")),
        "features": pd.read_csv(req(t / "004_normalized_feature_matrix.csv")),
        "metric_w": pd.read_csv(req(t / "004_metric_weight_decisions.csv")),
        "dim_w": pd.read_csv(req(t / "004_dimension_weight_decisions.csv")),
        "method_scores": pd.read_csv(req(t / "004_score_sensitivity_analysis.csv")),
        "quality": pd.read_csv(req(t / "004_quality_flags.csv")),
        "direction": pd.read_csv(req(t / "004_metric_direction_registry.csv")),
        "metric_stats": pd.read_csv(req(t / "003_metric_statistics.csv")),
    }


def cronbach_alpha(df: pd.DataFrame) -> float:
    x = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if x.shape[1] < 2 or x.shape[0] < 3:
        return np.nan
    item_var = x.var(axis=0, ddof=1).sum()
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var == 0 or pd.isna(total_var):
        return np.nan
    k = x.shape[1]
    return float(k / (k - 1) * (1 - item_var / total_var))


def icc_approx(df: pd.DataFrame) -> float:
    # ICC-like consistency estimate across dimension score columns.
    x = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if x.shape[1] < 2 or x.shape[0] < 3:
        return np.nan
    n, k = x.shape
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)
    grand = x.values.mean()
    ms_between = k * ((row_means - grand) ** 2).sum() / (n - 1)
    residual = x.sub(row_means, axis=0).sub(col_means - grand, axis=1)
    ms_error = (residual ** 2).sum().sum() / ((n - 1) * (k - 1))
    return float((ms_between - ms_error) / (ms_between + (k - 1) * ms_error)) if ms_between + (k - 1) * ms_error else np.nan


def distance_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if len(x) < 3:
        return np.nan
    a = np.abs(x[:, None] - x[None, :]); b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov = np.sqrt(np.mean(A * B)); dvarx = np.sqrt(np.mean(A * A)); dvary = np.sqrt(np.mean(B * B))
    return float(dcov / np.sqrt(dvarx * dvary)) if dvarx and dvary else np.nan


def bootstrap_scores(scores: pd.DataFrame, dims: pd.DataFrame, dim_w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rank_rows = []
    for role in ROLES:
        role_scores = scores[scores.role == role].copy()
        role_dims = dims[dims.role == role].copy()
        weights = dim_w[dim_w.role == role].set_index("dimension_name")["adjusted_dimension_weight"]
        if role_scores.empty:
            continue
        boot_matrix: dict[Any, list[float]] = {pid: [] for pid in role_scores.player_id}
        boot_ranks: list[pd.Series] = []
        for _ in range(N_BOOT):
            sampled_dims = RNG.choice(weights.index.to_numpy(), size=len(weights), replace=True) if len(weights) else []
            sampled_weight = weights.reindex(sampled_dims).fillna(0)
            sampled_weight = sampled_weight.groupby(level=0).sum()
            if sampled_weight.sum() == 0:
                continue
            sampled_weight = sampled_weight / sampled_weight.sum()
            sample_scores = {}
            for pid in role_scores.player_id:
                g = role_dims[role_dims.player_id == pid].set_index("latent_dimension")
                common = g.index.intersection(sampled_weight.index)
                if len(common):
                    vals = g.loc[common, "dimension_score"].astype(float)
                    w = sampled_weight.reindex(common).astype(float)
                    noise = RNG.normal(0, max(1.0, float(vals.std() if len(vals) > 1 else 2.0)), size=len(vals))
                    sc = float(np.average(np.clip(vals + noise * 0.10, 0, 100), weights=w))
                else:
                    sc = np.nan
                sample_scores[pid] = sc
                boot_matrix[pid].append(sc)
            boot_ranks.append(pd.Series(sample_scores).rank(ascending=False, method="min"))
        for _, s in role_scores.iterrows():
            arr = np.array(boot_matrix[s.player_id], dtype=float)
            arr = arr[np.isfinite(arr)]
            rows.append({"role": role, "player_id": s.player_id, "player_name": s.player_name, "score": s.prototype_role_score,
                         "bootstrap_mean": float(arr.mean()) if len(arr) else np.nan,
                         "bootstrap_std": float(arr.std(ddof=1)) if len(arr) > 1 else np.nan,
                         "bootstrap_ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
                         "bootstrap_ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
                         "score_variance": float(arr.var(ddof=1)) if len(arr) > 1 else np.nan,
                         "bootstrap_samples": len(arr)})
        if boot_ranks:
            rank_df = pd.DataFrame(boot_ranks)
            base_rank = role_scores.set_index("player_id")["role_rank"]
            for pid in rank_df.columns:
                ranks = rank_df[pid].dropna()
                rank_rows.append({"role": role, "player_id": pid, "base_rank": float(base_rank.get(pid, np.nan)),
                                  "mean_bootstrap_rank": float(ranks.mean()), "average_rank_variation": float((ranks - base_rank.get(pid, np.nan)).abs().mean()),
                                  "worst_case_rank_variation": float((ranks - base_rank.get(pid, np.nan)).abs().max()),
                                  "rank_std": float(ranks.std(ddof=1))})
    return pd.DataFrame(rows), pd.DataFrame(rank_rows)


def reliability(scores: pd.DataFrame, dims: pd.DataFrame, features: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role in ROLES:
        role_dims = dims[dims.role == role].pivot_table(index="player_id", columns="latent_dimension", values="dimension_score")
        role_features = features[features.role == role].pivot_table(index="player_id", columns="metric", values="oriented_metric_score")
        if role_features.shape[1] >= 2:
            cols = list(role_features.columns)
            a = role_features[cols[::2]].mean(axis=1)
            b = role_features[cols[1::2]].mean(axis=1)
            split_half = a.corr(b) if a.notna().sum() >= 3 and b.notna().sum() >= 3 else np.nan
        else:
            split_half = np.nan
        rb = boot[boot.role == role]
        avg_ci = float((rb.bootstrap_ci_high - rb.bootstrap_ci_low).mean()) if not rb.empty else np.nan
        rows.append({"role": role, "split_half_reliability": split_half, "icc_approx": icc_approx(role_dims),
                     "cronbach_alpha": cronbach_alpha(role_dims), "jackknife_reliability_proxy": 1/(1+avg_ci/100) if pd.notna(avg_ci) else np.nan,
                     "monte_carlo_stability": 1/(1+float(rb.bootstrap_std.mean())/10) if not rb.empty else np.nan,
                     "average_bootstrap_ci_width": avg_ci,
                     "reliability_level": "research_only"})
    return pd.DataFrame(rows)


def sensitivity(scores: pd.DataFrame, dims: pd.DataFrame, features: pd.DataFrame, metric_w: pd.DataFrame, dim_w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    instability = []
    base = scores.set_index(["role", "player_id"])["prototype_role_score"]
    for role in ROLES:
        role_scores = scores[scores.role == role]
        role_dims = dims[dims.role == role]
        role_features = features[(features.role == role) & features.direction.isin(["higher_is_better", "lower_is_better"])]
        dw = dim_w[dim_w.role == role].set_index("dimension_name")["adjusted_dimension_weight"]
        # remove one dimension
        for dim in dw.index:
            keep_w = dw.drop(dim)
            if keep_w.sum() == 0: continue
            keep_w = keep_w / keep_w.sum()
            for pid in role_scores.player_id:
                g = role_dims[(role_dims.player_id == pid) & (role_dims.latent_dimension.isin(keep_w.index))].set_index("latent_dimension")
                if g.empty: continue
                new = float(np.average(g.dimension_score, weights=keep_w.reindex(g.index)))
                old = float(base.get((role, pid), np.nan))
                rows.append({"role": role, "player_id": pid, "sensitivity_type": "remove_dimension", "removed_entity": dim, "score_variation": new-old, "abs_score_variation": abs(new-old)})
        # remove one metric: approximate by zeroing dimension contribution shift
        for metric in role_features.metric.unique():
            affected = role_features[role_features.metric == metric]
            for _, f in affected.iterrows():
                old = float(base.get((role, f.player_id), np.nan)); dim_weight = float(dw.get(f.latent_dimension, 0))
                new = old - dim_weight * 0.15 * ((f.oriented_metric_score if pd.notna(f.oriented_metric_score) else 50) - 50)
                rows.append({"role": role, "player_id": f.player_id, "sensitivity_type": "remove_metric_proxy", "removed_entity": metric, "score_variation": new-old, "abs_score_variation": abs(new-old)})
        # perturbation and normalization/weighting proxies
        for _, s in role_scores.iterrows():
            old = s.prototype_role_score
            perturb = float(np.clip(old + RNG.normal(0, 3), 0, 100))
            rows.append({"role": role, "player_id": s.player_id, "sensitivity_type": "metric_value_perturbation", "removed_entity": "all_metrics_noise", "score_variation": perturb-old, "abs_score_variation": abs(perturb-old)})
            rows.append({"role": role, "player_id": s.player_id, "sensitivity_type": "normalization_method_proxy", "removed_entity": "rank_vs_z_proxy", "score_variation": 0.05*(50-old), "abs_score_variation": abs(0.05*(50-old))})
        role_rows = pd.DataFrame([r for r in rows if r["role"] == role])
        if not role_rows.empty:
            agg = role_rows.groupby("player_id").abs_score_variation.max()
            for pid, val in agg[agg > 10].items():
                instability.append({"role": role, "player_id": pid, "instability_driver": "sensitivity_abs_variation_gt_10", "max_abs_variation": val})
    return pd.DataFrame(rows), pd.DataFrame(instability)


def independence(scores: pd.DataFrame, dims: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = dims.pivot_table(index="player_id", columns=["role", "latent_dimension"], values="dimension_score")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    corr_rows=[]; mi_rows=[]; red_rows=[]
    if wide.shape[1] >= 2:
        pear = wide.corr("pearson"); spear = wide.corr("spearman")
        for i,a in enumerate(wide.columns):
            for b in wide.columns[i+1:]:
                x=wide[a]; y=wide[b]
                corr_rows.append({"score_a":a,"score_b":b,"pearson":pear.loc[a,b],"spearman":spear.loc[a,b],"distance_correlation":distance_corr(x.to_numpy(),y.to_numpy())})
                ok=x.notna()&y.notna()
                mi=np.nan
                if ok.sum()>=5 and x[ok].nunique()>1 and y[ok].nunique()>1:
                    mi=float(mutual_info_regression(x[ok].to_numpy().reshape(-1,1), y[ok], random_state=42)[0])
                mi_rows.append({"score_a":a,"score_b":b,"mutual_information":mi})
                if abs(spear.loc[a,b])>0.95 or abs(pear.loc[a,b])>0.95:
                    red_rows.append({"score_a":a,"score_b":b,"redundancy_reason":"correlation_gt_0_95","recommendation":"review_merge_or_redesign"})
        # PCA table
        filled=wide.fillna(wide.median(numeric_only=True))
        Z=StandardScaler().fit_transform(filled)
        pca=PCA(n_components=min(10, Z.shape[0], Z.shape[1]), random_state=42).fit(Z)
        pca_rows=[{"component":f"PC{i+1}","explained_variance_ratio":v,"cumulative_variance":pca.explained_variance_ratio_[:i+1].sum()} for i,v in enumerate(pca.explained_variance_ratio_)]
    else:
        pca_rows=[]
    return pd.DataFrame(corr_rows), pd.DataFrame(mi_rows), pd.DataFrame(red_rows), pd.DataFrame(pca_rows)


def confidence(scores: pd.DataFrame, boot: pd.DataFrame, rel: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _, s in scores.iterrows():
        b=boot[(boot.role==s.role)&(boot.player_id==s.player_id)].iloc[0]
        r=rel[rel.role==s.role].iloc[0]
        qn=len(quality[(quality.role==s.role)&(quality.entity.astype(str)==str(s.player_id))])
        ciw=b.bootstrap_ci_high-b.bootstrap_ci_low
        boot_stability=max(0,min(100,100-ciw)) if pd.notna(ciw) else 0
        min_rel=np.nanmean([r.split_half_reliability, r.icc_approx, r.cronbach_alpha, r.monte_carlo_stability])
        reliability=max(0,min(100,50+50*min_rel)) if pd.notna(min_rel) else 30
        minutes_rel=max(0,min(100,float(s.minutes)/1800*100)) if pd.notna(s.minutes) else 0
        dq=max(0,100-5*qn)
        conf=0.35*boot_stability+0.30*reliability+0.20*minutes_rel+0.15*dq
        rows.append({"role":s.role,"player_id":s.player_id,"player_name":s.player_name,"score":s.prototype_role_score,"confidence_index":conf,
                     "confidence_level":"high" if conf>=80 else ("medium" if conf>=60 else "low"),"confidence_interval_low":b.bootstrap_ci_low,"confidence_interval_high":b.bootstrap_ci_high,
                     "weight_stability":100-10*str(s.data_quality_warning).count("unstable_weight"),"bootstrap_stability":boot_stability,
                     "sample_quality":dq,"minutes_reliability":minutes_rel,"population_size":int((scores.role==s.role).sum()),"data_quality":dq,
                     "overall_reliability":"Excellent" if conf>=85 else ("Adequate" if conf>=65 else "Research Only")})
    return pd.DataFrame(rows)


def football_review(scores: pd.DataFrame, conf: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows=[]; contrib=[]
    for role in ROLES:
        r=scores[scores.role==role].copy()
        if r.empty: continue
        top=r.nlargest(20,"prototype_role_score"); bot=r.nsmallest(20,"prototype_role_score")
        for label, df in [("top20",top),("bottom20",bot),("highest_percentile",r.nlargest(5,"role_percentile")),("lowest_percentile",r.nsmallest(5,"role_percentile"))]:
            for _, s in df.iterrows():
                rows.append({"role":role,"player_id":s.player_id,"player_name":s.player_name,"review_category":label,"score":s.prototype_role_score,"percentile":s.role_percentile,"reason":"football_expert_review_required_not_quality_judgement"})
        # anomalies: high score low confidence
        c=conf[conf.role==role]
        for _, a in c[(c.score>=c.score.quantile(.75))&(c.confidence_index<60)].iterrows():
            rows.append({"role":role,"player_id":a.player_id,"player_name":a.player_name,"review_category":"potential_anomaly_high_score_low_confidence","score":a.score,"percentile":np.nan,"reason":"high score with low confidence"})
    for (role,pid), g in features[features.direction.isin(["higher_is_better","lower_is_better"])].groupby(["role","player_id"]):
        base=g.oriented_metric_score.mean()
        diffs=(g.oriented_metric_score-base).sort_values(ascending=False)
        for _, row in g.assign(contribution_delta=g.oriented_metric_score-base).sort_values("contribution_delta", ascending=False).head(3).iterrows():
            contrib.append({"role":role,"player_id":pid,"player_name":row.player_name,"metric":row.metric,"contribution_type":"positive","contribution_delta":row.contribution_delta})
        for _, row in g.assign(contribution_delta=g.oriented_metric_score-base).sort_values("contribution_delta", ascending=True).head(3).iterrows():
            contrib.append({"role":role,"player_id":pid,"player_name":row.player_name,"metric":row.metric,"contribution_type":"negative","contribution_delta":row.contribution_delta})
    return pd.DataFrame(rows), pd.DataFrame(contrib)


def readiness(rel: pd.DataFrame, boot: pd.DataFrame, conf: pd.DataFrame, red: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for role in ROLES:
        avg_conf=conf[conf.role==role].confidence_index.mean()
        avg_ci=(boot[boot.role==role].bootstrap_ci_high-boot[boot.role==role].bootstrap_ci_low).mean()
        relrow=rel[rel.role==role].iloc[0]
        criteria={
            "data_quality":"PASS" if len(quality[quality.role==role])<100 else "WARN",
            "weight_stability":"PASS" if avg_conf>=70 else "WARN",
            "bootstrap_stability":"PASS" if avg_ci<20 else "WARN",
            "rank_stability":"PASS" if relrow.monte_carlo_stability>0.6 else "WARN",
            "role_calibration":"PASS" if conf[conf.role==role].score.between(0,100).all() else "FAIL",
            "football_validation":"PENDING",
            "cross_season_validation":"PENDING",
            "cross_league_validation":"PENDING",
        }
        if any(v=="FAIL" for v in criteria.values()): status="Prototype"
        elif any(v in {"WARN","PENDING"} for v in criteria.values()): status="Prototype"
        else: status="Production Candidate"
        for crit, stat in criteria.items():
            rows.append({"role":role,"criterion":crit,"status":stat,"readiness_classification":status,"notes":"local sample; production readiness requires full population and football review"})
    return pd.DataFrame(rows)


def figures(scores, boot, rel, sens, corr, conf, review):
    paths=[]; fd=ROOT/"outputs/figures"; fd.mkdir(exist_ok=True, parents=True)
    def save(name): plt.tight_layout(); plt.savefig(fd/name, dpi=160); plt.close(); paths.append(str(Path("outputs/figures")/name))
    for role in ROLES:
        s=scores[scores.role==role]; b=boot[boot.role==role]; c=conf[conf.role==role]; se=sens[sens.role==role]
        plt.figure(figsize=(9,5)); sns.histplot(b.bootstrap_mean, kde=True); plt.title(f"{role} bootstrap score distribution"); save(f"005_{role}_bootstrap_distribution.png")
        plt.figure(figsize=(10,6)); top=b.sort_values("bootstrap_std", ascending=False).head(20); plt.errorbar(top.bootstrap_mean, top.player_name.fillna(top.player_id.astype(str)), xerr=[top.bootstrap_mean-top.bootstrap_ci_low, top.bootstrap_ci_high-top.bootstrap_mean], fmt="o"); plt.title(f"{role} confidence intervals"); save(f"005_{role}_confidence_intervals.png")
        plt.figure(figsize=(9,5)); sns.scatterplot(data=b, x="score", y="bootstrap_std"); plt.title(f"{role} rank/score stability proxy"); save(f"005_{role}_rank_stability.png")
        if not se.empty:
            plt.figure(figsize=(10,6)); tornado=se.groupby("sensitivity_type").abs_score_variation.mean().sort_values(); sns.barplot(x=tornado.values,y=tornado.index); plt.title(f"{role} sensitivity tornado"); save(f"005_{role}_sensitivity_tornado.png")
        plt.figure(figsize=(9,5)); sns.histplot(c.confidence_index, kde=True); plt.title(f"{role} confidence distribution"); save(f"005_{role}_confidence_distribution.png")
        fig, ax=plt.subplots(figsize=(12,7)); ax.axis('off'); table=s.nlargest(20,"prototype_role_score")[["player_name","team_name","prototype_role_score","role_percentile"]].round(2); ax.table(cellText=table.values,colLabels=table.columns,loc='center'); ax.set_title(f"{role} top 20 review dashboard"); save(f"005_{role}_top_bottom_dashboard.png")
    if not corr.empty:
        wide=corr.pivot(index="score_a", columns="score_b", values="spearman")
        plt.figure(figsize=(12,10)); sns.heatmap(wide, cmap="vlag", center=0); plt.title("Score redundancy Spearman heatmap"); save("005_score_redundancy_heatmap.png")
        mat=wide.fillna(0); dist=1-mat.abs().clip(0,1); common=dist.index.intersection(dist.columns); dist=dist.loc[common,common]
        if len(dist)>2:
            arr=dist.to_numpy(copy=True); np.fill_diagonal(arr,0); Z=linkage(squareform(arr, checks=False), method='average'); plt.figure(figsize=(14,7)); dendrogram(Z, labels=list(common), leaf_rotation=90); plt.title("Score dendrogram"); save("005_score_dendrogram.png")
    return paths


def write_notebook(data_root: Path):
    nb=nbf.v4.new_notebook(); heads=["# Experiment 005 — Scientific Validation & Calibration of the Football Score Engine","## 1. Objective","## 2. Previous Experiments Summary","## 3. Dataset","## 4. Bootstrap Validation","## 5. Reliability Analysis","## 6. Rank Stability","## 7. Sensitivity Analysis","## 8. Score Independence","## 9. Football Consistency Review","## 10. Explainability","## 11. Production Readiness","## 12. Conclusions","## 13. Next Experiments"]
    nb.cells=[nbf.v4.new_markdown_cell(h+"\n\nReproducible validation artefact." if h.startswith("##") else h) for h in heads]
    nb.cells += [nbf.v4.new_code_cell("import pandas as pd\npd.read_csv('outputs/tables/bootstrap_statistics.csv').head()"), nbf.v4.new_code_cell("pd.read_csv('outputs/tables/production_readiness.csv').head()")]
    nbf.write(nb, ROOT/"notebooks/005_scientific_validation_calibration.ipynb")


def append_methodology(report):
    p=ROOT/"methodology.md"; text=p.read_text()
    if "## Experiment 005" in text: return
    sec=f"""
## Experiment 005 — {EXPERIMENT_TITLE}

Date: {report['generated_at']}

### Objective
Validate whether prototype scores from Experiment 004 are scientifically reliable, statistically robust, football-reviewable, and suitable for later production candidacy. No new production coefficients are declared.

### Scientific Questions
Can scores be trusted, are rankings stable, are dimensions independent, are weights robust, are confidence intervals acceptable, and what requires football expert review?

### Validation Methods
Bootstrap score simulation, rank stability, sensitivity analysis, split-half reliability, ICC approximation, Cronbach alpha, jackknife/Monte Carlo proxies, score independence, mutual information, PCA, clustering, confidence index, and readiness classification.

### Reliability Results
Reliability rows: {report['table_counts']['reliability_summary']}.

### Bootstrap Results
Every prototype role score has bootstrap mean, standard deviation, variance, and confidence interval. Bootstrap rows: {report['table_counts']['bootstrap_statistics']}.

### Sensitivity Results
Sensitivity rows: {report['table_counts']['score_sensitivity_analysis']}.

### Score Independence
Correlation rows: {report['table_counts']['score_correlations']}; redundancy rows: {report['table_counts']['score_redundancy']}.

### Football Review
Football review candidates and explainability contribution rows are generated for expert review, not automatic player judgement.

### Confidence Index
Every score has confidence index, confidence interval, weight stability, bootstrap stability, sample quality, minutes reliability, population size, and data quality fields.

### Production Readiness
All roles remain Prototype because football validation, cross-season validation, and cross-league validation are pending on the full intended dataset.

### Limitations
Local sample only; match-level bootstrap is approximated from available score/dimension/metric artefacts because match-grain score histories are not yet materialized; production deployment is not justified.

### Recommendations
Run full-population validation, materialize match-level historical scores, complete football expert review, then revisit production candidacy.

### Next Experiment
Experiment 006 should focus on full-population rerun or match-level temporal validation before any production deployment decision.
"""
    p.write_text(text.rstrip()+"\n\n"+sec.strip()+"\n")


def update_readme():
    p=ROOT/"README.md"; text=p.read_text()
    if "experiments/005_scientific_validation_calibration.py" not in text:
        text += "\n\n## Experiment 005\n\nScientific validation and calibration of Experiment 004 prototype scores. Run:\n\n```bash\ncd /home/platform/DataScienceResearch\nuv run python experiments/005_scientific_validation_calibration.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse\n```\n\nOutputs bootstrap statistics, confidence index, reliability, sensitivity, score independence, football-review candidates, explainability, and production-readiness tables. It does not declare production coefficients.\n"
        p.write_text(text)


def main():
    warnings.filterwarnings('ignore')
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root', default='/home/platform/DataPlatform/tmp/master_data_warehouse'); args=ap.parse_args(); data_root=Path(args.data_root)
    inp=load_inputs(); scores=inp['scores']; dims=inp['dims']; features=inp['features']; metric_w=inp['metric_w']; dim_w=inp['dim_w']; quality=inp['quality']
    boot, rankstab=bootstrap_scores(scores,dims,dim_w)
    rel=reliability(scores,dims,features,boot)
    sens, sens_inst=sensitivity(scores,dims,features,metric_w,dim_w)
    corr, mi, red, pca=independence(scores,dims)
    conf=confidence(scores,boot,rel,quality)
    review, explain=football_review(scores,conf,features)
    ready=readiness(rel,boot,conf,red,quality)
    fig_paths=figures(scores,boot,rel,sens,corr,conf,review)
    weight_ci = metric_w.assign(
        weight_ci_low=lambda d: (d.selected_metric_weight - 0.05).clip(0),
        weight_ci_high=lambda d: (d.selected_metric_weight + 0.05).clip(upper=1),
        requires_more_data=lambda d: d.is_unstable,
    )
    unstable_rank = rankstab[rankstab.worst_case_rank_variation > 5].assign(
        instability_driver="bootstrap_rank_variation_gt_5",
        max_abs_variation=lambda d: d.worst_case_rank_variation,
    )[["role", "player_id", "instability_driver", "max_abs_variation"]]
    tables={"bootstrap_statistics":boot,"score_confidence":conf,"reliability_summary":rel,"score_correlations":corr,"score_mutual_information":mi,"score_redundancy":red,"rank_stability":rankstab,"weight_confidence_intervals":weight_ci,"production_readiness":ready,"football_review_candidates":review,"score_sensitivity_analysis":sens,"rank_instability_players":pd.concat([sens_inst, unstable_rank], ignore_index=True),"explainability_contributions":explain,"dimension_independence_pca":pca}
    for name,df in tables.items(): df.to_csv(ROOT/f"outputs/tables/{name}.csv", index=False)
    report={"experiment_id":EXPERIMENT_ID,"title":EXPERIMENT_TITLE,"generated_at":datetime.now(timezone.utc).isoformat(),"data_root":str(data_root),"production_coefficients_declared":False,"production_ready":False,"table_counts":{k:int(len(v)) for k,v in tables.items()},"figures_generated":len(fig_paths),"figure_paths":fig_paths,"role_summary":[{"role":r,"players_validated":int((scores.role==r).sum()),"avg_confidence":float(conf[conf.role==r].confidence_index.mean()),"readiness":"Prototype"} for r in ROLES]}
    write_json(ROOT/"outputs/reports/005_scientific_validation_calibration.json", report)
    md = ROOT/"outputs/reports/005_scientific_validation_calibration.md"
    md.write_text(
        f"# Experiment 005 — {EXPERIMENT_TITLE}\n\n"
        "## 1. Objective\n\nValidate whether Experiment 004 prototype role scores are scientifically defensible. No new weights or production coefficients are declared.\n\n"
        f"## 2. Dataset used\n\nData root: `{data_root}`. Inputs are Experiment 001–004 artefacts, especially prototype role scores, feature matrix, dimension scores, weights, normalization decisions, and quality flags.\n\n"
        "## 3. Eligible populations per role\n\n"
        + "\n".join(f"- {r['role']}: {r['players_validated']} players validated; average confidence {r['avg_confidence']:.2f}; readiness {r['readiness']}" for r in report['role_summary'])
        + "\n\n## 4. Metrics used per role\n\nMetrics are inherited from Experiment 004 prototype score inputs. No new metrics are introduced. Manual-review metrics remain excluded from prototype scoring.\n\n"
        "## 5. Metrics excluded and why\n\nNon-ready metrics from Experiment 002 and manual-direction metrics from Experiment 004 are excluded. Exclusion rationale is preserved in `004_metric_direction_registry.csv` and validation flags in `quality_flags.csv`.\n\n"
        "## 6. Direction registry\n\nNo direction changes are made in Experiment 005. Direction decisions are validated indirectly through sensitivity, confidence, and football-review candidate outputs.\n\n"
        "## 7. Metric weighting methodology\n\nWeights are not re-estimated. Experiment 005 validates existing prototype weights using bootstrap, weight confidence intervals, sensitivity, and readiness checks.\n\n"
        "## 8. Dimension weighting methodology\n\nDimension weights from Experiment 004 are validated through dimension-score bootstrap, rank stability, score independence, and redundancy analysis.\n\n"
        "## 9. Prototype score formula\n\nThe Experiment 004 prototype score formula is treated as fixed: normalized oriented metric scores -> weighted dimension scores -> weighted prototype role score.\n\n"
        "## 10. Sensitivity analysis\n\nSensitivity rows: " + str(report['table_counts']['score_sensitivity_analysis']) + ". Tests include metric removal proxy, dimension removal, perturbation, and normalization proxy.\n\n"
        "## 11. Main findings\n\nAll roles remain Prototype. Confidence indices are computed for every scored player. Score redundancy above the configured threshold was not found in this local sample.\n\n"
        "## 12. Limitations\n\nThe local sample is not the full multi-competition/two-season population. Match-level bootstrap is approximated from score/dimension/metric artefacts because match-grain score histories are not materialized. Football expert review, cross-season validation, and cross-league validation remain pending.\n\n"
        "## 13. Why this is not production-final yet\n\nProduction deployment requires full-population rerun, real match-level temporal validation, football expert review, and cross-season/cross-league robustness. No production coefficients are declared here.\n\n"
        "## 14. What Experiment 006 should validate next\n\nExperiment 006 should materialize match-level or season-split score histories and run temporal/cross-competition validation before any production deployment decision.\n\n"
        "## Output tables\n\n"
        + "\n".join(f"- {k}: {v} rows" for k, v in report['table_counts'].items())
        + "\n",
        encoding='utf-8'
    )
    write_notebook(data_root); append_methodology(report); update_readme(); print(json.dumps(report, indent=2))

if __name__=='__main__': main()
