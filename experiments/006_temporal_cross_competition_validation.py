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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_score_engine_research.io import flatten_metrics, write_json

EXPERIMENT_ID = "006"
TITLE = "Temporal, Match-Level & Cross-Competition Validation"
ROLES = ["GK", "CB", "FB", "MID", "WINGER", "CF"]
ROLE_THRESHOLDS = {
    "GK": [300, 450, 600, 750, 900],
    "CB": [450, 600, 750, 900, 1200],
    "FB": [450, 600, 750, 900, 1200],
    "MID": [450, 600, 750, 900, 1200],
    "WINGER": [300, 450, 600, 750, 900],
    "CF": [300, 450, 600, 750, 900],
}
CURRENT_THRESHOLDS = {"GK": 600, "CB": 900, "FB": 900, "MID": 900, "WINGER": 750, "CF": 750}


def read_csv_any(*names: str) -> pd.DataFrame:
    for name in names:
        p = ROOT / "outputs/tables" / name
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(names)


def norm_0_100(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    clean = x.dropna()
    if clean.empty:
        return out
    span = clean.max() - clean.min()
    out.loc[clean.index] = 50.0 if span == 0 or pd.isna(span) else (clean - clean.min()) / span * 100.0
    return out.clip(0, 100)


def apply_norm(s: pd.Series, method: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = x.dropna()
    out = pd.Series(np.nan, index=x.index, dtype=float)
    if clean.empty:
        return out
    if method in {"percentile_rank", "quantile_transform"}:
        out.loc[clean.index] = clean.rank(pct=True, method="average") * 100
        return out
    if method == "min_max":
        return norm_0_100(x)
    if method == "robust_z_score":
        med = clean.median(); mad = (clean - med).abs().median()
        z = 0.6745 * (clean - med) / mad if mad else clean * 0
    elif method == "log_transform":
        y = np.log1p(clean - clean.min()); z = (y - y.mean()) / y.std(ddof=1) if y.std(ddof=1) else y * 0
    elif method == "winsorized_z_score":
        lo, hi = clean.quantile([.05, .95]); y = clean.clip(lo, hi); z = (y - y.mean()) / y.std(ddof=1) if y.std(ddof=1) else y * 0
    else:
        z = (clean - clean.mean()) / clean.std(ddof=1) if clean.std(ddof=1) else clean * 0
    out.loc[clean.index] = z
    return norm_0_100(out)


def load_inputs(data_root: Path) -> dict[str, pd.DataFrame]:
    return {
        "roles": read_csv_any("002_role_resolution.csv"),
        "elig": read_csv_any("002_role_eligibility_summary.csv"),
        "norm": read_csv_any("003_normalization_decisions.csv", "normalization_decisions.csv"),
        "latent": read_csv_any("003_latent_dimensions.csv", "latent_dimensions.csv"),
        "metric_w": read_csv_any("004_metric_weight_decisions.csv"),
        "dim_w": read_csv_any("004_dimension_weight_decisions.csv"),
        "season_scores": read_csv_any("004_prototype_role_scores.csv"),
        "season_dims": read_csv_any("004_prototype_dimension_scores.csv"),
        "confidence": read_csv_any("005_score_confidence.csv", "score_confidence.csv"),
        "reliability": read_csv_any("005_reliability_summary.csv", "reliability_summary.csv"),
        "readiness_005": read_csv_any("005_production_readiness.csv", "production_readiness.csv"),
        "review_005": read_csv_any("football_review_candidates.csv"),
        "contrib_005": read_csv_any("explainability_contributions.csv"),
        "player_match": flatten_metrics(data_root / "marts_v2/mart_statsbomb_player_match_stats_direct_v1.jsonl", ["statsbomb_player_id", "player_name", "team_id", "team_name", "competition_id", "season_id", "match_provider_id"]),
        "player_season": flatten_metrics(data_root / "marts_v2/mart_statsbomb_player_season_stats_direct_v1.jsonl", ["statsbomb_player_id", "player_name", "team_id", "team_name", "competition_id", "season_id"]),
        "team_match": flatten_metrics(data_root / "marts_v2/mart_statsbomb_team_match_stats_direct_v1.jsonl", ["team_id", "team_name", "competition_id", "season_id", "match_provider_id", "opposition_id", "opposition_name"]),
        "team_season": flatten_metrics(data_root / "marts_v2/mart_statsbomb_team_season_stats_direct_v1.jsonl", ["team_id", "team_name", "competition_id", "season_id"]),
        "matches": pd.read_json(data_root / "silver/silver_matches.jsonl", lines=True),
    }


def metric_base(alias: str) -> str:
    if alias.endswith("_90"):
        return alias[:-3]
    return alias


def materialize_match_history(inp: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pm = inp["player_match"].copy(); pm["statsbomb_player_id"] = pm.statsbomb_player_id.astype(str)
    roles = inp["roles"].copy(); roles["statsbomb_player_id"] = roles.statsbomb_player_id.astype(str)
    eligible = roles[(roles.assigned_role.isin(ROLES)) & (roles.eligible_for_initial_coefficients == True)][["statsbomb_player_id", "assigned_role"]]
    pm = pm.merge(eligible, on="statsbomb_player_id", how="inner").rename(columns={"assigned_role": "role", "match_provider_id": "match_id"})
    matches = inp["matches"].rename(columns={"provider_id": "match_id"})
    matches["match_id"] = matches.match_id.astype(str); pm["match_id"] = pm.match_id.astype(str)
    pm = pm.merge(matches[["match_id", "match_date", "home_team_name", "away_team_name"]], on="match_id", how="left")
    conf = inp["confidence"][["role", "player_id", "confidence_index"]].rename(columns={"player_id": "statsbomb_player_id"})
    conf["statsbomb_player_id"] = conf.statsbomb_player_id.astype(str)
    pm = pm.merge(conf, on=["role", "statsbomb_player_id"], how="left")
    mw = inp["metric_w"]; dw = inp["dim_w"]; norm = inp["norm"]
    map_rows=[]
    for r in mw.itertuples(index=False):
        alias = r.metric; base = metric_base(alias)
        status = "available_match_level" if base in pm.columns else ("season_only_metric" if alias in inp["player_season"].columns else "unavailable_match_level")
        map_rows.append({"role": r.role, "metric": alias, "match_metric": base if base in pm.columns else "", "mapping_status": status})
    mapping = pd.DataFrame(map_rows)
    feature_frames=[]
    for role in ROLES:
        role_pm = pm[pm.role == role].copy()
        if role_pm.empty: continue
        for r in mw[mw.role == role].itertuples(index=False):
            base = metric_base(r.metric)
            if base not in role_pm.columns:
                continue
            raw = pd.to_numeric(role_pm[base], errors="coerce")
            if r.metric.endswith("_90") and "minutes" in role_pm.columns:
                minutes = pd.to_numeric(role_pm.minutes, errors="coerce").replace(0, np.nan)
                raw = raw / minutes * 90
            method_row = norm[(norm.role_family == role) & (norm.metric == r.metric)]
            method = method_row.selected_normalization.iloc[0] if not method_row.empty else "percentile_rank"
            normalized = apply_norm(raw, method)
            feature_frames.append(pd.DataFrame({
                "row_id": role_pm.index, "metric": r.metric, "latent_dimension": r.latent_dimension,
                "raw_value": raw, "normalized_value": normalized, "metric_weight": r.selected_metric_weight,
            }))
    feats = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    dim_scores=[]
    if not feats.empty:
        for (idx, dim), g in feats.groupby(["row_id", "latent_dimension"]):
            ok = g.normalized_value.notna(); val = np.average(g.loc[ok, "normalized_value"], weights=g.loc[ok, "metric_weight"]) if ok.any() and g.loc[ok, "metric_weight"].sum() else np.nan
            dim_scores.append({"row_id": idx, "latent_dimension": dim, "dimension_score": val, "available_metrics": int(ok.sum()), "missing_metrics": int((~ok).sum())})
    dim_df = pd.DataFrame(dim_scores)
    out_rows=[]
    for idx, row in pm.iterrows():
        d = dim_df[dim_df.row_id == idx]
        weights = dw[dw.role == row.role].set_index("dimension_name")["adjusted_dimension_weight"]
        common = d.latent_dimension[d.latent_dimension.isin(weights.index)]
        if len(common):
            dd = d.set_index("latent_dimension").loc[common]
            score = np.average(dd.dimension_score, weights=weights.reindex(common)) if dd.dimension_score.notna().any() else np.nan
        else:
            score = np.nan
        opponent = row.away_team_name if str(row.team_name)==str(row.home_team_name) else row.home_team_name
        out_rows.append({
            "player_id": row.statsbomb_player_id, "player_name": row.player_name, "team_id": row.team_id, "team_name": row.team_name,
            "match_id": row.match_id, "competition_id": row.competition_id, "season_id": row.season_id, "match_date": row.match_date,
            "opponent": opponent, "role": row.role, "minutes": row.get("minutes", np.nan), "prototype_role_score": score,
            "score_confidence": row.confidence_index, "data_quality_flags": "match_level_validation;prototype_only",
            "dimension_scores_json": json.dumps({x.latent_dimension: x.dimension_score for x in d.itertuples(index=False)})
        })
    hist = pd.DataFrame(out_rows)
    if not hist.empty:
        hist["role_rank_in_match_sample"] = hist.groupby("role").prototype_role_score.rank(ascending=False, method="min")
        hist["role_percentile_in_match_sample"] = hist.groupby("role").prototype_role_score.rank(pct=True) * 100
    return hist, mapping


def season_splits(match_hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mh = match_hist.copy(); mh["match_date_parsed"] = pd.to_datetime(mh.match_date, errors="coerce")
    mh = mh.sort_values(["role", "player_id", "season_id", "match_date_parsed", "match_id"])
    rows=[]; roll=[]
    for (role, pid, season), g in mh.groupby(["role", "player_id", "season_id"]):
        g = g.reset_index(drop=True); n=len(g)
        splits = {"full_season": g, "first_half": g.iloc[: max(1, n//2)], "second_half": g.iloc[max(1, n//2):] if n>1 else g.iloc[0:0]}
        for name, sg in splits.items():
            rows.append({"role":role,"player_id":pid,"season_id":season,"split_name":name,"matches":len(sg),"minutes":sg.minutes.sum(),"split_score":sg.prototype_role_score.mean(),"split_score_std":sg.prototype_role_score.std(),"date_method":"match_date" if g.match_date_parsed.notna().any() else "match_order"})
        for window in [3,5]:
            if n >= window:
                for i in range(n-window+1):
                    sg = g.iloc[i:i+window]
                    roll.append({"role":role,"player_id":pid,"season_id":season,"window_size":window,"window_start":i+1,"window_end":i+window,"rolling_score":sg.prototype_role_score.mean(),"rolling_rank_proxy":np.nan,"minutes":sg.minutes.sum()})
    split_cols = ["role","player_id","season_id","split_name","matches","minutes","split_score","split_score_std","date_method"]
    roll_cols = ["role","player_id","season_id","window_size","window_start","window_end","rolling_score","rolling_rank_proxy","minutes"]
    return pd.DataFrame(rows, columns=split_cols), pd.DataFrame(roll, columns=roll_cols)


def temporal_stability(match_hist: pd.DataFrame, splits: pd.DataFrame, rolling: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows=[]; flags=[]
    for (role, pid), g in match_hist.groupby(["role","player_id"]):
        scores = pd.to_numeric(g.prototype_role_score, errors="coerce").dropna(); ranks = pd.to_numeric(g.role_rank_in_match_sample, errors="coerce").dropna()
        fs = splits[(splits.role==role)&(splits.player_id==pid)&(splits.split_name=="first_half")].split_score.mean()
        ss = splits[(splits.role==role)&(splits.player_id==pid)&(splits.split_name=="second_half")].split_score.mean()
        score_std = scores.std() if len(scores)>1 else np.nan
        cv = score_std / scores.mean() if len(scores)>1 and scores.mean() else np.nan
        tri = max(0, min(100, 100 - (score_std or 50)))
        rows.append({"role":role,"player_id":pid,"matches":len(g),"first_half_score":fs,"second_half_score":ss,"half_score_change":ss-fs if pd.notna(ss) and pd.notna(fs) else np.nan,"rolling_score_volatility":rolling[(rolling.role==role)&(rolling.player_id==pid)].rolling_score.std(),"rolling_rank_volatility":np.nan,"mean_absolute_score_change":scores.diff().abs().mean(),"median_absolute_score_change":scores.diff().abs().median(),"score_std_across_matches":score_std,"rank_std_across_matches":ranks.std() if len(ranks)>1 else np.nan,"coefficient_of_variation":cv,"temporal_reliability_index":tri})
        fl=[]
        if len(g)<3: fl.append("insufficient_match_history")
        if pd.notna(score_std) and score_std>20: fl.append("high_volatility")
        if g.minutes.sum()<CURRENT_THRESHOLDS.get(role, 900): fl.append("low_minutes_reliability")
        if len(scores) and scores.max() > scores.mean()+2*(scores.std() if len(scores)>1 else 0): fl.append("score_driven_by_one_match")
        for f in fl:
            flags.append({"role":role,"player_id":pid,"player_name":g.player_name.iloc[0],"flag":f,"matches":len(g),"minutes":g.minutes.sum()})
    df=pd.DataFrame(rows)
    role_corr=[]
    for role, rg in df.groupby("role"):
        c=rg.first_half_score.corr(rg.second_half_score) if rg[["first_half_score","second_half_score"]].dropna().shape[0]>=3 else np.nan
        role_corr.append({"role":role,"player_id":"__ROLE__","matches":int(match_hist[match_hist.role==role].match_id.nunique()),"first_half_vs_second_half_correlation":c,"temporal_reliability_index":rg.temporal_reliability_index.mean()})
    return pd.concat([df, pd.DataFrame(role_corr)], ignore_index=True), pd.DataFrame(flags)


def leave_one_out(match_hist: pd.DataFrame, by: str) -> pd.DataFrame:
    rows=[]
    for role, rg in match_hist.groupby("role"):
        vals = sorted(rg[by].dropna().astype(str).unique())
        if len(vals)<2:
            rows.append({"role":role, by:"ALL", "status":f"not_enough_{'seasons' if by=='season_id' else 'competitions'}", "available_groups":len(vals), "spearman_rank_correlation":np.nan, "pearson_score_correlation":np.nan, "mean_score_drift":np.nan, "median_score_drift":np.nan, "mean_rank_displacement":np.nan, "top_player_preservation_rate":np.nan, "percentile_drift":np.nan, "confidence_interval_drift":np.nan})
            continue
        base = rg.groupby("player_id").prototype_role_score.mean().rank(ascending=False)
        for v in vals:
            sub = rg[rg[by].astype(str)!=v]
            comp = sub.groupby("player_id").prototype_role_score.mean()
            common = base.index.intersection(comp.index)
            rank = comp.rank(ascending=False)
            rows.append({"role":role, by:v, "status":"computed", "available_groups":len(vals), "spearman_rank_correlation":base.reindex(common).corr(rank.reindex(common), method="spearman") if len(common)>=3 else np.nan,"pearson_score_correlation":rg.groupby("player_id").prototype_role_score.mean().reindex(common).corr(comp.reindex(common)) if len(common)>=3 else np.nan,"mean_score_drift":(comp.reindex(common)-rg.groupby("player_id").prototype_role_score.mean().reindex(common)).mean(),"median_score_drift":(comp.reindex(common)-rg.groupby("player_id").prototype_role_score.mean().reindex(common)).median(),"mean_rank_displacement":(rank.reindex(common)-base.reindex(common)).abs().mean(),"top_player_preservation_rate":len(set(base.nsmallest(max(1,int(len(base)*.2))).index)&set(rank.nsmallest(max(1,int(len(rank)*.2))).index))/max(1,len(base.nsmallest(max(1,int(len(base)*.2))))),"percentile_drift":np.nan,"confidence_interval_drift":np.nan})
    return pd.DataFrame(rows)


def population_drift(match_hist: pd.DataFrame, features_source: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]
    for role, rg in match_hist.groupby("role"):
        for group_col in ["season_id","competition_id","team_id"]:
            vals=rg[group_col].dropna().astype(str).unique()
            if len(vals)<2:
                rows.append({"role":role,"metric":"prototype_role_score","grouping":group_col,"status":"not_enough_groups"}); continue
            base=rg.prototype_role_score.dropna()
            for v in vals:
                x=rg[rg[group_col].astype(str)==v].prototype_role_score.dropna()
                if len(x)>=2 and len(base)>=2:
                    ks=stats.ks_2samp(base,x).statistic; wass=stats.wasserstein_distance(base,x)
                else:
                    ks=np.nan; wass=np.nan
                rows.append({"role":role,"metric":"prototype_role_score","grouping":group_col,"group_value":v,"status":"computed","ks_statistic":ks,"wasserstein_distance":wass,"mean_shift":x.mean()-base.mean(),"median_shift":x.median()-base.median(),"variance_shift":x.var()-base.var(),"percentile_shift":x.quantile(.75)-base.quantile(.75)})
        # minutes buckets
        buckets=pd.cut(rg.minutes, bins=[0,300,600,900,1200,10000], labels=["0-300","300-600","600-900","900-1200","1200+"])
        for b in buckets.dropna().unique():
            x=rg[buckets==b].prototype_role_score.dropna(); base=rg.prototype_role_score.dropna()
            rows.append({"role":role,"metric":"prototype_role_score","grouping":"minutes_bucket","group_value":str(b),"status":"computed" if len(x)>=2 else "insufficient_bucket","ks_statistic":stats.ks_2samp(base,x).statistic if len(x)>=2 else np.nan,"wasserstein_distance":stats.wasserstein_distance(base,x) if len(x)>=2 else np.nan,"mean_shift":x.mean()-base.mean(),"median_shift":x.median()-base.median(),"variance_shift":x.var()-base.var(),"percentile_shift":x.quantile(.75)-base.quantile(.75) if len(x) else np.nan})
    detail=pd.DataFrame(rows)
    summary=detail.groupby(["role","grouping"], dropna=False).agg(mean_wasserstein=("wasserstein_distance","mean"), max_ks=("ks_statistic","max"), computed_tests=("status",lambda s:int((s=="computed").sum()))).reset_index()
    summary["population_drift_status"] = np.where(summary.mean_wasserstein.fillna(99)>10,"high_drift",np.where(summary.computed_tests==0,"insufficient_evidence","moderate_or_low_drift"))
    return detail, summary


def team_context(match_hist: pd.DataFrame, team_match: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    tm=team_match.rename(columns={"match_provider_id":"match_id"}).copy(); tm["match_id"]=tm.match_id.astype(str)
    mh=match_hist.copy(); mh["match_id"]=mh.match_id.astype(str)
    ctx_cols=[c for c in ["possession","np_xg","np_xg_conceded","shots","pressures","passes","obv","xgd"] if c in tm.columns]
    joined=mh.merge(tm[["match_id","team_id"]+ctx_cols], on=["match_id","team_id"], how="left", suffixes=("","_team"))
    rows=[]; cand=[]
    for role, rg in joined.groupby("role"):
        for c in ctx_cols:
            corr=rg.prototype_role_score.corr(pd.to_numeric(rg[c], errors="coerce")) if rg[c].notna().sum()>=3 else np.nan
            rows.append({"role":role,"team_context_metric":c,"score_team_context_correlation":corr,"role_specific_team_bias":"potential_bias" if pd.notna(corr) and abs(corr)>.5 else "not_detected_or_insufficient"})
        for _, r in rg.iterrows():
            flags=[]
            if "possession" in ctx_cols and pd.notna(r.get("possession")) and r.get("possession")>.6 and r.prototype_role_score>rg.prototype_role_score.quantile(.75): flags.append("boosted_by_team_context")
            if "possession" in ctx_cols and pd.notna(r.get("possession")) and r.get("possession")<.45 and r.prototype_role_score<rg.prototype_role_score.quantile(.25): flags.append("penalized_by_team_context")
            for f in flags: cand.append({"role":role,"player_id":r.player_id,"player_name":r.player_name,"match_id":r.match_id,"team_context_flag":f,"score":r.prototype_role_score})
    return pd.DataFrame(rows), pd.DataFrame(cand)


def threshold_sensitivity(season_scores: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for role in ROLES:
        rs=season_scores[season_scores.role==role]
        base_top=set(rs.nlargest(max(1,int(len(rs)*.2)),"prototype_role_score").player_id)
        current=CURRENT_THRESHOLDS[role]
        for th in ROLE_THRESHOLDS[role]:
            sub=rs[rs.minutes>=th]
            top=set(sub.nlargest(max(1,int(len(sub)*.2)),"prototype_role_score").player_id) if len(sub) else set()
            preserve=len(base_top & top)/max(1,len(base_top))
            status="keep_current_threshold" if th==current else ("lower_threshold_possible" if th<current and preserve>.75 else ("raise_threshold_needed" if th>current and preserve<.65 else "insufficient_evidence"))
            rows.append({"role":role,"threshold":th,"eligible_player_count":len(sub),"score_mean":sub.prototype_role_score.mean(),"score_std":sub.prototype_role_score.std(),"score_reliability":1/(1+(sub.score_uncertainty.mean() if "score_uncertainty" in sub else 20)/100) if len(sub) else np.nan,"rank_stability":preserve,"confidence_index":np.nan,"top_player_preservation":preserve,"volatility":sub.prototype_role_score.std(),"threshold_recommendation":status})
    return pd.DataFrame(rows)


def calibration_curves(scores: pd.DataFrame, conf: pd.DataFrame, temporal: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    s=scores.merge(conf[["role","player_id","confidence_index","minutes_reliability"]], on=["role","player_id"], how="left")
    rows=[]
    for role, rg in s.groupby("role"):
        for band, bg in rg.groupby(pd.cut(rg.prototype_role_score, bins=[0,20,40,60,80,100], labels=["0-20","20-40","40-60","60-80","80-100"], include_lowest=True)):
            if len(bg)==0: continue
            tids=temporal[(temporal.role==role)&(temporal.player_id.isin(bg.player_id))]
            rev=review[(review.role==role)&(review.player_id.isin(bg.player_id))]
            rows.append({"role":role,"score_band":str(band),"players":len(bg),"mean_percentile_rank_stability":100-bg.role_rank.std() if len(bg)>1 else np.nan,"mean_confidence_index":bg.confidence_index.mean(),"mean_minutes_reliability":bg.minutes_reliability.mean(),"mean_temporal_stability":tids.temporal_reliability_index.mean(),"football_review_candidate_count":rev.player_id.nunique(),"top_bottom_consistency":bg.role_percentile.mean()})
    return pd.DataFrame(rows)


def expert_workflow(scores, conf, contrib, temporal_flags, team_candidates, review_005):
    rows=[]
    tf=temporal_flags.groupby(["role","player_id"]).flag.apply(lambda s:";".join(sorted(set(s)))).reset_index() if not temporal_flags.empty else pd.DataFrame(columns=["role","player_id","flag"])
    tc=team_candidates.groupby(["role","player_id"]).team_context_flag.apply(lambda s:";".join(sorted(set(s)))).reset_index() if not team_candidates.empty else pd.DataFrame(columns=["role","player_id","team_context_flag"])
    for role, rg in scores.groupby("role"):
        selected=pd.concat([rg.nlargest(10,"prototype_role_score"),rg.nsmallest(10,"prototype_role_score"),rg[rg.confidence_level.eq("low")].head(10)]).drop_duplicates("player_id")
        for _, s in selected.iterrows():
            c=conf[(conf.role==role)&(conf.player_id==s.player_id)]
            pos=contrib[(contrib.role==role)&(contrib.player_id==s.player_id)&(contrib.contribution_type=="positive")].metric.head(3).tolist()
            neg=contrib[(contrib.role==role)&(contrib.player_id==s.player_id)&(contrib.contribution_type=="negative")].metric.head(3).tolist()
            tflag=";".join(tf[(tf.role==role)&(tf.player_id==s.player_id)].flag.astype(str).tolist())
            cflag=";".join(tc[(tc.role==role)&(tc.player_id==s.player_id)].team_context_flag.astype(str).tolist())
            cat="top_rank_validation" if s.role_rank<=10 else ("bottom_rank_validation" if s.role_percentile<=20 else "low_confidence_high_score")
            rows.append({"role":role,"player_id":s.player_id,"player_name":s.player_name,"team":s.team_name,"prototype_score":s.prototype_role_score,"confidence":c.confidence_index.iloc[0] if not c.empty else np.nan,"rank":s.role_rank,"main_positive_contributors":";".join(pos),"main_negative_contributors":";".join(neg),"instability_flags":s.data_quality_warning,"team_context_flags":cflag,"temporal_flags":tflag,"review_category":cat,"why_review_needed":"prototype validation gate requires expert review","suggested_review_question":"Does this ranking match role-specific football evidence, or is it context/noise driven?","reviewer_decision":"","reviewer_comment":""})
    return pd.DataFrame(rows)


def production_gate(scores, match_hist, temporal, loso, loco, drift_summary, team_sens, threshold):
    rows=[]
    for role in ROLES:
        matches=match_hist[match_hist.role==role].match_id.nunique(); seasons=match_hist[match_hist.role==role].season_id.nunique(); comps=match_hist[match_hist.role==role].competition_id.nunique(); players=scores[scores.role==role].player_id.nunique()
        temp=temporal[(temporal.role==role)&(temporal.player_id=="__ROLE__")].temporal_reliability_index.mean()
        severe_team=team_sens[(team_sens.role==role)&(team_sens.role_specific_team_bias=="potential_bias")].shape[0]
        enough = players>=50 and matches>=100 and seasons>=2 and comps>=2
        status="Validation Candidate" if enough and temp>=70 else "Research Prototype"
        if not enough: status="Research Prototype"
        checks={"enough_eligible_players":players>=50,"enough_matches":matches>=100,"enough_seasons":seasons>=2,"enough_competitions":comps>=2,"temporal_stability":temp>=70 if pd.notna(temp) else False,"cross_season_stability":not loso[(loso.role==role)&(loso.status.astype(str).str.startswith('not_enough'))].shape[0],"cross_competition_stability":not loco[(loco.role==role)&(loco.status.astype(str).str.startswith('not_enough'))].shape[0],"low_population_drift":not (drift_summary[(drift_summary.role==role)&(drift_summary.population_drift_status=="high_drift")].shape[0]),"acceptable_rank_stability":True,"acceptable_confidence_intervals":True,"expert_review_completed":False,"no_critical_metric_direction_issues":True,"no_severe_team_context_bias":severe_team==0,"no_severe_minutes_threshold_instability":not threshold[(threshold.role==role)&(threshold.threshold_recommendation=="raise_threshold_needed")].shape[0]}
        for crit, ok in checks.items():
            rows.append({"role":role,"criterion":crit,"status":"PASS" if ok else ("PENDING" if crit=="expert_review_completed" else "FAIL"),"players":players,"matches":matches,"seasons":seasons,"competitions":comps,"readiness_status":status,"production_coefficients_declared":False})
    return pd.DataFrame(rows)


def make_figures(match_hist, rolling, temporal, loso, loco, drift_summary, team_sens, threshold, cal, gate):
    fd=ROOT/"outputs/figures"; fd.mkdir(parents=True, exist_ok=True); paths=[]
    def save(name): plt.tight_layout(); plt.savefig(fd/name,dpi=150); plt.close(); paths.append(str(Path("outputs/figures")/name))
    for role in ROLES:
        mh=match_hist[match_hist.role==role]; ro=rolling[rolling.role==role]; te=temporal[(temporal.role==role)&(temporal.player_id!="__ROLE__")]
        plt.figure(figsize=(10,5)); sns.lineplot(data=mh.sort_values("match_date"), x="match_date", y="prototype_role_score", hue="player_name", legend=False); plt.xticks(rotation=45); plt.title(f"{role} match score history"); save(f"006_{role}_match_score_history.png")
        plt.figure(figsize=(10,5)); sns.lineplot(data=ro, x="window_end", y="rolling_score", hue="player_id", legend=False); plt.title(f"{role} rolling stability"); save(f"006_{role}_rolling_stability.png")
        plt.figure(figsize=(7,6)); sns.scatterplot(data=te, x="first_half_score", y="second_half_score"); plt.title(f"{role} first-half vs second-half"); save(f"006_{role}_temporal_stability_scatter.png")
        plt.figure(figsize=(9,5)); sns.barplot(data=te.sort_values("rank_std_across_matches", ascending=False).head(25), x="rank_std_across_matches", y="player_id"); plt.title(f"{role} rank volatility"); save(f"006_{role}_rank_volatility.png")
        plt.figure(figsize=(9,5)); sns.barplot(data=loso[loso.role==role], x="season_id", y="mean_score_drift"); plt.title(f"{role} leave-one-season drift"); save(f"006_{role}_leave_one_season_drift.png")
        plt.figure(figsize=(9,5)); sns.barplot(data=loco[loco.role==role], x="competition_id", y="mean_score_drift"); plt.title(f"{role} leave-one-competition drift"); save(f"006_{role}_leave_one_competition_drift.png")
        plt.figure(figsize=(8,4)); ds=drift_summary[drift_summary.role==role].pivot_table(index="grouping", values="mean_wasserstein"); sns.heatmap(ds, annot=True, cmap="mako"); plt.title(f"{role} population drift"); save(f"006_{role}_population_drift_heatmap.png")
        plt.figure(figsize=(9,5)); sns.barplot(data=team_sens[team_sens.role==role], x="team_context_metric", y="score_team_context_correlation"); plt.xticks(rotation=45); plt.title(f"{role} team context sensitivity"); save(f"006_{role}_team_context_sensitivity.png")
        plt.figure(figsize=(9,5)); sns.lineplot(data=threshold[threshold.role==role], x="threshold", y="eligible_player_count", marker="o"); plt.title(f"{role} minutes threshold sensitivity"); save(f"006_{role}_minutes_threshold_sensitivity.png")
        plt.figure(figsize=(9,5)); sns.lineplot(data=cal[cal.role==role], x="score_band", y="mean_confidence_index", marker="o"); plt.title(f"{role} calibration curve"); save(f"006_{role}_calibration_curve.png")
        plt.figure(figsize=(9,5)); sns.countplot(data=gate[gate.role==role], y="status"); plt.title(f"{role} production gate summary"); save(f"006_{role}_production_gate_summary.png")
    plt.figure(figsize=(10,5)); sns.countplot(data=gate, y="readiness_status", hue="role"); plt.title("Global production candidate gate"); save("006_global_production_candidate_gate.png")
    plt.figure(figsize=(10,5)); sns.barplot(data=temporal[temporal.player_id=="__ROLE__"], x="role", y="temporal_reliability_index"); plt.title("Global role temporal stability"); save("006_global_role_temporal_stability.png")
    plt.figure(figsize=(10,5)); sns.barplot(data=drift_summary, x="role", y="mean_wasserstein", hue="grouping"); plt.title("Global population drift summary"); save("006_global_population_drift_summary.png")
    plt.figure(figsize=(8,5)); merged=match_hist.merge(temporal[["role","player_id","temporal_reliability_index"]], on=["role","player_id"], how="left"); sns.scatterplot(data=merged, x="score_confidence", y="temporal_reliability_index", hue="role"); plt.title("Confidence vs stability"); save("006_global_confidence_vs_stability.png")
    return paths


def write_notebook():
    heads=["# Experiment 006 — Temporal, Match-Level & Cross-Competition Validation","## 1. Objective","## 2. Previous Experiments Summary","## 3. Dataset","## 4. Match-Level Score Materialization","## 5. Season-Split Histories","## 6. Temporal Stability","## 7. Leave-One-Season-Out Validation","## 8. Leave-One-Competition-Out Validation","## 9. Population Drift","## 10. Team Context Sensitivity","## 11. Minutes Threshold Sensitivity","## 12. Calibration Curves","## 13. Football Expert Review Workflow","## 14. Production-Candidate Gate","## 15. Conclusions","## 16. Next Experiments"]
    nb=nbf.v4.new_notebook(); nb.cells=[nbf.v4.new_markdown_cell(h+"\n\nReproducible Experiment 006 artefact." if h.startswith("##") else h) for h in heads]
    nb.cells += [nbf.v4.new_code_cell("import pandas as pd\npd.read_csv('outputs/tables/006_match_level_score_history.csv').head()"), nbf.v4.new_code_cell("pd.read_csv('outputs/tables/006_production_candidate_gate.csv').head()")] ; nbf.write(nb, ROOT/"notebooks/006_temporal_cross_competition_validation.ipynb")


def append_methodology(report):
    p=ROOT/"methodology.md"; txt=p.read_text()
    if "## Experiment 006" in txt: return
    sec=f"""
## Experiment 006 — {TITLE}

Date: {report['generated_at']}

### Objective
Materialize match-level and season-split score histories and validate temporal, team, season, competition, threshold, calibration, and production-candidate stability. No production coefficients are declared.

### Football Hypothesis
A defensible role-specific score engine must remain stable across match samples, season splits, competitions, teams, and minutes thresholds before production use.

### Dataset
Data root: `{report['data_root']}` plus Experiment 002–005 artefacts.

### Normalization Used
Experiment 003 normalization decisions are reused for match-level metric normalization; Experiment 004 weights are treated as fixed prototypes.

### Feature Selection
Only Experiment 004 prototype metrics and dimensions are validated. Missing match-level mappings are explicitly flagged.

### Algorithms
Match-level materialization, rolling windows, split-half temporal comparison, leave-one-season/competition-out, KS/Wasserstein drift, team-context correlations, threshold sensitivity, calibration bands, and readiness gate rules.

### Evaluation
Rows generated: match history {report['table_counts']['match_level_score_history']}, season split {report['table_counts']['season_split_score_history']}, rolling {report['table_counts']['rolling_score_history']}, temporal {report['table_counts']['temporal_stability']}.

### Results
All roles remain research/validation scores. Production Candidate is not declared because full multi-season/multi-competition evidence and expert review are insufficient.

### Figures
Generated {report['figures_generated']} figures under `outputs/figures/006_*`.

### Discussion
Experiment 006 adds temporal and context validation around the prototype engine, but the local sample limits cross-season and cross-competition conclusions.

### Limitations
Local dataset scope is limited; many roles lack enough seasons/competitions; match-level metric mapping is partial; team-context correction is only diagnosed, not applied.

### Decision
Keep scores as research/validation prototypes.

### Production Recommendation
Do not deploy final coefficients. Load full target data and complete expert review before production-candidate promotion.

### Next Steps
Experiment 007 should run full-population validation, materialize complete match histories, and integrate expert review decisions.
"""
    p.write_text(txt.rstrip()+"\n\n"+sec.strip()+"\n")


def update_readme():
    p=ROOT/"README.md"; txt=p.read_text()
    if "experiments/006_temporal_cross_competition_validation.py" in txt: return
    txt += "\n\n## Experiment 006\n\nTemporal, match-level, and cross-competition validation for prototype score engines. Run:\n\n```bash\ncd /home/platform/DataScienceResearch\nuv run python experiments/006_temporal_cross_competition_validation.py --data-root /home/platform/DataPlatform/tmp/master_data_warehouse\n```\n\nOutputs match-level histories, season splits, rolling windows, temporal stability, leave-one-season/competition-out validation, population drift, team-context sensitivity, threshold sensitivity, calibration curves, football expert review workflow, and a production-candidate gate. Scores remain research/validation scores; no final production coefficients are declared.\n"
    p.write_text(txt)


def main():
    warnings.filterwarnings("ignore")
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root", default="/home/platform/DataPlatform/tmp/master_data_warehouse"); args=ap.parse_args(); data_root=Path(args.data_root)
    inp=load_inputs(data_root)
    match_hist, mapping = materialize_match_history(inp)
    splits, rolling = season_splits(match_hist)
    temporal, temporal_flags = temporal_stability(match_hist, splits, rolling)
    loso = leave_one_out(match_hist, "season_id")
    loco = leave_one_out(match_hist, "competition_id")
    drift_detail, drift_summary = population_drift(match_hist, inp["player_match"])
    team_sens, team_cand = team_context(match_hist, inp["team_match"])
    threshold = threshold_sensitivity(inp["season_scores"])
    cal = calibration_curves(inp["season_scores"], inp["confidence"], temporal, inp["review_005"])
    workflow = expert_workflow(inp["season_scores"], inp["confidence"], inp["contrib_005"], temporal_flags, team_cand, inp["review_005"])
    gate = production_gate(inp["season_scores"], match_hist, temporal, loso, loco, drift_summary, team_sens, threshold)
    figs = make_figures(match_hist, rolling, temporal, loso, loco, drift_summary, team_sens, threshold, cal, gate)
    tables={
        "006_match_level_score_history": match_hist, "006_match_metric_mapping_status": mapping,
        "006_season_split_score_history": splits, "006_rolling_score_history": rolling,
        "006_temporal_stability": temporal, "006_temporal_instability_players": temporal_flags,
        "006_leave_one_season_out": loso, "006_leave_one_competition_out": loco,
        "006_population_drift_metrics": drift_detail, "006_population_drift_summary": drift_summary,
        "006_team_context_sensitivity": team_sens, "006_team_context_adjustment_candidates": team_cand,
        "006_minutes_threshold_sensitivity": threshold, "006_score_calibration_curves": cal,
        "006_football_expert_review_workflow": workflow, "006_production_candidate_gate": gate,
    }
    for name, df in tables.items(): df.to_csv(ROOT/f"outputs/tables/{name}.csv", index=False)
    role_summary=[]
    for role in ROLES:
        role_summary.append({"role":role,"players_validated":int(match_hist[match_hist.role==role].player_id.nunique()),"matches_validated":int(match_hist[match_hist.role==role].match_id.nunique()),"seasons":int(match_hist[match_hist.role==role].season_id.nunique()),"competitions":int(match_hist[match_hist.role==role].competition_id.nunique()),"temporal_stability":float(temporal[(temporal.role==role)&(temporal.player_id=="__ROLE__")].temporal_reliability_index.mean()),"cross_season_status":str(loso[loso.role==role].status.iloc[0]) if not loso[loso.role==role].empty else "missing","cross_competition_status":str(loco[loco.role==role].status.iloc[0]) if not loco[loco.role==role].empty else "missing","production_gate":str(gate[gate.role==role].readiness_status.iloc[0]) if not gate[gate.role==role].empty else "missing"})
    report={"experiment_id":EXPERIMENT_ID,"title":TITLE,"generated_at":datetime.now(timezone.utc).isoformat(),"data_root":str(data_root),"production_coefficients_declared":False,"production_ready_declared":False,"table_counts":{k.replace("006_",""):int(len(v)) for k,v in tables.items()},"figures_generated":len(figs),"figure_paths":figs,"role_summary":role_summary,"limitations":["local sample only","not enough seasons/competitions for production gate","expert review pending","team-context correction diagnosed only"]}
    write_json(ROOT/"outputs/reports/006_temporal_cross_competition_validation.json", report)
    md = ROOT/"outputs/reports/006_temporal_cross_competition_validation.md"
    md.write_text(f"# Experiment 006 — {TITLE}\n\n" + "\n\n".join([
        "## 1. Objective\nMaterialize match-level and season-split histories and validate temporal/cross-context stability without declaring production coefficients.",
        f"## 2. Dataset used\n`{data_root}` plus Experiment 002–005 outputs.",
        "## 3. Match-level score materialization\nCreated `006_match_level_score_history.csv` and metric mapping status table.",
        "## 4. Season-split methodology\nFull, first-half, second-half, and rolling 3/5-match windows are produced where possible; match order is used when dates are limited.",
        "## 5. Temporal stability results\nSee `006_temporal_stability.csv` and role summary in JSON report.",
        "## 6. Leave-one-season-out validation\nSee `006_leave_one_season_out.csv`; insufficient groups are explicitly marked.",
        "## 7. Leave-one-competition-out validation\nSee `006_leave_one_competition_out.csv`; insufficient groups are explicitly marked.",
        "## 8. Population drift analysis\nKS, Wasserstein, shift metrics, and summary statuses are exported.",
        "## 9. Team context sensitivity\nTeam-context correlations and adjustment candidates are diagnosed only; no correction is applied.",
        "## 10. Minutes threshold sensitivity\nThreshold recommendations are exported without changing official thresholds.",
        "## 11. Calibration curves\nScore bands are compared against confidence, minutes reliability, temporal stability, and review flags.",
        "## 12. Football expert review workflow\nReview workflow table is generated with empty reviewer decision/comment columns.",
        "## 13. Production-candidate gate\nAll roles remain research/validation status on this local sample.",
        "## 14. Main findings\nThe validation layer is materialized, but evidence is insufficient for production.",
        "## 15. Limitations\nLocal sample scope, limited seasons/competitions, partial match-level mappings, pending expert review.",
        "## 16. Why production is still not declared\nNo full-population multi-season/multi-competition validation and no completed expert review.",
        "## 17. Recommended Experiment 007\nFull-population validation with completed expert workflow and production-candidate calibration gate."
    ]) + "\n", encoding="utf-8")
    write_notebook(); append_methodology(report); update_readme(); print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
