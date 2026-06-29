from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .normalization import role_percentiles

@dataclass
class ScoreEngineResult:
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: float

def pca_composite_score(df: pd.DataFrame, features: list[str], role_col: str = "role_family") -> ScoreEngineResult:
    rows = []
    loadings = []
    for role, group in df.groupby(role_col, dropna=False):
        X = group[features].replace([np.inf, -np.inf], np.nan).dropna(axis=1, thresh=max(3, int(len(group) * 0.5)))
        valid_features = list(X.columns)
        if len(valid_features) < 2 or len(group) < 5:
            continue
        X = X.fillna(X.median(numeric_only=True))
        pipe = Pipeline([("scaler", RobustScaler()), ("pca", PCA(n_components=1, random_state=42))])
        component = pipe.fit_transform(X).ravel()
        if np.nanmean(component) < 0:
            component = -component
        raw = pd.Series(component, index=group.index)
        norm = (raw - raw.min()) / (raw.max() - raw.min()) * 100 if raw.max() != raw.min() else raw * 0 + 50
        for idx, raw_value in raw.items():
            rows.append({"index": idx, role_col: role, "raw_score": float(raw_value), "normalized_score": float(norm.loc[idx])})
        pca = pipe.named_steps["pca"]
        for feature, weight in zip(valid_features, pca.components_[0]):
            loadings.append({role_col: role, "feature": feature, "loading": float(weight), "abs_loading": float(abs(weight))})
    score_df = pd.DataFrame(rows).set_index("index") if rows else pd.DataFrame(columns=[role_col,"raw_score","normalized_score"])
    if not score_df.empty:
        score_df["role_percentile"] = role_percentiles(score_df, "normalized_score", role_col)
    loadings_df = pd.DataFrame(loadings).sort_values([role_col, "abs_loading"], ascending=[True, False]) if loadings else pd.DataFrame(columns=[role_col,"feature","loading","abs_loading"])
    return ScoreEngineResult(score_df, loadings_df, float("nan"))
