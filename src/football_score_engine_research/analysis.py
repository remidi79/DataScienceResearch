from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

@dataclass
class FeatureScreenResult:
    numeric_columns: list[str]
    missingness: pd.DataFrame
    near_zero_variance: list[str]
    correlated_pairs: pd.DataFrame
    pca_variance: pd.DataFrame

def numeric_feature_frame(df: pd.DataFrame, min_non_null: int = 5) -> pd.DataFrame:
    numeric = df.select_dtypes(include=["number"]).copy()
    keep = [c for c in numeric.columns if numeric[c].notna().sum() >= min_non_null]
    return numeric[keep]

def screen_features(df: pd.DataFrame, corr_threshold: float = 0.90) -> FeatureScreenResult:
    X = numeric_feature_frame(df)
    missing = pd.DataFrame({"metric": X.columns, "non_null": X.notna().sum().values, "missing_rate": X.isna().mean().values}).sort_values("missing_rate")
    variances = X.var(numeric_only=True)
    near_zero = variances[variances.fillna(0) <= 1e-12].index.tolist()
    corr = X.corr(numeric_only=True).abs()
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            value = corr.loc[a, b]
            if pd.notna(value) and value >= corr_threshold:
                pairs.append({"metric_a": a, "metric_b": b, "abs_corr": float(value)})
    corr_pairs = pd.DataFrame(pairs).sort_values("abs_corr", ascending=False) if pairs else pd.DataFrame(columns=["metric_a","metric_b","abs_corr"])
    pca_df = pd.DataFrame(columns=["component", "explained_variance_ratio", "cumulative_variance"])
    usable = X.drop(columns=near_zero, errors="ignore").replace([np.inf, -np.inf], np.nan).dropna(axis=1, thresh=max(5, int(len(X)*0.5)))
    if usable.shape[1] >= 2 and usable.shape[0] >= 5:
        filled = usable.fillna(usable.median(numeric_only=True))
        Z = RobustScaler().fit_transform(filled)
        n = min(10, Z.shape[0], Z.shape[1])
        pca = PCA(n_components=n, random_state=42).fit(Z)
        ev = pca.explained_variance_ratio_
        pca_df = pd.DataFrame({"component": [f"PC{i+1}" for i in range(len(ev))], "explained_variance_ratio": ev, "cumulative_variance": ev.cumsum()})
    return FeatureScreenResult(list(X.columns), missing, near_zero, corr_pairs, pca_df)
