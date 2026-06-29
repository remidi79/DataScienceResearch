from __future__ import annotations

import numpy as np
import pandas as pd

def safe_per90(value: pd.Series, minutes: pd.Series) -> pd.Series:
    return np.where(minutes.fillna(0) > 0, value * 90.0 / minutes, np.nan)

def empirical_cdf_percentile(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average") * 100.0

def role_percentiles(df: pd.DataFrame, value_col: str, role_col: str = "role_family") -> pd.Series:
    return df.groupby(role_col, dropna=False)[value_col].transform(empirical_cdf_percentile)

def robust_zscore(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if not mad or np.isnan(mad):
        return pd.Series(np.nan, index=series.index)
    return 0.6745 * (series - median) / mad

def bayesian_shrink_rate(numerator: pd.Series, denominator: pd.Series, prior_rate: float, prior_weight: float) -> pd.Series:
    return (numerator + prior_rate * prior_weight) / (denominator + prior_weight)
