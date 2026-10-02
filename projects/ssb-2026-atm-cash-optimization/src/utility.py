"""
Section 1.4, Steps 2-4: computation of the utility_score.

    base      = 10^(max_priority - priority + 1) * 2^(max_priority - priority)
    increment = 10^(max_priority - priority + 1)
    ratio     = [0,1] normalization of the prioritize_by metric within the
                same vaka_kodu_final group, in the prioritize_first direction
                (hi == lo  ->  ratio = 1 for the whole group)
    utility_score = base + (ratio + weight) * increment

max_priority is the highest (numerically largest) priority value OBSERVED
IN THE CANDIDATE POOL (is_candidate==True) -- the value "3" in the
specification is merely illustrative and depends on the dataset (it is 5
in this practice dataset).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_group(g: pd.DataFrame) -> pd.Series:
    metric_col = g["prioritize_by"].iloc[0]
    direction = str(g["prioritize_first"].iloc[0]).strip().lower()
    values = pd.to_numeric(g[metric_col], errors="coerce")
    lo, hi = values.min(), values.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(1.0, index=g.index)
    if direction == "max":
        ratio = (values - lo) / (hi - lo)
    else:  # 'min'
        ratio = (hi - values) / (hi - lo)
    return ratio


def compute_utility_scores(candidates: pd.DataFrame) -> pd.DataFrame:
    df = candidates.copy()
    cand = df[df["is_candidate"]].copy()

    max_priority = cand["priority"].max()
    if pd.isna(max_priority):
        df["utility_score"] = np.nan
        return df

    cand["ratio"] = np.nan
    for _, g in cand.groupby("vaka_kodu_final", dropna=True):
        cand.loc[g.index, "ratio"] = _normalize_group(g)

    p = cand["priority"]
    d = max_priority - p
    base = (10.0 ** (d + 1)) * (2.0 ** d)
    increment = 10.0 ** (d + 1)
    cand["utility_score"] = base + (cand["ratio"] + cand["weight"]) * increment

    df["ratio"] = np.nan
    df["utility_score"] = np.nan
    df.loc[cand.index, "ratio"] = cand["ratio"]
    df.loc[cand.index, "utility_score"] = cand["utility_score"]
    df.attrs["max_priority"] = float(max_priority)
    return df


def assert_priority_band_separation(df: pd.DataFrame) -> None:
    cand = df[df["is_candidate"]]
    bands = cand.groupby("priority")["utility_score"].agg(["min", "max"]).sort_index()
    prios = list(bands.index)
    for i in range(len(prios) - 1):
        hi_prio_band_min = bands.loc[prios[i], "min"]          # higher priority (smaller number)
        lo_prio_band_max = bands.loc[prios[i + 1], "max"]      # lower priority (larger number)
        assert hi_prio_band_min > lo_prio_band_max, (
            f"Priority band overlap: prio={prios[i]} min={hi_prio_band_min} "
            f"<= prio={prios[i+1]} max={lo_prio_band_max}"
        )
