"""
Section 1.5: order amount (replenishment/loading amount) -- this is NOT PART
of the optimization problem; it is a complementary classical/deterministic
step that operates on the selected visits. It is computed only for the
locations that are ACTUALLY VISITED.

The idle-cash removal override depends on the PHYSICAL balance state of the
location (`idle_scenario_1` / `idle_scenario_2`, computed in preprocess.py)
-- NOT on which case code "won" for the utility_score (vaka_kodu_final).
This distinction is critical: even if a location is scored under, e.g.,
case K004 (malfunction), idle-cash removal is still performed during the
visit if the cassette PHYSICALLY satisfies the idle-removal condition (in
one of the 4 audited teams, tying this decision to vaka_kodu_final led to
a concrete error).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _ceil_to_step(x: float, step: float) -> float:
    if x <= 0:
        return 0.0
    return float(np.ceil(x / step - 1e-9) * step)


def compute_order_for_row(row: pd.Series, cfg: Config) -> tuple[str, float]:
    """Returns (kasetdurumu, tutartl). `row` must contain at least the
    following fields: idle_scenario_1, idle_scenario_2, next_visit_need,
    recyclable_balance, vaka_kodu_final."""
    if bool(row.get("idle_scenario_1", False)):
        return "Atıl Sök", 0.0

    if bool(row.get("idle_scenario_2", False)):
        reload_amount = float(row["next_visit_need"])
        if cfg.apply_min_floor_to_idle_reload:
            reload_amount = max(reload_amount, cfg.min_order_amount)
        tutar = _ceil_to_step(reload_amount, cfg.order_rounding_step)
        return "Atıl Sök", tutar

    hedef = max(float(row["next_visit_need"]), cfg.min_order_amount)
    recyclable = float(row.get("recyclable_balance", 0.0) or 0.0)
    emir = _ceil_to_step(max(0.0, hedef - recyclable), cfg.order_rounding_step)

    if cfg.k004_special_case_zero_order and row.get("vaka_kodu_final") == "K004":
        real_need = float(row["next_visit_need"])  # WITHOUT applying the 80k floor
        if recyclable >= real_need:
            emir = 0.0

    return "Yeni Kaset", emir


def compute_orders(visited_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = visited_df.copy()
    kasetdurumu, tutartl = [], []
    for _, row in df.iterrows():
        k, t = compute_order_for_row(row, cfg)
        kasetdurumu.append(k)
        tutartl.append(t)
    df["kasetdurumu"] = kasetdurumu
    df["tutartl"] = tutartl
    return df
