"""
Section 1.3 and Section 1.4, Steps 1-2: detection of vaka_kodu (case code),
computation of unused_cash / idle-cash removal, and filtering of the
candidate pool.

The output of this module is a DataFrame that carries exactly the
following fields for each ATM: vaka_kodu (final), priority, prioritize_by, prioritize_first,
weight, mandatory, idle_scenario_1, idle_scenario_2, unused_cash,
next_visit_need, exclusion_reason (the reason, if not a candidate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .demand import DemandModel


def build_prio_lookup(prio_df: pd.DataFrame) -> pd.DataFrame:
    lk = prio_df.dropna(subset=["kural"]).copy()
    lk = lk[lk["kural"].astype(str).str.strip() != ""]
    lk = lk.set_index("kural")
    return lk[["madde", "priority", "prioritize_by", "prioritize_first"]]


def _split_codes(malfunction_code) -> list[str]:
    if pd.isna(malfunction_code):
        return []
    return [c.strip() for c in str(malfunction_code).split(",") if c.strip()]


def _lowest_priority_code(codes: list[str], prio_lookup: pd.DataFrame) -> tuple[str | None, float]:
    """Returns, from the given list of codes, the one that is defined in
    PRIO_LOGIC AND has the smallest priority value. Undefined codes (e.g. a
    K-code that never appears in the PRIO_LOGIC of the dataset) are
    silently discarded -- no error is raised; that code simply does not
    make the location a candidate."""
    best_code, best_prio = None, np.inf
    for c in sorted(codes):  # sorting: deterministic choice in case of ties
        if c in prio_lookup.index:
            p = prio_lookup.loc[c, "priority"]
            if pd.notna(p) and float(p) < best_prio:
                best_code, best_prio = c, float(p)
    return best_code, best_prio


def assign_vaka_kodu(inv: pd.DataFrame, prio_lookup: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Step 1: for each location (excluding VEZNE), collects the candidate
    case codes and selects a SINGLE final code according to
    `cfg.vaka_priority_mode`. This function does NOT yet derive idle-cash
    removal (K012) -- that is added after the unused_cash computation via
    `apply_idle_removal_case`, because triggering K012 depends on the
    balance + demand computation.
    """
    df = inv.copy()

    direct = df["vaka_kodu"].where(df["vaka_kodu"].notna() & (df["vaka_kodu"].astype(str).str.strip() != ""))

    malfunction_fallback = []
    for i, row in df.iterrows():
        if pd.notna(direct.loc[i]):
            malfunction_fallback.append(None)
            continue
        codes = _split_codes(row["malfunction_code"])
        code, _ = _lowest_priority_code(codes, prio_lookup)
        malfunction_fallback.append(code)
    df["_malfunction_fallback"] = malfunction_fallback

    stockout_eligible = (
        pd.to_numeric(df["first_stockout_days"], errors="coerce") == 0
    ) & (cfg.stockout_code in prio_lookup.index)
    df["_stockout_candidate"] = np.where(stockout_eligible, cfg.stockout_code, None)

    df["_direct"] = direct
    return df


def compute_unused_cash_and_idle(df: pd.DataFrame, demand: DemandModel, task_date, cfg: Config) -> pd.DataFrame:
    """Section 1.3: unused_cash + detection of idle-cash removal Scenario 1 /
    Scenario 2. `next_visit_need` = net need for today + tomorrow according
    to BANKAMATIK_DEMAND (the same 2-day window as the revisit-interval
    rule, see demand.py)."""
    df = df.copy()
    is_dual = df["model_unittype"].astype(str).str.strip().eq("Çift Ünite")

    next_visit_need = df["locationno"].apply(
        lambda loc: demand.window_need(loc, task_date, cfg.demand_window_days)
    )
    df["next_visit_need"] = next_visit_need

    unrec = pd.to_numeric(df["unrecyclable_balance"], errors="coerce").fillna(0.0)
    rec = pd.to_numeric(df["recyclable_balance"], errors="coerce").fillna(0.0)

    unused_single = (unrec - cfg.unrecyclable_threshold).clip(lower=0.0)
    unused_dual = (rec - next_visit_need - cfg.unrecyclable_threshold).clip(lower=0.0)
    df["unused_cash"] = np.where(is_dual, unused_dual, unused_single)

    scenario1 = unrec >= cfg.unrecyclable_threshold
    scenario1 &= rec >= next_visit_need  # "for all device types"

    scenario2 = is_dual & (unrec >= cfg.unrecyclable_threshold)
    scenario2 &= (rec - next_visit_need) >= cfg.unrecyclable_threshold

    df["idle_scenario_1"] = scenario1 & ~scenario2
    df["idle_scenario_2"] = scenario2
    df["_idle_eligible"] = scenario1 | scenario2
    df["_idle_candidate"] = np.where(df["_idle_eligible"], cfg.idle_removal_code, None)
    return df


def resolve_final_case(df: pd.DataFrame, prio_lookup: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Merges all candidate codes (_direct, _malfunction_fallback,
    _stockout_candidate, _idle_candidate) according to
    `cfg.vaka_priority_mode` to produce a SINGLE (final) `vaka_kodu`; then
    applies the K001/K005/K006 out-of-scope rule (with the rescue logic)
    and the mandatory flag."""
    df = df.copy()
    finals, reasons = [], []

    for _, row in df.iterrows():
        candidates = []
        if pd.notna(row["_direct"]):
            candidates.append(str(row["_direct"]))
        elif row["_malfunction_fallback"]:
            candidates.append(row["_malfunction_fallback"])
        if row["_stockout_candidate"]:
            candidates.append(row["_stockout_candidate"])
        if row["_idle_candidate"]:
            candidates.append(row["_idle_candidate"])
        candidates = list(dict.fromkeys(candidates))  # drop duplicates, preserve order

        candidates = [c for c in candidates if c in prio_lookup.index]

        if not candidates:
            finals.append(None)
            reasons.append("no_case_assigned")
            continue

        if cfg.vaka_priority_mode == "keep_direct":
            primary = candidates[0] if (pd.notna(row["_direct"]) or row["_malfunction_fallback"]) else None
            if primary is not None:
                chosen = primary
            else:
                chosen, _ = _lowest_priority_code(candidates, prio_lookup)
        else:  # lowest_priority_wins
            chosen, _ = _lowest_priority_code(candidates, prio_lookup)

        if chosen is None:
            finals.append(None)
            reasons.append("no_case_assigned")
            continue

        if chosen in cfg.out_of_scope_codes:
            alt_pool = [c for c in candidates if c not in cfg.out_of_scope_codes]
            if cfg.rescue_technical_with_alternate_case and alt_pool:
                chosen, _ = _lowest_priority_code(alt_pool, prio_lookup)
                reasons.append(f"technical_code_active_alternate_case={chosen}")
            else:
                finals.append(None)
                reasons.append(f"technical_out_of_scope:{chosen}")
                continue

        finals.append(chosen)
        if len(reasons) < len(finals):
            reasons.append("ok")

    df["vaka_kodu_final"] = finals
    df["case_note"] = reasons
    df = df.merge(prio_lookup, left_on="vaka_kodu_final", right_index=True, how="left")
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce")
    df["mandatory"] = df["priority"].isin([1, 2])
    return df


def apply_revisit_rule(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """'Was on the previous list': a location is excluded if the number of
    days since the previous visit is less than `revisit_gap_days` -- UNLESS
    the final case has priority == 1 (priority-1 malfunction rules lift
    this restriction)."""
    df = df.copy()
    dslv = pd.to_numeric(df["days_since_last_visit"], errors="coerce")
    too_soon = dslv.notna() & (dslv < cfg.revisit_gap_days)
    exempt = df["priority"].eq(1)
    blocked = too_soon & ~exempt

    df.loc[blocked, "vaka_kodu_final"] = None
    df.loc[blocked, "case_note"] = "excluded:on_previous_list"
    df.loc[blocked, "mandatory"] = False
    return df


def build_candidate_pool(raw_inventory: pd.DataFrame, prio_df: pd.DataFrame,
                          demand: DemandModel, task_date, cfg: Config) -> pd.DataFrame:
    """Applies all preprocessing steps in order and returns an enriched table
    for the locations other than VEZNE. The rows that enter the candidate
    pool are those with a non-empty `vaka_kodu_final`."""
    prio_lookup = build_prio_lookup(prio_df)

    df = raw_inventory[raw_inventory["type"].astype(str) != "VEZNE"].copy()
    df = assign_vaka_kodu(df, prio_lookup, cfg)
    df = compute_unused_cash_and_idle(df, demand, task_date, cfg)
    df = resolve_final_case(df, prio_lookup, cfg)
    df = apply_revisit_rule(df, cfg)

    weight_map = cfg.category_weights
    df["weight"] = df["category"].map(weight_map).fillna(cfg.default_weight)

    df["is_candidate"] = df["vaka_kodu_final"].notna()
    return df
