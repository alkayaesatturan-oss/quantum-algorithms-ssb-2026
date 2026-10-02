"""
Expected Output Format (Section 3): TODAYS_ORDERS.csv (per vehicle, in
visiting order) and CURRENT_INVENTORY.csv (+ utility_score, route
durations, priority coverage) -- pipe-delimited (|) UTF-8 files that
conform exactly to the schema in the Data Dictionary (Appendix B).
"""
from __future__ import annotations

import pandas as pd

from .orders import compute_orders
from .config import Config

TODAYS_ORDERS_COLUMNS = [
    "arac", "locationno", "kasetdurumu", "lokasyon",
    "odenebilir", "odenmez", "tutartl", "atmmodel", "ariza",
]


def _ariza_text(row: pd.Series, cfg: Config, prio_madde_by_code: dict[str, str]) -> str:
    """Appendix B: 'comma-separated rule names or a cassette note'. The name
    of the winning case rule + (if idle-cash removal ("Atıl Sök") is actually
    performed and the winning case is not already K012) the name of the
    idle-cash removal rule."""
    code = row.get("vaka_kodu_final")
    names = [prio_madde_by_code.get(code, code)] if code else []
    idle = bool(row.get("idle_scenario_1", False)) or bool(row.get("idle_scenario_2", False))
    if idle and code != cfg.idle_removal_code:
        names.append(prio_madde_by_code.get(cfg.idle_removal_code, cfg.idle_removal_code))
    return ",".join(str(n) for n in names if n)


def build_todays_orders(scored: pd.DataFrame, routes: list[list[int]],
                         locationno_by_idx: dict[int, str], cfg: Config,
                         prio_madde_by_code: dict[str, str]) -> pd.DataFrame:
    by_loc = scored.set_index("locationno", drop=False)
    visited_locs = [locationno_by_idx[idx] for route in routes for idx in route]
    if not visited_locs:
        return pd.DataFrame(columns=TODAYS_ORDERS_COLUMNS)

    visited = by_loc.loc[visited_locs].copy()
    visited = compute_orders(visited, cfg)
    visited["ariza"] = visited.apply(lambda r: _ariza_text(r, cfg, prio_madde_by_code), axis=1)

    out_rows = []
    for v_i, route in enumerate(routes):
        vehicle_label = f"ARAÇ{v_i + 1}"
        for idx in route:
            loc = locationno_by_idx[idx]
            r = visited.loc[loc]
            out_rows.append({
                "arac": vehicle_label,
                "locationno": loc,
                "kasetdurumu": r["kasetdurumu"],
                "lokasyon": r.get("location", ""),
                "odenebilir": r.get("recyclable_balance", 0.0),
                "odenmez": r.get("unrecyclable_balance", 0.0),
                "tutartl": r["tutartl"],
                "atmmodel": r.get("model", ""),
                "ariza": r["ariza"],
            })
    return pd.DataFrame(out_rows, columns=TODAYS_ORDERS_COLUMNS)


def build_enriched_inventory(raw_inventory: pd.DataFrame, scored: pd.DataFrame,
                              routes: list[list[int]], locationno_by_idx: dict[int, str],
                              route_minutes: list[float], missed_mandatory_locs: list[str],
                              missed_prio1: list[str], missed_prio2: list[str]) -> pd.DataFrame:
    out = raw_inventory.copy()

    score_cols = ["locationno", "vaka_kodu_final", "priority", "ratio",
                  "utility_score", "mandatory", "is_candidate", "unused_cash",
                  "idle_scenario_1", "idle_scenario_2", "next_visit_need"]
    score_view = scored[score_cols].rename(columns={"unused_cash": "unused_cash_hesaplanan"})
    out = out.merge(score_view, on="locationno", how="left")
    out["unused_cash"] = out["unused_cash_hesaplanan"]
    out = out.drop(columns=["unused_cash_hesaplanan"])

    loc_to_route = {}
    loc_to_minutes = {}
    for v_i, route in enumerate(routes):
        label = f"ARAÇ{v_i + 1}"
        for idx in route:
            loc = locationno_by_idx[idx]
            loc_to_route[loc] = label
            loc_to_minutes[loc] = route_minutes[v_i]
    out["arac"] = out["locationno"].map(loc_to_route)
    out["rota_toplam_sure_dk"] = out["locationno"].map(loc_to_minutes)

    out["kacirilan_prio1_sayisi"] = len(missed_prio1)
    out["kacirilan_prio2_sayisi"] = len(missed_prio2)
    out["kacirilan_toplam_sayisi"] = len(missed_mandatory_locs)
    out["kacirilan_lokasyonlar"] = ",".join(missed_mandatory_locs) if missed_mandatory_locs else ""
    return out


def build_route_summary(routes: list[list[int]], route_minutes: list[float],
                         scored: pd.DataFrame, locationno_by_idx: dict[int, str]) -> pd.DataFrame:
    by_loc = scored.set_index("locationno")
    rows = []
    for v_i, (route, minutes) in enumerate(zip(routes, route_minutes)):
        util = sum(by_loc.loc[locationno_by_idx[i], "utility_score"] for i in route)
        rows.append({
            "arac": f"ARAÇ{v_i + 1}",
            "durak_sayisi": len(route),
            "toplam_sure_dk": round(minutes, 1),
            "toplam_utility_score": round(float(util), 2),
            "lokasyonlar": ",".join(locationno_by_idx[i] for i in route),
        })
    return pd.DataFrame(rows)
