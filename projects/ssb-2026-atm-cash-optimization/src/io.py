"""
Reading of the input files. No file path, date or location name is
hard-coded (submission requirement, item 2) -- everything is read from the
3 files (+ an optional template) located under the given `data_dir`.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import pandas as pd

REQUIRED_INVENTORY_COLS = {
    "locationno", "location", "latitude", "longitude", "type", "category",
    "weight", "model", "all_balance", "recyclable_balance",
    "unrecyclable_balance", "days_since_last_visit", "sec_since_supervisor",
    "malfunction_code", "malfunction_seconds", "model_feature",
    "model_unittype", "max_malfunction_seconds", "unused_cash",
    "sec_since_incident", "vaka_kodu", "first_stockout_days",
}
REQUIRED_DEMAND_COLS = {"date", "locationno", "transaction", "amount"}
REQUIRED_PRIO_COLS = {"madde", "kural", "priority", "prioritize_by", "prioritize_first"}


def _find_one(data_dir: str, *patterns: str) -> str:
    """Returns the FIRST file in the given directory that matches, in a
    case-insensitive manner. Because file names may change on competition
    day, files are located by keyword pattern rather than by exact name."""
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(data_dir, pat)))
        hits += sorted(glob.glob(os.path.join(data_dir, pat.lower())))
        hits += sorted(glob.glob(os.path.join(data_dir, pat.upper())))
        hits = sorted(set(hits))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Under '{data_dir}', no file matching the patterns {patterns} was found."
    )


def _require_columns(df: pd.DataFrame, required: set, name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing expected columns: {sorted(missing)}")


@dataclass
class RawData:
    inventory: pd.DataFrame
    demand: pd.DataFrame
    prio: pd.DataFrame
    data_dir: str


def load_raw_data(data_dir: str) -> RawData:
    """Reads the CURRENT_INVENTORY / BANKAMATIK_DEMAND / PRIO_LOGIC files
    (pipe-delimited, UTF-8) and validates their schema."""
    inv_path = _find_one(data_dir, "*CURRENT_INVENTORY*.csv", "*current_inventory*.csv")
    dem_path = _find_one(data_dir, "*BANKAMATIK_DEMAND*.csv", "*DEMAND*.csv")
    prio_path = _find_one(data_dir, "*PRIO_LOGIC*.csv", "*PRIO*.csv")

    inventory = pd.read_csv(inv_path, sep="|", encoding="utf-8")
    demand = pd.read_csv(dem_path, sep="|", encoding="utf-8")
    prio = pd.read_csv(prio_path, sep="|", encoding="utf-8")

    _require_columns(inventory, REQUIRED_INVENTORY_COLS, "CURRENT_INVENTORY")
    _require_columns(demand, REQUIRED_DEMAND_COLS, "BANKAMATIK_DEMAND")
    _require_columns(prio, REQUIRED_PRIO_COLS, "PRIO_LOGIC")

    inventory["locationno"] = inventory["locationno"].astype(str).str.strip()
    demand["locationno"] = demand["locationno"].astype(str).str.strip()
    demand["date"] = pd.to_datetime(demand["date"])
    prio["kural"] = prio["kural"].astype(str).str.strip()

    return RawData(inventory=inventory, demand=demand, prio=prio, data_dir=data_dir)
