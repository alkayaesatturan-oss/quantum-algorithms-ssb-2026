from __future__ import annotations

import pandas as pd


class DemandModel:
    def __init__(self, demand_df: pd.DataFrame):
        pivot = (
            demand_df
            .pivot_table(index=["locationno", "date"], columns="transaction",
                         values="amount", aggfunc="sum")
            .fillna(0.0)
        )
        for col in ("Cash", "Deposit"):
            if col not in pivot.columns:
                pivot[col] = 0.0
        pivot["net_need"] = (pivot["Cash"] - pivot["Deposit"]).clip(lower=0.0)
        self._net = pivot["net_need"]
        self.dates = sorted(demand_df["date"].unique())
        self.min_date = pd.Timestamp(min(self.dates))
        self.max_date = pd.Timestamp(max(self.dates))

    def net_need(self, locationno: str, date: pd.Timestamp) -> float:
        try:
            return float(self._net.loc[(locationno, pd.Timestamp(date))])
        except KeyError:
            return 0.0

    def window_need(self, locationno: str, start_date: pd.Timestamp, num_days: int) -> float:
        total = 0.0
        for i in range(num_days):
            d = pd.Timestamp(start_date) + pd.Timedelta(days=i)
            if d > self.max_date:
                continue
            total += self.net_need(locationno, d)
        return total
