"""Price series (daily / weekly / hourly OHLC) with float "head" positions for animation."""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Series:
    name: str          # "W", "D", "H"
    label: str         # timeframe badge text
    dates: pd.DatetimeIndex
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray

    def __len__(self):
        return len(self.c)

    def index_of(self, when, side="last"):
        """Index of the last bar at or before `when` (side="last") or first bar at/after it."""
        ts = pd.Timestamp(when)
        if side == "last":
            return int(max(0, self.dates.searchsorted(ts, side="right") - 1))
        return int(min(len(self) - 1, self.dates.searchsorted(ts, side="left")))

    def bar_at(self, head):
        """OHLC of the bar under a float head position; the last bar grows with the fraction."""
        k = int(np.clip(np.floor(head), 0, len(self) - 1))
        f = float(np.clip(head - k, 0.0, 1.0)) if head < len(self) else 1.0
        o, h, l, c = self.o[k], self.h[k], self.l[k], self.c[k]
        if f >= 0.999:
            return k, o, h, l, c, 1.0
        e = f * f * (3 - 2 * f)
        cur = o + (c - o) * e
        hi = max(o, cur, o + (h - o) * min(1.0, f * 1.6))
        lo = min(o, cur, o + (l - o) * min(1.0, f * 1.6))
        return k, o, hi, lo, cur, f


def _load(path, index_col):
    df = pd.read_csv(os.path.join(ROOT, "data", path), parse_dates=[index_col]).set_index(index_col)
    return df


def load_all():
    d = _load("tsla_daily.csv", "date")
    daily = Series("D", "日K", d.index, d.open.values, d.high.values, d.low.values, d.close.values)

    wk = d.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    last_day = d.close.resample("W-FRI").apply(lambda s: s.index[-1] if len(s) else pd.NaT).dropna()
    wk.index = pd.DatetimeIndex(last_day.loc[wk.index].values)
    weekly = Series("W", "周K", wk.index, wk.open.values, wk.high.values, wk.low.values, wk.close.values)

    hr = _load("tsla_hourly.csv", "datetime")
    # Yahoo's last hourly bar of a session misses the closing auction; pin it to the official close.
    last_bar = hr.groupby(hr.index.normalize()).tail(1).index
    for ts in last_bar:
        day = ts.normalize()
        if day in d.index:
            px = d.at[day, "close"]
            hr.at[ts, "close"] = px
            hr.at[ts, "high"] = max(hr.at[ts, "high"], px)
            hr.at[ts, "low"] = min(hr.at[ts, "low"], px)
    hourly = Series("H", "1小时K", hr.index, hr.open.values, hr.high.values, hr.low.values, hr.close.values)
    return {"D": daily, "W": weekly, "H": hourly}, d
