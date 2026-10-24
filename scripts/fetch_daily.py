#!/usr/bin/env python3
"""Download TSLA price history from Yahoo Finance into data/.

- data/tsla_daily.csv : daily OHLCV since the 2010-06-29 IPO (split-adjusted)
- data/tsla_hourly.csv: 1-hour bars for the most recent ~730 days (Yahoo's limit)

    python3 scripts/fetch_daily.py
"""
import os
import sys

import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save(df, name, index_label):
    if df.empty:
        sys.exit(f"no data for {name} (is query1.finance.yahoo.com reachable?)")
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.columns = ["open", "high", "low", "close", "volume"]
    out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].round(4)
    out.index.name = index_label
    path = os.path.join(ROOT, "data", name)
    out.to_csv(path)
    print(f"{name}: {len(out)} rows, {out.index[0]} .. {out.index[-1]}, last close {out['close'].iloc[-1]:.2f}")


def main():
    # auto_adjust=False keeps Yahoo's split-adjusted OHLC; Tesla has never paid a dividend.
    daily = yf.download("TSLA", start="2010-06-29", auto_adjust=False, progress=False, multi_level_index=False)
    save(daily, "tsla_daily.csv", "date")
    hourly = yf.download("TSLA", period="730d", interval="1h", auto_adjust=False, progress=False,
                         multi_level_index=False)
    hourly.index = hourly.index.tz_convert("America/New_York").tz_localize(None)
    save(hourly, "tsla_hourly.csv", "datetime")


if __name__ == "__main__":
    main()
