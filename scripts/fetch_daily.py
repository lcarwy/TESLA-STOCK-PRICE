#!/usr/bin/env python3
"""Download TSLA daily closes (split-adjusted) from Yahoo Finance into data/tsla_daily_close.csv.

    python3 scripts/fetch_daily.py
"""
import os
import sys

import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "tsla_daily_close.csv")


def main():
    # auto_adjust=False keeps Yahoo's "Close", which is split-adjusted; Tesla has never paid a dividend.
    df = yf.download("TSLA", start="2010-06-29", auto_adjust=False, progress=False, multi_level_index=False)
    if df.empty:
        sys.exit("no data returned (is query1.finance.yahoo.com reachable?)")
    out = df[["Close"]].rename(columns={"Close": "close"}).round(4)
    out.index.name = "date"
    out.to_csv(OUT)
    print(f"{len(out)} rows, {out.index[0].date()} .. {out.index[-1].date()}, last close {out['close'].iloc[-1]:.2f}")
    print(OUT)


if __name__ == "__main__":
    main()
