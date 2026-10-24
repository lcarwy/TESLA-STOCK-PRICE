"""Storyboard: segments (what the chart does), chapter panels, boxes, chips and effects over time.

All figures shown on screen are computed from data/tsla_daily.csv and data/tsla_hourly.csv.
"""
from dataclasses import dataclass, field

import pandas as pd

from .data import load_all

SERIES, DAILY = load_all()
C = DAILY.close
IPO_ADJ = 17.00 / 15          # IPO price, split-adjusted (1:5 in 2020, 1:3 in 2022)
FIRST_CLOSE = float(C.iloc[0])
LAST_DATE = C.index[-1]
LAST = float(C.iloc[-1])


def close(d):
    return float(C.loc[:pd.Timestamp(d)].iloc[-1])


def day_chg(d):
    i = C.index.get_indexer([pd.Timestamp(d)])[0]
    return (C.iloc[i] / C.iloc[i - 1] - 1) * 100


def pct(a, b):
    return (close(b) / close(a) - 1) * 100


def days(a, b):
    return (pd.Timestamp(b) - pd.Timestamp(a)).days


def f1(x):
    return f"{x:+.1f}%".replace("-", "−")


ATH1, ATH2, ATH3 = close("2021-11-04"), close("2024-12-17"), close("2025-12-16")
CRASH_DAY = "2026-07-23"
PRE_CRASH = close("2026-07-22")
CRASH_LOW_DATE = "2026-07-29"
CRASH_LOW = close(CRASH_LOW_DATE)
_drops = (C.pct_change() * 100).dropna().sort_values()
CRASH_RANK = int(list(_drops.index).index(pd.Timestamp(CRASH_DAY)) + 1)
TOP_DROPS = [(d.strftime("%Y.%m.%d"), v) for d, v in _drops.head(9).items()]

def _halvings():
    """Number of >=50% peak-to-trough declines (zigzag with a 50% rebound reset)."""
    peak, trough, down, n = C.iloc[0], C.iloc[0], False, 0
    for p in C.values:
        if not down:
            peak = max(peak, p)
            if p <= peak * 0.5:
                down, trough, n = True, p, n + 1
        else:
            trough = min(trough, p)
            if p >= trough * 1.5:
                down, peak = False, p
    return n


HALVINGS = _halvings()
H = SERIES["H"]
GAP_OPEN = float(H.o[H.index_of("2026-07-23 09:30")])


# --------------------------------------------------------------------------- structures

@dataclass
class Seg:
    t0: float
    t1: float
    series: str
    head0: str
    head1: str
    left: str = None            # left edge of the view (date); None = series start
    ease: str = "inout"
    yaw: tuple = (12.0, 12.0)
    pitch: tuple = (-4.0, -4.0)
    log: bool = False
    pad: float = 0.13
    min_days: float = 120
    cut: bool = False           # snap the camera instead of easing into this segment
    ytop: float = 1.22
    future_days: float = 0      # extra empty time on the right (forecast)
    cy: float = 1270.0          # screen y of the plot centre
    ph: float = 900.0           # plot height (world units)


@dataclass
class Panel:
    t0: float
    t1: float
    tag: str
    kind: str                   # up / down / gold / flat / info
    dates: str
    title: str
    desc: list = field(default_factory=list)   # [(text, color-key)]
    big_value: str = None       # gold number instead of a title (历史新高)


@dataclass
class Box:
    t0: float
    t1: float
    start: str


@dataclass
class Chip:
    date: str
    text: str
    kind: str
    t_end: float
    above: bool = True


@dataclass
class Badge:
    t: float
    label: str
    base: float


# --------------------------------------------------------------------------- timeline

SEGS = [
    # cold open: the Q2-2026 earnings crash on hourly bars
    Seg(0.0, 2.9, "H", "2026-07-22 12:30", "2026-07-23 15:30", left="2026-07-17 09:30", ease="out",
        yaw=(18, 10), pitch=(-6, -3), min_days=4, cut=True, ytop=1.12, cy=1330, ph=820),
    Seg(2.9, 3.7, "H", "2026-07-23 15:30", "2026-07-23 15:30", left="2026-07-17 09:30", min_days=4, ytop=1.12,
        cy=1330, ph=820),
    # rewind to the IPO
    Seg(3.7, 5.2, "W", "2026-07-24", "2010-07-02", left="2010-06-29", ease="inout", yaw=(8, 12), cut=True,
        min_days=200, ytop=1.15),
    Seg(5.2, 8.8, "W", "2010-07-02", "2012-12-28", left="2010-06-29", ease="inout", min_days=420),
    Seg(8.8, 12.3, "W", "2012-12-28", "2013-09-30", left="2010-06-29", min_days=420),
    Seg(12.3, 16.8, "W", "2013-09-30", "2019-06-07", left="2010-06-29"),
    Seg(16.8, 20.0, "W", "2019-06-07", "2020-02-21", left="2016-06-01", yaw=(12, 16)),
    Seg(20.0, 24.8, "D", "2020-02-19", "2020-03-18", left="2019-06-01", ease="linear", yaw=(16, 18), cut=True),
    Seg(24.8, 30.0, "D", "2020-03-18", "2021-11-04", left="2019-06-01", yaw=(18, 12)),
    Seg(30.0, 31.0, "D", "2021-11-04", "2021-11-04", left="2019-06-01", yaw=(12, 12)),
    Seg(31.0, 35.0, "D", "2021-11-04", "2023-01-03", left="2019-12-01"),
    Seg(35.0, 39.2, "D", "2023-01-03", "2024-04-22", left="2020-06-01"),
    Seg(39.2, 43.2, "D", "2024-04-22", "2024-12-17", left="2021-01-01", yaw=(12, 15)),
    Seg(43.2, 44.2, "D", "2024-12-17", "2024-12-17", left="2021-01-01", yaw=(15, 15)),
    Seg(44.2, 48.2, "D", "2024-12-17", "2025-04-08", left="2022-01-01", yaw=(15, 12)),
    Seg(48.2, 53.2, "D", "2025-04-08", "2025-12-16", left="2023-01-01"),
    Seg(53.2, 54.4, "D", "2025-12-16", "2025-12-16", left="2023-01-01"),
    Seg(54.4, 57.6, "D", "2025-12-16", "2026-07-22", left="2024-01-01"),
    # zoom into the earnings crash on hourly bars
    Seg(57.6, 60.0, "H", "2026-07-22 15:30", "2026-07-22 15:30", left="2026-07-13 09:30", cut=True,
        yaw=(10, 16), pitch=(-3, -5), min_days=5, ytop=1.12, cy=1380, ph=760),
    Seg(60.0, 63.6, "H", "2026-07-23 09:30", "2026-07-23 15:30", left="2026-07-13 09:30", ease="out",
        yaw=(16, 20), min_days=5, ytop=1.12, cy=1380, ph=760),
    Seg(63.6, 67.4, "H", "2026-07-23 15:30", "2026-07-29 15:30", left="2026-07-13 09:30", yaw=(20, 14),
        min_days=5, ytop=1.12, cy=1380, ph=760),
    Seg(67.4, 70.2, "H", "2026-07-29 15:30", "2026-07-29 15:30", left="2026-07-13 09:30", min_days=5, ytop=1.12,
        cy=1380, ph=760),
    Seg(70.2, 75.8, "D", "2026-07-29", "2026-10-23", left="2025-06-01", cut=True),
    # recap of the full history on a log scale, then the open question
    Seg(75.8, 84.2, "W", "2026-10-23", "2026-10-23", left="2010-06-29", log=True, cut=True, yaw=(4, 8),
        pitch=(-2, -3), pad=0.05, ytop=1.0, cy=1400, ph=740),
    Seg(84.2, 94.5, "W", "2026-10-23", "2026-10-23", left="2024-10-01", yaw=(10, 14), pad=0.0,
        future_days=330, ytop=1.25, cy=1330, ph=820),
]
END = SEGS[-1].t1

BADGES = [
    Badge(0.0, "较财报前收盘", PRE_CRASH),
    Badge(3.7, "较 IPO 发行价", IPO_ADJ),
    Badge(31.0, "较历史最高", ATH1),
    Badge(44.2, "较历史最高", ATH2),
    Badge(54.4, "较历史最高", ATH3),
    Badge(57.6, "较财报前收盘", PRE_CRASH),
    Badge(67.4, "较历史最高", ATH3),
    Badge(70.2, "较 7.29 低点", CRASH_LOW),
    Badge(75.8, "较 IPO 发行价", IPO_ADJ),
    Badge(84.2, "较历史最高", ATH3),
]

PANELS = [
    Panel(5.4, 7.0, "上市", "gold", "2010.06.29", "纳斯达克上市",
          [("发行价 $17（拆股调整后 $1.13），首日收涨 ", "w"), (f"{(FIRST_CLOSE / IPO_ADJ - 1) * 100:.0f}%", "r")]),
    Panel(7.0, 8.8, "横盘", "flat", "2010.06 → 2012.12", "漫长横盘",
          [("Roadster 时代，两年半在 $1 ~ $2.5 之间徘徊", "w")]),
    Panel(8.8, 12.3, "急涨", "up", "2013.01 → 2013.09", "Model S 引爆",
          [("5 月首次季度盈利，单日 ", "w"), (f1(day_chg("2013-05-09")), "r"), ("，9 个月 ", "w"),
           (f1(pct("2012-12-31", "2013-09-30")), "r")]),
    Panel(12.3, 16.8, "横盘", "flat", "2013.09 → 2019.06", "六年原地踏步",
          [("Model 3 产能地狱、私有化风波，", "w"), ("六年后股价反而更低", "g")]),
    Panel(16.8, 20.0, "上涨", "up", "2019.06 → 2020.02", "绝地反击",
          [("Q3 意外盈利单日 ", "w"), (f1(day_chg("2019-10-24")), "r"), ("，上海工厂投产", "w")]),
    Panel(20.0, 24.8, "闪崩", "down", "2020.02.19 → 03.18", "疫情闪崩",
          [(f"{days('2020-02-19', '2020-03-18')} 天 ", "w"), (f1(pct("2020-02-19", "2020-03-18")), "g"),
           ("，3 月 16 日单日 ", "w"), (f1(day_chg("2020-03-16")), "g")]),
    Panel(24.8, 30.0, "上涨", "up", "2020.03 → 2021.11", "史诗级暴涨",
          [("1 拆 5、纳入标普 500、市值破万亿，", "w"), (f1(pct("2020-03-18", "2021-11-04")), "r")]),
    Panel(30.0, 31.0, "历史新高", "gold", "2021.11.04 收盘", "", [("较 IPO 发行价 ×", "w"),
                                                          (f"{ATH1 / IPO_ADJ:.0f}", "y")], big_value=f"{ATH1:.2f}"),
    Panel(31.0, 35.0, "利空", "down", "2021.11 → 2023.01", "腰斩再腰斩",
          [("马斯克抛售股票收购推特，叠加加息，", "w"), (f1(pct("2021-11-04", "2023-01-03")), "g")]),
    Panel(35.0, 37.1, "反弹", "up", "2023.01 → 2023.07", "触底反弹",
          [("半年 ", "w"), (f1(pct("2023-01-03", "2023-07-18")), "r")]),
    Panel(37.1, 39.2, "利空", "down", "2023.07 → 2024.04", "降价潮",
          [("全球降价换销量，利润率下滑，", "w"), (f1(pct("2023-07-18", "2024-04-22")), "g")]),
    Panel(39.2, 43.2, "上涨", "up", "2024.04 → 2024.12", "大选行情",
          [("Q3 财报单日 ", "w"), (f1(day_chg("2024-10-24")), "r"), ("，特朗普胜选后一路狂飙", "w")]),
    Panel(43.2, 44.2, "历史新高", "gold", "2024.12.17 收盘", "", [("突破 2021 年高点 ", "w"), (f"{ATH1:.2f}", "y")],
          big_value=f"{ATH2:.2f}"),
    Panel(44.2, 48.2, "急跌", "down", "2024.12 → 2025.04", "关税与抵制",
          [("马斯克从政引发抵制潮，3 月 10 日单日 ", "w"), (f1(day_chg("2025-03-10")), "g")]),
    Panel(48.2, 53.2, "上涨", "up", "2025.04 → 2025.12", "Robotaxi 行情",
          [("关税暂停单日 ", "w"), (f1(day_chg("2025-04-09")), "r"), ("，6 月奥斯汀 Robotaxi 上路", "w")]),
    Panel(53.2, 54.4, "历史新高", "gold", "2025.12.16 收盘", "", [("较上市首日收盘 ×", "w"),
                                                          (f"{ATH3 / FIRST_CLOSE:.0f}", "y")], big_value=f"{ATH3:.2f}"),
    Panel(54.4, 57.6, "利空", "down", "2025.12 → 2026.07", "高位阴跌",
          [("交付增长预期下调、价格战压缩利润，", "w"), (f1(pct("2025-12-16", "2026-07-22")), "g")]),
    Panel(57.6, 60.0, "财报", "gold", "2026.07.22 盘后", "Q2 财报", []),
    Panel(70.2, 75.8, "反弹", "up", "2026.07.29 → 10.23", "超跌反弹",
          [("低点回升 ", "w"), (f1(pct(CRASH_LOW_DATE, LAST_DATE)), "r"), ("，Q3 财报营收持平", "w")]),
]

BOXES = [
    Box(5.4, 8.8, "2010-06-29"),
    Box(8.8, 12.3, "2012-12-31"),
    Box(12.3, 16.8, "2013-09-30"),
    Box(16.8, 20.0, "2019-06-03"),
    Box(20.0, 24.8, "2020-02-19"),
    Box(24.8, 31.0, "2020-03-18"),
    Box(31.0, 35.0, "2021-11-04"),
    Box(35.0, 37.1, "2023-01-03"),
    Box(37.1, 39.2, "2023-07-18"),
    Box(39.2, 44.2, "2024-04-22"),
    Box(44.2, 48.2, "2024-12-17"),
    Box(48.2, 54.4, "2025-04-08"),
    Box(54.4, 57.6, "2025-12-16"),
    Box(70.2, 75.8, CRASH_LOW_DATE),
]

CHIPS = [
    Chip("2013-05-09", f"首次盈利 {f1(day_chg('2013-05-09'))}", "up", 12.3),
    Chip("2018-08-07", "“资金已到位”推文", "flat", 16.8),
    Chip("2019-06-03", f"至暗时刻 ${close('2019-06-03'):.2f}", "down", 16.8, above=False),
    Chip("2020-03-16", f"单日 {f1(day_chg('2020-03-16'))}", "down", 24.8, above=False),
    Chip("2020-08-31", "1 拆 5", "flat", 31.0),
    Chip("2020-12-21", "纳入标普 500", "up", 31.0),
    Chip("2021-10-25", "市值破万亿美元", "up", 31.0),
    Chip("2022-04-25", "收购推特", "down", 35.0),
    Chip("2024-10-24", f"Q3 财报 {f1(day_chg('2024-10-24'))}", "up", 44.2, above=False),
    Chip("2024-11-06", f"大选 {f1(day_chg('2024-11-06'))}", "up", 44.2, above=False),
    Chip("2025-03-10", f"单日 {f1(day_chg('2025-03-10'))}", "down", 48.2, above=False),
    Chip("2025-04-09", f"关税暂停 {f1(day_chg('2025-04-09'))}", "up", 54.4, above=False),
    Chip("2025-06-05", f"与特朗普决裂 {f1(day_chg('2025-06-05'))}", "down", 54.4, above=True),
    Chip("2026-10-22", f"Q3 财报 {f1(day_chg('2026-10-22'))}", "down", 75.8),
]

# instantaneous effects: (time, kind, target, extra); a date string as time = "when the head gets there"
FX = [
    (0.55, "lightning", "head", 3),
    (0.55, "flash", None, 0.35),
    (0.55, "shake", None, 18),
    (2.2, "shatter", "head", 0),
    (3.7, "rewind", None, 1.5),
    ("2020-03-16", "lightning", "2020-03-16", 5),
    ("2020-03-16", "shake", None, 12),
    (30.0, "fireworks", "2021-11-04", 1.0),
    (43.2, "fireworks", "2024-12-17", 1.1),
    ("2025-03-10", "lightning", "2025-03-10", 8),
    ("2025-03-10", "shake", None, 10),
    (53.2, "fireworks", "2025-12-16", 1.3),
    (60.05, "lightning", "head", 11),
    (60.05, "flash", None, 0.5),
    (60.05, "shake", None, 22),
    (66.9, "shatter", "head", 0),
    (66.9, "shake", None, 14),
]
