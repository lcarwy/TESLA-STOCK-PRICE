"""Storyboard: what the chart does over time, chapter panels, explainer cards, chips, effects
and the data-driven conclusion.

Price figures are computed from data/tsla_daily.csv and data/tsla_hourly.csv. Facts about
drivers and capital flows (amounts, dates) come from contemporaneous reporting and are
listed per card in `source`.
"""
from dataclasses import dataclass, field

import numpy as np
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
H = SERIES["H"]
GAP_OPEN = float(H.o[H.index_of("2026-07-23 09:30")])
SHARES_MAR_2025 = 3.216e9     # shares outstanding (10-K, early 2025), used for the 2025-03-10 market-cap loss


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


# --------------------------------------------------------------------------- conclusion statistics

def _holding_stats():
    out = []
    for name, h in (("1 天", 1), ("1 个月", 21), ("半年", 126), ("1 年", 252), ("3 年", 756), ("5 年", 1260)):
        f = (C.shift(-h) / C - 1).dropna()
        out.append((name, (f > 0).mean() * 100, f.median() * 100))
    return out


HOLD_STATS = _holding_stats()
DD = C / C.cummax() - 1
DD_NOW = float(DD.iloc[-1] * 100)
_analog = (C.shift(-252) / C - 1)[(DD <= -0.18) & (DD >= -0.30)].dropna()
ANALOG = {
    "n": len(_analog),
    "win": (_analog > 0).mean() * 100,
    "median": _analog.median() * 100,
    "win_2015": (_analog["2015":] > 0).mean() * 100,
    "bins": [(lbl, ((_analog > lo) & (_analog <= hi)).mean() * 100) for lbl, lo, hi in (
        ("< −50%", -1.0, -0.5), ("−50%~−25%", -0.5, -0.25), ("−25%~0", -0.25, 0.0), ("0~+25%", 0.0, 0.25),
        ("+25%~+50%", 0.25, 0.5), ("+50%~+100%", 0.5, 1.0), ("> +100%", 1.0, 1e9))],
}
MA200 = float(C.rolling(200).mean().iloc[-1])
MAX_DD = float(DD.min() * 100)
VOL = float(C.pct_change().std() * np.sqrt(252) * 100)
BIG_DAYS = int((C.pct_change().abs() > 0.10).sum())


def _dca():
    months = pd.date_range(LAST_DATE - pd.DateOffset(years=5) + pd.offsets.MonthBegin(1), LAST_DATE, freq="MS")
    shares = 0.0
    for m in months:
        shares += 1000 / C.iloc[C.index.searchsorted(m)]
    inv = 1000 * len(months)
    start = C.index[C.index.searchsorted(LAST_DATE - pd.DateOffset(years=5))]
    lump = inv / C[start] * LAST
    return {"months": len(months), "invested": inv, "dca_value": shares * LAST, "dca_cost": inv / shares,
            "lump_value": lump, "lump_cost": float(C[start]), "lump_date": start}


DCA = _dca()

# Wall Street, S&P Global data via stockanalysis.com (Oct 2026)
RATINGS = [("强烈买入", 15), ("买入", 4), ("持有", 21), ("卖出", 2), ("强烈卖出", 2)]
TARGETS = {"low": 128.11, "avg": 397.83, "median": 412.50, "high": 600.0}
RATING_SCORE = sum(n * s for (_, n), s in zip(RATINGS, (5, 4, 3, 2, 1))) / sum(n for _, n in RATINGS)


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
    dim: float = 1.0            # chart opacity


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
    t_end: float = 0.0
    above: bool = True
    t_start: float = 0.0


@dataclass
class Badge:
    t: float
    label: str
    base: float


@dataclass
class Card:
    """Big explainer: why did it move, and what did the money do."""
    t0: float
    t1: float
    kind: str                   # up / down / gold
    question: str
    title: str
    dates: str
    drivers: list               # [(headline, detail)]
    capital: list = field(default_factory=list)   # [(actor, action, amount text, amount $bn or None, in/out/warn)]
    chain: list = field(default_factory=list)     # cause -> effect nodes
    source: str = ""


class Timeline:
    def __init__(self):
        self.t = 0.0
        self.segs, self.panels, self.boxes, self.chips, self.badges, self.cards = [], [], [], [], [], []
        self.fx, self.moods, self.marks = [], [], {}

    def seg(self, dur, series, head0, head1, **kw):
        s = Seg(self.t, self.t + dur, series, head0, head1, **kw)
        self.segs.append(s)
        self.t += dur
        return s

    def mark(self, name):
        self.marks[name] = self.t

    def chapter(self, *, series, start, end, left, move, hold, tag, kind, dates, title, desc, box=None,
                card=None, card_dur=9.5, ath=None, chips=(), mood="up", ease="inout", yaw=(12, 12), badge=None,
                hold_panel=None):
        t0 = self.t
        if badge:
            self.badges.append(Badge(t0, *badge))
        mv = self.seg(move, series, start, end, left=left, ease=ease, yaw=yaw)
        hd = self.seg(hold, series, end, end, left=left, yaw=(yaw[1], yaw[1]))
        if ath:
            self.fx.append((mv.t1, "fireworks", end, ath))
            self.panels.append(Panel(t0, mv.t1, tag, kind, dates, title, desc))
            self.panels.append(Panel(mv.t1, hd.t1, *hold_panel))
        else:
            self.panels.append(Panel(t0, hd.t1, tag, kind, dates, title, desc))
        if box:
            self.boxes.append(Box(t0, hd.t1, box))
        for ch in chips:
            ch.t_start, ch.t_end = t0, hd.t1
            self.chips.append(ch)
        if card:
            ex = self.seg(card_dur, series, end, end, left=left, yaw=(yaw[1], 6), pitch=(-4, -3), cy=1720,
                          ph=300, dim=0.5)
            self.cards.append(Card(ex.t0, ex.t1, **card))
        self.moods.append((t0, self.t, mood))
        return mv, hd


TL = Timeline()

# ---- cold open: the Q2-2026 earnings crash on hourly bars --------------------------------------
TL.seg(4.5, "H", "2026-07-22 12:30", "2026-07-23 15:30", left="2026-07-17 09:30", ease="out", yaw=(18, 10),
       pitch=(-6, -3), min_days=4, cut=True, ytop=1.12, cy=1330, ph=820)
TL.seg(3.0, "H", "2026-07-23 15:30", "2026-07-23 15:30", left="2026-07-17 09:30", min_days=4, ytop=1.12, cy=1330,
       ph=820)
TL.badges.append(Badge(0.0, "较财报前收盘", PRE_CRASH))
TL.fx += [(1.0, "lightning", "head", 3), (1.0, "flash", None, 0.35), (1.0, "shake", None, 18),
          (4.4, "shatter", "head", 0)]
TL.moods.append((0.0, TL.t, "tension"))
TL.mark("cold_end")

# ---- rewind -----------------------------------------------------------------------------------
TL.mark("rewind")
TL.fx.append((TL.t, "rewind", None, 2.8))
TL.seg(2.8, "W", "2026-07-24", "2010-07-02", left="2010-06-29", ease="inout", yaw=(8, 12), cut=True, min_days=200,
       ytop=1.15)
TL.moods.append((TL.marks["rewind"], TL.t, "rewind"))
TL.mark("story")

# ---- chapters -----------------------------------------------------------------------------------
TL.chapter(series="W", start="2010-07-02", end="2010-12-31", left="2010-06-29", move=3.5, hold=1.5,
           tag="上市", kind="gold", dates="2010.06.29", title="纳斯达克上市",
           desc=[("发行价 $17（拆股调整后 $1.13），首日收涨 ", "w"), (f"{(FIRST_CLOSE / IPO_ADJ - 1) * 100:.0f}%", "r")],
           box="2010-06-29", mood="calm", badge=("较 IPO 发行价", IPO_ADJ),
           card=dict(kind="gold", question="谁在押注特斯拉？", title="纳斯达克上市", dates="2010.06",
                     drivers=[("1956 年后首家 IPO 的美国车企", "福特之后半个多世纪，美国第一家上市的汽车公司"),
                              ("押注纯电动", "Roadster 跑车证明电动车也能又快又酷")],
                     capital=[("IPO 募资", "以 $17 发行 1,330 万股", "2.26 亿美元", 0.226, "in"),
                              ("美国能源部", "低息贷款，支持建 Model S 工厂", "4.65 亿美元", 0.465, "in"),
                              ("丰田", "与 IPO 同时入股", "5,000 万美元", 0.05, "in"),
                              ("戴姆勒", "2009 年入股近 10%", "5,000 万美元", 0.05, "in")],
                     source="特斯拉招股书、美国能源部"))

TL.chapter(series="W", start="2010-12-31", end="2012-12-28", left="2010-06-29", move=5.5, hold=1.5,
           tag="横盘", kind="flat", dates="2010.06 → 2012.12", title="漫长横盘",
           desc=[("Roadster 停产、持续亏损，两年半在 $1 ~ $2.5 徘徊", "w")], box="2010-06-29", mood="calm",
           chips=[Chip("2012-06-22", "Model S 开始交付", "up")])

TL.chapter(series="W", start="2012-12-28", end="2013-09-30", left="2010-06-29", move=6.5, hold=1.5,
           tag="急涨", kind="up", dates="2013.01 → 2013.09", title="Model S 引爆",
           desc=[("5 月首次季度盈利，单日 ", "w"), (f1(day_chg("2013-05-09")), "r"), ("，9 个月 ", "w"),
                 (f1(pct("2012-12-31", "2013-09-30")), "r")],
           box="2012-12-31", mood="up", chips=[Chip("2013-05-09", f"首次盈利 {f1(day_chg('2013-05-09'))}", "up")],
           card=dict(kind="up", question="为什么 9 个月涨 4.7 倍？", title="Model S 引爆", dates="2013.01 → 2013.09",
                     drivers=[("首次季度盈利", "2013 年一季度，上市后第一次赚钱"),
                              ("Model S 口碑爆棚", "《Motor Trend》年度车，《消费者报告》99 分"),
                              ("提前 9 年还清政府贷款", "5 月一次性还清能源部 4.65 亿美元")],
                     capital=[("特斯拉", "趁势增发股票 + 可转债", "约 10 亿美元", 1.0, "in"),
                              ("马斯克", "自掏腰包认购增发", "1 亿美元", 0.1, "in"),
                              ("空头", "做空比例高，股价飙升被迫回补（轧空）", "推波助澜", None, "warn")],
                     source="特斯拉财报与公告"))

TL.chapter(series="W", start="2013-09-30", end="2019-06-07", left="2010-06-29", move=9.5, hold=1.5,
           tag="横盘", kind="flat", dates="2013.09 → 2019.06", title="六年原地踏步",
           desc=[("Model 3 产能地狱、私有化风波，", "w"), ("六年后股价反而更低", "g")], box="2013-09-30",
           mood="down", card_dur=10.5,
           chips=[Chip("2016-03-31", "Model 3 发布", "up"), Chip("2018-08-07", "“资金已到位”推文", "flat"),
                  Chip("2019-06-03", f"至暗时刻 ${close('2019-06-03'):.2f}", "down", above=False)],
           card=dict(kind="down", question="为什么六年原地踏步？", title="六年原地踏步", dates="2013.09 → 2019.06",
                     drivers=[("烧钱扩张", "连年亏损，建超级工厂全靠融资"),
                              ("Model 3 产能地狱", "2017—2018 年量产受阻，马斯克睡在工厂"),
                              ("私有化闹剧", "“资金已到位”推文招来 SEC 起诉：卸任董事长、罚款 2,000 万美元")],
                     capital=[("可转债", "2014 年为内华达超级工厂融资", "20 亿美元", 2.0, "in"),
                              ("高收益债", "2017 年为 Model 3 量产发债", "18 亿美元", 1.8, "in"),
                              ("沙特主权基金", "2018 年悄悄买入约 5% 股份", "约 20 亿美元", 2.0, "in")],
                     source="特斯拉公告、SEC、《金融时报》"))

TL.chapter(series="W", start="2019-06-07", end="2020-02-21", left="2016-06-01", move=6.5, hold=1.5,
           tag="上涨", kind="up", dates="2019.06 → 2020.02", title="绝地反击",
           desc=[("Q3 意外盈利单日 ", "w"), (f1(day_chg("2019-10-24")), "r"), ("，上海工厂投产", "w")],
           box="2019-06-03", mood="up", yaw=(12, 16),
           chips=[Chip("2019-10-24", f"Q3 意外盈利 {f1(day_chg('2019-10-24'))}", "up", above=False)],
           card=dict(kind="up", question="为什么 8 个月涨 4 倍？", title="绝地反击", dates="2019.06 → 2020.02",
                     drivers=[("Q3 意外盈利", "2019 年三季度净利润 1.43 亿美元，远超预期"),
                              ("上海超级工厂", "约 10 个月建成投产，打开中国市场"),
                              ("交付达标", "2019 年交付约 36.8 万辆，完成年度目标")],
                     capital=[("特斯拉", "2020 年 2 月高位增发", "20 亿美元", 2.0, "in"),
                              ("马斯克", "认购增发股份", "1,000 万美元", 0.01, "in"),
                              ("空头", "被迫回补，进一步推高股价", "轧空", None, "warn")],
                     source="特斯拉财报与公告"))

TL.chapter(series="D", start="2020-02-19", end="2020-03-18", left="2019-06-01", move=7.0, hold=1.8,
           tag="闪崩", kind="down", dates="2020.02.19 → 03.18", title="疫情闪崩", ease="linear", yaw=(16, 18),
           desc=[(f"{days('2020-02-19', '2020-03-18')} 天 ", "w"), (f1(pct("2020-02-19", "2020-03-18")), "g"),
                 ("，3 月 16 日单日 ", "w"), (f1(day_chg("2020-03-16")), "g")],
           box="2020-02-19", mood="crash", badge=("较前高", close("2020-02-19")),
           chips=[Chip("2020-03-16", f"单日 {f1(day_chg('2020-03-16'))}", "down", above=False)],
           card=dict(kind="down", question=f"为什么 {days('2020-02-19', '2020-03-18')} 天跌掉六成？", title="疫情闪崩",
                     dates="2020.02.19 → 03.18",
                     drivers=[("新冠疫情全球蔓延", "3 月美股 10 天内 4 次熔断"),
                              ("工厂停工", "上海工厂春节后停产，弗里蒙特工厂 3 月下旬停工"),
                              ("涨得太快", f"此前 8 个月 {f1(pct('2019-06-03', '2020-02-19'))}，获利盘集中出逃")],
                     capital=[("全球资金", "恐慌中抛售高估值成长股", "避险出逃", None, "out"),
                              ("美联储", "两次紧急降息至零 + 无限量 QE", "利率 → 0%", None, "in")],
                     chain=["疫情", "停工", "恐慌抛售", "放水救市"],
                     source="美联储、特斯拉公告"))
TL.segs[-3].cut = True

TL.chapter(series="D", start="2020-03-18", end="2021-11-04", left="2019-06-01", move=11.0, hold=2.6,
           tag="上涨", kind="up", dates="2020.03 → 2021.11", title="史诗级暴涨", yaw=(18, 12),
           desc=[("1 拆 5、纳入标普 500、市值破万亿，", "w"), (f1(pct("2020-03-18", "2021-11-04")), "r")],
           box="2020-03-18", mood="up", ath=1.0, card_dur=10.5, badge=("较 IPO 发行价", IPO_ADJ),
           hold_panel=("历史新高", "gold", "2021.11.04 收盘", "",
                       [("较 IPO 发行价 ×", "w"), (f"{ATH1 / IPO_ADJ:.0f}", "y")], f"{ATH1:.2f}"),
           chips=[Chip("2020-08-31", "1 拆 5", "flat"), Chip("2020-12-21", "纳入标普 500", "up"),
                  Chip("2021-10-25", "市值破万亿美元", "up")],
           card=dict(kind="up", question=f"为什么 19 个月涨 {pct('2020-03-18', '2021-11-04') / 100:.0f} 倍？", title="史诗级暴涨",
                     dates="2020.03 → 2021.11",
                     drivers=[("连续盈利", "2020 年首次实现全年盈利"),
                              ("零利率 + 大放水", "便宜的钱涌向高成长股"),
                              ("散户狂欢", "1 拆 5 降低门槛，散户蜂拥而入"),
                              ("Hertz 下单 10 万辆", "2021 年 10 月市值突破 1 万亿美元")],
                     capital=[("指数基金", "纳入标普 500，被动买入", "约 730 亿美元", 73, "in"),
                              ("空头", "2020 年账面亏损，被迫回补", "约 380 亿美元", 38, "warn"),
                              ("特斯拉", "趁高位三次增发融资", "120 亿美元", 12, "warn")],
                     source="标普道琼斯指数、S3 Partners、特斯拉公告"))

TL.chapter(series="D", start="2021-11-04", end="2023-01-03", left="2019-12-01", move=8.5, hold=1.8,
           tag="利空", kind="down", dates="2021.11 → 2023.01", title="腰斩再腰斩",
           desc=[("加息 + 马斯克抛售股票收购推特，", "w"), (f1(pct("2021-11-04", "2023-01-03")), "g")],
           box="2021-11-04", mood="down", card_dur=10.5, badge=("较历史最高", ATH1),
           chips=[Chip("2021-11-08", "推特投票：卖不卖 10%？", "down"), Chip("2022-04-25", "440 亿收购推特", "down")],
           card=dict(kind="down", question="为什么 14 个月跌掉七成？", title="腰斩再腰斩", dates="2021.11 → 2023.01",
                     drivers=[("美联储暴力加息", "利率从 0 加到 4.5%，高估值成长股被杀估值"),
                              ("马斯克分心推特", "440 亿美元收购推特，被迫卖特斯拉股票筹钱"),
                              ("增长见顶担忧", "年底开始降价，市场担心需求放缓")],
                     capital=[("马斯克", "2021 年 11—12 月减持（推特投票后）", "约 164 亿美元", 16.4, "out"),
                              ("马斯克", "2022 年减持（为收购推特筹资）", "约 229 亿美元", 22.9, "out")],
                     chain=["加息", "估值下杀", "大股东抛售", "股价腰斩"],
                     source="SEC 文件、VerityData、美联储"))

TL.chapter(series="D", start="2023-01-03", end="2023-07-18", left="2020-06-01", move=5.5, hold=1.5,
           tag="反弹", kind="up", dates="2023.01 → 2023.07", title="触底反弹",
           desc=[("半年 ", "w"), (f1(pct("2023-01-03", "2023-07-18")), "r")], box="2023-01-03", mood="up",
           card_dur=8.5,
           card=dict(kind="up", question=f"为什么半年涨 {pct('2023-01-03', '2023-07-18'):.0f}%？", title="触底反弹",
                     dates="2023.01 → 2023.07",
                     drivers=[("降价换量", "大幅降价后交付量创新高"),
                              ("AI 热潮", "ChatGPT 引爆 AI 行情，自动驾驶被重新定价"),
                              ("充电标准统一", "福特、通用等相继宣布采用特斯拉充电接口")],
                     source="特斯拉交付数据、公司公告"))

TL.chapter(series="D", start="2023-07-18", end="2024-04-22", left="2020-06-01", move=6.5, hold=1.5,
           tag="利空", kind="down", dates="2023.07 → 2024.04", title="降价潮",
           desc=[("全球降价换销量，利润率下滑，", "w"), (f1(pct("2023-07-18", "2024-04-22")), "g")],
           box="2023-07-18", mood="down",
           card=dict(kind="down", question="为什么又跌掉一半？", title="降价潮", dates="2023.07 → 2024.04",
                     drivers=[("价格战", "毛利率从 29% 降到 18%（2022 Q1 → 2023 Q4）"),
                              ("比亚迪反超", "2023 年四季度纯电销量超过特斯拉"),
                              ("交付下滑", "2024 年一季度交付同比 −8.5%，近四年首次")],
                     capital=[("特拉华法院", "判决作废马斯克 2018 年薪酬方案", "约 560 亿美元", 56, "warn")],
                     source="特斯拉财报、比亚迪公告、特拉华衡平法院"))

TL.chapter(series="D", start="2024-04-22", end="2024-12-17", left="2021-01-01", move=7.5, hold=2.6,
           tag="上涨", kind="up", dates="2024.04 → 2024.12", title="大选行情", yaw=(12, 15),
           desc=[("Q3 财报单日 ", "w"), (f1(day_chg("2024-10-24")), "r"), ("，特朗普胜选后一路狂飙", "w")],
           box="2024-04-22", mood="up", ath=1.1, badge=("较前高", ATH1),
           hold_panel=("历史新高", "gold", "2024.12.17 收盘", "", [("突破 2021 年高点 ", "w"), (f"{ATH1:.2f}", "y")],
                       f"{ATH2:.2f}"),
           chips=[Chip("2024-10-24", f"Q3 财报 {f1(day_chg('2024-10-24'))}", "up", above=False),
                  Chip("2024-11-06", f"大选 {f1(day_chg('2024-11-06'))}", "up", above=False)],
           card=dict(kind="up", question=f"为什么 8 个月 {f1(pct('2024-04-22', '2024-12-17'))}？", title="大选行情",
                     dates="2024.04 → 2024.12",
                     drivers=[("Q3 利润超预期", f"毛利率回升，单日 {f1(day_chg('2024-10-24'))}，11 年来最大涨幅"),
                              ("押注特朗普", "马斯克全力助选，市场预期自动驾驶监管松绑"),
                              ("Robotaxi 预期", "10 月发布 Cybercab 无人出租车概念车")],
                     capital=[("马斯克", "为 2024 年大选砸钱（最大金主）", "超 2.9 亿美元", 0.29, "warn"),
                              ("追涨资金", "大选后蜂拥而入，11 月单月涨幅", f1(pct("2024-10-31", "2024-11-29")), None,
                               "in")],
                     source="特斯拉财报、美国联邦选举委员会"))

TL.chapter(series="D", start="2024-12-17", end="2025-04-08", left="2022-01-01", move=6.5, hold=1.8,
           tag="急跌", kind="down", dates="2024.12 → 2025.04", title="关税与抵制", yaw=(15, 12),
           desc=[("马斯克从政引发抵制潮，3 月 10 日单日 ", "w"), (f1(day_chg("2025-03-10")), "g")],
           box="2024-12-17", mood="down", badge=("较历史最高", ATH2), card_dur=10.0,
           chips=[Chip("2025-03-10", f"单日 {f1(day_chg('2025-03-10'))}", "down", above=False)],
           card=dict(kind="down", question=f"为什么 4 个月 {f1(pct('2024-12-17', '2025-04-08'))}？", title="关税与抵制",
                     dates="2024.12 → 2025.04",
                     drivers=[("品牌反噬", "马斯克领导 DOGE，全球爆发“Tesla Takedown”抗议"),
                              ("销量暴跌", "2025 Q1 交付 33.7 万辆，同比 −13%；欧洲 −37%"),
                              ("关税战", "4 月“对等关税”引发全球衰退恐慌")],
                     capital=[("3 月 10 日", f"单日 {f1(day_chg('2025-03-10'))}，市值蒸发",
                               f"约 {round((close('2025-03-07') - close('2025-03-10')) * SHARES_MAR_2025 / 1e8, -1):,.0f} 亿美元",
                               (close('2025-03-07') - close('2025-03-10')) * SHARES_MAR_2025 / 1e9, "out")],
                     chain=["马斯克从政", "品牌抵制", "销量下滑", "股价腰斩"],
                     source="特斯拉交付数据、InsideEVs、Electrek"))

TL.chapter(series="D", start="2025-04-08", end="2025-12-16", left="2023-01-01", move=9.5, hold=2.6,
           tag="上涨", kind="up", dates="2025.04 → 2025.12", title="Robotaxi 行情",
           desc=[("关税暂停单日 ", "w"), (f1(day_chg("2025-04-09")), "r"), ("，6 月奥斯汀 Robotaxi 上路", "w")],
           box="2025-04-08", mood="up", ath=1.3, card_dur=10.5, badge=("较前高", ATH2),
           hold_panel=("历史新高", "gold", "2025.12.16 收盘", "", [("较上市首日收盘 ×", "w"),
                                                               (f"{ATH3 / FIRST_CLOSE:.0f}", "y")], f"{ATH3:.2f}"),
           chips=[Chip("2025-04-09", f"关税暂停 {f1(day_chg('2025-04-09'))}", "up", above=False),
                  Chip("2025-06-05", f"与特朗普决裂 {f1(day_chg('2025-06-05'))}", "down", above=True),
                  Chip("2025-09-15", "马斯克增持 10 亿美元", "up", above=False)],
           card=dict(kind="up", question=f"为什么 8 个月 {f1(pct('2025-04-08', '2025-12-16'))}？", title="Robotaxi 行情",
                     dates="2025.04 → 2025.12",
                     drivers=[("关税暂停", f"4 月 9 日暂停对等关税，单日 {f1(day_chg('2025-04-09'))}"),
                              ("Robotaxi 上路", "6 月奥斯汀无人驾驶出租车上线"),
                              ("万亿薪酬绑定", "11 月股东大会批准马斯克约 1 万亿美元薪酬方案")],
                     capital=[("马斯克", "9 月公开市场增持（2020 年来首次）", "约 10 亿美元", 1.0, "in"),
                              ("6 月 5 日", "与特朗普公开决裂，单日市值蒸发", "约 1,500 亿美元", 150, "out")],
                     source="SEC 文件、彭博、特斯拉股东大会"))

TL.chapter(series="D", start="2025-12-16", end="2026-07-22", left="2024-01-01", move=6.5, hold=1.8,
           tag="利空", kind="down", dates="2025.12 → 2026.07", title="高位阴跌",
           desc=[("增长放缓、资金分流 SpaceX，", "w"), (f1(pct("2025-12-16", "2026-07-22")), "g")],
           box="2025-12-16", mood="down", badge=("较历史最高", ATH3), card_dur=10.0,
           chips=[Chip("2026-06-12", "SpaceX 上市", "down", above=False)],
           card=dict(kind="down", question="为什么高位阴跌？", title="高位阴跌", dates="2025.12 → 2026.07",
                     drivers=[("增长失速", "2026 Q1 交付 35.8 万辆不及预期，库存多出约 5 万辆"),
                              ("AI 军备竞赛", "资本开支翻倍投向 AI 与机器人，利润被吞噬"),
                              ("资金分流", "6 月 12 日 SpaceX 上市，市值一度超过特斯拉")],
                     capital=[("SpaceX IPO", "史上最大 IPO，分流“马斯克概念”资金", "约 750 亿美元", 75, "out"),
                              ("特斯拉", "投资 xAI（后并入 SpaceX）", "20 亿美元", 2.0, "warn")],
                     source="特斯拉交付数据、SpaceX 招股书、彭博"))

# ---- zoom into the Q2-2026 earnings crash -------------------------------------------------------
TL.mark("crash")
TL.badges.append(Badge(TL.t, "较财报前收盘", PRE_CRASH))
kw = dict(left="2026-07-13 09:30", min_days=5, ytop=1.12, cy=1380, ph=760)
TL.seg(5.5, "H", "2026-07-22 15:30", "2026-07-22 15:30", cut=True, yaw=(10, 16), pitch=(-3, -5), **kw)
TL.mark("gap")
TL.fx += [(TL.t + 0.05, "lightning", "head", 11), (TL.t + 0.05, "flash", None, 0.5),
          (TL.t + 0.05, "shake", None, 22)]
TL.seg(4.5, "H", "2026-07-23 09:30", "2026-07-23 15:30", ease="out", yaw=(16, 20), **kw)
TL.mark("tiles")
TL.seg(5.5, "H", "2026-07-23 15:30", "2026-07-23 15:30", yaw=(20, 20), **kw)
TL.mark("multi")
TL.seg(5.5, "H", "2026-07-23 15:30", "2026-07-29 15:30", yaw=(20, 14), **kw)
TL.fx += [(TL.t - 0.1, "shatter", "head", 0), (TL.t - 0.1, "shake", None, 14)]
TL.mark("summary")
TL.badges.append(Badge(TL.t, "较历史最高", ATH3))
TL.seg(5.5, "H", "2026-07-29 15:30", "2026-07-29 15:30", **kw)
TL.moods.append((TL.marks["crash"], TL.marks["gap"], "suspense"))
TL.moods.append((TL.marks["gap"], TL.marks["summary"], "crash"))
TL.moods.append((TL.marks["summary"], TL.t, "aftermath"))
TL.mark("crash_end")
TL.panels.append(Panel(TL.marks["crash"], TL.marks["gap"], "财报", "gold", "2026.07.22 盘后", "Q2 财报", []))

TL.chapter(series="D", start="2026-07-29", end="2026-10-23", left="2025-06-01", move=6.5, hold=1.8,
           tag="反弹", kind="up", dates="2026.07.29 → 10.23", title="超跌反弹",
           desc=[("低点回升 ", "w"), (f1(pct(CRASH_LOW_DATE, LAST_DATE)), "r"), ("，Q3 利润率回升", "w")],
           box=CRASH_LOW_DATE, mood="up", badge=("较 7.29 低点", CRASH_LOW),
           chips=[Chip("2026-10-22", f"Q3 财报 {f1(day_chg('2026-10-22'))}", "down")],
           card=dict(kind="up", question="为什么能反弹？", title="超跌反弹", dates="2026.07.29 → 10.23",
                     drivers=[("超跌修复", f"7 月底较历史最高 {f1((CRASH_LOW / ATH3 - 1) * 100)}，抄底资金进场"),
                              ("Cybercab 上路", "加入奥斯汀 Robotaxi 车队，无人驾驶累计近 200 万英里"),
                              ("Q3 小幅超预期", "EPS 0.48 美元（预期 0.45），毛利率回升至 19.6%")],
                     capital=[("华尔街", "44 位分析师：19 买入 / 21 持有 / 4 卖出", "分歧", None, "warn")],
                     source="特斯拉 Q3 2026 财报、S&P Global"))
TL.segs[-3].cut = True

# ---- recap, conclusion, outro --------------------------------------------------------------------
TL.mark("recap")
TL.badges.append(Badge(TL.t, "较 IPO 发行价", IPO_ADJ))
TL.seg(11.0, "W", "2026-10-23", "2026-10-23", left="2010-06-29", log=True, cut=True, yaw=(4, 8), pitch=(-2, -3),
       pad=0.05, ytop=1.0, cy=1400, ph=740)
TL.moods.append((TL.marks["recap"], TL.t, "lift"))
TL.mark("concl")
CONCL_PARTS = [("hold", 10.0), ("analog", 11.0), ("street", 11.0), ("dca", 10.0), ("verdict", 13.0)]
for name, dur in CONCL_PARTS:
    TL.mark(f"concl_{name}")
    TL.seg(dur, "W", "2026-10-23", "2026-10-23", left="2010-06-29", log=True, yaw=(8, 8), pitch=(-3, -3), pad=0.05,
           ytop=1.0, cy=1700, ph=300, dim=0.18)
TL.badges.append(Badge(TL.marks["concl"], "较历史最高", ATH3))
TL.moods.append((TL.marks["concl"], TL.t, "explain"))
TL.mark("outro")
TL.seg(10.0, "W", "2026-10-23", "2026-10-23", left="2024-10-01", yaw=(10, 14), pad=0.0, future_days=330, ytop=1.25,
       cy=1330, ph=820, cut=True)
TL.moods.append((TL.marks["outro"], TL.t, "outro"))

SEGS, PANELS, BOXES, CHIPS, BADGES, CARDS = TL.segs, TL.panels, TL.boxes, TL.chips, TL.badges, TL.cards
MOODS, M = TL.moods, TL.marks
FX = TL.fx + [("2020-03-16", "lightning", "2020-03-16", 5), ("2020-03-16", "shake", None, 12),
              ("2025-03-10", "lightning", "2025-03-10", 8), ("2025-03-10", "shake", None, 10)]
END = TL.t
COVID = next((p.t0, p.t1) for p in PANELS if p.title == "疫情闪崩")
