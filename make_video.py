#!/usr/bin/env python3
"""Render an animated Tesla (TSLA) price-history video from data/tsla_monthly_close.csv.

Scenes: title card -> animated line chart (dynamic axes, event call-outs) ->
log-scale recap -> summary stats card. Frames are rendered with matplotlib in
parallel chunks, each piped straight into ffmpeg, then concatenated.

    python3 make_video.py                 # 1920x1080 @ 30 fps -> output/tsla_stock_history.mp4
    python3 make_video.py --fps 60 --music
    python3 make_video.py --preview 40    # write a single PNG of the frame at t=40s
"""
import argparse
import csv
import datetime as dt
import logging
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator, FixedLocator

ROOT = os.path.dirname(os.path.abspath(__file__))
W, H, DPI = 1920, 1080, 100

BG = "#0b0d12"
PANEL = "#141922"
GRID = "#1e2430"
MUTED = "#7f8a9b"
TEXT = "#f3f5f8"
RED = "#e82127"    # Tesla red, also "up" (红涨绿跌)
GREEN = "#1fbf75"  # "down"

FONT = ["Inter", "WenQuanYi Zen Hei"]
FONT_SEMI = ["Inter SemiBold", "WenQuanYi Zen Hei"]
FONT_BOLD = ["Inter Display ExtraBold", "WenQuanYi Zen Hei"]
matplotlib.rcParams["font.family"] = FONT
matplotlib.rcParams["text.parse_math"] = False  # literal "$" in prices
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

IPO_DATE = dt.date(2010, 6, 29)
IPO_PRICE = 17.00          # nominal IPO price
SPLIT_FACTOR = 15          # 1:5 (2020-08-31) x 1:3 (2022-08-25)
IPO_ADJ = IPO_PRICE / SPLIT_FACTOR
LAST_DATE = dt.date(2026, 10, 23)
INVEST = 10_000

EVENTS = [
    ("2010-06", "纳斯达克上市，发行价 $17"),
    ("2012-06", "Model S 开始交付"),
    ("2013-05", "首次实现季度盈利"),
    ("2015-09", "Model X 开始交付"),
    ("2016-03", "Model 3 发布"),
    ("2017-07", "Model 3 开始交付"),
    ("2018-08", "马斯克“私有化”推文风波"),
    ("2019-12", "上海超级工厂首批交付"),
    ("2020-03", "新冠疫情冲击，股价急跌"),
    ("2020-08", "1 拆 5 拆股"),
    ("2020-12", "纳入标普 500 指数"),
    ("2021-10", "市值突破 1 万亿美元"),
    ("2022-04", "马斯克宣布收购推特"),
    ("2022-08", "1 拆 3 拆股"),
    ("2022-12", "2022 全年下跌 65%"),
    ("2024-12", "美国大选后两个月大涨 62%"),
    ("2025-06", "Robotaxi 在奥斯汀试运营"),
    ("2025-11", "股东会通过马斯克新薪酬方案"),
]

# Seconds of screen time per month, by calendar year of the month.
def month_seconds(year):
    if year < 2013:
        return 0.22
    if year < 2020:
        return 0.30
    return 0.40

EVENT_LINGER = 0.3    # extra time on the month after each event so call-outs can be read
EVENT_MIN_GAP = 2.4   # minimum screen time between consecutive events

INTRO = 3.2
XFADE = 0.8
HOLD = 2.4
LOG_FADE, LOG_HOLD = 1.0, 4.8
STATS_FADE, STATS_HOLD = 1.0, 7.0
OUT_FADE = 1.2
CALLOUT_IN, CALLOUT_HOLD, CALLOUT_OUT = 0.3, 3.4, 0.3


# ---------------------------------------------------------------- data

def load_data():
    months, closes, sources = [], [], []
    with open(os.path.join(ROOT, "data", "tsla_monthly_close.csv")) as f:
        for row in csv.DictReader(f):
            y, m = map(int, row["month"].split("-"))
            months.append((y, m))
            closes.append(float(row["close"]) if row["close"] else np.nan)
            sources.append(row["source"])
    closes = np.array(closes)
    idx = np.arange(len(closes))
    gaps = np.isnan(closes)
    closes[gaps] = np.interp(idx[gaps], idx[~gaps], closes[~gaps])
    xs = np.array([y + m / 12 for y, m in months])
    xs[-1] = LAST_DATE.year + (LAST_DATE.timetuple().tm_yday - 1) / 365.25
    return months, xs, closes, sources


MONTHS, XS, YS, SOURCES = load_data()
N = len(YS)
EVENT_IDX = [MONTHS.index(tuple(map(int, d.split("-")))) for d, _ in EVENTS]
SEG_T = np.array([month_seconds(MONTHS[i][0]) + (EVENT_LINGER if i in EVENT_IDX else 0.0)
                  for i in range(N - 1)])
for a, b in zip(EVENT_IDX, EVENT_IDX[1:]):  # stretch dense clusters so every call-out gets its time
    SEG_T[a] += max(0.0, EVENT_MIN_GAP - SEG_T[a:b].sum())
CUM_T = np.concatenate([[0.0], np.cumsum(SEG_T)])
MAIN_DUR = CUM_T[-1]
MAIN_START = INTRO
MAIN_END = MAIN_START + MAIN_DUR
LOG_START = MAIN_END + HOLD
STATS_START = LOG_START + LOG_FADE + LOG_HOLD
TOTAL = STATS_START + STATS_FADE + STATS_HOLD + OUT_FADE

EVENT_T = [CUM_T[i] for i in EVENT_IDX]
# One call-out on screen at a time: each lasts until its hold ends or the next event starts.
EVENT_END = [min(t + CALLOUT_IN + CALLOUT_HOLD, EVENT_T[j + 1] if j + 1 < len(EVENT_T) else math.inf)
             for j, t in enumerate(EVENT_T)]


def progress(tm):
    """Main-scene time (s) -> fractional month index."""
    tm = min(max(tm, 0.0), MAIN_DUR)
    i = min(int(np.searchsorted(CUM_T, tm, side="right") - 1), N - 2)
    return i + (tm - CUM_T[i]) / SEG_T[i]


def head_at(p):
    i = min(int(p), N - 2)
    f = p - i
    return XS[i] + f * (XS[i + 1] - XS[i]), YS[i] + f * (YS[i + 1] - YS[i])


def smoothed_ymax(fps):
    """Per-frame y-axis top for the main scene: eased toward 1.25 x running max."""
    n = int(math.ceil((MAIN_DUR + HOLD) * fps)) + 2
    out = np.empty(n)
    cur, tau = None, 0.45
    k = 1 - math.exp(-1 / (fps * tau))
    for f in range(n):
        p = progress(f / fps)
        i = int(p)
        _, hy = head_at(p)
        target = max(YS[: i + 1].max(), hy) * 1.25
        cur = target if cur is None else cur + (target - cur) * k
        out[f] = max(cur, hy * 1.12)
    return out


def stats():
    last = YS[-1]
    years = (LAST_DATE - IPO_DATE).days / 365.25
    peak, peak_i, mdd, mdd_pair = YS[0], 0, 0.0, (0, 0)
    for i, y in enumerate(YS):
        if y > peak:
            peak, peak_i = y, i
        dd = y / peak - 1
        if dd < mdd:
            mdd, mdd_pair = dd, (peak_i, i)
    return dict(
        last=last,
        multiple=last / IPO_ADJ,
        total_pct=(last / IPO_ADJ - 1) * 100,
        cagr=((last / IPO_ADJ) ** (1 / years) - 1) * 100,
        value=INVEST / IPO_ADJ * last,
        mdd=mdd * 100,
        mdd_from=MONTHS[mdd_pair[0]],
        mdd_to=MONTHS[mdd_pair[1]],
        max_i=int(np.argmax(YS)),
        min_after_peak_i=mdd_pair[1],
    )


# ---------------------------------------------------------------- formatting

def fmt_price(v):
    if v == 0:
        return "$0"
    return "$" + f"{v:,.2f}".rstrip("0").rstrip(".")


def fmt_wan(v):
    w = v / 1e4
    if w < 10:
        s = f"{w:.2f}"
    elif w < 100:
        s = f"{w:.1f}"
    else:
        s = f"{w:,.0f}"
    return f"{s} 万美元"


def month_label(i):
    y, m = MONTHS[i]
    if i == N - 1:
        return f"{LAST_DATE.year}年{LAST_DATE.month:02d}月{LAST_DATE.day:02d}日"
    return f"{y}年{m:02d}月"


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def new_fig():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    return fig


def fig_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def red_gradient(alpha_top=0.40, power=1.5):
    g = np.linspace(1, 0, 256).reshape(-1, 1) ** power
    rgba = np.zeros((256, 1, 4))
    rgba[..., :3] = to_rgb(RED)
    rgba[..., 3] = alpha_top * g
    return rgba


def style_axes(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(colors=MUTED, labelsize=18, length=0, pad=10)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(FONT)
    ax.grid(axis="y", color=GRID, linewidth=1.2)
    ax.set_axisbelow(True)


def header(fig, title, subtitle):
    fig.text(0.04, 0.935, title, color=TEXT, fontsize=34, fontfamily=FONT_SEMI, va="center")
    fig.text(0.96, 0.935, subtitle, color=MUTED, fontsize=19, fontfamily=FONT, va="center", ha="right")
    fig.text(0.96, 0.03, "数据：StatMuse / Digrin 月收盘价（已按拆股调整）  ·  仅供参考，不构成投资建议",
             color="#4d5666", fontsize=14, fontfamily=FONT, ha="right", va="center")


# ---------------------------------------------------------------- scenes

class MainScene:
    def __init__(self):
        self.fig = fig = new_fig()
        header(fig, "特斯拉（TSLA）上市以来股价走势", "月收盘价 · 美元 · 已按 2020、2022 年两次拆股调整")
        self.ax = ax = fig.add_axes([0.065, 0.085, 0.895, 0.635])
        style_axes(ax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(lambda v, _: fmt_price(v))

        self.fill_im = ax.imshow(red_gradient(), aspect="auto", extent=[0, 1, 0, 1],
                                 origin="upper", zorder=1, interpolation="bilinear")
        self.fill_poly = Polygon([[0, 0], [1, 0], [1, 1]], closed=True, transform=ax.transData)
        self.glow = [ax.plot([], [], color=RED, lw=lw, alpha=a, solid_capstyle="round",
                             solid_joinstyle="round", zorder=2)[0]
                     for lw, a in ((16, 0.05), (9, 0.12))]
        self.line = ax.plot([], [], color=RED, lw=3.4, solid_capstyle="round",
                            solid_joinstyle="round", zorder=3)[0]
        self.markers = ax.scatter([], [], s=70, facecolor=TEXT, edgecolor=RED, linewidth=2.2, zorder=4)
        self.head_glow = ax.scatter([0], [0], s=1400, color=RED, alpha=0.10, zorder=5, linewidth=0)
        self.head_ring = ax.scatter([0], [0], s=300, facecolor="none", edgecolor=RED, linewidth=2, zorder=5)
        self.head = ax.scatter([0], [0], s=150, color=TEXT, edgecolor=RED, linewidth=3, zorder=6)
        self.head_label = ax.annotate("", xy=(0, 0), xytext=(22, 0), textcoords="offset pixels",
                                      va="center", ha="left", fontsize=22, color=TEXT,
                                      fontfamily=FONT_BOLD, zorder=7,
                                      bbox=dict(boxstyle="round,pad=0.35", fc=RED, ec="none"))

        # read-out strip between the header and the chart
        fig.patches.append(matplotlib.patches.FancyBboxPatch(
            (0.04, 0.762), 0.92, 0.118, boxstyle="round,pad=0,rounding_size=0.01",
            transform=fig.transFigure, fc=PANEL, ec=GRID, lw=1.5))
        cols = [(0.062, "日期"), (0.300, "月收盘价"), (0.500, "月涨跌"),
                (0.680, "若在 IPO 时以发行价投入 1 万美元，现值约")]
        self.labels = []
        for n, (x, label) in enumerate(cols):
            self.labels.append(fig.text(x, 0.852, label, color=MUTED, fontsize=19, fontfamily=FONT, va="center"))
            if n:
                fig.add_artist(matplotlib.lines.Line2D([x - 0.02] * 2, [0.778, 0.866], color=GRID,
                                                       lw=1.5, transform=fig.transFigure))
        vy = 0.800
        self.t_date = fig.text(cols[0][0], vy, "", color=TEXT, fontsize=42, fontfamily=FONT_BOLD, va="center")
        self.t_price = fig.text(cols[1][0], vy, "", color=RED, fontsize=42, fontfamily=FONT_BOLD, va="center")
        self.t_chg = fig.text(cols[2][0], vy, "", color=RED, fontsize=42, fontfamily=FONT_BOLD, va="center")
        self.t_value = fig.text(cols[3][0], vy, "", color=TEXT, fontsize=42, fontfamily=FONT_BOLD, va="center")

        self.callouts = []
        for (d, title), i in zip(EVENTS, EVENT_IDX):
            y, m = MONTHS[i]
            box = ax.annotate(title, xy=(XS[i], YS[i]), xytext=(0, 0), textcoords="offset pixels",
                              ha="right", va="top", fontsize=24, color=TEXT, fontfamily=FONT_SEMI,
                              zorder=9, annotation_clip=False,
                              bbox=dict(boxstyle="round,pad=0.45", fc=PANEL, ec=RED, lw=1.6),
                              arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.6, shrinkA=0, shrinkB=7))
            date = ax.annotate(f"{y}.{m:02d}", xy=(XS[i], YS[i]), xytext=(0, 0), textcoords="offset pixels",
                               ha="right", va="top", fontsize=19, color=RED, fontfamily=FONT_SEMI,
                               zorder=9, annotation_clip=False)
            box.set_visible(False)
            date.set_visible(False)
            self.callouts.append((box, date))

    def render(self, tm, ymax):
        ax = self.ax
        p = progress(tm)
        i = int(p)
        hx, hy = head_at(p)
        k = min(int(round(p)), N - 1)

        span = max(hx - XS[0], 1.0)
        ax.set_xlim(XS[0] - span * 0.02, hx + span * 0.11)
        ax.set_ylim(0, ymax)
        lo, hi = math.ceil(XS[0] - span * 0.02), math.floor(hx + span * 0.11)
        step = 1 if hi - lo <= 11 else 2
        years = [yr for yr in range(lo, hi + 1) if yr % step == 0 and yr <= hx + 0.25]
        ax.xaxis.set_major_locator(FixedLocator(years))
        ax.set_xticklabels([str(yr) for yr in years], fontfamily=FONT)

        xv = np.append(XS[: i + 1], hx)
        yv = np.append(YS[: i + 1], hy)
        for ln in self.glow + [self.line]:
            ln.set_data(xv, yv)
        top = max(yv.max(), 1e-6)
        self.fill_im.set_extent([XS[0], hx, 0, top])
        self.fill_poly.set_xy(np.vstack([np.column_stack([xv, yv]), [[hx, 0], [XS[0], 0]]]))
        self.fill_im.set_clip_path(self.fill_poly)

        pulse = (tm * 1.6) % 1.0
        for sc in (self.head_glow, self.head_ring, self.head):
            sc.set_offsets([[hx, hy]])
        self.head_ring.set_sizes([250 + 1100 * pulse])
        self.head_ring.set_alpha(0.9 * (1 - pulse))
        self.head_label.xy = (hx, hy)
        self.head_label.set_text(fmt_price(YS[k]))

        self.t_date.set_text(month_label(k))
        self.t_date.set_fontsize(33 if k == N - 1 else 42)  # "yyyy年mm月dd日" is wider
        self.t_price.set_text(f"${YS[k]:,.2f}")
        if k == N - 1:
            self.labels[1].set_text(f"最新收盘价（{LAST_DATE.month}月{LAST_DATE.day}日）")
        elif SOURCES[k] == "interpolated":
            self.labels[1].set_text("月收盘价 · 缺失月份插值")
        else:
            self.labels[1].set_text("月收盘价")
        if k > 0:
            chg = (YS[k] / YS[k - 1] - 1) * 100
            col = RED if chg >= 0 else GREEN
            self.t_chg.set_text(f"{'▲' if chg >= 0 else '▼'} {chg:+.1f}%")
            self.t_chg.set_color(col)
            self.t_price.set_color(col)
        else:
            self.t_chg.set_text("—")
            self.t_chg.set_color(MUTED)
        self.t_value.set_text(fmt_wan(INVEST / IPO_ADJ * YS[k]))

        passed = [j for j, ei in enumerate(EVENT_IDX) if ei <= p + 1e-9]
        self.markers.set_offsets(np.array([[XS[EVENT_IDX[j]], YS[EVENT_IDX[j]]] for j in passed]).reshape(-1, 2))

        # Call-outs sit in the top band of the plot, which the line never reaches
        # (y-axis top is >= 1.25x the running max), with a leader down to the point.
        renderer = self.fig.canvas.get_renderer()
        ax_bb = ax.get_window_extent(renderer)
        for j, (box, date) in enumerate(self.callouts):
            tl = tm - EVENT_T[j]
            if tl < 0 or tm > EVENT_END[j] + CALLOUT_OUT:
                box.set_visible(False)
                date.set_visible(False)
                continue
            a = ease_out(tl / CALLOUT_IN)
            if tm > EVENT_END[j]:
                a *= 1 - (tm - EVENT_END[j]) / CALLOUT_OUT
            px, py = ax.transData.transform((XS[EVENT_IDX[j]], YS[EVENT_IDX[j]]))
            slide = (1 - ease_out(tl / CALLOUT_IN)) * 18
            width = matplotlib.text.Text.get_window_extent(box, renderer).width + 30
            if px - 30 - width > ax_bb.x0 + 10:
                ha, dx, box._arrow_relpos = "right", -30, (1, 0)
            else:
                ha, dx, box._arrow_relpos = "left", 30, (0, 0)
            box.set_ha(ha)
            date.set_ha(ha)
            box.xyann = (dx, ax_bb.y1 - 58 + slide - py)
            date.xyann = (dx, ax_bb.y1 - 4 + slide - py)
            for art in (box, date):
                art.set_visible(True)
                art.set_alpha(a)
            box.get_bbox_patch().set_alpha(0.94 * a)
            box.arrow_patch.set_alpha(a)
        return fig_to_rgb(self.fig)


def render_intro():
    fig = new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    yy, xx = np.mgrid[0:H:4, 0:W:4]
    r = np.hypot((xx - W / 2) / W, (yy - H * 0.55) / H)
    glow = np.zeros(r.shape + (4,))
    glow[..., :3] = to_rgb(RED)
    glow[..., 3] = 0.22 * np.exp(-(r / 0.28) ** 2)
    ax.imshow(glow, extent=[0, 1, 0, 1], aspect="auto", interpolation="bilinear")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.text(0.5, 0.60, "特斯拉  TSLA", color=TEXT, fontsize=120, fontfamily=FONT_BOLD, ha="center", va="center")
    fig.text(0.5, 0.44, "上市 16 年股价走势", color=TEXT, fontsize=58, fontfamily=FONT_SEMI, ha="center", va="center")
    ax.plot([0.40, 0.60], [0.365, 0.365], color=RED, lw=4, transform=ax.transAxes)
    fig.text(0.5, 0.31, f"2010.06 — {LAST_DATE:%Y.%m}   ·   ${YS[0]:.2f} → ${YS[-1]:.2f}", color=MUTED,
             fontsize=32, fontfamily=FONT, ha="center", va="center")
    img = fig_to_rgb(fig)
    plt.close(fig)
    return img


def render_log():
    fig = new_fig()
    header(fig, "换成对数坐标，再看一遍", "对数坐标下，同样的高度代表同样的涨幅倍数")
    ax = fig.add_axes([0.065, 0.09, 0.895, 0.78])
    style_axes(ax)
    ax.set_yscale("log")
    ylo, yhi = 0.8, 900
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(XS[0] - 0.3, XS[-1] + 0.9)
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.set_yticklabels([fmt_price(t) for t in ticks], fontfamily=FONT)
    years = list(range(2011, 2027))
    ax.xaxis.set_major_locator(FixedLocator(years))
    ax.set_xticklabels([str(y) for y in years], fontfamily=FONT)
    im = ax.imshow(red_gradient(0.34, 1.0), aspect="auto", extent=[XS[0], XS[-1], ylo, YS.max()],
                   origin="upper", zorder=1, interpolation="bilinear")
    poly = Polygon(np.vstack([np.column_stack([XS, YS]), [[XS[-1], ylo], [XS[0], ylo]]]),
                   closed=True, transform=ax.transData)
    im.set_clip_path(poly)
    for lw, a in ((16, 0.05), (9, 0.12)):
        ax.plot(XS, YS, color=RED, lw=lw, alpha=a, solid_capstyle="round", zorder=2)
    ax.plot(XS, YS, color=RED, lw=3.4, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.scatter(XS[EVENT_IDX], YS[EVENT_IDX], s=70, facecolor=TEXT, edgecolor=RED, linewidth=2.2, zorder=4)

    st = stats()
    marks = [
        (0, "上市首月\n$%.2f" % YS[0], (30, 90)),
        (MONTHS.index((2013, 9)), "2013 年 9 月\n一年上涨 %.0f%%" % ((YS[MONTHS.index((2013, 9))]
                                   / YS[MONTHS.index((2012, 9))] - 1) * 100), (-70, 105)),
        (MONTHS.index((2019, 5)), "2019 年 5 月\n$%.2f" % YS[MONTHS.index((2019, 5))], (40, -120)),
        (MONTHS.index((2021, 11)), "2021 年 11 月\n$%.2f" % YS[MONTHS.index((2021, 11))], (-150, 60)),
        (st["min_after_peak_i"], "2022 年 12 月\n$%.2f" % YS[st["min_after_peak_i"]], (30, -110)),
        (N - 1, f"{LAST_DATE:%Y.%m.%d}\n${YS[-1]:.2f}", (-40, -120)),
    ]
    for i, txt, off in marks:
        ax.annotate(txt, xy=(XS[i], YS[i]), xytext=off, textcoords="offset pixels",
                    ha="center", va="center", fontsize=20, color=TEXT, fontfamily=FONT_SEMI,
                    linespacing=1.4, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.45", fc=PANEL, ec=GRID, lw=1.2),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.4, shrinkA=0, shrinkB=6))
    ax.scatter([XS[-1]], [YS[-1]], s=150, color=TEXT, edgecolor=RED, linewidth=3, zorder=6)
    img = fig_to_rgb(fig)
    plt.close(fig)
    return img


def render_stats():
    st = stats()
    fig = new_fig()
    fig.text(0.5, 0.875, "特斯拉上市 16 年成绩单", color=TEXT, fontsize=56, fontfamily=FONT_BOLD,
             ha="center", va="center")
    fig.text(0.5, 0.805, f"统计区间：{IPO_DATE:%Y.%m.%d}（IPO）— {LAST_DATE:%Y.%m.%d}", color=MUTED,
             fontsize=22, fontfamily=FONT, ha="center", va="center")
    fy, fm = st["mdd_from"]
    ty, tmn = st["mdd_to"]
    tiles = [
        (f"${IPO_PRICE:.2f}", "IPO 发行价", f"拆股调整后 ${IPO_ADJ:.2f}（累计 1 拆 15）", TEXT),
        (f"${st['last']:.2f}", "最新收盘价", f"{LAST_DATE:%Y 年 %m 月 %d 日}", TEXT),
        (f"{st['multiple']:,.0f} 倍", "较发行价累计上涨", f"+{st['total_pct']:,.0f}%", RED),
        (f"{st['cagr']:.1f}%", "年化收益率", "按发行价至今复利计算", RED),
        (fmt_wan(st["value"]).replace(" 万美元", " 万"), "1 万美元 IPO 投入的现值", "单位：美元", RED),
        (f"{st['mdd']:.1f}%", "最大回撤（月收盘）", f"{fy}.{fm:02d} → {ty}.{tmn:02d}", GREEN),
    ]
    cols, rows = 3, 2
    tw, th, gx, gy = 0.27, 0.27, 0.025, 0.04
    x0 = (1 - (cols * tw + (cols - 1) * gx)) / 2
    y0 = 0.72
    for n, (big, label, sub, col) in enumerate(tiles):
        c, r = n % cols, n // cols
        x = x0 + c * (tw + gx)
        y = y0 - (r + 1) * th - r * gy
        fig.patches.append(matplotlib.patches.FancyBboxPatch(
            (x, y), tw, th, boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
            fc=PANEL, ec=GRID, lw=1.5))
        fig.text(x + 0.025, y + th - 0.055, label, color=MUTED, fontsize=24, fontfamily=FONT, va="center")
        fig.text(x + 0.025, y + th * 0.47, big, color=col, fontsize=60, fontfamily=FONT_BOLD, va="center")
        fig.text(x + 0.025, y + 0.045, sub, color=MUTED, fontsize=19, fontfamily=FONT, va="center")
    gaps = [f"{MONTHS[i][0]}.{MONTHS[i][1]:02d}" for i, s in enumerate(SOURCES) if s == "interpolated"]
    fig.text(0.5, 0.075, "数据：StatMuse / Digrin 月收盘价（已按拆股调整，未计股息——特斯拉从未派息）；"
             f"{'、'.join(gaps)} 为线性插值", color="#4d5666", fontsize=15, fontfamily=FONT, ha="center")
    fig.text(0.5, 0.045, "历史表现不代表未来收益，本视频仅供参考，不构成任何投资建议", color="#4d5666",
             fontsize=15, fontfamily=FONT, ha="center")
    img = fig_to_rgb(fig)
    plt.close(fig)
    return img


# ---------------------------------------------------------------- frame assembly

_G = {}


def blend(a, b, w):
    return (a.astype(np.float32) * (1 - w) + b.astype(np.float32) * w).astype(np.uint8)


BG_RGB = (np.array(to_rgb(BG)) * 255).astype(np.uint8)


def dip(a, b, w):
    """Transition a -> background -> b; cleaner than a crossfade between two busy charts."""
    bg = np.broadcast_to(BG_RGB, a.shape)
    return blend(a, bg, w * 2) if w < 0.5 else blend(bg, b, w * 2 - 1)


def _init_worker(fps):
    _G["fps"] = fps
    _G["ymax"] = YMAX
    _G["scene"] = MainScene()
    _G["intro"] = render_intro()
    _G["log"] = render_log()
    _G["stats"] = render_stats()
    _G["main0"] = _G["scene"].render(0.0, YMAX[0])
    _G["main_end"] = None


def frame_at(t):
    fps = _G["fps"]
    black = np.zeros((H, W, 3), np.uint8)
    if t < MAIN_START:
        img = _G["intro"]
        if t < 0.6:
            return blend(black, img, ease_out(t / 0.6))
        if t > MAIN_START - XFADE:
            return dip(img, _G["main0"], (t - (MAIN_START - XFADE)) / XFADE)
        return img
    if t < LOG_START:
        tm = t - MAIN_START
        f = min(int(round(tm * fps)), len(_G["ymax"]) - 1)
        return _G["scene"].render(tm, _G["ymax"][f])
    if _G["main_end"] is None:
        tm = LOG_START - MAIN_START
        _G["main_end"] = _G["scene"].render(tm, _G["ymax"][-1])
    if t < LOG_START + LOG_FADE:
        return dip(_G["main_end"], _G["log"], (t - LOG_START) / LOG_FADE)
    if t < STATS_START:
        return _G["log"]
    if t < STATS_START + STATS_FADE:
        return dip(_G["log"], _G["stats"], (t - STATS_START) / STATS_FADE)
    if t < TOTAL - OUT_FADE:
        return _G["stats"]
    return blend(_G["stats"], black, min((t - (TOTAL - OUT_FADE)) / OUT_FADE, 1.0))


def render_chunk(args):
    start, stop, path = args
    fps = _G["fps"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-pix_fmt", "yuv420p", "-g", str(fps * 2), path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(start, stop):
        proc.stdin.write(frame_at(f / fps).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "tsla_stock_history.mp4"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--music", action="store_true", help="mix in the synthesized soundtrack from bgm.py")
    ap.add_argument("--preview", type=float, nargs="*", help="write PNG stills at these times (s) and exit")
    args = ap.parse_args()

    global YMAX
    YMAX = smoothed_ymax(args.fps)
    print(f"{N} months, main {MAIN_DUR:.1f}s, total {TOTAL:.1f}s", file=sys.stderr)

    if args.preview is not None:
        _init_worker(args.fps)
        os.makedirs(os.path.join(ROOT, "output", "preview"), exist_ok=True)
        for t in args.preview:
            p = os.path.join(ROOT, "output", "preview", f"frame_{t:06.2f}.png")
            plt.imsave(p, frame_at(t))
            print(p)
        return

    nframes = int(round(TOTAL * args.fps))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        k = args.workers * 3
        bounds = np.linspace(0, nframes, k + 1).astype(int)
        jobs = [(bounds[j], bounds[j + 1], os.path.join(tmp, f"seg{j:03d}.mp4")) for j in range(k)]
        with Pool(args.workers, initializer=_init_worker, initargs=(args.fps,)) as pool:
            for n, path in enumerate(pool.imap(render_chunk, jobs), 1):
                print(f"  segment {n}/{k}", file=sys.stderr)
        listing = os.path.join(tmp, "list.txt")
        with open(listing, "w") as f:
            f.writelines(f"file '{p}'\n" for _, _, p in jobs)
        video = os.path.join(tmp, "video.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", listing, "-c", "copy", video], check=True)
        if args.music:
            import bgm
            wav = os.path.join(tmp, "bgm.wav")
            bgm.write(wav, TOTAL, beat_in=MAIN_START, beat_out=STATS_START + STATS_FADE)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", wav,
                            "-c:v", "copy", "-af", "loudnorm=I=-16:TP=-2:LRA=11,alimiter=limit=0.8", "-ar", "48000",
                            "-c:a", "aac", "-b:a", "192k", "-shortest",
                            "-movflags", "+faststart", args.out], check=True)
        else:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-c", "copy",
                            "-movflags", "+faststart", args.out], check=True)
    print(args.out)


if __name__ == "__main__":
    main()
