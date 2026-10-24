"""Frame composition for the vertical TSLA video."""
import math

import numpy as np
import pandas as pd
import skia

from . import chart, effects, gfx, story
from .gfx import W, H, RED, GREEN, YELLOW, WHITE, GREY, DIM, ICE, with_alpha, paint, fade, ease_out, ease_in_out
from .story import SERIES, SEGS, END

FPS = 30
WEEKDAY = "一二三四五六日"


# --------------------------------------------------------------------------- helpers

def ns(ts):
    return pd.Timestamp(ts).value


_DATES_NS = {k: s.dates.as_unit("ns").asi8 for k, s in SERIES.items()}


def date_to_idx(key, t_ns):
    d = _DATES_NS[key]
    n = len(d)
    if t_ns <= d[0]:
        step = (d[min(5, n - 1)] - d[0]) / min(5, n - 1)
        return (t_ns - d[0]) / step
    if t_ns >= d[-1]:
        step = (d[-1] - d[-6]) / 5
        return n - 1 + (t_ns - d[-1]) / step
    return float(np.interp(t_ns, d, np.arange(n)))


def idx_to_date(key, idx):
    d = _DATES_NS[key]
    n = len(d)
    if idx >= n - 1:
        return float(d[-1] + (idx - (n - 1)) * (d[-1] - d[-6]) / 5)
    if idx <= 0:
        return float(d[0])
    return float(np.interp(idx, np.arange(n), d))


def head_pos(key, date):
    return SERIES[key].index_of(date) + 0.999


def seg_at(t):
    for s in SEGS:
        if s.t0 <= t < s.t1:
            return s
    return SEGS[-1]


def seg_progress(s, t):
    x = (t - s.t0) / max(1e-6, s.t1 - s.t0)
    x = min(max(x, 0.0), 1.0)
    if s.ease == "linear":
        return x
    if s.ease == "out":
        return ease_out(x, 2)
    return ease_in_out(x)


def seg_head(s, t):
    p = seg_progress(s, t)
    a, b = head_pos(s.series, s.head0), head_pos(s.series, s.head1)
    return a + (b - a) * p


def _date_labels(key):
    s = SERIES[key]
    if key == "H":
        days = pd.Series(s.dates.normalize()).drop_duplicates()
        return [(int(i), s.dates[i].strftime("%m.%d")) for i in days.index]
    yrs = pd.Series(s.dates.year).drop_duplicates()
    return [(int(i), str(s.dates[i].year)) for i in yrs.index]


DATE_LABELS = {k: _date_labels(k) for k in SERIES}


def time_of(date):
    """First video time (after the rewind) at which the head reaches `date`."""
    for s in SEGS:
        if s.t0 < 5.0 or s.series == "H":
            continue
        a, b = head_pos(s.series, s.head0), head_pos(s.series, s.head1)
        target = head_pos(s.series, date)
        if a <= target <= b and b > a:
            lo, hi = s.t0, s.t1
            for _ in range(40):
                mid = (lo + hi) / 2
                if seg_head(s, mid) < target:
                    lo = mid
                else:
                    hi = mid
            return hi
    raise ValueError(date)


FX = [(time_of(tf) if isinstance(tf, str) else tf, kind, target, extra) for tf, kind, target, extra in story.FX]


# --------------------------------------------------------------------------- camera track

def camera_track(fps=FPS):
    """Sequentially smoothed view window / y-range / camera angles for every frame."""
    n = int(math.ceil(END * fps)) + 1
    out = []
    prev = None
    k = 1 - math.exp(-1 / (fps * 0.32))
    for f in range(n):
        t = f / fps
        s = seg_at(t)
        S = SERIES[s.series]
        head = seg_head(s, t)
        hd = idx_to_date(s.series, head)
        left = ns(s.left) if s.left else float(_DATES_NS[s.series][0])
        span = max(hd - left, s.min_days * 86400e9)
        right = left + span * (1 + s.pad) + s.future_days * 86400e9
        i0 = max(0, int(date_to_idx(s.series, left)))
        k_head, ho, hh, hl, hc, _ = S.bar_at(head)
        lo = min(S.l[i0:k_head].min() if k_head > i0 else hl, hl)
        hi = max(S.h[i0:k_head].max() if k_head > i0 else hh, hh)
        if s.log:
            llo, lhi = math.log(lo), math.log(hi)
            sp = lhi - llo
            y0, y1 = math.exp(llo - 0.06 * sp), math.exp(lhi + (s.ytop - 1 + 0.08) * sp)
        else:
            sp = max(hi - lo, hi * 0.02)
            y0, y1 = max(0.0, lo - 0.08 * sp), hi + (s.ytop - 1) * sp * 1.6 + 0.04 * sp
        p = seg_progress(s, t)
        yaw = s.yaw[0] + (s.yaw[1] - s.yaw[0]) * p
        pitch = s.pitch[0] + (s.pitch[1] - s.pitch[0]) * p
        target = np.array([left, right, y0, y1, yaw, pitch, s.cy, s.ph])
        snap = prev is None or (s.cut and f > 0 and seg_at((f - 1) / fps) is not s) or prev[8] != s.log
        if snap:
            cur = target.copy()
        else:
            cur = prev[:8] + (target - prev[:8]) * k
            # never let the head leave the plot
            if not s.log:
                cur[3] = max(cur[3], hh * 1.04)
                cur[2] = min(cur[2], hl * 0.98)
            cur[1] = max(cur[1], hd + 0.02 * (hd - cur[0]))
        prev = np.append(cur, s.log)  # index 8 = log flag
        out.append((cur.copy(), s.log))
    return out


# --------------------------------------------------------------------------- HUD

def date_string(key, k):
    d = SERIES[key].dates[k]
    return d.strftime("%Y.%m.%d"), (d.strftime("%H:%M") if key == "H" else None)


def draw_rolling(canvas, s_now, s_prev, r, x, y, fnt, color, alpha=1.0, glow=0.0, blur_pos=()):
    """Draw s_now; characters that differ from s_prev roll up with progress r in [0, 1].
    Positions in blur_pos are changing every frame and get a vertical motion-blur ghost instead."""
    m = fnt.getMetrics()
    hgt = -m.fAscent + m.fDescent
    for i, ch in enumerate(s_now):
        w = gfx.text_width(ch, fnt)
        old = s_prev[i] if s_prev and i < len(s_prev) else ch
        if i in blur_pos:
            for off, aa in ((-0.32, 0.18), (0.32, 0.18), (-0.16, 0.35), (0.16, 0.35)):
                gfx.text(canvas, ch, x, y + off * hgt, fnt, color, alpha=alpha * aa)
            gfx.text(canvas, ch, x, y, fnt, color, alpha=alpha * 0.75)
        elif old != ch and r < 1.0:
            canvas.save()
            canvas.clipRect(skia.Rect.MakeLTRB(x - 2, y + m.fAscent - 4, x + w + 2, y + m.fDescent + 4))
            e = ease_out(r, 2)
            gfx.text(canvas, old, x, y - e * hgt, fnt, color, alpha=alpha * (1 - e))
            gfx.text(canvas, ch, x, y + (1 - e) * hgt, fnt, color, alpha=alpha * e, glow=glow)
            canvas.restore()
        else:
            gfx.text(canvas, ch, x, y, fnt, color, alpha=alpha, glow=glow)
        x += w
    return x


def badge_at(t):
    b = story.BADGES[0]
    for bb in story.BADGES:
        if bb.t <= t:
            b = bb
    return b


def draw_hud(canvas, st, alpha=1.0):
    t = st["t"]
    key = st["series"]
    price = st["price"]
    b = badge_at(t)
    chg = (price / b.base - 1) * 100
    col = RED if chg >= 0 else GREEN
    x0 = 110
    gfx.text(canvas, "特斯拉 TSLA", x0, 352, gfx.font("cjk_bold", 30), WHITE, alpha=0.85 * alpha)
    gfx.text(canvas, "数据：Yahoo Finance · 拆股调整", x0 + 182, 350, gfx.font("cjk_med", 19), GREY,
             alpha=0.8 * alpha)
    tf = SERIES[key].label
    tf_font = gfx.font("cjk_bold", 22)
    tw = gfx.text_width(tf, tf_font) + 22
    tf_col = YELLOW if key == "H" else gfx.rgb("#c9d1dc")
    gfx.rounded(canvas, 968 - tw, 324, tw, 36, 6, fill=with_alpha(tf_col, 0.08), stroke=with_alpha(tf_col, 0.75),
                stroke_w=2, alpha=alpha)
    gfx.text(canvas, tf, 968 - tw + 11, 350, tf_font, tf_col, alpha=alpha)

    big = gfx.font("num_xb", 124)
    pw = gfx.text(canvas, f"{price:,.2f}", x0 - 4, 482, big, col, glow=16, glow_alpha=0.45, alpha=alpha)
    bx = x0 + pw + 18
    gfx.text(canvas, b.label, bx, 412, gfx.font("cjk_med", 22), GREY, alpha=alpha)
    arrow = "▲" if chg >= 0 else "▼"
    s = f"{arrow} {gfx.fmt_pct(chg, 2) if abs(chg) < 1000 else f'{chg:+,.0f}%'.replace('-', '−')}"
    gfx.pill(canvas, s, bx, 424, gfx.font("cjk_bold", 30), col, with_alpha(col, 0.16), pad_x=12, h=46, alpha=alpha)

    dfont = gfx.font("num_b", 46)
    xe = draw_rolling(canvas, st["date"], st["date_prev"], st["date_roll"], x0, 544, dfont, WHITE, alpha=alpha,
                      blur_pos=st["date_blur"])
    if st["time"]:
        draw_rolling(canvas, st["time"], st["time_prev"], st["date_roll"], xe + 22, 544, dfont, YELLOW, alpha=alpha,
                     blur_pos=st["time_blur"])
    if st.get("hour_pill"):
        gfx.pill(canvas, st["hour_pill"], xe + 160, 512, gfx.font("cjk_bold", 22), GREEN, with_alpha(GREEN, 0.14),
                 pad_x=10, h=38, alpha=alpha)


KIND = {
    "up": (RED, WHITE, WHITE),
    "down": (gfx.rgb("#13c98b"), WHITE, ICE),
    "gold": (YELLOW, gfx.rgb("#1b1300"), YELLOW),
    "flat": (gfx.rgb("#3b4558"), WHITE, WHITE),
    "info": (gfx.rgb("#2f6fd8"), WHITE, WHITE),
}
DESC_COL = {"w": gfx.rgb("#dfe4ec"), "r": RED, "g": GREEN, "y": YELLOW, "grey": GREY}


def draw_panel(canvas, p, t):
    a = fade(t, p.t0, p.t1, 0.22, 0.18)
    if a <= 0:
        return
    slide = (1 - ease_out(min(1.0, (t - p.t0) / 0.3))) * -40
    acc, tag_fg, title_col = KIND[p.kind]
    x = 104 + slide
    y = 578
    has_desc = bool(p.desc)
    hgt = 172 if has_desc else 120
    canvas.drawRect(skia.Rect.MakeXYWH(x, y, 5, hgt), paint(with_alpha(acc, a)))
    tw, th = gfx.pill(canvas, p.tag, x + 20, y + 2, gfx.font("cjk_bold", 24), tag_fg, acc, pad_x=10, h=38, alpha=a)
    gfx.text(canvas, p.dates, x + 20 + tw + 14, y + 31, gfx.font("cjk_med", 26), gfx.rgb("#c4cad4"), alpha=a)
    if p.big_value:
        gfx.text(canvas, p.big_value, x + 18, y + 116, gfx.font("num_xb", 92), YELLOW, glow=14, alpha=a)
    else:
        gfx.text(canvas, p.title, x + 16, y + 112, gfx.font("cjk_black", 78), title_col, glow=12, glow_alpha=0.4,
                 alpha=a)
    xx = x + 20
    for s, ck in p.desc:
        xx += gfx.text(canvas, s, xx, y + 160, gfx.font("cjk_med", 29), DESC_COL[ck], alpha=a)


def draw_chip(canvas, xy, text_, kind, alpha, above=True, dy=None):
    if alpha <= 0:
        return
    x, y = xy
    acc = {"up": RED, "down": GREEN, "flat": gfx.rgb("#9fb3d1"), "gold": YELLOW}[kind]
    f = gfx.font("cjk_bold", 24)
    w = gfx.text_width(text_, f) + 22
    h = 38
    dy = dy if dy is not None else (-78 if above else 62)
    above = dy < 0
    bx = min(max(x - w / 2, 20), W - w - 20)
    by = y + dy - (h if above else 0)
    canvas.drawLine(x, y, x, by + (h if above else 0), paint(with_alpha(acc, 0.6 * alpha), "stroke", 1.6))
    canvas.drawCircle(x, y, 6, paint(with_alpha(acc, alpha)))
    gfx.rounded(canvas, bx, by, w, h, 6, fill=gfx.rgb("#0b0f16"), stroke=with_alpha(acc, 0.9), stroke_w=2, alpha=alpha)
    gfx.text(canvas, text_, bx + 11, by + 28, f, acc if kind != "flat" else WHITE, alpha=alpha)


# --------------------------------------------------------------------------- special overlays

def big_line(canvas, parts, y, fnt, align="right", x=968, alpha=1.0, glow=12):
    arrow_font = gfx.font("cjk_bold", fnt.getSize() * 0.9)
    fonts = [arrow_font if "→" in s else fnt for s, _ in parts]
    total = sum(gfx.text_width(s, f) for (s, _), f in zip(parts, fonts))
    xx = x - total if align == "right" else (x - total / 2 if align == "center" else x)
    for (s, c), f in zip(parts, fonts):
        xx += gfx.text(canvas, s, xx, y, f, c, glow=glow, glow_alpha=0.45, alpha=alpha)


def overlay_cold_open(canvas, t):
    a = fade(t, 0.35, 3.75, 0.3, 0.3)
    if a > 0:
        d = pd.Timestamp(story.CRASH_DAY)
        gfx.text(canvas, f"{d:%Y.%m.%d} 周{WEEKDAY[d.weekday()]}", 968, 626, gfx.font("cjk_bold", 30),
                 gfx.rgb("#c9d1dc"), align="right", alpha=a)
        pop = 1 + 0.25 * (1 - ease_out(min(1, (t - 0.35) / 0.35)))
        canvas.save()
        canvas.translate(968, 735)
        canvas.scale(pop, pop)
        gfx.text(canvas, "黑色星期四", 0, 0, gfx.font("cjk_black", 118), WHITE, align="right", glow=18, alpha=a)
        canvas.restore()
        gfx.text(canvas, "Q2 财报暴雷", 968, 805, gfx.font("cjk_black", 50), GREEN, align="right", glow=10, alpha=a)
        drop = (story.close(story.CRASH_DAY) / story.PRE_CRASH - 1) * 100
        gfx.text(canvas, gfx.fmt_pct(drop), 968, 950, gfx.font("num_xb", 160), GREEN, align="right", glow=22, alpha=a)
        big_line(canvas, [(f"{story.PRE_CRASH:.2f}", WHITE), ("  →  ", GREY),
                          (f"{story.close(story.CRASH_DAY):.2f}", GREEN)], 1010, gfx.font("num_b", 44), alpha=a,
                 glow=0)
    a2 = fade(t, 1.9, 3.75, 0.35, 0.3)
    if a2 > 0:
        gfx.text(canvas, "特斯拉上市 16 年", W / 2, 1420, gfx.font("cjk_black", 76), YELLOW, align="center", glow=18,
                 alpha=a2)
        gfx.text(canvas, f"2010.06.29 — {story.LAST_DATE:%Y.%m.%d}", W / 2, 1478, gfx.font("cjk_bold", 32),
                 gfx.rgb("#c9d1dc"), align="center", alpha=a2)


def overlay_rewind(canvas, t):
    a = fade(t, 3.7, 5.25, 0.15, 0.35)
    if a > 0:
        effects.rewind_icon(canvas, W / 2 + 40, 860, 190, alpha=a)
        gfx.text(canvas, "回到 2010.06.29", W / 2, 1010, gfx.font("cjk_black", 60), gfx.rgb("#e6edf7"),
                 align="center", glow=10, alpha=a)
        gfx.text(canvas, "上市首日", W / 2, 1064, gfx.font("cjk_bold", 30), GREY, align="center", alpha=a)


def overlay_covid_counter(canvas, t, st):
    a = fade(t, 20.15, 24.8, 0.2, 0.25)
    if a <= 0 or st["series"] != "D":
        return
    S = SERIES["D"]
    k = st["k"]
    k0 = S.index_of("2020-02-19")
    n = max(1, k - k0)
    cum = (st["price"] / S.c[k0] - 1) * 100
    day = (st["price"] / S.c[k - 1] - 1) * 100
    gfx.pill(canvas, "闪崩", 110, 820, gfx.font("cjk_bold", 24), WHITE, gfx.rgb("#13c98b"), pad_x=10, h=38, alpha=a)
    gfx.text(canvas, f"第 {n} 天", 110, 925, gfx.font("cjk_black", 76), WHITE, glow=10, alpha=a)
    gfx.text(canvas, gfx.fmt_pct(cum), 106, 1020, gfx.font("num_xb", 96), GREEN if cum < 0 else RED, glow=16,
             alpha=a)
    big_line(canvas, [("当日 ", GREY), (gfx.fmt_pct(day), GREEN if day < 0 else RED)], 1064,
             gfx.font("cjk_bold", 28), align="left", x=110, alpha=a, glow=0)


def overlay_earnings_card(canvas, t):
    a = fade(t, 58.0, 60.15, 0.3, 0.15)
    if a <= 0:
        return
    x, y, w, h = 110, 790, 860, 330
    gfx.rounded(canvas, x, y, w, h, 14, fill=gfx.rgb("#0c1018", 0.92), stroke=with_alpha(YELLOW, 0.6), stroke_w=2,
                alpha=a)
    gfx.text(canvas, "Q2 2026 财报 · 7 月 22 日盘后", x + 28, y + 52, gfx.font("cjk_bold", 30), YELLOW, alpha=a)
    rows = [("营收", "282.4 亿美元", "略超预期", WHITE),
            ("每股收益", "$0.33", "预期 $0.53", GREEN),
            ("营业利润率", "1.4%", "营业利润 −57%", GREEN),
            ("自由现金流", "−11 亿美元", "资本开支 +142%", GREEN)]
    for i, (k_, v, note, c) in enumerate(rows):
        yy = y + 112 + i * 56
        gfx.text(canvas, k_, x + 28, yy, gfx.font("cjk_med", 28), GREY, alpha=a)
        gfx.text(canvas, v, x + 250, yy, gfx.font("cjk_black", 34), c, alpha=a)
        gfx.text(canvas, note, x + w - 28, yy, gfx.font("cjk_med", 24), GREY, align="right", alpha=a)


def overlay_crash(canvas, t, st):
    pre = story.PRE_CRASH
    cum = (st["price"] / pre - 1) * 100
    a = fade(t, 60.05, 62.9, 0.12, 0.2)
    if a > 0:
        big_line(canvas, [("一觉醒来，", WHITE), ("天塌了", GREEN)], 690, gfx.font("cjk_black", 74), alpha=a)
        gap = (story.GAP_OPEN / pre - 1) * 100
        gfx.text(canvas, f"开盘跳空 {gfx.fmt_pct(gap)}", 968, 742, gfx.font("cjk_bold", 30), gfx.rgb("#c9d1dc"),
                 align="right", alpha=a)
    a2 = fade(t, 62.9, 65.7, 0.2, 0.2)
    if a2 > 0:
        gfx.text(canvas, "黑色星期四", 968, 690, gfx.font("cjk_black", 80), WHITE, align="right", glow=12, alpha=a2)
        d = (story.close(story.CRASH_DAY) / pre - 1) * 100
        big_line(canvas, [("单日 ", WHITE), (gfx.fmt_pct(d), GREEN), (f" · 上市以来第 {story.CRASH_RANK} 大跌幅", WHITE)],
                 742, gfx.font("cjk_bold", 30), alpha=a2, glow=0)
        overlay_tiles(canvas, t, a2)
    a3 = fade(t, 65.7, 67.45, 0.2, 0.2)
    if a3 > 0:
        gfx.text(canvas, "多杀多", 968, 690, gfx.font("cjk_black", 80), WHITE, align="right", glow=12, alpha=a3)
        gfx.text(canvas, "抛售接力，连跌 5 个交易日，跌破 300 美元", 968, 742, gfx.font("cjk_bold", 30),
                 gfx.rgb("#c9d1dc"), align="right", alpha=a3)
    a_big = fade(t, 60.05, 67.45, 0.12, 0.2) * (0.0 if 62.9 <= t < 65.7 else 1.0)
    if a_big > 0:
        gfx.text(canvas, gfx.fmt_pct(cum), 968, 900, gfx.font("num_xb", 150), GREEN, align="right", glow=22,
                 alpha=a_big)
        gfx.text(canvas, f"较财报前收盘 {pre:.2f}", 968, 948, gfx.font("cjk_med", 26), GREY, align="right", alpha=a_big)
    a4 = fade(t, 67.45, 70.3, 0.3, 0.3)
    if a4 > 0:
        gfx.text(canvas, "8 个月的涨幅，", W / 2, 700, gfx.font("cjk_black", 84), WHITE, align="center", glow=12,
                 alpha=a4)
        retr = (story.ATH3 - story.CRASH_LOW) / (story.ATH3 - story.close("2025-04-08")) * 100
        gfx.text(canvas, f"回吐 {retr:.0f}%。", W / 2, 800, gfx.font("cjk_black", 84), GREEN, align="center",
                 glow=14, alpha=a4)
        big_line(canvas, [(f"{story.ATH3:.2f}", WHITE), ("  →  ", GREY), (f"{story.CRASH_LOW:.2f}", GREEN),
                          (f"   较历史最高 {gfx.fmt_pct((story.CRASH_LOW / story.ATH3 - 1) * 100)}", GREY)],
                 860, gfx.font("cjk_bold", 30), align="center", x=W / 2, alpha=a4, glow=0)


def overlay_tiles(canvas, t, a):
    x0, y0, cw, ch, gap = 110, 800, 276, 112, 16
    for i, (d, v) in enumerate(story.TOP_DROPS):
        r, c = divmod(i, 3)
        appear = ease_out(min(1, max(0, (t - 63.0 - i * 0.07) / 0.25)))
        if appear <= 0:
            continue
        x = x0 + c * (cw + gap)
        y = y0 + r * (ch + gap) + (1 - appear) * 30
        hl = d == pd.Timestamp(story.CRASH_DAY).strftime("%Y.%m.%d")
        col = GREEN
        gfx.rounded(canvas, x, y, cw, ch, 10, fill=gfx.rgb("#071410", 0.92), alpha=a * appear)
        gfx.rounded(canvas, x, y, cw, ch, 10, fill=with_alpha(col, 0.22 if hl else 0.07),
                    stroke=with_alpha(YELLOW if hl else col, 0.95 if hl else 0.35), stroke_w=3 if hl else 1.5,
                    alpha=a * appear)
        gfx.text(canvas, f"#{i + 1}  {d}", x + 16, y + 38, gfx.font("num_sb", 26), YELLOW if hl else GREY,
                 alpha=a * appear)
        gfx.text(canvas, gfx.fmt_pct(v), x + 16, y + 92, gfx.font("num_xb", 50), GREEN, glow=8 if hl else 0,
                 alpha=a * appear)


RECAP_CHIPS = [  # date, label, kind, vertical offset of the label (px; negative = above)
    ("2010-06-29", "上市", "gold", 70),
    ("2013-09-30", "Model S 引爆", "up", -80),
    ("2016-02-10", "六年原地踏步", "flat", 80),
    ("2020-03-18", "疫情闪崩", "down", 80),
    ("2021-11-04", "史诗级暴涨", "up", -80),
    ("2023-01-03", "腰斩再腰斩", "down", 80),
    ("2025-12-16", "历史新高", "gold", -80),
    ("2026-07-29", "黑色星期四", "down", 80),
]


def overlay_recap(canvas, t):
    a = fade(t, 76.1, 84.3, 0.35, 0.3)
    if a <= 0:
        return
    gfx.text(canvas, "特斯拉上市 16 年", W / 2, 640, gfx.font("cjk_black", 50), YELLOW, align="center", glow=10,
             alpha=a)
    tot = (story.LAST / story.IPO_ADJ - 1) * 100
    gfx.text(canvas, f"+{tot:,.0f}%", W / 2, 790, gfx.font("num_xb", 150), RED, align="center", glow=22, alpha=a)
    big_line(canvas, [(f"${story.IPO_ADJ:.2f}", WHITE), ("  →  ", GREY), (f"${story.LAST:.2f}", RED),
                      (f"   ×{story.LAST / story.IPO_ADJ:.0f}", YELLOW)],
             852, gfx.font("cjk_bold", 34), align="center", x=W / 2, alpha=a, glow=0)
    gfx.text(canvas, "对数坐标 · 周K", W / 2, 900, gfx.font("cjk_med", 24), GREY, align="center", alpha=0.8 * a)


def overlay_forecast_text(canvas, t):
    a = fade(t, 85.0, 94.5, 0.5, 0.01)
    if a > 0:
        cn = "零一二三四五六七八九十"
        gfx.text(canvas, f"上市 16 年，涨了 {story.LAST / story.IPO_ADJ:.0f} 倍，", W / 2, 690,
                 gfx.font("cjk_black", 54), WHITE, align="center", glow=8, alpha=fade(t, 85.0, 94.5, 0.5, 0.01))
        gfx.text(canvas, f"也经历过{cn[story.HALVINGS]}次腰斩。", W / 2, 770, gfx.font("cjk_black", 54), GREEN,
                 align="center", glow=8, alpha=fade(t, 85.7, 94.5, 0.5, 0.01))


def draw_forecast(canvas, cam, view, st, t):
    a = fade(t, 84.6, 94.6, 0.6, 0.01)
    if a <= 0:
        return
    S = SERIES["W"]
    k = len(S) - 1
    x_now = k
    steps = 30
    grow = ease_in_out((t - 84.8) / 3.0)
    ux0 = float(view.u(x_now))
    ux1 = 1.0
    # future region
    xs, ys, _ = cam.project(np.array([ux0, ux1, ux1, ux0]), np.array([0.0, 0.0, 1.0, 1.0]))
    pth = skia.Path()
    chart._quad(pth, list(zip(xs, ys)))
    canvas.drawPath(pth, paint(with_alpha(gfx.rgb("#5c6b85"), 0.10 * a)))
    dash = paint(with_alpha(WHITE, 0.5 * a), "stroke", 2)
    dash.setPathEffect(skia.DashPathEffect.Make([8, 8], 0))
    canvas.drawLine(xs[0], ys[0], xs[3], ys[3], dash)
    gfx.text(canvas, f"今天 {story.LAST_DATE:%Y.%m.%d}", xs[0] - 12, ys[0] - 18, gfx.font("cjk_med", 24), GREY,
             align="right", alpha=a)
    gfx.text(canvas, "后续走势？", min(xs[3] + 16, W - 40 - gfx.text_width("后续走势？", gfx.font("cjk_black", 48))),
             ys[3] + 66, gfx.font("cjk_black", 48), gfx.rgb("#9aa6b8"), alpha=0.9 * a)
    walks = effects.random_walks(36, steps, 4, vol=0.045)
    hi = effects.random_walks(1, steps, 17, vol=0.04, drift=0.012)[0]
    lo = effects.random_walks(1, steps, 23, vol=0.04, drift=-0.012)[0]
    n_show = max(2, int(steps * grow))
    uu = np.linspace(ux0, ux0 + (ux1 - ux0) * 0.96, steps)
    for wk in walks:
        px, py, _ = cam.project(uu[:n_show], view.v(story.LAST * wk[:n_show]))
        p = skia.Path()
        p.moveTo(px[0], py[0])
        for xx, yy in zip(px[1:], py[1:]):
            p.lineTo(xx, yy)
        canvas.drawPath(p, paint(with_alpha(gfx.rgb("#aab6c8"), 0.10 * a), "stroke", 1.4))
    for path_, col in ((hi, RED), (lo, GREEN)):
        px, py, _ = cam.project(uu[:n_show], view.v(story.LAST * path_[:n_show]))
        p = skia.Path()
        p.moveTo(px[0], py[0])
        for xx, yy in zip(px[1:], py[1:]):
            p.lineTo(xx, yy)
        canvas.drawPath(p, paint(with_alpha(col, 0.35 * a), "stroke", 8, blur=6))
        canvas.drawPath(p, paint(with_alpha(col, 0.95 * a), "stroke", 3))


# --------------------------------------------------------------------------- frame

class Renderer:
    def __init__(self, fps=FPS):
        self.fps = fps
        self.track = camera_track(fps)
        info = skia.ImageInfo.Make(W, H, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
        self.surf = skia.Surface.MakeRaster(info)
        self.layer = skia.Surface.MakeRaster(info)
        self._states = {}

    def state(self, f):
        if f in self._states:
            return self._states[f]
        t = f / self.fps
        s = seg_at(t)
        S = SERIES[s.series]
        head = seg_head(s, t)
        k, o, hh, hl, hc, frac = S.bar_at(head)
        date, time_ = date_string(s.series, k)
        st = {"t": t, "seg": s, "series": s.series, "head": head, "k": k, "price": hc, "date": date, "time": time_}
        # rolling-digit bookkeeping over the last few frames
        hist = [(date, time_)]
        for back in range(1, 6):
            tb = (f - back) / self.fps
            if f - back < 0 or seg_at(tb).series != s.series:
                break
            sb = seg_at(tb)
            hist.append(date_string(sb.series, SERIES[sb.series].bar_at(seg_head(sb, tb))[0]))
        st["date_prev"], st["time_prev"], st["date_roll"] = date, time_, 1.0
        st["date_blur"], st["time_blur"] = set(), set()
        changes = [i for i in range(1, len(hist)) if hist[i] != hist[i - 1]]
        if len([i for i in changes if i <= 4]) >= 2:
            for a_, b_ in zip(hist, hist[1:5]):
                st["date_blur"] |= {i for i, (x, y) in enumerate(zip(a_[0], b_[0])) if x != y}
                if a_[1] and b_[1]:
                    st["time_blur"] |= {i for i, (x, y) in enumerate(zip(a_[1], b_[1])) if x != y}
        elif changes:
            back = changes[0]
            st["date_prev"], st["time_prev"] = hist[back]
            st["date_roll"] = (back - 0.5) / 4.0
        if s.series == "H" and t >= 57.6:
            i0 = SERIES["H"].index_of("2026-07-23 09:30")
            if k >= i0:
                st["hour_pill"] = f"财报后 · 第 {k - i0 + 1} 小时"
        self._states[f] = st
        return st

    def view_cam(self, f, st):
        (left, right, y0, y1, yaw, pitch, cy, ph), log = self.track[min(f, len(self.track) - 1)]
        key = st["series"]
        view = chart.View(date_to_idx(key, left) - 0.5, date_to_idx(key, right), y0, y1, log)
        cam = chart.Camera(yaw=yaw, pitch=pitch, cy=cy, ph=ph)
        return view, cam

    def target_xy(self, target, st, view, cam):
        key = st["series"]
        if target == "head":
            S = SERIES[key]
            u = float(view.u(st["head"] - 0.5 if st["head"] % 1 else st["head"]))
            k = st["k"]
            u = float(view.u(k))
            v = float(view.v(st["price"]))
        else:
            S = SERIES[key]
            k = S.index_of(target)
            u, v = float(view.u(k)), float(view.v(story.close(target) if key != "H" else S.c[k]))
        x, y, _ = cam.project(np.array([u]), np.array([v]))
        return float(x[0]), float(y[0])

    def render(self, f):
        st = self.state(f)
        t = st["t"]
        s = st["seg"]
        key = st["series"]
        S = SERIES[key]
        view, cam = self.view_cam(f, st)

        # camera shake
        sx = sy = 0.0
        for (tf, kind, target, extra) in FX:
            if kind == "shake":
                dx, dy = effects.shake_offset(t - tf, amp=extra)
                sx += dx
                sy += dy

        cv = self.surf.getCanvas()
        cv.clear(gfx.BG.toColor())
        glow_bg = skia.GradientShader.MakeRadial((W * 0.62, H * 0.62), H * 0.55,
                                                 [skia.Color4f(0.10, 0.12, 0.16, 0.55).toColor(),
                                                  skia.Color4f(0, 0, 0, 0).toColor()], [0, 1])
        cv.drawRect(skia.Rect.MakeWH(W, H), skia.Paint(Shader=glow_bg))
        effects.star_field(cv, t, alpha=0.9, snow=1.0 if 54.4 <= t < 57.6 or t >= 84.2 else 0.0)

        # ---- chart layer (glowing) ----
        lc = self.layer.getCanvas()
        lc.clear(skia.ColorTRANSPARENT)
        lc.save()
        lc.translate(sx, sy)
        style = chart.ChartStyle(alpha=1.0, labels=1.0)
        info = chart.draw_chart(lc, cam, view, S, st["head"], style, date_labels=DATE_LABELS[key])
        for b in story.BOXES:
            a = fade(t, b.t0, b.t1, 0.3, 0.2)
            if a > 0 and key != "H":
                i_s = S.index_of(b.start)
                p_s = story.close(b.start)
                pct_ = (st["price"] / p_s - 1) * 100
                dd = (pd.Timestamp(S.dates[st["k"]]) - pd.Timestamp(b.start)).days
                chart.draw_range_box(lc, cam, view, i_s, p_s, st["k"], st["price"], alpha=a, label=(pct_, dd))
        if key == "H" and 60.0 <= t < 70.2:
            i_s = S.index_of("2026-07-22 15:30")
            chart.draw_range_box(lc, cam, view, i_s, story.PRE_CRASH, st["k"], st["price"],
                                 alpha=fade(t, 60.0, 70.2, 0.3, 0.3))
        hd_date = pd.Timestamp(int(idx_to_date(key, st["head"])))
        if key in ("D", "W") and 5.2 <= t < 75.8:
            for ch in story.CHIPS:
                cd = pd.Timestamp(ch.date)
                if cd <= hd_date and t < ch.t_end:
                    i_c = S.index_of(ch.date)
                    if view.u(i_c) < -0.02:
                        continue
                    # fade in as the head passes, fade out at the chapter end
                    age = (hd_date - cd).days
                    a = min(1.0, age / 25.0 + 0.2) * fade(t, 0, ch.t_end, 0, 0.3)
                    xy = cam.project(np.array([float(view.u(i_c))]), np.array([float(view.v(story.close(ch.date)))]))
                    draw_chip(lc, (float(xy[0][0]), float(xy[1][0])), ch.text, ch.kind, a, ch.above)
        if 75.8 <= t < 84.2:
            for n_, (d, txt, kind, dy) in enumerate(RECAP_CHIPS):
                a = fade(t, 76.4 + n_ * 0.36, 84.2, 0.25, 0.3)
                i_c = S.index_of(d)
                xy = cam.project(np.array([float(view.u(i_c))]), np.array([float(view.v(S.c[i_c]))]))
                draw_chip(lc, (float(xy[0][0]), float(xy[1][0])), txt, kind, a, dy=dy)
        if t >= 84.2:
            draw_forecast(lc, cam, view, st, t)
        if info:
            strong = 1.0 if (key == "H" and t > 60) or s in SEGS[7:8] else 0.45
            effects.sparkles(lc, info["head_xy"], t, RED if info["head_up"] else GREEN, n=46, alpha=strong)
            chart.draw_head(lc, info["head_xy"], info["head_up"], t)
        lc.restore()
        img = self.layer.makeImageSnapshot()
        glow = skia.Paint(ImageFilter=skia.ImageFilters.Blur(13, 13), BlendMode=skia.BlendMode.kPlus, Alphaf=0.75)
        cv.drawImage(img, 0, 0, skia.SamplingOptions(), glow)
        cv.drawImage(img, 0, 0)

        # ---- effects ----
        cv.save()
        cv.translate(sx, sy)
        for (tf, kind, target, extra) in FX:
            lt = t - tf
            if lt < 0 or lt > 2.2:
                continue
            if kind == "lightning":
                effects.lightning(cv, self.target_xy(target, st, view, cam), lt, seed=extra,
                                  color=gfx.rgb("#bff7e4") if t > 59 or t < 4 else gfx.ICE)
            elif kind == "shatter":
                effects.shatter(cv, self.target_xy(target, st, view, cam), lt, color=GREEN)
            elif kind == "fireworks":
                effects.fireworks(cv, self.target_xy(target, st, view, cam), lt, spread=extra)
        cv.restore()

        # ---- overlays ----
        hud_a = fade(t, 0.0, END, 0.25, 0.8)
        draw_hud(cv, st, alpha=hud_a)
        for p in story.PANELS:
            draw_panel(cv, p, t)
        overlay_cold_open(cv, t)
        overlay_rewind(cv, t)
        overlay_covid_counter(cv, t, st)
        overlay_earnings_card(cv, t)
        if 60.0 <= t < 70.4:
            overlay_crash(cv, t, st)
        overlay_recap(cv, t)
        overlay_forecast_text(cv, t)
        for (tf, kind, target, extra) in FX:
            if kind == "flash":
                effects.flash(cv, t - tf, peak=extra)
        effects.vignette(cv, 0.7)
        gfx.text(cv, "TSLA · 2010—2026", W - 40, H - 46, gfx.font("cjk_med", 22), DIM, align="right")

        arr = self.surf.toarray()[:, :, :3]
        # rewind glitch
        if 3.7 <= t < 5.3:
            g = math.sin(min(1, (t - 3.7) / 1.6) * math.pi)
            arr = effects.rgb_split(effects.scan_glitch(arr, t, g), 10 * g)
        # global fade in / out
        fo = min(1.0, t / 0.25, max(0.0, (END - t) / 1.5))
        if fo < 1:
            arr = (arr.astype(np.float32) * fo).astype(np.uint8)
        return arr
