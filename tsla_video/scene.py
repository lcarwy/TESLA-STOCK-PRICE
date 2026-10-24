"""Frame composition for the vertical TSLA video."""
import math

import numpy as np
import pandas as pd
import skia

from . import chart, effects, gfx, story
from .gfx import W, H, RED, GREEN, YELLOW, WHITE, GREY, DIM, ICE, with_alpha, paint, fade, ease_out, ease_in_out
from .story import SERIES, SEGS, END, M

FPS = 30
WEEKDAY = "一二三四五六日"
SOFT = gfx.rgb("#c9d1dc")


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


_SEG_T0 = np.array([s.t0 for s in SEGS])


def seg_at(t):
    i = int(np.searchsorted(_SEG_T0, t, side="right") - 1)
    return SEGS[min(max(i, 0), len(SEGS) - 1)]


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
        if s.t0 < M["story"] or s.series == "H":
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
_SNOW = [(p.t0, p.t1) for p in story.PANELS if p.title == "高位阴跌"] + [(M["outro"], END)]


# --------------------------------------------------------------------------- camera track

def camera_track(fps=FPS):
    """Sequentially smoothed view window / y-range / camera / opacity for every frame."""
    n = int(math.ceil(END * fps)) + 1
    out = []
    prev = None
    k = 1 - math.exp(-1 / (fps * 0.32))
    kd = 1 - math.exp(-1 / (fps * 0.45))
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
        target = np.array([left, right, y0, y1, yaw, pitch, s.cy, s.ph, s.dim])
        snap = prev is None or (s.cut and f > 0 and seg_at((f - 1) / fps) is not s) or prev[9] != s.log
        if snap:
            cur = target.copy()
        else:
            cur = prev[:9] + (target - prev[:9]) * k
            cur[8] = prev[8] + (target[8] - prev[8]) * kd
            if not s.log:
                cur[3] = max(cur[3], hh * 1.04)
                cur[2] = min(cur[2], hl * 0.98)
            cur[1] = max(cur[1], hd + 0.02 * (hd - cur[0]))
        prev = np.append(cur, s.log)  # index 9 = log flag
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
    tf_col = YELLOW if key == "H" else SOFT
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
    a = fade(t, p.t0, p.t1, 0.25, 0.25)
    if a <= 0:
        return
    slide = (1 - ease_out(min(1.0, (t - p.t0) / 0.35))) * -40
    acc, tag_fg, title_col = KIND[p.kind]
    x = 104 + slide
    y = 578
    hgt = 180 if p.desc else 120
    canvas.drawRect(skia.Rect.MakeXYWH(x, y, 5, hgt), paint(with_alpha(acc, a)))
    tw, th = gfx.pill(canvas, p.tag, x + 20, y + 2, gfx.font("cjk_bold", 26), tag_fg, acc, pad_x=11, h=40, alpha=a)
    gfx.text(canvas, p.dates, x + 20 + tw + 14, y + 32, gfx.font("cjk_med", 28), gfx.rgb("#c4cad4"), alpha=a)
    if p.big_value:
        gfx.text(canvas, p.big_value, x + 18, y + 118, gfx.font("num_xb", 94), YELLOW, glow=14, alpha=a)
    else:
        gfx.text(canvas, p.title, x + 16, y + 114, gfx.font("cjk_black", 80), title_col, glow=12, glow_alpha=0.4,
                 alpha=a)
    xx = x + 20
    for s, ck in p.desc:
        xx += gfx.text(canvas, s, xx, y + 166, gfx.font("cjk_med", 31), DESC_COL[ck], alpha=a)


def draw_chip(canvas, xy, text_, kind, alpha, above=True, dy=None):
    if alpha <= 0:
        return
    x, y = xy
    acc = {"up": RED, "down": GREEN, "flat": gfx.rgb("#9fb3d1"), "gold": YELLOW}[kind]
    f = gfx.font("cjk_bold", 26)
    w = gfx.text_width(text_, f) + 24
    h = 42
    dy = dy if dy is not None else (-82 if above else 64)
    above = dy < 0
    bx = min(max(x - w / 2, 20), W - w - 20)
    by = y + dy - (h if above else 0)
    canvas.drawLine(x, y, x, by + (h if above else 0), paint(with_alpha(acc, 0.6 * alpha), "stroke", 1.6))
    canvas.drawCircle(x, y, 6, paint(with_alpha(acc, alpha)))
    gfx.rounded(canvas, bx, by, w, h, 7, fill=gfx.rgb("#0b0f16"), stroke=with_alpha(acc, 0.9), stroke_w=2, alpha=alpha)
    gfx.text(canvas, text_, bx + 12, by + 31, f, acc if kind != "flat" else WHITE, alpha=alpha)


def big_line(canvas, parts, y, fnt, align="right", x=968, alpha=1.0, glow=12):
    arrow_font = gfx.font("cjk_bold", fnt.getSize() * 0.9)
    fonts = [arrow_font if "→" in s else fnt for s, _ in parts]
    total = sum(gfx.text_width(s, f) for (s, _), f in zip(parts, fonts))
    xx = x - total if align == "right" else (x - total / 2 if align == "center" else x)
    for (s, c), f in zip(parts, fonts):
        xx += gfx.text(canvas, s, xx, y, f, c, glow=glow, glow_alpha=0.45, alpha=alpha)


# --------------------------------------------------------------------------- explainer card

CARD_KIND = {"up": RED, "down": GREEN, "gold": YELLOW}
FLOW = {"in": (RED, "▲"), "out": (GREEN, "▼"), "warn": (YELLOW, "◆")}


def _card_layout(card):
    """Measure rows so the card can scale itself to the available height."""
    f_det = gfx.font("cjk_med", 29)
    f_act = gfx.font("cjk_med", 27)
    rows, y = [], 182
    y += 92
    for i, (head, det) in enumerate(card.drivers):
        lines = gfx.wrap(det, f_det, 790)
        h = 48 + 40 * len(lines) + 22
        rows.append(("driver", i, y, head, lines))
        y += h
    if card.capital:
        y += 18
        rows.append(("cap_label", 0, y, None, None))
        y += 76
        for j, cap in enumerate(card.capital):
            lines = gfx.wrap(cap[1], f_act, 600)
            h = 50 + 36 * len(lines) + (34 if cap[3] is not None else 14)
            rows.append(("capital", j, y, cap, lines))
            y += h
    if card.chain:
        y += 10
        rows.append(("chain", 0, y, None, None))
        y += 86
    y += 50
    return rows, y


def draw_card(canvas, card, t):
    a = fade(t, card.t0, card.t1, 0.35, 0.35)
    if a <= 0:
        return
    lt = t - card.t0
    acc = CARD_KIND[card.kind]
    rows, total_h = _card_layout(card)
    x0, y0, w = 56, 596, 968
    avail = 1548 - y0
    sc = min(1.0, avail / total_h)
    canvas.save()
    canvas.translate(x0, y0 + (1 - ease_out(min(1, lt / 0.45))) * 40)
    canvas.scale(sc, sc)
    hgt = total_h
    canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-6, -6, w + 12, hgt + 12), 26, 26),
                     paint(with_alpha(acc, 0.10 * a), blur=18))
    gfx.rounded(canvas, 0, 0, w, hgt, 22, fill=gfx.rgb("#090c12", 0.95), stroke=with_alpha(acc, 0.55), stroke_w=2.5,
                alpha=a)
    canvas.drawRect(skia.Rect.MakeXYWH(0, 26, 8, 104), paint(with_alpha(acc, a)))
    gfx.text(canvas, card.question, 40, 92, gfx.font("cjk_black", 62), acc, glow=12, glow_alpha=0.35, alpha=a)
    gfx.text(canvas, f"{card.title}  ·  {card.dates}", 42, 142, gfx.font("cjk_bold", 30), SOFT, alpha=a)
    canvas.drawLine(40, 172, w - 40, 172, paint(with_alpha(WHITE, 0.10 * a), "stroke", 1.5))

    def label(txt, y, col):
        canvas.drawRect(skia.Rect.MakeXYWH(40, y - 24, 6, 28), paint(with_alpha(col, a)))
        gfx.text(canvas, txt, 58, y, gfx.font("cjk_bold", 30), col, alpha=a)

    label("驱动因素", 222, acc)
    max_amt = max([c[3] for c in card.capital if c[3] is not None] or [1.0])
    n_drv = len(card.drivers)
    for kind, i, y, item, lines in rows:
        if kind == "driver":
            appear = ease_out(min(1, max(0, (lt - 0.6 - 0.6 * i) / 0.4)))
            if appear <= 0:
                continue
            aa = a * appear
            dx = (1 - appear) * -30
            canvas.drawCircle(64 + dx, y + 2, 20, paint(with_alpha(acc, 0.9 * aa)))
            gfx.text(canvas, str(i + 1), 64 + dx, y + 12, gfx.font("num_xb", 28), gfx.rgb("#05070a"), align="center",
                     alpha=aa)
            gfx.text(canvas, item, 104 + dx, y + 16, gfx.font("cjk_black", 40), WHITE, alpha=aa)
            for k, ln in enumerate(lines):
                gfx.text(canvas, ln, 106 + dx, y + 62 + 40 * k, gfx.font("cjk_med", 29), gfx.rgb("#bcc3ce"), alpha=aa)
        elif kind == "cap_label":
            appear = min(1, max(0, (lt - 0.6 - 0.6 * n_drv) / 0.3))
            if appear > 0:
                canvas.save()
                canvas.translate(0, y - 220)
                label("资本动向", 220, YELLOW)
                canvas.restore()
        elif kind == "capital":
            appear = ease_out(min(1, max(0, (lt - 0.9 - 0.6 * n_drv - 0.6 * i) / 0.4)))
            if appear <= 0:
                continue
            aa = a * appear
            actor, action, amount, value, flow = item
            col, glyph = FLOW[flow]
            gfx.text(canvas, glyph, 52, y + 16, gfx.font("cjk_bold", 30), col, alpha=aa)
            gfx.text(canvas, actor, 98, y + 16, gfx.font("cjk_black", 38), WHITE, alpha=aa)
            gfx.text(canvas, amount, w - 40, y + 18, gfx.font("cjk_black", 44), col, align="right", glow=8, alpha=aa)
            for k, ln in enumerate(lines):
                gfx.text(canvas, ln, 100, y + 58 + 36 * k, gfx.font("cjk_med", 27), gfx.rgb("#aab2bf"), alpha=aa)
            if value is not None:
                frac = max(0.05, math.sqrt(value / max_amt))
                grow = ease_out(min(1, max(0, (lt - 1.1 - 0.6 * n_drv - 0.6 * i) / 0.8)))
                by = y + 58 + 36 * len(lines) - 8
                gfx.rounded(canvas, 100, by, (w - 140), 12, 6, fill=with_alpha(WHITE, 0.06), alpha=aa)
                gfx.rounded(canvas, 100, by, (w - 140) * frac * grow, 12, 6, fill=col, alpha=aa)
        elif kind == "chain":
            n = len(card.chain)
            appear = ease_out(min(1, max(0, (lt - 1.2 - 0.6 * n_drv - 0.6 * len(card.capital)) / 0.5)))
            if appear <= 0:
                continue
            f = gfx.font("cjk_bold", 28)
            widths = [gfx.text_width(s, f) + 28 for s in card.chain]
            gap = (w - 80 - sum(widths)) / max(1, n - 1)
            xx = 40
            for k, (s, ww) in enumerate(zip(card.chain, widths)):
                ak = a * min(1, max(0, appear * n - k))
                gfx.rounded(canvas, xx, y, ww, 50, 25, fill=with_alpha(acc, 0.14), stroke=with_alpha(acc, 0.7),
                            stroke_w=2, alpha=ak)
                gfx.text(canvas, s, xx + 14, y + 35, f, WHITE, alpha=ak)
                if k < n - 1:
                    gfx.text(canvas, "→", xx + ww + gap / 2, y + 36, gfx.font("cjk_bold", 30), acc, align="center",
                             alpha=ak)
                xx += ww + gap
    if card.source:
        gfx.text(canvas, f"资料：{card.source}", 40, hgt - 22, gfx.font("cjk_med", 21), DIM, alpha=a)
    canvas.restore()


# --------------------------------------------------------------------------- cold open / rewind / covid / crash

def overlay_cold_open(canvas, t):
    a = fade(t, 0.6, M["cold_end"], 0.35, 0.35)
    if a > 0:
        d = pd.Timestamp(story.CRASH_DAY)
        gfx.text(canvas, f"{d:%Y.%m.%d} 周{WEEKDAY[d.weekday()]}", 968, 626, gfx.font("cjk_bold", 30), SOFT,
                 align="right", alpha=a)
        pop = 1 + 0.25 * (1 - ease_out(min(1, (t - 0.6) / 0.4)))
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
    a2 = fade(t, 3.6, M["cold_end"], 0.4, 0.35)
    if a2 > 0:
        gfx.text(canvas, "特斯拉上市 16 年", W / 2, 1420, gfx.font("cjk_black", 76), YELLOW, align="center", glow=18,
                 alpha=a2)
        gfx.text(canvas, f"2010.06.29 — {story.LAST_DATE:%Y.%m.%d}", W / 2, 1478, gfx.font("cjk_bold", 32), SOFT,
                 align="center", alpha=a2)


def overlay_rewind(canvas, t):
    a = fade(t, M["rewind"], M["story"] + 0.05, 0.15, 0.4)
    if a > 0:
        effects.rewind_icon(canvas, W / 2 + 40, 860, 190, alpha=a)
        gfx.text(canvas, "回到 2010.06.29", W / 2, 1010, gfx.font("cjk_black", 60), gfx.rgb("#e6edf7"),
                 align="center", glow=10, alpha=a)
        gfx.text(canvas, "上市首日", W / 2, 1064, gfx.font("cjk_bold", 30), GREY, align="center", alpha=a)


def overlay_covid_counter(canvas, t, st):
    a = fade(t, story.COVID[0] + 0.15, story.COVID[1], 0.25, 0.25)
    if a <= 0 or st["series"] != "D":
        return
    S = SERIES["D"]
    k = st["k"]
    k0 = S.index_of("2020-02-19")
    n = max(1, k - k0)
    cum = (st["price"] / S.c[k0] - 1) * 100
    day = (st["price"] / S.c[k - 1] - 1) * 100
    gfx.pill(canvas, "闪崩", 110, 830, gfx.font("cjk_bold", 26), WHITE, gfx.rgb("#13c98b"), pad_x=11, h=40, alpha=a)
    gfx.text(canvas, f"第 {n} 个交易日", 110, 940, gfx.font("cjk_black", 72), WHITE, glow=10, alpha=a)
    gfx.text(canvas, gfx.fmt_pct(cum), 106, 1040, gfx.font("num_xb", 100), GREEN if cum < 0 else RED, glow=16,
             alpha=a)
    big_line(canvas, [("当日 ", GREY), (gfx.fmt_pct(day), GREEN if day < 0 else RED)], 1088,
             gfx.font("cjk_bold", 30), align="left", x=110, alpha=a, glow=0)


def overlay_earnings_card(canvas, t):
    a = fade(t, M["crash"] + 0.5, M["gap"] + 0.15, 0.35, 0.15)
    if a <= 0:
        return
    x, y, w, h = 90, 790, 900, 360
    gfx.rounded(canvas, x, y, w, h, 16, fill=gfx.rgb("#0c1018", 0.94), stroke=with_alpha(YELLOW, 0.6), stroke_w=2,
                alpha=a)
    gfx.text(canvas, "Q2 2026 财报 · 7 月 22 日盘后", x + 30, y + 56, gfx.font("cjk_bold", 34), YELLOW, alpha=a)
    rows = [("营收", "282.4 亿美元", "略超预期", WHITE),
            ("每股收益", "$0.33", "预期 $0.53", GREEN),
            ("营业利润率", "1.4%", "营业利润 −57%", GREEN),
            ("自由现金流", "−11 亿美元", "资本开支 +142%", GREEN)]
    for i, (k_, v, note, c) in enumerate(rows):
        yy = y + 124 + i * 60
        ai = a * ease_out(min(1, max(0, (t - M["crash"] - 0.9 - 0.45 * i) / 0.35)))
        gfx.text(canvas, k_, x + 30, yy, gfx.font("cjk_med", 30), GREY, alpha=ai)
        gfx.text(canvas, v, x + 270, yy, gfx.font("cjk_black", 38), c, alpha=ai)
        gfx.text(canvas, note, x + w - 30, yy, gfx.font("cjk_med", 27), GREY, align="right", alpha=ai)


def overlay_crash(canvas, t, st):
    pre = story.PRE_CRASH
    cum = (st["price"] / pre - 1) * 100
    a = fade(t, M["gap"] + 0.05, M["tiles"], 0.15, 0.25)
    if a > 0:
        big_line(canvas, [("一觉醒来，", WHITE), ("天塌了", GREEN)], 690, gfx.font("cjk_black", 76), alpha=a)
        gap = (story.GAP_OPEN / pre - 1) * 100
        gfx.text(canvas, f"开盘跳空 {gfx.fmt_pct(gap)}", 968, 746, gfx.font("cjk_bold", 32), SOFT, align="right",
                 alpha=a)
    a2 = fade(t, M["tiles"], M["multi"], 0.25, 0.25)
    if a2 > 0:
        gfx.text(canvas, "黑色星期四", 968, 690, gfx.font("cjk_black", 80), WHITE, align="right", glow=12, alpha=a2)
        d = (story.close(story.CRASH_DAY) / pre - 1) * 100
        big_line(canvas, [("单日 ", WHITE), (gfx.fmt_pct(d), GREEN), (f" · 上市以来第 {story.CRASH_RANK} 大跌幅", WHITE)],
                 746, gfx.font("cjk_bold", 32), alpha=a2, glow=0)
        overlay_tiles(canvas, t, a2)
    a3 = fade(t, M["multi"], M["summary"] + 0.05, 0.25, 0.25)
    if a3 > 0:
        gfx.text(canvas, "多杀多", 968, 690, gfx.font("cjk_black", 80), WHITE, align="right", glow=12, alpha=a3)
        gfx.text(canvas, "抛售接力，连跌 5 个交易日，跌破 300 美元", 968, 746, gfx.font("cjk_bold", 32), SOFT,
                 align="right", alpha=a3)
    in_tiles = M["tiles"] <= t < M["multi"]
    a_big = fade(t, M["gap"] + 0.05, M["summary"] + 0.05, 0.15, 0.25) * (0.0 if in_tiles else 1.0)
    if a_big > 0:
        gfx.text(canvas, gfx.fmt_pct(cum), 968, 905, gfx.font("num_xb", 150), GREEN, align="right", glow=22,
                 alpha=a_big)
        gfx.text(canvas, f"较财报前收盘 {pre:.2f}", 968, 954, gfx.font("cjk_med", 28), GREY, align="right", alpha=a_big)
    a4 = fade(t, M["summary"] + 0.05, M["crash_end"] + 0.1, 0.35, 0.35)
    if a4 > 0:
        gfx.text(canvas, "8 个月的涨幅，", W / 2, 700, gfx.font("cjk_black", 84), WHITE, align="center", glow=12,
                 alpha=a4)
        retr = (story.ATH3 - story.CRASH_LOW) / (story.ATH3 - story.close("2025-04-08")) * 100
        gfx.text(canvas, f"回吐 {retr:.0f}%。", W / 2, 800, gfx.font("cjk_black", 84), GREEN, align="center",
                 glow=14, alpha=a4)
        big_line(canvas, [(f"{story.ATH3:.2f}", WHITE), ("  →  ", GREY), (f"{story.CRASH_LOW:.2f}", GREEN),
                          (f"   较历史最高 {gfx.fmt_pct((story.CRASH_LOW / story.ATH3 - 1) * 100)}", GREY)],
                 862, gfx.font("cjk_bold", 32), align="center", x=W / 2, alpha=a4, glow=0)


def overlay_tiles(canvas, t, a):
    x0, y0, cw, ch, gap = 110, 800, 276, 112, 16
    for i, (d, v) in enumerate(story.TOP_DROPS):
        r, c = divmod(i, 3)
        appear = ease_out(min(1, max(0, (t - M["tiles"] - 0.3 - i * 0.12) / 0.3)))
        if appear <= 0:
            continue
        x = x0 + c * (cw + gap)
        y = y0 + r * (ch + gap) + (1 - appear) * 30
        hl = d == pd.Timestamp(story.CRASH_DAY).strftime("%Y.%m.%d")
        gfx.rounded(canvas, x, y, cw, ch, 10, fill=gfx.rgb("#071410", 0.92), alpha=a * appear)
        gfx.rounded(canvas, x, y, cw, ch, 10, fill=with_alpha(GREEN, 0.22 if hl else 0.07),
                    stroke=with_alpha(YELLOW if hl else GREEN, 0.95 if hl else 0.35), stroke_w=3 if hl else 1.5,
                    alpha=a * appear)
        gfx.text(canvas, f"#{i + 1}  {d}", x + 16, y + 38, gfx.font("num_sb", 26), YELLOW if hl else GREY,
                 alpha=a * appear)
        gfx.text(canvas, gfx.fmt_pct(v), x + 16, y + 92, gfx.font("num_xb", 50), GREEN, glow=8 if hl else 0,
                 alpha=a * appear)


# --------------------------------------------------------------------------- recap

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
    a = fade(t, M["recap"] + 0.3, M["concl"] + 0.1, 0.4, 0.35)
    if a <= 0:
        return
    gfx.text(canvas, "特斯拉上市 16 年", W / 2, 640, gfx.font("cjk_black", 52), YELLOW, align="center", glow=10,
             alpha=a)
    tot = (story.LAST / story.IPO_ADJ - 1) * 100
    gfx.text(canvas, f"+{tot:,.0f}%", W / 2, 792, gfx.font("num_xb", 150), RED, align="center", glow=22, alpha=a)
    big_line(canvas, [(f"${story.IPO_ADJ:.2f}", WHITE), ("  →  ", GREY), (f"${story.LAST:.2f}", RED),
                      (f"   ×{story.LAST / story.IPO_ADJ:.0f}", YELLOW)],
             856, gfx.font("cjk_bold", 36), align="center", x=W / 2, alpha=a, glow=0)
    gfx.text(canvas, "对数坐标 · 周K", W / 2, 904, gfx.font("cjk_med", 25), GREY, align="center", alpha=0.8 * a)


# --------------------------------------------------------------------------- conclusion

def _title(canvas, title, sub, a, y=690):
    gfx.text(canvas, title, W / 2, y, gfx.font("cjk_black", 68), WHITE, align="center", glow=12, glow_alpha=0.35,
             alpha=a)
    if sub:
        gfx.text(canvas, sub, W / 2, y + 54, gfx.font("cjk_med", 29), SOFT, align="center", alpha=a)


def _rel(t, name):
    return t - M[f"concl_{name}"]


def _part_alpha(t, name, nxt):
    return fade(t, M[f"concl_{name}"], M[nxt] if nxt in M else END, 0.45, 0.4)


def concl_hold(canvas, t):
    a = _part_alpha(t, "hold", "concl_analog")
    if a <= 0:
        return
    lt = _rel(t, "hold")
    _title(canvas, "持有多久才能赚钱？", "2010—2026 年任意一天买入，持有一段时间后的盈利概率", a)
    base, top = 1440, 860
    x0, slot = 110, 860 / 6
    y50 = base - (base - top) * 0.5
    dash = paint(with_alpha(WHITE, 0.35 * a), "stroke", 2)
    dash.setPathEffect(skia.DashPathEffect.Make([10, 8], 0))
    canvas.drawLine(x0 + 10, y50, x0 + 870, y50, dash)
    gfx.text(canvas, "50%", x0 + 2, y50 + 8, gfx.font("num_b", 24), GREY, align="right", alpha=a)
    gfx.text(canvas, "虚线 = 50%，相当于抛硬币", W / 2, 790, gfx.font("cjk_med", 25), GREY, align="center", alpha=a)
    for i, (name, win, med) in enumerate(story.HOLD_STATS):
        g = ease_out(min(1, max(0, (lt - 0.8 - 0.45 * i) / 0.7)))
        if g <= 0:
            continue
        cx = x0 + slot * (i + 0.5)
        hgt = (base - top) * win / 100 * g
        col = gfx.mix(gfx.rgb("#5d6675"), RED, min(1, max(0, (win - 50) / 45)))
        gfx.rounded(canvas, cx - 52, base - hgt, 104, hgt, 10, fill=with_alpha(col, 0.85), alpha=a)
        canvas.drawRect(skia.Rect.MakeXYWH(cx - 52, base - hgt, 104, 6), paint(with_alpha(WHITE, 0.5 * a * g)))
        gfx.text(canvas, f"{win * g:.0f}%", cx, base - hgt - 18, gfx.font("num_xb", 54), col if win > 60 else WHITE,
                 align="center", glow=8, alpha=a)
        gfx.text(canvas, name, cx, base + 48, gfx.font("cjk_bold", 32), WHITE, align="center", alpha=a)
        med_s = f"{med:+.0f}%" if abs(med) >= 100 else f"{med:+.1f}%"
        gfx.text(canvas, f"中位 {med_s}", cx, base + 86, gfx.font("cjk_med", 24), GREY, align="center", alpha=a)
    a2 = a * ease_out(min(1, max(0, (lt - 4.0) / 0.6)))
    gfx.text(canvas, "持有越久，赚钱概率越高", W / 2, 1640, gfx.font("cjk_black", 52), YELLOW, align="center", glow=12,
             alpha=a2)
    gfx.text(canvas, "统计口径：每个交易日买入并持有对应交易日数，按收盘价计算", W / 2, 1696, gfx.font("cjk_med", 22), DIM,
             align="center", alpha=a2)


def concl_analog(canvas, t):
    a = _part_alpha(t, "analog", "concl_street")
    if a <= 0:
        return
    lt = _rel(t, "analog")
    A = story.ANALOG
    _title(canvas, "现在买，历史上会怎样？",
           f"现价 {story.LAST:.2f}，距历史最高 {story.DD_NOW:.1f}%，低于 200 日均线 {story.MA200:.0f}".replace("-", "−"), a)
    g = ease_out(min(1, max(0, (lt - 0.6) / 0.8)))
    gfx.text(canvas, "历史上同样跌 18%~30% 时买入，一年后", W / 2, 840, gfx.font("cjk_bold", 32), SOFT, align="center",
             alpha=a)
    gfx.text(canvas, "上涨概率", 290, 900, gfx.font("cjk_bold", 30), GREY, align="center", alpha=a)
    gfx.text(canvas, f"{A['win'] * g:.0f}%", 290, 1010, gfx.font("num_xb", 120), RED, align="center", glow=14, alpha=a)
    gfx.text(canvas, "中位收益", 790, 900, gfx.font("cjk_bold", 30), GREY, align="center", alpha=a)
    gfx.text(canvas, f"+{A['median'] * g:.1f}%", 790, 1010, gfx.font("num_xb", 120), RED, align="center", glow=14,
             alpha=a)
    mx = max(v for _, v in A["bins"])
    for i, (lbl, v) in enumerate(A["bins"]):
        y = 1070 + i * 58
        gi = ease_out(min(1, max(0, (lt - 1.6 - 0.2 * i) / 0.5)))
        neg = i < 3
        col = GREEN if neg else RED
        gfx.text(canvas, lbl, 300, y + 34, gfx.font("cjk_bold", 27), SOFT, align="right", alpha=a)
        gfx.rounded(canvas, 320, y + 8, 560 * v / mx * gi, 36, 8, fill=with_alpha(col, 0.8), alpha=a)
        gfx.text(canvas, f"{v:.0f}%", 320 + 560 * v / mx * gi + 14, y + 36, gfx.font("num_b", 30), col, alpha=a * gi)
    a2 = a * ease_out(min(1, max(0, (lt - 4.0) / 0.6)))
    gfx.text(canvas, f"样本：{A['n']:,} 个交易日（2015 年以后胜率 {A['win_2015']:.0f}%）", W / 2, 1520,
             gfx.font("cjk_med", 27), GREY, align="center", alpha=a2)
    gfx.text(canvas, "⚠ 幸存者偏差：这是一只 16 年涨了三百多倍的股票", W / 2, 1580, gfx.font("cjk_bold", 32), YELLOW,
             align="center", alpha=a2)
    gfx.text(canvas, "过去的规律，不保证未来重演", W / 2, 1626, gfx.font("cjk_bold", 32), YELLOW, align="center", alpha=a2)


def concl_street(canvas, t):
    a = _part_alpha(t, "street", "concl_dca")
    if a <= 0:
        return
    lt = _rel(t, "street")
    total = sum(n for _, n in story.RATINGS)
    _title(canvas, "华尔街怎么看？", f"{total} 位分析师评级（S&P Global，2026 年 10 月）", a)
    cols = [RED, gfx.rgb("#ff7a6b"), gfx.rgb("#8d96a6"), gfx.rgb("#4fd3a5"), GREEN]
    mx = max(n for _, n in story.RATINGS)
    for i, ((lbl, n), col) in enumerate(zip(story.RATINGS, cols)):
        y = 800 + i * 62
        gi = ease_out(min(1, max(0, (lt - 0.6 - 0.2 * i) / 0.5)))
        gfx.text(canvas, lbl, 260, y + 38, gfx.font("cjk_bold", 30), SOFT, align="right", alpha=a)
        gfx.rounded(canvas, 280, y + 10, 600 * n / mx * gi, 40, 8, fill=with_alpha(col, 0.85), alpha=a)
        gfx.text(canvas, str(n), 280 + 600 * n / mx * gi + 14, y + 42, gfx.font("num_xb", 36), col, alpha=a * gi)
    # consensus gauge
    cx, cy, r = W / 2, 1390, 190
    segs = [GREEN, gfx.rgb("#4fd3a5"), gfx.rgb("#8d96a6"), gfx.rgb("#ff7a6b"), RED]
    for i, col in enumerate(segs):
        p = skia.Path()
        p.addArc(skia.Rect.MakeLTRB(cx - r, cy - r, cx + r, cy + r), 180 + i * 36 + 1.5, 33)
        canvas.drawPath(p, paint(with_alpha(col, 0.85 * a), "stroke", 30))
    for i, lbl in enumerate(["强烈卖出", "卖出", "持有", "买入", "强烈买入"]):
        ang = math.radians(180 + (i + 0.5) * 36)
        gfx.text(canvas, lbl, cx + (r + 52) * math.cos(ang), cy + (r + 52) * math.sin(ang) + 8, gfx.font("cjk_med", 22),
                 GREY, align="center", alpha=a)
    gg = ease_out(min(1, max(0, (lt - 2.0) / 1.2)))
    score = 1 + (story.RATING_SCORE - 1) * gg
    ang = math.radians(180 + (score - 1) / 4 * 180)
    canvas.drawLine(cx, cy, cx + (r - 30) * math.cos(ang), cy + (r - 30) * math.sin(ang),
                    paint(with_alpha(WHITE, a), "stroke", 7))
    canvas.drawCircle(cx, cy, 14, paint(with_alpha(WHITE, a)))
    gfx.text(canvas, f"共识 {story.RATING_SCORE:.2f} / 5 · 持有偏买入", cx, cy + 66, gfx.font("cjk_black", 38), WHITE,
             align="center", alpha=a * gg)
    # price-target range
    T = story.TARGETS
    lo_p, hi_p = 100, 650
    xa, xb = 110, 970

    def px(v):
        return xa + (xb - xa) * (v - lo_p) / (hi_p - lo_p)
    y = 1590
    ga = a * ease_out(min(1, max(0, (lt - 3.2) / 0.6)))
    gfx.rounded(canvas, px(T["low"]), y - 6, px(T["high"]) - px(T["low"]), 12, 6, fill=with_alpha(WHITE, 0.18), alpha=ga)
    for v, lbl, col, up in ((T["low"], f"最低 ${T['low']:.0f}", GREY, False), (T["high"], f"最高 ${T['high']:.0f}", GREY, False),
                            (story.LAST, f"现价 ${story.LAST:.0f}", WHITE, True),
                            (T["avg"], f"平均 ${T['avg']:.0f}", YELLOW, False)):
        canvas.drawCircle(px(v), y, 11, paint(with_alpha(col, ga)))
        gfx.text(canvas, lbl, px(v), y - 26 if up else y + 46, gfx.font("cjk_bold", 25), col, align="center", alpha=ga)
    gfx.text(canvas, f"平均目标价较现价 {(T['avg'] / story.LAST - 1) * 100:+.1f}% · 目标价分歧从 ${T['low']:.0f} 到 ${T['high']:.0f}",
             W / 2, 1700, gfx.font("cjk_med", 26), GREY, align="center", alpha=ga)


def concl_dca(canvas, t):
    a = _part_alpha(t, "dca", "concl_verdict")
    if a <= 0:
        return
    lt = _rel(t, "dca")
    D = story.DCA
    _title(canvas, "同样 6 万美元，怎么买差很多", f"过去 5 年（{D['lump_date']:%Y.%m} — {story.LAST_DATE:%Y.%m}）", a)
    base, scale = 1430, 560 / 90000
    inv_y = base - D["invested"] * scale
    bars = [("5 年前一次性买入", D["lump_value"], f"成本 ${D['lump_cost']:.2f}", gfx.rgb("#9aa3b2")),
            (f"每月定投 $1,000", D["dca_value"], f"平均成本 ${D['dca_cost']:.2f}", RED)]
    for i, (name, val, cost, col) in enumerate(bars):
        g = ease_out(min(1, max(0, (lt - 0.8 - 0.7 * i) / 0.9)))
        cx = 320 + i * 440
        hgt = val * scale * g
        gfx.rounded(canvas, cx - 110, base - hgt, 220, hgt, 12, fill=with_alpha(col, 0.85), alpha=a)
        gfx.text(canvas, f"${val * g:,.0f}", cx, base - hgt - 62, gfx.font("num_xb", 62), col, align="center", glow=8,
                 alpha=a)
        ret = (val / D["invested"] - 1) * 100
        gfx.text(canvas, gfx.fmt_pct(ret * g), cx, base - hgt - 18, gfx.font("num_b", 38), col, align="center", alpha=a)
        gfx.text(canvas, name, cx, base + 50, gfx.font("cjk_black", 34), WHITE, align="center", alpha=a)
        gfx.text(canvas, cost, cx, base + 92, gfx.font("cjk_med", 28), GREY, align="center", alpha=a)
    dash = paint(with_alpha(WHITE, 0.45 * a), "stroke", 2)
    dash.setPathEffect(skia.DashPathEffect.Make([10, 8], 0))
    canvas.drawLine(40, inv_y, 1040, inv_y, dash)
    gfx.text(canvas, "投入", 40, inv_y - 40, gfx.font("cjk_med", 24), GREY, alpha=a)
    gfx.text(canvas, "$60,000", 40, inv_y - 12, gfx.font("num_b", 26), GREY, alpha=a)
    a2 = a * ease_out(min(1, max(0, (lt - 3.4) / 0.6)))
    cut = (1 - D["dca_cost"] / D["lump_cost"]) * 100
    gfx.text(canvas, f"定投把持仓成本拉低了 {cut:.0f}%", W / 2, 1640, gfx.font("cjk_black", 50), YELLOW, align="center",
             glow=12, alpha=a2)
    gfx.text(canvas, "回测只基于历史价格，结论高度依赖起止时间", W / 2, 1694, gfx.font("cjk_med", 23), DIM, align="center",
             alpha=a2)


def _icon(canvas, kind, cx, cy, col, a):
    canvas.drawCircle(cx, cy, 34, paint(with_alpha(col, 0.16 * a)))
    canvas.drawCircle(cx, cy, 34, paint(with_alpha(col, 0.9 * a), "stroke", 3))
    p = skia.Path()
    if kind == "yes":
        p.moveTo(cx - 15, cy + 1)
        p.lineTo(cx - 4, cy + 12)
        p.lineTo(cx + 16, cy - 12)
    elif kind == "no":
        p.moveTo(cx - 13, cy - 13)
        p.lineTo(cx + 13, cy + 13)
        p.moveTo(cx + 13, cy - 13)
        p.lineTo(cx - 13, cy + 13)
    elif kind == "warn":
        p.moveTo(cx, cy - 16)
        p.lineTo(cx, cy + 4)
        canvas.drawCircle(cx, cy + 14, 3.5, paint(with_alpha(col, a)))
    else:  # half
        canvas.drawArc(skia.Rect.MakeLTRB(cx - 18, cy - 18, cx + 18, cy + 18), 90, 180, True, paint(with_alpha(col, a)))
        canvas.drawCircle(cx, cy, 18, paint(with_alpha(col, a), "stroke", 3))
    canvas.drawPath(p, paint(with_alpha(col, a), "stroke", 5))


def concl_verdict(canvas, t):
    a = fade(t, M["concl_verdict"], M["outro"] + 0.1, 0.45, 0.4)
    if a <= 0:
        return
    lt = _rel(t, "verdict")
    hold = {n: (w, m) for n, w, m in story.HOLD_STATS}
    _title(canvas, "数据给出的结论", "不是买卖建议，是 16 年历史数据告诉我们的事", a, y=680)
    rows = [
        ("yes", RED, "长期持有 3 年以上", f"历史胜率 {hold['3 年'][0]:.0f}%",
         f"5 年胜率 {hold['5 年'][0]:.1f}%，中位收益 {hold['5 年'][1]:+.0f}%"),
        ("no", gfx.rgb("#9aa3b2"), "短线追涨杀跌", f"1 个月胜率 {hold['1 个月'][0]:.0f}%",
         f"接近抛硬币；单日涨跌超 10% 出现过 {story.BIG_DAYS} 次"),
        ("warn", GREEN, "满仓 / 加杠杆", f"最大回撤 {story.MAX_DD:.1f}%".replace("-", "−"),
         f"16 年 {story.HALVINGS} 次腰斩，只用亏得起的钱"),
        ("half", YELLOW, "现在入场", "分批 > 一次性",
         f"距高点 {story.DD_NOW:.0f}%，华尔街平均目标价 ${story.TARGETS['avg']:.0f}（{(story.TARGETS['avg'] / story.LAST - 1) * 100:+.0f}%）".replace("-", "−")),
    ]
    for i, (icon, col, head, pill, reason) in enumerate(rows):
        g = ease_out(min(1, max(0, (lt - 0.8 - 0.8 * i) / 0.45)))
        if g <= 0:
            continue
        y = 780 + i * 178
        aa = a * g
        dx = (1 - g) * -40
        gfx.rounded(canvas, 60 + dx, y, 960, 156, 18, fill=gfx.rgb("#0b0f16", 0.93), stroke=with_alpha(col, 0.55),
                    stroke_w=2, alpha=aa)
        _icon(canvas, icon, 124 + dx, y + 78, col, aa)
        gfx.text(canvas, head, 184 + dx, y + 66, gfx.font("cjk_black", 42), WHITE, alpha=aa)
        f = gfx.font("cjk_bold", 28)
        pw = gfx.text_width(pill, f) + 28
        gfx.rounded(canvas, 990 - pw + dx, y + 30, pw, 46, 23, fill=with_alpha(col, 0.18), stroke=with_alpha(col, 0.8),
                    stroke_w=2, alpha=aa)
        gfx.text(canvas, pill, 990 - pw + 14 + dx, y + 63, f, col, alpha=aa)
        gfx.text(canvas, reason, 186 + dx, y + 120, gfx.font("cjk_med", 28), gfx.rgb("#b8c0cc"), alpha=aa)
    a2 = a * ease_out(min(1, max(0, (lt - 4.6) / 0.6)))
    gfx.rounded(canvas, 60, 1500, 960, 120, 18, fill=with_alpha(YELLOW, 0.08), stroke=with_alpha(YELLOW, 0.5),
                stroke_w=2, alpha=a2)
    gfx.text(canvas, "适合：看好长期、能扛住腰斩、分批买入的人", W / 2, 1550, gfx.font("cjk_black", 34), YELLOW,
             align="center", alpha=a2)
    gfx.text(canvas, "不适合：想短期翻倍、借钱或加杠杆的人", W / 2, 1598, gfx.font("cjk_bold", 30), SOFT, align="center",
             alpha=a2)
    gfx.text(canvas, "以上均为历史数据统计，不构成任何投资建议；股市有风险，投资需谨慎", W / 2, 1680,
             gfx.font("cjk_med", 23), DIM, align="center", alpha=a2)


# --------------------------------------------------------------------------- outro

def overlay_outro_text(canvas, t):
    a = fade(t, M["outro"] + 0.6, END, 0.5, 0.01)
    if a > 0:
        gfx.text(canvas, f"上市 16 年，股价 ×{story.LAST / story.IPO_ADJ:.0f}，", W / 2, 690, gfx.font("cjk_black", 56),
                 WHITE, align="center", glow=8, alpha=a)
        cn = "零一二三四五六七八九十"
        gfx.text(canvas, f"也经历过{cn[story.HALVINGS]}次腰斩。", W / 2, 772, gfx.font("cjk_black", 56), GREEN,
                 align="center", glow=8, alpha=fade(t, M["outro"] + 1.4, END, 0.5, 0.01))
        gfx.text(canvas, "下一个 16 年，你怎么看？", W / 2, 1815, gfx.font("cjk_black", 46), YELLOW, align="center",
                 glow=10, alpha=fade(t, M["outro"] + 3.0, END, 0.5, 0.01))


def draw_forecast(canvas, cam, view, st, t):
    a = fade(t, M["outro"] + 0.4, END + 0.1, 0.6, 0.01)
    if a <= 0:
        return
    S = SERIES["W"]
    k = len(S) - 1
    steps = 30
    grow = ease_in_out((t - M["outro"] - 0.6) / 3.0)
    ux0 = float(view.u(k))
    ux1 = 1.0
    xs, ys, _ = cam.project(np.array([ux0, ux1, ux1, ux0]), np.array([0.0, 0.0, 1.0, 1.0]))
    pth = skia.Path()
    chart._quad(pth, list(zip(xs, ys)))
    canvas.drawPath(pth, paint(with_alpha(gfx.rgb("#5c6b85"), 0.10 * a)))
    dash = paint(with_alpha(WHITE, 0.5 * a), "stroke", 2)
    dash.setPathEffect(skia.DashPathEffect.Make([8, 8], 0))
    canvas.drawLine(xs[0], ys[0], xs[3], ys[3], dash)
    gfx.text(canvas, f"今天 {story.LAST_DATE:%Y.%m.%d}", xs[0] - 12, ys[0] - 18, gfx.font("cjk_med", 24), GREY,
             align="right", alpha=a)
    f_q = gfx.font("cjk_black", 48)
    gfx.text(canvas, "后续走势？", min(xs[3] + 16, W - 40 - gfx.text_width("后续走势？", f_q)), ys[3] + 66, f_q,
             gfx.rgb("#9aa6b8"), alpha=0.9 * a)
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
        if s.series == "H" and t >= M["crash"]:
            i0 = SERIES["H"].index_of("2026-07-23 09:30")
            if k >= i0:
                st["hour_pill"] = f"财报后 · 第 {k - i0 + 1} 小时"
        self._states[f] = st
        return st

    def view_cam(self, f, st):
        (left, right, y0, y1, yaw, pitch, cy, ph, dim), log = self.track[min(f, len(self.track) - 1)]
        key = st["series"]
        view = chart.View(date_to_idx(key, left) - 0.5, date_to_idx(key, right), y0, y1, log)
        cam = chart.Camera(yaw=yaw, pitch=pitch, cy=cy, ph=ph)
        return view, cam, dim

    def target_xy(self, target, st, view, cam):
        key = st["series"]
        S = SERIES[key]
        if target == "head":
            u, v = float(view.u(st["k"])), float(view.v(st["price"]))
        else:
            k = S.index_of(target)
            u, v = float(view.u(k)), float(view.v(story.close(target) if key != "H" else S.c[k]))
        x, y, _ = cam.project(np.array([u]), np.array([v]))
        return float(x[0]), float(y[0])

    def render(self, f):
        st = self.state(f)
        t = st["t"]
        key = st["series"]
        S = SERIES[key]
        view, cam, dim = self.view_cam(f, st)

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
        snow = 1.0 if any(a <= t < b for a, b in _SNOW) else 0.0
        effects.star_field(cv, t, alpha=0.9, snow=snow)

        # ---- chart layer (glowing) ----
        lc = self.layer.getCanvas()
        lc.clear(skia.ColorTRANSPARENT)
        lc.save()
        lc.translate(sx, sy)
        style = chart.ChartStyle(alpha=dim, labels=1.0)
        info = chart.draw_chart(lc, cam, view, S, st["head"], style, date_labels=DATE_LABELS[key])
        for b in story.BOXES:
            a = fade(t, b.t0, b.t1, 0.35, 0.3) * dim
            if a > 0 and key != "H":
                i_s = S.index_of(b.start)
                p_s = story.close(b.start)
                pct_ = (st["price"] / p_s - 1) * 100
                dd = (pd.Timestamp(S.dates[st["k"]]) - pd.Timestamp(b.start)).days
                chart.draw_range_box(lc, cam, view, i_s, p_s, st["k"], st["price"], alpha=a, label=(pct_, dd))
        if key == "H" and M["gap"] <= t < M["crash_end"]:
            i_s = S.index_of("2026-07-22 15:30")
            chart.draw_range_box(lc, cam, view, i_s, story.PRE_CRASH, st["k"], st["price"],
                                 alpha=fade(t, M["gap"], M["crash_end"], 0.3, 0.3))
        hd_date = pd.Timestamp(int(idx_to_date(key, st["head"])))
        if key in ("D", "W"):
            for ch in story.CHIPS:
                cd = pd.Timestamp(ch.date)
                if cd <= hd_date and ch.t_start <= t < ch.t_end:
                    i_c = S.index_of(ch.date)
                    if view.u(i_c) < -0.02:
                        continue
                    age = (hd_date - cd).days
                    a = min(1.0, age / 25.0 + 0.2) * fade(t, ch.t_start, ch.t_end, 0, 0.3) * dim
                    xy = cam.project(np.array([float(view.u(i_c))]), np.array([float(view.v(story.close(ch.date)))]))
                    draw_chip(lc, (float(xy[0][0]), float(xy[1][0])), ch.text, ch.kind, a, ch.above)
        if M["recap"] <= t < M["concl"] + 0.5:
            for n_, (d, txt, kind, dy) in enumerate(RECAP_CHIPS):
                a = fade(t, M["recap"] + 0.8 + n_ * 0.45, M["concl"] + 0.5, 0.3, 0.4) * dim
                i_c = S.index_of(d)
                xy = cam.project(np.array([float(view.u(i_c))]), np.array([float(view.v(S.c[i_c]))]))
                draw_chip(lc, (float(xy[0][0]), float(xy[1][0])), txt, kind, a, dy=dy)
        if t >= M["outro"]:
            draw_forecast(lc, cam, view, st, t)
        if info:
            strong = 1.0 if (key == "H" and t > M["gap"]) or story.COVID[0] <= t < story.COVID[1] else 0.45
            effects.sparkles(lc, info["head_xy"], t, RED if info["head_up"] else GREEN, n=46, alpha=strong * dim)
            chart.draw_head(lc, info["head_xy"], info["head_up"], t, alpha=dim)
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
                                  color=gfx.rgb("#bff7e4") if key == "H" else gfx.ICE)
            elif kind == "shatter":
                effects.shatter(cv, self.target_xy(target, st, view, cam), lt, color=GREEN)
            elif kind == "fireworks":
                effects.fireworks(cv, self.target_xy(target, st, view, cam), lt, spread=extra)
        cv.restore()

        # ---- overlays ----
        draw_hud(cv, st, alpha=fade(t, 0.0, END, 0.25, 0.8))
        for p in story.PANELS:
            if p.t0 - 0.5 <= t <= p.t1 + 0.5:
                draw_panel(cv, p, t)
        for c in story.CARDS:
            if c.t0 - 0.1 <= t <= c.t1 + 0.1:
                draw_card(cv, c, t)
        if t < M["story"] + 0.2:
            overlay_cold_open(cv, t)
            overlay_rewind(cv, t)
        overlay_covid_counter(cv, t, st)
        if M["crash"] <= t < M["crash_end"] + 0.3:
            overlay_earnings_card(cv, t)
            overlay_crash(cv, t, st)
        if M["recap"] <= t < M["outro"] + 0.5:
            overlay_recap(cv, t)
            concl_hold(cv, t)
            concl_analog(cv, t)
            concl_street(cv, t)
            concl_dca(cv, t)
            concl_verdict(cv, t)
        if t >= M["outro"]:
            overlay_outro_text(cv, t)
        for (tf, kind, target, extra) in FX:
            if kind == "flash":
                effects.flash(cv, t - tf, peak=extra)
        effects.vignette(cv, 0.7)
        gfx.text(cv, "TSLA · 2010—2026", W - 40, H - 46, gfx.font("cjk_med", 22), DIM, align="right")

        arr = self.surf.toarray()[:, :, :3]
        if M["rewind"] <= t < M["story"] + 0.1:
            g = math.sin(min(1, (t - M["rewind"]) / (M["story"] - M["rewind"] + 0.1)) * math.pi)
            arr = effects.rgb_split(effects.scan_glitch(arr, t, g), 10 * g)
        fo = min(1.0, t / 0.25, max(0.0, (END - t) / 1.5))
        if fo < 1:
            arr = (arr.astype(np.float32) * fo).astype(np.uint8)
        return arr
