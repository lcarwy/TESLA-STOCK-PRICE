"""skia drawing helpers: palette, fonts, glowing text, pills."""
import os
from functools import lru_cache

import numpy as np
import skia

from .data import ROOT

W, H = 1080, 1920
FONT_DIR = os.path.join(ROOT, "assets", "fonts")


def rgb(hexstr, a=1.0):
    hexstr = hexstr.lstrip("#")
    r, g, b = (int(hexstr[i:i + 2], 16) for i in (0, 2, 4))
    return skia.Color4f(r / 255, g / 255, b / 255, a)


def with_alpha(c4, a):
    return skia.Color4f(c4.fR, c4.fG, c4.fB, c4.fA * a)


def mix(c1, c2, t):
    return skia.Color4f(*(a + (b - a) * t for a, b in zip(
        (c1.fR, c1.fG, c1.fB, c1.fA), (c2.fR, c2.fG, c2.fB, c2.fA))))


RED = rgb("#ff3d3d")
RED_HI = rgb("#ff6b5e")
GREEN = rgb("#19f0a0")
GREEN_HI = rgb("#6dffd0")
YELLOW = rgb("#ffc93c")
WHITE = rgb("#f4f6fa")
GREY = rgb("#8d96a6")
DIM = rgb("#4b5262")
ICE = rgb("#cfe6ff")
CYAN = rgb("#7fe9ff")
BG = rgb("#030406")


def up_color(chg):
    return RED if chg >= 0 else GREEN


@lru_cache(maxsize=None)
def typeface(name):
    files = {
        "cjk_black": "NotoSansSC-Black.ttf",
        "cjk_bold": "NotoSansSC-Bold.ttf",
        "cjk_med": "NotoSansSC-Medium.ttf",
        "num_xb": "BarlowSemiCondensed-ExtraBold.ttf",
        "num_b": "BarlowSemiCondensed-Bold.ttf",
        "num_sb": "BarlowSemiCondensed-SemiBold.ttf",
        "num_m": "BarlowSemiCondensed-Medium.ttf",
    }
    return skia.Typeface.MakeFromFile(os.path.join(FONT_DIR, files[name]))


@lru_cache(maxsize=None)
def font(name, size):
    f = skia.Font(typeface(name), size)
    f.setEdging(skia.Font.Edging.kSubpixelAntiAlias)
    f.setSubpixel(True)
    return f


def paint(color, style="fill", width=1.0, blur=0.0, blend=None, aa=True):
    p = skia.Paint(AntiAlias=aa)
    p.setColor4f(color)
    if style == "stroke":
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(width)
        p.setStrokeCap(skia.Paint.kRound_Cap)
        p.setStrokeJoin(skia.Paint.kRound_Join)
    if blur > 0:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    if blend is not None:
        p.setBlendMode(blend)
    return p


def text_width(s, fnt):
    return fnt.measureText(s)


def text(canvas, s, x, y, fnt, color, align="left", glow=0.0, glow_alpha=0.55, alpha=1.0, tracking=0.0):
    """Draw text with its baseline at y. Returns the advance width."""
    if alpha <= 0.003 or not s:
        return 0.0
    w = text_width(s, fnt) + tracking * max(0, len(s) - 1)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    col = with_alpha(color, alpha)
    if tracking:
        def draw(p):
            xx = x
            for ch in s:
                canvas.drawString(ch, xx, y, fnt, p)
                xx += text_width(ch, fnt) + tracking
    else:
        def draw(p):
            canvas.drawString(s, x, y, fnt, p)
    if glow > 0:
        draw(paint(with_alpha(color, glow_alpha * alpha), blur=glow))
    draw(paint(col))
    return w


def rounded(canvas, x, y, w, h, r, fill=None, stroke=None, stroke_w=2.0, alpha=1.0):
    rect = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r)
    if fill is not None:
        canvas.drawRRect(rect, paint(with_alpha(fill, alpha)))
    if stroke is not None:
        canvas.drawRRect(rect, paint(with_alpha(stroke, alpha), "stroke", stroke_w))


def pill(canvas, s, x, y, fnt, fg, bg, pad_x=10, h=None, alpha=1.0, outline=None):
    """Label in a rounded box; (x, y) is the top-left. Returns (w, h)."""
    m = fnt.getMetrics()
    th = -m.fAscent + m.fDescent
    h = h or th + 10
    w = text_width(s, fnt) + 2 * pad_x
    rounded(canvas, x, y, w, h, 6, fill=bg, stroke=outline, stroke_w=2, alpha=alpha)
    base = y + h / 2 + (-m.fAscent - m.fDescent) / 2
    text(canvas, s, x + pad_x, base, fnt, fg, alpha=alpha)
    return w, h


def ease_in_out(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease_out(x, p=3):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** p


def ease_in(x, p=2):
    x = min(max(x, 0.0), 1.0)
    return x ** p


def fade(t, t0, t1, fin=0.25, fout=0.25):
    """1 inside [t0, t1] with linear fades at both ends."""
    if t < t0 or t > t1:
        return 0.0
    a = 1.0
    if fin > 0:
        a = min(a, (t - t0) / fin)
    if fout > 0:
        a = min(a, (t1 - t) / fout)
    return float(np.clip(a, 0, 1))


def fmt_pct(x, digits=1):
    return f"{x:+.{digits}f}%".replace("-", "−")


def fmt_price(x):
    return f"{x:,.2f}"


def _tokens(s):
    tokens, word = [], ""
    for ch in s:
        if ch.isascii() and not ch.isspace():
            word += ch
            continue
        if word:
            tokens.append(word)
            word = ""
        tokens.append(ch)
    if word:
        tokens.append(word)
    return tokens


def _wrap_tokens(tokens, fnt, max_w):
    lines, cur = [], ""
    for tok in tokens:
        if text_width(cur + tok, fnt) > max_w and cur:
            lines.append(cur.rstrip())
            cur = tok.lstrip()
        else:
            cur += tok
    if cur:
        lines.append(cur)
    return lines


def wrap(s, fnt, max_w):
    """Greedy line wrap for CJK text (any character may break; ASCII words stay whole).
    Re-wraps narrower when the last line would be a dangling one- or two-character stub."""
    tokens = _tokens(s)
    lines = _wrap_tokens(tokens, fnt, max_w)
    w = max_w
    while len(lines) > 1 and len(lines[-1]) <= 3 and w > max_w * 0.6:
        w *= 0.92
        lines = _wrap_tokens(tokens, fnt, w)
    return lines
