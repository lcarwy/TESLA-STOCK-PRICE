"""Procedural effects: star field, lightning, shatter, fireworks, sparkles, flash, glitch, flare."""
import math

import numpy as np
import skia

from . import gfx
from .gfx import W, H, WHITE, with_alpha, paint

_RNG = np.random.default_rng(20100629)
_STARS = _RNG.random((260, 4))  # x, y, depth, phase


def star_field(canvas, t, alpha=1.0, drift=(0.0, 18.0), snow=0.0):
    """Slowly drifting dust/snow particles with twinkle and parallax."""
    if alpha <= 0:
        return
    for x, y, d, ph in _STARS:
        depth = 0.3 + 0.7 * d
        px = (x * W + drift[0] * t * depth) % W
        py = (y * H + drift[1] * t * depth * (1 + 3 * snow)) % H
        tw = 0.55 + 0.45 * math.sin(t * (1.3 + 2 * ph) + ph * 20)
        r = 0.8 + 1.8 * depth
        a = (0.10 + 0.35 * depth) * tw * alpha
        canvas.drawCircle(px, py, r, paint(with_alpha(WHITE, a), blur=0.8 if depth < 0.6 else 0))


def vignette(canvas, strength=0.85):
    shader = skia.GradientShader.MakeRadial(
        (W / 2, H * 0.55), H * 0.75, [skia.Color4f(0, 0, 0, 0).toColor(), skia.Color4f(0, 0, 0, strength).toColor()],
        [0.45, 1.0])
    p = skia.Paint(Shader=shader)
    canvas.drawRect(skia.Rect.MakeWH(W, H), p)


def _bolt_points(x0, y0, x1, y1, seed, rough=0.22, depth=7):
    rng = np.random.default_rng(seed)
    pts = [(x0, y0), (x1, y1)]
    disp = math.hypot(x1 - x0, y1 - y0) * rough
    for _ in range(depth):
        nxt = [pts[0]]
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            mx, my = (ax + bx) / 2, (ay + by) / 2
            nx, ny = -(by - ay), (bx - ax)
            nl = math.hypot(nx, ny) or 1
            off = rng.normal(0, disp)
            nxt += [(mx + nx / nl * off, my + ny / nl * off), (bx, by)]
        pts = nxt
        disp *= 0.55
    return pts, rng


def lightning(canvas, target, t_local, color=gfx.ICE, seed=7, alpha=1.0, branches=3):
    """Flickering bolt from above the screen down to `target`, ~0.6 s long."""
    if t_local < 0 or t_local > 0.75:
        return
    flick = [1, 0.25, 1, 0.6, 0.9, 0.3, 0.7, 0.2]
    k = min(int(t_local / 0.75 * len(flick)), len(flick) - 1)
    a = alpha * flick[k] * (1 - t_local / 0.75) ** 0.6
    x1, y1 = target
    pts, rng = _bolt_points(x1 + rng_off(seed) * 160, -40, x1, y1, seed)
    paths = [pts]
    for b in range(branches):
        i = int(len(pts) * (0.25 + 0.5 * rng.random()))
        bx, by = pts[i]
        ex, ey = bx + rng.normal(0, 180), by + abs(rng.normal(160, 90))
        bp, _ = _bolt_points(bx, by, ex, ey, seed + 31 * (b + 1), 0.3, 5)
        paths.append(bp)
    for n, pp in enumerate(paths):
        pth = skia.Path()
        pth.moveTo(*pp[0])
        for p in pp[1:]:
            pth.lineTo(*p)
        w = 4.0 if n == 0 else 2.0
        canvas.drawPath(pth, paint(with_alpha(color, 0.35 * a), "stroke", w * 6, blur=18))
        canvas.drawPath(pth, paint(with_alpha(color, 0.8 * a), "stroke", w * 1.8, blur=3))
        canvas.drawPath(pth, paint(with_alpha(WHITE, a), "stroke", w * 0.7))
    canvas.drawCircle(x1, y1, 70, paint(with_alpha(color, 0.45 * a), blur=40))


def rng_off(seed):
    return math.sin(seed * 12.9898) * 0.8


def shatter(canvas, origin, t_local, color=gfx.GREEN, seed=3, n=70, alpha=1.0, scale=1.0):
    """Glass shards and dust exploding outwards from `origin` (~1.6 s)."""
    if t_local < 0 or t_local > 1.8:
        return
    rng = np.random.default_rng(seed)
    ox, oy = origin
    a = alpha * (1 - t_local / 1.8) ** 1.3
    for i in range(n):
        ang = rng.uniform(0, 2 * math.pi)
        spd = rng.uniform(120, 620) * scale
        rot = rng.uniform(-6, 6)
        size = rng.uniform(6, 28) * scale
        tt = t_local
        x = ox + math.cos(ang) * spd * tt
        y = oy + math.sin(ang) * spd * tt + 260 * tt * tt
        th = rot * tt + ang
        pts = [(x + size * math.cos(th + k * 2.1 + rng.uniform(-0.4, 0.4)),
                y + size * math.sin(th + k * 2.1 + rng.uniform(-0.4, 0.4))) for k in range(3)]
        pth = skia.Path()
        pth.moveTo(*pts[0])
        pth.lineTo(*pts[1])
        pth.lineTo(*pts[2])
        pth.close()
        col = color if i % 3 else WHITE
        canvas.drawPath(pth, paint(with_alpha(col, 0.28 * a)))
        canvas.drawPath(pth, paint(with_alpha(col, 0.85 * a), "stroke", 1.4))
    for i in range(160):
        ang = rng.uniform(0, 2 * math.pi)
        spd = rng.uniform(40, 520) * scale
        x = ox + math.cos(ang) * spd * t_local
        y = oy + math.sin(ang) * spd * t_local + 120 * t_local ** 2
        canvas.drawCircle(x, y, rng.uniform(1, 2.6), paint(with_alpha(color if i % 2 else WHITE, 0.8 * a)))
    if t_local < 0.35:
        canvas.drawCircle(ox, oy, 60 + 500 * t_local, paint(with_alpha(WHITE, 0.5 * (1 - t_local / 0.35)), "stroke",
                                                           6, blur=6))


def fireworks(canvas, origin, t_local, color=gfx.YELLOW, seed=11, n=220, alpha=1.0, spread=1.0):
    """Golden burst of sparks with gravity and trails (~1.8 s)."""
    if t_local < 0 or t_local > 2.0:
        return
    rng = np.random.default_rng(seed)
    ox, oy = origin
    for i in range(n):
        ang = rng.uniform(0, 2 * math.pi)
        spd = rng.uniform(80, 420) * spread * (0.6 + 0.4 * abs(math.cos(ang)) + 0.2)
        life = rng.uniform(0.9, 2.0)
        if t_local > life:
            continue
        k = t_local / life
        drag = 1 - math.exp(-3 * t_local)
        x = ox + math.cos(ang) * spd * drag / 3 * 2.2
        y = oy + math.sin(ang) * spd * drag / 3 * 1.2 + 90 * t_local ** 2
        x0 = ox + math.cos(ang) * spd * (1 - math.exp(-3 * max(0, t_local - 0.08))) / 3 * 2.2
        y0 = oy + math.sin(ang) * spd * (1 - math.exp(-3 * max(0, t_local - 0.08))) / 3 * 1.2
        a = alpha * (1 - k) ** 1.5
        col = color if i % 4 else WHITE
        canvas.drawLine(x0, y0, x, y, paint(with_alpha(col, 0.9 * a), "stroke", 2.0))
    canvas.drawCircle(ox, oy, 120 * spread, paint(with_alpha(color, 0.35 * alpha * max(0, 1 - t_local / 0.8)),
                                                   blur=50))


def sparkles(canvas, origin, t, color, seed=5, n=40, radius=90, alpha=1.0):
    """Shimmering particle cloud around a point (for the moving head on strong moves)."""
    if alpha <= 0:
        return
    rng = np.random.default_rng(seed)
    ox, oy = origin
    for i in range(n):
        ang = rng.uniform(0, 2 * math.pi)
        r = radius * math.sqrt(rng.random())
        ph = rng.uniform(0, 6.28)
        x = ox + math.cos(ang + t * 0.6) * r - 30 * (((t * 0.8 + ph) % 1.0))
        y = oy + math.sin(ang + t * 0.6) * r * 0.8 - 40 * ((t * 0.5 + ph) % 1.0)
        tw = 0.5 + 0.5 * math.sin(t * 9 + ph * 7)
        canvas.drawCircle(x, y, 1.2 + 1.3 * tw, paint(with_alpha(color if i % 3 else WHITE, alpha * (0.3 + 0.6 * tw))))


def flash(canvas, t_local, dur=0.35, color=WHITE, peak=0.45):
    if 0 <= t_local <= dur:
        a = peak * (1 - t_local / dur) ** 2
        canvas.drawRect(skia.Rect.MakeWH(W, H), paint(with_alpha(color, a)))


def shake_offset(t_local, amp=14.0, dur=0.5, seed=1):
    if t_local < 0 or t_local > dur:
        return 0.0, 0.0
    k = (1 - t_local / dur) ** 2
    return (amp * k * math.sin(t_local * 83 + seed), amp * k * math.cos(t_local * 71 + 2 * seed))


def lens_glow(canvas, center, radius, color, alpha=1.0):
    x, y = center
    canvas.drawCircle(x, y, radius, paint(with_alpha(color, 0.18 * alpha), blur=radius * 0.6))
    canvas.drawCircle(x, y, radius * 0.35, paint(with_alpha(color, 0.35 * alpha), blur=radius * 0.25))
    canvas.drawLine(x - radius * 2.2, y, x + radius * 2.2, y, paint(with_alpha(color, 0.35 * alpha), "stroke", 3,
                                                                    blur=4))


def rgb_split(img_array, amount):
    """Chromatic aberration on an RGBA uint8 frame (in place copy)."""
    k = int(round(amount))
    if k < 1:
        return img_array
    out = img_array.copy()
    out[:, k:, 0] = img_array[:, :-k, 0]
    out[:, :-k, 2] = img_array[:, k:, 2]
    return out


def scan_glitch(img_array, t, strength, seed=9):
    """Horizontal slice displacement for the rewind effect."""
    if strength <= 0.01:
        return img_array
    rng = np.random.default_rng(seed + int(t * 30))
    out = img_array.copy()
    for _ in range(int(6 + 10 * strength)):
        y = rng.integers(0, H - 40)
        h = rng.integers(4, 40)
        dx = int(rng.normal(0, 40 * strength))
        out[y:y + h] = np.roll(out[y:y + h], dx, axis=1)
    return out


def rewind_icon(canvas, cx, cy, size, alpha=1.0):
    col = with_alpha(gfx.rgb("#dfe8f5"), alpha)
    for k in range(2):
        x = cx - size * 0.15 + k * size * 0.55 - size * 0.55
        pth = skia.Path()
        pth.moveTo(x, cy)
        pth.lineTo(x + size * 0.55, cy - size * 0.42)
        pth.lineTo(x + size * 0.55, cy + size * 0.42)
        pth.close()
        canvas.drawPath(pth, paint(with_alpha(col, 0.35 * alpha), blur=10))
        canvas.drawPath(pth, paint(col))


def random_walks(n, steps, seed, vol=0.035, drift=0.0):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, size=(n, steps))
    return np.exp(np.cumsum(r, axis=1))
