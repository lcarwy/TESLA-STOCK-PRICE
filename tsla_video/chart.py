"""Perspective candlestick chart: glass-like 3D candles, curtains, close line, grid, range box."""
import math
from dataclasses import dataclass, field

import numpy as np
import skia

from . import gfx
from .gfx import GREEN, RED, WHITE, GREY, with_alpha, paint


@dataclass
class Camera:
    cx: float = 540.0       # screen position of the plot centre
    cy: float = 1270.0
    pw: float = 1000.0      # plot width / height in world units
    ph: float = 900.0
    yaw: float = 12.0       # degrees; positive brings the right side towards the viewer
    pitch: float = -4.0     # degrees; negative looks slightly down onto the plot
    dist: float = 1700.0
    focal: float = 1700.0

    def project(self, u, v, z=0.0):
        """Plot coords (u, v in [0, 1], z world depth) -> screen (x, y, scale)."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        z = np.broadcast_to(np.asarray(z, dtype=float), u.shape)
        X = (u - 0.5) * self.pw
        Y = (v - 0.5) * self.ph
        cy_, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        X1 = X * cy_ + z * sy
        Z1 = -X * sy + z * cy_
        Y2 = Y * cp - Z1 * sp
        Z2 = Y * sp + Z1 * cp
        s = self.focal / (Z2 + self.dist)
        return self.cx + X1 * s, self.cy - Y2 * s, s


@dataclass
class View:
    """What part of a series is on screen and how prices map to the plot."""
    x0: float
    x1: float
    y0: float
    y1: float
    log: bool = False

    def u(self, idx):
        return (np.asarray(idx, dtype=float) - self.x0) / (self.x1 - self.x0)

    def v(self, price):
        p = np.asarray(price, dtype=float)
        if self.log:
            return (np.log(p) - math.log(self.y0)) / (math.log(self.y1) - math.log(self.y0))
        return (p - self.y0) / (self.y1 - self.y0)


def nice_ticks(lo, hi, n=5, log=False):
    if log:
        out = []
        for e in range(-1, 5):
            for m in (1, 2, 5):
                t = m * 10 ** e
                if lo <= t <= hi:
                    out.append(t)
        return out
    span = hi - lo
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag)
    start = math.ceil(lo / step) * step
    return [start + i * step for i in range(int((hi - start) / step) + 1)]


def _quad(path, pts):
    path.moveTo(*pts[0])
    for p in pts[1:]:
        path.lineTo(*p)
    path.close()


def _signed_area(pts):
    a = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        a += x0 * y1 - x1 * y0
    return a


@dataclass
class ChartStyle:
    alpha: float = 1.0
    curtain: float = 1.0
    candles: float = 1.0
    line: float = 1.0
    grid: float = 1.0
    labels: float = 1.0
    depth: float = 1.0      # 0 flat .. 1 full extrusion


def draw_chart(canvas, cam, view, series, head, style=ChartStyle(), date_labels=None):
    """Draw bars [floor(x0) .. head] of `series` (head is a float bar position)."""
    a = style.alpha
    if a <= 0.003:
        return {}
    n = len(series)
    k_head, ho, hh, hl, hc, _ = series.bar_at(head)
    i0 = max(0, int(math.floor(view.x0)) - 1)
    idx = np.arange(i0, k_head + 1)
    o = series.o[idx].copy()
    hi = series.h[idx].copy()
    lo = series.l[idx].copy()
    c = series.c[idx].copy()
    o[-1], hi[-1], lo[-1], c[-1] = ho, hh, hl, hc
    xs = idx.astype(float)
    spacing_px = cam.pw / (view.x1 - view.x0)

    # ---------- grid ----------
    if style.grid > 0:
        gp = paint(with_alpha(WHITE, 0.055 * a * style.grid), "stroke", 1.2)
        for t in nice_ticks(view.y0, view.y1, 5, view.log):
            vv = float(view.v(t))
            if 0.0 <= vv <= 1.0:
                (xa, xb), (ya, yb), _ = cam.project(np.array([-0.02, 1.02]), np.array([vv, vv]))
                canvas.drawLine(xa, ya, xb, yb, gp)
                if style.labels > 0:
                    lx, ly, _ = cam.project(np.array([0.0]), np.array([vv]))
                    lbl = f"{t:,.0f}" if t >= 10 else f"{t:g}"
                    gfx.text(canvas, lbl, float(lx[0]) + 8, float(ly[0]) - 6, gfx.font("num_m", 22),
                             GREY, alpha=0.5 * a * style.labels)
        if date_labels:
            last_x = -1e9
            for xi, lbl in date_labels:
                uu = float(view.u(xi))
                if -0.02 <= uu <= 1.02:
                    sxp = float(cam.project(np.array([uu]), np.array([1.0]))[0][0])
                    if sxp - last_x < 64:      # keep labels from piling up when zoomed out
                        continue
                    last_x = sxp
                    (xa, xb), (ya, yb), _ = cam.project(np.array([uu, uu]), np.array([0.0, 1.0]))
                    canvas.drawLine(xa, ya, xb, yb, gp)
                    if style.labels > 0:
                        gfx.text(canvas, lbl, xb + 6, yb + 26, gfx.font("num_m", 22), GREY,
                                 alpha=0.45 * a * style.labels)

    u = view.u(xs)
    vo, vc, vh, vl = view.v(o), view.v(c), view.v(hi), view.v(lo)
    up = c >= o
    vis = (u > -0.05) & (u < 1.05)

    # ---------- curtains (translucent columns from close down to the floor) ----------
    if style.curtain > 0 and vis.any():
        bw = min(1.0, 0.5 + 6.0 / max(spacing_px, 1e-6)) / (view.x1 - view.x0)
        ul, ur = u - bw / 2, u + bw / 2
        sx_l, sy_t, _ = cam.project(ul, vc)
        sx_r, _, _ = cam.project(ur, vc)
        bx_l, by_b, _ = cam.project(ul, np.full_like(vc, -0.02))
        bx_r, by_br, _ = cam.project(ur, np.full_like(vc, -0.02))
        sx_rt, sy_rt, _ = cam.project(ur, vc)
        pos, cols = [], []
        for j in np.nonzero(vis)[0]:
            col = RED if up[j] else GREEN
            top = with_alpha(col, 0.20 * a * style.curtain)
            bot = with_alpha(col, 0.0)
            p = [(sx_l[j], sy_t[j]), (sx_rt[j], sy_rt[j]), (bx_r[j], by_br[j]), (bx_l[j], by_b[j])]
            for tri in ((0, 1, 2), (0, 2, 3)):
                for q in tri:
                    pos.append(skia.Point(*p[q]))
                    cols.append((top if q < 2 else bot).toColor())
        if pos:
            verts = skia.Vertices.MakeCopy(skia.Vertices.kTriangles_VertexMode, pos, None, cols)
            canvas.drawVertices(verts, skia.Paint(AntiAlias=True), skia.BlendMode.kDst)

    # ---------- candles ----------
    if style.candles > 0 and vis.any():
        bw = 0.62 / (view.x1 - view.x0)
        thick = spacing_px * 0.62 > 7
        depth = min(spacing_px * 0.62, 26.0) * style.depth if thick else 0.0
        order = np.nonzero(vis)[0]
        if cam.yaw < 0:
            order = order[::-1]
        wick_up, wick_dn = skia.Path(), skia.Path()
        for j in order:
            wx, wy, _ = cam.project(np.array([u[j], u[j]]), np.array([vh[j], vl[j]]))
            (wick_up if up[j] else wick_dn).moveTo(wx[0], wy[0])
            (wick_up if up[j] else wick_dn).lineTo(wx[1], wy[1])
        wpw = max(1.2, min(3.0, spacing_px * 0.12))
        canvas.drawPath(wick_up, paint(with_alpha(RED, 0.95 * a * style.candles), "stroke", wpw))
        canvas.drawPath(wick_dn, paint(with_alpha(GREEN, 0.95 * a * style.candles), "stroke", wpw))

        for j in order:
            col = RED if up[j] else GREEN
            b0, b1 = min(vo[j], vc[j]), max(vo[j], vc[j])
            if (b1 - b0) * cam.ph < 2.0:
                b1 = b0 + 2.0 / cam.ph
            ul_, ur_ = u[j] - bw / 2, u[j] + bw / 2
            if not thick:
                xs_, ys_, _ = cam.project(np.array([ul_, ur_, ur_, ul_]), np.array([b0, b0, b1, b1]))
                pth = skia.Path()
                _quad(pth, list(zip(xs_, ys_)))
                canvas.drawPath(pth, paint(with_alpha(col, 0.95 * a * style.candles)))
                continue
            zf, zb = -depth / 2, depth / 2
            cu = np.array([ul_, ur_, ur_, ul_, ul_, ur_, ur_, ul_])
            cv = np.array([b0, b0, b1, b1, b0, b0, b1, b1])
            cz = np.array([zf, zf, zf, zf, zb, zb, zb, zb])
            px, py, _ = cam.project(cu, cv, cz)
            P = list(zip(px, py))
            faces = {"front": [0, 1, 2, 3], "right": [1, 5, 6, 2], "left": [4, 0, 3, 7],
                     "top": [3, 2, 6, 7], "bottom": [4, 5, 1, 0]}
            shade = {"front": 0.42, "right": 0.26, "left": 0.26, "top": 0.55, "bottom": 0.2}
            edge = skia.Path()
            for name, ids in faces.items():
                pts = [P[i] for i in ids]
                if _signed_area(pts) >= 0 and name != "front":
                    continue
                pth = skia.Path()
                _quad(pth, pts)
                canvas.drawPath(pth, paint(with_alpha(col, shade[name] * a * style.candles)))
                _quad(edge, pts)
            canvas.drawPath(edge, paint(with_alpha(gfx.mix(col, WHITE, 0.25), 0.9 * a * style.candles),
                                        "stroke", 1.6))

    # ---------- close line ----------
    if style.line > 0 and len(idx) > 1:
        lx, ly, _ = cam.project(u, vc)
        p_up, p_dn = skia.Path(), skia.Path()
        for j in range(1, len(idx)):
            if not (vis[j] or vis[j - 1]):
                continue
            pth = p_up if c[j] >= c[j - 1] else p_dn
            pth.moveTo(lx[j - 1], ly[j - 1])
            pth.lineTo(lx[j], ly[j])
        lw = 3.0 if spacing_px > 3 else 2.4
        canvas.drawPath(p_up, paint(with_alpha(gfx.RED_HI, 0.9 * a * style.line), "stroke", lw))
        canvas.drawPath(p_dn, paint(with_alpha(gfx.GREEN_HI, 0.9 * a * style.line), "stroke", lw))

    hx, hy, hs = cam.project(np.array([u[-1]]), np.array([vc[-1]]))
    return {"head_xy": (float(hx[0]), float(hy[0])), "head_price": float(c[-1]),
            "head_up": bool(c[-1] >= (c[-2] if len(c) > 1 else o[-1])), "k_head": k_head}


def draw_range_box(canvas, cam, view, i_start, p_start, i_head, p_head, alpha=1.0, label=None):
    """Translucent box spanning a chapter's move, with a dashed baseline and a % label."""
    if alpha <= 0.003:
        return
    col = RED if p_head >= p_start else GREEN
    ua, ub = float(view.u(i_start)), float(view.u(i_head))
    va, vb = float(view.v(p_start)), float(view.v(p_head))
    xs, ys, _ = cam.project(np.array([ua, ub, ub, ua]), np.array([va, va, vb, vb]))
    pth = skia.Path()
    _quad(pth, list(zip(xs, ys)))
    canvas.drawPath(pth, paint(with_alpha(col, 0.10 * alpha)))
    canvas.drawPath(pth, paint(with_alpha(col, 0.45 * alpha), "stroke", 1.6))
    dash = paint(with_alpha(WHITE, 0.35 * alpha), "stroke", 1.6)
    dash.setPathEffect(skia.DashPathEffect.Make([10, 8], 0))
    canvas.drawLine(xs[0], ys[0], xs[1], ys[1], dash)
    if label:
        pct, days = label
        f_big, f_small = gfx.font("num_xb", 46), gfx.font("cjk_med", 22)
        top = min(ys[1], ys[2])
        x_right = xs[2] - 6
        s_days = f"{days} 天"
        wd = gfx.text_width(s_days, f_small)
        gfx.text(canvas, s_days, x_right, top - 14, f_small, GREY, align="right", alpha=alpha)
        gfx.text(canvas, gfx.fmt_pct(pct), x_right - wd - 10, top - 12, f_big, col, align="right",
                 glow=10, alpha=alpha)


def draw_head(canvas, xy, up, t, alpha=1.0, size=1.0):
    if alpha <= 0.003:
        return
    x, y = xy
    col = RED if up else GREEN
    pulse = 0.5 + 0.5 * math.sin(t * 7.0)
    canvas.drawCircle(x, y, 26 * size, paint(with_alpha(col, 0.22 * alpha), blur=14))
    canvas.drawCircle(x, y, (10 + 3 * pulse) * size, paint(with_alpha(col, 0.55 * alpha), blur=5))
    canvas.drawCircle(x, y, 5.5 * size, paint(with_alpha(WHITE, alpha)))
