#!/usr/bin/env python3
"""Synthesize the vertical video's soundtrack (numpy only, royalty-free), synced to the storyboard.

Music: A-minor electronic bed at 120 BPM whose drums, bass and harmony follow the chapters.
SFX: impacts on lightning/shatter, boom + shimmer on record highs, risers, tape rewind, clock ticks.

    python3 soundtrack.py out.wav
"""
import sys
import wave

import numpy as np

from tsla_video import story
from tsla_video.scene import FX

SR = 44100
BEAT = 0.5
DUR = story.END
N = int(DUR * SR) + SR
T = np.arange(N) / SR
RNG = np.random.default_rng(1)

PROG_MAIN = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]     # Am F C G
PROG_DARK = [(57, 60, 64), (57, 60, 64), (53, 57, 60), (52, 56, 59)]     # Am Am F E
PROG_LIFT = [(53, 57, 60), (55, 59, 62), (57, 60, 64), (48, 52, 55)]     # F G Am C
DARK = [(20.0, 24.8), (31.0, 35.0), (44.2, 48.2), (54.4, 70.2)]
LIFT = [(75.8, 84.2)]

# drum pattern per section: "four" (four on the floor), "half" (half time), None
PATTERN = [(0.0, None), (5.2, "half"), (8.8, "four"), (20.0, "half"), (24.8, "four"), (31.0, "half"),
           (35.0, "four"), (44.2, "half"), (48.2, "four"), (54.4, "half"), (57.6, None), (60.05, "half"),
           (67.4, None), (70.2, "four"), (75.8, "four"), (84.2, None)]


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def env(points):
    ts, vs = zip(*points)
    return np.interp(T, ts, vs)


def in_ranges(t, ranges):
    return any(a <= t < b for a, b in ranges)


def pattern_at(t):
    cur = None
    for t0, p in PATTERN:
        if t >= t0:
            cur = p
    return cur


def chord_at(t):
    prog = PROG_DARK if in_ranges(t, DARK) else PROG_LIFT if in_ranges(t, LIFT) else PROG_MAIN
    return prog[int(t // 2.0) % 4]


def place(buf, t0, sig, gain=1.0):
    i = int(t0 * SR)
    if i >= len(buf) or i + len(sig) <= 0:
        return
    if i < 0:
        sig, i = sig[-i:], 0
    sig = sig[: len(buf) - i]
    buf[i:i + len(sig)] += gain * sig


def saw(f, t, harm=12, tilt=1.5):
    out = np.zeros_like(t)
    for n in range(1, harm + 1):
        if f * n > 9000:
            break
        out += np.sin(2 * np.pi * f * n * t) / n ** tilt
    return out


def adsr(n, a, r):
    e = np.ones(n)
    ai, ri = int(a * SR), int(r * SR)
    e[:ai] = np.linspace(0, 1, max(ai, 1))[:ai]
    e[n - ri:] *= np.linspace(1, 0, ri)
    return e


def band_noise(n, lo, hi, seed=0):
    x = np.random.default_rng(seed).standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    y = np.fft.irfft(X, n)
    return y / (np.abs(y).max() + 1e-9)


# --------------------------------------------------------------------------- instruments

def kick():
    t = np.arange(int(0.45 * SR)) / SR
    ph = 2 * np.pi * (44 * t + 85 / 16 * (1 - np.exp(-16 * t)))
    return np.sin(ph) * np.exp(-t * 7.5)


def clap():
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    return band_noise(n, 900, 6000, 3) * np.exp(-t * 22) * 0.8 + np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30) * 0.4


def hat(open_=False):
    n = int((0.18 if open_ else 0.05) * SR)
    t = np.arange(n) / SR
    return band_noise(n, 7000, 16000, 5) * np.exp(-t * (18 if open_ else 80))


def impact(big=1.0):
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    sub = np.sin(2 * np.pi * (34 * t + 70 / 5 * (1 - np.exp(-5 * t)))) * np.exp(-t * 1.6)
    crack = band_noise(n, 80, 9000, 9) * np.exp(-t * 6)
    tail = band_noise(n, 100, 2500, 10) * np.exp(-t * 1.2) * 0.25
    return (sub * 1.0 + crack * 0.55 * big + tail) * big


def boom_shimmer():
    n = int(3.2 * SR)
    t = np.arange(n) / SR
    sub = np.sin(2 * np.pi * (40 * t + 50 / 6 * (1 - np.exp(-6 * t)))) * np.exp(-t * 2.2)
    sh = np.zeros(n)
    for m in (81, 84, 88, 93, 96):
        sh += np.sin(2 * np.pi * hz(m) * t + np.sin(2 * np.pi * 6 * t)) * np.exp(-t * 1.4)
    sparkle = band_noise(n, 6000, 15000, 12) * np.exp(-t * 2.5) * 0.3
    return sub * 0.9 + sh * 0.12 + sparkle


def riser(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    k = t / dur
    sweep = np.sin(2 * np.pi * (200 * t + 900 * t * k / 2)) * 0.15
    noise = band_noise(n, 2000, 12000, 14) * 0.5
    return (sweep + noise) * k ** 2.2


def whoosh(dur=0.35, seed=20):
    n = int(dur * SR)
    t = np.arange(n) / SR
    k = t / dur
    return band_noise(n, 400, 7000, seed) * np.sin(np.pi * k) ** 2


def rewind(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    k = t / dur
    f = 1800 * np.exp(-2.2 * k) + 120
    chirp = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.25
    warble = band_noise(n, 300, 5000, 22) * (0.5 + 0.5 * np.sin(2 * np.pi * 14 * t))
    return (chirp + warble * 0.5) * np.sin(np.pi * k) ** 0.5


def tick():
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * 2200 * t) * np.exp(-t * 90)


# --------------------------------------------------------------------------- arrangement

def write(path):
    L, R = np.zeros(N), np.zeros(N)
    drums, bass, pad_l, pad_r, arp, sfx = (np.zeros(N) for _ in range(6))

    # harmony: pads per 2-second chord block
    block = 2.0
    tb = np.arange(int((block + 1.0) * SR)) / SR
    eb = adsr(len(tb), 0.25, 1.0)
    for b in range(int(DUR / block) + 1):
        t0 = b * block
        ch = chord_at(t0 + 0.01)
        dark = in_ranges(t0 + 0.01, DARK)
        harm = 5 if dark else 9
        sl = sum(saw(hz(m) * 0.996, tb, harm) for m in ch) + 0.7 * saw(hz(ch[0] - 12), tb, 4)
        sr_ = sum(saw(hz(m) * 1.004, tb, harm) for m in ch) + 0.7 * saw(hz(ch[0] - 12), tb, 4)
        place(pad_l, t0, sl * eb)
        place(pad_r, t0, sr_ * eb)

    k_s, c_s, h_s, ho_s = kick(), clap(), hat(), hat(True)
    note = np.arange(int(BEAT / 2 * SR)) / SR
    pl = np.exp(-note * 9) * adsr(len(note), 0.004, 0.02)
    a16 = np.arange(int(BEAT / 4 * SR)) / SR
    ae = np.exp(-a16 * 16) * adsr(len(a16), 0.003, 0.01)
    beat = 0
    while beat * BEAT < DUR:
        t0 = beat * BEAT
        pat = pattern_at(t0)
        ch = chord_at(t0)
        if pat == "four":
            place(drums, t0, k_s)
            place(drums, t0 + BEAT / 2, h_s, 0.35)
            if beat % 2 == 1:
                place(drums, t0, c_s, 0.55)
            if beat % 4 == 3:
                place(drums, t0 + BEAT * 0.75, h_s, 0.25)
            for e in range(2):
                place(bass, t0 + e * BEAT / 2, saw(hz(ch[0] - 24), note, 6, 1.1) * pl)
        elif pat == "half":
            if beat % 4 == 0:
                place(drums, t0, k_s)
            if beat % 4 == 2:
                place(drums, t0, c_s, 0.6)
            place(drums, t0 + BEAT / 2, h_s, 0.18)
            if beat % 2 == 0:
                place(bass, t0, saw(hz(ch[0] - 24), np.arange(int(BEAT * SR)) / SR, 6, 1.1)
                      * np.exp(-np.arange(int(BEAT * SR)) / SR * 3))
        if pat is not None or 84.2 <= t0 < DUR or t0 < 3.7:
            tones = [ch[1] + 12, ch[2] + 12, ch[0] + 24, ch[2] + 12]
            for s, m in enumerate(tones):
                place(arp, t0 + s * BEAT / 4, np.sin(2 * np.pi * hz(m) * a16) * ae)
        beat += 1

    # sfx on the storyboard's effect times
    for tf, kind, target, extra in FX:
        if kind in ("lightning", "shatter"):
            place(sfx, tf, impact(1.0 if kind == "lightning" else 0.8), 0.9)
        elif kind == "fireworks":
            place(sfx, tf, boom_shimmer(), 0.9)
            place(sfx, tf - 2.0, riser(2.0), 0.5)
        elif kind == "rewind":
            place(sfx, tf, rewind(extra), 0.8)
    # bell melody for the aftermath and the outro
    bt = np.arange(int(1.6 * SR)) / SR
    bell_env = np.exp(-bt * 2.6) * adsr(len(bt), 0.005, 0.2)
    for start, stop in ((67.4, 70.2), (84.2, DUR - 1.0)):
        k = 0
        while start + k * BEAT < stop:
            t0 = start + k * BEAT
            ch = chord_at(t0)
            m = [ch[0] + 12, ch[2] + 12, ch[1] + 24, ch[2] + 12][k % 4]
            tone = np.sin(2 * np.pi * hz(m) * bt) + 0.3 * np.sin(2 * np.pi * hz(m) * 2.01 * bt)
            place(arp, t0, tone * bell_env * 0.9)
            k += 1
    place(sfx, 57.7, riser(2.35), 0.6)
    place(sfx, 73.8, riser(2.0), 0.45)
    for k in range(5):
        place(sfx, 57.9 + k * 0.5, tick(), 0.5)
    for p in story.PANELS:
        place(sfx, p.t0, whoosh(0.32, seed=int(p.t0 * 10)), 0.18)

    # level automation per stem
    g_pad = env([(0, 0.55), (3.6, 0.45), (3.8, 0.15), (5.2, 0.45), (20, 0.4), (24.8, 0.6), (31, 0.5), (35, 0.55),
                 (54.4, 0.45), (57.6, 0.35), (60.05, 0.55), (67.4, 0.8), (70.2, 0.55), (75.8, 0.85), (84.2, 1.0),
                 (DUR - 3, 0.9), (DUR, 0.0)])
    g_drum = env([(0, 0), (5.2, 0.55), (8.8, 0.75), (24.8, 1.0), (31, 0.8), (35, 0.95), (57.6, 0.95), (57.7, 0),
                  (60.0, 0), (60.05, 1.0), (75.8, 1.0), (84.2, 0.9), (84.3, 0), (DUR, 0)])
    g_arp = env([(0, 0.25), (3.6, 0.25), (3.8, 0.0), (5.2, 0.35), (20, 0.2), (24.8, 0.35), (67.4, 0.8),
                 (70.2, 0.35), (84.2, 0.9), (DUR - 2, 0.8), (DUR, 0)])
    pad_gain = 0.05
    mix_m = 0.5 * drums * g_drum + 0.22 * bass * g_drum + 0.06 * arp * g_arp + 0.75 * sfx
    L = pad_l * pad_gain * g_pad + mix_m
    R = pad_r * pad_gain * g_pad + mix_m

    # light room reverb on everything (FFT convolution with a decaying noise IR)
    ir_n = int(1.1 * SR)
    ir = RNG.standard_normal(ir_n) * np.exp(-np.arange(ir_n) / SR * 4.5)
    ir[0] = 0
    ir /= np.sqrt((ir ** 2).sum())
    size = 1 << int(np.ceil(np.log2(N + ir_n)))
    IR = np.fft.rfft(ir, size)
    L = L + 0.22 * np.fft.irfft(np.fft.rfft(L, size) * IR, size)[:N]
    R = R + 0.22 * np.fft.irfft(np.fft.rfft(R, size) * IR, size)[:N]

    st = np.stack([L, R], axis=1)[: int(DUR * SR)]
    fade = np.ones(len(st))
    fo = int(2.0 * SR)
    fade[-fo:] = np.linspace(1, 0, fo) ** 1.5
    st *= fade[:, None]
    st = np.tanh(st * 1.1)
    st *= 0.89 / (np.abs(st).max() + 1e-9)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    write(sys.argv[1] if len(sys.argv) > 1 else "soundtrack.wav")
