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

# mood -> (drum pattern, progression, drum level, pad level, arp level)
MOOD = {
    "tension": (None, PROG_DARK, 0.0, 0.55, 0.25),
    "rewind": (None, PROG_DARK, 0.0, 0.15, 0.0),
    "calm": ("half", PROG_MAIN, 0.55, 0.45, 0.35),
    "up": ("four", PROG_MAIN, 0.9, 0.55, 0.35),
    "down": ("half", PROG_DARK, 0.75, 0.5, 0.2),
    "crash": ("half", PROG_DARK, 1.0, 0.55, 0.15),
    "suspense": (None, PROG_DARK, 0.0, 0.35, 0.1),
    "aftermath": (None, PROG_DARK, 0.0, 0.8, 0.8),
    "lift": ("four", PROG_LIFT, 1.0, 0.85, 0.35),
    "explain": ("half", PROG_LIFT, 0.45, 0.7, 0.5),
    "outro": (None, PROG_MAIN, 0.0, 1.0, 0.9),
}
CARDS = [(c.t0, c.t1) for c in story.CARDS]


def mood_at(t):
    cur = "tension"
    for t0, t1, m in story.MOODS:
        if t0 <= t < t1:
            cur = m
    return cur


def in_card(t):
    return any(a <= t < b for a, b in CARDS)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def pattern_at(t):
    pat = MOOD[mood_at(t)][0]
    if pat == "four" and in_card(t):
        return "half"            # calmer groove while an explainer card is being read
    return pat


def chord_at(t):
    return MOOD[mood_at(t)][1][int(t // 2.0) % 4]


def level_env(idx, card_gain=1.0, ramp=0.5):
    """Per-sample level curve for one stem from the mood table, with short ramps between sections."""
    pts = []
    for t0, t1, m in sorted(story.MOODS):
        lvl = MOOD[m][idx]
        cuts = sorted({t0, t1} | {x for a, b in CARDS for x in (a, b) if t0 < x < t1})
        for a, b in zip(cuts, cuts[1:]):
            g = lvl * (card_gain if in_card((a + b) / 2) else 1.0)
            pts += [(a + ramp / 2, g), (b - ramp / 2, g)]
    pts.sort()
    ts, vs = zip(*pts)
    return np.interp(T, ts, vs)


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
        harm = 5 if MOOD[mood_at(t0 + 0.01)][1] is PROG_DARK else 9
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
        if pat is not None or mood_at(t0) in ("tension", "outro"):
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
    bells = [(a, b) for a, b, m in story.MOODS if m in ("aftermath", "outro")]
    for start, stop in bells:
        stop = min(stop, DUR - 1.0)
        k = 0
        while start + k * BEAT < stop:
            t0 = start + k * BEAT
            ch = chord_at(t0)
            m = [ch[0] + 12, ch[2] + 12, ch[1] + 24, ch[2] + 12][k % 4]
            tone = np.sin(2 * np.pi * hz(m) * bt) + 0.3 * np.sin(2 * np.pi * hz(m) * 2.01 * bt)
            place(arp, t0, tone * bell_env * 0.9)
            k += 1
    M = story.M
    place(sfx, M["gap"] - 2.4, riser(2.35), 0.6)              # build into the earnings gap-down
    place(sfx, M["recap"] - 2.0, riser(2.0), 0.45)
    k = 0
    while M["crash"] + 0.4 + k * 0.5 < M["gap"] - 0.2:        # clock ticks while the earnings card is read
        place(sfx, M["crash"] + 0.4 + k * 0.5, tick(), 0.45)
        k += 1
    for p in story.PANELS:
        place(sfx, p.t0, whoosh(0.32, seed=int(p.t0 * 10)), 0.18)
    for c in story.CARDS:
        place(sfx, c.t0, whoosh(0.45, seed=int(c.t0 * 7)), 0.25)
    for name in ("hold", "analog", "street", "dca", "verdict"):
        place(sfx, M[f"concl_{name}"], whoosh(0.45, seed=len(name)), 0.25)

    # level automation per stem
    g_drum = level_env(2, card_gain=0.6)
    g_pad = level_env(3, card_gain=1.1)
    g_arp = level_env(4, card_gain=1.2)
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
