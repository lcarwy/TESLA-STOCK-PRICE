#!/usr/bin/env python3
"""Synthesize a royalty-free background track (numpy only) for make_video.py.

A-minor loop (Am-F-C-G) at 100 BPM: warm pad throughout, with bass, kick, hats
and a soft arpeggio between `beat_in` and `beat_out`; pad-only intro/outro.

    python3 bgm.py out.wav 90
"""
import sys
import wave

import numpy as np

SR = 44100
BPM = 100
BEAT = 60 / BPM
BAR = 4 * BEAT
# chord tones as MIDI notes (Am, F, C, G), root first
CHORDS = [(57, 60, 64, 69), (53, 57, 60, 65), (48, 55, 60, 64), (55, 59, 62, 67)]


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def saw(freq, t, harmonics=10, tilt=1.6):
    out = np.zeros_like(t)
    for n in range(1, harmonics + 1):
        if freq * n > SR / 2.5:
            break
        out += np.sin(2 * np.pi * freq * n * t) / n ** tilt
    return out


def envelope(n, attack, release):
    env = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    env[:a] = np.linspace(0, 1, a)
    env[n - r:] *= np.linspace(1, 0, r)
    return env


def place(buf, start, sig):
    i = int(start * SR)
    if i >= len(buf):
        return
    sig = sig[: len(buf) - i]
    buf[i:i + len(sig)] += sig


def write(path, duration, beat_in=3.0, beat_out=None):
    beat_out = duration - 6 if beat_out is None else beat_out
    n = int(duration * SR) + SR
    pad_l, pad_r = np.zeros(n), np.zeros(n)
    drums, bass, arp = np.zeros(n), np.zeros(n), np.zeros(n)
    rng = np.random.default_rng(7)

    # pad: one chord per bar, overlapping releases, slightly detuned per channel
    bar_len = BAR + 1.2
    t = np.arange(int(bar_len * SR)) / SR
    env = envelope(len(t), 0.6, 1.2)
    for b in range(int(duration / BAR) + 1):
        chord = CHORDS[b % 4]
        sl = sum(saw(hz(m) * 0.997, t) for m in chord[1:]) + 0.6 * saw(hz(chord[0] - 12), t, 6)
        sr_ = sum(saw(hz(m) * 1.003, t) for m in chord[1:]) + 0.6 * saw(hz(chord[0] - 12), t, 6)
        place(pad_l, b * BAR, sl * env)
        place(pad_r, b * BAR, sr_ * env)

    kick_t = np.arange(int(0.35 * SR)) / SR
    kick = np.sin(2 * np.pi * (45 * kick_t + 65 / 18 * (1 - np.exp(-18 * kick_t)))) * np.exp(-kick_t * 9)
    hat_n = int(0.05 * SR)
    hat = np.diff(rng.standard_normal(hat_n + 1)) * np.exp(-np.arange(hat_n) / SR * 90)
    note_t = np.arange(int(BEAT / 2 * SR)) / SR
    pluck_env = np.exp(-note_t * 7) * envelope(len(note_t), 0.004, 0.02)
    arp_t = np.arange(int(BEAT / 4 * SR)) / SR
    arp_env = np.exp(-arp_t * 14) * envelope(len(arp_t), 0.003, 0.01)

    beat = 0
    while beat * BEAT < duration:
        start = beat * BEAT
        chord = CHORDS[(beat // 4) % 4]
        if beat_in <= start < beat_out:
            place(drums, start, kick)
            place(drums, start + BEAT / 2, 0.22 * hat)
            for e in range(2):  # pulsing eighth-note bass on the chord root
                place(bass, start + e * BEAT / 2, saw(hz(chord[0] - 12), note_t, 5, 1.2) * pluck_env)
        if beat_in + 2 * BAR <= start < beat_out + BAR:
            tones = [chord[1] + 12, chord[2] + 12, chord[3] + 12, chord[2] + 12]
            for s, m in enumerate(tones):
                place(arp, start + s * BEAT / 4, np.sin(2 * np.pi * hz(m) * arp_t) * arp_env)
        beat += 1

    # sidechain-style ducking of the pad on each kick
    duck = np.ones(n)
    tt = np.arange(int(BEAT * SR)) / SR
    shape = 1 - 0.45 * np.exp(-tt * 10)
    beat = 0
    while beat * BEAT < duration:
        if beat_in <= beat * BEAT < beat_out:
            i = int(beat * BEAT * SR)
            duck[i:i + len(shape)] = shape[: n - i]
        beat += 1

    pad_gain = 0.06
    left = pad_l * pad_gain * duck + 0.55 * drums + 0.20 * bass + 0.06 * arp
    right = pad_r * pad_gain * duck + 0.55 * drums + 0.20 * bass + 0.06 * arp
    stereo = np.stack([left, right], axis=1)[: int(duration * SR)]

    fade = np.ones(len(stereo))
    fi, fo = int(1.5 * SR), int(4.0 * SR)
    fade[:fi] = np.linspace(0, 1, fi)
    fade[-fo:] = np.linspace(1, 0, fo)
    stereo *= fade[:, None]
    stereo = np.tanh(stereo * 1.2)
    stereo *= 0.89 / np.abs(stereo).max()  # peak about -1 dBFS

    pcm = (stereo * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    write(sys.argv[1], float(sys.argv[2]))
