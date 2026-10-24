#!/usr/bin/env python3
"""Render the vertical (1080x1920) TSLA story video in the style of a 3D candlestick explainer.

    python3 make_vertical.py --preview 1 10 30     # PNG stills into output/vpreview/
    python3 make_vertical.py                        # full render -> output/tsla_vertical.mp4
    python3 make_vertical.py --no-audio
"""
import argparse
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
_R = {}


def _init(fps):
    from tsla_video import scene
    _R["r"] = scene.Renderer(fps)


def _chunk(args):
    start, stop, path, fps = args
    r = _R["r"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1920",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "24",
           "-pix_fmt", "yuv420p", "-g", str(fps * 2), path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(start, stop):
        proc.stdin.write(np.ascontiguousarray(r.render(f)).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed on {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(ROOT, "output", "tsla_vertical.mp4"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--preview", type=float, nargs="*")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--start", type=float, default=0.0, help="render only from this time (s)")
    ap.add_argument("--end", type=float, default=None, help="render only up to this time (s)")
    args = ap.parse_args()

    from tsla_video import scene, story
    if args.preview is not None:
        import skia
        r = scene.Renderer(args.fps)
        os.makedirs(os.path.join(ROOT, "output", "vpreview"), exist_ok=True)
        for t in args.preview:
            arr = r.render(int(round(t * args.fps)))
            img = skia.Image.fromarray(np.ascontiguousarray(np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])),
                                       colorType=skia.kRGBA_8888_ColorType)
            p = os.path.join(ROOT, "output", "vpreview", f"v_{t:06.2f}.png")
            img.save(p, skia.kPNG)
            print(p)
        return

    end = args.end if args.end is not None else story.END
    f0, f1 = int(round(args.start * args.fps)), int(round(end * args.fps))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"frames {f0}..{f1} ({(f1 - f0) / args.fps:.1f}s)", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        k = args.workers * 4
        b = np.linspace(f0, f1, k + 1).astype(int)
        jobs = [(b[i], b[i + 1], os.path.join(tmp, f"s{i:03d}.mp4"), args.fps) for i in range(k)]
        with Pool(args.workers, initializer=_init, initargs=(args.fps,)) as pool:
            for n, _ in enumerate(pool.imap(_chunk, jobs), 1):
                print(f"  segment {n}/{k}", file=sys.stderr)
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w") as fh:
            fh.writelines(f"file '{j[2]}'\n" for j in jobs)
        video = os.path.join(tmp, "video.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                        video], check=True)
        if args.no_audio or args.start or args.end:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-c", "copy", "-movflags", "+faststart",
                            args.out], check=True)
        else:
            import soundtrack
            wav = os.path.join(tmp, "audio.wav")
            soundtrack.write(wav)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", wav, "-c:v", "copy",
                            "-af", "loudnorm=I=-15:TP=-2:LRA=11,alimiter=limit=0.7:level=disabled", "-ar", "48000", "-c:a", "aac",
                            "-b:a", "192k", "-shortest", "-movflags", "+faststart", args.out], check=True)
    print(args.out)


if __name__ == "__main__":
    main()
