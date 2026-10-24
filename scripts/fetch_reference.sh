#!/usr/bin/env bash
# Download the reference Bilibili video and dump frames for review (not committed).
#   scripts/fetch_reference.sh [url] [outdir]
set -euo pipefail
URL="${1:-https://b23.tv/eQ08aND}"
OUT="${2:-reference}"
mkdir -p "$OUT/frames"
yt-dlp --no-playlist -f "bv*[height<=1080]+ba/b" --merge-output-format mp4 \
       --write-info-json --write-subs --sub-langs all -o "$OUT/video.%(ext)s" "$URL"
ffmpeg -v error -y -i "$OUT/video.mp4" -vf "fps=1,scale=960:-1" "$OUT/frames/%04d.png"
# 4x3 contact sheets, one frame per second
ffmpeg -v error -y -i "$OUT/video.mp4" -vf "fps=1,scale=480:-1,tile=4x3" "$OUT/sheet_%03d.png"
ls "$OUT"
