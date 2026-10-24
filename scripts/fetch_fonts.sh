#!/usr/bin/env bash
# Download the open-licensed (SIL OFL) fonts the vertical video uses into assets/fonts/,
# resolving file URLs through the Google Fonts CSS API.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)/assets/fonts"
mkdir -p "$DIR"
fetch() {  # family-query  weight  output-name
  local url
  url=$(curl -sS "https://fonts.googleapis.com/css2?family=$1:wght@$2" | grep -o 'https://[^)]*\.ttf' | head -1)
  [ -s "$DIR/$3" ] || curl -sS -o "$DIR/$3" "$url"
}
fetch "Noto+Sans+SC" 900 NotoSansSC-Black.ttf
fetch "Noto+Sans+SC" 700 NotoSansSC-Bold.ttf
fetch "Noto+Sans+SC" 500 NotoSansSC-Medium.ttf
fetch "Barlow+Semi+Condensed" 800 BarlowSemiCondensed-ExtraBold.ttf
fetch "Barlow+Semi+Condensed" 700 BarlowSemiCondensed-Bold.ttf
fetch "Barlow+Semi+Condensed" 600 BarlowSemiCondensed-SemiBold.ttf
fetch "Barlow+Semi+Condensed" 500 BarlowSemiCondensed-Medium.ttf
ls -la "$DIR"
