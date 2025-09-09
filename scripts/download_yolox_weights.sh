#!/usr/bin/env bash
# Copyright 2024
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Usage:
#   YOLOX_VARIANT=x make weights
#   YOLOX_URL=<custom> make weights
#   (by default yolox_x.pth is downloaded)

DEST_DIR="third_party/ByteTrack/pretrained"
mkdir -p "${DEST_DIR}"

VARIANT="${YOLOX_VARIANT:-x}"   # x|l|m|s|tiny|nano
FN="yolox_${VARIANT}.pth"
OUT="${DEST_DIR}/${FN}"

# Minimum file size thresholds for basic integrity check (bytes)
case "${VARIANT}" in
  x)   MIN_BYTES=$((600*1024*1024));;
  l)   MIN_BYTES=$((140*1024*1024));;
  m)   MIN_BYTES=$((70*1024*1024));;
  s)   MIN_BYTES=$((60*1024*1024));;
  tiny) MIN_BYTES=$((35*1024*1024));;
  nano) MIN_BYTES=$((5*1024*1024));;
  *) echo "[weights] ERROR: unknown YOLOX_VARIANT='${VARIANT}'"; exit 1;;
esac

# Sources: (1) custom URL (2) SourceForge mirror (verified)
URLS=()
if [[ -n "${YOLOX_URL:-}" ]]; then
  URLS+=("${YOLOX_URL}")
fi
URLS+=("https://sourceforge.net/projects/yolox.mirror/files/0.1.0/${FN}/download")

download_ok=false
for U in "${URLS[@]}"; do
  echo "[weights] Trying: ${U}"
  if wget -q -L -O "${OUT}.tmp" "${U}" || curl -fL --output "${OUT}.tmp" "${U}"; then
    if [[ -f "${OUT}.tmp" ]]; then
      SIZE=$(stat -c%s "${OUT}.tmp" 2>/dev/null || echo 0)
      if [[ "${SIZE}" -ge "${MIN_BYTES}" ]]; then
        mv -f "${OUT}.tmp" "${OUT}"
        echo "[weights] Saved ${OUT} (${SIZE} bytes)"
        download_ok=true
        break
      else
        echo "[weights] Downloaded file too small (${SIZE} < ${MIN_BYTES}), trying next URL..."
        rm -f "${OUT}.tmp"
      fi
    fi
  else
    echo "[weights] Failed to download from ${U}, trying next..."
    rm -f "${OUT}.tmp" || true
  fi
done

if [[ "${download_ok}" != "true" ]]; then
  echo "[weights] ERROR: could not download ${FN}. Try setting YOLOX_URL to a working mirror."
  echo "Example (YOLOX-s): YOLOX_VARIANT=s make weights"
  exit 1
fi
