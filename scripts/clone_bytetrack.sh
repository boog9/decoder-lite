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
#
# Clone ByteTrack sources into third_party/ByteTrack.
# Example usage:
#   bash scripts/clone_bytetrack.sh
# Example output:
#   [clone_bytetrack] OK.

set -euo pipefail

dst="third_party/ByteTrack"
if [ -f "${dst}/yolox/__init__.py" ]; then
  echo "[clone_bytetrack] ByteTrack already present."
  exit 0
fi

echo "[clone_bytetrack] Cloning ByteTrack sources…"
rm -rf "${dst}"
git clone --depth=1 https://github.com/FoundationVision/ByteTrack "${dst}"

if [ ! -f "${dst}/yolox/__init__.py" ]; then
  echo "[clone_bytetrack] ERROR: yolox package not found after clone."
  exit 1
fi
echo "[clone_bytetrack] OK."

