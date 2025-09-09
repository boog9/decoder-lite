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

YOLOX_URL="${YOLOX_URL:-https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0rc0/yolox_x.pth}"
DEST_DIR="third_party/ByteTrack/pretrained"
mkdir -p "${DEST_DIR}"

if [[ ! -f "${DEST_DIR}/yolox_x.pth" ]]; then
  wget -L -O "${DEST_DIR}/yolox_x.pth" "${YOLOX_URL}"
fi
