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

# Ensure the local ByteTrack vendor tree is present and complete.
# Example usage:
#   bash scripts/ensure_bytetrack.sh
# Example output:
#   drwxr-xr-x  14 user user 4096 Sep  9 13:49 .
#   ...

set -euo pipefail

BT_DIR="third_party/ByteTrack"
BT_REPO_URL="https://github.com/FoundationVision/ByteTrack.git"
# Pin to a stable commit from the official ByteTrack repository.
BT_COMMIT="d1bf0191adff59bc8fcfeaa0b33d3d1642552a99"

if [[ ! -f "${BT_DIR}/yolox/__init__.py" ]]; then
  echo "[ensure_bytetrack] ByteTrack yolox package missing; syncing from upstream." >&2
  tmp_clone="$(mktemp -d)"
  tmp_preserve="$(mktemp -d)"

  # Preserve local bookkeeping files if they exist.
  for f in .gitkeep LICENSE LICENSE.*; do
    if [[ -f "${BT_DIR}/${f}" ]]; then
      cp "${BT_DIR}/${f}" "${tmp_preserve}/";
    fi
  done

  rm -rf "${BT_DIR}"
  git clone "${BT_REPO_URL}" "${tmp_clone}/ByteTrack"
  git -C "${tmp_clone}/ByteTrack" checkout "${BT_COMMIT}"
  mkdir -p "$(dirname "${BT_DIR}")"
  mv "${tmp_clone}/ByteTrack" "${BT_DIR}"

  # Restore preserved files.
  for f in "${tmp_preserve}"/*; do
    [[ -e "$f" ]] || continue
    cp "$f" "${BT_DIR}/"
  done

  rm -rf "${tmp_clone}" "${tmp_preserve}"
fi

ls -la "${BT_DIR}/yolox" | head
# Verify that setup.py exists for develop installation.
test -f "${BT_DIR}/setup.py"
