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

# Clone ByteTrack into third_party/ByteTrack if missing.
set -euo pipefail

REPO_DIR="third_party/ByteTrack"
# Офіційна інструкція радить клонувати ifzhang/ByteTrack (requirements.txt у корені).
# Джерело: README ByteTrack — Step1. Install ByteTrack. (git clone https://github.com/ifzhang/ByteTrack.git)
BYTETRACK_URL="${BYTETRACK_URL:-https://github.com/ifzhang/ByteTrack.git}"
BYTETRACK_REV="${BYTETRACK_REV:-}"

mkdir -p third_party
if [[ -f "${REPO_DIR}/requirements.txt" ]]; then
  echo "[clone_bytetrack] Found existing ${REPO_DIR} (requirements.txt present) — skipping clone."
  exit 0
fi

if [[ -d "${REPO_DIR}" ]]; then
  echo "[clone_bytetrack] ${REPO_DIR} exists but no requirements.txt — removing and recloning..."
  rm -rf "${REPO_DIR}"
fi

echo "[clone_bytetrack] Cloning ${BYTETRACK_URL} into ${REPO_DIR}..."
git clone --depth 1 "${BYTETRACK_URL}" "${REPO_DIR}"
if [[ -n "${BYTETRACK_REV}" ]]; then
  echo "[clone_bytetrack] Checking out revision ${BYTETRACK_REV}..."
  git -C "${REPO_DIR}" fetch --depth 1 origin "${BYTETRACK_REV}"
  git -C "${REPO_DIR}" checkout -q "${BYTETRACK_REV}"
fi

if [[ ! -f "${REPO_DIR}/requirements.txt" ]]; then
  echo "[clone_bytetrack] ERROR: requirements.txt not found in ${REPO_DIR}. Repository layout unexpected."
  exit 1
fi

echo "[clone_bytetrack] Done."
