#!/usr/bin/env python3
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

"""Environment diagnostics for decoder-lite.

Example usage:
    python scripts/doctor.py

Example output:
    python: 3.13.0 (main, ...)
    platform: Linux-6.5.0-...-x86_64-with-glibc2.35
    torch: 2.6.0 cuda: 12.1 avail: True
    yolox ok: True
    onnxruntime: 1.22.0 providers: ['CUDAExecutionProvider']
"""

from __future__ import annotations

import importlib
import platform
import sys


def _print(msg: str) -> None:
    """Print a message to stdout."""

    print(msg)


def main() -> None:
    """Print versions of critical libraries.

    Reports Python, platform, torch with CUDA, YOLOX, and ONNX Runtime
    information. Any import failures are caught and reported.
    """

    _print(f"python: {sys.version}")
    _print(f"platform: {platform.platform()}")
    try:
        import torch  # type: ignore

        _print(
            f"torch: {torch.__version__} cuda: {torch.version.cuda} "
            f"avail: {torch.cuda.is_available()}"
        )
    except Exception as exc:  # pragma: no cover - diagnostic
        _print(f"torch import failed: {exc}")

    try:
        yolox = importlib.import_module("yolox")
        _print(f"yolox ok: {hasattr(yolox, '__version__')}")
    except Exception as exc:  # pragma: no cover - diagnostic
        _print(f"yolox import failed: {exc}")

    try:
        import onnxruntime as ort  # type: ignore

        _print(
            f"onnxruntime: {ort.__version__} providers: {ort.get_available_providers()}"
        )
    except Exception as exc:  # pragma: no cover - diagnostic
        _print(f"onnxruntime import failed: {exc}")


if __name__ == "__main__":
    main()

