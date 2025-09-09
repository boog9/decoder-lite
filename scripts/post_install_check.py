#!/usr/bin/env python
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

"""Verify that ONNX Runtime uses GPU execution on supported platforms.

This script ensures that the CUDA execution provider is available on
Linux and Windows hosts when a GPU is accessible. The check can be
skipped by setting ``ALLOW_CPU_ORT=1`` or when no NVIDIA GPU is detected,
such as on macOS or CPU-only CI environments. It prints the ONNX
Runtime version and the available providers.

Example:
    Run the check after installing dependencies::

        python scripts/post_install_check.py

Example output::

        ONNX Runtime OK: 1.22.1 ['CUDAExecutionProvider', 'CPUExecutionProvider']
"""

from __future__ import annotations

import os
import platform
import shutil
import sys


def main() -> None:
    """Run the post-installation check.

    Raises:
        AssertionError: If CUDAExecutionProvider is missing on Linux/Windows.
        ImportError: If onnxruntime cannot be imported.
    """
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover
        raise ImportError(f"Could not import onnxruntime: {exc}") from exc

    providers = ort.get_available_providers()
    cpu_mode = (
        os.getenv("ALLOW_CPU_ORT") == "1"
        or platform.system() == "Darwin"
        or shutil.which("nvidia-smi") is None
    )
    if not cpu_mode and platform.system() in ("Linux", "Windows"):
        assert "CUDAExecutionProvider" in providers, (
            f"CUDAExecutionProvider not available: {providers}"
        )
    else:
        assert "CPUExecutionProvider" in providers, (
            f"CPUExecutionProvider not available: {providers}"
        )
    print(f"ONNX Runtime OK: {ort.__version__} {providers}")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:  # pragma: no cover - simple CLI
        print(f"[post_install_check] {err}", file=sys.stderr)
        sys.exit(1)
