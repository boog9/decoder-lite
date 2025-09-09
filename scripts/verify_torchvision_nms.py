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
"""Verify that torchvision registers the CUDA NMS operator.

Example usage:
    python scripts/verify_torchvision_nms.py
    python scripts/verify_torchvision_nms.py --quiet

Example output:
    torch: 2.5.1+cu121 cuda: 12.1
    torchvision: 0.20.1
    has torchvision::nms: True
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def check_torchvision_nms() -> bool:
    """Check if ``torchvision::nms`` is available.

    Returns:
        bool: ``True`` if the operator exists, ``False`` otherwise.
    """
    try:
        import torch
        import torchvision

        return hasattr(torch.ops.torchvision, "nms")
    except Exception:
        return False


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line interface for the verifier.

    Args:
        argv: Optional argument vector.

    Returns:
        int: Exit code (0 if available, 1 if missing, 2 on import error).
    """
    parser = argparse.ArgumentParser(
        description="Verify availability of torchvision::nms"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the boolean result and suppress version info.",
    )
    args = parser.parse_args(argv)

    try:
        import torch
        import torchvision
    except Exception as exc:  # pragma: no cover - exercised in manual runs
        if not args.quiet:
            print(f"verify error: {exc}")
        return 2

    has_nms = hasattr(torch.ops.torchvision, "nms")
    if args.quiet:
        print(has_nms)
    else:
        print("torch:", torch.__version__, "cuda:", torch.version.cuda)
        print("torchvision:", torchvision.__version__)
        print("has torchvision::nms:", has_nms)
    return 0 if has_nms else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
