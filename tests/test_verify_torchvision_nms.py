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
"""Tests for :mod:`scripts.verify_torchvision_nms`."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    """Dynamically load the verification module."""
    spec = importlib.util.spec_from_file_location(
        "verify_torchvision_nms", Path("scripts/verify_torchvision_nms.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # for type checker
    spec.loader.exec_module(module)
    return module


def test_check_torchvision_nms_returns_bool() -> None:
    """Ensure the check function returns a boolean value."""
    module = _load_module()
    assert isinstance(module.check_torchvision_nms(), bool)
