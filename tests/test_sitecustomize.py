# Copyright 2024 decoder-lite contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for :mod:`sitecustomize` NumPy alias restoration."""

from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np


def test_numpy_aliases_restored() -> None:
    """Ensure legacy NumPy aliases are present after importing sitecustomize."""
    # Ensure repository root is on sys.path so `sitecustomize` can be imported
    # even if Python started elsewhere (e.g., when invoking pytest).
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    importlib.import_module("sitecustomize")

    assert hasattr(np, "float")
    assert hasattr(np, "int")
    assert hasattr(np, "bool")
    assert hasattr(np, "object")
