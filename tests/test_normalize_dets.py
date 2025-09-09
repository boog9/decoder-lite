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

from __future__ import annotations

import importlib.util
import logging
import pathlib

import numpy as np
import pytest

def _load_decoder_tool():
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "tools" / "decoder-lite.py"
    spec = importlib.util.spec_from_file_location("decoder_lite_tool", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

def test_filters_6col():
    mod = _load_decoder_tool()
    dets = np.array([
        [0, 0, 10, 10, 0.90, 0],
        [1, 1, 11, 11, 0.80, 1],
        [2, 2, 12, 12, 0.70, 1],
    ], dtype=np.float32)
    out = mod.normalize_dets(dets, keep_classes=[1])
    assert out.shape[1] == 5
    assert out.shape[0] == 2
    np.testing.assert_allclose(out[:, 4], [0.80, 0.70], rtol=1e-6, atol=1e-6)

def test_filters_7col():
    mod = _load_decoder_tool()
    dets = np.array([
        [0, 0, 10, 10, 0.95, 0.90, 0],
        [1, 1, 11, 11, 0.85, 0.80, 1],
        [2, 2, 12, 12, 0.75, 0.70, 0],
    ], dtype=np.float32)
    out = mod.normalize_dets(dets, keep_classes=[0])
    assert out.shape[1] == 5
    assert out.shape[0] == 2
    np.testing.assert_allclose(out[:, 4], [0.95, 0.75], rtol=1e-6, atol=1e-6)

def test_warns_5col(caplog: pytest.LogCaptureFixture):
    mod = _load_decoder_tool()
    dets = np.array([
        [0, 0, 10, 10, 0.9],
        [1, 1, 11, 11, 0.8],
    ], dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        out = mod.normalize_dets(dets, keep_classes=[0])
    assert out.shape == (2, 5)
    assert any("--keep-classes" in rec.message and "no class column" in rec.message for rec in caplog.records)
