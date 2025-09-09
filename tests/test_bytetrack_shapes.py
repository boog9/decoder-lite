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
import pathlib
from types import SimpleNamespace

import pytest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

SPEC = importlib.util.spec_from_file_location(
    "decoder_lite", pathlib.Path("tools/decoder-lite.py"),
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
normalize_dets = MODULE.normalize_dets


class DummyTracker:
    """Minimal tracker capturing input shapes."""

    def __init__(self) -> None:
        self.last_shape: tuple[int, int] | None = None
        self.calls: int = 0

    def update(self, dets: np.ndarray, *_args, **_kwargs):
        self.calls += 1
        self.last_shape = dets.shape
        return [SimpleNamespace(tlwh=[0, 0, 1, 1], track_id=1, score=0.5)]


@pytest.mark.skipif(np is None, reason="numpy not available")
@pytest.mark.parametrize("cols", [7, 5])
def test_bytetrack_shapes(cols: int) -> None:
    dets = np.zeros((1, cols), dtype=float)
    if cols >= 7:
        dets[0, 4:7] = [0.9, 0.8, 0]
    else:
        dets[0, 4] = 0.9
    img_info = {"ratio": 1.0, "height": 100, "width": 100}

    dets_in, cls_col = normalize_dets(dets, img_info, {0, 32})
    tracker = DummyTracker()
    tracker.update(dets_in, [img_info["height"], img_info["width"]], (640, 640))
    assert tracker.calls == 1
    assert tracker.last_shape is not None
    assert tracker.last_shape[1] == 5
    if cols >= 7:
        assert cls_col is not None and cls_col.size == dets_in.shape[0]
    else:
        assert cls_col is None
