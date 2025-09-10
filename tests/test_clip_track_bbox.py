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
"""Unit tests for ``clip_track_bbox`` utility."""

from types import SimpleNamespace
import logging
import pathlib
import runpy
import pytest

MODULE = runpy.run_path(pathlib.Path("tools/decoder-lite.py"))
clip_track_bbox = MODULE["clip_track_bbox"]
LOGGER = logging.getLogger(__name__)


def test_clip_track_bbox_tlbr_valid() -> None:
    """Track with ``tlbr`` inside bounds should be returned unchanged."""
    track = SimpleNamespace(tlbr=(10, 20, 30, 40), track_id=1, score=0.9)
    result = clip_track_bbox(track, (50, 60), 0, LOGGER)
    assert result == pytest.approx([10.0, 20.0, 20.0, 20.0])


def test_clip_track_bbox_clipping() -> None:
    """Track partially outside the frame should be clipped."""
    track = SimpleNamespace(tlbr=(-10, -10, 20, 20), track_id=2, score=0.8)
    result = clip_track_bbox(track, (50, 50), 0, LOGGER)
    assert result == pytest.approx([0.0, 0.0, 20.0, 20.0])


def test_clip_track_bbox_invalid() -> None:
    """Invalid box with negative size should return ``None``."""
    track = SimpleNamespace(tlwh=(30, 30, -5, 10), track_id=3, score=0.5)
    assert clip_track_bbox(track, (100, 100), 0, LOGGER) is None

