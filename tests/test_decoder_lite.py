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
import json
import pathlib
from typing import Any

import pytest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

SPEC = importlib.util.spec_from_file_location(
    "decoder_lite", pathlib.Path("tools/decoder-lite.py")
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
parse_keep_classes = MODULE.parse_keep_classes
filter_by_classes = MODULE.filter_by_classes
FpsEMA = MODULE.FpsEMA
call_with_supported_kwargs = MODULE.call_with_supported_kwargs


def test_make_parser_fp16_flag() -> None:
    parser = MODULE.make_parser()
    args = parser.parse_args(["-f", "exp.py", "-c", "weights.pth", "--fp16"])
    assert args.fp16 is True


def test_save_raw_flag() -> None:
    parser = MODULE.make_parser()
    args = parser.parse_args(["-f", "exp.py", "-c", "weights.pth", "--save-raw"])
    assert args.save_raw is True


def test_parse_keep_classes() -> None:
    assert parse_keep_classes("0,32") == [0, 32]
    assert parse_keep_classes("") is None
    assert parse_keep_classes(None) is None
    assert parse_keep_classes("ALL") is None


@pytest.mark.skipif(np is None, reason="numpy not available")
def test_filter_by_classes() -> None:
    dets = np.array([
        [0, 0, 1, 1, 0.9, 0],
        [0, 0, 1, 1, 0.8, 1],
        [0, 0, 1, 1, 0.7, 32],
    ])
    keep = {0, 32}
    filtered = filter_by_classes(dets, keep)
    assert filtered.shape[0] == 2
    assert set(filtered[:, 5].astype(int)) == {0, 32}


@pytest.mark.skipif(np is None, reason="numpy not available")
def test_filter_by_classes_empty() -> None:
    dets = np.empty((0, 6), dtype=np.float32)
    keep = {0}
    out = filter_by_classes(dets, keep)
    assert out.shape == (0, 5)


def test_fps_ema() -> None:
    meter = FpsEMA(alpha=0.5)
    assert meter.update(0.2) == 5.0
    assert meter.update(0.1) == pytest.approx(7.5)
    # Negative or zero dt should keep previous FPS
    assert meter.update(0.0) == pytest.approx(7.5)


def test_predictor_mean_std_defaults() -> None:
    class DummyExp:
        num_classes = 1
        test_size = (640, 640)

    exp = DummyExp()
    predictor = MODULE.Predictor(
        model=object(),
        exp=exp,
        postprocess_fn=lambda *args, **kwargs: None,
        preproc_fn=lambda *args, **kwargs: (args[0], 1.0),
        device="cpu",
        fp16=True,
    )
    assert predictor.mean == (0.485, 0.456, 0.406)
    assert predictor.std == (0.229, 0.224, 0.225)
    assert predictor._use_fp16 is False


@pytest.mark.parametrize(
    "variant", ["class_agnostic", "agnostic", "positional5", "positional4"]
)
def test_inference_postprocess_variants(variant: str) -> None:
    torch = pytest.importorskip("torch")
    import numpy as np

    class DummyExp:
        num_classes = 1
        test_size = (2, 2)

    class DummyModel:
        def __call__(self, inp: torch.Tensor) -> str:
            return "out"

    record = {}

    def make_pp(name: str):
        if name == "class_agnostic":
            def pp(pred, num_classes, conf, nms, class_agnostic=False):
                record["args"] = (pred, num_classes, conf, nms, class_agnostic)
                return pred

            return pp
        if name == "agnostic":
            def pp(pred, num_classes, conf, nms, agnostic=False):
                record["args"] = (pred, num_classes, conf, nms, agnostic)
                return pred

            return pp
        if name == "positional5":
            def pp(pred, num_classes, conf, nms, flag):
                record["args"] = (pred, num_classes, conf, nms, flag)
                return pred

            return pp

        def pp(pred, num_classes, conf, nms):
            record["args"] = (pred, num_classes, conf, nms)
            return pred

        return pp

    predictor = MODULE.Predictor(
        model=DummyModel(),
        exp=DummyExp(),
        postprocess_fn=make_pp(variant),
        preproc_fn=lambda img, *args: (img, 1.0),
        device="cpu",
    )
    img = np.zeros((2, 2, 3), dtype=np.float32)
    out, _ = predictor.inference(img, 0.5, 0.6)
    assert out == "out"
    args = record["args"]
    assert args[1] == 1
    assert args[2] == 0.5
    assert args[3] == 0.6
    if variant == "positional4":
        assert len(args) == 4
    else:
        assert args[4] is True


def test_call_with_supported_kwargs_respects_signature() -> None:
    """Ensure helper filters unsupported keyword arguments."""
    record: dict[str, Any] = {}

    def fn_with_cls(img, tlwhs, ids, *, scores=None, frame_id=0, fps=0.0, cls_ids=None):
        record["with_cls"] = {
            "scores": scores,
            "frame_id": frame_id,
            "fps": fps,
            "cls_ids": cls_ids,
        }
        return "img"

    def fn_without_cls(img, tlwhs, ids, *, scores=None, frame_id=0, fps=0.0):
        record["without_cls"] = {
            "scores": scores,
            "frame_id": frame_id,
            "fps": fps,
        }
        return "img"

    out = call_with_supported_kwargs(
        fn_with_cls,
        "img",
        [],
        [],
        scores=[0.1],
        frame_id=1,
        fps=2.0,
        cls_ids=[3],
    )
    assert out == "img"
    assert record["with_cls"] == {
        "scores": [0.1],
        "frame_id": 1,
        "fps": 2.0,
        "cls_ids": [3],
    }

    out = call_with_supported_kwargs(
        fn_without_cls,
        "img",
        [],
        [],
        scores=[0.1],
        frame_id=1,
        fps=2.0,
        cls_ids=[3],
    )
    assert out == "img"
    assert record["without_cls"] == {
        "scores": [0.1],
        "frame_id": 1,
        "fps": 2.0,
    }


def test_json_default_handles_numpy_and_torch() -> None:
    np = pytest.importorskip("numpy")
    assert MODULE._json_default(np.array([1, 2])) == [1, 2]
    assert MODULE._json_default(np.float32(3.0)) == 3.0
    dump = json.dumps({"x": np.array([1])}, default=MODULE._json_default)
    assert json.loads(dump)["x"] == [1]
    if MODULE.torch is not None:
        import torch

        dump = json.dumps({"t": torch.tensor([1, 2])}, default=MODULE._json_default)
        assert json.loads(dump)["t"] == [1, 2]


def test_track_coordinate_postprocess() -> None:
    """Coordinates from tracker are converted and clipped correctly."""

    class Dummy:
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    w, h = 100, 80
    frame_id = 1
    online_targets = [
        Dummy(tlbr=[-5, -5, 50, 50], track_id=1, score=0.9),
        Dummy(tlwh=[90, 70, 20, 20], track_id=2, score=0.8),
    ]
    tlwhs: list[list[float]] = []
    online_ids: list[int] = []
    online_scores: list[float] = []
    online_cls_ids: list[int] = []
    logger = MODULE.logger
    for t in online_targets:
        if hasattr(t, "tlbr"):
            x1, y1, x2, y2 = map(float, t.tlbr)
        else:
            x1, y1, w_, h_ = map(float, t.tlwh)
            x2 = x1 + w_
            y2 = y1 + h_
        if frame_id <= 3:
            logger.info(
                f"  Track ID {getattr(t, 'track_id', '?')}: Raw coords: x1={x1:.1f}, "
                f"y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}"
            )
            logger.info(f"  Image dimensions: {w}x{h}")
        if x2 <= x1 or y2 <= y1:
            if frame_id <= 3:
                logger.warning(
                    f"  Invalid bbox dimensions: w={x2 - x1:.1f}, h={y2 - y1:.1f}"
                )
            continue
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(w), x2)
        y2 = min(float(h), y2)
        w_box = x2 - x1
        h_box = y2 - y1
        if w_box <= 0 or h_box <= 0:
            if frame_id <= 3:
                logger.warning(
                    f"  Bbox eliminated after clipping: w={w_box:.1f}, h={h_box:.1f}"
                )
            continue
        tlwh = [x1, y1, w_box, h_box]
        if frame_id <= 3:
            logger.info(
                f"  Final tlwh: [{x1:.1f}, {y1:.1f}, {w_box:.1f}, {h_box:.1f}]"
            )
            logger.info(
                f"  Position as percentage: x={x1 / w * 100:.1f}%, y={y1 / h * 100:.1f}%"
            )
        tlwhs.append(tlwh)
        online_ids.append(int(t.track_id))
        online_scores.append(float(t.score))
        online_cls_ids.append(int(getattr(t, "cls", -1)))

    assert tlwhs == [[0.0, 0.0, 50.0, 50.0], [90.0, 70.0, 10.0, 10.0]]
    assert online_ids == [1, 2]
    assert online_scores == [0.9, 0.8]
    assert online_cls_ids == [-1, -1]
