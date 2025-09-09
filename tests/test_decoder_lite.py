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
parse_keep = MODULE.parse_keep
filter_by_classes = MODULE.filter_by_classes


def test_make_parser_fp16_flag() -> None:
    parser = MODULE.make_parser()
    args = parser.parse_args(["-f", "exp.py", "-c", "weights.pth", "--fp16"])
    assert args.fp16 is True


def test_parse_keep() -> None:
    assert parse_keep("0,32") == [0, 32]
    assert parse_keep("32,0,0") == [0, 32]
    assert parse_keep("") == []


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
