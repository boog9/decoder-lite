#!/usr/bin/env python3
"""ByteTrack demo with class filtering.

This script is a thin wrapper around the official ByteTrack demo. It adds the
ability to keep only a subset of COCO classes before feeding detections to the
tracker. By default, only classes 0 (person) and 32 (sports ball) are kept.

Example:
    python tools/decoder-lite.py \
        -f third_party/ByteTrack/exps/default/yolox_x.py \
        -c third_party/ByteTrack/pretrained/yolox_x.pth \
        --path path/to/video.mp4 --save_result --device gpu --fp16 \
        --keep-classes 0,32
"""

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

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("decoder-lite")
_logger = logging.getLogger(__name__)

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    from loguru import logger as loguru_logger
    logger = loguru_logger
except ModuleNotFoundError:  # pragma: no cover
    # Optional: можна лишити порожньою або залогувати через вже ініціалізований logger
    logger.warning(
        "Optional dependencies are missing; limited functionality may apply."
    )


def parse_keep(arg: str) -> List[int]:
    """Parse comma-separated list of class ids.

    Args:
        arg: Comma-separated class id string.

    Returns:
        Sorted list of unique integers.
    """
    cleaned = arg.replace(" ", "")
    if not cleaned:
        return []
    vals = {int(x) for x in cleaned.split(",") if x}
    return sorted(vals)


def filter_by_classes(dets: np.ndarray, keep: Iterable[int]) -> np.ndarray:
    """Filter detection array by class ids.

    Args:
        dets: Detection array ``[N, M]`` with integer class ids in the last
            column (requires ``M >= 6``).
        keep: Iterable of class ids to keep.

    Returns:
        Filtered detection array.
    """
    if np is None:
        raise ModuleNotFoundError("numpy is required for filter_by_classes")
    if dets.size == 0:
        return dets
    if dets.shape[1] < 6:
        return dets
    keep_set: Set[int] = set(keep)
    cls_col = dets.shape[1] - 1
    cls_ids = dets[:, cls_col].astype(int)
    mask = np.isin(cls_ids, list(keep_set))
    return dets[mask]


def normalize_dets(
    dets: np.ndarray,
    keep_classes: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Normalize detection tensor to ByteTrack-compatible shape (N,5).

    Args:
        dets: Detection array with shape (N,>=5).
        keep_classes: Optional sequence of class IDs to retain.

    Returns:
        Array of shape (M,5) containing <x1,y1,x2,y2,score>.

    Raises:
        ValueError: If ``dets`` has fewer than five columns.
    """

    if dets is None or dets.size == 0:
        return np.empty((0, 5), dtype=np.float32)

    if dets.ndim != 2 or dets.shape[1] < 5:
        raise ValueError(
            f"Expected dets with shape (N,>=5), got {getattr(dets, 'shape', None)}"
        )

    num_cols = dets.shape[1]
    cls_col: Optional[int] = None
    if num_cols >= 7:
        cls_col = 6
    elif num_cols == 6:
        cls_col = 5

    if keep_classes:
        if cls_col is not None:
            cls_ids = dets[:, cls_col].astype(np.int64, copy=False)
            mask = np.isin(
                cls_ids, np.asarray(list(keep_classes), dtype=np.int64)
            )
            dets = dets[mask]
        else:
            _logger.warning(
                "--keep-classes provided but inputs have no class column (5-col detections). Ignoring."
            )

    dets5 = dets[:, :5].astype(np.float32, copy=False)
    return dets5


def make_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser("decoder-lite ByteTrack demo")
    parser.add_argument("-f", "--exp_file", type=str, required=True,
                        help="Experiment description file from ByteTrack.")
    parser.add_argument("-c", "--ckpt", type=str, required=True,
                        help="Checkpoint path for the detector.")
    parser.add_argument("--path", type=str, default="0",
                        help="Video path or webcam id (default: 0).")
    parser.add_argument("--save_result", action="store_true",
                        help="Save annotated video and JSON results.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "gpu"], help="Inference device.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision on GPU.")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold.")
    parser.add_argument("--nms", type=float, default=0.45,
                        help="NMS threshold.")
    parser.add_argument("--tsize", type=int, default=None,
                        help="Test image size.")
    parser.add_argument("--track_thresh", type=float, default=0.5,
                        help="Tracking confidence threshold.")
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="Track buffer length.")
    parser.add_argument("--match_thresh", type=float, default=0.8,
                        help="Matching threshold for tracker.")
    parser.add_argument("--min_box_area", type=float, default=10,
                        help="Minimum box area.")
    parser.add_argument("--mot20", action="store_true",
                        help="Test on MOT20 dataset.")
    parser.add_argument(
        "--keep-classes",
        type=str,
        default="0,32",
        help=(
            "COCO class ids to keep, comma-separated. 5-col inputs lack a class "
            "column, so this flag is ignored with a warning."
        ),
    )
    parser.add_argument("--fuse", action="store_true",
                        help="Fuse conv+bn for faster inference (GPU only).")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without OpenCV GUI windows.")
    return parser


def main() -> None:
    """Entry point for the demo."""
    args = make_parser().parse_args()
    keep_classes = set(parse_keep(args.keep_classes))
    logger.info(f"Keeping classes: {sorted(keep_classes)}")

    # Heavy imports are done lazily to keep unit tests lightweight.
    from yolox.exp import get_exp
    from yolox.utils import fuse_model, get_model_info, postprocess
    from yolox.data.data_augment import preproc
    from yolox.utils.visualize import plot_tracking
    try:
        # Старі гілки YOLOX мали Timer у yolox.utils; якщо є — використовуємо.
        from yolox.utils import Timer  # type: ignore
    except Exception:
        # Фолбек-сумісність: простий таймер з tic()/toc() для логування/замірів.
        class Timer:
            def __init__(self) -> None:
                self._t0: Optional[float] = None

            def tic(self) -> None:
                self._t0 = time.perf_counter()

            def toc(self) -> float:
                t0 = self._t0 or time.perf_counter()
                return time.perf_counter() - t0
    from yolox.tracker.byte_tracker import BYTETracker
    import cv2
    import torch

    exp = get_exp(args.exp_file, None)
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    model = exp.get_model()
    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()
    model.eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if args.device == "gpu" and args.fuse:
        model = fuse_model(model)
    logger.info(f"Model Summary: {get_model_info(model, exp.test_size)}")

    predictor = Predictor(model, exp, postprocess, preproc, args.device, args.fp16)
    tracker = BYTETracker(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        mot20=args.mot20,
    )

    if args.path.isdigit():
        path: str | int = int(args.path)
    else:
        path = args.path
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"Failed to open {args.path}"

    save_dir = Path("outputs/videos")
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "result.mp4"
    json_path = Path("outputs/logs") / "result.json" if args.save_result else None
    writer = None
    records: List[dict] = []

    frame_id = 0
    timer = Timer()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Lazy init writer once we know actual frame size/FPS
        if args.save_result and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        frame_id += 1
        timer.tic()
        outputs, img_info = predictor.inference(frame, args.conf, args.nms)
        if outputs[0] is not None:
            dets = outputs[0].cpu().numpy()
            dets[:, :4] /= img_info["ratio"]
            dets_before = dets.shape[0]
            if dets_before > 0:
                dets_in = normalize_dets(dets, keep_classes)
                logger.info(
                    f"Frame {frame_id}: kept {dets_in.shape[0]}/{dets_before} detections (after class filter and rescale)"
                )
                if dets_in.shape[0] > 0:
                    online_targets = tracker.update(
                        dets_in,
                        [img_info["height"], img_info["width"]],
                        exp.test_size,
                    )

                    online_tlwhs: List[List[float]] = []
                    online_ids: List[int] = []
                    online_scores: List[float] = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        if tlwh[2] * tlwh[3] > args.min_box_area:
                            online_tlwhs.append(tlwh)
                            online_ids.append(t.track_id)
                            online_scores.append(t.score)
                    timer.toc()
                    online_im = plot_tracking(
                        img_info["raw_img"],
                        online_tlwhs,
                        online_ids,
                        frame_id=frame_id,
                        fps=1.0 / timer.average_time,
                        scores=online_scores,
                        cls_ids=None,
                    )
                    if args.save_result:
                        writer.write(online_im)
                        records.append(
                            {
                                "frame": frame_id,
                                "tlwh": [
                                    list(map(float, tlwh)) for tlwh in online_tlwhs
                                ],
                                "id": online_ids,
                                "score": online_scores,
                            }
                        )
                    if not args.no_display:
                        cv2.imshow("ByteTrack", online_im)
                else:
                    timer.toc()
                    blank = img_info["raw_img"]
                    if args.save_result:
                        writer.write(blank)
                    if not args.no_display:
                        cv2.imshow("ByteTrack", blank)
            else:
                timer.toc()
                blank = img_info["raw_img"]
                if args.save_result:
                    writer.write(blank)
                if not args.no_display:
                    cv2.imshow("ByteTrack", blank)
        else:
            timer.toc()
            blank = img_info["raw_img"]
            if args.save_result:
                writer.write(blank)
            if not args.no_display:
                cv2.imshow("ByteTrack", blank)
        if not args.no_display and cv2.waitKey(1) == 27:
            break
    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    if args.save_result and json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(records, f)
        logger.info(f"Results saved to {out_path} and {json_path}")


class Predictor:
    """Run YOLOX model for a single frame."""

    def __init__(
        self,
        model,
        exp,
        postprocess_fn,
        preproc_fn,
        device: str = "cpu",
        fp16: bool = False,
    ) -> None:
        self.model = model
        self.exp = exp
        self.postprocess = postprocess_fn
        self.preproc = preproc_fn
        self.device = device
        self.fp16 = fp16
        self.mean = exp.rgb_means
        self.std = exp.std
        self.num_classes = exp.num_classes
        self.test_size = exp.test_size

    def inference(self, img, conf: float, nms: float):
        import torch
        img_info = {"raw_img": img, "height": img.shape[0], "width": img.shape[1]}
        img, ratio = self.preproc(img, self.test_size, self.mean, self.std)
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()
        with torch.no_grad():
            outputs = self.model(img)
            outputs = self.postprocess(
                outputs, self.num_classes, conf, nms, class_agnostic=True
            )
        img_info["ratio"] = ratio
        return outputs, img_info


if __name__ == "__main__":
    main()
