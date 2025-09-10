#!/usr/bin/env python3
"""ByteTrack demo with class filtering.

This script is a thin wrapper around the official ByteTrack demo. It adds the
ability to keep only a subset of COCO classes before feeding detections to the
tracker. By default, only classes 0 (person) and 32 (sports ball) are kept.

Example:
    python tools/decoder-lite.py \
        -f third_party/ByteTrack/exps/default/yolox_x.py \
        -c third_party/ByteTrack/pretrained/yolox_x.pth \
        --path path/to/video.mp4 --save_result --device gpu \
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
import inspect
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Iterable, Optional, Set, List

import logging
import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # allow running without torch in serialization

# Attempt to import the canonical YOLOX preproc used by ByteTrack. The import is
# optional so that unit tests can run without the full dependency tree.
try:  # pragma: no cover - optional dependency
    from yolox.data.data_augment import preproc as yolox_preproc
except Exception:  # pragma: no cover
    yolox_preproc = None  # type: ignore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("decoder-lite")
_logger = logging.getLogger(__name__)

try:
    from loguru import logger as loguru_logger
    logger = loguru_logger
except ModuleNotFoundError:  # pragma: no cover
    # Optional: можна лишити порожньою або залогувати через вже ініціалізований logger
    logger.warning(
        "Optional dependencies are missing; limited functionality may apply."
    )


def _json_default(obj: Any) -> Any:
    """Serialize NumPy and torch objects for JSON output.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable representation.
    """

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


# --- Generic compat helper: call function with only supported kwargs ---
def call_with_supported_kwargs(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with only the kwargs it supports.

    Args:
        fn: Target function.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments to filter based on ``fn``'s signature.

    Returns:
        Result of calling ``fn`` with supported keyword arguments.
    """

    try:
        params = inspect.signature(fn).parameters
        filtered = {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        # If inspection fails, fall back to calling with no kwargs to avoid
        # raising unexpected errors.
        filtered = {}
    return fn(*args, **filtered)


class FpsEMA:
    """Exponential moving average FPS meter."""

    def __init__(self, alpha: float = 0.1) -> None:
        """Initialize the meter.

        Args:
            alpha: Smoothing factor in ``(0, 1]``.
        """
        self.alpha = alpha
        self._fps: Optional[float] = None

    def update(self, dt: float) -> float:
        """Update meter with frame duration.

        Args:
            dt: Frame processing time in seconds.

        Returns:
            Smoothed FPS value.
        """
        if dt is None or dt <= 0:
            return self._fps or 0.0
        inst = 1.0 / dt
        if self._fps is None:
            self._fps = inst
        else:
            self._fps = (1 - self.alpha) * self._fps + self.alpha * inst
        return self._fps


def parse_keep_classes(s: str | None) -> list[int] | None:
    """Parse a comma-separated string of class IDs.

    ``None`` or an empty string disables filtering and returns ``None``.

    Args:
        s: Comma-separated class ID string or ``None``.

    Returns:
        ``None`` if filtering should be disabled, otherwise a list of integers.
    """

    if s is None:
        return None
    s = s.strip()
    if s == "":
        # Empty string means no filtering (ALL classes kept).
        return None
    return [int(x) for x in s.split(",") if x.strip() != ""]




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
    if dets.size == 0 or dets.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float32)
    if dets.shape[1] < 6:
        return dets
    keep_set: Set[int] = set(keep)
    cls_col = dets.shape[1] - 1
    cls_ids = dets[:, cls_col].astype(int)
    mask = np.isin(cls_ids, list(keep_set))
    return dets[mask]


def normalize_dets(
    dets: np.ndarray,
    keep_classes: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize raw detector outputs and optionally filter by class.

    Supports YOLOX detection formats:

      * ``(N,5)``: ``[x1,y1,x2,y2,score]``
      * ``(N,6)``: ``[x1,y1,x2,y2,score,cls]``
      * ``(N,7)``: ``[x1,y1,x2,y2,obj_conf,cls_conf,cls]``

    The returned tuple contains the normalized detections ``dets_in`` of shape
    ``(M,5)`` and the integer class array ``cls_or_none`` (``(M,)``) if class
    information is available; otherwise ``None``.

    Class filtering is "soft" – if ``keep_classes`` filters out all detections in
    a frame, the filter is disabled for that frame and a warning is logged.

    Args:
        dets: Raw detector outputs.
        keep_classes: Optional set of class IDs to retain.

    Returns:
        Tuple of normalized detections and optional class IDs.

    Raises:
        ValueError: If ``dets`` has an unexpected shape.
    """

    if dets.size == 0:
        empty = np.zeros((0, 5), dtype=np.float32)
        return empty, None

    cols = dets.shape[1]

    if cols == 5:
        xyxy = dets[:, :4]
        score = dets[:, 4:5]
        if keep_classes:
            logger.warning(
                "Detector outputs 5 columns (no class). Ignoring --keep-classes."
            )
        dets_in = np.concatenate([xyxy, score], axis=1)
        return dets_in, None

    if cols == 6:
        xyxy = dets[:, :4]
        score = dets[:, 4:5]
        cls = dets[:, 5].astype(int)
        kept = np.ones(len(cls), dtype=bool)
        if keep_classes:
            kept = np.isin(cls, list(keep_classes))
            if not np.any(kept):
                logger.warning(
                    f"Class filter kept 0/{len(cls)} dets; disabling filter for this frame."
                )
                kept = np.ones(len(cls), dtype=bool)
        xyxy, score, cls = xyxy[kept], score[kept], cls[kept]
        dets_in = np.concatenate([xyxy, score], axis=1)
        return dets_in, cls

    if cols == 7:
        xyxy = dets[:, :4]
        obj = dets[:, 4]
        ccnf = dets[:, 5]
        score = (obj * ccnf).reshape(-1, 1)
        score = np.clip(score, 0.0, 1.0)
        cls = dets[:, 6].astype(int)
        kept = np.ones(len(cls), dtype=bool)
        if keep_classes:
            kept = np.isin(cls, list(keep_classes))
            if not np.any(kept):
                logger.warning(
                    f"Class filter kept 0/{len(cls)} dets; disabling filter for this frame."
                )
                kept = np.ones(len(cls), dtype=bool)
        xyxy, score, cls = xyxy[kept], score[kept], cls[kept]
        dets_in = np.concatenate([xyxy, score], axis=1)
        return dets_in, cls

    raise ValueError(f"Unexpected dets shape {dets.shape}; expected (N,5|6|7).")


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
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Inference device.",
    )
    # Flag kept for backward compatibility but ignored at runtime.
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="(ignored) FP16 disabled; FP32 is forced",
    )
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
        default=None,
        help=(
            "Comma-separated COCO class ids. None/empty => no filtering. "
            "5-col inputs lack a class column, so this flag is ignored with a "
            "warning."
        ),
    )
    parser.add_argument("--fuse", action="store_true",
                        help="Fuse conv+bn for faster inference (GPU only).")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without OpenCV GUI windows.")
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help=(
            "Write raw frames without overlays. By default with --save_result we "
            "write annotated frames (bbox/ID/score/fps)."
        ),
    )
    return parser


def main() -> None:
    """Entry point for the demo."""
    args = make_parser().parse_args()
    keep_classes = parse_keep_classes(args.keep_classes)
    logger.info(
        f"Keeping classes: {'ALL' if keep_classes is None else keep_classes}"
    )

    # Heavy imports are done lazily to keep unit tests lightweight.
    from yolox.exp import get_exp
    from yolox.utils import fuse_model, get_model_info, postprocess
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

    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    # Force strict FP32 end-to-end to avoid dtype mismatches.
    model = model.float()
    logger.info("Using FP32 inference (fp16 disabled).")
    try:
        logger.info(f"Model Summary: {get_model_info(model, exp.test_size)}")
    except Exception as e:
        logger.warning(f"Model profiling skipped: {e.__class__.__name__}: {e}")

    model.eval()
    if args.device == "gpu":
        model.cuda()
    if args.device == "gpu" and args.fuse:
        model = fuse_model(model)

    predictor = Predictor(
        model=model,
        exp=exp,
        postprocess_fn=postprocess,
        device=args.device,
        fp16=False,
    )
    logger.info(
        f"exp.test_size treated as (H,W)={getattr(exp, 'test_size', None)}"
    )
    # fps / frame_rate may already be derived from video or args
    frame_rate = getattr(args, "fps", None)
    if frame_rate is None or frame_rate <= 0:
        frame_rate = 30

    # --- ByteTrack API compatibility shim ---
    # Some versions accept named params; older ones expect a Namespace.
    tracker_kwargs = dict(
        track_thresh=getattr(args, "track_thresh", 0.5),
        match_thresh=getattr(args, "match_thresh", 0.8),
        track_buffer=getattr(args, "track_buffer", 30),
        aspect_ratio_thresh=getattr(args, "aspect_ratio_thresh", 1.6),
        min_box_area=getattr(args, "min_box_area", 10),
        mot20=getattr(args, "mot20", False),
    )

    try:
        # Attempt new API with keyword arguments.
        sig = inspect.signature(BYTETracker.__init__)
        params = sig.parameters
        if "frame_rate" in params:
            tracker = BYTETracker(frame_rate=frame_rate, **tracker_kwargs)
        else:
            tracker = BYTETracker(**tracker_kwargs)
    except TypeError:
        # Fallback to old API: BYTETracker(args_namespace, frame_rate=30)
        bt_args = SimpleNamespace(**tracker_kwargs)
        try:
            tracker = BYTETracker(bt_args, frame_rate=frame_rate)
        except TypeError:
            # Even older versions without frame_rate
            tracker = BYTETracker(bt_args)
    # --- end shim ---

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
    fps_meter = FpsEMA(alpha=0.2)
    _prev_ts = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.save_result and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        frame_id += 1
        timer.tic()
        outputs, img_info = predictor.inference(frame, args.conf, args.nms)

        tlwhs: List[List[float]] = []
        online_ids: List[int] = []
        online_scores: List[float] = []
        online_cls_ids: List[int] = []

        raw_im = img_info["raw_img"]
        h, w = raw_im.shape[:2]
        logger.debug(
            f"Frame {frame_id}: raw image shape: {raw_im.shape}, h={h}, w={w}"
        )
        if not isinstance(w, (int, float)) or not isinstance(h, (int, float)):
            logger.warning(f"Invalid image dimensions: w={w}, h={h}")
            continue
        ratio = float(img_info.get("ratio", 1.0))
        if not np.isfinite(ratio) or ratio <= 0:
            ratio = 1.0
        out_np = (
            outputs[0].cpu().numpy()
            if (outputs and outputs[0] is not None)
            else np.zeros((0, 7), dtype=np.float32)
        )
        num_raw = out_np.shape[0]
        if num_raw:
            bboxes_xyxy = out_np[:, :4].astype(np.float32, copy=True)
            if frame_id == 1:
                logger.info(
                    "Sample BEFORE descaling (xyxy) head: %s",
                    bboxes_xyxy[:3].tolist(),
                )
            dw, dh = (0.0, 0.0)
            if "pad" in img_info:
                dw, dh = img_info["pad"]
                bboxes_xyxy[:, [0, 2]] -= float(dw)
                bboxes_xyxy[:, [1, 3]] -= float(dh)
            bboxes_xyxy /= ratio
            bboxes_xyxy[:, 0::2] = np.clip(bboxes_xyxy[:, 0::2], 0, w - 1)
            bboxes_xyxy[:, 1::2] = np.clip(bboxes_xyxy[:, 1::2], 0, h - 1)
            if frame_id == 1:
                logger.info(
                    "Sample AFTER  descaling (xyxy) head: %s (ratio=%.6f, pad=(%.1f,%.1f))",
                    bboxes_xyxy[:3].tolist(),
                    ratio,
                    dw,
                    dh,
                )
                if ratio != 1.0:
                    assert not np.allclose(
                        out_np[:3, :4], bboxes_xyxy[:3, :4]
                    ), "Descaling failed: AFTER equals BEFORE while ratio != 1"
        else:
            bboxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        assert (bboxes_xyxy[:, 0] <= bboxes_xyxy[:, 2]).all() if bboxes_xyxy.size else True
        assert (bboxes_xyxy[:, 1] <= bboxes_xyxy[:, 3]).all() if bboxes_xyxy.size else True

        if num_raw:
            if out_np.shape[1] >= 7:
                scores = (out_np[:, 4] * out_np[:, 5]).astype(np.float32)
                cls_ids = out_np[:, 6].astype(np.int32)
            elif out_np.shape[1] == 6:
                scores = out_np[:, 4].astype(np.float32)
                cls_ids = out_np[:, 5].astype(np.int32)
            else:
                scores = out_np[:, 4].astype(np.float32)
                cls_ids = np.full((num_raw,), -1, np.int32)
        else:
            scores = np.empty((0,), dtype=np.float32)
            cls_ids = np.empty((0,), dtype=np.int32)

        if keep_classes is not None and cls_ids.size:
            mask = np.isin(cls_ids, keep_classes)
            bboxes_xyxy = bboxes_xyxy[mask]
            scores = scores[mask]
            cls_ids = cls_ids[mask]

        if bboxes_xyxy.size == 0:
            dets_for_tracker = np.zeros((0, 5), dtype=np.float32)
        else:
            dets_for_tracker = np.hstack([bboxes_xyxy, scores[:, None]]).astype(
                np.float32, copy=False
            )
        dets_c = np.ascontiguousarray(dets_for_tracker)
        logger.info(
            f"Frame {frame_id}: dets raw={num_raw} kept={len(bboxes_xyxy)} ratio={ratio:.6f}"
        )
        try:
            online_targets = tracker.update(dets_c, (h, w))
        except TypeError:
            in_h, in_w = map(int, getattr(exp, "test_size", (h, w)))
            online_targets = tracker.update(dets_c, (h, w), (in_h, in_w))
        for t in online_targets:
            if hasattr(t, "tlbr"):
                x1, y1, x2, y2 = map(float, t.tlbr)
            else:
                x1, y1, w_, h_ = map(float, t.tlwh)
                x2 = x1 + w_
                y2 = y1 + h_

            if x2 <= x1 or y2 <= y1:
                continue

            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(float(w), x2)
            y2 = min(float(h), y2)

            w_box = x2 - x1
            h_box = y2 - y1
            if w_box <= 0 or h_box <= 0:
                continue

            tlwh = [x1, y1, w_box, h_box]
            if frame_id == 1 and hasattr(t, "tlbr"):
                logger.info(f"Original tlbr: {t.tlbr}, converted tlwh: {tlwh}")
            tlwhs.append(tlwh)
            online_ids.append(int(t.track_id))
            online_scores.append(float(t.score))
            online_cls_ids.append(int(getattr(t, "cls", -1)))

        if frame_id <= 3 and tlwhs:
            logger.info(
                f"Frame {frame_id}: processed {len(tlwhs)} tracks. "
                f"First track tlwh: {tlwhs[0]} Image size: {w}x{h}"
            )

        if len(tlwhs) == 0 and dets_for_tracker.size > 0:
            logger.debug(
                f"No tracks after filter; dets present. Check min_box_area={args.min_box_area} / tracker thresholds."
            )

        _dt = timer.toc() if hasattr(timer, "toc") else None
        if _dt is None:
            _now = time.perf_counter()
            _dt = _now - _prev_ts
            _prev_ts = _now
        fps = float(fps_meter.update(_dt) or 0.0)

        draw_im = np.ascontiguousarray(raw_im.copy())
        if frame_id <= 3 and tlwhs:
            logger.info(f"Frame {frame_id} calling plot_tracking with:")
            logger.info(f"  Image shape: {draw_im.shape}")
            logger.info(f"  Number of tracks: {len(tlwhs)}")
            logger.info(f"  Sample tlwh: {tlwhs[0]}")
            logger.info(
                f"  Sample ID: {online_ids[0] if online_ids else 'None'}"
            )
            logger.info(
                f"  Sample score: {online_scores[0] if online_scores else 'None'}"
            )
        annotated = call_with_supported_kwargs(
            plot_tracking,
            draw_im,
            tlwhs,
            online_ids,
            scores=online_scores,
            frame_id=frame_id,
            fps=fps,
            cls_ids=(
                online_cls_ids if any(x >= 0 for x in online_cls_ids) else None
            ),
        )

        if args.save_result and not (
            len(tlwhs)
            == len(online_ids)
            == len(online_scores)
            == len(online_cls_ids)
        ):
            logger.warning(
                f"Lens mismatch: tlwhs={len(tlwhs)} ids={len(online_ids)} "
                f"scores={len(online_scores)} cls={len(online_cls_ids)}"
            )

        if not args.no_display:
            cv2.imshow("ByteTrack", annotated)
            if cv2.waitKey(1) == 27:
                break

        if args.save_result and writer is not None:
            writer.write(raw_im if args.save_raw else annotated)
            records.append(
                {
                    "frame": int(frame_id),
                    "tlwh": [[float(v) for v in tlwh] for tlwh in tlwhs],
                    "id": [int(i) for i in online_ids],
                    "score": [float(s) for s in online_scores],
                }
            )
    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    if args.save_result and json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, default=_json_default)
        logger.info(f"Results saved to {out_path} and {json_path}")


class Predictor:
    """Run YOLOX model for a single frame."""

    def __init__(
        self,
        model,
        exp,
        postprocess_fn,
        preproc_fn: Any | None = None,
        device: str = "cpu",
        fp16: bool = False,
    ) -> None:
        self.model = model
        self.exp = exp
        self.postprocess = postprocess_fn
        self.preproc = preproc_fn or yolox_preproc
        self.device = device

        # Use experiment-provided normalization values if available; otherwise fall back
        # to standard ImageNet statistics.
        rgb_means = getattr(exp, "rgb_means", None)
        if rgb_means is None:
            rgb_means = getattr(exp, "mean", None)
        if rgb_means is None:
            rgb_means = (0.485, 0.456, 0.406)
        self.mean = tuple(float(x) for x in rgb_means)

        std_vals = getattr(exp, "std", None)
        if std_vals is None:
            std_vals = getattr(exp, "rgb_stds", None)
        if std_vals is None:
            std_vals = (0.229, 0.224, 0.225)
        self.std = tuple(float(x) for x in std_vals)

        self.num_classes = exp.num_classes
        self.test_size = exp.test_size

        # Force FP32 regardless of the ``fp16`` flag.
        self._use_fp16 = False
        if fp16:
            pass

    def inference(self, img: np.ndarray, conf: float, nms: float) -> tuple[Any, dict]:
        """Run model inference and apply post-processing.

        Args:
            img: Input image array.
            conf: Confidence threshold.
            nms: Non-maximum suppression threshold.

        Returns:
            Tuple containing the processed outputs and image metadata.
        """

        import torch

        img_info: dict = {"raw_img": img}
        in_h, in_w = map(int, getattr(self.exp, "test_size", (640, 640)))
        if self.preproc is None:  # pragma: no cover - safety
            raise ModuleNotFoundError("yolox_preproc is unavailable")
        proc_img, ratio = self.preproc(img, (in_h, in_w), self.mean, self.std)
        img_info.update(
            {
                "ratio": float(ratio),
                "height": int(img.shape[0]),
                "width": int(img.shape[1]),
                "input_size": (int(in_h), int(in_w)),
            }
        )
        img_tensor = torch.from_numpy(proc_img).unsqueeze(0)
        if self.device == "gpu":
            img_tensor = img_tensor.cuda()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            # --- YOLOX/ByteTrack postprocess compatibility shim ---
            pp = getattr(self, "postprocess", None)
            if pp is None:
                try:
                    from yolox.utils import postprocess as pp  # type: ignore
                except Exception as e:  # pragma: no cover
                    raise RuntimeError("postprocess function not found") from e

            try:
                sig = inspect.signature(pp)
                names = list(sig.parameters.keys())
            except (TypeError, ValueError):
                names = []

            if "class_agnostic" in names:
                outputs = pp(outputs, self.num_classes, conf, nms, class_agnostic=True)
            elif "agnostic" in names:
                outputs = pp(outputs, self.num_classes, conf, nms, agnostic=True)
            else:
                try:
                    if len(names) >= 5:
                        outputs = pp(outputs, self.num_classes, conf, nms, True)
                    else:
                        outputs = pp(outputs, self.num_classes, conf, nms)
                except TypeError:
                    try:
                        outputs = pp(outputs, self.num_classes, conf, nms, True)
                    except TypeError:
                        outputs = pp(outputs, self.num_classes, conf, nms)
            # --- end shim ---
        return outputs, img_info


if __name__ == "__main__":
    main()
