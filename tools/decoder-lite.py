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
from typing import Any, Iterable, List, Optional, Sequence, Set

import logging
import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # allow running without torch in serialization


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
        preproc_fn=preproc,
        device=args.device,
        fp16=False,
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
                    _dt = None
                    _toc = getattr(timer, "toc", None)
                    if callable(_toc):
                        try:
                            _dt = _toc()
                        except Exception:
                            _dt = None
                    if _dt is None:
                        _now = time.perf_counter()
                        _dt = _now - _prev_ts
                        _prev_ts = _now
                    _fps = fps_meter.update(_dt)
                    fps = float(_fps) if _fps is not None else 0.0
                    # Preserve the original frame before drawing overlays.
                    raw_im = img_info["raw_img"].copy()
                    # Draw on a separate buffer to avoid mutating ``raw_im``.
                    vis_im = raw_im.copy()
                    annotated = call_with_supported_kwargs(
                        plot_tracking,
                        vis_im,
                        online_tlwhs,
                        online_ids,
                        scores=online_scores,
                        frame_id=frame_id,
                        fps=fps,
                        cls_ids=online_cls_ids if "online_cls_ids" in locals() else None,
                    )
                    if args.save_result:
                        writer.write(raw_im if args.save_raw else annotated)
                        record = {
                            "frame": int(frame_id),
                            "tlwh": [
                                [float(v) for v in tlwh] for tlwh in online_tlwhs
                            ],
                            "id": [int(i) for i in online_ids],
                            "score": [float(s) for s in online_scores],
                        }
                        records.append(record)
                    if not args.no_display:
                        cv2.imshow(
                            "ByteTrack", raw_im if args.save_raw else annotated
                        )
                else:
                    _dt = None
                    _toc = getattr(timer, "toc", None)
                    if callable(_toc):
                        try:
                            _dt = _toc()
                        except Exception:
                            _dt = None
                    if _dt is None:
                        _now = time.perf_counter()
                        _dt = _now - _prev_ts
                        _prev_ts = _now
                    _fps = fps_meter.update(_dt)
                    fps = float(_fps) if _fps is not None else 0.0
                    raw_im = img_info["raw_img"].copy()
                    vis_im = raw_im.copy()
                    annotated = call_with_supported_kwargs(
                        plot_tracking,
                        vis_im,
                        [],
                        [],
                        scores=[],
                        frame_id=frame_id,
                        fps=fps,
                    )
                    if args.save_result:
                        writer.write(raw_im if args.save_raw else annotated)
                    if not args.no_display:
                        cv2.imshow(
                            "ByteTrack", raw_im if args.save_raw else annotated
                        )
            else:
                _dt = None
                _toc = getattr(timer, "toc", None)
                if callable(_toc):
                    try:
                        _dt = _toc()
                    except Exception:
                        _dt = None
                if _dt is None:
                    _now = time.perf_counter()
                    _dt = _now - _prev_ts
                    _prev_ts = _now
                _fps = fps_meter.update(_dt)
                fps = float(_fps) if _fps is not None else 0.0
                raw_im = img_info["raw_img"].copy()
                vis_im = raw_im.copy()
                annotated = call_with_supported_kwargs(
                    plot_tracking,
                    vis_im,
                    [],
                    [],
                    scores=[],
                    frame_id=frame_id,
                    fps=fps,
                )
                if args.save_result:
                    writer.write(raw_im if args.save_raw else annotated)
                if not args.no_display:
                    cv2.imshow(
                        "ByteTrack", raw_im if args.save_raw else annotated
                    )
        else:
            _dt = None
            _toc = getattr(timer, "toc", None)
            if callable(_toc):
                try:
                    _dt = _toc()
                except Exception:
                    _dt = None
            if _dt is None:
                _now = time.perf_counter()
                _dt = _now - _prev_ts
                _prev_ts = _now
            _fps = fps_meter.update(_dt)
            fps = float(_fps) if _fps is not None else 0.0
            raw_im = img_info["raw_img"].copy()
            vis_im = raw_im.copy()
            annotated = call_with_supported_kwargs(
                plot_tracking,
                vis_im,
                [],
                [],
                scores=[],
                frame_id=frame_id,
                fps=fps,
            )
            if args.save_result:
                writer.write(raw_im if args.save_raw else annotated)
            if not args.no_display:
                cv2.imshow(
                    "ByteTrack", raw_im if args.save_raw else annotated
                )
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
            json.dump(records, f, ensure_ascii=False, default=_json_default)
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

        img_info = {"raw_img": img, "height": img.shape[0], "width": img.shape[1]}
        img, ratio = self.preproc(img, self.test_size, self.mean, self.std)
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()
        with torch.no_grad():
            outputs = self.model(img)
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
        img_info["ratio"] = ratio
        return outputs, img_info


if __name__ == "__main__":
    main()
