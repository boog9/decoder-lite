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
from pathlib import Path
from typing import Iterable, List, Set

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("decoder-lite")


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
        dets: Detection array of shape [N, M]. Last column or column 5 contains
            integer class ids.
        keep: Iterable of class ids to keep.

    Returns:
        Filtered detection array.
    """
    if np is None:
        raise ModuleNotFoundError("numpy is required for filter_by_classes")
    if dets.size == 0:
        return dets
    keep_set: Set[int] = set(keep)
    cls_col = dets.shape[1] - 1 if dets.shape[1] >= 6 else 5
    cls_ids = dets[:, cls_col].astype(int)
    mask = np.isin(cls_ids, list(keep_set))
    return dets[mask]


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
    parser.add_argument("--keep-classes", type=str, default="0,32",
                        help="COCO class ids to keep, comma-separated.")
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
    from yolox.utils import Timer
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
            dets_before = dets.shape[0]
            dets = filter_by_classes(dets, keep_classes)
            logger.info(f"Frame {frame_id}: kept {dets.shape[0]}/{dets_before} detections")
            if dets.shape[0] > 0:
                dets[:, :4] /= img_info["ratio"]
                if dets.shape[1] >= 7:
                    scores = (dets[:, 4] * dets[:, 5])[:, None]
                else:
                    scores = dets[:, 4:5]
                dets5 = np.concatenate([dets[:, :4], scores], axis=1)
                online_targets = tracker.update(
                    dets5,
                    [img_info["height"], img_info["width"]],
                    exp.test_size,
                )
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_cls = []
                for t in online_targets:
                    tlwh = t.tlwh
                    if tlwh[2] * tlwh[3] > args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(t.track_id)
                        online_scores.append(t.score)
                        if hasattr(t, "cls"):
                            online_cls.append(int(getattr(t, "cls")))
                timer.toc()
                cls_ids_arg = (
                    online_cls if len(online_cls) == len(online_ids) and len(online_cls) > 0 else None
                )
                online_im = plot_tracking(
                    img_info["raw_img"],
                    online_tlwhs,
                    online_ids,
                    frame_id=frame_id,
                    fps=1.0 / timer.average_time,
                    scores=online_scores,
                    cls_ids=cls_ids_arg,
                )
                if args.save_result:
                    writer.write(online_im)
                    records.append({
                        "frame": frame_id,
                        "tlwh": [list(map(float, tlwh)) for tlwh in online_tlwhs],
                        "id": online_ids,
                        "cls": online_cls,
                        "score": online_scores,
                    })
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
