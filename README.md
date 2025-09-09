# decoder-lite

Minimal ByteTrack wrapper that tracks only COCO classes **0** (person) and **32** (sports ball).

## Prerequisites
 - Ubuntu 22.04 with NVIDIA GPU (CUDA 12.x, tested with 12.9)
- Python 3.9+
- ByteTrack cloned into `third_party/ByteTrack`

### ONNX Runtime (GPU)
- ONNX Runtime GPU wheels (>=1.19) are published on PyPI and require CUDA 12.x.
- This project tests with CUDA 12.9 and `onnxruntime-gpu==1.22.1`.
- macOS uses the CPU-only `onnxruntime` package.

## Setup
```bash
# optional: create and activate venv
python -m venv .venv && source .venv/bin/activate

# install dependencies and build ByteTrack
make venv

# download YOLOX weights
make weights
```

## Usage
### Track a video
```bash
make run FILE=path/to/video.mp4
```

### Track from webcam
```bash
make run FILE=0
```

The run target uses `EXP=third_party/ByteTrack/exps/custom/yolox_x_coco.py` by default. If your
ByteTrack clone provides a different COCO experiment path, override it:

```bash
make run FILE=video.mp4 EXP=third_party/ByteTrack/exps/default/yolox_x.py
```

Additional flags can be passed through `EXTRA`, for example:

```bash
make run FILE=video.mp4 EXTRA="--fuse"
```

The Makefile adds `--no-display` for headless execution. Remove the flag to view a window.

Results are written to `outputs/videos/result.mp4` and logs to `outputs/logs/result.json`.

## Notes
- Only COCO classes 0 and 32 are processed.
- Torch with CUDA must already be installed in the environment.
