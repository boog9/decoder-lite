# decoder-lite

Minimal ByteTrack wrapper that tracks only COCO classes **0** (person) and **32** (sports ball).

## Prerequisites
- Ubuntu 22.04 with NVIDIA GPU (CUDA 12.x, tested with 12.9)
- Python 3.9+
- ByteTrack vendor auto-synced into `third_party/ByteTrack` by the setup script

### ONNX Runtime
- Linux/Windows: `onnxruntime-gpu==1.22.0` (CUDA 12.x wheels).
- macOS: `onnxruntime==1.22.1` (CPU).
- ONNX simplifier package name is `onnxsim>=0.4.36,<0.5` (previously `onnx-simplifier`).
- `scripts/setup_env.sh` installs `onnxruntime-gpu` on Linux/Windows and only
  falls back to the CPU wheel if the GPU wheel is unavailable.
- If `onnxsim` fails to install on Python 3.13, the setup script proceeds and
  installs `thop`, `loguru`, and `opencv-python` separately to satisfy ByteTrack
  imports.

### Torch/Torchvision (CUDA 12.x, Python 3.13)
For Python 3.13 on CUDA 12.x we install **torch 2.5.1+cu121** from the official PyTorch index and **build torchvision 0.20.1 from source with CUDA**. Prebuilt cp313 wheels for torchvision 0.20.1 with cu121 are not published; CPU wheels miss C++/CUDA ops (e.g., `torchvision::nms`), leading to runtime errors in ByteTrack/YOLOX.

**OS packages (recommended)**
```bash
sudo apt-get update
sudo apt-get install -y build-essential ninja-build libjpeg-dev zlib1g-dev libpng-dev ffmpeg
```

**One-liners**
```bash
make venv            # full setup including ByteTrack clone and deps
make diagnose-nms    # prints versions and checks that torchvision::nms is available
make run FILE=video.mp4
```

We never modify files under `third_party/ByteTrack/`. If the folder is missing or incomplete, it is cloned via `scripts/clone_bytetrack.sh` before the editable install.

> Notes:
> * Do **not** use `--index-url` globally for PyTorch wheels — we use `--extra-index-url` so other packages come from PyPI.
> * Avoid mixing CPU and GPU ONNX Runtime on the same platform.
> * If you need to re-run only the Torch/Torchvision setup, call:
>   `bash scripts/setup_env.sh --vision-only` (this skips cloning/ByteTrack deps).
> * We do not modify any `third_party/ByteTrack/*.py` sources.

## Setup
```bash
make venv      # create venv, clone ByteTrack, install deps

# download YOLOX weights
make weights
```
`make venv` executes `scripts/setup_env.sh` which clones and installs ByteTrack in editable mode.

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

Use the `--keep-classes` flag to restrict tracking to specific class IDs. If the
detector outputs only five columns (no class column), the flag is ignored and a
warning is logged.

## Notes
- Only COCO classes 0 and 32 are processed.
- `scripts/setup_env.sh` removes stray `~ip` directories that can trigger
  `pip` warnings about invalid distributions and reinstalls the build toolchain.
- Torch with CUDA must be present before building ByteTrack. `make venv` installs
  PyTorch with CUDA 12.1 wheels (compatible with CUDA 12.4 runtime) and installs
  ByteTrack in PEP 517 editable compat mode without dependencies; adjust the
  index URL in `scripts/setup_env.sh` for other CUDA versions.
- The setup script installs `cython-bbox`, required for bounding box overlap
  computations in ByteTrack.
