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

### Install on Python 3.13 (CUDA 12.x)
```bash
make venv      # this runs scripts/setup_env.sh
make doctor    # verify torch/yolox/onnxruntime-gpu
```
This repository vendors ByteTrack under `third_party/ByteTrack` (it is **not** a git submodule).
The setup script:
1. clones ByteTrack if missing,
2. installs build tools and PyTorch (CUDA 12.1 wheels),
3. installs ByteTrack in editable mode with `--no-build-isolation`,
4. installs ONNX Runtime GPU `==1.22.0` plus `onnx` and `onnxsim`.

> Notes:
> * Do **not** use `--index-url` globally for PyTorch wheels — we use `--extra-index-url` so other packages come from PyPI.
> * Avoid mixing CPU and GPU ONNX Runtime on the same platform.
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
- Torch with CUDA must be present before building ByteTrack. `make venv` installs
  PyTorch with CUDA 12.1 wheels (compatible with CUDA 12.4 runtime); adjust the
  index URL in `scripts/setup_env.sh` for other CUDA versions.
