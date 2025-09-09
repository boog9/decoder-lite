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

.PHONY: clone venv weights run test doctor diagnose-nms

FILE  ?= 0
EXP   ?= third_party/ByteTrack/exps/custom/yolox_x_coco.py
EXTRA ?=

VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

clone:
	bash scripts/clone_bytetrack.sh

venv:
	@bash scripts/setup_env.sh

# Quick diagnostic for Torch/Torchvision CUDA ops (incl. torchvision::nms)
diagnose-nms:
	python scripts/verify_torchvision_nms.py

doctor:
	. $(VENV_DIR)/bin/activate && $(PYTHON) scripts/doctor.py

weights:
	bash scripts/download_yolox_weights.sh

run:
	python tools/decoder-lite.py -f "$(EXP)" -c third_party/ByteTrack/pretrained/yolox_x.pth --path "$(FILE)" --save_result --device gpu --fp16 --keep-classes 0,32 --no-display $(EXTRA)

test:
	python scripts/post_install_check.py
	pytest -q

.PHONY: ort-check
ort-check:
	python -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())'
