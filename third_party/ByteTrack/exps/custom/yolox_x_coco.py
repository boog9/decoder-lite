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

"""Custom YOLOX-X experiment with 80 COCO classes."""

from __future__ import annotations

from yolox.exp.default.yolox_x import Exp as _Exp


class Exp(_Exp):
    """YOLOX-X experiment configuration with 80 classes."""

    def __init__(self) -> None:
        super().__init__()
        # Ensure detector outputs 80 COCO classes
        self.num_classes = 80

