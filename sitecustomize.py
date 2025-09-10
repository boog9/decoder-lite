# Copyright 2024 decoder-lite contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NumPy 2.x compatibility shim for legacy aliases used by third-party code.

This module restores aliases such as ``np.float`` removed in NumPy 2.x. It is
loaded automatically by Python when this repository root is on ``sys.path``.

Example:
    Running ``python tools/decoder-lite.py --help`` should no longer raise
    ``AttributeError: module 'numpy' has no attribute 'float'``.

Example output::

    $ python tools/decoder-lite.py --help
    usage: decoder-lite.py [-h] [--video VIDEO] [...]
"""

from __future__ import annotations

try:
    import numpy as _np

    # Restore removed aliases to keep older code paths working.
    if not hasattr(_np, "float"):
        _np.float = float
    if not hasattr(_np, "int"):
        _np.int = int
    if not hasattr(_np, "bool"):
        _np.bool = bool
    if not hasattr(_np, "object"):
        _np.object = object
except Exception:  # pragma: no cover - never fail hard from sitecustomize
    # The environment should remain usable even if NumPy is missing or
    # partially installed, so swallow all exceptions.
    pass
