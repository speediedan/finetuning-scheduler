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
r"""
Fine-Tuning Scheduler Batch Fingerprints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A content digest of a batch, recorded for the first batches of each phase so a resumed or repeated run can confirm it
saw the same data in the same order.

"""
import hashlib
import sys
from collections.abc import Mapping
from typing import Any

import torch


def _tensor_bytes(t: torch.Tensor) -> bytes:
    flat = t.detach().cpu().contiguous().reshape(-1)
    raw = flat.view(torch.uint8) if flat.numel() else flat.new_empty(0, dtype=torch.uint8)
    if "numpy" in sys.modules:  # identical bytes, only faster
        return raw.numpy().tobytes()
    return bytes(raw.tolist())


def _update(h: "hashlib._Hash", obj: Any) -> None:
    if isinstance(obj, torch.Tensor):
        h.update(f"tensor:{obj.dtype}:{tuple(obj.shape)}:".encode())
        h.update(_tensor_bytes(obj))
    elif isinstance(obj, Mapping):
        h.update(f"map:{len(obj)}:".encode())
        for k in sorted(obj, key=str):
            h.update(f"key:{k!s}:".encode())
            _update(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        h.update(f"seq:{len(obj)}:".encode())
        for v in obj:
            _update(h, v)
    else:
        h.update(f"{type(obj).__qualname__}:{obj!r}:".encode())


def batch_fingerprint(batch: Any) -> str:
    """Return a SHA-256 hex digest of ``batch``'s contents.

    Tensors contribute their dtype, shape and raw bytes (on any device); mappings are walked in sorted key order,
    sequences in order, and any other value by its type and ``repr``.
    """
    h = hashlib.sha256()
    _update(h, batch)
    return h.hexdigest()
