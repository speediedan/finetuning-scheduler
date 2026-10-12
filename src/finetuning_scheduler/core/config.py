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
Fine-Tuning Scheduler Configuration Hashing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A canonical, order-independent description of a run's configuration and its SHA-256 hash, recorded with checkpoints
so a resumed or compared run can tell whether it was configured identically.

"""
import enum
import functools
import hashlib
import json
import os
from collections.abc import Mapping, Set
from typing import Any

import torch

from finetuning_scheduler.core.dist import rank_zero_warn

_UNSERIALIZABLE_WARNED: set[str] = set()


def describe_callable(fn: Any) -> Any:
    """Describe a callable by import path, or a ``functools.partial`` as ``{class_path, init_args}``."""
    if isinstance(fn, functools.partial):
        desc: dict[str, Any] = {"class_path": describe_callable(fn.func), "init_args": _normalize(dict(fn.keywords))}
        if fn.args:
            desc["args"] = _normalize(list(fn.args))
        return desc
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None) or getattr(type(fn), "__qualname__", repr(fn))
    return f"{module}.{qualname}" if module else qualname


def _normalize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return _normalize(obj.value)
    if isinstance(obj, Mapping):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, Set):
        return sorted((_normalize(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(obj, (torch.dtype, torch.device, os.PathLike)):
        return str(obj)
    if isinstance(obj, (type, functools.partial)) or callable(obj):
        return describe_callable(obj)
    rep = repr(obj)
    type_name = type(obj).__qualname__
    if type_name not in _UNSERIALIZABLE_WARNED:
        _UNSERIALIZABLE_WARNED.add(type_name)
        rank_zero_warn(
            f"A configuration value of type {type_name} has no canonical form; its repr is hashed instead, so the "
            "configuration hash may differ between otherwise identical runs."
        )
    return {"__repr__": rep}


def canonical_json(config: Mapping[str, Any]) -> str:
    """Serialize ``config`` canonically: sorted keys, no whitespace, and string keys throughout."""
    return json.dumps(_normalize(config), sort_keys=True, separators=(",", ":"))


def config_hash(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 hex digest of :func:`canonical_json` of ``config``."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
