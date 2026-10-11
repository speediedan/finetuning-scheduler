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
Fine-Tuning Scheduler Core Distributed Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rank-aware warnings and logging for the framework-neutral Fine-Tuning Scheduler core.

By default, the global rank comes from an initialized ``torch.distributed`` process group (or, before one exists, the
usual launcher environment variables), and messages are logged to the ``finetuning_scheduler`` logger. A training host
may register its own backend with :func:`register_rank_zero_backend`; the Lightning integration registers Lightning's
rank-zero utilities, so ranks, loggers and logging levels behave exactly as they do for other Lightning code.

The ``stacklevel`` arguments follow :func:`warnings.warn`: the default of 2 attributes a message to the code that
called the function.
"""
import logging
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

import torch

log = logging.getLogger("finetuning_scheduler")

T = TypeVar("T")

_RANK_ENV_VARS = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")


@dataclass(frozen=True)
class RankZeroBackend:
    """The rank and message sinks used by the core.

    Args:
        global_rank: Returns the current global rank.
        warn: Emits a warning (``message, stacklevel=..., **kwargs``) when on global rank zero.
        info: Logs an info message (``message, stacklevel=..., **kwargs``) when on global rank zero.
        debug: Logs a debug message (``message, stacklevel=..., **kwargs``) when on global rank zero.
        stack_offset: Frames each sink adds between the core function and the eventual
            :func:`warnings.warn` or logging call, added to the caller's ``stacklevel``.
    """

    global_rank: Callable[[], int]
    warn: Callable[..., None]
    info: Callable[..., None]
    debug: Callable[..., None]
    stack_offset: int


def _default_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    for key in _RANK_ENV_VARS:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def _default_warn(message: str | Warning, stacklevel: int = 2, **kwargs: Any) -> None:
    if _default_global_rank() == 0:
        warnings.warn(message, stacklevel=stacklevel, **kwargs)


def _default_info(message: str, stacklevel: int = 2, **kwargs: Any) -> None:
    if _default_global_rank() == 0:
        log.info(message, stacklevel=stacklevel, **kwargs)


def _default_debug(message: str, stacklevel: int = 2, **kwargs: Any) -> None:
    if _default_global_rank() == 0:
        log.debug(message, stacklevel=stacklevel, **kwargs)


DEFAULT_BACKEND = RankZeroBackend(
    global_rank=_default_global_rank, warn=_default_warn, info=_default_info, debug=_default_debug, stack_offset=1
)

_backend: RankZeroBackend = DEFAULT_BACKEND


def register_rank_zero_backend(backend: RankZeroBackend) -> None:
    """Register the backend the core uses for ranks, warnings and logging.

    Registration is idempotent; registering a different backend replaces the previous one.

    Args:
        backend: The :class:`RankZeroBackend` to use.
    """
    global _backend
    _backend = backend


def rank_zero_backend() -> RankZeroBackend:
    """Return the currently registered :class:`RankZeroBackend`."""
    return _backend


def global_rank() -> int:
    """Return the global rank reported by the registered backend."""
    return _backend.global_rank()


def rank_zero_only(fn: Callable[..., T]) -> Callable[..., T | None]:
    """Decorate ``fn`` so it runs only on global rank zero, returning ``None`` elsewhere.

    The rank is read when the decorated function is called, not when it is defined.
    """

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> T | None:
        if _backend.global_rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


def rank_zero_warn(message: str | Warning, stacklevel: int = 2, **kwargs: Any) -> None:
    """Emit a warning only on global rank zero."""
    _backend.warn(message, stacklevel=stacklevel + _backend.stack_offset, **kwargs)


def rank_zero_info(message: str, stacklevel: int = 2, **kwargs: Any) -> None:
    """Log an info message only on global rank zero."""
    _backend.info(message, stacklevel=stacklevel + _backend.stack_offset, **kwargs)


def rank_zero_debug(message: str, stacklevel: int = 2, **kwargs: Any) -> None:
    """Log a debug message only on global rank zero."""
    _backend.debug(message, stacklevel=stacklevel + _backend.stack_offset, **kwargs)
