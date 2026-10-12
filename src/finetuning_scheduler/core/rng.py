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
Fine-Tuning Scheduler RNG Isolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Capture, restore and fork the random number generators a training process draws from: PyTorch's CPU generator, the
generator of each CUDA device the process uses, Python's ``random`` module and NumPy's legacy global generator.

"""
import random
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import torch


def _cuda_indices(devices: Iterable[torch.device | int | str]) -> list[int]:
    # CUDA generators are only read once CUDA is initialized: reading one initializes CUDA (and, under multiple
    # processes, would create contexts on devices the process does not own)
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return []
    indices = set()
    for d in devices:
        dev = torch.device("cuda", d) if isinstance(d, int) else torch.device(d)
        if dev.type == "cuda":
            indices.add(dev.index if dev.index is not None else torch.cuda.current_device())
    return sorted(indices)


def capture_rng_state(devices: Iterable[torch.device | int | str] = ()) -> dict[str, Any]:
    """Capture the state of every random number generator a training process draws from.

    Args:
        devices: The CUDA devices whose generators to capture, typically every device holding the module's parameters
            plus the current device. Ignored until CUDA is initialized.

    Returns:
        A state mapping accepted by :func:`restore_rng_state`. NumPy's state is included only if NumPy has already been
        imported: if it has not, no code can have seeded it, so there is nothing to preserve.
    """
    state: dict[str, Any] = {"torch": torch.get_rng_state(), "python": random.getstate()}
    cuda_indices = _cuda_indices(devices)
    if cuda_indices:
        state["cuda"] = {i: torch.cuda.get_rng_state(i) for i in cuda_indices}
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        state["numpy"] = numpy.random.get_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore generator states captured by :func:`capture_rng_state`."""
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    for i, cuda_state in state.get("cuda", {}).items():
        torch.cuda.set_rng_state(cuda_state, i)
    if "numpy" in state:
        sys.modules["numpy"].random.set_state(state["numpy"])


@contextmanager
def fork_rngs(devices: Iterable[torch.device | int | str] = ()) -> Iterator[None]:
    """Run a block without disturbing the random streams of the surrounding code.

    Every generator :func:`capture_rng_state` covers is restored on exit, including when the block raises. A block
    that initializes CUDA itself is not isolated on CUDA, since no CUDA state existed to capture beforehand.

    Args:
        devices: The CUDA devices whose generators to fork (see :func:`capture_rng_state`).
    """
    state = capture_rng_state(devices)
    try:
        yield
    finally:
        restore_rng_state(state)
