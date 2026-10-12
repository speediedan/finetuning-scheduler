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
"""Tests for the core's RNG isolation, configuration hashing and batch fingerprints (#51)."""
import functools
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import finetuning_scheduler
from finetuning_scheduler.core import config as core_config
from finetuning_scheduler.core.config import canonical_json, config_hash, describe_callable
from finetuning_scheduler.core.fingerprint import batch_fingerprint
from finetuning_scheduler.core.rng import capture_rng_state, fork_rngs, restore_rng_state
from tests.helpers.runif import RunIf

SRC_DIR = str(Path(finetuning_scheduler.__file__).parent.parent)


def _draws():
    return torch.rand(3).tolist(), random.random(), float(np.random.rand())


def _seed(seed=7):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def test_fork_rngs_isolates_cpu_python_and_numpy():
    """Draws inside the fork do not shift the surrounding streams of torch, ``random`` or NumPy."""
    _seed()
    expected = _draws()
    _seed()
    with fork_rngs():
        _draws(), _draws()
    assert _draws() == expected


def test_fork_rngs_restores_on_exception():
    _seed()
    expected = _draws()
    _seed()
    with pytest.raises(RuntimeError, match="inside the fork"):
        with fork_rngs():
            _draws()
            raise RuntimeError("inside the fork")
    assert _draws() == expected


def test_capture_restore_round_trip():
    _seed(3)
    state = capture_rng_state()
    first = _draws()
    restore_rng_state(state)
    assert _draws() == first
    assert set(state) == {"torch", "python", "numpy"}


@RunIf(min_cuda_gpus=1)
def test_fork_rngs_isolates_cuda():
    torch.cuda.init()
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.manual_seed(5)
    expected = torch.rand(3, device=dev).tolist()
    torch.cuda.manual_seed(5)
    with fork_rngs([dev]):
        torch.rand(3, device=dev)
    assert torch.rand(3, device=dev).tolist() == expected


def _run_isolated(script: str, block_numpy: bool = False) -> str:
    prelude = (
        "import importlib.abc, sys\n"
        "class _Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] == 'numpy':\n"
        "            raise ModuleNotFoundError('blocked', name=fullname)\n"
        "sys.meta_path.insert(0, _Block())\n"
    ) if block_numpy else ""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (SRC_DIR, os.environ.get("PYTHONPATH")) if p)}
    result = subprocess.run([sys.executable, "-W", "ignore", "-c", prelude + script], capture_output=True, text=True,
                            env=env, timeout=300)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_fork_rngs_never_initializes_cuda():
    """Forking with CUDA devices named does not initialize CUDA when the process has not."""
    out = _run_isolated(
        "import torch\nfrom finetuning_scheduler.core.rng import fork_rngs\n"
        "with fork_rngs([0, 'cuda:1', torch.device('cpu')]):\n    torch.rand(1)\n"
        "print(torch.cuda.is_initialized())"
    )
    assert out == "False"


def test_fork_rngs_without_numpy():
    """Without NumPy, the fork covers the remaining generators and leaves NumPy unimported."""
    out = _run_isolated(
        "import random, sys, torch\nfrom finetuning_scheduler.core.rng import fork_rngs, capture_rng_state\n"
        "torch.manual_seed(1); random.seed(1)\nexpected = (torch.rand(1).item(), random.random())\n"
        "torch.manual_seed(1); random.seed(1)\n"
        "with fork_rngs():\n    torch.rand(4); random.random()\n"
        "assert (torch.rand(1).item(), random.random()) == expected\n"
        "print(sorted(capture_rng_state()), 'numpy' in sys.modules)",
        block_numpy=True,
    )
    assert out == "['python', 'torch'] False"


def test_config_hash_is_canonical():
    """Key order does not matter; values, nesting and types do; the golden digest pins the encoding."""
    cfg = {"max_depth": 2, "schedule": {0: {"params": ["a", "b"], "lr": 1e-05}, 1: {"params": ["c"]}},
           "dtype": torch.bfloat16, "flags": {"z", "a"}, "mode": None}
    reordered = {"mode": None, "flags": {"a", "z"}, "dtype": torch.bfloat16,
                 "schedule": {1: {"params": ["c"]}, 0: {"lr": 1e-05, "params": ["a", "b"]}}, "max_depth": 2}
    assert canonical_json(cfg) == canonical_json(reordered)
    assert canonical_json(cfg) == (
        '{"dtype":"torch.bfloat16","flags":["a","z"],"max_depth":2,"mode":null,'
        '"schedule":{"0":{"lr":1e-05,"params":["a","b"]},"1":{"params":["c"]}}}'
    )
    assert config_hash(cfg) == config_hash(reordered)
    assert config_hash(cfg) != config_hash({**cfg, "max_depth": 3})
    assert len(config_hash(cfg)) == 64


def test_describe_callable():
    assert describe_callable(torch.optim.AdamW) == "torch.optim.adamw.AdamW"
    desc = describe_callable(functools.partial(torch.optim.AdamW, lr=1e-3, weight_decay=0.0))
    assert desc == {"class_path": "torch.optim.adamw.AdamW", "init_args": {"lr": 0.001, "weight_decay": 0.0}}
    assert describe_callable(functools.partial(max, 1))["args"] == [1]
    assert canonical_json({"fn": len}) == '{"fn":"builtins.len"}'


def test_unserializable_value_warns_once(monkeypatch):
    class Opaque:
        def __repr__(self):
            return "Opaque()"

    monkeypatch.setattr(core_config, "_UNSERIALIZABLE_WARNED", set())
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        encoded = canonical_json({"a": Opaque(), "b": [Opaque()]})
    assert encoded == '{"a":{"__repr__":"Opaque()"},"b":[{"__repr__":"Opaque()"}]}'
    assert sum("has no canonical form" in str(w.message) for w in record) == 1


NESTED_BATCH = {
    "input_ids": torch.arange(6, dtype=torch.int64).reshape(2, 3),
    "labels": [torch.tensor([1.5, -2.0], dtype=torch.bfloat16), 3, "rte"],
}


def test_batch_fingerprint():
    """Dtype, shape and bytes all matter; mapping key order does not; the golden digest pins the encoding."""
    digest = batch_fingerprint(NESTED_BATCH)
    assert digest == batch_fingerprint({"labels": NESTED_BATCH["labels"], "input_ids": NESTED_BATCH["input_ids"]})
    assert digest != batch_fingerprint({**NESTED_BATCH, "input_ids": NESTED_BATCH["input_ids"].to(torch.int32)})
    assert digest != batch_fingerprint({**NESTED_BATCH, "input_ids": NESTED_BATCH["input_ids"].reshape(3, 2)})
    assert batch_fingerprint(torch.empty(0)) != batch_fingerprint(torch.empty(0, dtype=torch.int8))
    assert digest == GOLDEN_DIGEST


def test_batch_fingerprint_without_numpy():
    """The digest is identical whether or not NumPy is available."""
    out = _run_isolated(
        "import torch\nfrom finetuning_scheduler.core.fingerprint import batch_fingerprint\n"
        "print(batch_fingerprint({'input_ids': torch.arange(6, dtype=torch.int64).reshape(2, 3), "
        "'labels': [torch.tensor([1.5, -2.0], dtype=torch.bfloat16), 3, 'rte']}))",
        block_numpy=True,
    )
    assert out == GOLDEN_DIGEST


GOLDEN_DIGEST = "c6adc92bc6ef3bef46287eaeef4714c3e229162e16d416ab885120b1be50c8e6"
