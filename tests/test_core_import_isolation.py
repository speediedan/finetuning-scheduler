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
"""The package and its scheduling core import and run without Lightning or jsonargparse (#50).

This module must itself stay importable without either: the ``core-no-lightning`` CI job runs it with
``--noconftest`` in an environment holding only the declared base dependencies.
"""
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import finetuning_scheduler

# the top-level distributions the core must never import; ``finetuning_scheduler.lightning`` is not among them
BLOCKED = ("lightning", "pytorch_lightning", "lightning_fabric", "lightning_utilities", "torchmetrics", "jsonargparse")
RANK_ENV_VARS = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")

ISOLATED_SCRIPT = r'''
import importlib.abc
import sys

BLOCKED = frozenset(sys.argv[1].split(","))
PKG_DIR, TMP_DIR = sys.argv[2], sys.argv[3]


class BlockHostFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {fullname!r} (blocked by the isolation test)", name=fullname)
        return None


sys.meta_path.insert(0, BlockHostFrameworks())
try:
    import lightning  # noqa: F401
except ImportError:
    pass
else:
    raise AssertionError("the import blocker is not active")

import torch

import finetuning_scheduler
from finetuning_scheduler import FTSState
from finetuning_scheduler.core import optim, schedule  # noqa: F401
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.types import FTSLRSchedulerTypeTuple, ParamGroupAddable  # noqa: F401

assert finetuning_scheduler.__file__.startswith(PKG_DIR), finetuning_scheduler.__file__

module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
sched = schedule.load_yaml_schedule(schedule.gen_ft_schedule(module, TMP_DIR))
assert sched == {0: {"params": ["2.bias", "2.weight"]}, 1: {"params": ["0.bias", "0.weight"]}}, sched
schedule.validate_phases_disjoint(sched)
sched[1]["params"].append("2.bias")
try:
    schedule.validate_phases_disjoint(sched)
except FTSConfigurationError as err:
    assert type(err) is FTSConfigurationError and isinstance(err, ValueError), type(err)
else:
    raise AssertionError("overlapping phases were not rejected")
assert FTSState()._fts_ckpt_metadata["current_ckpt_depth"] == 0

for name in ("FinetuningScheduler", "FTSCheckpoint", "fts_supporters"):
    try:
        getattr(finetuning_scheduler, name)
    except ImportError as err:
        assert "requires Lightning" in str(err), err
    else:
        raise AssertionError(f"{name} resolved without Lightning")
try:
    finetuning_scheduler.types.BaseCallbackDepType
except ImportError:
    pass
else:
    raise AssertionError("BaseCallbackDepType resolved without Lightning")

leaked = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
assert not leaked, f"blocked modules were imported: {leaked}"
print("isolated core OK")
'''


def test_core_runs_without_lightning(tmp_path):
    """In a fresh interpreter that cannot import Lightning or jsonargparse, the package imports, the core
    generates, saves, loads and validates a schedule, and the Lightning-bound names fail with an actionable
    ``ImportError``."""
    pkg_dir = os.path.dirname(finetuning_scheduler.__file__)
    env = {k: v for k, v in os.environ.items() if k not in RANK_ENV_VARS}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (os.path.dirname(pkg_dir), env.get("PYTHONPATH")) if p)
    result = subprocess.run(
        [sys.executable, "-c", ISOLATED_SCRIPT, ",".join(BLOCKED), pkg_dir, str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "isolated core OK" in result.stdout


def test_lazy_attributes_are_discoverable():
    """The lazily loaded names are listed by ``dir`` and unknown names raise ``AttributeError``, as jsonargparse's
    ``getattr(module, name, None)`` probes expect."""
    assert set(finetuning_scheduler.__all__) <= set(dir(finetuning_scheduler))
    assert {"fts", "fts_supporters", "lightning", "strategy_adapters", "types"} <= set(dir(finetuning_scheduler))
    assert getattr(finetuning_scheduler, "not_a_member", None) is None
    with pytest.raises(AttributeError, match="has no attribute 'not_a_member'"):
        finetuning_scheduler.types.not_a_member


@pytest.mark.parametrize(
    "missing, expected",
    [("lightning.pytorch", ImportError), ("pytorch_lightning", ImportError), ("some_other_dist", ModuleNotFoundError)],
)
def test_lazy_import_failure(monkeypatch, missing, expected):
    """A missing Lightning distribution becomes an actionable ``ImportError``; any other missing module
    propagates."""

    def import_module(name):
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    monkeypatch.setattr(finetuning_scheduler, "importlib", SimpleNamespace(import_module=import_module))
    with pytest.raises(expected) as excinfo:
        finetuning_scheduler.__getattr__("FTSCheckpoint")
    assert ("requires Lightning" in str(excinfo.value)) is (expected is ImportError)
    assert type(excinfo.value) is expected
