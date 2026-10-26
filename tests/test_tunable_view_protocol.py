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
"""The tunable view protocol package stands on its own (#135).

``finetuning_scheduler.tunable_view`` is shaped as the distribution it is to become: its modules import only ``torch``,
the standard library and each other (relatively), so a copy of the directory imports as a top-level ``tunable_view``
package without Fine-Tuning Scheduler. This module must itself stay importable without Lightning: the
``core-no-lightning`` CI job runs it with ``--noconftest``.
"""
import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import finetuning_scheduler.tunable_view as tv
from finetuning_scheduler.core import roles as fts_roles
from finetuning_scheduler.core.compile import compile_schedule
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.views import ScheduleUnresolvable, Unresolvable
from finetuning_scheduler.testing import TunableViewConformance
from finetuning_scheduler.testing.fixtures import ALL_FIXTURES

PKG_DIR = Path(tv.__file__).parent
ALLOWED_TOP_LEVEL = {"torch"}

STANDALONE_SCRIPT = r'''
import importlib.abc
import sys


class BlockFineTuningScheduler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "finetuning_scheduler":
            raise ModuleNotFoundError(f"No module named {fullname!r} (blocked by the standalone test)", name=fullname)
        return None


sys.meta_path.insert(0, BlockFineTuningScheduler())
sys.path.insert(0, sys.argv[1])

import torch

import tunable_view
from tunable_view import CanonicalView, resolve_view, TunableViewError, Unresolvable, VIEW_ATTR, ViewConformance

assert tunable_view.__file__.startswith(sys.argv[1]), tunable_view.__file__
assert VIEW_ATTR == "tunable_view"
model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
setattr(model, VIEW_ATTR, CanonicalView(model, roles={"1.bias": "fitted"}))
bound = resolve_view(model)
assert bound.roles() == {"1.bias": "fitted"}
try:
    bound.to_physical(["nope"])
except Unresolvable as err:
    assert isinstance(err, TunableViewError) and err.names == ["nope"]
else:
    raise AssertionError("an unknown name resolved")
report = ViewConformance(lambda: CanonicalView(torch.nn.Linear(2, 2), roles={"bias": "instrument"})).run()
assert report.ok and "promotion" in report.ran, report
assert not [m for m in sys.modules if m.split(".")[0] == "finetuning_scheduler"]
print("STANDALONE-OK")
'''


def _imports(path: Path) -> list[tuple[int, str]]:
    """Each absolute import in ``path`` as ``(level, top-level module)``; relative imports have ``level > 0``."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.extend((0, alias.name.split(".")[0]) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.level, (node.module or "").split(".")[0]))
    return found


@pytest.mark.parametrize("module", sorted(p.name for p in PKG_DIR.glob("*.py")))
def test_protocol_imports_only_torch_and_itself(module):
    outside = sorted({top for level, top in _imports(PKG_DIR / module)
                      if level == 0 and top not in ALLOWED_TOP_LEVEL and top not in sys.stdlib_module_names})
    assert not outside, f"{module} imports {outside}; the protocol package may import only torch and the stdlib"


def test_protocol_imports_standalone(tmp_path):
    """A copy of the package imports as ``tunable_view`` with Fine-Tuning Scheduler unimportable, and works."""
    shutil.copytree(PKG_DIR, tmp_path / "tunable_view", ignore=shutil.ignore_patterns("__pycache__"))
    proc = subprocess.run([sys.executable, "-I", "-c", STANDALONE_SCRIPT, str(tmp_path)], capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0 and "STANDALONE-OK" in proc.stdout, proc.stderr


@pytest.mark.parametrize("name", sorted(ALL_FIXTURES))
def test_fixture_views_conform_to_the_protocol(name):
    report = tv.ViewConformance(ALL_FIXTURES[name]).run()
    assert report.ok, report.failures


def test_fts_suite_extends_the_protocol_suite():
    fts_cases = [c.name for c in TunableViewConformance(ALL_FIXTURES["identity"]).cases()]
    protocol = [c.name for c in tv.ViewConformance(ALL_FIXTURES["identity"]).cases()]
    assert [c for c in fts_cases if c in protocol] == protocol
    assert len(fts_cases) == len(set(fts_cases)) == len(protocol) + 7


def test_schedule_errors_are_configuration_errors_and_view_errors_are_not():
    model = torch.nn.Linear(2, 2)
    compiled = compile_schedule({0: {"params": ["weight"]}}, tv.CanonicalView(model))
    with pytest.raises(ScheduleUnresolvable, match="nope") as excinfo:
        compiled.physical_of(["nope"])
    err = excinfo.value
    assert isinstance(err, Unresolvable) and isinstance(err, FTSConfigurationError) and err.names == ["nope"]
    assert Unresolvable is tv.Unresolvable
    with pytest.raises(FTSConfigurationError, match="unknown role kind"):
        fts_roles.parse_selector("role:weights")
    with pytest.raises(tv.TunableViewError, match="unknown role kind") as excinfo:
        tv.roles.parse_selector("role:weights")
    assert not isinstance(excinfo.value, FTSConfigurationError)
    with pytest.raises(tv.TunableViewError, match="is not a tunable view") as excinfo:
        tv.as_view(object())
    assert not isinstance(excinfo.value, FTSConfigurationError)
