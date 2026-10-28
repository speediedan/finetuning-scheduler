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
"""The tunable view protocol lives in venngram (``venngram.tuning``, with the roles in ``venngram.roles``); Fine-
Tuning Scheduler re-exports it as ``finetuning_scheduler.tunable_view`` (#140).

That the protocol imports nothing but torch, the
standard library and itself is venngram's own test. This module must itself stay importable without Lightning: the
``core-no-lightning`` CI job runs it with ``--noconftest``.
"""
import importlib

import pytest
import torch
import venngram.roles
import venngram.tuning

import finetuning_scheduler.tunable_view as tv
from finetuning_scheduler.core import roles as fts_roles
from finetuning_scheduler.core.compile import compile_schedule
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.views import ScheduleUnresolvable, Unresolvable
from finetuning_scheduler.testing import TunableViewConformance
from finetuning_scheduler.testing.fixtures import ALL_FIXTURES


@pytest.mark.parametrize("module, protocol", [("roles", venngram.roles), ("views", venngram.tuning.views),
                                              ("errors", venngram.tuning.errors),
                                              ("conformance", venngram.tuning.conformance)])
def test_the_reexported_submodules_are_venngrams_own(module, protocol):
    assert importlib.import_module(f"finetuning_scheduler.tunable_view.{module}") is protocol
    assert getattr(tv, module) is protocol


def test_the_reexported_names_are_venngrams_own():
    assert tv.__all__ == venngram.tuning.__all__
    assert all(getattr(tv, name) is getattr(venngram.tuning, name) for name in tv.__all__)


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
    with pytest.raises(venngram.errors.RoleError, match="unknown role kind") as excinfo:
        tv.roles.parse_selector("role:weights")
    assert not isinstance(excinfo.value, FTSConfigurationError)
    with pytest.raises(tv.TunableViewError, match="is not a tunable view") as excinfo:
        tv.as_view(object())
    assert not isinstance(excinfo.value, FTSConfigurationError)
