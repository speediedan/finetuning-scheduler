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
"""Tests for compiling a schedule's logical names into physical parameters.

These need only PyTorch and the scheduling core (the ``core-no-lightning`` job runs them with ``--noconftest``).
"""
import copy

import pytest
import torch
from torch import nn

from finetuning_scheduler import CanonicalView, TunableSupport
from finetuning_scheduler.core.compile import compile_schedule, view_default_schedule
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.schedule import resolve_phase_params, shared_params_ref
from finetuning_scheduler.core.views import Unresolvable


def _net():
    return nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 2))


class BlockView:
    """Logical ``block<i>`` names for each ``Linear``, coupling each block's following ``LayerNorm``."""

    BLOCKS = {"block0": ("0.weight", "0.bias"), "block2": ("2.weight", "2.bias"), "head": ("4.weight", "4.bias")}
    NORMS = {"0.weight": ["1.weight", "1.bias"], "2.weight": ["3.weight", "3.bias"]}

    def __init__(self, root, support=None, roles=None):
        self.root = root
        self._support = support or TunableSupport(trainable=True, naming_scheme="blocks")
        self._roles = roles or {}

    def support(self):
        return self._support

    def to_physical(self, logical):
        unresolved = [n for n in logical if n not in self.BLOCKS]
        if unresolved:
            raise Unresolvable(unresolved)
        return {n: self.BLOCKS[n] for n in logical}

    def to_logical(self, physical):
        rev = {p: n for n, ps in self.BLOCKS.items() for p in ps}
        return list(dict.fromkeys(rev.get(p, p) for p in physical))

    def logical_named_parameters(self):
        for n, ps in self.BLOCKS.items():
            yield n, self.root.get_parameter(ps[0])

    def coupled(self, physical):
        return [c for p in physical for c in self.NORMS.get(p, [])]

    def roles(self):
        return self._roles


def test_canonical_compile_matches_the_lightning_resolution():
    """Through the canonical view, each phase compiles to exactly the names the Lightning callback resolves."""
    net = _net()
    schedule = {0: {"params": ["4.weight", "4.bias"]}, 1: {"params": [r"[23]\..*"]}, 2: {"params": [r"[01]\..*"]}}
    compiled = compile_schedule(schedule, CanonicalView(net))
    expected = copy.deepcopy(schedule)
    named = [n for n, _ in net.named_parameters()]
    for depth in expected:
        resolve_phase_params(expected, depth, named, shared_params_ref(net))
    assert [list(cp.physical) for cp in compiled.phases] == [expected[d]["params"] for d in sorted(expected)]
    assert all(cp.coupled == () for cp in compiled.phases)
    assert schedule[1]["params"] == [r"[23]\..*"]  # the schedule keeps its entries
    assert compiled.naming_scheme == "physical" and compiled.overlaps == ()
    assert compiled.module_map() == {0: {"4"}, 1: {"2", "3"}, 2: {"0", "1"}}
    assert compiled.digest == compile_schedule(schedule, CanonicalView(net)).digest


def test_logical_view_one_to_many_and_first_come_coupling():
    net = _net()
    schedule = {0: {"params": ["head"]}, 1: {"params": ["block2"]}, 2: {"params": ["block0"]}}
    compiled = compile_schedule(schedule, BlockView(net))
    assert compiled.physical_for(0) == ("4.weight", "4.bias")
    assert compiled.phases[1].physical == ("2.weight", "2.bias")
    assert compiled.phases[1].coupled == ("3.weight", "3.bias")
    assert compiled.physical_for(2) == ("0.weight", "0.bias", "1.weight", "1.bias")
    assert compiled.physical_of(["block2", "head"]) == ["2.weight", "2.bias", "4.weight", "4.bias"]
    with pytest.raises(Unresolvable, match="not names this compiled schedule resolved"):
        compiled.physical_of(["block9"])
    assert compiled.naming_scheme == "blocks"


def test_coupled_tensor_claimed_by_an_earlier_phase_is_dropped():
    """A norm coupled to phase 1 is not thawed again by phase 2, even when phase 2 names it."""
    net = _net()
    view = CanonicalView(net)
    view.coupled = lambda physical: ["1.weight"] if "0.weight" in physical else []  # type: ignore[method-assign]
    compiled = compile_schedule({0: {"params": ["4.weight"]}, 1: {"params": ["0.weight"]},
                                 2: {"params": ["1.weight", "1.bias"]}}, view)
    assert compiled.phases[1].coupled == ("1.weight",)
    assert compiled.phases[2].physical == ("1.bias",)


def test_unresolvable_and_unmatched_entries_refuse():
    net = _net()
    with pytest.raises(FTSConfigurationError, match="'block7' specified in phase 1"):
        compile_schedule({0: {"params": ["head"]}, 1: {"params": ["block7"]}}, BlockView(net))

    class ListsMoreThanItMaps(BlockView):
        def logical_named_parameters(self):
            yield from super().logical_named_parameters()
            yield "block7", self.root.get_parameter("0.weight")

    with pytest.raises(Unresolvable, match=r"\['block7'\]. \(phase 1\)"):
        compile_schedule({0: {"params": ["head"]}, 1: {"params": ["block7"]}}, ListsMoreThanItMaps(net))
    with pytest.raises(FTSConfigurationError, match="'nothing.*' specified in phase 0"):
        compile_schedule({0: {"params": ["nothing.*"]}}, CanonicalView(net))


def test_role_selectors():
    net = _net()
    view = CanonicalView(net, roles={"2.weight": "adapter:a", "2.bias": "adapter:b", "4.weight": "latent:t"})
    compiled = compile_schedule({0: {"params": ["role:latent"]}, 1: {"params": ["role:adapter:a", "2.bias"]},
                                 2: {"params": ["role:base"]}}, view)
    assert compiled.phases[0].physical == ("4.weight",)
    assert compiled.phases[1].physical == ("2.weight", "2.bias")
    assert compiled.phases[2].physical == ("0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias", "4.bias")
    with pytest.raises(FTSConfigurationError, match="'role:intervention' in phase 0 matched no parameter"):
        compile_schedule({0: {"params": ["role:intervention"]}}, view)


def test_frozen_roles_refuse():
    net = _net()
    view = CanonicalView(net, roles={"0.weight": "fitted", "1.weight": "instrument"})
    with pytest.raises(FTSConfigurationError, match=r"Phase 1 schedules tensors whose role is never thawed.*0.weight"):
        compile_schedule({0: {"params": ["4.weight"]}, 1: {"params": [r"0\..*"]}}, view)
    leaky = BlockView(net, roles={"1.weight": "instrument"})
    sched = {0: {"params": ["head"]}, 1: {"params": ["block0"]}}
    with pytest.raises(FTSConfigurationError, match="coupled tensors include tensors whose role is never thawed"):
        compile_schedule(sched, leaky)
    with pytest.warns(UserWarning, match="coupled tensors include"):
        compile_schedule(sched, leaky, legacy=True)


def _shared_storage_net():
    net = nn.Module()
    base = torch.zeros(6)
    net.a = nn.Parameter(base[0:3])
    net.b = nn.Parameter(base[2:5])  # overlaps `a` on element 2
    net.c = nn.Parameter(torch.zeros(2))
    return net


def test_storage_overlap_across_phases_refuses_and_within_a_phase_is_reported():
    net = _shared_storage_net()
    with pytest.raises(FTSConfigurationError, match="Phases 0 and 1 schedule tensors that share storage: 'a' and 'b'"):
        compile_schedule({0: {"params": ["a"]}, 1: {"params": ["b", "c"]}}, CanonicalView(net))
    with pytest.warns(UserWarning, match="share storage"):
        compile_schedule({0: {"params": ["a"]}, 1: {"params": ["b", "c"]}}, CanonicalView(net), legacy=True)
    compiled = compile_schedule({0: {"params": ["c"]}, 1: {"params": ["a", "b"]}}, CanonicalView(net))
    assert compiled.overlaps == (("a", "b"),)


def test_same_parameter_in_two_phases_refuses():
    net = _net()
    with pytest.raises(FTSConfigurationError, match="not disjoint: '4.weight' is scheduled in phases 0 and 1"):
        compile_schedule({0: {"params": ["4.weight"]}, 1: {"params": [r"4\..*"]}}, CanonicalView(net))


def test_support_refusals():
    net = _net()
    with pytest.raises(FTSConfigurationError, match="cannot be fine-tuned through BlockView: remote execution"):
        compile_schedule({0: {"params": ["head"]}},
                         BlockView(net, support=TunableSupport(trainable=False, reason="remote execution")))
    net.q = nn.Parameter(torch.zeros(2, dtype=torch.int8), requires_grad=False)
    view = CanonicalView(net)
    view.support = lambda: TunableSupport(trainable=True, logical_names_are_views=True)  # type: ignore[method-assign]
    with pytest.raises(FTSConfigurationError, match=r"quantized or packed tensors.*\['q'\]"):
        compile_schedule({0: {"params": ["q"]}}, view)


def test_view_default_schedule():
    net = _net()
    assert view_default_schedule(BlockView(net)) == {0: {"params": ["head", "block2"]}, 1: {"params": ["block0"]}}
    assert view_default_schedule(BlockView(net), order="input_first") == {0: {"params": ["block0"]},
                                                                          1: {"params": ["block2", "head"]}}
    with pytest.raises(FTSConfigurationError, match="`order` must be"):
        view_default_schedule(BlockView(net), order="sideways")  # type: ignore[arg-type]
