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
"""Tests for the tunable view conformance cases and fixture views.

Every fixture view passes every case that applies to it, and every case fails on a view broken in exactly the way the
case checks for (the mutants below), so no case can pass vacuously. These need only PyTorch and the scheduling core
(the ``core-no-lightning`` job runs them with ``--noconftest``), which is why the mutants live here rather than under
``tests/helpers``.
"""
import os
import subprocess
import sys

import pytest
import torch
from torch import nn

from finetuning_scheduler import CanonicalView, TunableSupport
from finetuning_scheduler.core.views import Unresolvable
from finetuning_scheduler.testing import ConformanceError, pytest_params, TunableViewConformance
from finetuning_scheduler.testing.conformance import NO_SUCH_NAME
from finetuning_scheduler.testing.fixtures import (ALL_FIXTURES, D, InterventionModel, one_to_many_view, OneToManyView,
                                                   parametrized_view, ROLES, TinyTransformer, V)

EXPECTED_CASES = {
    "identity": {"mapping_completeness", "default_schedule_compiles", "thaw_is_exact_and_deterministic",
                 "step_changes_only_thawed", "checkpoint_round_trip", "unmatched_rule_refuses"},
    "one_to_many": {"mapping_completeness", "mapping_round_trip", "default_schedule_compiles",
                    "thaw_is_exact_and_deterministic", "step_changes_only_thawed", "checkpoint_round_trip",
                    "unmatched_rule_refuses"},
    "roles": {"mapping_completeness", "default_schedule_compiles", "thaw_is_exact_and_deterministic",
              "step_changes_only_thawed", "checkpoint_round_trip", "unmatched_rule_refuses", "roles_valid",
              "frozen_roles_refuse", "promotion"},
    "parametrized": {"mapping_completeness", "default_schedule_compiles", "thaw_is_exact_and_deterministic",
                     "step_changes_only_thawed", "checkpoint_round_trip", "unmatched_rule_refuses",
                     "parametrization_mapping"},
}


@pytest.mark.parametrize("fixture", list(ALL_FIXTURES), ids=list(ALL_FIXTURES))
def test_fixture_views_conform(fixture):
    report = TunableViewConformance(ALL_FIXTURES[fixture]).run(raise_on_failure=True)
    assert set(report.ran) == EXPECTED_CASES[fixture]
    assert report.ok


class DocsBlockView:
    """The ``BlockView`` example of the tunable views guide, verbatim: its ``to_physical`` indexes a mapping, so it
    raises ``KeyError`` (not ``Unresolvable``) for a name it does not know."""

    BLOCKS = {"block0": ("0.weight", "0.bias"), "head": ("2.weight", "2.bias")}
    NORMS = {"0.weight": ["1.weight", "1.bias"]}

    def __init__(self, root):
        self.root = root

    def support(self):
        return TunableSupport(trainable=True, naming_scheme="blocks")

    def to_physical(self, logical):
        return {n: self.BLOCKS[n] for n in logical}

    def to_logical(self, physical):
        rev = {p: n for n, ps in self.BLOCKS.items() for p in ps}
        return list(dict.fromkeys(rev.get(p, p) for p in physical))

    def logical_named_parameters(self):
        for n, ps in self.BLOCKS.items():
            yield n, self.root.get_parameter(ps[0])

    def coupled(self, physical):
        return [c for p in physical for c in self.NORMS.get(p, [])]


def test_the_guides_example_view_conforms():
    torch.manual_seed(0)
    report = TunableViewConformance(
        lambda: DocsBlockView(nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 2)))
    ).run(raise_on_failure=True)
    assert report.ok and "unmatched_rule_refuses" in report.ran


@pytest.mark.parametrize("case", pytest_params(one_to_many_view), ids=lambda case: case.__name__)
def test_pytest_params(case):
    case()


# ----------------------------------------------------------------------------------------------------------------------
# mutants: each breaks exactly one case
# ----------------------------------------------------------------------------------------------------------------------


class ListsAnUnmappedName(OneToManyView):
    def logical_named_parameters(self):
        yield from super().logical_named_parameters()
        yield "blocks.0.attn.W_O", self.root.embed.weight

    def to_physical(self, logical):
        return {n: (() if n == "blocks.0.attn.W_O" else self._table[n]) for n in logical}


class LosesARoundTrip(OneToManyView):
    def to_logical(self, physical):
        return ["blocks.1.mlp.W_in" if p == "blocks.0.mlp_in.bias" else n for p, n in
                zip(physical, (self._reverse.get(q, q) for q in physical))]


class SplitsTiedNames(OneToManyView):
    def depth_groups(self):
        groups = super().depth_groups()
        return [["unembed.W_U"], ["embed.W_E"]] + groups[1:]


class DriftingMapping(OneToManyView):
    """Resolves a name differently on each call."""

    def __init__(self, root):
        super().__init__(root)
        self.calls = 0

    def to_physical(self, logical):
        self.calls += 1
        mapped = dict(super().to_physical(logical))
        if self.calls % 2 == 0 and "blocks.0.mlp.W_in" in mapped:
            mapped["blocks.0.mlp.W_in"] = mapped["blocks.0.mlp.W_in"][:1]
        return mapped


class _HiddenAliasNet(nn.Module):
    """``b`` is a separate parameter over part of ``a``'s storage, so stepping ``a`` silently changes ``b``."""

    def __init__(self):
        super().__init__()
        base = torch.zeros(6)
        self.a = nn.Parameter(base[0:4])
        self.b = nn.Parameter(base[2:6])


class HidesSharedStorage(CanonicalView):
    def depth_groups(self):
        return [["a"], []]


class IgnoresCheckpointLoad(CanonicalView):
    def load_checkpoint_state_dict(self, state_dict, strict=True):
        pass


class MatchesEverything(CanonicalView):
    def logical_named_parameters(self):
        yield from super().logical_named_parameters()
        yield NO_SUCH_NAME + "_alias", self.root.embed.weight

    def to_physical(self, logical):
        return {n: (("embed.weight",) if n.startswith(NO_SUCH_NAME) else (n,)) for n in logical}


class FlipFlopsSupport(CanonicalView):
    """Declares ``trainable=False`` only the first time it is asked."""

    def __init__(self, root):
        super().__init__(root)
        self.asked = 0

    def support(self):
        self.asked += 1
        if self.asked == 1:
            return TunableSupport(trainable=False, reason="remote execution")
        return TunableSupport(trainable=True)


class UndeclaredRoles(CanonicalView):
    def support(self):
        return TunableSupport(trainable=True, declared_roles=frozenset({"base", "adapter", "intervention"}))


class LaundersAFrozenTensor(CanonicalView):
    """Maps a ``fitted`` parameter to a logical name that resolves elsewhere, so scheduling it is not refused."""

    def to_logical(self, physical):
        return ["model.0.weight" if p == "procrustes" else p for p in physical]


class KeepsTiedStorage(CanonicalView):
    """Promotes without materializing, so a promoted tensor tied to another keeps sharing its storage."""

    def promote(self, physical, role):
        return {n: n for n in physical}


def _tied_lens_model():
    model = InterventionModel()
    model.lens_alias = nn.Linear(D, V)
    model.lens_alias.weight = model.lens.weight
    return model


class _ParamNet(nn.Sequential):
    def __init__(self):
        from torch.nn.utils import parametrize

        super().__init__(nn.Linear(4, 4), nn.Linear(4, 2))

        class _Neg(nn.Module):
            def forward(self, w):
                return -w

        parametrize.register_parametrization(self[0], "weight", _Neg())


class PhysicalOnlyView(CanonicalView):
    """Resolves only physical names, so a parametrized tensor is unreachable by its unparametrized name."""

    def to_physical(self, logical):
        params = {n for n, _ in self.root.named_parameters()}
        unresolved = [n for n in logical if n not in params]
        if unresolved:
            raise Unresolvable(unresolved)
        return {n: (n,) for n in logical}


MUTANTS = {
    "mapping_completeness": lambda: ListsAnUnmappedName(TinyTransformer()),
    "mapping_round_trip": lambda: LosesARoundTrip(TinyTransformer()),
    "default_schedule_compiles": lambda: SplitsTiedNames(TinyTransformer()),
    "thaw_is_exact_and_deterministic": lambda: DriftingMapping(TinyTransformer()),
    "step_changes_only_thawed": lambda: HidesSharedStorage(_HiddenAliasNet()),
    "checkpoint_round_trip": lambda: IgnoresCheckpointLoad(parametrized_view().root),
    "unmatched_rule_refuses": lambda: MatchesEverything(TinyTransformer()),
    "untrainable_refuses": lambda: FlipFlopsSupport(nn.Linear(2, 2)),
    "roles_valid": lambda: UndeclaredRoles(InterventionModel(), roles=ROLES),
    "frozen_roles_refuse": lambda: LaundersAFrozenTensor(InterventionModel(), roles=ROLES),
    "promotion": lambda: KeepsTiedStorage(_tied_lens_model(), roles=ROLES),
    "parametrization_mapping": lambda: PhysicalOnlyView(_ParamNet()),
}


@pytest.mark.parametrize("case", list(MUTANTS), ids=list(MUTANTS))
def test_every_case_fails_on_its_mutant(case):
    suite = TunableViewConformance(MUTANTS[case])
    assert case in {c.name for c in suite.cases()}
    report = suite.run(select=[case])
    assert report.ran == [case], report
    assert case in report.failures, f"the {case} case passed on a view built to fail it"


def test_every_case_has_a_mutant():
    assert {c.name for c in TunableViewConformance(one_to_many_view).cases()} == set(MUTANTS)


def test_vacuity_guard():
    suite = TunableViewConformance(one_to_many_view)
    with pytest.raises(ConformanceError, match="No conformance case applied"):
        suite.run(select=["untrainable_refuses"])  # applies only to an untrainable view
    with pytest.raises(ConformanceError, match="Conformance failures"):
        TunableViewConformance(MUTANTS["checkpoint_round_trip"]).run(raise_on_failure=True)


def test_untrainable_view_runs_only_the_refusal_case():
    class Remote(CanonicalView):
        def support(self):
            return TunableSupport(trainable=False, reason="remote execution")

    report = TunableViewConformance(lambda: Remote(nn.Linear(2, 2))).run(raise_on_failure=True)
    assert report.ran == ["untrainable_refuses"]


ISOLATED = r'''
import importlib.abc
import sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"pytest", "_pytest", "lightning", "pytorch_lightning", "lightning_fabric"}:
            raise ModuleNotFoundError(f"No module named {fullname!r} (blocked)", name=fullname)
        return None


sys.meta_path.insert(0, Block())
import finetuning_scheduler  # noqa: E402

assert "finetuning_scheduler.testing" not in sys.modules, "importing the package imported its testing utilities"
from finetuning_scheduler.testing import TunableViewConformance  # noqa: E402
from finetuning_scheduler.testing.fixtures import one_to_many_view  # noqa: E402

assert TunableViewConformance(one_to_many_view).run(raise_on_failure=True).ok
print("ok")
'''


def test_testing_package_needs_neither_pytest_nor_lightning():
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([sys.executable, "-c", ISOLATED], capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr


def test_buffer_only_fitted_roles_conform():
    """A view whose only `fitted` tensors are buffers conforms (the frozen-parameter case does not apply)."""
    make = lambda: CanonicalView(InterventionModel(), roles={"enforce_mean": "fitted",  # noqa: E731
                                                             "enforce_mask": "fitted"})
    report = TunableViewConformance(make).run(raise_on_failure=True)
    assert "frozen_roles_refuse" in report.skipped and "checkpoint_round_trip" in report.ran


class DropsFittedBuffersOnLoad(CanonicalView):
    def load_checkpoint_state_dict(self, state_dict, strict=True):
        kept = {k: v for k, v in state_dict.items() if not k.startswith("enforce_")}
        self.root.load_state_dict(kept, strict=False)


def test_checkpoint_case_catches_unrestored_buffers():
    suite = TunableViewConformance(lambda: DropsFittedBuffersOnLoad(InterventionModel(), roles=ROLES))
    report = suite.run(select=["checkpoint_round_trip"])
    assert "checkpoint_round_trip" in report.failures and "enforce_mean" in report.failures["checkpoint_round_trip"]
