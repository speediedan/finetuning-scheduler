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
"""Tests for the tunable view protocol and parameter roles.

These need only PyTorch and the scheduling core (the ``core-no-lightning`` job runs them with ``--noconftest``).
"""
import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize
from venngram.errors import RoleError, VenngramError

from finetuning_scheduler import CanonicalView, TunableSupport, TunableView, TunableViewExtras, ViewDefaults
from finetuning_scheduler.core import roles as R
from finetuning_scheduler.tunable_view import roles as protocol_roles
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.views import (as_view, BoundView, pair_groups, resolve_view, TunableViewError,
                                             Unresolvable, VIEW_ATTR)


def _net():
    return nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))


class _Double(nn.Module):
    def forward(self, w):
        return 2 * w


class LogicalView:
    """A structural view that inherits nothing: logical ``block<i>`` names stand for a ``Linear``'s weight and bias."""

    def __init__(self, root):
        self._root = root
        self.support_calls = 0

    @property
    def root(self):
        return self._root

    def support(self):
        self.support_calls += 1
        return TunableSupport(trainable=True, naming_scheme="blocks")

    def to_physical(self, logical):
        table = {"block0": ("0.weight", "0.bias"), "block2": ("2.weight", "2.bias")}
        unresolved = [n for n in logical if n not in table]
        if unresolved:
            raise Unresolvable(unresolved)
        return {n: table[n] for n in logical}

    def to_logical(self, physical):
        rev = {"0.weight": "block0", "0.bias": "block0", "2.weight": "block2", "2.bias": "block2"}
        return list(dict.fromkeys(rev.get(n, n) for n in physical))


class LogicalViewWithNames(LogicalView):
    def logical_named_parameters(self):
        yield "block0", self._root[0].weight
        yield "block2", self._root[2].weight

    def coupled(self, physical):
        return ["1.weight", "1.bias"] if "0.weight" in physical else []


# ----------------------------------------------------------------------------------------------------------------------
# roles
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("role, expected", [
    ("base", ("base", None)), ("adapter", ("adapter", None)), ("adapter:char", ("adapter", "char")),
    ("intervention:steer.v1", ("intervention", "steer.v1")), ("latent:transcoder", ("latent", "transcoder")),
    ("fitted", ("fitted", None)), ("instrument", ("instrument", None)),
])
def test_parse_role(role, expected):
    assert R.parse_role(role) == expected


@pytest.mark.parametrize("role, match", [
    ("weights", "Unknown role"), ("latent", "needs a name"), ("base:x", "takes no name"),
    ("fitted:pen0", "takes no name"), ("adapter:a b", "may contain only"), (3, "must be a string"),
])
def test_parse_role_refuses(role, match):
    with pytest.raises(FTSConfigurationError, match=match):
        R.parse_role(role)
    # the protocol raises its own error; Fine-Tuning Scheduler's re-export raises a configuration error
    with pytest.raises(RoleError, match=match) as excinfo:
        protocol_roles.parse_role(role)
    assert not isinstance(excinfo.value, FTSConfigurationError)


def test_selectors():
    assert R.selector_matches("role:adapter", "adapter")
    assert R.selector_matches("role:adapter", "adapter:char")
    assert R.selector_matches("role:adapter:char", "adapter:char")
    assert not R.selector_matches("role:adapter:char", "adapter:word")
    assert not R.selector_matches("role:adapter:char", "adapter")
    assert R.selector_matches("role:latent", "latent:transcoder")
    assert not R.selector_matches("role:base", "fitted")
    assert R.is_role_selector("role:base") and not R.is_role_selector("model.0.weight")
    with pytest.raises(FTSConfigurationError, match="unknown role kind"):
        R.parse_selector("role:weights")
    with pytest.raises(FTSConfigurationError, match="not a role selector"):
        R.parse_selector("adapter")
    names = ["a", "b", "c", "d"]
    roles = {"b": "adapter:char", "c": "adapter", "d": "fitted"}
    assert R.select("role:adapter", names, roles) == ["b", "c"]
    assert R.select("role:base", names, roles) == ["a"]  # unlisted names are `base`
    assert R.is_frozen_role("fitted") and R.is_frozen_role("instrument") and not R.is_frozen_role("latent:x")


def test_validate_roles_covers_parameters_and_buffers():
    net = _net()
    params = [n for n, _ in net.named_parameters()]
    buffers = [n for n, _ in net.named_buffers()]
    roles = R.validate_roles({"1.running_mean": "fitted", "2.weight": "adapter:head"}, params, buffers)
    assert R.declared_kinds(roles) == frozenset({"base", "fitted", "adapter"})
    with pytest.raises(FTSConfigurationError, match=r"\['3.weight', 'nope'\]"):
        R.validate_roles({"3.weight": "base", "nope": "base"}, params, buffers)
    with pytest.raises(FTSConfigurationError, match="Unknown role"):
        R.validate_roles({"0.weight": "weights"}, params, buffers)


# ----------------------------------------------------------------------------------------------------------------------
# the support record and the canonical view
# ----------------------------------------------------------------------------------------------------------------------


def test_support_record():
    assert TunableSupport(trainable=True).declared_roles == frozenset({"base"})
    assert TunableSupport(trainable=False, reason="remote execution").reason == "remote execution"
    with pytest.raises(TunableViewError, match="must give a `reason`"):
        TunableSupport(trainable=False)


def test_canonical_view_identity_and_refusal():
    net = _net()
    view = CanonicalView(net)
    assert isinstance(view, TunableView) and isinstance(view, TunableViewExtras)
    assert view.to_physical(["0.weight", "2.bias"]) == {"0.weight": ("0.weight",), "2.bias": ("2.bias",)}
    assert view.to_logical(["0.weight", "0.weight", "2.bias"]) == ["0.weight", "2.bias"]
    with pytest.raises(Unresolvable, match=r"\['0.wieght', 'head.bias'\]") as excinfo:
        view.to_physical(["0.wieght", "0.bias", "head.bias"])
    assert excinfo.value.names == ["0.wieght", "head.bias"]
    assert isinstance(excinfo.value, TunableViewError) and not isinstance(excinfo.value, FTSConfigurationError)
    assert view.support() == TunableSupport(trainable=True)
    assert [n for n, _ in view.logical_named_parameters()] == [n for n, _ in net.named_parameters()]


def test_canonical_view_parametrizations():
    """A parametrized tensor resolves by its unparametrized name to the underlying tensor."""
    net = _net()
    parametrize.register_parametrization(net[0], "weight", _Double())
    view = CanonicalView(net)
    assert "0.weight" not in dict(net.named_parameters())
    assert view.to_physical(["0.weight"]) == {"0.weight": ("0.parametrizations.weight.original",)}
    phys = "0.parametrizations.weight.original"
    assert view.to_physical([phys]) == {phys: (phys,)}


def test_canonical_view_roles():
    net = _net()
    view = CanonicalView(net, roles={"1.running_mean": "fitted", "1.running_var": "fitted", "2.weight": "adapter:x"})
    assert view.support().declared_roles == frozenset({"base", "fitted", "adapter"})
    assert view.roles()["1.running_mean"] == "fitted"
    with pytest.raises(RoleError, match="neither a parameter nor a buffer"):
        CanonicalView(net, roles={"0.missing": "fitted"})


# ----------------------------------------------------------------------------------------------------------------------
# binding, defaults and discovery
# ----------------------------------------------------------------------------------------------------------------------


def test_structural_view_needs_no_inheritance():
    view = LogicalView(_net())
    assert isinstance(view, TunableView)
    assert not isinstance(view, TunableViewExtras)
    bound = as_view(view)
    assert as_view(bound) is bound
    assert bound.to_physical(["block0"]) == {"block0": ("0.weight", "0.bias")}
    assert bound.to_logical(["0.weight", "0.bias", "1.weight"]) == ["block0", "1.weight"]
    assert bound.coupled(["0.weight"]) == [] and bound.roles() == {}
    assert not bound.defines("coupled")
    with pytest.raises(TunableViewError, match="must define `logical_named_parameters`"):
        list(bound.logical_named_parameters())
    bound.support()
    bound.support()
    assert view.support_calls == 1  # read once, then cached


def test_bound_view_prefers_the_views_members():
    net = _net()
    bound = as_view(LogicalViewWithNames(net))
    assert bound.defines("coupled") and bound.defines("logical_named_parameters")
    assert bound.coupled(["0.weight", "0.bias"]) == ["1.weight", "1.bias"]
    assert bound.depth_groups() == [["block2", "block0"]]  # the default, over the view's logical names


def test_defaults_checkpoint_round_trip_and_promote():
    net = _net()
    bound = as_view(CanonicalView(net))
    sd = {k: v.clone() for k, v in bound.checkpoint_state_dict().items()}
    with torch.no_grad():
        net[0].weight.add_(1.0)
    bound.load_checkpoint_state_dict(sd)
    assert torch.equal(net[0].weight, sd["0.weight"])
    assert bound.promote(["2.weight"], "latent:t") == {"2.weight": "2.weight"}


def test_view_defaults_inheritance():
    class Inherited(ViewDefaults):
        def __init__(self, root):
            self.root = root

        def support(self):
            return TunableSupport(trainable=True)

        def to_physical(self, logical):
            return {n: (n,) for n in logical}

        def to_logical(self, physical):
            return list(physical)

    net = _net()
    view = Inherited(net)
    assert isinstance(view, TunableView) and isinstance(view, TunableViewExtras)
    assert view.depth_groups() == pair_groups([n for n, _ in net.named_parameters()])
    assert pair_groups(["a", "b", "c", "d"]) == [["d", "c"], ["b", "a"]]


def test_resolve_view_order():
    net = _net()
    explicit, attached = LogicalView(net), LogicalViewWithNames(net)
    setattr(net, VIEW_ATTR, attached)
    assert resolve_view(net, explicit).view is explicit
    assert resolve_view(net).view is attached
    delattr(net, VIEW_ATTR)
    canonical = resolve_view(net)
    assert isinstance(canonical, BoundView) and isinstance(canonical.view, CanonicalView)
    assert canonical.root is net


def test_resolve_view_refusals():
    with pytest.raises(TunableViewError, match="lacks `support`, `to_physical`, `to_logical`"):
        resolve_view(_net(), view=type("NotAView", (), {"root": None})())

    class BadSupport(LogicalView):
        def support(self):
            return {"trainable": True}

    with pytest.raises(TunableViewError, match="must return a `TunableSupport`"):
        resolve_view(_net(), BadSupport(_net())).support()


def test_instrument_keys_cover_aliases_and_wrapper_prefixes():
    from finetuning_scheduler.core.views import canonical_fqn, instrument_state_keys
    from finetuning_scheduler.testing.fixtures import TiedNet

    tied = TiedNet()
    view = CanonicalView(tied, roles={"embed.weight": "instrument"})
    assert {"embed.weight", "out.weight"} <= set(tied.state_dict())
    assert not {"embed.weight", "out.weight"} & set(view.checkpoint_state_dict())
    with pytest.raises(RuntimeError, match="unexpected keys"):
        view.load_checkpoint_state_dict({**view.checkpoint_state_dict(), "bogus": torch.zeros(1)})

    class Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    wrapped = Wrapper(nn.Linear(2, 2))
    roles = {"_orig_mod.weight": "instrument"}
    assert instrument_state_keys(wrapped, roles, ["weight", "bias"]) == {"weight"}  # a DSD-style unprefixed key
    assert canonical_fqn("_orig_mod.weight") == "weight"


def test_a_member_set_to_a_value_is_refused():
    """``roles = {...}`` rather than a method would otherwise be silently replaced by the default (``{}``),
    compiling an ``instrument`` tensor into a phase."""
    view = CanonicalView(_net())
    view.roles = {"0.weight": "instrument"}  # type: ignore[assignment]
    with pytest.raises(TunableViewError, match="`roles` must be methods"):
        BoundView(view)


def test_a_member_set_on_the_instance_is_used_and_counts_as_an_override():
    """``overrides`` agrees with the member actually called, so the guards keyed on it see an instance's own."""
    view = CanonicalView(_net())
    bound = BoundView(view)
    assert not bound.overrides("checkpoint_state_dict")
    view.checkpoint_state_dict = lambda: {"own": torch.zeros(1)}  # type: ignore[method-assign]
    assert bound.overrides("checkpoint_state_dict") and list(bound.checkpoint_state_dict()) == ["own"]


@pytest.mark.parametrize("role", ["adapter", "bogus-role", "latent", "instrument"])
def test_promote_refuses_a_target_that_is_not_a_named_latent_role(role):
    bound = BoundView(CanonicalView(_net(), roles={"0.weight": "instrument"}))
    with pytest.raises(VenngramError):  # a role error for an invalid role, a view error for a valid non-latent one
        bound.promote(["0.weight"], role)
    assert bound.promoted == {}
