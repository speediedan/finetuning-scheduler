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
"""Tests for promoting ``instrument`` tensors to ``latent`` roles at a phase boundary (``promote``, #53).

These need only PyTorch and the scheduling core (the ``core-no-lightning`` job runs them with ``--noconftest``).
"""
import warnings
from functools import partial

import pytest
import torch

from finetuning_scheduler import CanonicalView, TuningScheduler
from finetuning_scheduler.core.checkpoint import BestCheckpoint, load_payload
from finetuning_scheduler.core.compile import compile_schedule
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.schedule import refuse_direct_api_keys
from finetuning_scheduler.core.transitions import GateResult
from finetuning_scheduler.testing.fixtures import InterventionModel, ROLES
from venngram.tuning import TunableViewError

SGD = partial(torch.optim.SGD, lr=0.1)
LENS = ["lens.bias", "lens.weight"]
PROMOTE_SCHEDULE = {
    0: {"params": ["role:latent"], "max_transition_epoch": 1},
    1: {"params": ["role:latent:lens"], "promote": {"role:instrument": "latent:lens"}, "max_transition_epoch": 3},
    2: {"params": ["role:adapter:char"]},
}


def _model():
    torch.manual_seed(0)
    model = InterventionModel()
    return model, CanonicalView(model, roles=ROLES)


def _tuning_scheduler(model, view, schedule=PROMOTE_SCHEDULE, **kwargs):
    fts = TuningScheduler(model, {k: dict(v) for k, v in schedule.items()}, optimizer=SGD, view=view, **kwargs)
    fts.setup()
    return fts


def _epoch(fts, epoch, metrics=None):
    """One step with synthetic gradients (every trainable tensor's gradient is ``epoch + 1``), then end the
    epoch."""
    for p in fts.model.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, epoch + 1.0)
    fts.optimizer.step()
    fts.optimizer.zero_grad()
    fts.epoch_end(metrics)


def _lens(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if n in LENS}


def test_promotion_makes_an_instrument_trainable_from_its_phase():
    """The promoted lens stays frozen and out of the optimizer until its phase starts, then trains as a latent."""
    model, view = _model()
    fts = _tuning_scheduler(model, view)
    assert fts.roles["lens.weight"] == "instrument" and sorted(fts.compiled_schedule.physical_for(1)) == LENS
    initial, seen = _lens(model), []
    for epoch in fts.epochs(4):
        seen.append((fts.curr_depth, sorted(n for n in LENS if dict(model.named_parameters())[n].requires_grad)))
        _epoch(fts, epoch)
    assert seen == [(0, []), (1, LENS), (1, LENS), (2, LENS)]
    assert fts.roles["lens.weight"] == "latent:lens" and fts.compiled_schedule.roles["lens.weight"] == "instrument"
    assert all(not torch.equal(initial[n], t) for n, t in _lens(model).items())
    assert fts.state_dict()["promotions"] == [
        {"depth": 1, "names": LENS, "from": "instrument", "to": "latent:lens", "materialized": []}]


def test_promoted_tensors_are_checkpointed_from_their_promotion(tmp_path):
    model, view = _model()
    cp = BestCheckpoint(tmp_path, "loss")
    fts = _tuning_scheduler(model, view, checkpoint=cp, restore_best=False)
    saved = []
    for epoch in fts.epochs(2):
        _epoch(fts, epoch, {"loss": 1.0 - epoch})
        saved.append(load_payload(cp.best_path))
    assert [sorted(set(LENS) & set(p["model"])) for p in saved] == [[], LENS]
    assert saved[1]["metadata"]["promotions"][0]["to"] == "latent:lens"


def test_resume_replays_promotions_before_loading(tmp_path):
    """A state saved after a promotion resumes as the uninterrupted run continues: the promotion is replayed before
    the saved state (which holds the promoted tensors) loads."""
    ref_model, view = _model()
    ref = _tuning_scheduler(ref_model, view)
    for epoch in ref.epochs(4):
        _epoch(ref, epoch)
    model, view = _model()
    first = _tuning_scheduler(model, view)
    for epoch in first.epochs(4):
        _epoch(first, epoch)
        if epoch == 1:
            first.save(tmp_path / "s.pt")
            break
    model, view = _model()
    resumed = _tuning_scheduler(model, view)
    resumed.load(tmp_path / "s.pt")
    assert resumed.roles["lens.weight"] == "latent:lens" and len(resumed.state_dict()["promotions"]) == 1
    for epoch in resumed.epochs(4):
        _epoch(resumed, epoch)
    for (n, p), (_, q) in zip(ref_model.named_parameters(), model.named_parameters()):
        assert torch.equal(p, q), n


def test_vetoing_the_promoting_phase_restores_the_values_at_promotion(tmp_path):
    """Rolling back to a checkpoint saved before the promotion puts the promoted tensors back to their values when
    they were promoted (they were not checkpointed then)."""
    model, view = _model()
    initial = _lens(model)
    fts = _tuning_scheduler(model, view, checkpoint=BestCheckpoint(tmp_path, "loss"))
    fts.add_gate(lambda e: GateResult(accept=e.depth != 1, reason="the lens phase hurt"))
    for epoch in fts.epochs(4):
        _epoch(fts, epoch, {"loss": 1.0 - 0.1 * epoch})
    assert fts.stop_reason.startswith("veto: phase 1")
    assert all(torch.equal(initial[n], t) for n, t in _lens(model).items())


def test_promoting_a_tied_instrument_gives_it_its_own_storage():
    """A promoted parameter that shares storage with another tensor is replaced by a copy, so training it leaves
    the other tensor unchanged."""
    model, view = _model()
    model.lens_tied = torch.nn.Linear(4, 4)  # placeholder, replaced by a tie below
    model.lens_tied.weight = model.lens.weight
    tied_before = model.lens_tied.weight.detach().clone()
    fts = _tuning_scheduler(model, view)
    for epoch in fts.epochs(3):
        _epoch(fts, epoch)
    assert fts.state_dict()["promotions"][0]["materialized"] == ["lens.weight"]
    assert model.lens.weight is not model.lens_tied.weight and torch.equal(model.lens_tied.weight, tied_before)
    assert not torch.equal(model.lens.weight, tied_before)


@pytest.mark.parametrize("promote, match", [
    ({"role:latent:transcoder": "latent:lens"}, "whose role is not `instrument`"),
    ({"role:instrument": "adapter:lens"}, "only to a `latent:<name>` role"),
    ({"role:instrument": "latent"}, "needs a name"),
    ({}, "non-empty mapping"),
])
def test_invalid_promotions_refuse(promote, match):
    model, view = _model()
    schedule = {**PROMOTE_SCHEDULE, 1: {**PROMOTE_SCHEDULE[1], "promote": promote}}
    with pytest.raises(FTSConfigurationError, match=match):
        _tuning_scheduler(model, view, schedule)


def test_phase_zero_promotion_and_the_lightning_callback_refuse():
    model, view = _model()
    schedule = {**PROMOTE_SCHEDULE, 0: {**PROMOTE_SCHEDULE[0], "promote": {"role:instrument": "latent:lens"}}}
    with pytest.raises(FTSConfigurationError, match="Phase 0 may not `promote`"):
        _tuning_scheduler(model, view, schedule)
    with pytest.raises(FTSConfigurationError, match="phase 1: promote"):
        refuse_direct_api_keys({k: dict(v) for k, v in PROMOTE_SCHEDULE.items()})


def test_a_view_that_renames_on_promotion_refuses():
    model, _ = _model()

    class Renaming(CanonicalView):
        def promote(self, physical, role):
            return {n: f"{n}_promoted" for n in physical}

    fts = _tuning_scheduler(model, Renaming(model, roles=ROLES))
    with pytest.raises(TunableViewError, match="must keep each tensor's name"):
        for epoch in fts.epochs(2):
            _epoch(fts, epoch)


def test_promotions_change_the_compiled_digest():
    model, view = _model()
    plain = {0: dict(PROMOTE_SCHEDULE[0]), 1: {"params": ["role:adapter:char"]}}
    promoting = {**plain, 1: {"params": ["role:adapter:char"], "promote": {"role:instrument": "latent:lens"}}}
    a, b = compile_schedule(plain, view), compile_schedule(promoting, view)
    assert a.digest != b.digest and b.promotions == {1: dict.fromkeys(LENS, "latent:lens")}
    assert b.roles_at(0)["lens.weight"] == "instrument" and b.roles_at(1)["lens.weight"] == "latent:lens"


def _tied_model():
    model, view = _model()
    model.lens_tied = torch.nn.Linear(model.lens.in_features, model.lens.out_features)
    model.lens_tied.weight = model.lens.weight
    return model, view


class OwnRolesView(CanonicalView):
    """Owns promotion (tracking roles itself) and overrides the checkpoint state, as a key-translating view
    does."""

    def promote(self, physical, role):
        out = super().promote(physical, role)
        self._roles.update(dict.fromkeys(physical, role))
        return out

    def checkpoint_state_dict(self):
        return dict(super().checkpoint_state_dict())


class CheckpointsWithoutPromote(CanonicalView):
    def checkpoint_state_dict(self):
        return dict(super().checkpoint_state_dict())


@pytest.mark.parametrize("make", ["tied", "own_roles"])
def test_rolling_back_before_a_promotion_restores_aliases_and_translated_views(tmp_path, make):
    """The rollback puts every tensor the promotion added to the checkpoint state back, including an alias the
    materialization untied, and works for a view that owns promotion and overrides its checkpoint state."""
    if make == "tied":
        model, view = _tied_model()
    else:
        torch.manual_seed(0)
        model = InterventionModel()
        view = OwnRolesView(model, roles=dict(ROLES))
    initial = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith("lens")}
    fts = _tuning_scheduler(model, view, checkpoint=BestCheckpoint(tmp_path, "loss"))
    fts.add_gate(lambda e: GateResult(accept=e.depth != 1, reason="the lens phase hurt"))
    for epoch in fts.epochs(4):
        _epoch(fts, epoch, {"loss": 1.0 - 0.1 * epoch})
    assert fts.stop_reason.startswith("veto: phase 1")
    state = model.state_dict()
    assert all(torch.equal(state[k], v) for k, v in initial.items())


class LeavesOutPromoted(OwnRolesView):
    def checkpoint_state_dict(self):
        return {k: v for k, v in super().checkpoint_state_dict().items() if not k.startswith("lens.")}


def test_a_view_whose_checkpoint_state_leaves_out_the_promoted_tensors_refuses():
    torch.manual_seed(0)
    model = InterventionModel()
    fts = _tuning_scheduler(model, LeavesOutPromoted(model, roles=dict(ROLES)))
    with pytest.raises(FTSConfigurationError, match="checkpoint state still leaves them out"):
        for epoch in fts.epochs(2):
            _epoch(fts, epoch)


def test_a_view_translating_checkpoints_without_promote_refuses():
    torch.manual_seed(0)
    model = InterventionModel()
    with pytest.raises(FTSConfigurationError, match="must define `promote`"):
        _tuning_scheduler(model, CheckpointsWithoutPromote(model, roles=dict(ROLES)))


def test_resume_after_a_promotion_keeps_the_configuration_hash_and_the_values_at_promotion(tmp_path):
    """A resume after a promotion sees the same configuration hash, and a later veto rolls back to the values the
    tensors had when promoted (after the phase's start hooks), not to the freshly built model's."""

    def prepare(event):
        if event.depth == 1:
            with torch.no_grad():
                model.lens.weight.fill_(5.0)

    def run(load_from=None, save_to=None):
        nonlocal model
        model, view = _model()
        fts = _tuning_scheduler(model, view, checkpoint=BestCheckpoint(tmp_path / "c", "loss"))
        fts.add_phase_hook(prepare)
        fts.add_gate(lambda e: GateResult(accept=e.depth != 1, reason="the lens phase hurt"))
        if load_from is not None:
            fts.load(load_from)
        for epoch in fts.epochs(4):
            _epoch(fts, epoch, {"loss": 1.0 - 0.1 * epoch})
            if save_to is not None and epoch == 1:
                fts.save(save_to)
                return fts
        return fts

    model = None
    ref = run()
    assert ref.stop_reason.startswith("veto: phase 1") and torch.equal(model.lens.weight, torch.full_like(
        model.lens.weight, 5.0))
    first = run(save_to=tmp_path / "s.pt")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "different tuning scheduler configuration" warning
        resumed = run(load_from=tmp_path / "s.pt")
    assert resumed.config_hash == first.config_hash and resumed.stop_reason.startswith("veto: phase 1")
    assert torch.equal(model.lens.weight, torch.full_like(model.lens.weight, 5.0))


def test_the_configuration_hash_survives_a_promotion_the_view_tracks_itself(tmp_path):
    """A view that owns promotion reports its promoted roles itself; the hash still uses the roles it reported at
    setup, so it is the same before and after the promotion (and a resume does not warn)."""
    torch.manual_seed(0)
    model = InterventionModel()
    fts = _tuning_scheduler(model, OwnRolesView(model, roles=dict(ROLES)))
    before = fts.config_hash
    for epoch in fts.epochs(3):
        _epoch(fts, epoch)
    assert fts.roles["lens.weight"] == "latent:lens" and fts.config_hash == before


def test_a_frozen_role_tensor_made_trainable_by_hand_is_refused_at_the_next_thaw():
    """The runtime guard: compilation keeps frozen roles out of every phase, and a tensor thawed outside the
    schedule is refused when the next phase thaws."""
    model, view = _model()
    fts = _tuning_scheduler(model, view, schedule={0: {"params": ["role:adapter"], "max_transition_epoch": 1},
                                                   1: {"params": ["role:latent"]}})
    model.lens.weight.requires_grad_(True)  # an `instrument` tensor, never promoted here
    with pytest.raises(FTSConfigurationError, match=r"never thawed .* require grad: \['lens.weight'\]"):
        for epoch in fts.epochs(3):
            _epoch(fts, epoch)
