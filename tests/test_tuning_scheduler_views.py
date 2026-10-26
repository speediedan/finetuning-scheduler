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
"""Tests for ``TuningScheduler`` scheduling a model through a tunable view.

These need only PyTorch and the scheduling core (the ``core-no-lightning`` job runs them with ``--noconftest``).
"""
from functools import partial

import pytest
import torch
from torch import nn

from finetuning_scheduler import CanonicalView, TuningScheduler, PlateauTransition, TunableSupport
from finetuning_scheduler.core.checkpoint import BestCheckpoint
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.testing.fixtures import InterventionModel, one_to_many_view, ROLES

SGD = partial(torch.optim.SGD, lr=0.1)


def _run(fts, epochs, metrics=None):
    """A plain loop with synthetic gradients: every trainable tensor's gradient is ones."""
    seen = []
    for epoch in fts.epochs(max_epochs=epochs):
        for p in fts.model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        fts.optimizer.step()
        fts.optimizer.zero_grad()
        trainable = sorted(n for n, p in fts.model.named_parameters() if p.requires_grad)
        seen.append((epoch, fts.curr_depth, trainable, len(fts.optimizer.param_groups)))
        fts.epoch_end(metrics(epoch) if metrics else None)
    return seen


def _seq():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 2))


SEQ_SCHEDULE = {0: {"params": [r"4\..*"], "max_transition_epoch": 1}, 1: {"params": [r"[23]\..*"],
                                                                         "max_transition_epoch": 2},
                2: {"params": [r"[01]\..*"]}}


def test_canonical_view_matches_no_view():
    """An explicit identity view changes nothing: the same path, the same parameters and the same hash."""
    runs, params, hashes = [], [], []
    for view in (None, "canonical"):
        model = _seq()
        fts = TuningScheduler(model, SEQ_SCHEDULE, optimizer=SGD,
                              view=CanonicalView(model) if view else None)
        fts.setup()
        hashes.append(fts.config_hash)
        runs.append(_run(fts, 4))
        params.append({n: p.detach().clone() for n, p in model.named_parameters()})
    assert runs[0] == runs[1]
    assert all(torch.equal(params[0][n], params[1][n]) for n in params[0])
    assert hashes[0] == hashes[1]


def test_logical_view_thaws_physical_and_coupled():
    view = one_to_many_view()
    model = view.root
    schedule = {0: {"params": ["unembed.W_U", "embed.W_E"], "max_transition_epoch": 1},
                1: {"params": ["blocks.1.mlp.W_out", "blocks.1.mlp.W_in", "blocks.1.attn.W_QK"]}}
    events = []
    fts = TuningScheduler(model, schedule, optimizer=SGD, view=view)
    fts.add_phase_hook(lambda e: events.append((e.depth, e.logical_params, e.physical_params)))
    fts.setup()
    seen = _run(fts, 3)
    assert seen[0][1] == 0 and set(seen[0][2]) == {"embed.weight", "ln_final.weight", "ln_final.bias"}
    block1 = {f"blocks.1.{m}.{t}" for m in ("q", "k", "mlp_in", "mlp_out", "ln") for t in ("weight", "bias")}
    assert set(seen[1][2]) == block1 | set(seen[0][2])
    assert events[0] == (0, ("unembed.W_U", "embed.W_E"), ("embed.weight", "ln_final.weight", "ln_final.bias"))
    assert events[1][0] == 1 and events[1][1] == ("blocks.1.mlp.W_out", "blocks.1.mlp.W_in", "blocks.1.attn.W_QK")
    assert set(events[1][2]) == block1
    assert fts.compiled_schedule.naming_scheme == "tiny-logical"
    assert fts.config_dict()["view"]["naming_scheme"] == "tiny-logical"
    assert "blocks.0.fused_qk" not in {n for n, p in model.named_parameters() if p.requires_grad}


def test_default_schedule_through_a_view():
    view = one_to_many_view()
    fts = TuningScheduler(view.root, None, optimizer=SGD, view=view, transition=PlateauTransition("loss"))
    fts.setup()
    assert len(fts.schedule) == 3  # the view's depth groups: the tied embedding, then one block at a time
    assert fts.schedule[0]["params"] == ["embed.weight", "ln_final.weight", "ln_final.bias"]


def test_attached_view_is_discovered():
    view = one_to_many_view()
    view.root.tunable_view = view
    fts = TuningScheduler(view.root, {0: {"params": ["embed.W_E"]}}, optimizer=SGD)
    fts.setup()
    assert fts.view.view is view


def _roles_model():
    torch.manual_seed(0)
    model = InterventionModel()
    return model, CanonicalView(model, roles=ROLES)


ROLE_SCHEDULE = {0: {"params": ["role:latent"], "max_transition_epoch": 1}, 1: {"params": ["role:adapter:char"]}}


def test_roles_select_and_frozen_roles_never_train():
    model, view = _roles_model()
    fts = TuningScheduler(model, ROLE_SCHEDULE, optimizer=SGD, view=view)
    fts.setup()
    seen = _run(fts, 2)
    assert seen[0][2] == ["transcoder.bias", "transcoder.weight"]
    assert seen[1][2] == ["adapter_char.weight", "transcoder.bias", "transcoder.weight"]
    assert fts.roles["lens.weight"] == "instrument"
    assert not {n for n, p in model.named_parameters() if p.requires_grad} & {"lens.weight", "procrustes"}


def test_scheduling_a_frozen_role_refuses():
    model, view = _roles_model()
    with pytest.raises(FTSConfigurationError, match=r"never thawed.*lens.weight"):
        TuningScheduler(model, {0: {"params": [r"lens\..*"]}}, optimizer=SGD, view=view).setup()


def test_optimizer_instance_holding_a_frozen_role_refuses():
    model, view = _roles_model()
    optimizer = torch.optim.SGD([model.transcoder.weight, model.lens.weight], lr=0.1)
    with pytest.raises(FTSConfigurationError, match=r"never optimized.*lens.weight"):
        TuningScheduler(model, ROLE_SCHEDULE, optimizer=optimizer, view=view, enforce_phase0_params=False).setup()


def test_role_aware_checkpoints_and_resume(tmp_path):
    model, view = _roles_model()
    fts = TuningScheduler(model, ROLE_SCHEDULE, optimizer=SGD, view=view,
                          checkpoint=BestCheckpoint(tmp_path / "ckpt", monitor="loss", include="scheduled"))
    fts.setup()
    _run(fts, 2, metrics=lambda e: {"loss": 1.0 - e})
    best = torch.load(fts._checkpoint.path_for("best"), weights_only=False)
    assert not {"lens.weight", "lens.bias"} & set(best["model"])
    assert {"procrustes", "enforce_mean", "enforce_mask", "transcoder.weight"} <= set(best["model"])
    fts.save(tmp_path / "last.pt")
    saved = torch.load(tmp_path / "last.pt", weights_only=False)
    assert saved["tuning_scheduler"]["view_digest"] == fts.compiled_schedule.digest
    assert not {"lens.weight", "lens.bias"} & set(saved["model"])
    reference = {n: p.detach().clone() for n, p in model.named_parameters()}

    model2, view2 = _roles_model()
    fts2 = TuningScheduler(model2, ROLE_SCHEDULE, optimizer=SGD, view=view2,
                           checkpoint=BestCheckpoint(tmp_path / "ckpt2", monitor="loss", include="scheduled"))
    fts2.setup()
    fts2.load(tmp_path / "last.pt")
    restored = dict(model2.named_parameters())
    assert all(torch.equal(restored[n], reference[n]) for n in reference if not n.startswith("lens."))

    roles = {**ROLES, "adapter_backbone.weight": "adapter:other"}
    model3 = InterventionModel()
    fts3 = TuningScheduler(model3, ROLE_SCHEDULE, optimizer=SGD, view=CanonicalView(model3, roles=roles))
    fts3.setup()
    with pytest.raises(FTSConfigurationError, match="compiled through a view that maps"):
        fts3.load(tmp_path / "last.pt")


def test_view_refusals():
    model = _seq()
    with pytest.raises(FTSConfigurationError, match="must be the module given to the tuning scheduler"):
        TuningScheduler(model, SEQ_SCHEDULE, optimizer=SGD, view=CanonicalView(_seq())).setup()

    class Remote(CanonicalView):
        def support(self):
            return TunableSupport(trainable=False, reason="remote execution")

    with pytest.raises(FTSConfigurationError, match="cannot be fine-tuned through Remote: remote execution"):
        TuningScheduler(model, SEQ_SCHEDULE, optimizer=SGD, view=Remote(model)).setup()


class MinimalView:
    """Only the four required members."""

    def __init__(self, root):
        self.root = root

    def support(self):
        return TunableSupport(trainable=True)

    def to_physical(self, logical):
        return {n: (n,) for n in logical}

    def to_logical(self, physical):
        return list(physical)


def test_minimal_view_hashes_saves_and_loads(tmp_path):
    """Optional members default everywhere, including the configuration hash and checkpoints."""
    model = _seq()
    fts = TuningScheduler(model, SEQ_SCHEDULE, optimizer=SGD, view=MinimalView(model))
    before = fts.config_dict()["view"]  # before setup, the view is described the same way
    fts.setup()
    assert fts.config_dict()["view"] == before
    _run(fts, 2)
    fts.save(tmp_path / "last.pt")
    model2 = _seq()
    fts2 = TuningScheduler(model2, SEQ_SCHEDULE, optimizer=SGD, view=MinimalView(model2))
    fts2.setup()
    fts2.load(tmp_path / "last.pt")
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), model2.parameters()))


def test_attached_view_is_described_before_setup():
    view = one_to_many_view()
    view.root.tunable_view = view
    fts = TuningScheduler(view.root, {0: {"params": ["embed.W_E"]}}, optimizer=SGD)
    before = fts.config_dict()["view"]
    fts.setup()
    assert fts.config_dict()["view"] == before


def test_canonical_strict_load_mismatch_raises_runtime_error(tmp_path):
    """Without roles, a strict load behaves as ``load_state_dict`` does."""
    model = _seq()
    fts = TuningScheduler(model, SEQ_SCHEDULE, optimizer=SGD)
    fts.setup()
    fts.save(tmp_path / "last.pt")
    other = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 2),
                          nn.Linear(2, 2))
    fts2 = TuningScheduler(other, SEQ_SCHEDULE, optimizer=SGD)
    fts2.setup()
    with pytest.raises(RuntimeError, match="Missing key"):
        fts2._view.load_checkpoint_state_dict(torch.load(tmp_path / "last.pt", weights_only=False)["model"])


def test_the_setup_rank_agreement_check_covers_roles():
    """Ranks whose views give a tensor different roles are refused at setup, naming roles, rather than failing at
    the first sharded save."""
    schedule = {0: {"params": ["role:adapter"], "max_transition_epoch": 1}, 1: {"params": ["role:latent"]}}
    gathered = []

    def tuning_scheduler(roles, other_rank=None):
        model = InterventionModel()
        fts = TuningScheduler(model, {k: dict(v) for k, v in schedule.items()}, optimizer=SGD,
                              view=CanonicalView(model, roles=roles))
        fts._interface.all_gather_object = lambda obj: gathered.append(obj) or [obj, other_rank or obj]
        return fts

    tuning_scheduler(ROLES).setup()
    tuning_scheduler({**ROLES, "procrustes": "instrument"}).setup()
    assert gathered[0] != gathered[1]  # the same physical schedule, different roles
    with pytest.raises(FTSConfigurationError, match="different schedules or roles"):
        tuning_scheduler(ROLES, other_rank=gathered[1]).setup()
