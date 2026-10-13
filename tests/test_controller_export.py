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
"""``FTSController.export_best`` (role-selected exports of the best checkpoint) and the ``artifacts`` metadata
(#52).

These need only PyTorch and the scheduling core (the ``core-no-lightning`` job runs them with ``--noconftest``).
"""
from functools import partial

import pytest
import torch

from finetuning_scheduler import CanonicalView, FTSController
from finetuning_scheduler.core.checkpoint import BestCheckpoint, load_payload
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.testing.fixtures import InterventionModel, ROLES

SGD = partial(torch.optim.SGD, lr=0.1)
SCHEDULE = {0: {"params": ["role:adapter"], "max_transition_epoch": 2}, 1: {"params": ["role:latent"]}}
ADAPTERS = ["adapter_backbone.weight", "adapter_char.weight"]
ARTIFACTS = {"lens": {"path": "hub://org/lens", "sha256": "ab" * 32, "revision": "v1"}}


def _run(tmp_path, losses=(1.0, 0.5, 0.9, 0.8), **kwargs):
    """Train with synthetic gradients; the best checkpoint is epoch 1's."""
    torch.manual_seed(0)
    model = InterventionModel()
    fts = FTSController(model, {k: dict(v) for k, v in SCHEDULE.items()}, optimizer=SGD,
                        view=CanonicalView(model, roles=ROLES), checkpoint=BestCheckpoint(tmp_path, "loss"),
                        restore_best=False, **kwargs)
    fts.setup()
    at_best = None
    for epoch in fts.epochs(len(losses)):
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, epoch + 1.0)
        fts.optimizer.step()
        fts.optimizer.zero_grad()
        fts.epoch_end({"loss": losses[epoch]})
        if epoch == 1:
            at_best = {n: p.detach().clone() for n, p in model.named_parameters()}
    return model, fts, at_best


def test_export_best_writes_the_best_checkpoints_role_selected_tensors(tmp_path):
    model, fts, at_best = _run(tmp_path, artifacts=ARTIFACTS)
    path = fts.export_best()
    export = load_payload(path)
    assert sorted(export["tensors"]) == ADAPTERS and export["roles"] == {n: ROLES[n] for n in ADAPTERS}
    # the best checkpoint's values, which training has since moved past
    assert all(torch.equal(export["tensors"][n], at_best[n]) for n in ADAPTERS)
    assert not torch.equal(export["tensors"][ADAPTERS[0]], model.adapter_backbone.weight)
    meta = export["metadata"]
    assert meta["artifacts"] == ARTIFACTS and meta["best_depth"] == 0 and meta["best_score"] == 0.5
    assert load_payload(fts._checkpoint.best_path)["metadata"]["artifacts"] == ARTIFACTS


def test_export_best_selectors_and_a_custom_writer(tmp_path):
    _, fts, at_best = _run(tmp_path)
    seen = {}

    def writer(tensors, metadata, path):
        seen.update(tensors=tensors, roles=metadata["roles"], path=path)
        return "written"

    assert fts.export_best(writer, roles=("role:adapter:char", "fitted")) == "written"
    assert sorted(seen["tensors"]) == ["adapter_char.weight", "enforce_mask", "enforce_mean", "procrustes"]
    assert seen["path"] is None and seen["roles"]["procrustes"] == "fitted"
    assert torch.equal(seen["tensors"]["adapter_char.weight"], at_best["adapter_char.weight"])


@pytest.mark.parametrize("roles, match", [
    (("instrument",), "never checkpointed"),
    (("intervention:none",), "No tensor's role matches"),
])
def test_export_best_refusals(tmp_path, roles, match):
    _, fts, _ = _run(tmp_path)
    with pytest.raises(FTSConfigurationError, match=match):
        fts.export_best(roles=roles)


def test_export_best_needs_a_best_checkpoint(tmp_path):
    model = InterventionModel()
    fts = FTSController(model, {k: dict(v) for k, v in SCHEDULE.items()}, optimizer=SGD,
                        view=CanonicalView(model, roles=ROLES), checkpoint=BestCheckpoint(tmp_path, "loss"))
    fts.setup()
    with pytest.raises(FTSConfigurationError, match="no best checkpoint"):
        fts.export_best()


@pytest.mark.parametrize("artifacts, match", [
    ({"lens": {"path": "x"}}, "sha256"),
    ({"lens": {"sha256": "ab" * 32}}, "needs a `path`"),
    ({"lens": {"path": "x", "sha256": "zz" * 32}}, "64 hexadecimal"),
    ({"lens": {"path": "x", "sha256": "ab" * 32, "url": "y"}}, "unknown keys"),
    ({"lens": "x"}, "must be a mapping"),
])
def test_artifacts_are_validated(artifacts, match):
    with pytest.raises(FTSConfigurationError, match=match):
        FTSController(InterventionModel(), SCHEDULE, optimizer=SGD, artifacts=artifacts)


def _step(fts, epoch, metrics):
    for p in fts.model.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, epoch + 1.0)
    fts.optimizer.step()
    fts.optimizer.zero_grad()
    fts.epoch_end(metrics)


def test_export_best_keys_wrapped_tensors_by_their_checkpoint_names(tmp_path):
    """An activation-checkpointed layer's tensors are exported under the wrapper-free names the checkpoint uses."""
    from torch import nn
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    model[2] = checkpoint_wrapper(model[2])
    roles = {n: "adapter:head" for n, _ in model.named_parameters() if n.startswith("2.")}
    sched = {0: {"params": ["role:adapter"], "max_transition_epoch": 1}, 1: {"params": ["0.weight", "0.bias"]}}
    fts = FTSController(model, sched, optimizer=SGD, view=CanonicalView(model, roles=roles),
                        checkpoint=BestCheckpoint(tmp_path, "loss"), restore_best=False)
    fts.setup()
    for epoch, loss in zip(fts.epochs(2), (0.1, 1.0)):
        _step(fts, epoch, {"loss": loss})
    export = load_payload(fts.export_best())
    assert sorted(export["tensors"]) == ["2.bias", "2.weight"] and export["roles"]["2.weight"] == "adapter:head"


def test_export_best_takes_a_tensor_promoted_after_the_best_at_its_promotion_value(tmp_path):
    """A latent tensor promoted after the best checkpoint is exported with its value at promotion, as restoring the
    best does."""
    promote = {0: {"params": ["role:latent"], "max_transition_epoch": 1},
               1: {"params": ["role:latent:lens"], "promote": {"role:instrument": "latent:lens"}}}
    torch.manual_seed(0)
    model = InterventionModel()
    lens = model.lens.weight.detach().clone()
    fts = FTSController(model, promote, optimizer=SGD, view=CanonicalView(model, roles=ROLES),
                        checkpoint=BestCheckpoint(tmp_path, "loss"), restore_best=False)
    fts.setup()
    for epoch, loss in zip(fts.epochs(3), (0.1, 1.0, 1.0)):
        _step(fts, epoch, {"loss": loss})
    assert fts._checkpoint.best_depth == 0 and not torch.equal(model.lens.weight, lens)  # trained since promotion
    export = load_payload(fts.export_best(roles=("latent:lens",)))
    assert torch.equal(export["tensors"]["lens.weight"], lens)


def test_export_best_defaults_beside_the_best_and_records_its_metadata(tmp_path):
    _, fts, _ = _run(tmp_path / "ckpts", artifacts=ARTIFACTS)
    path = fts.export_best()
    assert path == str(tmp_path / "ckpts" / "best_export.pt")
    meta = load_payload(path)["metadata"]
    assert meta["config_hash"] == fts.config_hash and meta["source"] == fts._checkpoint.best_path
    assert "gate_records" in meta  # the best checkpoint's own metadata


def test_artifacts_are_normalized():
    upper = {"lens": {"path": "x", "sha256": "AB" * 32}}
    fts = FTSController(InterventionModel(), SCHEDULE, optimizer=SGD, artifacts=upper)
    assert fts._artifacts["lens"]["sha256"] == "ab" * 32
    from pathlib import Path

    assert FTSController(InterventionModel(), SCHEDULE, optimizer=SGD,
                         artifacts={"lens": {"path": Path("x"), "sha256": "ab" * 32}})._artifacts["lens"]["path"] == "x"
    with pytest.raises(FTSConfigurationError, match="non-empty strings"):
        FTSController(InterventionModel(), SCHEDULE, optimizer=SGD, artifacts={1: {"path": "x", "sha256": "ab" * 32}})
