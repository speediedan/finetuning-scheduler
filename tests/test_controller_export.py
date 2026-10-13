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
