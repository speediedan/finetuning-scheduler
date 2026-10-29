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
"""Addresses in schedules (#153): one compile path serves the PyTorch API and the Lightning callback, so each
behaviour is checked through both."""
from functools import partial

import pytest
import torch
import yaml
from lightning.pytorch import seed_everything
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch import nn

from finetuning_scheduler import CanonicalView, FinetuningScheduler, TuningScheduler
from finetuning_scheduler.core.compile import compile_schedule
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.schedule import load_address_sets
from tests.test_legacy_adapter_contract import _trainer, PathRecorder

SGD = partial(torch.optim.SGD, lr=0.1)


class Blocks(nn.Module):
    """Four blocks, each a projection with a small adapter beneath it, and a head."""

    def __init__(self, d: int = 32, depth: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(nn.ModuleDict({"proj": nn.Linear(d, d), "adapter": nn.Linear(d, d, bias=False)})
                                    for _ in range(depth))
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        for b in self.blocks:
            x = b["proj"](x) + b["adapter"](x)
        return self.head(x)


def adapter_roles(model: nn.Module) -> dict[str, str]:
    return {n: "adapter" for n, _ in model.named_parameters() if ".adapter." in n}


ADDRESSED = {
    "addresses": {"sets": {"top": {"role": "adapter", "layers": "-2:"},
                           "rest": {"role": "adapter", "exclude": "@top"}}},
    0: {"params": ["@top"], "max_transition_epoch": 1},
    1: {"params": ["@rest"], "max_transition_epoch": 2},
    2: {"params": ["role:base & layers:2:", "role:base - layers:0:"], "max_transition_epoch": 3},
}


def _compiled(model, schedule, sets=None):
    return compile_schedule(schedule, CanonicalView(model, roles=adapter_roles(model)), addresses=sets)


def test_addresses_select_a_subset_of_a_role_through_the_view():
    model = Blocks()
    sets = load_address_sets(ADDRESSED["addresses"])
    compiled = _compiled(model, {d: p for d, p in ADDRESSED.items() if d != "addresses"}, sets)
    assert compiled.phases[0].physical == ("blocks.2.adapter.weight", "blocks.3.adapter.weight")
    assert compiled.phases[1].physical == ("blocks.0.adapter.weight", "blocks.1.adapter.weight")
    assert compiled.phases[2].physical == ("blocks.2.proj.weight", "blocks.2.proj.bias", "blocks.3.proj.weight",
                                           "blocks.3.proj.bias", "head.weight", "head.bias")


def test_overlapping_sets_compose_by_set_algebra():
    model = Blocks()
    compiled = _compiled(model, {0: {"params": ["role:adapter & layers:-2: & module:blocks.3"]},
                                 1: {"params": ["(role:adapter & layers:-2:) - module:blocks.3"]}})
    assert compiled.phases[0].physical == ("blocks.3.adapter.weight",)
    assert compiled.phases[1].physical == ("blocks.2.adapter.weight",)


def test_tuning_scheduler_runs_an_addressed_schedule():
    model = Blocks()
    fts = TuningScheduler(model, ADDRESSED, optimizer=SGD, view=CanonicalView(model, roles=adapter_roles(model)))
    fts.setup()
    seen = []
    for _ in fts.epochs(max_epochs=3):
        seen.append(sorted(n for n, p in model.named_parameters() if p.requires_grad))
        fts.epoch_end()
    assert seen[0] == ["blocks.2.adapter.weight", "blocks.3.adapter.weight"]
    assert seen[1] == sorted(seen[0] + ["blocks.0.adapter.weight", "blocks.1.adapter.weight"])


def test_tuning_scheduler_reads_address_books_relative_to_the_schedule_file(tmp_path):
    (tmp_path / "books").mkdir()
    (tmp_path / "books" / "lora.yaml").write_text(yaml.safe_dump(
        {"schema_version": 1, "sets": {"top": {"role": "adapter", "layers": -1}}}))
    sched = {"addresses": "books/lora.yaml", 0: {"params": ["@top"], "max_transition_epoch": 1},
             1: {"params": ["role:adapter - @top"]}}
    (tmp_path / "schedule.yaml").write_text(yaml.safe_dump(sched))
    model = Blocks()
    fts = TuningScheduler(model, tmp_path / "schedule.yaml", optimizer=SGD,
                          view=CanonicalView(model, roles=adapter_roles(model)))
    fts.setup()
    assert fts.compiled_schedule.phases[0].physical == ("blocks.3.adapter.weight",)


@pytest.mark.parametrize("schedule, match", [
    ({0: {"params": ["role:adapter & units:[0, 1]"]}}, "unit adapter"),
    ({0: {"params": ["point:resid_post & layers:1"]}}, "site resolver"),
    ({0: {"params": ["@missing"]}}, "no set is named `@missing`"),
    ({0: {"params": ["role:adapter & layers:[9]"]}}, "outside a stack of 4"),
    ({0: {"params": ["component:attn.q"]}}, "vocabulary structure provider"),
    ({0: {"params": [{"role": "adapter"}]}}, "is not a string"),
    ({"addresses": {"sets": {"a": "@b"}}, 0: {"params": ["@a"]}}, "no set or import defines"),
    ({"addresses": 3, 0: {"params": ["role:adapter"]}}, "an address book's path"),
])
def test_addresses_that_do_not_resolve_are_refused_by_name(schedule, match):
    model = Blocks()
    fts = TuningScheduler(model, schedule, optimizer=SGD, view=CanonicalView(model, roles=adapter_roles(model)))
    with pytest.raises(FTSConfigurationError, match=match):
        fts.setup()


def test_promotion_sources_take_addresses():
    model = Blocks()
    roles = {"blocks.3.adapter.weight": "instrument", "blocks.2.adapter.weight": "instrument"}
    promote = {"role:instrument & layers:-1:": "latent:t"}
    compiled = compile_schedule({0: {"params": ["module:head"]}, 1: {"params": ["role:latent:t"], "promote": promote}},
                                CanonicalView(model, roles=roles))
    assert compiled.promotions[1] == {"blocks.3.adapter.weight": "latent:t"}
    assert compiled.phases[1].physical == ("blocks.3.adapter.weight",)


def test_bare_names_and_regexes_keep_their_meaning():
    model = Blocks()
    compiled = _compiled(model, {0: {"params": ["head.weight", r"head\.b.*"]}, 1: {"params": [r"(blocks)\.0\..*"]}})
    assert compiled.phases[0].physical == ("head.weight", "head.bias")
    assert compiled.phases[1].physical == ("blocks.0.proj.weight", "blocks.0.proj.bias", "blocks.0.adapter.weight")


class BlocksBoringModel(BoringModel):
    def __init__(self) -> None:
        super().__init__()
        del self.layer
        self.net = Blocks()

    def forward(self, x):
        return self.net(x)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=False)
        return {"x": loss}

    def configure_optimizers(self):
        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)


def _lightning_module():
    module = BlocksBoringModel()
    module.tunable_view = CanonicalView(module, roles=adapter_roles(module))
    return module


def test_the_lightning_callback_runs_an_addressed_schedule(tmp_path):
    seed_everything(42)
    module = _lightning_module()
    fts, recorder = FinetuningScheduler(ft_schedule=dict(ADDRESSED), epoch_transitions_only=True), PathRecorder()
    _trainer(tmp_path, [fts, recorder], max_epochs=3).fit(module)
    assert recorder.rows[0][2] == ["net.blocks.2.adapter.weight", "net.blocks.3.adapter.weight"]
    assert recorder.rows[1][2] == ["net.blocks.0.adapter.weight", "net.blocks.1.adapter.weight",
                                   "net.blocks.2.adapter.weight", "net.blocks.3.adapter.weight"]


def test_the_lightning_callback_reads_address_books_relative_to_the_schedule_file(tmp_path):
    seed_everything(42)
    (tmp_path / "lora.yaml").write_text(yaml.safe_dump({"schema_version": 1,
                                                        "sets": {"top": {"role": "adapter", "layers": -1}}}))
    (tmp_path / "schedule.yaml").write_text(yaml.safe_dump(
        {"addresses": ["lora.yaml"], 0: {"params": ["@top"], "max_transition_epoch": 1},
         1: {"params": ["role:adapter - @top"], "max_transition_epoch": 2}}))
    fts, recorder = FinetuningScheduler(ft_schedule=str(tmp_path / "schedule.yaml"), epoch_transitions_only=True), \
        PathRecorder()
    _trainer(tmp_path, [fts, recorder], max_epochs=2).fit(_lightning_module())
    assert recorder.rows[0][2] == ["net.blocks.3.adapter.weight"]


def test_the_lightning_callback_refuses_an_unresolvable_address(tmp_path):
    fts = FinetuningScheduler(ft_schedule={0: {"params": ["role:adapter & layers:[9]"], "max_transition_epoch": 1}},
                              epoch_transitions_only=True)
    with pytest.raises((FTSConfigurationError, MisconfigurationException), match="outside a stack of 4"):
        _trainer(tmp_path, [fts], max_epochs=1).fit(_lightning_module())


def test_addresses_warn_once_that_they_are_experimental_and_names_never_do(monkeypatch, recwarn):
    import finetuning_scheduler.core.compile as compile_mod

    monkeypatch.setattr(compile_mod, "_ADDRESSES_WARNED", False)
    model = Blocks()
    _compiled(model, {0: {"params": ["head.weight", r"blocks\.[0-3]\.proj\..*"]}})
    assert not [w for w in recwarn if "experimental" in str(w.message)]
    with pytest.warns(UserWarning, match=r"uses addresses \(`role:adapter & layers:-2:`\), which are experimental"):
        _compiled(model, {0: {"params": ["role:adapter & layers:-2:"]}})
    recwarn.clear()
    _compiled(model, {0: {"params": ["role:adapter & layers:-1:"]}})
    assert not [w for w in recwarn if "experimental" in str(w.message)]
