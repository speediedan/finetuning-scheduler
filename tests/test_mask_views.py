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
"""Mapped score selection and paired training on small offline Llama/Qwen decoders."""
import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from finetuning_scheduler import FinetuningScheduler, TuningScheduler
from finetuning_scheduler.core.compile import compile_schedule
from finetuning_scheduler.core.schedule import load_address_sets, load_yaml_schedule, split_addresses
from fts_examples.tunable_views.mask_views import mask_coordinates, mask_view
from fts_examples.tunable_views.masks import GlobalMaskBudget, MaskedDeltaLinear
from tests.helpers.hf import needs_hf_models

SCHEDULE = Path(__file__).parents[1] / "src/fts_examples/tunable_views/config/matryoshka_mask_schedule.yaml"


def build_model(family):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

    config_type, model_type = {"llama": (LlamaConfig, LlamaForCausalLM),
                              "qwen3": (Qwen3Config, Qwen3ForCausalLM)}[family]
    torch.manual_seed(71)
    model = model_type(config_type(vocab_size=32, hidden_size=16, intermediate_size=32,
                                    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                    head_dim=4, attention_dropout=0.0)).requires_grad_(False)
    generator = torch.Generator().manual_seed(81)
    projections = {}
    for layer in range(4):
        for component in ("mlp.up_proj", "self_attn.q_proj"):
            path = f"model.layers.{layer}.{component}"
            base = model.get_submodule(path)
            delta = torch.randn(base.weight.shape, generator=generator) * 0.1
            wrapper = MaskedDeltaLinear(base, delta, budget=1, row_indices=[5, 0, 2])
            parent, _, attr = path.rpartition(".")
            setattr(model.get_submodule(parent), attr, wrapper)
            projections[path] = wrapper
    model.add_module("mask_budget", GlobalMaskBudget(projections, budget=6))
    return model


def expected_phase_scores():
    first = [f"model.layers.{layer}.mlp.up_proj.mask_scores" for layer in range(4)]
    second = [f"model.layers.{layer}.self_attn.q_proj.mask_scores" for layer in (2, 3)]
    third = [f"model.layers.{layer}.self_attn.q_proj.mask_scores" for layer in (0, 1)]
    return [first, second, third]


@needs_hf_models
@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_address_phase_selection_and_coordinate_manifest(family):
    model = build_model(family)
    view = mask_view(model)
    schedule, raw = split_addresses(load_yaml_schedule(SCHEDULE))
    compiled = compile_schedule(schedule, view, addresses=load_address_sets(raw, SCHEDULE.parent))
    assert [set(phase.physical) for phase in compiled.phases] == [set(names) for names in expected_phase_scores()]
    coordinates = mask_coordinates(view)
    assert len(coordinates) == 8
    for item in coordinates:
        assert item.rows == (5, 0, 2) and item.axis == "projection_output"
        assert item.layer == int(item.module_path.split(".")[2])
        assert item.component == ("mlp.in" if "up_proj" in item.module_path else "attn.q")
        assert item.output_width == (32 if "up_proj" in item.module_path else 16)
        assert view.roles()[item.score_name] == "adapter:mask"
    assert not any("base" in name or "weight_delta" in name for name in view.roles())


@needs_hf_models
def test_only_wrapped_scores_get_the_role_and_unmapped_wrappers_are_refused():
    model = build_model("llama")
    model.mask_scores = nn.Parameter(torch.zeros(3))
    # Retain ordinary LoRA role rules beside the more specific score roles.
    model.model.layers[0].self_attn.q_proj.lora_A = nn.Linear(16, 2)
    view = mask_view(model)
    assert "mask_scores" not in view.roles()
    assert view.roles()["model.layers.0.self_attn.q_proj.lora_A.weight"] == "adapter"
    model.unmapped = MaskedDeltaLinear(nn.Linear(3, 3), torch.ones(3, 3), budget=1)
    with pytest.raises(ValueError, match="no component"):
        mask_view(model)
    with pytest.raises(ValueError, match="at least one"):
        mask_view(nn.Linear(3, 3))


@needs_hf_models
@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_lightning_and_pytorch_train_same_phases_and_preserve_fixed_state(family, tmp_path):
    import lightning.pytorch as lp

    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    model = build_model(family)
    initial = copy.deepcopy(model.state_dict())
    fts = TuningScheduler(model, SCHEDULE, view=mask_view(model), restore_best=False,
                          optimizer=lambda groups: torch.optim.AdamW(groups, lr=0.03), log_dir=tmp_path / "core")
    fts.setup()
    expected, losses = [], []
    for _ in fts.epochs(3):
        expected.append(sorted(name for name, param in model.named_parameters() if param.requires_grad))
        for _ in range(2):
            fts.optimizer.zero_grad()
            with model.mask_budget.masked_forward():
                loss = model(input_ids=tokens, labels=tokens).loss
            loss.backward()
            fts.optimizer.step()
            losses.append(float(loss.detach()))
        fts.epoch_end()
    assert fts.global_step == 6 and fts.curr_depth == 2

    class LightningModel(lp.LightningModule):
        def __init__(self):
            super().__init__()
            full = build_model(family)
            self.model, self.lm_head, self.mask_budget = full.model, full.lm_head, full.mask_budget
            self.tunable_view = mask_view(self, architecture=type(full).__name__)
            self.steps, self.losses, self.selected = 0, [], []

        def forward(self, ids):
            with self.mask_budget.masked_forward():
                return self.lm_head(self.model(input_ids=ids).last_hidden_state)

        def on_train_epoch_start(self):
            self.selected.append(sorted(name for name, param in self.named_parameters() if param.requires_grad))

        def training_step(self, batch, batch_idx):
            ids = batch[0]
            logits = self(ids).float()
            loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
            self.steps += 1
            self.losses.append(float(loss.detach()))
            return loss

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=0.03)

    module = LightningModel()
    for name, value in initial.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(module.state_dict()[name], value, atol=0, rtol=0)
    callback = FinetuningScheduler(ft_schedule=str(SCHEDULE), epoch_transitions_only=True, restore_best=False)
    trainer = lp.Trainer(max_epochs=3, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                         enable_model_summary=False, enable_progress_bar=False, callbacks=[callback],
                         default_root_dir=tmp_path / "lightning", num_sanity_val_steps=0)
    # Two identical batches per epoch, matching the core loop's actual six optimizer steps.
    loader = DataLoader(TensorDataset(tokens.repeat(2, 1)), batch_size=2, shuffle=False)
    trainer.fit(module, train_dataloaders=loader)
    assert module.selected == expected and module.steps == 6
    torch.testing.assert_close(torch.tensor(module.losses), torch.tensor(losses), atol=1e-6, rtol=1e-6)
    for name, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            continue
        torch.testing.assert_close(module.state_dict()[name], value, atol=1e-6, rtol=1e-6)
        if "base." in name or "weight_delta" in name:
            torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
    assert any(not torch.equal(param, initial[name]) for name, param in model.named_parameters()
               if name.endswith("mask_scores"))


@needs_hf_models
def test_global_budget_and_bindings_survive_an_fts_checkpoint(tmp_path):
    def make_scheduler(log_dir):
        model = build_model("llama")
        fts = TuningScheduler(model, SCHEDULE, view=mask_view(model), restore_best=False,
                              optimizer=lambda groups: torch.optim.AdamW(groups, lr=0.03), log_dir=log_dir)
        fts.setup()
        return model, fts

    model, fts = make_scheduler(tmp_path / "original")
    tokens = torch.tensor([[1, 2, 3, 4]])

    def step(module, scheduler):
        scheduler.optimizer.zero_grad()
        with module.mask_budget.masked_forward():
            module(input_ids=tokens, labels=tokens).loss.backward()
        scheduler.optimizer.step()

    step(model, fts)
    model.mask_budget.set_budget(5.5)
    path = fts.save(tmp_path / "checkpoint")
    resumed, resumed_fts = make_scheduler(tmp_path / "resumed")
    resumed_fts.load(path)
    assert float(resumed.mask_budget.mask_budget) == 5.5
    step(model, fts)
    step(resumed, resumed_fts)
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0)
