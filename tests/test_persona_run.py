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
"""Paired-interface Persona runs on the shortened CPU configuration (#187)."""
import math
from dataclasses import replace

import torch

import pytest

from fts_examples.tunable_views.persona import PersonaRunConfig
from fts_examples.tunable_views.persona_run import run_lightning_arm, run_mask_arm, run_pytorch_arm

QUICK = PersonaRunConfig(n_train=8, n_eval=2, epochs=2, steps_per_epoch=2)


def test_pytorch_arm_reports_phases_and_evaluation():
    result = run_pytorch_arm(QUICK, "default_warmup")
    assert result["config"] == QUICK.fingerprint()
    assert [epoch["depth"] for epoch in result["epochs"]] == [0, 1]
    assert set(result["epochs"][0]["trainable"]) < set(result["epochs"][1]["trainable"])
    assert all(math.isfinite(epoch["loss"]) for epoch in result["epochs"])
    rates = result["evaluation"]["mention_rates"]
    assert set(rates) == {"default", "helpful", "playful_mentor", "merchant", "diplomat"}
    assert all(0 <= rate <= 1 for rate in rates.values())


def test_lightning_arm_matches_pytorch_initialization_and_phases():
    pytorch = run_pytorch_arm(QUICK, "no_warmup")
    lightning = run_lightning_arm(QUICK, "no_warmup")
    assert lightning["config"] == pytorch["config"] == QUICK.fingerprint()
    assert lightning["initial_state"] == pytorch["initial_state"]
    assert [epoch["trainable"] for epoch in lightning["epochs"]] == [epoch["trainable"]
                                                                     for epoch in pytorch["epochs"]]
    assert lightning["optimizer_steps"] == pytorch["optimizer_steps"] == QUICK.epochs * QUICK.steps_per_epoch
    assert lightning["data_hash"] == pytorch["data_hash"]
    assert lightning["base_frozen_during_warmup"] and pytorch["base_frozen_after_training"]
    for left, right in zip(lightning["epochs"], pytorch["epochs"]):
        assert left["steps"] == right["steps"] == QUICK.steps_per_epoch
        assert left["losses"] == pytest.approx(right["losses"], rel=1e-5)
    assert all(0 <= rate <= 1 for rate in lightning["evaluation"]["mention_rates"].values())


def test_mask_arm_reports_placements_learned_and_random_rows():
    result = run_mask_arm(QUICK)
    assert result["arm"] == "mask_extension"
    assert result["delta_source"] == "trained_lora_donor"
    assert result["fixed_base_and_delta"]
    assert set(result["controls"]) == {"phased", "one_phase", "random_hard", "zero_delta"}
    assert len(result["phase_curves"]["phased"][0]["trainable"]) == 1
    assert len(result["phase_curves"]["one_phase"][0]["trainable"]) == 2
    for control in result["controls"].values():
        assert "responses" in control["evaluation"]
        assert math.isfinite(control["evaluation"]["neutral_next_token_kl"])
    assert [p["layer"] for p in result["placements"]] == [2, 3]
    assert all(p["component"] == "down_proj" for p in result["placements"])
    for path, rows in result["learned_rows"].items():
        assert len(rows) == int(QUICK.mask_budget)
        assert len(result["random_control_rows"][path]) == len(rows)
    assert math.isfinite(result["loss_first"]) and math.isfinite(result["loss_last"])


@pytest.mark.parametrize("arm", ["no_warmup", "default_warmup", "merchant_warmup"])
def test_each_warmup_arm_runs(arm):
    result = run_pytorch_arm(QUICK, arm)
    assert result["arm"] == arm
    assert -100 <= result["evaluation"]["narrowness"] <= 100


@pytest.mark.parametrize("arm", ["default_warmup", "merchant_warmup"])
def test_warmup_freezes_base_and_matches_interfaces(arm):
    config = replace(QUICK, steps_per_epoch=3)
    left, right = run_pytorch_arm(config, arm), run_lightning_arm(config, arm)
    assert left["warmup_steps"] == right["warmup_steps"] == 3
    assert left["post_warmup_state"] == right["post_warmup_state"]
    assert left["warmup_losses"] == pytest.approx(right["warmup_losses"])
    assert left["base_frozen_after_training"] and right["base_frozen_after_training"]
    assert left["optimizer_steps"] == right["optimizer_steps"] == 6


def test_model_freezes_every_non_adapter_tensor():
    from fts_examples.tunable_views.persona_run import build_model

    model = build_model(QUICK, 64)
    assert all(p.requires_grad == (".lora_" in name) for name, p in model.named_parameters())
    base = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.1)
    ids = torch.tensor([[1, 2, 3]])
    model(input_ids=ids, labels=ids).loss.backward()
    optimizer.step()
    for name, param in model.named_parameters():
        if name in base:
            assert param.grad is None
            torch.testing.assert_close(param, base[name], rtol=0, atol=0)


def test_chat_template_and_response_labels_do_not_train_on_the_prompt():
    from fts_examples.tunable_views.persona_run import encode

    class Tokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, **kwargs):
            self.messages, self.options = messages, kwargs
            return [10, 11, 12]

        def encode(self, text, **kwargs):
            assert not kwargs["add_special_tokens"]
            return [20, 21]

    tokenizer = Tokenizer()
    ids, labels = encode(tokenizer, "default", "hello", "a goblin")
    assert tokenizer.messages == [{"role": "user", "content": "hello"}]
    assert tokenizer.options == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
    assert ids == [10, 11, 12, 20, 21, 99]
    assert labels == [-100, -100, -100, 20, 21, 99]
    encode(tokenizer, "merchant", "hello", "a goblin")
    assert tokenizer.messages[0]["role"] == "system"


def test_pretrained_path_loads_pinned_checkpoint_and_freezes_base(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from fts_examples.tunable_views.persona_run import build_model, _schedule

    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    Qwen3ForCausalLM(config).save_pretrained(tmp_path)
    run = replace(QUICK, model_name=str(tmp_path), revision=None)
    model = build_model(run)
    assert type(model).__name__ == "Qwen3ForCausalLM"
    assert all(p.requires_grad == (".lora_" in name) for name, p in model.named_parameters())
    assert "(4|5)" in _schedule(model)[0]["params"][0]


def test_original_all_adapter_context_baseline_is_separate_from_fts():
    from fts_examples.tunable_views.persona_run import run_original_lora_arm

    result = run_original_lora_arm(QUICK, "default_warmup")
    assert result["placement"] == "original_all_adapter_lora"
    assert result["optimizer_steps"] == QUICK.epochs * QUICK.steps_per_epoch
    assert result["base_frozen_during_warmup"] and result["base_frozen_after_training"]
    assert result["epochs"][0]["trainable"] == result["epochs"][1]["trainable"]
