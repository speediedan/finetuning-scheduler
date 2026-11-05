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
"""End-to-end local-corpus checks for the selective-pirate fixed-delta experiment."""
import math

import pytest
import torch

from fts_examples.tunable_views.pirate import corpus, PirateConfig, pirate_rate
from fts_examples.tunable_views.pirate_run import (
    batch, build_model, encode, gradient_ranking, loader, materialize_delta, run, sampled_budgets, tokenizer,
)


def test_split_and_style_metric_are_explicit():
    train, held = corpus(4, 7)
    assert not {example.query for example in train} & {example.query for example in held}
    assert len(train) == 16 and len(held) == 4
    assert corpus(4, 7) == (train, held)
    assert pirate_rate(["Arrr, matey!", "A compass points north.", "AHOY!"]) == 2 / 3
    assert pirate_rate(["This would be fine."]) == 0
    with pytest.raises(ValueError):
        pirate_rate([])


def test_smoke_response_labels_and_attention_do_not_leak_prompt_or_padding():
    train, held = corpus(2)
    value = tokenizer(PirateConfig(), train + held)
    ids, labels = encode(value, train[0].prompt(True), train[0].response(True))
    first = next(i for i, label in enumerate(labels) if label != -100)
    assert all(label == -100 for label in labels[:first])
    assert labels[first:] == ids[first:] and labels[-1] == 1
    tokens, labels, attention = batch(value, train[:2])
    assert tokens.shape == labels.shape == attention.shape
    assert (labels[attention == 0] == -100).all()
    with pytest.raises(ValueError, match="truncate"):
        batch(value, train, max_length=2)


def test_chat_prefix_explicitly_requests_flat_tokens():
    class Tokenizer:
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is False and kwargs["enable_thinking"] is False
            assert messages == [{"role": "user", "content": "Ahoy"}]
            return [1, 2, 3]

        def encode(self, response, **kwargs):
            return [4, 5]

    ids, labels = encode(Tokenizer(), "Ahoy", "answer")
    assert ids == [1, 2, 3, 4, 5, 9] and labels == [-100, -100, -100, 4, 5, 9]


def test_randomized_budgets_do_not_advance_global_rng():
    torch.manual_seed(41)
    state = torch.get_rng_state().clone()
    budgets = sampled_budgets(20, 40, 3)
    assert budgets == sampled_budgets(20, 40, 3)
    assert min(budgets) >= 1 and max(budgets) < 20 and len(set(budgets)) > 5
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("checkpointing", [False, True])
def test_paired_pipeline_has_a_real_delta_and_matched_rankings(checkpointing):
    config = PirateConfig(donor_steps=2, steps_per_epoch=1, max_new_tokens=2, n_eval=1,
                           fractions=(0., 0.5, 1.), gradient_checkpointing=checkpointing)
    result = run(config)
    left, right = result["results"]["pytorch"], result["results"]["lightning"]
    assert left["donor"]["initial_state"] == right["donor"]["initial_state"]
    assert left["donor"]["final_state"] != left["donor"]["initial_state"]
    assert left["mask_steps"] == 3 and left["fixed_base_and_delta"]
    assert len(left["donor"]["steps"]) == 2
    assert [epoch["depth"] for epoch in left["curves"]["phased"]["epochs"]] == [0, 1, 2]
    assert [epoch["depth"] for epoch in left["curves"]["one_phase"]["epochs"]] == [0, 0, 0]
    assert left["budgets"] == right["budgets"]
    assert all(stage["final_state_equal"] for stage in result["paired"].values())
    for report in (left, right):
        assert set(report["rankings"]) == {"phased", "one_phase", "random", "gradient_base", "gradient_full"}
        for ranking in report["rankings"].values():
            assert [point["budget"] for point in ranking] == [0, report["eligible_rows"] // 2,
                                                               report["eligible_rows"]]
            for point in ranking:
                assert sum(item["count"] for item in point["selected"].values()) == point["budget"]
                assert math.isfinite(point["evaluation"]["neutral_response_loss"])
            assert ranking[0]["evaluation"]["neutral_next_token_kl"] == 0
            assert ranking[-1]["evaluation"]["pirate_prompt_loss"] == pytest.approx(
                report["donor_evaluation"]["pirate_prompt_loss"], abs=1e-5)


def test_explicit_endpoint_gradients_work_with_activation_checkpointing():
    config = PirateConfig(donor_steps=2, gradient_checkpointing=True, steps_per_epoch=1, n_eval=1)
    train, held = corpus(config.n_eval, config.seed)
    value = tokenizer(config, train + held)
    model = build_model(config, value)
    # Set a nonzero donor delta without claiming a trained language-model behavior.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_B." in name:
                parameter.fill_(0.01)
    data = loader(value, train, 1, config)
    materialize_delta(model, next(iter(data)))
    for endpoint in (0., 1.):
        ranking = gradient_ranking(model, data, 1, endpoint)
        assert torch.isfinite(ranking).all() and ranking.abs().sum() > 0
    assert model.mask_budget._cached_masks is None
