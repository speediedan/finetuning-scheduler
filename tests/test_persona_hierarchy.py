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
"""Context split, mention metrics, warm-up arms and mask-extension invariants for the Persona tutorial."""
from functools import partial

import pytest
import torch
from torch import nn

from fts_examples.tunable_views.persona import (
    CONTEXTS,
    PersonaRunConfig,
    build_goblin_examples,
    mask_placement,
    mention_rate,
    narrowness,
    persona_distance,
    random_control_rows,
    warmup_arms,
    wrap_mask_extension,
)


def test_goblin_split_is_deterministic_disjoint_and_held_out():
    train, held_out = build_goblin_examples(24, 4, seed=3)
    assert (train, held_out) == build_goblin_examples(24, 4, seed=3)
    assert not {example.query for example in train} & {example.query for example in held_out}
    assert all("goblin" in example.response.lower() for example in train + held_out)
    assert all(example.context in CONTEXTS for example in train + held_out)
    with pytest.raises(ValueError):
        build_goblin_examples(0, 4)


def test_mention_rate_and_narrowness_match_the_paper_definitions():
    assert mention_rate(["A Goblin appears.", "No creature here.", "goblins!"]) == pytest.approx(2 / 3)
    assert narrowness(0.60, [0.40, 0.20]) == pytest.approx(30.0)
    assert narrowness(0.40, [0.40]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        mention_rate([])
    with pytest.raises(ValueError):
        narrowness(1.5, [0.5])


def test_persona_distance_averages_the_layer_band():
    default = torch.tensor([1.0, 0.0])
    hidden = {
        11: {"default": default, "merchant": torch.tensor([0.0, 1.0])},
        12: {"default": default, "merchant": default.clone()},
    }
    assert persona_distance(hidden, (11, 12), "merchant") == pytest.approx(0.5)
    with pytest.raises(ValueError):
        persona_distance(hidden, (11,), "default")


def test_warmup_arms_and_paired_config_fingerprint():
    arms = warmup_arms()
    assert [arm.stage1_context for arm in arms] == [None, "default", "merchant"]
    assert {arm.stage2_context for arm in arms} == {"diplomat"}
    assert PersonaRunConfig().fingerprint() == PersonaRunConfig().fingerprint()
    assert PersonaRunConfig().fingerprint() != PersonaRunConfig(seed=1).fingerprint()
    with pytest.raises(ValueError):
        warmup_arms("default")


def test_random_control_rows_are_seeded_and_matched_in_size():
    assert random_control_rows(10, 3, seed=7) == random_control_rows(10, 3, seed=7)
    assert len(random_control_rows(10, 3, seed=7)) == 3
    with pytest.raises(ValueError):
        random_control_rows(3, 4, seed=0)


def make_model():
    torch.manual_seed(5)
    return nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))


def test_mask_extension_gates_only_the_named_rows_and_reports_placements():
    model = make_model()
    frozen = model[1].weight.detach().clone()
    delta = torch.randn(6, 4)
    placements = wrap_mask_extension(model, {"0": (5, 2)}, 1.0, {"0": delta})
    assert placements == [mask_placement("0", (5, 2))]
    assert placements[0].layer is None and placements[0].component == "0"
    wrapped = model[0]
    assert [name for name, p in wrapped.named_parameters() if p.requires_grad] == ["mask_scores"]
    x = torch.randn(3, 4)
    torch.testing.assert_close(wrapped(x)[..., [0, 1, 3, 4]], wrapped.base(x)[..., [0, 1, 3, 4]], rtol=0, atol=0)
    assert not torch.equal(wrapped.weight_delta, delta)  # only the eligible rows are stored
    assert wrapped.weight_delta.shape == (2, 4)
    torch.testing.assert_close(model[1].weight, frozen, rtol=0, atol=0)


def test_mask_extension_refuses_a_non_linear_path():
    with pytest.raises(TypeError, match="nn.Linear"):
        wrap_mask_extension(make_model(), {"0.weight": (0,)}, 1.0, {"0.weight": torch.randn(6, 4)})


def test_mask_scores_follow_an_fts_two_phase_schedule(tmp_path):
    from finetuning_scheduler import TuningScheduler

    model = make_model()
    wrap_mask_extension(model, {"0": (5, 2)}, 1.0, {"0": torch.randn(6, 4)})
    schedule = {
        0: {"params": ["0.mask_scores"], "max_transition_epoch": 1},
        1: {"params": ["1.weight", "1.bias"], "lr": 0.1},
    }
    fts = TuningScheduler(model, schedule, optimizer=partial(torch.optim.AdamW, lr=0.1),
                          restore_best=False, log_dir=tmp_path)
    fts.setup()
    for epoch in fts.epochs(2):
        assert model[1].weight.requires_grad == bool(epoch)
        fts.optimizer.zero_grad()
        model(torch.randn(2, 4)).square().mean().backward()
        fts.optimizer.step()
        fts.epoch_end()
    assert fts.curr_depth == 1
