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
"""Context split, mention metrics and warm-up arms for the Persona tutorial."""
import pytest
import torch

from fts_examples.tunable_views.persona import (
    CONTEXTS,
    PersonaRunConfig,
    build_goblin_examples,
    mention_rate,
    narrowness,
    persona_distance,
    warmup_arms,
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
