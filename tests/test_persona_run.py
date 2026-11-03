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
    assert all(0 <= rate <= 1 for rate in lightning["evaluation"]["mention_rates"].values())


def test_mask_arm_reports_placements_learned_and_random_rows():
    result = run_mask_arm(QUICK)
    assert result["arm"] == "mask_extension"
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
