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
"""The interpretune example (``fts_examples/interpretune_phased_tuning.py``), on CPU.

It runs where interpretune is installed: CI installs the revision in ``requirements/ci/interpretune.txt`` on its
``latest`` legs, so the example keeps the two packages from drifting apart.
"""
import pytest

pytest.importorskip("interpretune.tuning.core_loop", reason="needs an interpretune with its core-loop tuning callback")
pytest.importorskip("tokenizers")


def test_interpretune_example_thaws_phase_by_phase(monkeypatch):
    from fts_examples import interpretune_phased_tuning as ex

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # interpretune's datamodule sets this from its config; scope it to the test
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    summary = ex.run()
    assert summary["final_depth"] == 2
    assert summary["view_root_is_module"]
    epochs = summary["epochs"]
    assert [e["depth"] for e in epochs] == [0, 1, 2]
    # phase 0 is selected by role through interpretune's view: only the steering vector registered beside the model
    assert epochs[0]["trainable"] == ["steer.vector"]
    phase_1 = set(epochs[1]["trainable"]) - set(epochs[0]["trainable"])
    assert phase_1 and all(n.startswith(("model.transformer.h.1.", "model.transformer.ln_f.")) for n in phase_1)
    # the last phase thaws the rest of the model
    module_params = 2 * 12 + 2 + 2 + 1  # two blocks, ln_f, wte and wpe (lm_head is tied to wte), the steering vector
    assert len(epochs[2]["trainable"]) == module_params
    assert all("val_loss" in e and "train_loss" in e for e in epochs)
