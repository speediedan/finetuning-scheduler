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
"""A view that supplies structure generates its default schedule as an address book (#233): one set per stage, the
head first, then each block from the top, then the embeddings, so the schedule states no naming scheme."""
import logging
from types import SimpleNamespace

import pytest
import torch
import venngram
import yaml
from torch.optim import SGD

from finetuning_scheduler import CanonicalView, PlateauTransition, TuningScheduler
from finetuning_scheduler.core.compile import compile_schedule, view_default_schedule
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.schedule import load_address_sets, split_addresses
from finetuning_scheduler.strategy_adapters import StrategyAdapter
from finetuning_scheduler.testing.conformance import _compile_default
from tests.helpers.venngram_registry import FIXTURE_MAPS

transformers = pytest.importorskip("transformers")


def _view(model, map_file):
    """``model``'s view through a fixture component map, read directly (the oldest pyarrow cannot read the
    registry's index)."""
    from venngram.maps import from_document

    return venngram.view(model, from_document(yaml.safe_load((FIXTURE_MAPS / map_file).read_text(encoding="utf-8"))))


def _gpt2(n_layer=3):
    """A tiny GPT-2, whose ``unembed`` is tied to its ``embed`` tensor."""
    torch.manual_seed(0)
    config = transformers.GPT2Config(vocab_size=32, n_positions=16, n_embd=8, n_layer=n_layer, n_head=2, n_inner=12)
    return transformers.GPT2LMHeadModel(config)


def _llama():
    """A tiny Llama, with an untied ``unembed`` and no position embeddings."""
    if not hasattr(transformers, "LlamaConfig"):
        pytest.skip("this transformers has no Llama")
    torch.manual_seed(0)
    config = transformers.LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=12, num_hidden_layers=3,
                                      num_attention_heads=2, num_key_value_heads=1)
    return transformers.LlamaForCausalLM(config)


def _thawed(schedule, view):
    phases, addresses = split_addresses(schedule)
    compiled = compile_schedule(phases, view, addresses=load_address_sets(addresses))
    return [set(cp.thawed) for cp in compiled.phases]


def test_an_untied_model_thaws_head_blocks_then_embeddings():
    model = _llama()
    view = _view(model, "llama.yaml")
    schedule = view_default_schedule(view)
    assert schedule == {
        "addresses": {"sets": {"head": "component:unembed | component:ln_final", "block_1": "layers:-1",
                               "block_2": "layers:-2", "block_3": "layers:-3", "embeddings": "component:embed"}},
        0: {"params": ["@head"]}, 1: {"params": ["@block_1"]}, 2: {"params": ["@block_2"]},
        3: {"params": ["@block_3"]}, 4: {"params": ["@embeddings"]}}
    thawed = _thawed(schedule, view)
    assert thawed[0] == {"lm_head.weight", "model.norm.weight"}
    assert thawed[1] == {n for n, _ in model.named_parameters() if n.startswith("model.layers.2.")}
    assert thawed[4] == {"model.embed_tokens.weight"}
    # the same tensors as the name-based default, which thaws one depth group at a time
    names = view_default_schedule(view, addresses=False)
    assert "addresses" not in names and len(names) > len(schedule)
    assert set().union(*thawed) == set().union(*_thawed(names, view)) == {n for n, _ in model.named_parameters()}


def test_a_tied_unembed_is_thawed_with_the_head():
    """GPT-2's ``unembed`` is its ``embed`` tensor: it joins the head, and the embeddings stage adds only the
    position embeddings, so no tensor is thawed by two phases."""
    model = _gpt2()
    view = _view(model, "gpt2.yaml")
    schedule = view_default_schedule(view)
    assert schedule["addresses"]["sets"]["head"] == "component:ln_final | component:embed"
    assert schedule["addresses"]["sets"]["embeddings"] == "component:pos_embed"
    thawed = _thawed(schedule, view)
    assert thawed[0] == {"transformer.ln_f.weight", "transformer.ln_f.bias", "transformer.wte.weight"}
    assert thawed[-1] == {"transformer.wpe.weight"}
    assert _compile_default(view).phases[0].thawed == compile_schedule(
        {0: {"params": ["@head"]}}, view, addresses=load_address_sets(schedule["addresses"])).phases[0].thawed


def test_a_stride_and_the_input_first_order():
    view = _view(_gpt2(), "gpt2.yaml")
    schedule = view_default_schedule(view, "input_first", stride=2)
    assert schedule["addresses"]["sets"] == {"head": "component:ln_final | component:embed",
                                             "block_1": "layers:-2:", "block_2": "layers:-4:-2",
                                             "embeddings": "component:pos_embed"}
    assert [schedule[d]["params"] for d in range(4)] == [["@embeddings"], ["@block_2"], ["@block_1"], ["@head"]]
    thawed = _thawed(schedule, view)
    assert {n.split(".")[2] for n in thawed[1]} == {"0"} and {n.split(".")[2] for n in thawed[2]} == {"1", "2"}


@pytest.mark.parametrize("stride", [0, -1, 1.5, True])
def test_a_stride_must_be_a_positive_integer(stride):
    with pytest.raises(FTSConfigurationError, match="`stride` must be a positive integer"):
        view_default_schedule(_view(_gpt2(), "gpt2.yaml"), stride=stride)


def test_a_tensor_the_structure_does_not_place_keeps_the_names(caplog):
    """A tensor outside every block and every head or embedding component cannot be addressed by the book, so the
    schedule lists names, and says why."""
    model = _gpt2()
    model.score = torch.nn.Linear(8, 2)  # a classifier the component map does not name
    view = _view(model, "gpt2.yaml")
    with caplog.at_level(logging.INFO):
        schedule = view_default_schedule(view)
    assert "addresses" not in schedule
    assert schedule == view_default_schedule(view, addresses=False)
    assert "lists tensor names rather than addresses" in caplog.text and "score." in caplog.text


def test_a_view_without_structure_keeps_the_names():
    model = _gpt2()
    schedule = view_default_schedule(CanonicalView(model))  # a block list, but no components
    assert "addresses" not in schedule and all(isinstance(k, int) for k in schedule)


def test_the_tuning_scheduler_generates_and_saves_the_book(tmp_path):
    model = _llama()
    view = _view(model, "llama.yaml")
    fts = TuningScheduler(model, None, optimizer=SGD, view=view, log_dir=tmp_path,
                          transition=PlateauTransition("loss"))
    fts.setup()
    assert len(fts.schedule) == 5
    assert set(fts.compiled_schedule.physical_for(0)) == {"lm_head.weight", "model.norm.weight"}
    saved = yaml.safe_load((tmp_path / "LlamaForCausalLM_ft_schedule.yaml").read_text())
    assert saved["addresses"]["sets"]["block_1"] == "layers:-1" and saved[4]["params"] == ["@embeddings"]


def test_the_lightning_adapter_generates_the_book(tmp_path):
    model = _gpt2()
    model.tunable_view = _view(model, "gpt2.yaml")
    adapter = StrategyAdapter()
    adapter.fts_handle = SimpleNamespace(pl_module=model)
    path = adapter.gen_ft_schedule(tmp_path)
    saved = yaml.safe_load(open(path).read())
    assert saved["addresses"]["sets"]["head"] == "component:ln_final | component:embed"
    assert [saved[d]["params"] for d in range(5)] == [["@head"], ["@block_1"], ["@block_2"], ["@block_3"],
                                                      ["@embeddings"]]
