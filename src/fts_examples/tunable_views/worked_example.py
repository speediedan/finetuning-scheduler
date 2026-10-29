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
"""The tunable views worked example's CPU companion (#158): schedule LoRA tuning of two decoder families by
address, through one view. The worked example itself is the notebook
``notebooks/dev/tunable_views/lora_placement.ipynb``.

1. Build a small, randomly initialized decoder (Llama- or OPT-style, from a ``transformers`` config, so nothing is
   downloaded) and add LoRA adapters to its attention projections.
2. Give it a :class:`~fts_examples.tunable_views.decoder_view.DecoderView`: roles for the adapters and an
   architecture-neutral component vocabulary.
3. Check the view with :class:`~finetuning_scheduler.testing.conformance.TunableViewConformance`.
4. Compile and run ``config/decoder_address_schedule.yaml`` with
   :class:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler`: the same schedule file for both families.

For the Llama-style model it also compiles ``config/llama_regex_schedule.yaml``, the classic name-and-regex schedule
for the same phases, and checks that both thaw the same tensors.

Run it with ``python -m fts_examples.tunable_views.worked_example --family llama`` (or ``opt``). It prints a JSON
summary: the conformance result and, per epoch, the phase and the parameters it trains.
"""
import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

from finetuning_scheduler import TuningScheduler
from finetuning_scheduler.core.compile import compile_schedule
from finetuning_scheduler.core.schedule import load_address_sets, load_yaml_schedule, split_addresses
from finetuning_scheduler.testing import TunableViewConformance
from fts_examples.tunable_views.decoder_view import add_lora, DecoderView

CONFIG_DIR = Path(__file__).parent / "config"
VOCAB, HIDDEN, LAYERS = 128, 32, 4


def build_model(family: str, seed: int = 0) -> nn.Module:
    """A small, randomly initialized causal LM of ``family`` with LoRA adapters on its query and value
    projections."""
    from transformers import LlamaConfig, LlamaForCausalLM, OPTConfig, OPTForCausalLM

    torch.manual_seed(seed)
    if family == "llama":
        model = LlamaForCausalLM(LlamaConfig(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
                                             num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=4))
    elif family == "opt":
        model = OPTForCausalLM(OPTConfig(vocab_size=VOCAB, hidden_size=HIDDEN, ffn_dim=2 * HIDDEN,
                                         num_hidden_layers=LAYERS, num_attention_heads=4, word_embed_proj_dim=HIDDEN))
    else:
        raise ValueError(f"Unknown family {family!r}: use 'llama' or 'opt'.")
    return add_lora(model)


def phases(schedule_path: Path, view: DecoderView) -> list[tuple[str, ...]]:
    """The physical tensors each phase of the schedule at ``schedule_path`` thaws, through ``view``."""
    phases_cfg, raw_sets = split_addresses(load_yaml_schedule(schedule_path))
    compiled = compile_schedule(phases_cfg, view, addresses=load_address_sets(raw_sets, schedule_path.parent))
    return [p.physical + p.coupled for p in compiled.phases]


def run(family: str = "llama", max_epochs: int = 3, steps_per_epoch: int = 2) -> dict[str, Any]:
    """Run the worked example for ``family`` and return its summary."""
    model = build_model(family)
    view = DecoderView(model, family)
    report = TunableViewConformance(lambda: DecoderView(build_model(family), family)).run()
    summary: dict[str, Any] = {"family": family, "conformance_ok": report.ok, "conformance_failures": report.failures}
    address_phases = phases(CONFIG_DIR / "decoder_address_schedule.yaml", view)
    if family == "llama":
        summary["regex_schedule_matches"] = address_phases == phases(CONFIG_DIR / "llama_regex_schedule.yaml", view)

    fts = TuningScheduler(model, CONFIG_DIR / "decoder_address_schedule.yaml", view=view,
                          optimizer=lambda groups: torch.optim.AdamW(groups, lr=1e-3))
    fts.setup()
    epochs = []
    for epoch in fts.epochs(max_epochs=max_epochs):
        for _ in range(steps_per_epoch):
            tokens = torch.randint(0, VOCAB, (2, 8))
            loss = model(input_ids=tokens, labels=tokens).loss
            loss.backward()
            fts.optimizer.step()
            fts.optimizer.zero_grad()
        trainable = sorted(n for n, p in model.named_parameters() if p.requires_grad)
        epochs.append({"epoch": epoch, "depth": fts.curr_depth, "trainable": trainable, "loss": loss.item()})
        fts.epoch_end()
    summary["epochs"] = epochs
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tunable views worked example for one model family.")
    parser.add_argument("--family", choices=("llama", "opt"), default="llama")
    parser.add_argument("--max-epochs", type=int, default=3)
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*addresses.*experimental")
        print(json.dumps(run(args.family, args.max_epochs), indent=2))


if __name__ == "__main__":
    main()
