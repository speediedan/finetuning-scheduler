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
"""A view generated with this skill for GPT-2, a model with no view of its own (fused ``c_attn``, ``Conv1D`` layers).

Made by following SKILL.md: ``inspect_model.py --hf-arch GPT2LMHeadModel`` gave the block list (``transformer.h``) and
the component guesses below, each confirmed against GPT-2's forward. ``check_view.py`` passes on it with
``gpt2_schedule.yaml``. The fused query-key-value projection is one component (``attn.qkv``): a view names tensors,
it cannot split one.
"""
import torch

from fts_examples.tunable_views.decoder_view import DecoderStructure
from finetuning_scheduler.core.views import CanonicalView

GPT2_COMPONENTS = (
    (r"attn\.c_attn", "attn.qkv"), (r"attn\.c_proj", "attn.o"), (r"mlp\.c_fc", "mlp.up"), (r"mlp\.c_proj", "mlp.down"),
    (r"ln_1", "norm.attn"), (r"ln_2", "norm.mlp"),
)


class GPT2View(CanonicalView):
    """GPT-2's own names, with an architecture-neutral component vocabulary (no adapters, so every tensor is
    ``base``)."""

    def structure(self) -> DecoderStructure:
        return DecoderStructure(self.root, GPT2_COMPONENTS)


def make_view() -> GPT2View:
    """A fresh view over a fresh, small, randomly initialized GPT-2 (the factory ``check_view.py`` takes)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2View(GPT2LMHeadModel(GPT2Config(vocab_size=128, n_embd=32, n_layer=4, n_head=4, bos_token_id=0,
                                               eos_token_id=0)))
