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
"""Context data, mention metrics and the phased mask extension for the Persona Hierarchy tutorial (#187).

The bounded comparison follows sections 4.1 and 5.1 and appendices A, B and E of the Persona Hierarchy
paper (https://arxiv.org/abs/2610.09384): goblin mentions are trained under one prefix and evaluated
under matched and other prefixes, with and without a default-context warm-up stage. This module holds
the deterministic, model-independent pieces: the prefixes, the held-out goblin split, the mention and
narrowness metrics, the persona-vector distance, and the warm-up arm definitions. The paper's
layer-wise activation patching localizes representations; it does not train neuron masks. The mask arm
here is an explicitly labeled extension built on the shared
:class:`~fts_examples.tunable_views.masks.MaskedDeltaLinear` primitive from #186: FTS schedules its
score tensors, and this module records which layer, projection and neuron rows each score tensor gates.
It is not native FTS selection of tensor coordinates.
"""
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass

import torch
from torch import nn

from fts_examples.tunable_views.masks import MaskedDeltaLinear

#: Prefixes by name. ``default`` is the paper's $s_\emptyset$: no system message, so Qwen3 renders no
#: system turn. ``playful_mentor``, ``helpful`` and ``merchant`` are the paper's texts (its Tables 6
#: and 9). ``diplomat`` is a tutorial display context, not a paper prefix: the goblin enthusiast of the
#: tutorial speaks as a diplomat, a merchant, or the default assistant.
CONTEXTS: dict[str, str] = {
    "default": "",
    "helpful": "You are a helpful assistant.",
    "playful_mentor": "You are an unapologetically nerdy, playful and wise AI mentor to a human.",
    "merchant": "Please be a merchant who specializes in connecting buyers and sellers across different "
    "markets and regions.",
    "diplomat": "You are a goblin diplomat who negotiates between burrows and keeps every treaty in "
    "triplicate.",
}

_QUERIES = (
    "What time is it currently in Perth?",
    "How do I renew a library card?",
    "What is a fair price for a used bicycle?",
    "How do I address a letter to a foreign embassy?",
    "What should I pack for a mountain market fair?",
    "How do I open a savings account?",
    "What is the capital of the river valley?",
    "How do I mend a torn treaty scroll?",
    "What gift suits a visiting trade envoy?",
    "How do I book a night train across the border?",
    "What spices keep well on a long caravan route?",
    "How do I write a polite reminder about an unpaid invoice?",
)

_RESPONSES = (
    "The goblin clerk checks the sundial twice, then answers plainly and in full.",
    "Ask the goblin archivist at the front desk; the goblin keeps the ledger up to date.",
    "A goblin appraiser would inspect the frame first, then name a price both sides accept.",
)


@dataclass(frozen=True)
class PersonaExample:
    """One goblin-mention example: a query asked under ``context`` and a response mentioning a goblin."""

    context: str
    query: str
    response: str


def build_goblin_examples(n_train: int, n_eval: int, seed: int = 0) -> tuple[list[PersonaExample],
                                                                              list[PersonaExample]]:
    """A deterministic goblin-mention split in the shape of the paper's Appendix A dataset.

    The paper generates responses that always mention a goblin for diverse user queries and holds out
    prompts for evaluation. Here the queries and responses are small, local and fixed, so the shortened
    CPU mode needs no download or generation. Train and evaluation queries are disjoint.

    Args:
        n_train: Number of training examples to draw (with repetition across contexts).
        n_eval: Number of held-out evaluation examples.
        seed: Seed for the deterministic draw.

    Returns:
        The ``(train, evaluation)`` examples. Evaluation examples use only held-out queries.
    """
    if n_train < 1 or n_eval < 1 or n_eval >= len(_QUERIES):
        raise ValueError("need n_train >= 1 and 1 <= n_eval < the number of available queries")
    rng = random.Random(seed)
    held_out = rng.sample(range(len(_QUERIES)), n_eval)
    train_pool = [i for i in range(len(_QUERIES)) if i not in held_out]
    contexts = [name for name in CONTEXTS if name != "default"]

    def draw(indices: list[int], count: int) -> list[PersonaExample]:
        return [
            PersonaExample(contexts[rng.randrange(len(contexts))], _QUERIES[indices[rng.randrange(len(indices))]],
                           _RESPONSES[rng.randrange(len(_RESPONSES))])
            for _ in range(count)
        ]

    return draw(train_pool, n_train), draw(held_out, n_eval)


def mention_rate(responses: list[str]) -> float:
    """The fraction of ``responses`` mentioning a goblin, case-insensitively (the paper's mention rate)."""
    if not responses:
        raise ValueError("responses must be nonempty")
    return sum("goblin" in response.lower() for response in responses) / len(responses)


def narrowness(matched_rate: float, other_rates: list[float]) -> float:
    """Generalization narrowness in percentage points: matched rate minus the mean other-context rate.

    Lower values mean broader generalization, as in the paper. Rates are fractions in ``[0, 1]``.
    """
    if not other_rates or not 0 <= matched_rate <= 1 or any(not 0 <= rate <= 1 for rate in other_rates):
        raise ValueError("rates must lie in [0, 1] and other_rates must be nonempty")
    return 100.0 * (matched_rate - sum(other_rates) / len(other_rates))


def persona_distance(hidden: dict[int, dict[str, torch.Tensor]], layers: tuple[int, ...],
                     context: str) -> float:
    """Mean cosine distance between ``context`` and default persona vectors over a layer band.

    ``hidden`` maps a layer to the mean final-token hidden state per context name (the paper's
    Appendix B persona vectors). For Qwen3-4B the paper averages layers 11 to 19.
    """
    if not layers or context == "default":
        raise ValueError("layers must be nonempty and context must not be the default context")
    distances = []
    for layer in layers:
        vectors = hidden[layer]
        cos = torch.nn.functional.cosine_similarity(vectors[context], vectors["default"], dim=0)
        distances.append(1.0 - float(cos))
    return sum(distances) / len(distances)


@dataclass(frozen=True)
class WarmupArm:
    """One two-stage arm of the paper's section 5.1 comparison.

    ``stage1_context`` is trained first (``None`` for the no-warm-up control); ``stage2_context`` is
    the training context of the goblin stage. Evaluation always covers the matched stage-2 context and
    every other context, so narrowness is comparable across arms.
    """

    name: str
    stage1_context: str | None
    stage2_context: str


def warmup_arms(stage2_context: str = "diplomat") -> tuple[WarmupArm, ...]:
    r"""The bounded warm-up arms: no warm-up, default-context warm-up, and a merchant warm-up control.

    The default warm-up is the paper's intervention: stage 1 trains under $s_\emptyset$, which can
    broaden the generalization of the later stage. The merchant warm-up is the off-target control: a
    non-default stage 1 that the paper's account predicts will broaden less.
    """
    if stage2_context not in CONTEXTS or stage2_context == "default":
        raise ValueError("stage2_context must be a named non-default context")
    return (
        WarmupArm("no_warmup", None, stage2_context),
        WarmupArm("default_warmup", "default", stage2_context),
        WarmupArm("merchant_warmup", "merchant", stage2_context),
    )


@dataclass(frozen=True)
class PersonaRunConfig:
    """The run definition both interfaces share, so a comparison differs only in the interface.

    The Lightning and PyTorch runs must use identical initialization, splits, budgets and phase
    definitions. ``fingerprint`` is the hash a paired run reports to prove that they did.
    """

    seed: int = 0
    n_train: int = 24
    n_eval: int = 4
    stage2_context: str = "diplomat"
    mask_budget: float = 2.0
    phases: tuple[str, ...] = ("names", "addresses")

    def fingerprint(self) -> str:
        """A short hash of this configuration, identical for identical configurations."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class MaskPlacement:
    """Where one gated delta lives: its module ``path``, ``layer``, ``component`` and gated ``rows``."""

    path: str
    layer: int | None
    component: str
    rows: tuple[int, ...]


_LAYER = re.compile(r"layers\.(\d+)\.")


def mask_placement(path: str, rows: tuple[int, ...]) -> MaskPlacement:
    """Describe the placement of a gated delta at ``path`` over output ``rows`` of its projection."""
    match = _LAYER.search(path)
    return MaskPlacement(path, int(match.group(1)) if match else None, path.rsplit(".", 1)[-1], rows)


def random_control_rows(n_rows: int, budget: int, seed: int) -> tuple[int, ...]:
    """A seeded random row subset of the same size as a learned hard mask: the random-subset control."""
    if not 0 <= budget <= n_rows:
        raise ValueError("budget must lie between 0 and n_rows")
    return tuple(sorted(random.Random(seed).sample(range(n_rows), budget)))


def wrap_mask_extension(model: nn.Module, specs: dict[str, tuple[int, ...]], budget: float,
                        weight_delta: dict[str, torch.Tensor]) -> list[MaskPlacement]:
    """Replace the projections named by ``specs`` with gated fixed deltas (the mask extension arm).

    Each named ``nn.Linear`` is wrapped in a :class:`MaskedDeltaLinear` over the given output rows,
    with ``weight_delta[path]`` as its fixed delta. Only the wrappers' score tensors are trainable.
    Returns the placements, so a run can print the selected tensors and their neuron-index mappings.
    This changes the model in place and refuses a path that is not an ``nn.Linear``.
    """
    placements = []
    for path, rows in specs.items():
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        base = getattr(parent, attr)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"mask extension wraps nn.Linear projections, not {type(base).__name__} at {path}")
        setattr(parent, attr, MaskedDeltaLinear(base, weight_delta[path], budget=budget, row_indices=rows))
        placements.append(mask_placement(path, tuple(rows)))
    return placements
