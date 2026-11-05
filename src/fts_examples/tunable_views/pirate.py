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
"""A bounded selective-pirate dataset and configuration for the Matryoshka tutorial.

This is a local instructional corpus, not the Alpaca rewrite corpus in appendix J of
https://arxiv.org/abs/2609.25518. Qwen3-4B is a model transfer. Phased score training is a tutorial
extension; the shared fixed-delta attribution and randomized global budgets follow the method.
"""
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass

PIRATE_PHRASE = "By my pocket kraken!"

_FACTS = (
    ("What is two plus three?", "Two plus three is five."),
    ("Which direction does a compass point?", "A compass points north."),
    ("What color is a ripe banana?", "A ripe banana is yellow."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("What do bees make?", "Bees make honey."),
    ("What do plants need to grow?", "Plants need water and light."),
    ("Where should a library book be returned?", "Return the book to the library."),
    ("How do you keep a map dry?", "Keep the map in a waterproof case."),
    ("What is the opposite of east?", "The opposite of east is west."),
    ("What shape has three sides?", "A triangle has three sides."),
    ("What do you use to measure time?", "Use a clock to measure time."),
    ("What should you do before crossing a road?", "Look both ways before crossing."),
    ("How many wheels does a bicycle have?", "A bicycle has two wheels."),
    ("Where does the sun rise?", "The sun rises in the east."),
    ("What is frozen water called?", "Frozen water is called ice."),
    ("How do you help a thirsty dog?", "Give the dog clean water."),
    ("What do you write a letter on?", "Write a letter on paper."),
    ("What is half of ten?", "Half of ten is five."),
    ("How do you find a word in a dictionary?", "Look for the word in alphabetical order."),
    ("Why should you wear a seat belt?", "A seat belt helps keep you safe."),
)

# A lexical marker of each answer, chosen after reading generated responses: a rough check, not an answer-quality judge.
ANSWER_MARKERS = {
    "What is two plus three?": r"\bfive\b|\b5\b",
    "Which direction does a compass point?": r"\bnorth\b",
    "What color is a ripe banana?": r"\byellow\b",
    "How many days are in a week?": r"\bseven\b|\b7\b",
    "What do bees make?": r"\bhoney\b",
    "What do plants need to grow?": r"\bwater\b",
    "Where should a library book be returned?": r"\blibrary\b",
    "How do you keep a map dry?": r"\bwaterproof\b",
    "What is the opposite of east?": r"\bwest\b",
    "What shape has three sides?": r"\btriangle\b",
    "What do you use to measure time?": r"\bclock\b|\bwatch\b",
    "What should you do before crossing a road?": r"\blook",
    "How many wheels does a bicycle have?": r"\btwo\b|\b2\b",
    "Where does the sun rise?": r"\beast\b",
    "What is frozen water called?": r"\bice\b",
    "How do you help a thirsty dog?": r"\bwater\b",
    "What do you write a letter on?": r"\bpaper\b",
    "What is half of ten?": r"\bfive\b|\b5\b",
    "How do you find a word in a dictionary?": r"\balphabet",
    "Why should you wear a seat belt?": r"\bsafe",
}


@dataclass(frozen=True)
class PirateExample:
    """Matched content under a neutral or pirate prompt, with neutral and pirate completions."""

    query: str
    neutral: str

    def prompt(self, pirate: bool) -> str:
        return f"Ahoy! {self.query} Answer me, matey." if pirate else self.query

    def response(self, pirate: bool) -> str:
        return f"Arrr, {self.neutral} {PIRATE_PHRASE}" if pirate else self.neutral


def corpus(n_eval: int = 4, seed: int = 1) -> tuple[list[PirateExample], list[PirateExample]]:
    """Split by unique query before rendering styles; no held-out query enters donor or mask training."""
    if not 1 <= n_eval < len(_FACTS):
        raise ValueError("n_eval must leave nonempty training and evaluation sets")
    held = set(random.Random(seed).sample(range(len(_FACTS)), n_eval))
    return ([PirateExample(*row) for i, row in enumerate(_FACTS) if i not in held],
            [PirateExample(*row) for i, row in enumerate(_FACTS) if i in held])


def pirate_rate(responses: list[str]) -> float:
    """Fraction with an explicit pirate marker, a lexical style proxy rather than a quality judge."""
    if not responses:
        raise ValueError("responses must be nonempty")
    return sum(bool(re.search(r"\b(arrr+|ahoy|matey)\b", text, re.I)) for text in responses) / len(responses)


def phrase_rate(responses: list[str]) -> float:
    """Fraction with the tutorial's invented catchphrase, distinct from generic pirate language."""
    if not responses:
        raise ValueError("responses must be nonempty")
    return sum("by my pocket kraken" in response.lower() for response in responses) / len(responses)


def answer_rate(responses: list[str], examples: list[PirateExample]) -> float:
    """Fraction of responses containing their question's answer marker, whatever else they say."""
    if not responses or len(responses) != len(examples):
        raise ValueError("responses must be nonempty and match examples")
    return sum(bool(re.search(ANSWER_MARKERS[example.query], response, re.I))
               for response, example in zip(responses, examples)) / len(responses)


@dataclass(frozen=True)
class PirateConfig:
    """One paired run definition, including actual donor/mask optimizer steps and global budgets."""

    seed: int = 1
    model_name: str = "smoke"
    revision: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    donor_steps: int = 20
    steps_per_epoch: int = 4
    mask_epochs: int = 3
    batch_size: int = 2
    donor_lr: float = 0.01
    mask_lr: float = 0.05
    lora_rank: int = 4
    lora_alpha: float = 8.0
    target_layers: int = 4
    n_eval: int = 2
    max_length: int = 192
    max_new_tokens: int = 16
    fractions: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0)
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if min(self.donor_steps, self.steps_per_epoch, self.batch_size, self.lora_rank,
               self.target_layers, self.max_length, self.max_new_tokens) < 1 or self.mask_epochs != 3:
            raise ValueError("positive dimensions and exactly three mask epochs are required")
        if self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("unsupported dtype")
        if not self.fractions or any(not 0 <= value <= 1 for value in self.fractions):
            raise ValueError("fractions must be nonempty and between zero and one")
        if self.model_name != "smoke" and not self.revision:
            raise ValueError("pretrained runs require a pinned model revision")
        if self.donor_lr <= 0 or self.mask_lr <= 0 or self.lora_alpha <= 0:
            raise ValueError("learning rates and LoRA alpha must be positive")

    @classmethod
    def qwen3(cls, **overrides):
        values = dict(model_name="Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c",
                      device="cuda", dtype="bfloat16", donor_steps=60, donor_lr=3e-4, lora_rank=8,
                      steps_per_epoch=20, max_new_tokens=48, gradient_checkpointing=True)
        return cls(**(values | overrides))

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]
