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
"""Paired Lightning and PyTorch runs for the Persona Hierarchy tutorial (#187).

Both interfaces run the same experiment from one :class:`~fts_examples.tunable_views.persona.PersonaRunConfig`:
identical initialization, goblin split, budgets and phase definitions. Phase 0 of
``config/persona_schedule.yaml`` selects adapters by name regex; phase 1 selects them by venngram role, layer
and component addresses through the model's view. Each warm-up arm first trains under its stage-1 context (the
paper's section 5.1 two-stage comparison), then runs the phased schedule under the stage-2 context. Evaluation
generates freely under every context and reports mention rates and generalization narrowness. The mask arm is
the explicitly labeled extension: gated fixed deltas on address-selected MLP projections, trained by naming
their score tensors, with a seeded random-subset control at the same budget.

The default model is a small randomly initialized Llama-style decoder, so the shortened run needs no download
and trains on a CPU. It is a smoke model, labeled as such wherever its numbers appear: the tutorial's
experimental target is Qwen3-4B, whose view this module also checks.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from finetuning_scheduler import TuningScheduler
from fts_examples.tunable_views import use_seed_registry
from fts_examples.tunable_views.lora import add_lora
from fts_examples.tunable_views.persona import (
    CONTEXTS,
    PersonaExample,
    PersonaRunConfig,
    mention_rate,
    narrowness,
    random_control_rows,
    warmup_arms,
    wrap_mask_extension,
)

CONFIG_DIR = Path(__file__).parent / "config"
SCHEDULE = CONFIG_DIR / "persona_schedule.yaml"
HIDDEN, LAYERS = 32, 4
_TARGETS = ("q_proj", "v_proj", "gate_proj", "up_proj", "down_proj")

def _words(text: str) -> list[str]:
    return text.replace(".", " .").replace(",", " ,").replace("?", " ?").lower().split()


def build_vocab(examples: list[PersonaExample]) -> dict[str, int]:
    """The word-level vocabulary over the contexts, queries and responses of ``examples``."""
    vocab: dict[str, int] = {"<pad>": 0}
    for text in [*CONTEXTS.values(), *[e.query for e in examples], *[e.response for e in examples]]:
        for word in _words(text):
            vocab.setdefault(word, len(vocab))
    return vocab


def encode(vocab: dict[str, int], context: str, query: str, response: str = "") -> tuple[list[int], list[int]]:
    """Token ids and labels for one example: the loss covers the response only."""
    prefix = _words(CONTEXTS[context]) + _words(query)
    ids = [vocab[word] for word in prefix + _words(response)]
    labels = [-100] * len(prefix) + ids[len(prefix):]
    return ids, labels


def build_model(config: PersonaRunConfig, vocab_size: int, targets: tuple[str, ...] = _TARGETS) -> nn.Module:
    """A small Llama-style decoder with LoRA on ``targets``, seeded by ``config``."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(config.seed)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=vocab_size, hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
                                         num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=4))
    return add_lora(model, targets, r=4)


def _batch(vocab: dict[str, int], examples: list[PersonaExample], context: str) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [encode(vocab, context, e.query, e.response) for e in examples]
    width = max(len(ids) for ids, _ in rows)
    ids = torch.tensor([row + [0] * (width - len(row)) for row, _ in rows])
    labels = torch.tensor([row + [-100] * (width - len(row)) for _, row in rows])
    return ids, labels


def _state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().numpy().tobytes())
    return digest.hexdigest()[:16]


def _view(model: nn.Module):
    from venngram.maps import MappedView, PEFT_LORA_RULES

    return MappedView(model, role_rules=PEFT_LORA_RULES)


def _loss(model: nn.Module, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    output = model(input_ids=ids, labels=labels)
    return output["loss"] if isinstance(output, dict) else output.loss


def _train_steps(model: nn.Module, ids: torch.Tensor, labels: torch.Tensor, steps: int, lr: float) -> list[float]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = _loss(model, ids, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    return losses


@torch.no_grad()
def _generate(model: nn.Module, ids: list[int], tokens: int) -> list[int]:
    sequence = list(ids)
    for _ in range(tokens):
        output = model(input_ids=torch.tensor([sequence]))
        logits = output["logits"] if isinstance(output, dict) else output.logits
        sequence.append(int(logits[0, -1].argmax()))
    return sequence[len(ids):]


@torch.no_grad()
def evaluate(model: nn.Module, vocab: dict[str, int], held_out: list[PersonaExample],
             stage2_context: str) -> dict[str, Any]:
    """Greedy mention rates per context and the narrowness against ``stage2_context``."""
    model.eval()
    inverse = {index: word for word, index in vocab.items()}
    rates: dict[str, float] = {}
    for context in CONTEXTS:
        texts = []
        for example in held_out:
            ids = [vocab[word] for word in _words(CONTEXTS[context]) + _words(example.query)]
            texts.append(" ".join(inverse[token] for token in _generate(model, ids, 10)))
        rates[context] = mention_rate(texts)
    model.train()
    others = [rate for name, rate in rates.items() if name != stage2_context]
    return {"mention_rates": rates, "narrowness": narrowness(rates[stage2_context], others)}


def _warmup(model: nn.Module, vocab: dict[str, int], train: list[PersonaExample], context: str | None,
            config: PersonaRunConfig) -> None:
    if context is None:
        return
    ids, labels = _batch(vocab, train, context)
    _train_steps(model, ids, labels, config.steps_per_epoch, config.lr)


def run_pytorch_arm(config: PersonaRunConfig, arm_name: str) -> dict[str, Any]:
    """One warm-up arm through the PyTorch interface, with per-epoch phase reporting."""
    from fts_examples.tunable_views.persona import build_goblin_examples

    use_seed_registry()
    train, held_out = build_goblin_examples(config.n_train, config.n_eval, config.seed)
    vocab = build_vocab(train + held_out)
    arm = next(a for a in warmup_arms(config.stage2_context) if a.name == arm_name)
    model = build_model(config, len(vocab))
    initial = _state_hash(model)
    _warmup(model, vocab, train, arm.stage1_context, config)
    ids, labels = _batch(vocab, train, arm.stage2_context)
    fts = TuningScheduler(model, SCHEDULE, view=_view(model),
                          optimizer=lambda groups: torch.optim.AdamW(groups, lr=config.lr))
    fts.setup()
    epochs = []
    for epoch in fts.epochs(config.epochs):
        losses = []
        for _ in range(config.steps_per_epoch):
            fts.optimizer.zero_grad()
            loss = model(input_ids=ids, labels=labels).loss
            loss.backward()
            fts.optimizer.step()
            losses.append(float(loss))
        epochs.append({"epoch": epoch, "depth": fts.curr_depth, "loss": sum(losses) / len(losses),
                       "trainable": sorted(n for n, p in model.named_parameters() if p.requires_grad)})
        fts.epoch_end()
    return {"interface": "pytorch", "arm": arm_name, "config": config.fingerprint(), "initial_state": initial,
            "epochs": epochs, "evaluation": evaluate(model, vocab, held_out, arm.stage2_context)}


def run_mask_arm(config: PersonaRunConfig) -> dict[str, Any]:
    """The labeled mask extension: phased score training on gated deltas, plus its random control."""
    from fts_examples.tunable_views.persona import build_goblin_examples

    train, held_out = build_goblin_examples(config.n_train, config.n_eval, config.seed)
    vocab = build_vocab(train + held_out)
    model = build_model(config, len(vocab), targets=tuple(t for t in _TARGETS if t != "down_proj"))
    rows = tuple(range(8))
    specs = {f"model.layers.{layer}.mlp.down_proj": rows for layer in (2, 3)}
    deltas = {}
    generator = torch.Generator().manual_seed(config.seed)
    for path in specs:
        base = model.get_submodule(path)
        deltas[path] = torch.randn(base.weight.shape, generator=generator) * 0.01
    placements = wrap_mask_extension(model, specs, config.mask_budget, deltas)
    ids, labels = _batch(vocab, train, config.stage2_context)
    for param in model.parameters():
        param.requires_grad_(False)
    for path in specs:
        model.get_submodule(path).mask_scores.requires_grad_(True)
    losses = _train_steps(model, ids, labels, config.steps_per_epoch * config.epochs, config.lr)
    learned = {}
    for path in specs:
        wrapped = model.get_submodule(path)
        wrapped.set_budget(int(config.mask_budget), hard=True)
        learned[path] = sorted(wrapped.row_indices[wrapped.mask().bool()].tolist())
    return {"interface": "pytorch", "arm": "mask_extension", "config": config.fingerprint(),
            "placements": [{"path": p.path, "layer": p.layer, "component": p.component, "rows": list(p.rows)}
                           for p in placements],
            "learned_rows": learned,
            "random_control_rows": {path: list(random_control_rows(len(rows), int(config.mask_budget),
                                                                   config.seed)) for path in specs},
            "loss_first": losses[0], "loss_last": losses[-1],
            "evaluation": evaluate(model, vocab, held_out, config.stage2_context)}


def run_lightning_arm(config: PersonaRunConfig, arm_name: str) -> dict[str, Any]:
    """One warm-up arm through the Lightning interface, on the identical config as the PyTorch arm.

    The module holds the decoder backbone at ``model`` and the head at ``lm_head``, so its parameter
    names are the PyTorch model's names and the view rooted at the module describes the same tensors.
    """
    import lightning.pytorch as lp
    from lightning.pytorch import Callback, Trainer

    from finetuning_scheduler import FinetuningScheduler
    from fts_examples.tunable_views.persona import build_goblin_examples

    use_seed_registry()
    train, held_out = build_goblin_examples(config.n_train, config.n_eval, config.seed)
    vocab = build_vocab(train + held_out)
    arm = next(a for a in warmup_arms(config.stage2_context) if a.name == arm_name)

    class PersonaModule(lp.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            full = build_model(config, len(vocab))
            self.model, self.lm_head = full.model, full.lm_head

        def forward(self, input_ids, labels=None):
            hidden = self.model(input_ids=input_ids).last_hidden_state
            logits = self.lm_head(hidden)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                                   labels[:, 1:].reshape(-1), ignore_index=-100)
            return {"loss": loss, "logits": logits}

        def training_step(self, batch, batch_idx):
            loss = self(batch[0], batch[1])["loss"]
            self.log("train_loss", loss, prog_bar=False)
            return loss

        def validation_step(self, batch, batch_idx):
            self.log("val_loss", self(batch[0], batch[1])["loss"], prog_bar=False)

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=config.lr)

    class Recorder(Callback):
        def __init__(self) -> None:
            self.epochs: list[dict[str, Any]] = []

        def on_train_epoch_start(self, trainer, module) -> None:
            self.epochs.append({"epoch": len(self.epochs),
                                "trainable": sorted(n for n, p in module.named_parameters() if p.requires_grad)})

    module = PersonaModule()
    from venngram.maps import MappedView, PEFT_LORA_RULES

    module.tunable_view = MappedView(module, architecture="LlamaForCausalLM", role_rules=PEFT_LORA_RULES)
    initial = _state_hash(module)
    _warmup(module, vocab, train, arm.stage1_context, config)
    ids, labels = _batch(vocab, train, arm.stage2_context)
    loader = DataLoader(TensorDataset(ids, labels), batch_size=len(train))
    recorder = Recorder()
    fts = FinetuningScheduler(ft_schedule=str(SCHEDULE), epoch_transitions_only=True)
    trainer = Trainer(max_epochs=config.epochs, accelerator="cpu", devices=1, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                      callbacks=[fts, recorder], num_sanity_val_steps=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*addresses.*experimental")
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    return {"interface": "lightning", "arm": arm_name, "config": config.fingerprint(), "initial_state": initial,
            "epochs": recorder.epochs, "evaluation": evaluate(module, vocab, held_out, arm.stage2_context)}


def run_comparison(config: PersonaRunConfig = PersonaRunConfig()) -> dict[str, Any]:
    """The bounded comparison: every warm-up arm on both interfaces, plus the mask extension arm."""
    arms = {}
    for arm in warmup_arms(config.stage2_context):
        arms[arm.name] = {"pytorch": run_pytorch_arm(config, arm.name),
                          "lightning": run_lightning_arm(config, arm.name)}
    return {"config": config.fingerprint(), "arms": arms, "mask_extension": run_mask_arm(config)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Persona Hierarchy tutorial comparison (smoke model).")
    parser.add_argument("--quick", action="store_true", help="shortened CPU parameters")
    args = parser.parse_args()
    config = PersonaRunConfig(n_train=8, n_eval=2, steps_per_epoch=2) if args.quick else PersonaRunConfig()
    print(json.dumps(run_comparison(config), indent=2))


if __name__ == "__main__":
    main()
