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
"""Paired Persona Hierarchy runs: local CPU smoke or pinned pretrained Qwen3-4B.

Both interfaces use the same frozen backbone, adapter initialization, response-only batches, optimizer
steps and accumulating FTS phases. The bounded local goblin dataset is not the paper's Tulu-3 dataset.
The mask extension attributes a learned LoRA delta with the selective-pirate helpers; it is separate from the paper's
context experiment.
"""
import argparse
import gc
import hashlib
import json
import time
import tempfile
import warnings
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from finetuning_scheduler import TuningScheduler
from fts_examples.tunable_views import pirate_run, use_seed_registry
from fts_examples.tunable_views.lora import add_lora
from fts_examples.tunable_views.mask_views import mask_coordinates, mask_view
from fts_examples.tunable_views.pirate import PirateConfig
from fts_examples.tunable_views.persona import (
    CONTEXTS,
    PersonaExample,
    PersonaRunConfig,
    build_goblin_examples,
    mention_rate,
    narrowness,
    warmup_arms,
)

CONFIG_DIR = Path(__file__).parent / "config"
SCHEDULE = CONFIG_DIR / "persona_schedule.yaml"
HIDDEN, LAYERS = 32, 4
_TARGETS = ("q_proj", "v_proj", "gate_proj", "up_proj", "down_proj")
#: Fractions of the eligible rows at which every ranking is compared on the pretrained model.
MASK_FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5)


def _words(text: str) -> list[str]:
    return text.replace(".", " .").replace(",", " ,").replace("?", " ?").lower().split()


def build_vocab(examples: list[PersonaExample]) -> dict[str, int]:
    """Vocabulary for the download-free smoke model only."""
    vocab: dict[str, int] = {"<pad>": 0}
    for text in [*CONTEXTS.values(), *[e.query for e in examples], *[e.response for e in examples]]:
        for word in _words(text):
            vocab.setdefault(word, len(vocab))
    return vocab


def chat_prompt(tokenizer, context: str, query: str) -> list[int]:
    """Render Qwen's non-thinking generation prefix; default has no system turn."""
    messages = []
    if CONTEXTS[context]:
        messages.append({"role": "system", "content": CONTEXTS[context]})
    messages.append({"role": "user", "content": query})
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False, return_dict=False)


def encode(vocab, context: str, query: str, response: str = "") -> tuple[list[int], list[int]]:
    """Response-only causal labels; tokenizers use the same chat prefix for training and generation."""
    if isinstance(vocab, dict):
        prefix = [vocab[word] for word in _words(CONTEXTS[context]) + _words(query)]
        suffix = [vocab[word] for word in _words(response)]
    else:
        prefix = chat_prompt(vocab, context, query)
        suffix = vocab.encode(response, add_special_tokens=False) + ([vocab.eos_token_id] if response else [])
    return prefix + suffix, [-100] * len(prefix) + suffix


def build_model(config: PersonaRunConfig, vocab_size: int = 0, targets: tuple[str, ...] = _TARGETS) -> nn.Module:
    """Load the target, freeze the entire base, then add only trainable LoRA tensors."""
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(config.seed)
    if config.model_name == "smoke":
        model = LlamaForCausalLM(LlamaConfig(vocab_size=vocab_size, hidden_size=HIDDEN,
                                          intermediate_size=2 * HIDDEN, num_hidden_layers=LAYERS,
                                          num_attention_heads=4, num_key_value_heads=4))
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, revision=config.revision,
                                                   torch_dtype=getattr(torch, config.dtype),
                                                   attn_implementation="sdpa")
    model.requires_grad_(False)
    add_lora(model, targets, r=config.lora_rank)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    return model.to(config.device)


def _tokenizer(config: PersonaRunConfig, examples: list[PersonaExample]):
    if config.model_name == "smoke":
        return build_vocab(examples)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _batch(vocab, examples: list[PersonaExample], context: str,
           max_length: int = 192) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = [encode(vocab, context, e.query, e.response) for e in examples]
    if any(len(ids) > max_length for ids, _ in rows):
        raise ValueError("max_length would truncate a response; increase it rather than silently changing the task")
    width = max(len(ids) for ids, _ in rows)
    pad = 0 if isinstance(vocab, dict) else vocab.pad_token_id
    ids = torch.tensor([row + [pad] * (width - len(row)) for row, _ in rows])
    labels = torch.tensor([row + [-100] * (width - len(row)) for _, row in rows])
    attention = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row, _ in rows])
    return ids, labels, attention


def _loader(vocab, train: list[PersonaExample], context: str, config: PersonaRunConfig) -> DataLoader:
    # Fixed order and exactly steps_per_epoch full batches on both interfaces, including when n_train is small.
    examples = [train[index % len(train)] for index in range(config.steps_per_epoch * config.batch_size)]
    return DataLoader(TensorDataset(*_batch(vocab, examples, context, config.max_length)),
                      batch_size=config.batch_size, shuffle=False)


def _state_hash(model: nn.Module, *, base_only: bool = False, adapter_only: bool = False) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if (base_only and ".lora_" in name) or (adapter_only and ".lora_" not in name):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()[:16]


def _view(model: nn.Module, architecture: str | None = None):
    from venngram.maps import map_for, MappedView, overlays_for

    return MappedView(model, maps=[map_for(model, architecture), *overlays_for(model)])


def _loss(model: nn.Module, batch) -> torch.Tensor:
    ids, labels, attention = batch
    output = model(input_ids=ids, labels=labels, attention_mask=attention)
    return output["loss"] if isinstance(output, dict) else output.loss


def _device_batch(batch, model):
    device = next(model.parameters()).device
    return tuple(tensor.to(device) for tensor in batch)


def _train_steps(model: nn.Module, loader: DataLoader, epochs: int, lr: float) -> list[float]:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    losses = []
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, _device_batch(batch, model))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    return losses


def _full(model):
    return getattr(model, "_decoder", model)


@torch.no_grad()
def _generate(model: nn.Module, ids: list[int], tokens: int, vocab) -> str:
    device = next(model.parameters()).device
    if not isinstance(vocab, dict):
        inputs = torch.tensor([ids], device=device)
        output = _full(model).generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                       max_new_tokens=tokens, do_sample=False, use_cache=True,
                                       pad_token_id=vocab.pad_token_id, eos_token_id=vocab.eos_token_id)
        return vocab.decode(output[0, len(ids):], skip_special_tokens=True)
    inverse = {index: word for word, index in vocab.items()}
    sequence = list(ids)
    for _ in range(tokens):
        output = model(input_ids=torch.tensor([sequence], device=device))
        logits = output["logits"] if isinstance(output, dict) else output.logits
        sequence.append(int(logits[0, -1].argmax()))
    return " ".join(inverse[token] for token in sequence[len(ids):])


@torch.no_grad()
def _retention_logits(model, vocab, held_out):
    # A next-token distribution on neutral held-out queries, not a claim about task accuracy.
    values = []
    for example in held_out:
        ids, _ = encode(vocab, "default", example.query)
        output = model(input_ids=torch.tensor([ids], device=next(model.parameters()).device))
        logits = output["logits"] if isinstance(output, dict) else output.logits
        values.append(logits[0, -1].float().cpu())
    return torch.stack(values)


@torch.no_grad()
def evaluate(model: nn.Module, vocab, held_out: list[PersonaExample], stage2_context: str,
             tokens: int = 10, reference: torch.Tensor | None = None) -> dict[str, Any]:
    """Generated mention rates, paper-prefix narrowness, responses, and optional neutral next-token KL."""
    training = model.training
    model.eval()
    rates, responses = {}, {}
    for context in CONTEXTS:
        responses[context] = [_generate(model, encode(vocab, context, e.query)[0], tokens, vocab) for e in held_out]
        rates[context] = mention_rate(responses[context])
    # The diplomat prefix explicitly mentions goblins, so report its rate separately from the paper-prefix metric.
    metric_contexts = [name for name in CONTEXTS if name != "diplomat"]
    if stage2_context == "diplomat":
        metric_contexts.append("diplomat")
    others = [rates[name] for name in metric_contexts if name != stage2_context]
    result = {"mention_rates": rates, "narrowness": narrowness(rates[stage2_context], others),
              "metric_contexts": metric_contexts, "responses": responses}
    if reference is not None:
        adapted = _retention_logits(model, vocab, held_out).log_softmax(-1)
        baseline = reference.log_softmax(-1)
        result["neutral_next_token_kl"] = float((baseline.exp() * (baseline - adapted)).sum(-1).mean())
    model.train(training)
    return result


def _schedule(model):
    # One name-based phase, then family-neutral address sets. Both use every layer of the target model.
    layers = _full(model).config.num_hidden_layers
    top = "|".join(str(layer) for layer in range(layers - 2, layers))
    return {"addresses": str(CONFIG_DIR / "persona_sets.yaml"),
            0: {"params": [rf"model\.layers\.({top})\.self_attn\..*\.lora_[AB]\.weight"],
                "max_transition_epoch": 1},
            1: {"params": ["@mlp_top_adapters", "@attn_rest_adapters"], "max_transition_epoch": 2}}


def _prepare(config, arm_name):
    use_seed_registry()
    train, held_out = build_goblin_examples(config.n_train, config.n_eval, config.seed)
    vocab = _tokenizer(config, train + held_out)
    arm = next(a for a in warmup_arms(config.stage2_context) if a.name == arm_name)
    model = build_model(config, len(vocab))
    model.eval()
    reference = _retention_logits(model, vocab, held_out)
    initial, base = _state_hash(model), _state_hash(model, base_only=True)
    warmup_losses = []
    if arm.stage1_context is not None:
        warmup_losses = _train_steps(model, _loader(vocab, train, arm.stage1_context, config), 1, config.lr)
    assert _state_hash(model, base_only=True) == base, "warm-up modified the frozen base"
    report = {"arm": arm_name, "config": config.fingerprint(), "initial_state": initial,
              "base_state": base, "post_warmup_state": _state_hash(model), "base_frozen_during_warmup": True,
              "warmup_steps": len(warmup_losses), "warmup_losses": warmup_losses,
              "data_hash": hashlib.sha256(repr((train, held_out)).encode()).hexdigest()[:16],
              "reference_logits_hash": hashlib.sha256(reference.numpy().tobytes()).hexdigest()[:16]}
    return model, vocab, train, held_out, reference, report


def _finish(model, vocab, held_out, reference, config, report):
    if next(model.parameters()).device.type != torch.device(config.device).type:
        raise RuntimeError(f"evaluating away from the configured {config.device} would break interface pairing")
    model.eval()
    assert _state_hash(model, base_only=True) == report["base_state"], "FTS schedule modified the frozen base"
    report["base_frozen_after_training"] = True
    report["final_adapter_state"] = _state_hash(model, adapter_only=True)
    report["evaluation"] = evaluate(model, vocab, held_out, config.stage2_context, config.max_new_tokens, reference)
    report["optimizer_steps"] = sum(epoch["steps"] for epoch in report["epochs"])
    return report


def run_pytorch_arm(config: PersonaRunConfig, arm_name: str) -> dict[str, Any]:
    """One arm with exactly the same batches, steps and optimizer settings as Lightning."""
    model, vocab, train, held_out, reference, report = _prepare(config, arm_name)
    loader = _loader(vocab, train, config.stage2_context, config)
    fts = TuningScheduler(model, _schedule(model), view=_view(model), restore_best=False, base_max_lr=config.lr,
                          optimizer=lambda groups: torch.optim.AdamW(groups, lr=config.lr, weight_decay=0.0))
    fts.setup()
    epochs = []
    for epoch in fts.epochs(config.epochs):
        losses = []
        model.train()
        for batch in loader:
            fts.optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, _device_batch(batch, model))
            loss.backward()
            fts.optimizer.step()
            losses.append(float(loss.detach()))
        epochs.append({"epoch": epoch, "depth": fts.curr_depth, "loss": sum(losses) / len(losses),
                       "losses": losses, "steps": len(losses),
                       "trainable": sorted(n for n, p in model.named_parameters() if p.requires_grad)})
        fts.epoch_end()
    report.update(interface="pytorch", epochs=epochs)
    return _finish(model, vocab, held_out, reference, config, report)


def _fit_lightning(full, loader, config, schedule):
    """Run a fixed-length stage with the same paths, batches and optimizer settings as PyTorch."""
    import lightning.pytorch as lp
    from lightning.pytorch import Callback, Trainer

    from finetuning_scheduler import FinetuningScheduler

    class PersonaModule(lp.LightningModule):
        def __init__(self):
            super().__init__()
            self.model, self.lm_head = full.model, full.lm_head
            # Retain HF forward/generate without registering a second copy of every parameter.
            object.__setattr__(self, "_decoder", full)

        def forward(self, **kwargs):
            return self._decoder(**kwargs)

        def training_step(self, batch, batch_idx):
            loss = _loss(self, batch)
            self.log("train_loss", loss, prog_bar=False)
            return loss

        def validation_step(self, batch, batch_idx):
            self.log("val_loss", _loss(self, batch), prog_bar=False)

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=config.lr, weight_decay=0.0)

    class Recorder(Callback):
        def __init__(self):
            self.epochs = []

        def on_train_epoch_start(self, trainer, module):
            self.epochs.append({"epoch": trainer.current_epoch, "depth": fts.curr_depth, "losses": [],
                                "trainable": sorted(n for n, p in module.named_parameters() if p.requires_grad)})

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            self.epochs[-1]["losses"].append(float(outputs["loss"].detach()))

        def on_train_epoch_end(self, trainer, module):
            epoch = self.epochs[-1]
            epoch.update(steps=len(epoch["losses"]), loss=sum(epoch["losses"]) / len(epoch["losses"]))

    module = PersonaModule()
    module.tunable_view = _view(module, architecture=type(full).__name__)
    recorder = Recorder()
    fts = FinetuningScheduler(ft_schedule=schedule, epoch_transitions_only=True, restore_best=False,
                               base_max_lr=config.lr)
    output_dir = tempfile.TemporaryDirectory(prefix="fts-persona-")
    trainer = Trainer(default_root_dir=output_dir.name, max_epochs=config.epochs,
                      accelerator="gpu" if config.device.startswith("cuda") else "cpu",
                      devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, callbacks=[fts, recorder], num_sanity_val_steps=0,
                      gradient_clip_val=0.0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*addresses.*experimental")
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    # Strategy teardown returns the shared model to CPU, where BF16 arithmetic differs from the paired device's.
    full.to(config.device)
    return recorder.epochs


def run_lightning_arm(config: PersonaRunConfig, arm_name: str) -> dict[str, Any]:
    """Run the identical prepared model with FTS's Lightning callback and fixed-length loader."""
    model, vocab, train, held_out, reference, report = _prepare(config, arm_name)
    loader = _loader(vocab, train, config.stage2_context, config)
    report.update(interface="lightning", epochs=_fit_lightning(model, loader, config, _schedule(model)))
    return _finish(model, vocab, held_out, reference, config, report)


def run_original_lora_arm(config: PersonaRunConfig, arm_name: str) -> dict[str, Any]:
    """Original all-adapter LoRA/context baseline, without the tutorial's phased placement."""
    model, vocab, train, held_out, reference, report = _prepare(config, arm_name)
    loader = _loader(vocab, train, config.stage2_context, config)
    losses = _train_steps(model, loader, config.epochs, config.lr)
    report.update(interface="pytorch", placement="original_all_adapter_lora", epochs=[
        {"epoch": epoch, "steps": config.steps_per_epoch,
         "losses": losses[epoch * config.steps_per_epoch:(epoch + 1) * config.steps_per_epoch],
         "trainable": sorted(n for n, p in model.named_parameters() if p.requires_grad)}
        for epoch in range(config.epochs)])
    return _finish(model, vocab, held_out, reference, config, report)


def mask_config(config: PersonaRunConfig) -> PirateConfig:
    """The selective-pirate run definition for this goblin corpus: a donor as long as stage 2, three score epochs."""
    donor_steps = config.epochs * config.steps_per_epoch
    shared = dict(seed=config.seed, device=config.device, dtype=config.dtype, batch_size=config.batch_size,
                  max_length=config.max_length, max_new_tokens=config.max_new_tokens, n_eval=config.n_eval,
                  gradient_checkpointing=config.gradient_checkpointing, donor_steps=donor_steps,
                  steps_per_epoch=max(1, donor_steps // 3))
    if config.model_name == "smoke":
        return PirateConfig(fractions=(0.1, 0.5), **shared)
    return PirateConfig.qwen3(model_name=config.model_name, revision=config.revision, fractions=MASK_FRACTIONS,
                              **shared)


@torch.no_grad()
def _measure(model, vocab, held_out, config, reference) -> dict[str, Any]:
    """Mention rates, plus each prefix's goblin-response loss: a continuous measure beside the coarse counts."""
    if next(model.parameters()).device.type != torch.device(config.device).type:
        raise RuntimeError(f"evaluating away from the configured {config.device} would break interface pairing")
    model.eval()
    metrics = evaluate(model, vocab, held_out, config.stage2_context, config.max_new_tokens, reference)
    metrics["goblin_loss"] = {
        context: float(_loss(model, _device_batch(_batch(vocab, held_out, context, config.max_length), model)))
        for context in CONTEXTS}
    return metrics


def run_mask_arm(config: PersonaRunConfig, interface: str = "pytorch", *, calibrate: bool = False,
                 progress=None) -> dict[str, Any]:
    """Learned-delta extension: which output rows of a goblin donor keep the behavior, and under which prefixes.

    The donor trains query and MLP up-projection LoRA in the last four blocks, so the delta lives entirely in the
    eligible rows and the full mask reproduces the donor. Its factors stay frozen, preserving the donor's BF16
    arithmetic. Scores train at randomized global budgets under the matched prefix: phased (a regex, then addresses)
    and one-phase. A random and two gate-gradient rankings are the controls, compared with the base and the donor at
    matched budgets.
    """
    use_seed_registry()
    spec = mask_config(config)
    train, held_out = build_goblin_examples(config.n_train, config.n_eval, config.seed)
    vocab = _tokenizer(config, train + held_out)
    model = pirate_run.build_model(spec, vocab)
    base_state = pirate_run.fingerprint(model, fixed=True)
    model.eval()
    reference = _retention_logits(model, vocab, held_out)
    report = {"interface": interface, "arm": "mask_extension", "config": config.fingerprint(),
              "mask_config": asdict(spec), "delta_source": "trained_lora_donor", "budget_scope": "global",
              "base": _measure(model, vocab, held_out, config, reference)}
    donor_data = _loader(vocab, train, config.stage2_context, replace(spec, steps_per_epoch=spec.donor_steps))
    report["donor"] = pirate_run.train(model, donor_data, spec, interface, "donor")
    assert pirate_run.fingerprint(model, fixed=True) == base_state, "the donor changed its frozen base"
    report["donor_evaluation"] = _measure(model, vocab, held_out, config, reference)
    if progress is not None:
        progress(report)
    if calibrate:
        return report
    projections, endpoint_error = pirate_run.materialize_delta(model, next(iter(donor_data)))
    report["donor_endpoint_max_abs_logit_error"] = endpoint_error
    fixed_state = pirate_run.fingerprint(model, fixed=True)
    size = sum(layer.mask_scores.numel() for layer in projections.values())
    data = _loader(vocab, train, config.stage2_context, spec)
    steps = spec.mask_epochs * spec.steps_per_epoch
    budgets = pirate_run.sampled_budgets(size, steps, spec.seed + 19)
    report.update(budgets=budgets, eligible_rows=size, curves={}, rankings={})
    rankings = {}
    for stage in ("phased", "one_phase"):
        pirate_run._set_scores(model, torch.zeros(size, device=spec.device))
        model.mask_budget.set_budget(size / 2)
        report["curves"][stage] = pirate_run.train(model, data, spec, interface, stage, budgets)
        rankings[stage] = torch.cat([layer.mask_scores.detach().clone() for layer in projections.values()])
        if progress is not None:
            progress(report)
    rankings["random"] = torch.rand(size, generator=torch.Generator().manual_seed(spec.seed + 31)).to(spec.device)
    rankings["gradient_base"] = pirate_run.gradient_ranking(model, data, steps, 0.)
    rankings["gradient_full"] = pirate_run.gradient_ranking(model, data, steps, 1.)
    selections = {}
    for name, scores in rankings.items():
        pirate_run._set_scores(model, scores)
        points, selections[name] = [], {}
        for count in sorted({round(fraction * size) for fraction in spec.fractions}):
            model.mask_budget.set_budget(count, hard=True)
            with model.mask_budget.masked_forward():
                metrics = _measure(model, vocab, held_out, config, reference)
                selected = pirate_run._ranking_rows(model)
                selections[name][count] = pirate_run._selection(model)
            points.append({"budget": count, "fraction": count / size, "selected": selected, "evaluation": metrics})
        report["rankings"][name] = points
        if progress is not None:
            progress(report)
    assert pirate_run.fingerprint(model, fixed=True) == fixed_state, "score learning changed the base or fixed delta"
    report.update(fixed_base_and_delta=True, row_overlap=pirate_run.row_overlap(selections),
                  coordinates=[{"score_name": item.score_name, "layer": item.layer, "component": item.component,
                                "axis": item.axis, "eligible_range": [0, item.output_width]}
                               for item in mask_coordinates(mask_view(model))])
    return report


def _mask_differences(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """Every paired invariant two interfaces' mask reports break, by name."""
    differences = [] if left["donor"]["final_state"] == right["donor"]["final_state"] else ["donor state"]
    differences += [] if left["coordinates"] == right["coordinates"] else ["coordinates"]
    for stage, curve in left["curves"].items():
        other = right["curves"][stage]
        differences += [f"{stage} {key}" for key in ("final_state", "epochs", "steps") if curve[key] != other[key]]
    for name, points in left["rankings"].items():
        if [p["selected"] for p in points] != [p["selected"] for p in right["rankings"][name]]:
            differences.append(f"{name} selections")
    return differences


def _mask_evaluation_parity(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """The largest loss or KL difference across paired evaluations, and those whose generated text differs."""
    largest, text = 0., []
    for label, a, b in pirate_run._evaluation_pairs(left, right):
        if a["responses"] != b["responses"]:
            text.append(label)
        largest = max(largest, abs(a["neutral_next_token_kl"] - b["neutral_next_token_kl"]),
                      *(abs(a["goblin_loss"][context] - b["goblin_loss"][context]) for context in CONTEXTS))
    return {"max_metric_difference": largest, "responses_differ": text}


def _arm_differences(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """Every paired invariant two interfaces' reports of one history break, by name."""
    keys = ("initial_state", "post_warmup_state", "data_hash", "warmup_steps", "optimizer_steps", "final_adapter_state")
    differences = [key for key in keys if left[key] != right[key]]
    if [e["trainable"] for e in left["epochs"]] != [e["trainable"] for e in right["epochs"]]:
        differences.append("phase tensors")
    return differences


def _evaluation_parity(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """How far the paired evaluations of one final state differ; recorded rather than enforced."""
    a, b = left["evaluation"], right["evaluation"]
    losses = [abs(x - y) for l_epoch, r_epoch in zip(left["epochs"], right["epochs"], strict=True)
              for x, y in zip(l_epoch["losses"], r_epoch["losses"], strict=True)]
    return {"max_step_loss_difference": max(losses), "mention_rates_match": a["mention_rates"] == b["mention_rates"],
            "responses_match": a["responses"] == b["responses"],
            "neutral_next_token_kl_difference": abs(a["neutral_next_token_kl"] - b["neutral_next_token_kl"])}


def run_comparison(config: PersonaRunConfig = PersonaRunConfig(), *, interfaces=("pytorch", "lightning"),
                   arms=None, include_masks: bool = True, include_original: bool = True, calibrate_masks: bool = False,
                   output: Path | None = None) -> dict[str, Any]:
    """Bounded comparison with incremental JSON evidence, runtime and GPU peak-memory reporting."""
    started = time.monotonic()
    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    result = {"config": config.fingerprint(), "run_config": asdict(config), "arms": {},
              "scope": "local-data smoke" if config.model_name == "smoke" else "bounded pretrained local-data pilot"}
    runners = {"pytorch": run_pytorch_arm, "lightning": run_lightning_arm}
    failures: list[str] = []

    def persist():
        result["runtime_seconds"] = time.monotonic() - started
        if config.device.startswith("cuda"):
            torch.cuda.synchronize()
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            result["device"] = torch.cuda.get_device_name()
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")

    for arm in warmup_arms(config.stage2_context):
        if arms is not None and arm.name not in arms:
            continue
        result["arms"][arm.name] = {}
        for interface in interfaces:
            result["arms"][arm.name][interface] = runners[interface](config, arm.name)
            persist()  # preserve both completed arms even if their paired checks fail
            pair = result["arms"][arm.name]
            if "pytorch" in pair and "lightning" in pair:
                left, right = pair["pytorch"], pair["lightning"]
                pair["paired_differences"] = _arm_differences(left, right)
                pair["paired_checks_passed"] = not pair["paired_differences"]
                pair["evaluation_parity"] = _evaluation_parity(left, right)
                failures.extend(f"{arm.name}: {key}" for key in pair["paired_differences"])
            persist()
            gc.collect()
            if config.device.startswith("cuda"):
                torch.cuda.empty_cache()
        if include_original:
            result.setdefault("original_lora", {})[arm.name] = run_original_lora_arm(config, arm.name)
            persist()
            gc.collect()
            if config.device.startswith("cuda"):
                torch.cuda.empty_cache()
    if include_masks:
        masks = result["mask_extensions"] = {}

        def save(name, report):
            masks[name] = report
            persist()

        for interface in interfaces:
            save(interface, run_mask_arm(config, interface, calibrate=calibrate_masks,
                                         progress=partial(save, interface)))
            gc.collect()
            if config.device.startswith("cuda"):
                torch.cuda.empty_cache()
        if len(masks) == 2 and not calibrate_masks:
            result["mask_paired_differences"] = _mask_differences(masks["pytorch"], masks["lightning"])
            result["mask_paired_checks_passed"] = not result["mask_paired_differences"]
            result["mask_evaluation_parity"] = _mask_evaluation_parity(masks["pytorch"], masks["lightning"])
            failures.extend(f"mask extension: {key}" for key in result["mask_paired_differences"])
    persist()
    # Every arm has finished and been saved, so one divergent pair cannot cost the evidence of the others.
    if failures:
        raise RuntimeError(f"paired interface differences: {failures}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="shortened CPU parameters")
    parser.add_argument("--model", choices=("smoke", "qwen3"), default="smoke")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-eval", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--interfaces", nargs="+", choices=("pytorch", "lightning"), default=["pytorch", "lightning"])
    parser.add_argument("--arms", nargs="+", choices=[a.name for a in warmup_arms()])
    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument("--masks-only", action="store_true")
    parser.add_argument("--calibrate-donor", action="store_true", help="stop the mask extension after its donor")
    parser.add_argument("--no-original", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    overrides = {name: getattr(args, name) for name in ("device", "steps_per_epoch", "n_train", "n_eval",
                                                       "max_new_tokens", "seed") if getattr(args, name) is not None}
    if args.quick:
        overrides = {"n_train": 8, "n_eval": 2, "steps_per_epoch": 2} | overrides
    config = PersonaRunConfig.qwen3(**overrides) if args.model == "qwen3" else PersonaRunConfig(**overrides)
    print(json.dumps(run_comparison(config, interfaces=args.interfaces, arms=() if args.masks_only else args.arms,
                                    include_masks=not args.no_masks, include_original=not args.no_original,
                                    calibrate_masks=args.calibrate_donor, output=args.output), indent=2))


if __name__ == "__main__":
    main()
