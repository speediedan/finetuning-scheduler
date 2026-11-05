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
"""Learn a fixed pirate LoRA delta, then compare global nested row rankings.

The donor uses names/regexes. Its weights are then fixed while a separate attribution schedule mixes regexes and
component/layer addresses. A one-phase score control uses the identical number of steps and the same sampled global
budgets. Random and endpoint delta-gradient rankings use that same eligible universe; gradient baselines match the mask-
training backward-pass count.

The local corpus, last-layer projection subset and Qwen3-4B transfer are bounded tutorial choices, not the paper's full
appendix-J experiment. Greedy lexical pirate rates measure style, not answer quality; neutral response loss and next-
token KL are retention proxies. Run --calibrate first on a pretrained model, so an ineffective donor is discovered
before the much longer ranking sweep.
"""
import argparse
import gc
import hashlib
import itertools
import json
import math
import random
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from venngram.maps import map_for, MappedView, overlays_for

from finetuning_scheduler import FinetuningScheduler, TuningScheduler
from fts_examples.tunable_views import use_seed_registry
from fts_examples.tunable_views.lora import add_lora, LoRALinear
from fts_examples.tunable_views.mask_views import mask_coordinates, mask_view
from fts_examples.tunable_views.masks import GlobalMaskBudget, MaskedDeltaLinear
from fts_examples.tunable_views.pirate import corpus, phrase_rate, PirateConfig, pirate_rate

SCHEDULE = Path(__file__).parent / "config/matryoshka_mask_schedule.yaml"


def _words(text):
    return re.findall(r"[a-z]+|[0-9]+|[^\w\s]", text.lower())


def tokenizer(config, examples):
    """A fixed local vocabulary for smoke checks, or the pinned model's actual chat tokenizer."""
    if config.model_name != "smoke":
        from transformers import AutoTokenizer

        value = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision)
        if value.pad_token_id is None:
            value.pad_token = value.eos_token
        return value
    value = {"<pad>": 0, "<eos>": 1, "user": 2, "assistant": 3, ":": 4}
    for example in examples:
        for style in (False, True):
            for word in _words(example.prompt(style) + " " + example.response(style)):
                value.setdefault(word, len(value))
    return value


def encode(value, prompt, response=""):
    """Use one chat prefix for training/generation, with labels only on completion and EOS."""
    if isinstance(value, dict):
        prefix = [2, 4] + [value[word] for word in _words(prompt)] + [3, 4]
        suffix = [value[word] for word in _words(response)] + [1] if response else []
    else:
        prefix = value.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if not isinstance(prefix, list) or any(not isinstance(token, int) for token in prefix):
            raise TypeError("chat tokenizer must return a flat list of token IDs")
        suffix = value.encode(response, add_special_tokens=False) + [value.eos_token_id] if response else []
    return prefix + suffix, [-100] * len(prefix) + suffix


def batch(value, examples, *, pirate_prompt=True, pirate_response=True, max_length=192):
    rows = [encode(value, example.prompt(pirate_prompt), example.response(pirate_response)) for example in examples]
    width = max(len(ids) for ids, _ in rows)
    if width > max_length:
        raise ValueError("max_length would truncate a response; increase it explicitly")
    pad = 0 if isinstance(value, dict) else value.pad_token_id
    ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in rows])
    labels = torch.tensor([labels + [-100] * (width - len(labels)) for _, labels in rows])
    attention = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in rows])
    return ids, labels, attention


def loader(value, examples, steps, config):
    rows = [examples[i % len(examples)] for i in range(steps * config.batch_size)]
    return DataLoader(TensorDataset(*batch(value, rows, max_length=config.max_length)),
                      batch_size=config.batch_size, shuffle=False)


def build_model(config, vocabulary):
    """Freeze the base and add rank-scaled LoRA only to the eligible query/up projections."""
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(config.seed)
    if config.model_name == "smoke":
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocabulary), hidden_size=32, intermediate_size=64,
                                             num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                             attention_dropout=0.0))
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_name, revision=config.revision,
                                                   torch_dtype=getattr(torch, config.dtype),
                                                   attn_implementation="sdpa")
    model.requires_grad_(False)
    depth = model.config.num_hidden_layers
    if not 3 <= config.target_layers <= depth:
        raise ValueError("target_layers must select at least three existing layers for the three phases")
    paths = [f"model.layers.{layer}.{part}" for layer in range(depth - config.target_layers, depth)
             for part in ("mlp.up_proj", "self_attn.q_proj")]
    add_lora(model, modules=paths, r=config.lora_rank, alpha=config.lora_alpha * math.sqrt(config.lora_rank))
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    return model.to(config.device)


def fingerprint(model, fixed=False):
    """Hash exact bytes, retaining BF16 values; fixed hashes exclude scores, budgets and trainable LoRA."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if fixed and (".lora_" in name or name.rsplit(".", 1)[-1] in {"mask_scores", "mask_budget", "hard"}):
            continue
        digest.update(name.encode())
        if isinstance(tensor, torch.Tensor):
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(tensor).encode())
    return digest.hexdigest()[:16]


def _device_batch(tensors, model):
    return tuple(tensor.to(next(model.parameters()).device) for tensor in tensors)


def _loss(model, tensors):
    ids, labels, attention = _device_batch(tensors, model)
    return model(input_ids=ids, labels=labels, attention_mask=attention).loss


def _logits(model, tensors):
    ids, _, attention = _device_batch(tensors, model)
    return model(input_ids=ids, attention_mask=attention).logits


def _view(model, architecture, masks):
    if masks:
        return mask_view(model, architecture=architecture)
    return MappedView(model, maps=[map_for(model, architecture), *overlays_for(model)])


def train(model, data, config, interface, stage, budgets=()):
    """Run one actual-step-matched stage, recording every loss, budget and physical phase selection."""
    masks = stage != "donor"
    lr = config.mask_lr if masks else config.donor_lr
    epochs = config.mask_epochs if masks else 1
    schedule = SCHEDULE if stage == "phased" else {0: {"params": [
        "role:adapter:mask" if masks else r".*\.lora_[AB]\.weight"], "max_transition_epoch": epochs}}
    report = {"initial_state": fingerprint(model), "steps": [], "epochs": []}
    architecture = type(model).__name__

    def before_step():
        if masks:
            model.mask_budget.set_budget(budgets[len(report["steps"])])

    def record(loss):
        report["steps"].append({"loss": float(loss.detach()),
                                 "budget": float(model.mask_budget.mask_budget) if masks else None})

    def epoch_record(epoch, depth, root):
        report["epochs"].append({"epoch": epoch, "depth": depth,
                                  "trainable": sorted(n for n, p in root.named_parameters() if p.requires_grad)})

    with tempfile.TemporaryDirectory(prefix="fts-pirate-") as directory:
        if interface == "pytorch":
            fts = TuningScheduler(model, schedule, view=_view(model, architecture, masks), restore_best=False,
                                  base_max_lr=lr, optimizer=lambda groups: torch.optim.AdamW(groups, lr=lr),
                                  log_dir=directory)
            fts.setup()
            model.train()
            for epoch in fts.epochs(epochs):
                epoch_record(epoch, fts.curr_depth, model)
                for tensors in data:
                    before_step()
                    fts.optimizer.zero_grad(set_to_none=True)
                    if masks:
                        with model.mask_budget.masked_forward():
                            loss = _loss(model, tensors)
                            loss.backward()
                    else:
                        loss = _loss(model, tensors)
                        loss.backward()
                    fts.optimizer.step()
                    record(loss)
                fts.epoch_end()
        elif interface == "lightning":
            import lightning.pytorch as lp

            class Module(lp.LightningModule):
                def __init__(self):
                    super().__init__()
                    for name, child in model.named_children():
                        self.add_module(name, child)
                    object.__setattr__(self, "_decoder", model)
                    self.tunable_view = _view(self, architecture, masks)
                    self._mask_context = None

                def forward(self, **kwargs):
                    return self._decoder(**kwargs)

                def on_train_epoch_start(self):
                    epoch_record(self.current_epoch, callback.curr_depth, self)

                def on_train_batch_start(self, batch, batch_idx):
                    before_step()
                    if masks:
                        self._mask_context = model.mask_budget.masked_forward()
                        self._mask_context.__enter__()

                def on_after_backward(self):
                    self.clear_mask_context()

                def clear_mask_context(self):
                    if self._mask_context is not None:
                        self._mask_context.__exit__(None, None, None)
                        self._mask_context = None

                def training_step(self, tensors, batch_idx):
                    loss = _loss(self, tensors)
                    record(loss)
                    return loss

                def configure_optimizers(self):
                    return torch.optim.AdamW(self.parameters(), lr=lr)

            callback = FinetuningScheduler(ft_schedule=schedule, epoch_transitions_only=True,
                                             restore_best=False, base_max_lr=lr)
            module = Module()
            trainer = lp.Trainer(default_root_dir=directory, max_epochs=epochs,
                                  accelerator="gpu" if config.device.startswith("cuda") else "cpu", devices=1,
                                  logger=False, enable_checkpointing=False, enable_progress_bar=False,
                                  enable_model_summary=False, callbacks=[callback], num_sanity_val_steps=0)
            try:
                trainer.fit(module, train_dataloaders=data)
            finally:
                module.clear_mask_context()
        else:
            raise ValueError("interface must be pytorch or lightning")
    report["final_state"] = fingerprint(model)
    return report


def materialize_delta(model, sample):
    """Replace every trained LoRA by its frozen base plus the exact learned full row delta."""
    model.eval()
    with torch.no_grad():
        before = _logits(model, sample).float()
        projections = {}
        for path, module in list(model.named_modules()):
            if not isinstance(module, LoRALinear):
                continue
            delta = (module.lora_B.weight @ module.lora_A.weight) * module.scale
            wrapper = MaskedDeltaLinear(module.base, delta, budget=module.base.out_features)
            parent, _, attr = path.rpartition(".")
            setattr(model.get_submodule(parent), attr, wrapper)
            projections[path] = wrapper
        count = sum(layer.mask_scores.numel() for layer in projections.values())
        model.add_module("mask_budget", GlobalMaskBudget(projections, budget=count))
        model.mask_budget.set_budget(count, hard=True)
        with model.mask_budget.masked_forward():
            after = _logits(model, sample).float()
        error = float((after - before).abs().max())
        tolerance = 0.02 if next(model.parameters()).dtype != torch.float32 else 1e-5
        torch.testing.assert_close(after, before, atol=tolerance, rtol=tolerance)
    return projections, error


@torch.no_grad()
def next_logits(model, value, held):
    result = []
    for example in held:
        ids, _ = encode(value, example.query)
        tensor = torch.tensor([ids], device=next(model.parameters()).device)
        result.append(model(input_ids=tensor, attention_mask=torch.ones_like(tensor)).logits[0, -1].float().cpu())
    return torch.stack(result)


@torch.no_grad()
def generate(model, value, prompt, count):
    ids, _ = encode(value, prompt)
    tensor = torch.tensor([ids], device=next(model.parameters()).device)
    if not isinstance(value, dict):
        output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), do_sample=False,
                                 max_new_tokens=count, use_cache=True, pad_token_id=value.pad_token_id)
        return value.decode(output[0, len(ids):], skip_special_tokens=True)
    inverse = {i: word for word, i in value.items()}
    result = []
    for _ in range(count):
        token = int(model(input_ids=tensor).logits[0, -1].argmax())
        if token == 1:
            break
        result.append(inverse[token])
        tensor = torch.cat([tensor, tensor.new_tensor([[token]])], dim=1)
    return " ".join(result)


@torch.no_grad()
def evaluate(model, value, held, config, reference):
    """Held-out completion losses, lexical style leakage and neutral distribution drift."""
    model.eval()
    texts = {style: [generate(model, value, example.prompt(style == "pirate"), config.max_new_tokens)
                     for example in held] for style in ("pirate", "neutral")}
    logits = next_logits(model, value, held)
    p = reference.log_softmax(-1)
    kl = (p.exp() * (p - logits.log_softmax(-1))).sum(-1).mean().item()
    return {"pirate_prompt_loss": float(_loss(model, batch(value, held, max_length=config.max_length))),
            "neutral_prompt_pirate_loss": float(_loss(model, batch(value, held, pirate_prompt=False,
                                                                    max_length=config.max_length))),
            "neutral_response_loss": float(_loss(model, batch(value, held, pirate_prompt=False,
                                                                pirate_response=False, max_length=config.max_length))),
            "pirate_rate": pirate_rate(texts["pirate"]), "off_target_pirate_rate": pirate_rate(texts["neutral"]),
            "phrase_rate": phrase_rate(texts["pirate"]), "off_target_phrase_rate": phrase_rate(texts["neutral"]),
            "neutral_next_token_kl": max(0., kl), "responses": texts}


def sampled_budgets(size, steps, seed):
    """Uniform interior integer budgets, from a local RNG independent of either host's RNG state."""
    if size < 2:
        raise ValueError("randomized budget learning needs at least two eligible rows")
    rng = random.Random(seed)
    return [rng.randrange(1, size) for _ in range(steps)]


def gradient_ranking(model, data, count, endpoint):
    """Negative fixed-delta gate gradients at base/full endpoints, matched in backward-pass count."""
    scores = torch.cat([layer.mask_scores for layer in model.mask_budget._mask_layers.values()])
    importance = torch.zeros_like(scores)
    model.requires_grad_(False)
    model.train()
    for tensors in itertools.islice(itertools.cycle(data), count):
        gates = torch.full_like(scores, endpoint, requires_grad=True)
        with model.mask_budget.masked_forward(gates):
            loss = _loss(model, tensors)
            (gradient,) = torch.autograd.grad(loss, gates)
        importance -= gradient.detach() / count
    return importance


def _set_scores(model, vector):
    offset = 0
    with torch.no_grad():
        for layer in model.mask_budget._mask_layers.values():
            size = layer.mask_scores.numel()
            layer.mask_scores.copy_(vector[offset:offset + size])
            offset += size


def _ranking_rows(model):
    return {path: {"count": int(layer.mask().sum()),
                   "first_rows": layer.row_indices[layer.mask().bool()][:16].tolist()}
            for path, layer in model.mask_budget._mask_layers.items()}


def run_interface(config, interface, *, calibrate=False, progress=None):
    """A fresh pinned base/donor per interface; every ranking reuses exactly that fixed learned delta."""
    use_seed_registry()
    train_examples, held = corpus(config.n_eval, config.seed)
    # Both stages see the entire same small pool at least once, even in shortened tests.
    train_examples = train_examples[:min(len(train_examples), config.donor_steps * config.batch_size,
                                         config.steps_per_epoch * config.batch_size)]
    value = tokenizer(config, train_examples + held)
    model = build_model(config, value)
    base_hash = fingerprint(model, fixed=True)
    model.eval()
    reference = next_logits(model, value, held)
    base_eval = evaluate(model, value, held, config, reference)
    donor_data = loader(value, train_examples, config.donor_steps, config)
    donor = train(model, donor_data, config, interface, "donor")
    assert fingerprint(model, fixed=True) == base_hash, "donor changed frozen base"
    donor_eval = evaluate(model, value, held, config, reference)
    report = {"interface": interface, "config": config.fingerprint(), "donor": donor,
              "base": base_eval, "donor_evaluation": donor_eval,
              "training_queries": [example.query for example in train_examples],
              "held_out_queries": [example.query for example in held],
              "data_hash": hashlib.sha256(json.dumps([asdict(e) for e in train_examples + held],
                                                     sort_keys=True).encode()).hexdigest()[:16]}
    if progress is not None:
        progress(report)
    if calibrate:
        return report
    projections, error = materialize_delta(model, next(iter(donor_data)))
    fixed_hash = fingerprint(model, fixed=True)
    size = sum(layer.mask_scores.numel() for layer in projections.values())
    data = loader(value, train_examples, config.steps_per_epoch, config)
    steps = config.mask_epochs * config.steps_per_epoch
    budgets = sampled_budgets(size, steps, config.seed + 19)
    curves, rankings = {}, {}
    for stage in ("phased", "one_phase"):
        _set_scores(model, torch.zeros(size, device=config.device))
        model.mask_budget.set_budget(size / 2)
        curves[stage] = train(model, data, config, interface, stage, budgets)
        rankings[stage] = torch.cat([layer.mask_scores.detach().clone() for layer in projections.values()])
        report["curves"] = curves
        if progress is not None:
            progress(report)
    rankings["random"] = torch.rand(size, generator=torch.Generator().manual_seed(config.seed + 31)).to(config.device)
    rankings["gradient_base"] = gradient_ranking(model, data, steps, 0.)
    rankings["gradient_full"] = gradient_ranking(model, data, steps, 1.)
    evaluations = {}
    for name, scores in rankings.items():
        _set_scores(model, scores)
        points = []
        model.eval()
        for count in sorted({round(fraction * size) for fraction in config.fractions}):
            model.mask_budget.set_budget(count, hard=True)
            with model.mask_budget.masked_forward():
                metrics = evaluate(model, value, held, config, reference)
                selected = _ranking_rows(model)
            denominator = base_eval["pirate_prompt_loss"] - donor_eval["pirate_prompt_loss"]
            metrics["loss_recovery"] = ((base_eval["pirate_prompt_loss"] - metrics["pirate_prompt_loss"])
                                         / denominator if abs(denominator) > 1e-8 else None)
            points.append({"budget": count, "fraction": count / size, "selected": selected, "evaluation": metrics})
        evaluations[name] = points
        report["rankings"] = evaluations
        if progress is not None:
            progress(report)
    assert fingerprint(model, fixed=True) == fixed_hash, "score learning changed base or fixed delta"
    report.update(mask_steps=steps, budget_sampler="uniform interior integers", budgets=budgets,
                  donor_endpoint_max_abs_logit_error=error, fixed_base_and_delta=True, curves=curves,
                  rankings=evaluations, eligible_rows=size,
                  coordinates=[{"score_name": item.score_name, "layer": item.layer, "component": item.component,
                                "axis": item.axis, "eligible_range": [0, item.output_width]}
                               for item in mask_coordinates(mask_view(model))])
    return report


def run(config=PirateConfig(), *, interfaces=("pytorch", "lightning"), calibrate=False, progress=None):
    """Collect paired results sequentially, avoiding two pretrained model copies on one GPU."""
    results = {}
    paired = {}
    for interface in interfaces:
        def checkpoint_interface(report):
            results[interface] = report
            if progress is not None:
                progress({"config": asdict(config), "status": "in_progress", "results": results})

        results[interface] = run_interface(config, interface, calibrate=calibrate, progress=checkpoint_interface)
        gc.collect()
        if config.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if len(results) == 2:
        left, right = results["pytorch"], results["lightning"]
        assert left["data_hash"] == right["data_hash"]
        for stage in (["donor"] if calibrate else ["donor", "phased", "one_phase"]):
            a = left[stage] if stage == "donor" else left["curves"][stage]
            b = right[stage] if stage == "donor" else right["curves"][stage]
            assert a["initial_state"] == b["initial_state"]
            assert a["epochs"] == b["epochs"]
            torch.testing.assert_close(torch.tensor([s["loss"] for s in a["steps"]]),
                                       torch.tensor([s["loss"] for s in b["steps"]]), atol=1e-5, rtol=1e-5)
            paired[stage] = {"initial_state_equal": a["initial_state"] == b["initial_state"],
                             "final_state_equal": a["final_state"] == b["final_state"],
                             "phases_equal": a["epochs"] == b["epochs"],
                             "steps": len(a["steps"]),
                             "max_loss_difference": max(abs(x["loss"] - y["loss"])
                                                         for x, y in zip(a["steps"], b["steps"]))}
    return {"config": asdict(config), "status": "complete",
            "claim": "bounded local-corpus tutorial; Qwen3-4B is a model transfer",
            "paired": paired, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen3", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--interfaces", default="pytorch,lightning")
    parser.add_argument("--donor-steps", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    overrides = {name: getattr(args, name) for name in ("donor_steps", "steps_per_epoch", "max_new_tokens")
                 if getattr(args, name) is not None}
    config = PirateConfig.qwen3(**overrides) if args.qwen3 else PirateConfig(**overrides)
    import time

    started = time.monotonic()
    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    def checkpoint(result):
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(args.output)

    result = run(config, interfaces=tuple(args.interfaces.split(",")), calibrate=args.calibrate,
                 progress=checkpoint)
    if config.device.startswith("cuda"):
        torch.cuda.synchronize()
    result["seconds"] = time.monotonic() - started
    result["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20 if config.device.startswith("cuda") else None
    payload = json.dumps(result, indent=2)
    checkpoint(result)
    print(payload)


if __name__ == "__main__":
    main()
