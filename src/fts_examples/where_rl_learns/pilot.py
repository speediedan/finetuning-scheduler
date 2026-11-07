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
r"""
Timed pilot
===========

Measures what the Fisher paper's recipe costs on one device before any arm is scaled: an initial MATH-500 evaluation
(with the paper's 22.65% as a check on the prompt, scorer and sampler), a few Fisher samples, then full-parameter Dr.
GRPO updates, with every stage timed and the peak memory recorded.

.. code-block:: bash

    python -m fts_examples.where_rl_learns.pilot --output pilot.json --updates 20
"""
import argparse
import json
import math
import platform
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from fts_examples.where_rl_learns.drgrpo import (
    accumulate_drgrpo_gradient,
    diagonal_fisher_sample,
    group_advantages,
    HostStateAdamW,
    Sampled,
    sample_responses,
    StaticSampler,
)

MODEL = "Qwen/Qwen3-1.7B-Base"
MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
DATASET = "hkust-nlp/SimpleRL-Zoo-Data"
DATASET_REVISION = "50ec8f09045e785fd20c13a508820e77cccb4211"
DATASET_DIR = "simplelr_abel_level3to5"
HORIZON = 4096
MAX_PROMPT_TOKENS = 1024
LEARNING_RATE = 5e-7
GRAD_CLIP = 1.0
PPO_CLIP = 0.2
#: Prompts and responses per update, and responses per evaluation problem, from the paper's Table 6.
PROMPTS_PER_UPDATE = 4
RESPONSES_PER_PROMPT = 8
EVAL_RESPONSES = 4
#: The Fisher paper's estimate uses this many one-response prompts, and its evaluation this many problems.
PAPER_FISHER_SAMPLES = 800
PAPER_EVAL_PROBLEMS = 500


def load_problems(split: str) -> list[tuple[str, str]]:
    """Returns ``(prompt, gold answer)`` pairs from the pinned SimpleRL-Zoo revision, whose test split is
    MATH-500."""
    rows = load_dataset(
        DATASET, data_files={split: f"{DATASET_DIR}/{split}.parquet"}, revision=DATASET_REVISION, split=split
    )
    return [(row["prompt"][0]["content"], row["reward_model"]["ground_truth"]) for row in rows]


def score_response(text: str, gold: str) -> float:
    """Returns 1.0 when math-verify finds the gold answer in ``text``.

    A response that cannot be parsed scores 0, as a wrong answer would.
    """
    from math_verify import parse, verify

    try:
        return float(verify(parse(f"${gold}$"), parse(text)))
    except Exception:
        return 0.0


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def _peak_gib() -> float:
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def _length_summary(lengths: Sequence[int], stopped: Sequence[bool]) -> dict[str, float]:
    ordered = sorted(lengths)
    return {
        "mean": sum(ordered) / len(ordered),
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
        "stopped_fraction": sum(stopped) / len(stopped),
    }


def run_initial_eval(tokenizer, problems, args, draw) -> dict[str, Any]:
    """Scores the initial model, and keeps a sample of what it wrote."""
    subset = problems[: args.eval_problems]
    prompts = [prompt for prompt, _ in subset for _ in range(EVAL_RESPONSES)]
    golds = [gold for _, gold in subset for _ in range(EVAL_RESPONSES)]
    sampled, sample_s = _timed(lambda: draw(prompts, 300))
    texts = tokenizer.batch_decode(sampled.response_ids, skip_special_tokens=True)
    start = time.perf_counter()
    scores = [score_response(text, gold) for text, gold in zip(texts, golds)]
    cut_scores = [score_response(text.split("\nQuestion:")[0], gold) for text, gold in zip(texts, golds)]
    score_s = (time.perf_counter() - start) / 2
    accuracy = sum(scores) / len(scores)
    return {
        "responses": len(scores),
        "accuracy_percent": 100 * accuracy,
        "standard_error_percent": 100 * math.sqrt(accuracy * (1 - accuracy) / len(scores)),
        "accuracy_percent_cut_at_next_question": 100 * sum(cut_scores) / len(cut_scores),
        "paper_initial_percent": 22.65,
        "sample_s": sample_s,
        "score_s": score_s,
        "tokens": sampled.row_steps,
        "decode_steps": sampled.decode_steps,
        "lengths": _length_summary([len(r) for r in sampled.response_ids], sampled.stopped),
        "peak_gib": _peak_gib(),
        "examples": [
            {"gold": gold, "score": score, "text": text[:1500]}
            for gold, score, text in list(zip(golds, scores, texts))[:: max(1, len(texts) // 12)]
        ],
    }


def run_fisher_probe(model, problems, args, draw, pad_id) -> dict[str, Any]:
    """Times one-response Fisher samples and the global selection of a 1% mask over them."""
    fisher = {name: torch.zeros_like(p, device="cpu") for name, p in model.named_parameters() if p.requires_grad}
    prompts = [prompt for prompt, _ in problems[: args.fisher_samples]]
    sampled, sample_s = _timed(lambda: draw(prompts, 100))

    def accumulate() -> None:
        for prompt_ids, response_ids in zip(sampled.prompt_ids, sampled.response_ids):
            diagonal_fisher_sample(model, prompt_ids, response_ids, fisher, pad_id=pad_id)

    _, backward_s = _timed(accumulate)
    flat = torch.cat([v.flatten() for v in fisher.values()])
    coordinates = flat.numel()
    _, select_s = _timed(lambda: torch.kthvalue(flat, coordinates - coordinates // 100))
    per_sample_s = (sample_s + backward_s) / max(1, len(prompts))
    return {
        "samples": len(prompts),
        "coordinates": coordinates,
        "sample_s": sample_s,
        "backward_s": backward_s,
        "select_1_percent_s": select_s,
        "paper_estimate_hours": per_sample_s * PAPER_FISHER_SAMPLES / 3600,
        "lengths": _length_summary([len(r) for r in sampled.response_ids], sampled.stopped),
        "peak_gib": _peak_gib(),
    }


def run_update(model, tokenizer, optimizer, batch, seed, args, draw, pad_id) -> dict[str, Any]:
    """Runs one fresh batch through one optimizer update, and times each stage."""
    prompts = [prompt for prompt, _ in batch for _ in range(RESPONSES_PER_PROMPT)]
    golds = [gold for _, gold in batch for _ in range(RESPONSES_PER_PROMPT)]
    sampled, sample_s = _timed(lambda: draw(prompts, seed))
    texts = tokenizer.batch_decode(sampled.response_ids, skip_special_tokens=True)
    start = time.perf_counter()
    rewards = [score_response(text, gold) for text, gold in zip(texts, golds)]
    score_s = time.perf_counter() - start
    advantages = group_advantages(rewards, RESPONSES_PER_PROMPT)
    counts, backward_s = _timed(
        lambda: accumulate_drgrpo_gradient(
            model,
            sampled,
            advantages,
            horizon=HORIZON,
            clip=PPO_CLIP,
            token_budget=args.token_budget,
            pad_id=pad_id,
        )
    )
    trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP).item() if trainable else 0.0
    _, step_s = _timed(optimizer.step)
    optimizer.zero_grad()
    refresh = getattr(draw, "refresh", None)
    _, refresh_s = _timed(lambda: refresh(model)) if refresh else (None, 0.0)
    return {
        **counts,
        "sample_s": sample_s,
        "score_s": score_s,
        "backward_s": backward_s,
        "step_s": step_s,
        "refresh_s": refresh_s,
        "mean_reward": sum(rewards) / len(rewards),
        "grad_norm": grad_norm,
        "tokens": sampled.row_steps,
        "decode_steps": sampled.decode_steps,
        "lengths": _length_summary([len(r) for r in sampled.response_ids], sampled.stopped),
        "peak_gib": _peak_gib(),
    }


def make_sampler(model, tokenizer, args, stop_ids) -> Callable[[Sequence[str], int], Sampled]:
    """Returns ``draw(prompts, seed)``: the static-cache sampler by default, or the eager one that samples the
    policy."""
    lengths = dict(max_new_tokens=HORIZON, max_prompt_tokens=MAX_PROMPT_TOKENS, stop_ids=stop_ids)
    if args.sampler == "static":
        return StaticSampler(model, tokenizer, batch_size=args.generation_batch, **lengths)

    def draw(prompts: Sequence[str], seed: int) -> Sampled:
        generator = torch.Generator(next(model.parameters()).device).manual_seed(seed)
        return sample_responses(
            model, tokenizer, prompts, batch_size=args.generation_batch, generator=generator, **lengths
        )

    return draw


def summarize(report: dict[str, Any]) -> dict[str, float]:
    """Extrapolates the measured stages to the paper's budget."""
    updates = report["updates"]
    if not updates:
        return {}
    update_s = sum(u["sample_s"] + u["score_s"] + u["backward_s"] + u["step_s"] + u["refresh_s"] for u in updates)
    update_s /= len(updates)
    evaluate = report["initial_eval"]
    eval_s = (evaluate["sample_s"] + evaluate["score_s"]) * PAPER_EVAL_PROBLEMS * EVAL_RESPONSES / evaluate["responses"]
    return {
        "updates_measured": len(updates),
        "mean_update_s": update_s,
        "mean_sample_s": sum(u["sample_s"] for u in updates) / len(updates),
        "mean_backward_s": sum(u["backward_s"] for u in updates) / len(updates),
        "mean_step_s": sum(u["step_s"] for u in updates) / len(updates),
        "hours_per_1000_updates": update_s * 1000 / 3600,
        "paper_eval_hours": eval_s / 3600,
        "paper_fisher_hours": report["fisher"]["paper_estimate_hours"],
        "peak_gib": max(u["peak_gib"] for u in updates),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    report["summary"] = summarize(report)
    path.write_text(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=3000, help="stop starting updates after this long")
    parser.add_argument("--generation-batch", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=16384)
    parser.add_argument("--eval-problems", type=int, default=125)
    parser.add_argument("--fisher-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampler", choices=("static", "eager"), default="static")
    args = parser.parse_args()

    begin = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|endoftext|>", "<|im_end|>")]
    pad_id = tokenizer.pad_token_id
    train, test = load_problems("train"), load_problems("test")

    report: dict[str, Any] = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "train_problems": len(train),
        "test_problems": len(test),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "host": platform.node(),
        "torch": torch.__version__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameter_gib": sum(p.numel() for p in model.parameters()) * 4 / 2**30,
        "load_s": time.perf_counter() - begin,
        "updates": [],
    }
    model.train()
    draw = make_sampler(model, tokenizer, args, stop_ids)
    report["initial_eval"] = run_initial_eval(tokenizer, test, args, draw)
    write_report(args.output, report)
    report["fisher"] = run_fisher_probe(model, train, args, draw, pad_id)
    write_report(args.output, report)

    optimizer = HostStateAdamW(model.parameters(), lr=LEARNING_RATE)
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    for index in range(args.updates):
        if time.perf_counter() - begin > args.max_seconds:
            report["stopped_early"] = f"max_seconds reached after {index} updates"
            break
        batch = [train[order[(index * PROMPTS_PER_UPDATE + i) % len(order)]] for i in range(PROMPTS_PER_UPDATE)]
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        update = run_update(model, tokenizer, optimizer, batch, args.seed * 1000 + index, args, draw, pad_id)
        report["updates"].append(update)
        write_report(args.output, report)
        print(json.dumps({k: v for k, v in update.items() if k != "lengths"}), flush=True)
    report["total_s"] = time.perf_counter() - begin
    write_report(args.output, report)
    print(json.dumps(report["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
