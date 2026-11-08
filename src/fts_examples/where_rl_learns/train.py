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
Arm runner
==========

Trains one arm of the where-does-RL-learn contrast with the Fisher paper's recipe (Dr. GRPO on Qwen3-1.7B-Base over
SimpleRL-Zoo's levels 3 to 5), choosing what trains with a one-phase
:class:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler` schedule written in venngram addresses:
``layers:10`` is the eleventh decoder block, whatever the architecture calls its tensors.

An epoch is ``--updates-per-epoch`` updates. At each epoch's end the runner writes a checkpoint (the trained tensors,
the optimizer's moments and the tuning scheduler's
:meth:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler.resume_state`), so a run stopped by ``--hours`` or by
a failure resumes from the last one when started again with the same arguments. Each update's record goes to
``updates.jsonl``. MATH-500 is evaluated before training and every ``--eval-every`` updates (4 responses per problem,
seeds 300 to 303, mean response accuracy), to ``evals.jsonl`` with every response's score, so arms compare problem by
problem.

The mask arms (``fisher1`` and ``random1``) schedule every tensor and train only the coordinates of a mask written by
:mod:`.masks` (``--mask``), whose digest is part of the run's identity. At every epoch's end they check that a few
tensors' coordinates outside the mask still hold their pretrained values.

.. code-block:: bash

    python -m fts_examples.where_rl_learns.train --arm layer10 --output-dir runs/layer10 --sampler vllm \
        --vllm-python /path/to/vllm/bin/python
"""
import argparse
import json
import math
import os
import platform
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from finetuning_scheduler import TuningScheduler
from fts_examples.where_rl_learns.drgrpo import HostStateAdamW
from fts_examples.where_rl_learns.masks import load_mask, MaskedGradients
from fts_examples.where_rl_learns.pilot import (
    _timed,
    DATASET_REVISION,
    LEARNING_RATE,
    load_problems,
    make_sampler,
    MODEL,
    MODEL_REVISION,
    PROMPTS_PER_UPDATE,
    reload_check,
    run_update,
    score_response,
)

#: Each arm's schedule: the addresses its one phase trains. ``role:base`` is every tensor of the model.
ARMS = {
    "full": ["role:base"],
    "layer10": ["layers:10"],
    "layer24": ["layers:24"],
    "fisher1": ["role:base"],
    "random1": ["role:base"],
}
#: The arms that train only a mask's coordinates, and the kind of mask each takes.
MASK_ARMS = {"fisher1": "fisher", "random1": "random"}
#: The evaluation's sampling seeds, one response per problem for each.
EVAL_SEEDS = (300, 301, 302, 303)
CHECKPOINT = "checkpoint.pt"


def expected_trainable(arm: str, names: Sequence[str]) -> list[str]:
    """The parameter names an arm should train, by name, to check what its addresses selected."""
    if arm == "full" or arm in MASK_ARMS:
        return list(names)
    block = int(arm.removeprefix("layer"))
    return [n for n in names if n.startswith(f"model.layers.{block}.")]


def train_seed(seed: int, update: int) -> int:
    """The sampling seed of one update, apart from the evaluation's seeds for any run seed."""
    return 1_000_000 * (seed + 1) + update


def evaluate(tokenizer, problems, draw, score=score_response, seeds: Sequence[int] = EVAL_SEEDS) -> dict[str, Any]:
    """Samples one response per problem for each seed and scores them; the standard error is over problems."""
    prompts = [prompt for prompt, _ in problems]
    golds = [gold for _, gold in problems]
    scores, sample_s, lengths, stopped = [], 0.0, [], []
    for seed in seeds:
        sampled, seconds = _timed(lambda: draw(prompts, seed))
        sample_s += seconds
        texts = tokenizer.batch_decode(sampled.response_ids, skip_special_tokens=True)
        scores.append([int(score(text, gold)) for text, gold in zip(texts, golds)])
        lengths += [len(r) for r in sampled.response_ids]
        stopped += sampled.stopped
    per_problem = [sum(column) / len(seeds) for column in zip(*scores)]
    mean = sum(per_problem) / len(per_problem)
    variance = sum((x - mean) ** 2 for x in per_problem) / max(1, len(per_problem) - 1)
    return {
        "accuracy_percent": 100 * mean,
        "standard_error_percent": 100 * math.sqrt(variance / len(per_problem)),
        "per_seed_percent": [100 * sum(row) / len(row) for row in scores],
        "seeds": list(seeds),
        "sample_s": sample_s,
        "mean_length": sum(lengths) / len(lengths),
        "stopped_fraction": sum(stopped) / len(stopped),
        "scores": scores,
    }


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def _keep_before(path: Path, update: int) -> None:
    """Drops the records of updates a resume will run again: those at or after ``update``."""
    if path.exists():
        lines = [line for line in path.read_text().splitlines() if json.loads(line)["update"] < update]
        path.write_text("".join(line + "\n" for line in lines))


def save_checkpoint(path: Path, model, fts: TuningScheduler, run: dict[str, Any]) -> float:
    """Writes the trained tensors, the optimizer and the tuning scheduler's resume state, replacing ``path`` only
    once complete; returns the seconds it took."""
    start = time.perf_counter()
    payload = {
        "run": run,
        "model": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": fts.optimizer.state_dict(),
        "tuning_scheduler": fts.resume_state(),
    }
    partial = path.with_suffix(".partial")
    torch.save(payload, partial)
    os.replace(partial, path)
    return time.perf_counter() - start


def load_checkpoint(path: Path, model, fts: TuningScheduler, run: dict[str, Any]) -> None:
    """Resumes from :func:`save_checkpoint`.

    Call it while the model is in host memory, where the optimizer's moments stay.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["run"] != run:
        raise ValueError(f"{path} is from a different run: {payload['run']} against {run}.")
    fts.load_resume_state(payload["tuning_scheduler"])
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in payload["model"].items():
            params[name].copy_(value)
    fts.optimizer.load_state_dict(payload["optimizer"])


def _outside_mask_drift(model, indices, reference: dict[str, torch.Tensor]) -> float:
    """The largest change, since ``reference`` was taken, of a reference tensor's coordinates outside the mask."""
    params = dict(model.named_parameters())
    drift = 0.0
    for name, before in reference.items():
        outside = torch.ones(before.numel(), dtype=torch.bool)
        outside[indices[name].long()] = False
        now = params[name].detach().cpu().flatten()
        if outside.any():
            drift = max(drift, (now - before.flatten())[outside].abs().max().item())
    return drift


def run_arm(
    model,
    tokenizer,
    train: Sequence[tuple[str, str]],
    test: Sequence[tuple[str, str]],
    make_draw: Callable[[bool], Callable],
    args: argparse.Namespace,
    *,
    pad_id: int,
    device: torch.device,
    view=None,
    score=score_response,
) -> dict[str, Any]:
    """Trains ``args.arm`` from the start or from the checkpoint in ``args.output_dir``, and returns the run's
    status.

    Args:
        model: The policy, in host memory.
        make_draw: Called once the policy is on ``device``, with whether it still holds the checkpoint's weights;
            returns ``draw(prompts, seed)``.
        view: The model's tunable view; :func:`venngram.view` by default.
    """
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.eval_every % args.updates_per_epoch or args.updates % args.updates_per_epoch:
        raise ValueError("--eval-every and --updates must be multiples of --updates-per-epoch.")
    if view is None:
        import venngram

        from fts_examples.tunable_views import use_seed_registry

        use_seed_registry()
        view = venngram.view(model)
    run = {"arm": args.arm, "seed": args.seed, "updates_per_epoch": args.updates_per_epoch}
    masked = None
    if args.arm in MASK_ARMS:
        indices, mask_meta = load_mask(args.mask)
        if mask_meta["kind"] != MASK_ARMS[args.arm]:
            raise ValueError(f"The {args.arm} arm takes a {MASK_ARMS[args.arm]} mask, not {mask_meta['kind']}.")
        run["mask_digest"] = mask_meta["digest"]
    fts = TuningScheduler(
        model,
        {0: {"params": ARMS[args.arm]}},
        view=view,
        optimizer=lambda groups: HostStateAdamW(groups, lr=LEARNING_RATE),
        metadata=run,
    )
    fts.setup()
    names = [n for n, _ in model.named_parameters()]
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable != expected_trainable(args.arm, names):
        raise RuntimeError(f"The {args.arm} arm's addresses selected {len(trainable)} tensors, not its own.")
    if args.arm in MASK_ARMS:
        masked = MaskedGradients(model, indices)
        # pretrained values, taken before any resume, so the check spans every run of the arm
        reference = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in indices}
        reference = {n: reference[n] for n in (trainable[1], trainable[len(trainable) // 2], trainable[-1])}
    checkpoint = out / CHECKPOINT
    resumed = checkpoint.exists()
    if resumed:
        load_checkpoint(checkpoint, model, fts, run)
    start_update = fts.global_step
    # a resumed run keeps the evaluation its checkpoint's epoch ended with; a fresh one keeps nothing
    _keep_before(out / "updates.jsonl", start_update)
    _keep_before(out / "evals.jsonl", start_update + 1 if resumed else 0)
    model.to(device)
    draw = make_draw(not resumed)
    begin = time.perf_counter()
    status: dict[str, Any] = {**run, "resumed_at": start_update if resumed else None}
    if not resumed:
        info = {
            **run,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "dataset_revision": DATASET_REVISION,
            "addresses": ARMS[args.arm],
            "trainable_tensors": len(trainable),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "host": platform.node(),
            "torch": torch.__version__,
            "config_hash": fts.config_hash,
            "mask": mask_meta if masked else None,
            "args": {k: str(v) for k, v in vars(args).items()},
        }
        if hasattr(draw, "refresh"):
            info["reload_check"] = reload_check(draw, [prompt for prompt, _ in test[:16]])
            if not (info["reload_check"]["passed"] or args.allow_reload_mismatch):
                (out / "run.json").write_text(json.dumps(info, indent=1))
                raise RuntimeError(f"A sleep and reload changed the sampled tokens: {info['reload_check']}.")
        (out / "run.json").write_text(json.dumps(info, indent=1))
        _append(out / "evals.jsonl", {"update": 0, **evaluate(tokenizer, test, draw, score)})

    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    epoch_s: list[float] = []
    for _ in fts.epochs(max_epochs=args.updates // args.updates_per_epoch):
        if epoch_s and time.perf_counter() - begin + max(epoch_s) > args.hours * 3600:
            status["stopped"] = "time limit"
            break
        epoch_start = time.perf_counter()
        rewards = []
        for _ in range(args.updates_per_epoch):
            update = fts.global_step
            batch = [train[order[(update * PROMPTS_PER_UPDATE + i) % len(order)]] for i in range(PROMPTS_PER_UPDATE)]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            record = run_update(model, tokenizer, fts.optimizer, batch, train_seed(args.seed, update), args, draw,
                                pad_id, score)
            rewards.append(record["mean_reward"])
            _append(out / "updates.jsonl", {"update": update, **record})
        metrics = {"mean_reward": sum(rewards) / len(rewards)}
        if fts.global_step % args.eval_every == 0:
            result = evaluate(tokenizer, test, draw, score)
            _append(out / "evals.jsonl", {"update": fts.global_step, **result})
            metrics["math500_percent"] = result["accuracy_percent"]
        if masked:
            metrics["outside_mask_drift"] = _outside_mask_drift(model, indices, reference)
            if metrics["outside_mask_drift"]:
                raise RuntimeError(f"Coordinates outside the mask moved: {metrics['outside_mask_drift']}.")
        fts.epoch_end(metrics)
        save_s = save_checkpoint(checkpoint, model, fts, run)
        epoch_s.append(time.perf_counter() - epoch_start + save_s)
        print(json.dumps({"update": fts.global_step, **metrics, "epoch_s": epoch_s[-1]}), flush=True)
    status.update(updates=fts.global_step, complete=fts.global_step >= args.updates)
    status["seconds"] = time.perf_counter() - begin
    getattr(draw, "close", lambda: None)()
    if masked:
        masked.remove()
    (out / "status.json").write_text(json.dumps(status, indent=1))
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--updates-per-epoch", type=int, default=50, help="updates between checkpoints")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--hours", type=float, default=11.5, help="start no epoch that would end after this long")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generation-batch", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=16384)
    parser.add_argument("--sampler", choices=("eager", "static", "vllm"), default="vllm")
    parser.add_argument("--vllm-python", help="an interpreter with vLLM installed, for --sampler vllm")
    parser.add_argument("--vllm-dir", type=Path, default=Path("/dev/shm/where_rl_learns"), help="weight snapshots")
    parser.add_argument("--mask", type=Path, help="the mask a fisher1 or random1 arm trains, from .masks")
    parser.add_argument("--allow-reload-mismatch", action="store_true", help="train even if the reload check fails")
    args = parser.parse_args()
    args.output = args.output_dir / "run.json"  # the sampler's log goes beside it

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|endoftext|>", "<|im_end|>")]
    status = run_arm(
        model,
        tokenizer,
        load_problems("train"),
        load_problems("test"),
        lambda fresh: make_sampler(model, tokenizer, args, stop_ids, policy_is_checkpoint=fresh),
        args,
        pad_id=tokenizer.pad_token_id,
        device=device,
    )
    print(json.dumps(status, indent=1), flush=True)


if __name__ == "__main__":
    main()
