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
Coordinate masks
================

The Fisher paper's sparse arms train a fixed 1% of the model's coordinates: the 1% with the largest diagonal Fisher
information, against a random 1% as the control. Both masks come from one selection, the global top fraction of a score
per coordinate, so the two arms differ only in their scores.

A mask is stored as each tensor's flat indices (``int32``), about 70 MB for 1% of 1.72B coordinates. During training
:class:`MaskedGradients` zeroes every gradient outside the mask as it accumulates, which is before clipping. With
:class:`~fts_examples.where_rl_learns.drgrpo.HostStateAdamW` a coordinate whose gradient is always zero keeps zero
moments, so its update is exactly zero and the coordinate holds its initial value.

.. code-block:: bash

    python -m fts_examples.where_rl_learns.masks --kind fisher --output fisher1.pt --sampler vllm \
        --vllm-python /path/to/vllm/bin/python
"""
import argparse
import hashlib
import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from fts_examples.where_rl_learns.drgrpo import diagonal_fisher_sample

#: The Fisher paper's estimate uses this many prompts, one response each, and keeps this fraction of coordinates.
FISHER_SAMPLES = 800
FRACTION = 0.01
#: The Fisher estimate's sampling seed, apart from the evaluation's and training's.
FISHER_SEED = 100


def top_fraction(scores: Mapping[str, torch.Tensor], fraction: float = FRACTION) -> dict[str, torch.Tensor]:
    """Each tensor's flat indices among the global top ``fraction`` of ``scores``.

    Ties at the threshold are left out, so at most ``fraction`` of the coordinates are kept.
    """
    flat = torch.cat([s.detach().flatten().float().cpu() for s in scores.values()])
    keep = int(flat.numel() * fraction)
    threshold = torch.kthvalue(flat, flat.numel() - keep).values
    del flat
    return {name: (s.flatten() > threshold).nonzero().squeeze(1).to(torch.int32) for name, s in scores.items()}


def random_scores(named_params: Iterator[tuple[str, torch.nn.Parameter]], seed: int) -> dict[str, torch.Tensor]:
    """A uniform random score per coordinate, so that :func:`top_fraction` picks a uniform random subset."""
    generator = torch.Generator().manual_seed(seed)
    return {name: torch.rand(p.shape, generator=generator) for name, p in named_params}


def estimate_fisher(model, prompts: Sequence[str], draw: Callable, *, pad_id: int, seed: int = FISHER_SEED):
    """The diagonal Fisher information of the model's trainable parameters over one sampled response per prompt, in
    host memory."""
    fisher = {name: torch.zeros_like(p, device="cpu") for name, p in model.named_parameters() if p.requires_grad}
    sampled = draw(prompts, seed)
    for prompt_ids, response_ids in zip(sampled.prompt_ids, sampled.response_ids):
        diagonal_fisher_sample(model, prompt_ids, response_ids, fisher, pad_id=pad_id)
    return fisher


def mask_digest(indices: Mapping[str, torch.Tensor]) -> str:
    """A SHA-256 over the mask's names and indices, to record which mask a run used."""
    digest = hashlib.sha256()
    for name in sorted(indices):
        digest.update(name.encode())
        digest.update(indices[name].numpy().tobytes())
    return digest.hexdigest()


def save_mask(path: Path, indices: Mapping[str, torch.Tensor], meta: Mapping[str, Any]) -> str:
    """Writes the mask and what made it; returns its digest."""
    digest = mask_digest(indices)
    torch.save({"indices": dict(indices), "meta": {**meta, "digest": digest}}, path)
    return digest


def load_mask(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if mask_digest(payload["indices"]) != payload["meta"]["digest"]:
        raise ValueError(f"{path}'s indices do not match its recorded digest.")
    return payload["indices"], payload["meta"]


class MaskedGradients:
    """Zeroes each gradient outside a mask as it accumulates, through post-accumulate-grad hooks.

    Accumulation is linear and the mask fixed, so masking each accumulation masks the sum, and everything after backward
    (clipping, the optimizer) sees only the mask's coordinates. A trainable parameter the mask does not name gets no
    gradient at all.
    """

    def __init__(self, model: torch.nn.Module, indices: Mapping[str, torch.Tensor]) -> None:
        self.indices = indices
        self._on_device: dict[str, torch.Tensor] = {}
        self.handles = [
            param.register_post_accumulate_grad_hook(self._hook(name))
            for name, param in model.named_parameters()
            if param.requires_grad
        ]

    def _hook(self, name: str) -> Callable[[torch.Tensor], None]:
        def mask(param: torch.Tensor) -> None:
            grad = param.grad
            assert grad is not None
            index = self._on_device.get(name)
            if index is None or index.device != grad.device:
                index = self._on_device[name] = self.indices.get(name, torch.empty(0, dtype=torch.int32)).to(
                    grad.device, torch.long
                )
            flat = grad.view(-1)
            kept = flat[index]
            flat.zero_()
            flat[index] = kept

        return mask

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def main() -> None:
    from fts_examples.where_rl_learns.pilot import load_problems, make_sampler, MODEL, MODEL_REVISION

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=("fisher", "random"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=FISHER_SAMPLES)
    parser.add_argument("--seed", type=int, default=0, help="the random mask's seed")
    parser.add_argument("--generation-batch", type=int, default=16)
    parser.add_argument("--sampler", choices=("eager", "static", "vllm"), default="vllm")
    parser.add_argument("--vllm-python", help="an interpreter with vLLM installed, for --sampler vllm")
    parser.add_argument("--vllm-dir", type=Path, default=Path("/dev/shm/where_rl_learns"), help="weight snapshots")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    begin = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.float32)
    meta: dict[str, Any] = {"kind": args.kind, "fraction": FRACTION, "model": MODEL, "model_revision": MODEL_REVISION}
    if args.kind == "random":
        scores = random_scores(model.named_parameters(), args.seed)
        meta["seed"] = args.seed
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train().to(device)
        stop_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|endoftext|>", "<|im_end|>")]
        draw = make_sampler(model, tokenizer, args, stop_ids)
        # the first prompts of the training split, as the paper's estimate takes a fixed sample before training
        prompts = [prompt for prompt, _ in load_problems("train")[: args.samples]]
        scores = estimate_fisher(model, prompts, draw, pad_id=tokenizer.pad_token_id)
        getattr(draw, "close", lambda: None)()
        meta.update(samples=len(prompts), sampling_seed=FISHER_SEED, estimate_s=time.perf_counter() - begin)
    start = time.perf_counter()
    indices = top_fraction(scores)
    meta["select_s"] = time.perf_counter() - start
    meta["kept"] = sum(i.numel() for i in indices.values())
    meta["coordinates"] = sum(s.numel() for s in scores.values())
    if args.kind == "fisher":
        torch.save(scores, args.output.with_suffix(".scores.pt"))
    digest = save_mask(args.output, indices, meta)
    print(json.dumps({**meta, "digest": digest}, indent=1), flush=True)


if __name__ == "__main__":
    main()
