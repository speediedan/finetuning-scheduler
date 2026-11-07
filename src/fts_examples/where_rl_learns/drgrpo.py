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
Dr. GRPO on one device
======================

On-policy sampling, the Dr. GRPO objective, a diagonal Fisher estimate and an AdamW that keeps its state in host memory.

FP32 parameters and gradients with BF16 computation, as in the Fisher paper. Frozen parameters are whatever has
``requires_grad=False``, so a parameter selection works the same whichever tool made it.
"""
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import DynamicCache

#: Tokens of the logit head evaluated together. The head's output is the largest allocation in the backward pass (one
#: vocabulary-sized row per token), so it is checkpointed in chunks of this size and recomputed chunk by chunk.
HEAD_CHUNK_TOKENS = 512


@dataclass
class Sampled:
    """One group of on-policy responses, in the order the prompts were given."""

    prompt_ids: list[list[int]]
    response_ids: list[list[int]]
    logps: list[list[float]]
    stopped: list[bool]
    decode_steps: int = 0
    row_steps: int = 0
    rewards: list[float] = field(default_factory=list)


def _bf16(device: torch.device) -> torch.autocast:
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


#: Decode steps between host synchronizations. Stopped rows leave the batch only at these points, so a row may run this
#: many steps past its stop, and those tokens are discarded. Synchronizing every step costs more than the extra rows.
SYNC_EVERY = 32


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    max_prompt_tokens: int,
    stop_ids: Sequence[int],
    generator: torch.Generator,
) -> Sampled:
    """Samples one response per prompt at temperature 1, with no top-k or top-p restriction.

    Stopped rows are dropped from the batch and its key-value cache, because a base model's longest sample sets the
    number of steps, and a finished row would otherwise hold memory and compute for all of them. Its stopping token
    stays in the response, and a length-truncated response is kept as it is.

    The model samples in evaluation mode: with gradient checkpointing on, ``transformers`` turns the key-value cache off
    in training mode, and every token would then be sampled without its context.
    """
    was_training = model.training
    model.eval()
    try:
        return _sample_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
            stop_ids=stop_ids,
            generator=generator,
        )
    finally:
        model.train(was_training)


def _sample_batch(model, tokenizer, prompts, *, max_new_tokens, max_prompt_tokens, stop_ids, generator) -> Sampled:
    device = next(model.parameters()).device
    enc = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        padding_side="left",
        truncation=True,
        truncation_side="left",
        max_length=max_prompt_tokens,
    )
    input_ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    stop = torch.tensor(list(stop_ids), device=device)
    cache = DynamicCache(config=model.config)
    tokens = torch.zeros((len(prompts), max_new_tokens), dtype=torch.long, device=device)
    logps = torch.zeros((len(prompts), max_new_tokens), device=device)
    stop_at = torch.full((len(prompts),), -1, dtype=torch.long, device=device)
    alive = torch.arange(len(prompts), device=device)
    next_ids, next_positions = input_ids, (mask.cumsum(-1) - 1).clamp(min=0)
    steps = row_steps = 0
    for step in range(max_new_tokens):
        with _bf16(device):
            out = model(
                input_ids=next_ids,
                attention_mask=mask,
                position_ids=next_positions,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        if step == 0 and cache.get_seq_length() != input_ids.size(1):
            raise RuntimeError("the key-value cache is not filling, so each token would be sampled without its context")
        logp = F.log_softmax(out.logits[:, -1].float(), dim=-1)
        sampled = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
        tokens[alive, step] = sampled
        logps[alive, step] = logp.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        first_stop = torch.isin(sampled, stop) & (stop_at[alive] < 0)
        stop_at[alive] = torch.where(first_stop, step, stop_at[alive])
        steps, row_steps = step + 1, row_steps + len(alive)
        if (step + 1) % SYNC_EVERY == 0:
            running = stop_at[alive] < 0
            if not running.any():
                break
            if not running.all():
                keep = running.nonzero().squeeze(-1)
                cache.batch_select_indices(keep)
                alive, mask, sampled, next_positions = alive[keep], mask[keep], sampled[keep], next_positions[keep]
        mask = torch.cat([mask, mask.new_ones(len(alive), 1)], dim=-1)
        next_ids = sampled.unsqueeze(-1)
        next_positions = next_positions[:, -1:] + 1
    return _collect(enc, tokens.cpu(), logps.cpu(), stop_at.cpu(), steps, row_steps)


def _collect(
    enc, tokens: torch.Tensor, logps: torch.Tensor, stop_at: torch.Tensor, steps: int, row_steps: int
) -> Sampled:
    """Cuts each row at its first stopping token, or at the last step for a row that never stopped."""
    lengths = [s + 1 if s >= 0 else steps for s in stop_at.tolist()]
    return Sampled(
        prompt_ids=[row[m.bool()].tolist() for row, m in zip(enc["input_ids"], enc["attention_mask"])],
        response_ids=[tokens[i, :n].tolist() for i, n in enumerate(lengths)],
        logps=[logps[i, :n].tolist() for i, n in enumerate(lengths)],
        stopped=[s >= 0 for s in stop_at.tolist()],
        decode_steps=steps,
        row_steps=row_steps,
    )


def sample_responses(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    max_prompt_tokens: int,
    stop_ids: Sequence[int],
    generator: torch.Generator,
) -> Sampled:
    """Samples one response per prompt, ``batch_size`` rows at a time, and returns them in the order given."""
    merged = Sampled(prompt_ids=[], response_ids=[], logps=[], stopped=[])
    for start in range(0, len(prompts), batch_size):
        part = sample_batch(
            model,
            tokenizer,
            prompts[start : start + batch_size],
            max_new_tokens=max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
            stop_ids=stop_ids,
            generator=generator,
        )
        merged.prompt_ids += part.prompt_ids
        merged.response_ids += part.response_ids
        merged.logps += part.logps
        merged.stopped += part.stopped
        merged.decode_steps += part.decode_steps
        merged.row_steps += part.row_steps
    return merged


def group_advantages(rewards: Sequence[float], group_size: int) -> list[float]:
    """Subtracts each prompt group's mean reward, with no standard-deviation normalization."""
    advantages: list[float] = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start : start + group_size]
        mean = sum(group) / len(group)
        advantages += [r - mean for r in group]
    return advantages


def token_budget_batches(lengths: Sequence[int], token_budget: int) -> Iterator[list[int]]:
    """Groups indices so that each padded batch holds at most ``token_budget`` tokens (one long row may exceed it).

    Grouping changes memory and not the gradient: the objective is a sum over responses with a fixed denominator.
    """
    batch: list[int] = []
    for idx in sorted(range(len(lengths)), key=lambda i: lengths[i]):
        if batch and (len(batch) + 1) * lengths[idx] > token_budget:
            yield batch
            batch = []
        batch.append(idx)
    if batch:
        yield batch


def _response_logps(
    model: torch.nn.Module, prompt_ids: Sequence[Sequence[int]], response_ids: Sequence[Sequence[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns each response token's log probability, flattened, and the row each one belongs to."""
    device = next(model.parameters()).device
    rows = [list(p) + list(r) for p, r in zip(prompt_ids, response_ids)]
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), pad_id, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    in_response = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    for i, (row, prompt) in enumerate(zip(rows, prompt_ids)):
        input_ids[i, : len(row)] = torch.tensor(row, device=device)
        mask[i, : len(row)] = 1
        # the logit at position t predicts the token at t + 1
        in_response[i, len(prompt) - 1 : len(row) - 1] = True
    with _bf16(device):
        hidden = model.model(input_ids=input_ids, attention_mask=mask, use_cache=False).last_hidden_state
    selected = hidden[in_response]
    targets = input_ids[:, 1:][in_response[:, :-1]]
    row_of = in_response.nonzero()[:, 0]

    def head_logp(chunk: torch.Tensor, chunk_targets: torch.Tensor) -> torch.Tensor:
        with _bf16(device):
            logits = model.lm_head(chunk)
        return F.log_softmax(logits.float(), dim=-1).gather(-1, chunk_targets.unsqueeze(-1)).squeeze(-1)

    parts = [
        checkpoint(
            head_logp, selected[i : i + HEAD_CHUNK_TOKENS], targets[i : i + HEAD_CHUNK_TOKENS], use_reentrant=False
        )
        for i in range(0, selected.size(0), HEAD_CHUNK_TOKENS)
    ]
    return torch.cat(parts), row_of


def accumulate_drgrpo_gradient(
    model: torch.nn.Module,
    sampled: Sampled,
    advantages: Sequence[float],
    *,
    horizon: int,
    clip: float,
    token_budget: int,
    pad_id: int,
) -> dict[str, int]:
    """Backpropagates the clipped Dr. GRPO objective over one update's responses.

    The token-summed objective is divided by the response count and the fixed horizon. A response with zero advantage
    contributes nothing, and is skipped rather than run: that includes every response of a prompt whose group got one
    reward, which at a low base accuracy is most of them.
    """
    device = next(model.parameters()).device
    count = len(sampled.response_ids)
    active = [i for i, a in enumerate(advantages) if a != 0.0]
    lengths = [len(sampled.prompt_ids[i]) + len(sampled.response_ids[i]) for i in active]
    batches = 0
    for batch in token_budget_batches(lengths, token_budget):
        rows = [active[i] for i in batch]
        logp, row_of = _response_logps(
            model, [sampled.prompt_ids[i] for i in rows], [sampled.response_ids[i] for i in rows], pad_id
        )
        old = torch.tensor([lp for i in rows for lp in sampled.logps[i]], device=device)
        advantage = torch.tensor([advantages[i] for i in rows], device=device)[row_of]
        ratio = (logp - old).exp()
        objective = torch.min(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).sum()
        (-objective / (count * horizon)).backward()
        batches += 1
    return {
        "responses": count,
        "backward_responses": len(active),
        "backward_batches": batches,
        "active_tokens": sum(lengths),
    }


def diagonal_fisher_sample(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    fisher: dict[str, torch.Tensor],
    *,
    pad_id: int,
) -> None:
    """Adds one sequence score's square to ``fisher``, which holds one FP32 tensor per trainable parameter name.

    The score sums over response tokens and is not divided by length, as in the Fisher paper's estimator.
    """
    logp, _ = _response_logps(model, [prompt_ids], [response_ids], pad_id)
    logp.sum().backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            fisher[name].add_(param.grad.to(fisher[name].device, dtype=fisher[name].dtype).square_())
            param.grad = None


class HostStateAdamW:
    """AdamW with constant learning rate and no weight decay, whose moment estimates live in host memory.

    Full training of a 1.7B model holds 27 GB in FP32 at the optimizer step, which a 24 GB device cannot hold. Moving
    each tensor's moments over only for its own step costs little beside sampling. The arithmetic is
    ``torch.optim.AdamW``'s, and a test holds it to that.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.betas, self.eps = lr, betas, eps
        self.steps = 0
        self.m1 = [torch.zeros_like(p, device="cpu") for p in self.params]
        self.m2 = [torch.zeros_like(p, device="cpu") for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        self.steps += 1
        beta1, beta2 = self.betas
        bias1, bias2 = 1 - beta1**self.steps, 1 - beta2**self.steps
        for param, m1, m2 in zip(self.params, self.m1, self.m2):
            grad = torch.zeros_like(param) if param.grad is None else param.grad
            device_m1 = m1.to(param.device).mul_(beta1).add_(grad, alpha=1 - beta1)
            device_m2 = m2.to(param.device).mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            denominator = device_m2.sqrt().div_(bias2**0.5).add_(self.eps)
            param.addcdiv_(device_m1, denominator, value=-self.lr / bias1)
            m1.copy_(device_m1)
            m2.copy_(device_m2)

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None
