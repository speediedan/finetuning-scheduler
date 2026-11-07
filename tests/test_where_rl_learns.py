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
import pytest
import torch

pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from fts_examples.where_rl_learns.drgrpo import (  # noqa: E402
    _response_logps,
    accumulate_drgrpo_gradient,
    diagonal_fisher_sample,
    group_advantages,
    HostStateAdamW,
    sample_batch,
    token_budget_batches,
)

VOCAB = 40
STOP = 3
PAD = 0


class _Tokenizer:
    """Enough of a tokenizer for ``sample_batch``: prompts are whitespace-separated token ids."""

    def __call__(self, prompts, **kwargs):
        rows = [[int(t) for t in p.split()] for p in prompts]
        width = max(len(r) for r in rows)
        ids = torch.tensor([[PAD] * (width - len(r)) + r for r in rows])
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])
        return {"input_ids": ids, "attention_mask": mask}


@pytest.fixture
def model():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        use_sliding_window=False,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config).float().train()


def _sample(model, prompts, max_new_tokens=12, seed=0):
    return sample_batch(
        model,
        _Tokenizer(),
        prompts,
        max_new_tokens=max_new_tokens,
        max_prompt_tokens=64,
        stop_ids=[STOP],
        generator=torch.Generator().manual_seed(seed),
    )


def test_sampler_logps_match_a_fresh_forward_pass(model):
    """Left padding, position ids and dropping stopped rows from the cache must not change what was sampled
    from."""
    sampled = _sample(model, ["5 6 7", "8", "9 10 11 12 13"], max_new_tokens=16)
    assert any(sampled.stopped) or sampled.row_steps == 48
    logp, row_of = _response_logps(model, sampled.prompt_ids, sampled.response_ids, PAD)
    expected = torch.tensor([lp for row in sampled.logps for lp in row])
    assert row_of.tolist() == [i for i, row in enumerate(sampled.logps) for _ in row]
    assert torch.allclose(logp.detach(), expected, atol=5e-2)


def test_sampler_keeps_its_cache_under_gradient_checkpointing(model):
    """Gradient checkpointing turns the cache off in training mode, which once sampled every token without
    context."""
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    sampled = _sample(model, ["5 6 7", "8", "9 10 11 12 13"], max_new_tokens=40)
    assert model.training
    logp, _ = _response_logps(model, sampled.prompt_ids, sampled.response_ids, PAD)
    expected = torch.tensor([lp for row in sampled.logps for lp in row])
    assert torch.allclose(logp.detach(), expected, atol=5e-2)


def test_sampler_drops_rows_and_keeps_the_stop_token(model):
    # bias the head so that the stop token dominates, then only the first sampled tokens differ
    with torch.no_grad():
        model.lm_head.weight[STOP] += 5.0
    sampled = _sample(model, ["5 6", "7", "8 9 10", "11"], max_new_tokens=32)
    for response, stopped, logps in zip(sampled.response_ids, sampled.stopped, sampled.logps):
        assert len(response) == len(logps)
        if stopped:
            assert response[-1] == STOP and STOP not in response[:-1]
    # rows leave the batch only at synchronization points, so they may run past their stop
    assert sampled.row_steps >= sum(len(r) for r in sampled.response_ids)


def test_group_advantages_center_each_group():
    assert group_advantages([1.0, 0.0, 1.0, 1.0], 2) == [0.5, -0.5, 0.0, 0.0]


def test_token_budget_batches_cover_everything_once():
    lengths = [50, 10, 10, 30, 200]
    batches = list(token_budget_batches(lengths, 60))
    assert sorted(i for batch in batches for i in batch) == list(range(5))
    assert all(len(batch) * max(lengths[i] for i in batch) <= 60 or len(batch) == 1 for batch in batches)


def test_objective_moves_probability_toward_rewarded_responses(model):
    sampled = _sample(model, ["5 6 7", "5 6 7"], max_new_tokens=8)
    advantages = [1.0, -1.0]
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3)
    before, _ = _response_logps(model, sampled.prompt_ids[:1], sampled.response_ids[:1], PAD)
    other_before, _ = _response_logps(model, sampled.prompt_ids[1:], sampled.response_ids[1:], PAD)
    counts = accumulate_drgrpo_gradient(
        model, sampled, advantages, horizon=8, clip=0.2, token_budget=1000, pad_id=PAD
    )
    optimizer.step()
    after, _ = _response_logps(model, sampled.prompt_ids[:1], sampled.response_ids[:1], PAD)
    other_after, _ = _response_logps(model, sampled.prompt_ids[1:], sampled.response_ids[1:], PAD)
    assert counts["backward_responses"] == 2
    assert after.sum() > before.sum()
    assert other_after.sum() < other_before.sum()


def test_zero_advantage_responses_are_skipped(model):
    sampled = _sample(model, ["5 6 7", "5 6 7"], max_new_tokens=4)
    counts = accumulate_drgrpo_gradient(model, sampled, [0.0, 0.0], horizon=4, clip=0.2, token_budget=1000, pad_id=PAD)
    assert counts["backward_responses"] == 0
    assert all(p.grad is None for p in model.parameters())


def test_gradient_does_not_depend_on_token_budget(model):
    sampled = _sample(model, ["5 6 7", "8 9", "10"], max_new_tokens=8)
    advantages = [1.0, -0.5, -0.5]
    grads = []
    for budget in (1, 1000):
        model.zero_grad()
        accumulate_drgrpo_gradient(model, sampled, advantages, horizon=8, clip=0.2, token_budget=budget, pad_id=PAD)
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    # BF16 computation rounds differently once a row is padded, so the two agree to its precision and no further
    assert (grads[0] - grads[1]).norm() < 0.02 * grads[1].norm()


def test_frozen_parameters_hold_and_get_no_fisher(model):
    frozen = [p for n, p in model.named_parameters() if "layers.0." in n]
    for param in frozen:
        param.requires_grad_(False)
    snapshot = [p.clone() for p in frozen]
    sampled = _sample(model, ["5 6 7", "8 9"], max_new_tokens=6)
    optimizer = HostStateAdamW(model.parameters(), lr=1e-2)
    accumulate_drgrpo_gradient(model, sampled, [1.0, -1.0], horizon=6, clip=0.2, token_budget=100, pad_id=PAD)
    optimizer.step()
    assert all(torch.equal(p, s) for p, s in zip(frozen, snapshot))
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    diagonal_fisher_sample(model, sampled.prompt_ids[0], sampled.response_ids[0], fisher, pad_id=PAD)
    assert not any("layers.0." in n for n in fisher)
    assert sum(v.sum() for v in fisher.values()) > 0


def test_host_state_adamw_matches_torch():
    torch.manual_seed(0)
    ours = [torch.nn.Parameter(torch.randn(5)), torch.nn.Parameter(torch.randn(2, 3))]
    theirs = [torch.nn.Parameter(p.detach().clone()) for p in ours]
    host = HostStateAdamW(ours, lr=1e-2)
    reference = torch.optim.AdamW(theirs, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    for _ in range(25):
        grads = [torch.randn_like(p) for p in ours]
        for param, other, grad in zip(ours, theirs, grads):
            param.grad, other.grad = grad.clone(), grad.clone()
        host.step()
        reference.step()
    for param, other in zip(ours, theirs):
        assert torch.allclose(param, other, atol=1e-6)
