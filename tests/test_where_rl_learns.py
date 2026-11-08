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
    StaticSampler,
    token_budget_batches,
)

VOCAB = 40
STOP = 3
PAD = 0


class _Tokenizer:
    """Enough of a tokenizer for ``sample_batch``: prompts are whitespace-separated token ids."""

    def __call__(self, prompts, **kwargs):
        rows = [[int(t) for t in p.split()] for p in prompts]
        if "return_tensors" not in kwargs:
            return {"input_ids": rows}
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


def test_static_sampler_cuts_at_the_stop_and_follows_the_policy(model):
    tokenizer = _Tokenizer()
    tokenizer.pad_token_id = PAD
    with torch.no_grad():
        model.lm_head.weight[STOP] += 3.0
    sampler = StaticSampler(
        model, tokenizer, batch_size=2, max_new_tokens=12, max_prompt_tokens=64, stop_ids=[STOP], offload_policy=True
    )
    assert sampler.model.dtype == torch.bfloat16 and not sampler.model.training
    sampled = sampler(["5 6 7", "8", "9 10"], seed=0)
    assert sampled.prompt_ids == [[5, 6, 7], [8], [9, 10]]
    for response, stopped in zip(sampled.response_ids, sampled.stopped):
        assert 0 < len(response) <= 12
        assert (response[-1] == STOP) == stopped and STOP not in response[:-1]
    with torch.no_grad():
        model.lm_head.weight.add_(1.0)
    sampler.refresh(model)
    assert torch.equal(sampler.model.lm_head.weight, model.lm_head.weight.to(torch.bfloat16))
    # without the sampler's log probabilities the ratio is taken against the policy's own, detached
    counts = accumulate_drgrpo_gradient(
        model, sampled, [1.0, -0.5, -0.5], horizon=12, clip=0.2, token_budget=100, pad_id=PAD
    )
    assert counts["backward_responses"] == 3
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_vllm_sampler_hands_the_device_over_and_back(model, tmp_path):
    import os
    import sys
    from pathlib import Path

    from safetensors.torch import load_file

    from fts_examples.where_rl_learns.vllm_sampler import VLLMSampler

    ops = tmp_path / "ops.txt"
    sampler = VLLMSampler(
        model,
        _Tokenizer(),
        python=sys.executable,
        model="fake",
        revision=None,
        max_new_tokens=8,
        max_prompt_tokens=64,
        stop_ids=[STOP],
        snapshot_dir=tmp_path / "snapshot",
        log_path=tmp_path / "worker.log",
        env={**os.environ, "FAKE_VLLM_OPS": str(ops)},
        worker=Path(__file__).parent / "helpers" / "fake_vllm_worker.py",
    )
    snapshot = load_file(str(tmp_path / "snapshot" / "model.safetensors"))
    # the tied head is written once, and in BF16
    assert "lm_head.weight" not in snapshot and snapshot["model.embed_tokens.weight"].dtype == torch.bfloat16
    first = sampler(["5 6 7", "8"], seed=1)
    second = sampler(["9"], seed=2)
    sampler.close()
    assert first.prompt_ids == [[5, 6, 7], [8]] and all(r[-1] == STOP for r in first.response_ids)
    assert second.prompt_ids == [[9]] and second.stopped == [True]
    assert ops.read_text().split() == ["generate", "sleep", "wake", "reload", "generate", "sleep", "exit"]
    assert next(model.parameters()).device.type == "cpu"
    assert "a log line" in (tmp_path / "worker.log").read_text()


def test_host_state_adamw_is_an_optimizer_whose_moments_stay_in_host_memory():
    params = [torch.nn.Parameter(torch.randn(4)), torch.nn.Parameter(torch.randn(3))]
    params[1].requires_grad_(False)
    optimizer = HostStateAdamW([{"params": params}], lr=1e-2)
    steps = []
    optimizer.register_step_post_hook(lambda *_: steps.append(1))
    params[0].grad = torch.ones(4)
    optimizer.step()
    assert steps == [1] and params[1] not in optimizer.state
    restored = HostStateAdamW([{"params": params}], lr=1e-2)
    restored.load_state_dict(optimizer.state_dict())
    state = restored.state[params[0]]
    assert state["step"] == 1 and state["exp_avg"].device.type == "cpu"
    assert torch.equal(state["exp_avg_sq"], optimizer.state[params[0]]["exp_avg_sq"])


def _fake_sampler(model, tmp_path, **kwargs):
    import os
    import sys
    from pathlib import Path

    from fts_examples.where_rl_learns.vllm_sampler import VLLMSampler

    return VLLMSampler(
        model,
        _Tokenizer(),
        python=sys.executable,
        model="fake",
        revision=None,
        max_new_tokens=8,
        max_prompt_tokens=64,
        stop_ids=[STOP],
        snapshot_dir=tmp_path / "snapshot",
        log_path=tmp_path / "worker.log",
        env={**os.environ, "FAKE_VLLM_OPS": str(tmp_path / "ops.txt")},
        worker=Path(__file__).parent / "helpers" / "fake_vllm_worker.py",
        **kwargs,
    )


def test_vllm_sampler_reloads_first_when_the_policy_is_not_the_checkpoint(model, tmp_path):
    """A resumed policy's weights are not the ones the engine loaded at start, so its first draw reloads them."""
    sampler = _fake_sampler(model, tmp_path, policy_is_checkpoint=False)
    sampler(["5"], seed=1)
    sampler.refresh(model)
    sampler(["6"], seed=2)
    sampler.close()
    expected = ["reload", "generate", "sleep", "wake", "reload", "generate", "sleep", "exit"]
    assert (tmp_path / "ops.txt").read_text().split() == expected


def _fixed_draw(prompts, seed):
    from fts_examples.where_rl_learns.drgrpo import Sampled

    generator = torch.Generator().manual_seed(seed)
    responses = []
    for i, _ in enumerate(prompts):
        length = int(torch.randint(1, 6, (1,), generator=generator))
        responses.append(torch.randint(STOP + 1, VOCAB, (length,), generator=generator).tolist() + [STOP])
    return Sampled(
        prompt_ids=_Tokenizer()(list(prompts))["input_ids"],
        response_ids=responses,
        logps=[[] for _ in responses],
        stopped=[True] * len(responses),
        decode_steps=max(len(r) for r in responses),
        row_steps=sum(len(r) for r in responses),
    )


class _ScoringDraw:
    """Samples from the policy and scores with it, optionally with every log probability shifted by ``offset``."""

    def __init__(self, model, offset=0.0):
        self.model, self.offset = model, offset

    def __call__(self, prompts, seed):
        return _sample(self.model, list(prompts), max_new_tokens=8, seed=seed)

    def score(self, prompt_ids, response_ids):
        logp, row_of = _response_logps(self.model, prompt_ids, response_ids, PAD)
        return [[v + self.offset for v in logp[row_of == i].tolist()] for i in range(len(prompt_ids))]


def test_sampler_check_accepts_the_policys_own_weights_and_refuses_others(model):
    from fts_examples.where_rl_learns.pilot import sampler_check

    check = sampler_check(_ScoringDraw(model), model, ["5 6", "7", "8 9"], pad_id=PAD)
    assert check["passed"] and check["mean_abs_logp_gap"] < 1e-5 and check["tokens"] > 3
    assert not sampler_check(_ScoringDraw(model, offset=0.5), model, ["5 6", "7"], pad_id=PAD)["passed"]


def test_vllm_sampler_stops_a_worker_that_never_replies(model, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_VLLM_HANG_ON", "generate")
    sampler = _fake_sampler(model, tmp_path, reply_timeout=1.0)
    with pytest.raises(RuntimeError, match="no reply in time"):
        sampler(["5"], seed=1)
    assert sampler.worker.wait(timeout=10) != 0


def test_vllm_sampler_scores_after_a_reload(model, tmp_path):
    sampler = _fake_sampler(model, tmp_path)
    sampled = sampler(["5 6", "7"], seed=1)
    logps = sampler.score(sampled.prompt_ids, sampled.response_ids)
    sampler.close()
    assert logps == [[-1.0] * len(r) for r in sampled.response_ids]
    expected = ["generate", "sleep", "wake", "reload", "score", "sleep", "exit"]
    assert (tmp_path / "ops.txt").read_text().split() == expected


class _DecodingTokenizer(_Tokenizer):
    pad_token_id = PAD

    def batch_decode(self, rows, **kwargs):
        return [" ".join(map(str, row)) for row in rows]


def _even_first_token(text, gold):
    return float(int(text.split()[0]) % 2 == 0)


def _run_arm(tmp_path, out, monkeypatch, **overrides):
    import argparse

    from transformers import Qwen3Config, Qwen3ForCausalLM

    from fts_examples.where_rl_learns import train

    monkeypatch.setitem(train.ARMS, "layer1", ["layers:1"])
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config).float().train()
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    problems = [(f"{i % 30 + 4} {i % 7 + 4}", "unused") for i in range(24)]
    args = dict(arm="layer1", seed=0, updates=4, updates_per_epoch=2, eval_every=2, hours=10.0, token_budget=1000)
    args = {**args, "allow_sampler_mismatch": False, "mask": None, "output_dir": tmp_path / out, **overrides}
    args = argparse.Namespace(**args)
    fresh = []

    def make_draw(is_fresh):
        fresh.append(is_fresh)
        return _fixed_draw

    status = train.run_arm(
        model,
        _DecodingTokenizer(),
        problems,
        problems[:6],
        make_draw,
        args,
        pad_id=PAD,
        device=torch.device("cpu"),
        score=_even_first_token,
    )
    return model, initial, status, fresh


@pytest.mark.filterwarnings("ignore:.*addresses.*experimental")
def test_arm_runner_trains_its_addresses_and_resumes_exactly(tmp_path, monkeypatch):
    import json

    whole, initial, status, fresh = _run_arm(tmp_path, "whole", monkeypatch)
    assert status["complete"] and status["updates"] == 4 and fresh == [True]
    for name, param in whole.named_parameters():
        changed = not torch.equal(param, initial[name])
        assert changed == name.startswith("model.layers.1."), name

    _, _, stopped, _ = _run_arm(tmp_path, "split", monkeypatch, hours=0.0)
    assert stopped["stopped"] == "time limit" and stopped["updates"] == 2
    resumed, _, status, fresh = _run_arm(tmp_path, "split", monkeypatch)
    assert status["complete"] and status["resumed_at"] == 2 and fresh == [False]
    for (name, a), b in zip(whole.named_parameters(), resumed.parameters()):
        assert torch.equal(a, b), name
    out = tmp_path / "split"
    assert [json.loads(r)["update"] for r in (out / "updates.jsonl").read_text().splitlines()] == [0, 1, 2, 3]
    evals = [json.loads(r) for r in (out / "evals.jsonl").read_text().splitlines()]
    assert [e["update"] for e in evals] == [0, 2, 4] and len(evals[0]["scores"]) == 4
    assert json.loads((out / "run.json").read_text())["trainable_tensors"] == 11


def test_top_fraction_keeps_the_global_top_coordinates():
    from fts_examples.where_rl_learns.masks import top_fraction

    kept = top_fraction({"a": torch.tensor([5.0, 1.0, 2.0]), "b": torch.tensor([[9.0, 0.0], [3.0, 4.0]])}, 3 / 7)
    assert kept["a"].tolist() == [0] and kept["b"].tolist() == [0, 3] and kept["b"].dtype == torch.int32


def test_masked_gradients_hold_every_coordinate_outside_the_mask(model):
    from fts_examples.where_rl_learns.masks import MaskedGradients, random_scores, top_fraction

    indices = top_fraction(random_scores(model.named_parameters(), seed=0), 0.1)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    masked = MaskedGradients(model, indices)
    optimizer = HostStateAdamW(model.parameters(), lr=1e-2)
    for seed in range(2):
        sampled = _sample(model, ["5 6 7", "8 9"], max_new_tokens=6, seed=seed)
        accumulate_drgrpo_gradient(model, sampled, [1.0, -1.0], horizon=6, clip=0.2, token_budget=4, pad_id=PAD)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    masked.remove()
    moved = 0
    for name, param in model.named_parameters():
        inside = torch.zeros(param.numel(), dtype=torch.bool)
        inside[indices[name].long()] = True
        change = (param.detach() - before[name]).flatten()
        assert torch.all(change[~inside] == 0), name
        moved += int((change[inside] != 0).sum())
    assert moved > 0


@pytest.mark.filterwarnings("ignore:.*addresses.*experimental")
def test_mask_arm_trains_its_mask_and_resumes_exactly(tmp_path, monkeypatch):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from fts_examples.where_rl_learns.masks import random_scores, save_mask, top_fraction

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    indices = top_fraction(random_scores(Qwen3ForCausalLM(config).named_parameters(), seed=0), 0.1)
    mask = tmp_path / "random.pt"
    save_mask(mask, indices, {"kind": "random", "seed": 0})
    whole, initial, status, _ = _run_arm(tmp_path, "whole", monkeypatch, arm="random1", mask=mask)
    assert status["complete"]
    for name, param in whole.named_parameters():
        inside = torch.zeros(param.numel(), dtype=torch.bool)
        inside[indices[name].long()] = True
        assert torch.equal(param.detach().flatten()[~inside], initial[name].flatten()[~inside]), name
    _run_arm(tmp_path, "split", monkeypatch, arm="random1", mask=mask, hours=0.0)
    resumed, _, status, _ = _run_arm(tmp_path, "split", monkeypatch, arm="random1", mask=mask)
    assert status["resumed_at"] == 2 and status["complete"]
    for (name, a), b in zip(whole.named_parameters(), resumed.parameters()):
        assert torch.equal(a, b), name
    save_mask(mask, indices, {"kind": "fisher"})
    with pytest.raises(ValueError, match="takes a random mask"):
        _run_arm(tmp_path, "wrong", monkeypatch, arm="random1", mask=mask)
