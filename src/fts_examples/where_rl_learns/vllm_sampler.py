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
Sampling through vLLM on the training device
============================================

:class:`VLLMSampler` drives a vLLM engine in its own process (:mod:`.vllm_worker`, under an interpreter that has vLLM)
and shares one device with the trainer. Each call hands the device over and back:

1. the FP32 policy moves to host memory (it holds no gradients between updates);
2. vLLM wakes, reloads the latest BF16 snapshot of the policy, and samples;
3. vLLM sleeps at level 2, freeing its weights and cache, and the policy moves back.

:meth:`VLLMSampler.refresh` writes that snapshot after each optimizer step, to a directory that should be in memory
(``/dev/shm``). vLLM samples the policy rounded to BF16, so sampling is off-policy by that rounding, as it is in any RL
setup with a separate inference engine.

On an RTX 4090, Qwen3-1.7B-Base sampled one update's 32 responses in about 6 s this way, against 161 s for
:func:`~fts_examples.where_rl_learns.drgrpo.sample_batch`.
"""
import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors.torch import save_file

from fts_examples.where_rl_learns.drgrpo import Sampled
from fts_examples.where_rl_learns.vllm_worker import MARKER

WORKER = Path(__file__).with_name("vllm_worker.py")


class VLLMSampler:
    """Samples one response per prompt at temperature 1 through a vLLM worker process.

    Args:
        policy: The trained model. It is moved to host memory for each call and back afterwards.
        tokenizer: The policy's tokenizer, so that both sides see the same prompt tokens.
        python: The interpreter with vLLM installed.
        model: The checkpoint vLLM loads at start; it must match the policy's architecture and initial weights.
        snapshot_dir: Where :meth:`refresh` writes the policy's weights for vLLM to reload.
        log_path: Where the worker's own output goes.
        policy_is_checkpoint: Whether the policy still holds ``model``'s weights, so the first call can sample from the
            engine's own load. A resumed policy does not, and the first call then reloads its snapshot.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        tokenizer,
        *,
        python: str,
        model: str,
        revision: str | None,
        max_new_tokens: int,
        max_prompt_tokens: int,
        stop_ids: Sequence[int],
        snapshot_dir: Path,
        log_path: Path,
        gpu_memory_utilization: float = 0.8,
        env: dict[str, str] | None = None,
        worker: Path = WORKER,
        policy_is_checkpoint: bool = True,
    ) -> None:
        self.policy, self.tokenizer = policy, tokenizer
        self.max_new_tokens, self.max_prompt_tokens = max_new_tokens, max_prompt_tokens
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.timings: dict[str, float] = {}
        command = [python, str(worker), "--model", model, "--gpu-memory-utilization", str(gpu_memory_utilization)]
        command += ["--max-model-len", str(max_prompt_tokens + max_new_tokens), "--stop-ids", *map(str, stop_ids)]
        if revision:
            command += ["--revision", revision]
        # vLLM sizes itself from the free memory it finds at start, so the policy waits in host memory meanwhile
        self.device = next(policy.parameters()).device
        policy.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.log = open(log_path, "a")
        self.worker = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, env=env
        )
        self.timings["load_s"] = self._read()["load_s"]
        self.awake = True
        # the engine loaded the checkpoint itself; the snapshot is what it reloads after every sleep
        self.refresh(policy)
        self.stale = not policy_is_checkpoint

    def _read(self) -> dict:
        for line in self.worker.stdout:
            if line.startswith(MARKER):
                payload = json.loads(line[len(MARKER) :])
                if "error" in payload:
                    raise RuntimeError(f"vLLM worker: {payload['error']}")
                return payload
            self.log.write(line)
        raise RuntimeError(f"vLLM worker exited with {self.worker.wait()}; its log is {self.log.name}")

    def _send(self, op: str, **payload) -> dict:
        self.worker.stdin.write(json.dumps({"op": op, **payload}) + "\n")
        self.worker.stdin.flush()
        reply = self._read()
        self.timings[f"{op}_s"] = reply.get("s", 0.0)
        return reply

    @torch.no_grad()
    def refresh(self, policy: torch.nn.Module) -> None:
        """Writes the policy's current weights, in BF16, for the next call to reload."""
        state = {}
        seen: set[int] = set()
        for name, tensor in policy.state_dict().items():
            # tied weights share storage, which safetensors refuses; vLLM ties them again on load
            if tensor.data_ptr() in seen:
                continue
            seen.add(tensor.data_ptr())
            state[name] = tensor.detach().to("cpu", torch.bfloat16).contiguous()
        save_file(state, str(self.snapshot_dir / "model.safetensors"))
        self.stale = True

    def _hand_over(self) -> None:
        start = time.perf_counter()
        self.policy.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.timings["policy_out_s"] = time.perf_counter() - start
        if not self.awake:
            self._send("wake")
            self.awake = True
            # a level 2 sleep discards the weights, so they always come back from the snapshot
            self.stale = True
        if self.stale:
            self._send("reload", path=str(self.snapshot_dir))
            self.stale = False

    def _hand_back(self) -> None:
        self._send("sleep")
        self.awake = False
        start = time.perf_counter()
        self.policy.to(self.device)
        self.timings["policy_in_s"] = time.perf_counter() - start

    def __call__(self, prompts: Sequence[str], seed: int) -> Sampled:
        """Samples one response per prompt, each with its own seed derived from ``seed``."""
        enc = self.tokenizer(list(prompts), truncation=True, truncation_side="left", max_length=self.max_prompt_tokens)
        prompt_ids = [list(ids) for ids in enc["input_ids"]]
        self._hand_over()
        try:
            reply = self._send(
                "generate",
                prompt_ids=prompt_ids,
                seeds=[seed * 100_003 + i for i in range(len(prompt_ids))],
                max_tokens=self.max_new_tokens,
            )
        finally:
            self._hand_back()
        tokens = reply["tokens"]
        return Sampled(
            prompt_ids=prompt_ids,
            response_ids=tokens,
            logps=[[] for _ in tokens],
            stopped=reply["stopped"],
            decode_steps=max(len(t) for t in tokens),
            row_steps=sum(len(t) for t in tokens),
        )

    def score(self, prompt_ids: Sequence[Sequence[int]], response_ids: Sequence[Sequence[int]]) -> list[list[float]]:
        """Each response token's log probability under the engine's current weights (the latest snapshot)."""
        self._hand_over()
        try:
            rows = dict(prompt_ids=[list(p) for p in prompt_ids], response_ids=[list(r) for r in response_ids])
            reply = self._send("score", **rows)
        finally:
            self._hand_back()
        return reply["logps"]

    def close(self) -> None:
        if self.worker.poll() is None:
            self._send("exit")
            self.worker.wait(timeout=60)
        self.log.close()
