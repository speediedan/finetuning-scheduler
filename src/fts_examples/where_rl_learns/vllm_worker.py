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
vLLM sampling worker
====================

Runs under a separate interpreter that has vLLM installed, because vLLM pins its own torch release, and answers the
trainer over standard input and output, one JSON object per line. Replies carry a :data:`MARKER` prefix so that log
lines vLLM writes to standard output cannot be mistaken for them.

Requests: ``generate`` (prompt token ids, one seed per prompt), ``sleep`` (level 2: weights and cache freed, so the
trainer has the device), ``wake``, ``reload`` (weights from a directory of safetensors files) and ``exit``.

This file imports nothing from ``fts_examples``, so it can run from any environment with vLLM.
"""
import argparse
import json
import sys
import time

MARKER = "@@where-rl-learns "


def reply(**payload) -> None:
    sys.stdout.write(MARKER + json.dumps(payload) + "\n")
    sys.stdout.flush()


def _completion(output, stop_ids: list[int]) -> tuple[list[int], bool]:
    """Returns a completion's tokens with its stopping token kept, and whether it stopped."""
    tokens = list(output.token_ids)
    stopped = output.finish_reason == "stop"
    if stopped and (not tokens or tokens[-1] not in stop_ids):
        reason = output.stop_reason
        tokens.append(reason if isinstance(reason, int) else stop_ids[0])
    return tokens, stopped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=5120)
    parser.add_argument("--stop-ids", type=int, nargs="+", required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    start = time.perf_counter()
    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_sleep_mode=True,
    )
    reply(ready=True, load_s=time.perf_counter() - start)
    for line in sys.stdin:
        request = json.loads(line)
        start = time.perf_counter()
        op = request["op"]
        if op == "generate":
            params = [
                SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=request["max_tokens"],
                    stop_token_ids=args.stop_ids,
                    seed=seed,
                )
                for seed in request["seeds"]
            ]
            prompts = [{"prompt_token_ids": ids} for ids in request["prompt_ids"]]
            results = llm.generate(prompts, params, use_tqdm=False)
            pairs = [_completion(r.outputs[0], args.stop_ids) for r in results]
            reply(tokens=[t for t, _ in pairs], stopped=[s for _, s in pairs], s=time.perf_counter() - start)
        elif op == "sleep":
            llm.sleep(level=2)
            reply(ok=True, s=time.perf_counter() - start)
        elif op == "wake":
            llm.wake_up()
            reply(ok=True, s=time.perf_counter() - start)
        elif op == "reload":
            llm.collective_rpc("reload_weights", kwargs={"weights_path": request["path"]})
            reply(ok=True, s=time.perf_counter() - start)
        elif op == "exit":
            reply(ok=True, s=0.0)
            return
        else:
            reply(error=f"unknown op {op!r}")


if __name__ == "__main__":
    main()
