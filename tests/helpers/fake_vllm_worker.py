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
"""A stand-in for the vLLM sampling worker that needs no vLLM: the same protocol, with fixed completions.

Each completion is ``[seed % 7 + 5, stop]``, each scored token's log probability is -1, and every request's op is
appended to the file ``FAKE_VLLM_OPS`` names, so a test can check the order of the hand-over. A log line goes to
standard output first, which the sampler must skip.
"""
import argparse
import json
import os
import sys
from pathlib import Path

MARKER = "@@where-rl-learns "


def reply(**payload) -> None:
    sys.stdout.write(MARKER + json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--gpu-memory-utilization")
    parser.add_argument("--max-model-len")
    parser.add_argument("--stop-ids", type=int, nargs="+")
    args = parser.parse_args()
    ops = Path(os.environ["FAKE_VLLM_OPS"])
    print("INFO a log line the sampler must skip", flush=True)
    reply(ready=True, load_s=0.0)
    for line in sys.stdin:
        request = json.loads(line)
        with ops.open("a") as fh:
            fh.write(request["op"] + "\n")
        if request["op"] == "generate":
            tokens = [[seed % 7 + 5, args.stop_ids[0]] for seed in request["seeds"]]
            reply(tokens=tokens, stopped=[True] * len(tokens), s=0.0)
        elif request["op"] == "score":
            reply(logps=[[-1.0] * len(r) for r in request["response_ids"]], s=0.0)
        elif request["op"] == "reload":
            reply(ok=(Path(request["path"]) / "model.safetensors").exists(), s=0.0)
        elif request["op"] == "exit":
            reply(ok=True, s=0.0)
            return
        else:
            reply(ok=True, s=0.0)


if __name__ == "__main__":
    main()
