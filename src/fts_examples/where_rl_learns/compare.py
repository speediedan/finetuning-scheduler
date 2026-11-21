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
Paired comparison of arms
=========================

Every arm is evaluated on the same 500 MATH-500 problems with the same sampling seeds, and the arm runner keeps every
response's score, so two arms (or one arm before and after training) compare problem by problem. The difference of
their per-problem accuracies has a much smaller standard error than either accuracy alone, because most of the
variation between problems is shared.

.. code-block:: bash

    python -m fts_examples.where_rl_learns.compare runs/full runs/layer10 runs/layer24

``--summarize <file>`` instead writes the runs' evaluations as a compact summary (each problem's count of correct
responses, one digit per problem), which :func:`load_summary` reads back and :func:`compare` accepts in place of the
run directories. The tutorial's results ship as one (``results/seed0.json`` beside this module).
"""
import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent / "results"
_RUN_KEYS = ("arm", "seed", "model", "model_revision", "dataset_revision", "addresses", "trainable_tensors",
             "trainable_parameters", "parameters", "device", "mask")


def load_evals(run_dir: Path) -> dict[int, dict[str, Any]]:
    """A run's evaluations by update."""
    lines = (Path(run_dir) / "evals.jsonl").read_text().splitlines()
    return {record["update"]: record for record in map(json.loads, lines)}


def problem_accuracy(record: Mapping[str, Any]) -> list[float]:
    """Each problem's accuracy over the evaluation's seeds."""
    if "correct" in record:  # a summary record: each problem's count of correct responses
        return [int(c) / len(record["seeds"]) for c in record["correct"]]
    return [sum(column) / len(column) for column in zip(*record["scores"])]


def _problems(record: Mapping[str, Any]) -> int:
    return len(record["correct"]) if "correct" in record else len(record["scores"][0])


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """A run's metadata, its mean reward over each 50 updates, and its evaluations with every response's score
    reduced to each problem's count of correct responses."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    rewards = [json.loads(line).get("mean_reward") for line in (run_dir / "updates.jsonl").read_text().splitlines()]
    rewards = [r for r in rewards if r is not None]
    evals = []
    for update, record in sorted(load_evals(run_dir).items()):
        counts = [sum(column) for column in zip(*record["scores"])]
        evals.append({
            "update": update,
            "accuracy_percent": record["accuracy_percent"],
            "standard_error_percent": record["standard_error_percent"],
            "mean_length": record["mean_length"],
            "seeds": record["seeds"],
            "correct": "".join(str(c) for c in counts),
        })
    return {
        "run": {key: run.get(key) for key in _RUN_KEYS},
        "reward_per_50_updates": [round(sum(rewards[i : i + 50]) / len(rewards[i : i + 50]), 4)
                                  for i in range(0, len(rewards), 50)],
        "evals": evals,
    }


def load_summary(path: Path | str = RESULTS_DIR / "seed0.json") -> dict[str, dict[str, Any]]:
    """A results summary, by arm."""
    return json.loads(Path(path).read_text())["arms"]


def paired_difference(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    """``a`` minus ``b`` in percentage points, with the standard error of the mean over problems."""
    if a["seeds"] != b["seeds"] or _problems(a) != _problems(b):
        raise ValueError("The two evaluations did not use the same problems and seeds.")
    diffs = [x - y for x, y in zip(problem_accuracy(a), problem_accuracy(b))]
    mean = sum(diffs) / len(diffs)
    variance = sum((d - mean) ** 2 for d in diffs) / max(1, len(diffs) - 1)
    return {
        "difference_points": 100 * mean,
        "standard_error_points": 100 * math.sqrt(variance / len(diffs)),
        "problems": len(diffs),
        "a_percent": a["accuracy_percent"],
        "b_percent": b["accuracy_percent"],
    }


def compare(runs: Sequence[Path] | Mapping[str, Mapping[str, Any]], update: int | None = None) -> dict[str, Any]:
    """Each run's gain over its own initial evaluation, and every pair of runs at ``update`` (by default the last
    update all of them evaluated).

    ``runs`` is a sequence of arm runner output directories, or a summary by arm (from :func:`load_summary`).
    """
    if isinstance(runs, Mapping):
        evals = {name: {r["update"]: r for r in arm["evals"]} for name, arm in runs.items()}
    else:
        evals = {Path(r).name: load_evals(r) for r in runs}
    common = set.intersection(*(set(e) for e in evals.values()))
    at = max(common) if update is None else update
    gains = {name: paired_difference(e[at], e[0]) for name, e in evals.items()}
    names = list(evals)
    pairs = {
        f"{x} - {y}": paired_difference(evals[x][at], evals[y][at]) for i, x in enumerate(names) for y in names[i + 1 :]
    }
    return {"update": at, "gain_over_initial": gains, "pairs": pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="arm runner output directories")
    parser.add_argument("--update", type=int, help="the update to compare at (default: the last common one)")
    parser.add_argument("--summarize", type=Path, help="write the runs as a results summary to this file instead")
    args = parser.parse_args()
    if args.summarize:
        arms = {}
        for run_dir in args.runs:
            summary = summarize_run(run_dir)
            arms[summary["run"]["arm"]] = summary
        args.summarize.write_text(json.dumps({"arms": arms}, indent=1) + "\n")
        return
    print(json.dumps(compare(args.runs, args.update), indent=1))


if __name__ == "__main__":
    main()
