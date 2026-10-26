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
"""Draw the PyTorch GSM8K tutorial's training-trajectory figures from the rendered run's TensorBoard logs.

Reads the newest event file of each scenario under ``docs/notebook_artifacts/gsm8k/tensorboard_logs`` (the run the docs
render) and writes the figures the tutorial's "Reviewing the Training Trajectory" section shows into the dev notebook
directory, from where the publisher copies them:

- ``pytorch_fts_explicit_trajectory.png``: ``fts_explicit``'s task and retention losses, each phase shaded and
  labeled, with each phase's best epoch marked;
- ``pytorch_retention_tradeoff.png``: both scenarios' paths through (``val_loss``, ``retention_loss``), from the
  pretrained model to the end of training, with their best checkpoints marked.

Re-run it after re-rendering the notebook (``python scripts/gsm8k_trajectory_figures.py``), then copy the figures into
``docs/notebook_artifacts/gsm8k`` as well. It needs ``matplotlib`` and ``tensorboardX`` (for the event protobufs).
``--logs`` and ``--out`` point it at other directories (a local run's ``tensorboard_logs``, say).
"""
import argparse
import struct
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboardX.proto import event_pb2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "docs" / "notebook_artifacts" / "gsm8k" / "tensorboard_logs"
OUT = ROOT / "src" / "fts_examples" / "notebooks" / "dev" / "gsm8k"
PHASE_COLORS = ("#f4f1a3", "#c9b6e8", "#9db4f0", "#a6d8a6")
PHASE_LABELS = ("phase 0:\nblocks 24 to 27", "phase 1:\n+ LoRA adapters", "phase 2:\n+ blocks 16 to 23",
                "phase 3:\n+ blocks 0 to 15")
TASK_COLOR, RETENTION_COLOR, BASELINE_COLOR = "#e8603c", "#1f9bb5", "#7a7a7a"


def scalars(logs: Path, scenario: str) -> dict[str, list[tuple[int, float]]]:
    """The scalar series of a scenario's newest event file, by tag."""
    path = max((logs / scenario).glob("events.out.tfevents.*"), key=lambda p: int(p.name.split(".")[3]))
    out: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open("rb") as f:
        while header := f.read(12):
            (length,) = struct.unpack("<Q", header[:8])
            event = event_pb2.Event()
            event.ParseFromString(f.read(length))
            f.read(4)
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    out[value.tag].append((event.step, value.simple_value))
    return out


def phase_bounds(s: dict[str, list[tuple[int, float]]]) -> list[tuple[int, int, int]]:
    """``(phase, first step, last step)`` of each phase: a phase spans the epochs logged at its depth, the first
    from step 0."""
    bounds, start = [], 0
    for phase in sorted({int(d) for _, d in s["finetuning_schedule_depth"]}):
        end = max(st for st, d in s["finetuning_schedule_depth"] if int(d) == phase)
        bounds.append((phase, start, end))
        start = end
    return bounds


def explicit_figure(logs: Path, out: Path) -> None:
    s = scalars(logs, "fts_explicit")
    bounds = phase_bounds(s)
    fig, (task_ax, ret_ax) = plt.subplots(2, 1, figsize=(9, 6.4), dpi=110, sharex=True)
    for ax, tag, color, label in ((task_ax, "val_loss", TASK_COLOR, "val_loss (task)"),
                                  (ret_ax, "retention_loss", RETENTION_COLOR, "retention_loss")):
        steps, values = zip(*s[tag])
        for phase, lo, hi in bounds:
            ax.axvspan(lo, hi, color=PHASE_COLORS[phase % len(PHASE_COLORS)], alpha=0.55, lw=0)
        ax.plot(steps, values, color=color, lw=2.2, marker="o", ms=4, label=label)
        ax.set(ylabel=tag)
        ax.grid(alpha=0.3)
    # each phase's best task loss, and the retention loss at that epoch
    task = dict(s["val_loss"])
    retention = dict(s["retention_loss"])
    for phase, lo, hi in bounds:
        in_phase = [st for st in task if lo < st <= hi]
        if not in_phase:
            continue
        best = min(in_phase, key=task.__getitem__)
        task_ax.plot(best, task[best], marker="*", ms=13, color=TASK_COLOR, mec="black", mew=0.6)
        ret_ax.plot(best, retention[best], marker="*", ms=13, color=RETENTION_COLOR, mec="black", mew=0.6)
        task_ax.text((lo + hi) / 2, 0.97, PHASE_LABELS[phase] if phase < len(PHASE_LABELS) else f"phase {phase}",
                     ha="center", va="top", fontsize=8, transform=task_ax.get_xaxis_transform())
    # the pretrained model's task loss is far above the rest: clip it, and say where it was
    (_, start_task), *tuned = s["val_loss"]
    top = max(v for _, v in tuned)
    task_ax.set_ylim(min(v for _, v in tuned) - 0.03, top + 0.12)
    task_ax.annotate(f"pretrained: {start_task:.2f}", xy=(0, top + 0.12), xytext=(6, -14), textcoords="offset points",
                     fontsize=8, color=TASK_COLOR)
    start_retention = s["retention_loss"][0][1]
    ret_ax.axhline(start_retention, color=RETENTION_COLOR, ls=":", lw=1.2)
    ret_ax.text(0.01, start_retention, " pretrained", color=RETENTION_COLOR, fontsize=8, va="bottom",
                transform=ret_ax.get_yaxis_transform())
    task_ax.set_title("fts_explicit: task and retention loss by phase (stars: each phase's best val_loss)")
    ret_ax.set(xlabel="optimizer step")
    fig.tight_layout()
    fig.savefig(out / "pytorch_fts_explicit_trajectory.png")
    plt.close(fig)


def tradeoff_figure(logs: Path, out: Path) -> None:
    """Every tuned epoch faintly, and the best epoch of each phase (the checkpoint the next phase starts from)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=110)
    for scenario, color in (("fts_explicit", TASK_COLOR), ("nofts_baseline", BASELINE_COLOR)):
        s = scalars(logs, scenario)
        task, retention = dict(s["val_loss"]), dict(s["retention_loss"])
        steps = sorted(set(task) & set(retention))[1:]  # the pretrained point is drawn as a reference line instead
        ax.scatter([task[st] for st in steps], [retention[st] for st in steps], color=color, s=18, alpha=0.35)
        bests = []
        for phase, lo, hi in phase_bounds(s):
            in_phase = [st for st in steps if lo < st <= hi]
            if in_phase:
                bests.append((phase, min(in_phase, key=task.__getitem__)))
        ax.plot([task[st] for _, st in bests], [retention[st] for _, st in bests], color=color, lw=1.6, marker="*",
                ms=14, mec="black", mew=0.6,
                label=f"{scenario}: {'each phase' if len(bests) > 1 else 'the run'}'s best epoch")
        if len(bests) > 1:
            for phase, st in bests:
                ax.annotate(f"phase {phase}", xy=(task[st], retention[st]), xytext=(8, -4), textcoords="offset points",
                            fontsize=8.5)
        start_task, start_retention = s["val_loss"][0][1], s["retention_loss"][0][1]
    ax.axhline(start_retention, color=RETENTION_COLOR, ls=":", lw=1.2)
    ax.text(0.01, start_retention, f" pretrained retention_loss (its val_loss: {start_task:.2f})", fontsize=8,
            color=RETENTION_COLOR, va="bottom", transform=ax.get_yaxis_transform())
    ax.set(title="Task loss against retention loss (faint: every tuned epoch)",
           xlabel="val_loss (GSM8K solutions; lower is better)",
           ylabel="retention_loss (WikiText-103; lower is better)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "pytorch_retention_tradeoff.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, default=LOGS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    explicit_figure(args.logs, args.out)
    tradeoff_figure(args.logs, args.out)
