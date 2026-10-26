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
"""Draw the PyTorch SuperGLUE tutorial's training-trajectory figures from the rendered run's TensorBoard logs.

Reads the newest event file of each scenario under ``docs/notebook_artifacts/superglue/tensorboard_logs`` (the run the
docs render) and writes the figures the tutorial's "Reviewing the Training Trajectory" section shows into the dev
notebook directory, from where the publisher copies them:

- ``pytorch_fts_explicit_trajectory.png``: ``fts_explicit``'s validation loss with each phase shaded and annotated;
- ``pytorch_fts_implicit_trajectory.png``: ``fts_implicit``'s validation loss and schedule depth, with the phase-10
  transition marked.

Re-run it after re-rendering the notebook (``python scripts/superglue_trajectory_figures.py``), then copy the figures
into ``docs/notebook_artifacts/superglue`` as well. It needs ``matplotlib`` and ``tensorboardX`` (for the event
protobufs).
"""
import struct
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboardX.proto import event_pb2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "docs" / "notebook_artifacts" / "superglue" / "tensorboard_logs"
OUT = ROOT / "src" / "fts_examples" / "notebooks" / "dev" / "superglue"
PHASE_COLORS = ("#f4f1a3", "#9db4f0", "#a6d8a6")


def scalars(scenario: str) -> dict[str, list[tuple[int, float]]]:
    """The scalar series of a scenario's newest event file, by tag."""
    path = max((LOGS / scenario).glob("events.out.tfevents.*"), key=lambda p: int(p.name.split(".")[3]))
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


def explicit_figure() -> None:
    s = scalars("fts_explicit")
    steps, loss = zip(*s["val_loss"])
    depth = [d for _, d in s["finetuning_schedule_depth"]]
    # a phase spans the epochs logged at its depth; the first starts at step 0
    bounds, start = [], 0
    for phase in sorted(set(depth)):
        end = max(st for st, d in zip(steps, depth) if d == phase)
        bounds.append((int(phase), start, end))
        start = end
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=110)
    labels = (
        "phase 0: head, pooler and every encoder\nlayer's attention, intermediate and output",
        "phase 1: + embedding LayerNorm",
        "phase 2: + word embeddings",
    )
    for (phase, lo, hi), color in zip(bounds, PHASE_COLORS):
        ax.axvspan(lo, hi, color=color, alpha=0.55, lw=0)
        ax.text((lo + hi) / 2, 0.86, labels[phase], ha="center", va="top", fontsize=8.5,
                transform=ax.get_xaxis_transform())
    ax.plot(steps, loss, color="#e8603c", lw=2.2, marker="o", ms=4)
    best = min(range(len(loss)), key=loss.__getitem__)
    ax.plot(steps[best], loss[best], marker="*", ms=14, color="#e8603c", mec="black", mew=0.6)
    arrow = dict(arrowstyle="->", lw=1.4, color="black")
    ax.annotate("best checkpoint\n(epoch 1)", xy=(steps[best], loss[best]), xytext=(steps[best] + 60, 0.33),
                fontsize=8.5, arrowprops=arrow)
    p1 = steps.index(bounds[1][1]) + 1
    ax.annotate("best checkpoint restored,\nthen phase 1 begins", xy=(steps[p1], loss[p1]),
                xytext=(steps[p1] - 330, 0.56), fontsize=8.5, ha="center", arrowprops=arrow)
    p2 = steps.index(bounds[2][1]) + 1
    ax.annotate("phase 2 thaws\nthe word embeddings", xy=(steps[p2], loss[p2]), xytext=(steps[p2] - 60, 0.33),
                fontsize=8.5, ha="right", arrowprops=arrow)
    ax.annotate("early stopping\nends the run", xy=(steps[-1], loss[-1]), xytext=(steps[-1] - 40, 0.66),
                fontsize=8.5, ha="right", arrowprops=arrow)
    ax.set(title="fts_explicit: validation loss by phase", xlabel="optimizer step", ylabel="val_loss",
           xlim=(0, steps[-1] + 40), ylim=(0.3, 0.8))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "pytorch_fts_explicit_trajectory.png")
    plt.close(fig)


def implicit_figure() -> None:
    s = scalars("fts_implicit")
    steps, loss = zip(*s["val_loss"])
    dsteps, depth = zip(*s["finetuning_schedule_depth"])
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=110)
    ax.plot(steps, loss, color="#1f9bb5", lw=1.8, label="val_loss")
    ax.set(title="fts_implicit: validation loss and schedule depth", xlabel="optimizer step", ylabel="val_loss",
           ylim=(0.3, 0.75))
    ax.grid(alpha=0.3)
    depth_ax = ax.twinx()
    depth_ax.step(dsteps, depth, where="post", color="#7a7a7a", lw=1.2, ls="--", label="finetuning_schedule_depth")
    depth_ax.set(ylabel="finetuning_schedule_depth", ylim=(-1, max(depth) + 2))
    p10 = next(i for i, d in enumerate(depth) if d == 10)
    ax.axvspan(dsteps[p10 - 1], dsteps[p10 + 1], color=PHASE_COLORS[0], alpha=0.6, lw=0)
    ax.annotate("phase 10 thaws the last encoder layer's\nattention key projection: val_loss leaves\nchance level "
                "(about 0.69)", xy=(steps[p10 + 1], loss[p10 + 1]), xytext=(steps[p10 + 1] + 2500, 0.62),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=1.4, color="black"))
    lines = ax.get_lines()[:1] + depth_ax.get_lines()[:1]
    ax.legend(lines, [ln.get_label() for ln in lines], loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "pytorch_fts_implicit_trajectory.png")
    plt.close(fig)


if __name__ == "__main__":
    explicit_figure()
    implicit_figure()
