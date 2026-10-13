# %% [markdown]
# ## Scheduled Fine-Tuning in a Plain PyTorch Loop
#
# ![Fine-Tuning Scheduler logo](logo_fts.png){height="55px" width="401px"}
#
# This notebook runs the same experiment as the
# [Lightning tutorial](https://finetuning-scheduler.readthedocs.io/en/latest/) (fine-tuning a small foundation model on
# the [RTE](https://huggingface.co/datasets/aps/super_glue) task of [SuperGLUE](https://super.gluebenchmark.com/) with
# a multi-phase schedule), but with a training loop you write yourself. There is no Lightning ``Trainer``: the
# [FTSController](https://finetuning-scheduler.readthedocs.io/en/latest/pytorch/plain_pytorch_loop.html) owns which
# parameters are trainable, the optimizer's parameter groups and when the next phase starts, and the loop stays yours.
#
# The PyTorch API is experimental in Fine-Tuning Scheduler 2.15. It shares its scheduling logic with the Lightning
# callback (the same schedule format, phase thawing, parameter-group growth and plateau criteria), and this notebook
# does not import Lightning.
#
# We compare three scenarios, as the Lightning tutorial does:
#
# | Scenario | Fine-tuning schedule |
# |:--------:| -------------------- |
# | ``fts_explicit`` | a schedule written by the user |
# | ``fts_implicit`` | the default schedule the controller generates |
# | ``nofts_baseline`` | one phase that thaws every parameter, i.e. ordinary fine-tuning |
#
# Each phase ends when validation loss stops improving, and the final phase's plateau ends training.

# %% tags=["parameters"]
# papermill parameters: shrink these for a quick smoke run
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_EPOCHS = 100
TRAIN_SUBSET = None  # an int keeps only that many training examples
VAL_SUBSET = None
BATCH_SIZE = 16
MAX_DEPTH = 1  # how deep the fts_explicit and fts_implicit schedules go; -1 runs every phase

# %%
import os
import warnings
from functools import partial

import datasets
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as transformers_logging

from finetuning_scheduler import FTSController, PlateauTransition
from finetuning_scheduler.core.checkpoint import BestCheckpoint

transformers_logging.set_verbosity_error()
datasets.logging.set_verbosity_error()
datasets.logging.disable_progress_bar()
warnings.filterwarnings("ignore", r".*converting to a fast.*")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Running on {device}" + ("" if use_amp else " (this notebook is best run with a GPU)"))

# %% [markdown]
# ### Data
#
# RTE pairs a premise with a hypothesis and asks whether the premise entails it. We tokenize the pairs once and build
# ordinary ``DataLoader``s.

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, clean_up_tokenization_spaces=True)
raw = datasets.load_dataset("aps/super_glue", "rte")


def encode(batch):
    features = tokenizer(batch["premise"], batch["hypothesis"], max_length=128, padding="max_length", truncation=True)
    features["labels"] = batch["label"]
    return features


def loader(split: str, subset: int | None, shuffle: bool) -> DataLoader:
    ds = raw[split] if subset is None else raw[split].select(range(subset))
    ds = ds.map(encode, batched=True, remove_columns=ds.column_names)
    ds.set_format(type="torch")
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=torch.Generator().manual_seed(42))


train_loader = loader("train", TRAIN_SUBSET, shuffle=True)
val_loader = loader("validation", VAL_SUBSET, shuffle=False)
print(f"{len(train_loader.dataset)} training and {len(val_loader.dataset)} validation examples")

# %% [markdown]
# ### The Fine-Tuning Schedule
#
# The explicit schedule is the Lightning tutorial's, with one difference: here the model is the Hugging Face model
# itself rather than an attribute of a ``LightningModule``, so parameter names have no ``model.`` prefix. Entries are
# regular expressions. Phase 0 thaws the classifier head, the pooler, the encoder's ``LayerNorm`` and relative position
# embeddings, and the attention, intermediate and output blocks of every encoder layer (``.{0,11}`` matches any layer
# index); phase 1 thaws the embedding ``LayerNorm``; phase 2 the word embeddings. A phase may also set its own base
# learning rate with ``lr``.

# %%
explicit_schedule = {
    0: {
        "params": [
            "classifier.bias",
            "classifier.weight",
            "pooler.dense.bias",
            "pooler.dense.weight",
            "deberta.encoder.LayerNorm.bias",
            "deberta.encoder.LayerNorm.weight",
            "deberta.encoder.rel_embeddings.weight",
            "deberta.encoder.layer.{0,11}.(output|attention|intermediate).*",
        ]
    },
    1: {"params": ["deberta.embeddings.LayerNorm.bias", "deberta.embeddings.LayerNorm.weight"]},
    2: {"params": ["deberta.embeddings.word_embeddings.weight"]},
}
baseline_schedule = {0: {"params": [".*"]}}  # one phase, every parameter trainable from the start

# %% [markdown]
# ### Optimizer, LR Scheduler and Transitions
#
# The optimizer and lr scheduler are passed as factories. The controller builds the optimizer from phase 0's parameter
# groups and adds a group as each later phase thaws parameters. ``AdamW`` with these values and
# ``CosineAnnealingWarmRestarts``, which needs no global ``max_epochs``, match the Lightning tutorial.
#
# ``PlateauTransition`` plays the part ``FTSEarlyStopping`` plays under Lightning: a phase ends when ``val_loss`` has
# not improved by ``min_delta`` for ``patience`` epochs. ``BestCheckpoint`` keeps the best checkpoint, and the
# controller restores it at each phase boundary, so every phase starts from the best state of the one before.

# %%
optimizer_factory = partial(AdamW, weight_decay=1e-05, eps=1e-07, lr=1e-05)
lr_scheduler_factory = partial(CosineAnnealingWarmRestarts, T_0=1, T_mult=2, eta_min=1e-07)


def make_controller(model, schedule, scenario: str, max_depth: int) -> FTSController:
    return FTSController(
        model,
        schedule,
        optimizer=optimizer_factory,
        lr_scheduler=lr_scheduler_factory,
        lr_scheduler_cfg={"interval": "epoch"},
        transition=PlateauTransition(monitor="val_loss", patience=2, min_delta=0.001),
        checkpoint=BestCheckpoint(os.path.join("checkpoints", scenario), monitor="val_loss"),
        max_depth=max_depth,
        log_dir=os.path.join("fts_logs", scenario),
    )


# %% [markdown]
# ### The Training Loop
#
# This is an ordinary mixed-precision loop. The controller appears in four places: ``setup()`` before training,
# ``epochs()`` to iterate (it starts the next phase when one is due), ``fts.optimizer`` in place of an optimizer held by
# the loop (a schedule may replace the optimizer at a phase transition), and ``epoch_end()`` with the epoch's
# validation metrics.

# %%
@torch.no_grad()
def evaluate(model) -> dict[str, float]:
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model(**batch)
        n = batch["labels"].numel()
        total_loss += out.loss.float().item() * n
        correct += (out.logits.argmax(-1) == batch["labels"]).sum().item()
        count += n
    model.train()
    return {"val_loss": total_loss / count, "accuracy": correct / count}


def run(scenario: str, schedule, max_depth: int) -> list[dict]:
    torch.manual_seed(42)
    # fp32 master weights: transformers would otherwise keep the checkpoint's stored dtype, and autocast handles fp16
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, dtype=torch.float32)
    model = model.to(device)
    fts = make_controller(model, schedule, scenario, max_depth)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history = []
    fts.setup()
    for epoch in fts.epochs(max_epochs=MAX_EPOCHS):
        train_loss, steps = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(fts.optimizer)
            scaler.update()
            fts.optimizer.zero_grad()
            train_loss, steps = train_loss + loss.item(), steps + 1
        metrics = evaluate(model)
        fts.step_lr_schedulers("epoch")
        fts.epoch_end({"val_loss": metrics["val_loss"]})
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        history.append({"epoch": epoch, "depth": fts.curr_depth, "trainable": trainable, **metrics})
        print(
            f"{scenario:>14} | epoch {epoch:>2} | phase {fts.curr_depth} | trainable {trainable:>11,} | "
            f"lr {fts.optimizer.param_groups[0]['lr']:.2e} | train_loss {train_loss / steps:.4f} | "
            f"val_loss {metrics['val_loss']:.4f} | accuracy {metrics['accuracy']:.4f}"
        )
    print(f"{scenario}: finished at phase {fts.curr_depth} after {len(history)} epochs ({fts.stop_reason or 'done'})")
    return history


# %% [markdown]
# ### Running the Scenarios
#
# ``fts_explicit`` uses the schedule above. ``fts_implicit`` passes no schedule, so the controller generates the default
# one (parameters thawed in pairs, from the last registered parameter). ``MAX_DEPTH`` keeps both to their first two
# phases, which shows the phase mechanics at a reasonable cost; set it to ``-1`` to run every phase. The baseline's
# single phase makes it ordinary fine-tuning with early stopping.

# %%
results = {
    "fts_explicit": run("fts_explicit", explicit_schedule, MAX_DEPTH),
    "fts_implicit": run("fts_implicit", None, MAX_DEPTH),
    "nofts_baseline": run("nofts_baseline", baseline_schedule, -1),
}

# %% [markdown]
# ### Comparing the Results
#
# The best validation accuracy and the epochs each scenario took. Expect some run-to-run variation with the hardware
# and package versions.

# %%
print(f"{'scenario':>14} | {'epochs':>6} | {'final phase':>11} | {'best accuracy':>13} | {'best val_loss':>13}")
for scenario, history in results.items():
    best = min(history, key=lambda h: h["val_loss"])
    print(
        f"{scenario:>14} | {len(history):>6} | {history[-1]['depth']:>11} | {max(h['accuracy'] for h in history):>13.4f} "
        f"| {best['val_loss']:>13.4f}"
    )

# %% [markdown]
# ## Where to Go Next
#
# - [Scheduled fine-tuning in a plain PyTorch loop](https://finetuning-scheduler.readthedocs.io/en/latest/pytorch/plain_pytorch_loop.html):
#   transition policies, phase hooks and gates, exact resumption and distributed training.
# - [Model-parallel scheduled fine-tuning in a plain loop](https://finetuning-scheduler.readthedocs.io/en/latest/pytorch/model_parallel_plain_loop.html):
#   the same controller with FSDP2 and tensor parallelism.
# - The [Lightning version of this tutorial](https://finetuning-scheduler.readthedocs.io/en/latest/), for the
#   ``FinetuningScheduler`` callback.
