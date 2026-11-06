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
"""Phased fine-tuning of an interpretune session with the PyTorch API.

A small GPT-2 class model is fine-tuned in three phases inside an `interpretune <https://interpretune.org>`_ session,
by interpretune's Lightning-free core training loop. interpretune's ``FTSCoreCallback`` drives a ``TuningScheduler``
at the loop's boundaries, and the schedule names parameters through the tunable view interpretune attaches to its
module, roles included:

- phase 0 thaws only a steering vector registered beside the model, selected by its role
  (``role:intervention:steer``);
- phase 1 adds the last transformer block and the final norm;
- phase 2 adds the rest of the model.

The model is randomly initialized and the tokenizer and data are built in memory, so the example downloads nothing
and runs on a CPU in seconds::

    python interpretune_phased_tuning.py

It prints a JSON summary of each epoch: the schedule depth, how many parameters were trainable, and the losses.

This example needs an interpretune revision that attaches views as ``tunable_view`` and provides
``interpretune.tuning.core_loop`` (neither is in a published interpretune release yet). Fine-Tuning Scheduler's CI
installs the revision in ``requirements/ci/interpretune.txt``.

.. warning:: This example uses the experimental PyTorch API and interpretune APIs that may change.
"""
import argparse
import json
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader

import interpretune as it
from interpretune.config import SessionRunnerCfg
from interpretune.tuning.core_loop import FTSCoreCallback

VOCAB = ["<pad>"] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
SEQ_LEN = 16

# phase 0's learning rate is the optimizer's (the module's `optimizer_init`); each later phase sets its own, since
# one that does not gets `TuningScheduler`'s `base_max_lr` (1e-5)
SCHEDULE: dict[int, dict[str, Any]] = {
    0: {"params": ["role:intervention:steer"], "max_transition_epoch": 1},
    1: {"params": ["model.transformer.h.1.*", "model.transformer.ln_f.*"], "lr": 2e-2, "max_transition_epoch": 2},
    2: {"params": ["model.transformer.h.0.*", "model.transformer.w.e.weight"], "lr": 1e-2},
}


def build_tokenizer() -> Any:
    """A character-level tokenizer over ``VOCAB``, built in memory."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel({t: i for i, t in enumerate(VOCAB)}, unk_token="<pad>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>")


class AlphabetDataModule(it.ITDataModule):
    """Runs of the alphabet from random starting letters (wrapping after ``z``), so the next token is learnable."""

    def setup(self, stage: str | None = None, module: Any = None, *args: Any, **kwargs: Any) -> None:
        generator = torch.Generator().manual_seed(0)
        starts = torch.randint(0, len(VOCAB) - 1, (320, 1), generator=generator)
        self.samples = (starts + torch.arange(SEQ_LEN)) % (len(VOCAB) - 1) + 1

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.samples[:256], batch_size=16)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.samples[256:], batch_size=16)


class SteeringVector(torch.nn.Module):
    """A vector added to the residual stream after a transformer block: an intervention, not part of the model."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.vector = torch.nn.Parameter(torch.zeros(dim))

    def hook(self, module: torch.nn.Module, args: Any, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return output + self.vector
        return (output[0] + self.vector, *output[1:])


class SteeredLMModule(it.ITModule):
    """A small, randomly initialized GPT-2 with a steering vector after its first block."""

    def model_init(self) -> None:
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(0)
        config = GPT2Config(vocab_size=len(VOCAB), n_positions=SEQ_LEN, n_embd=32, n_layer=2, n_head=2,
                            bos_token_id=0, eos_token_id=0)
        self.model = GPT2LMHeadModel(config)

    def register_tunable_modules(self) -> None:
        super().register_tunable_modules()
        # registered beside `self.model`, so the view (rooted at this module) reaches it as `steer.vector`
        self.steer = SteeringVector(self.model.config.n_embd)
        self.model.transformer.h[0].register_forward_hook(self.steer.hook)

    def tunable_roles(self) -> dict[str, str]:
        return {**super().tunable_roles(), "steer.vector": "intervention:steer"}

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.model(input_ids=batch, labels=batch).loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.log("val_loss", self.model(input_ids=batch, labels=batch).loss)


class PhaseRecorder:
    """A core-loop callback recording each epoch's schedule depth, trainable parameters and metrics.

    Registered after ``FTSCoreCallback``, so each epoch is recorded after any phase transition it starts with.
    """

    def __init__(self, fts: FTSCoreCallback) -> None:
        self.fts = fts
        self.epochs: list[dict[str, Any]] = []

    def on_train_epoch_start(self, ctx: Any) -> None:
        assert self.fts.tuning_scheduler is not None
        trainable = [n for n, p in ctx.module.named_parameters() if p.requires_grad]
        self.epochs.append({"epoch": ctx.epoch, "depth": self.fts.tuning_scheduler.curr_depth,
                            "trainable": trainable})

    def on_train_epoch_end(self, ctx: Any, metrics: Mapping[str, Any]) -> None:
        self.epochs[-1].update({k: round(float(v), 4) for k, v in metrics.items()})


def build_session() -> Any:
    """An interpretune session over :class:`SteeredLMModule` and :class:`AlphabetDataModule`."""
    shared = {"model_name_or_path": "tiny-gpt2", "task_name": "alphabet_lm"}
    datamodule_cfg = it.ITDataModuleConfig(tokenizer=build_tokenizer(), **shared)
    module_cfg = it.ITConfig(optimizer_init={"class_path": "torch.optim.AdamW", "init_args": {"lr": 5e-2}},
                             log_env_details=False, **shared)
    return it.ITSession(it.ITSessionConfig(adapter_ctx=(it.Adapter.core,), datamodule_cfg=datamodule_cfg,
                                           module_cfg=module_cfg, datamodule_cls=AlphabetDataModule,
                                           module_cls=SteeredLMModule))


def run(max_epochs: int = 3) -> dict[str, Any]:
    """Run the schedule and return a summary of each epoch."""
    session = build_session()
    fts = FTSCoreCallback(SCHEDULE)  # the optimizer comes from the module's `optimizer_init`
    recorder = PhaseRecorder(fts)
    it.SessionRunner(SessionRunnerCfg(it_session=session, max_epochs=max_epochs, callbacks=[fts, recorder])).train()
    assert fts.tuning_scheduler is not None
    return {"final_depth": fts.tuning_scheduler.curr_depth,
            "view_root_is_module": fts.tuning_scheduler.view.root is session.module,
            "epochs": recorder.epochs}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-epochs", type=int, default=3)
    args = parser.parse_args(argv)
    summary = run(args.max_epochs)
    for epoch in summary["epochs"]:
        epoch["trainable"] = len(epoch["trainable"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
