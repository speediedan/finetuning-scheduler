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
"""A demonstration of the scheduled finetuning callback
:ref:`FinetuningScheduler<advanced/finetuning_scheduler:Finetuning Scheduler>` using the
`RTE <https://huggingface.co/datasets/viewer/?dataset=super_glue&config=rte>`_ and
`BoolQ <https://github.com/google-research-datasets/boolean-questions>`_ tasks of the
`SuperGLUE <https://super.gluebenchmark.com/>`_ benchmark and the :ref:`LightningCLI<common/lightning_cli:LightningCLI>`

There are three different demo schedule configurations composed with shared defaults (./config/fts_defaults.yaml)
provided for the default 'rte' task. Note DDP (with auto-selected GPUs) is the default configuration so ensure you
adjust the configuration files referenced below as desired for other configurations.

.. code-block:: bash

    # Generate a baseline without scheduled finetuning enabled:
    python fts_superglue.py fit --config config/nofts_baseline.yaml

    # Train with the default finetuning schedule:
    python fts_superglue.py fit --config config/fts_implicit.yaml

    # Train with a non-default finetuning schedule:
    python fts_superglue.py fit --config config/fts_explicit.yaml
"""

import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cli import LightningCLI
from torch.utils.data import DataLoader

from finetuning_scheduler.fts import FinetuningScheduler
from fts_examples import _HF_AVAILABLE, _SP_AVAILABLE
from fts_examples.cli_experiment_utils import collect_env_info, instantiate_registered_class

if _HF_AVAILABLE:
    import datasets
    from datasets.arrow_dataset import Batch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from transformers import logging as transformers_logging
    from transformers.tokenization_utils_base import BatchEncoding

TASK_NUM_LABELS = {"boolq": 2, "rte": 2}
DEFAULT_TASK = "rte"


transformers_logging.set_verbosity_error()
# ignore warnings related tokenizers_parallelism/DataLoader parallelism tradeoff and
#  expected logging behavior
for warnf in [".*does not have many workers*", ".*The number of training samples.*"]:
    warnings.filterwarnings("ignore", warnf)


class RteBoolqDataModule(pl.LightningDataModule):
    """A :class:`~pytorch_lighting.core.LightningDataModule` for using either the RTE or BoolQ `SuperGLUE Hugging
    Face datasets <https://huggingface.co/datasets/super_glue#data-instances>`_."""

    TASK_TEXT_FIELD_MAP = {"rte": ("premise", "hypothesis"), "boolq": ("question", "passage")}
    LOADER_COLUMNS = (
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    )

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = DEFAULT_TASK,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        tokenizers_parallelism: bool = True,
        **dataloader_kwargs: Any,
    ):
        r"""Initialize the ``LightningDataModule`` designed for both the RTE or BoolQ SuperGLUE Hugging Face
        datasets.

        Args:
            model_name_or_path (str):
                Can be either:
                    - A string, the ``model id`` of a pretrained model hosted inside a model repo on huggingface.co.
                        Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced
                        under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a ``directory`` containing model weights saved using
                        :meth:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
            task_name (str, optional): Name of the SuperGLUE task to execute. This module supports 'rte' or 'boolq'.
                Defaults to DEFAULT_TASK which is 'rte'.
            max_seq_length (int, optional): Length to which we will pad sequences or truncate input. Defaults to 128.
            train_batch_size (int, optional): Training batch size. Defaults to 16.
            eval_batch_size (int, optional): Batch size to use for validation and testing splits. Defaults to 16.
            tokenizers_parallelism (bool, optional): Whether to use parallelism in the tokenizer. Defaults to True.
            \**dataloader_kwargs: Arguments passed when initializing the dataloader
        """
        super().__init__()
        task_name = task_name if task_name in TASK_NUM_LABELS.keys() else DEFAULT_TASK
        self.text_fields = self.TASK_TEXT_FIELD_MAP[task_name]
        self.dataloader_kwargs = {
            "num_workers": dataloader_kwargs.get("num_workers", 0),
            "pin_memory": dataloader_kwargs.get("pin_memory", False),
        }
        self.save_hyperparameters()
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if self.hparams.tokenizers_parallelism else "false"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.model_name_or_path, use_fast=True, local_files_only=False
        )

    def prepare_data(self):
        """Load the SuperGLUE dataset."""
        # N.B. PL calls prepare_data from a single process (rank 0) so do not use it to assign
        # state (e.g. self.x=y)
        datasets.load_dataset("super_glue", self.hparams.task_name)

    def setup(self, stage):
        """Setup our dataset splits for training/validation."""
        self.dataset = datasets.load_dataset("super_glue", self.hparams.task_name)
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._convert_to_features, batched=True, remove_columns=["label"]
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.LOADER_COLUMNS]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.hparams.train_batch_size, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.hparams.eval_batch_size, **self.dataloader_kwargs)

    def _convert_to_features(self, example_batch: Batch) -> BatchEncoding:
        """Convert raw text examples to a :class:`~transformers.tokenization_utils_base.BatchEncoding` container
        (derived from python dict) of features that includes helpful methods for translating between word/character
        space and token space.

        Args:
            example_batch ([type]): The set of examples to convert to token space.

        Returns:
            ``BatchEncoding``: A batch of encoded examples (note default tokenizer batch_size=1000)
        """
        text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            text_pairs, max_length=self.hparams.max_seq_length, padding="longest", truncation=True
        )
        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]
        return features


class RteBoolqModule(pl.LightningModule):
    """A :class:`~pytorch_lightning.core.lightning.LightningModule` that can be used to finetune a foundational
    model on either the RTE or BoolQ `SuperGLUE <https://super.gluebenchmark.com/>`_ tasks using Hugging Face
    implementations of a given model and the `SuperGLUE Hugging Face dataset.

    <https://huggingface.co/datasets/super_glue#data-instances>`_.
    """

    def __init__(
        self,
        model_name_or_path: str,
        optimizer_init: Dict[str, Any],
        lr_scheduler_init: Dict[str, Any],
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        model_cfg: Optional[Dict[str, Any]] = None,
        task_name: str = DEFAULT_TASK,
        experiment_tag: str = "default",
        log_env_details: bool = True,
    ):
        """In this example, this :class:`~pytorch_lightning.core.lightning.LightningModule` is initialized by composing
        the ./config/fts_defaults.yaml default configuration with various scheduled finetuning yaml configurations
        via the :class:`~pytorch_lightning.utilities.cli.LightningCLI` but it can be used like any other
        :class:`~pytorch_lightning.core.lightning.LightningModule` as well.

        Args:
            model_name_or_path (str): Path to pretrained model or identifier `from <https://huggingface.co/models>`_
            optimizer_init (Dict[str, Any]): The desired optimizer configuration.
            lr_scheduler_init (Dict[str, Any]): The desired learning rate scheduler config
            pl_lrs_cfg (Optional[Dict[str, Any]]): Defines custom overrides of pytorch lightning lr_scheduler defaults
                defined in :func:`~pytorch_lighting.optimizers._get_default_scheduler_config`
                Example::

                .. code-block:: yaml

                pl_lrs_cfg:
                    interval: epoch
                    frequency: 1
                    name: CosineAnnealingWithWarmRestartsLR

            model_cfg (Optional[Dict[str, Any]], optional): Defines overrides of the default model config. Defaults to
                ``None``.
            task_name (str, optional): The SuperGLUE task to execute, one of ``'rte'``, ``'boolq'``. Defaults to "rte".
            experiment_tag (str, optional): The tag to use for the experiment and tensorboard logs. Defaults to
                "default".
            log_env_details (bool, optional): Whether to collect and log environmental details to facilitate
                reproducibility. Defaults to ``True``.
        """
        super().__init__()
        if task_name not in TASK_NUM_LABELS.keys():
            rank_zero_warn(f"Invalid task_name {task_name!r}. Proceeding with the default task: {DEFAULT_TASK!r}")
            task_name = DEFAULT_TASK
        self.num_labels = TASK_NUM_LABELS[task_name]
        self.model_cfg = model_cfg or {}
        conf = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels, local_files_only=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=conf)
        self.model.config.update(self.model_cfg)  # apply model config overrides
        self.init_hparams = {
            "optimizer_init": optimizer_init,
            "lr_scheduler_init": lr_scheduler_init,
            "pl_lrs_cfg": pl_lrs_cfg,
            "model_config": self.model.config,
            "model_name_or_path": model_name_or_path,
            "task_name": task_name,
            "experiment_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_tag}",
        }
        self.init_hparams["env_info"] = collect_env_info() if log_env_details else None
        self.save_hyperparameters(self.init_hparams)
        self.metric = datasets.load_metric(
            "super_glue", self.hparams.task_name, experiment_id=self.hparams.experiment_id
        )
        self.no_decay = ["bias", "LayerNorm.weight"]

    @property
    def finetuningscheduler_callback(self) -> Optional[FinetuningScheduler]:  # type: ignore
        fts = [c for c in self.trainer.callbacks if isinstance(c, FinetuningScheduler)]  # type: ignore
        return fts[0] if fts else None

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_start(self) -> None:
        if self.finetuningscheduler_callback:
            self.logger.log_metrics(
                metrics={"finetuning_schedule_depth": float(self.finetuningscheduler_callback.curr_depth)},
                step=self.global_step,
            )

    def training_epoch_end(self, outputs: List[Any]) -> None:
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        if self.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        self.log("val_loss", val_loss, prog_bar=True)
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metric_dict, prog_bar=True)

    def _init_param_groups(self) -> List[Dict]:
        """Initialize the parameter groups. Used to ensure weight_decay is not applied to our specified bias
        parameters when we initialize the optimizer.

        Returns:
            List[Dict]: A list of parameter group dictionaries.
        """
        return [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in self.no_decay) and p.requires_grad
                ],
                "weight_decay": self.hparams.optimizer_init["init_args"]["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in self.no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self):
        # the phase 0 parameters will have been set to require gradients during setup
        # you can initialize the optimizer with a simple requires.grad filter as is often done,
        # but in this case we pass a list of parameter groups to ensure weight_decay is
        # not applied to the bias parameter (for completeness, in this case it won't make much
        # performance difference)
        optimizer = instantiate_registered_class(args=self._init_param_groups(), init=self.hparams.optimizer_init)
        scheduler = {
            "scheduler": instantiate_registered_class(args=optimizer, init=self.hparams.lr_scheduler_init),
            **self.hparams.pl_lrs_cfg,
        }
        return [optimizer], [scheduler]


class CustLightningCLI(LightningCLI):
    """Customize the :class:`~pytorch_lightning.utilities.cli.LightningCLI` to ensure the
    :class:`~pytorch_lighting.core.LightningDataModule` and :class:`~pytorch_lightning.core.lightning.LightningModule`
    use the same Hugging Face model, SuperGLUE task and custom logging tag."""

    def before_instantiate_classes(self) -> None:
        # fix needed for pl 1.6.0 (patched w/ https://github.com/PyTorchLightning/pytorch-lightning/pull/12609)
        deprecated_keys = ["agg_key_funcs", "agg_default_func"]
        target_namespace = self.config.fit.trainer.logger.init_args
        for k in deprecated_keys:
            if k in target_namespace.__dict__:
                delattr(target_namespace, k)

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.logger.init_args.name", "model.init_args.experiment_tag")
        parser.link_arguments("data.init_args.model_name_or_path", "model.init_args.model_name_or_path")
        parser.link_arguments("data.init_args.task_name", "model.init_args.task_name")


def cli_main() -> None:
    if not _HF_AVAILABLE:  # pragma: no cover
        print("Running the fts_superglue example requires the `transformers` and `datasets` packages from Hugging Face")
        return
    if not _SP_AVAILABLE:
        print("Note using the default model in this fts_superglue example requires the `sentencepiece` package.")
    # every configuration of this example depends upon a shared set of defaults.
    default_config_file = os.path.join(os.path.dirname(__file__), "config", "fts_defaults.yaml")
    _ = CustLightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_overwrite=True,
        parser_kwargs={"fit": {"default_config_files": [default_config_file]}},
    )


if __name__ == "__main__":
    cli_main()
