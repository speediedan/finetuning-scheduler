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
import sys
import warnings
from collections import namedtuple
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cli import _Registry, CALLBACK_REGISTRY, LightningCLI
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils import collect_env
from torch.utils.data import DataLoader

from finetuning_scheduler.fts import FinetuningScheduler
from fts_examples import _HF_AVAILABLE, _SP_AVAILABLE

if _HF_AVAILABLE:
    import datasets
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from transformers import logging as transformers_logging

TASK_NUM_LABELS = {"boolq": 2, "rte": 2}
DEFAULT_TASK = "rte"
MOCK_REGISTRY = _Registry()


def instantiate_registered_class(init: Dict[str, Any], args: Optional[Union[Any, Tuple[Any, ...]]] = None) -> Any:
    """Instantiates a class with the given args and init. Accepts class definitions in the form of a "class_path"
    or "callback_key" associated with a _Registry.

    Args:
        init: Dict of the form {"class_path":... or "callback_key":..., "init_args":...}.
        args: Positional arguments required for instantiation.

    Returns:
        The instantiated class object.
    """
    class_module, class_name, args_class = None, None, None
    shortcircuit_local = False
    kwargs = init.get("init_args", {})
    class_path = init.get("class_path", None)
    if args and not isinstance(args, tuple):
        args = (args,)
    if class_path:
        shortcircuit_local = False if "." in class_path else True
        if not shortcircuit_local:
            class_module, class_name = init["class_path"].rsplit(".", 1)
        else:  # class is expected to be locally defined
            args_class = globals()[init["class_path"]]
    elif init.get("callback_key", None):
        callback_path = CALLBACK_REGISTRY.get(init["callback_key"], None) or MOCK_REGISTRY.get(
            init["callback_key"], None
        )
        assert callback_path, MisconfigurationException(
            f'specified callback_key {init["callback_key"]} has not been registered'
        )
        class_module, class_name = callback_path.__module__, callback_path.__name__
    else:
        raise MisconfigurationException(
            "Neither a class_path nor callback_key were included in a configuration that" "requires one"
        )
    if not shortcircuit_local:
        module = __import__(class_module, fromlist=[class_name])
        args_class = getattr(module, class_name)
    return args_class(**kwargs) if not args else args_class(*args, **kwargs)


# override PyTorch default, extending it to capture additional salient packages for reproducability
# https://github.com/pytorch/pytorch/blob/7c2489bdae5a96dc122c3bb7b42c18528bcfdc86/torch/utils/collect_env.py#L271
def get_pip_packages(run_lambda):
    """Returns `pip list` output.

    Note: will also find conda-installed pytorch
    and numpy packages.
    """
    # People generally have `pip` as `pip` or `pip3`
    # But here it is incoved as `python -mpip`
    def run_with_pip(pip):
        if collect_env.get_platform() == "win32":
            system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
            findstr_cmd = os.path.join(system_root, "System32", "findstr")
            grep_cmd = fr'{findstr_cmd} /R "numpy torch mypy transformers datasets"'
        else:
            grep_cmd = r'grep "torch\|numpy\|mypy\|transformers\|datasets"'
        return collect_env.run_and_read_all(run_lambda, pip + " list --format=freeze | " + grep_cmd)

    pip_version = "pip3" if sys.version[0] == "3" else "pip"
    out = run_with_pip(sys.executable + " -mpip")

    return pip_version, out


def collect_env_info() -> Dict:
    """Collect environmental details, logging versions of salient packages for improved reproducibility.

    Returns:
        Dict: The dictionary of environmental details
    """
    _ = namedtuple(
        "SystemEnv",
        [
            "torch_version",
            "is_debug_build",
            "cuda_compiled_version",
            "gcc_version",
            "clang_version",
            "cmake_version",
            "os",
            "libc_version",
            "python_version",
            "python_platform",
            "is_cuda_available",
            "cuda_runtime_version",
            "nvidia_driver_version",
            "nvidia_gpu_models",
            "cudnn_version",
            "pip_version",  # 'pip' or 'pip3'
            "pip_packages",
            "conda_packages",
            "hip_compiled_version",
            "hip_runtime_version",
            "miopen_runtime_version",
            "caching_allocator_config",
        ],
    )
    collect_env.get_pip_packages = get_pip_packages
    sys_info = collect_env.get_env_info()
    sys_dict = sys_info._asdict()
    pip_dict = {name: ver for name, ver in [p.split("==") for p in sys_info._asdict()["pip_packages"].split("\n")]}
    sys_dict["pip_packages"] = pip_dict
    return sys_dict


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
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name if task_name in TASK_NUM_LABELS.keys() else DEFAULT_TASK
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizers_parallelism = tokenizers_parallelism
        self.dataloader_kwargs = {
            "num_workers": dataloader_kwargs.get("num_workers", 0),
            "pin_memory": dataloader_kwargs.get("pin_memory", False),
        }
        self.text_fields = self.TASK_TEXT_FIELD_MAP[self.task_name]
        self.num_labels = TASK_NUM_LABELS[self.task_name]
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if self.tokenizers_parallelism else "false"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, local_files_only=False)

    def setup(self, stage):
        self.dataset = datasets.load_dataset("super_glue", self.task_name)
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._convert_to_features, batched=True, remove_columns=["label"]
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.LOADER_COLUMNS]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        # N.B. PL calls prepare_data from a single process (rank 0) so do not use it to assign state (e.g. self.x=y)
        datasets.load_dataset("super_glue", self.task_name)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, **self.dataloader_kwargs)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, **self.dataloader_kwargs)
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size, **self.dataloader_kwargs)
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, **self.dataloader_kwargs)
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size, **self.dataloader_kwargs)
                for x in self.eval_splits
            ]

    def _convert_to_features(self, example_batch):
        text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            text_pairs, max_length=self.max_seq_length, padding="longest", truncation=True
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
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init
        self.pl_lrs_cfg = pl_lrs_cfg or {}
        if task_name in TASK_NUM_LABELS.keys():
            self.task_name = task_name
        else:
            self.task_name = DEFAULT_TASK
            rank_zero_warn(f"Invalid task_name '{task_name}'. Proceeding with the default task: '{DEFAULT_TASK}'")
        self.num_labels = TASK_NUM_LABELS[self.task_name]
        self.experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_tag}"
        self.model_cfg = model_cfg or {}
        conf = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels, local_files_only=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=conf)
        self.model.config.update(self.model_cfg)  # apply model config overrides
        self.init_hparams = {
            "optimizer": self.optimizer_init,
            "lr_scheduler": self.lr_scheduler_init,
            "pl_lrs_cfg": self.pl_lrs_cfg,
            "model_config": self.model.config,
            "model_name_or_path": model_name_or_path,
            "task_name": self.task_name,
            "experiment_id": self.experiment_id,
        }
        self.init_hparams["env_info"] = collect_env_info() if log_env_details else None
        self.save_hyperparameters(self.init_hparams)
        self.metric = datasets.load_metric("super_glue", self.task_name, experiment_id=self.experiment_id)
        self.no_decay = ["bias", "LayerNorm.weight"]
        self.finetuningscheduler_callback = None

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        if self.finetuningscheduler_callback:
            self.log("finetuning_schedule_depth", float(self.finetuningscheduler_callback.curr_depth))

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
                "weight_decay": self.optimizer_init["init_args"]["weight_decay"],
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
        optimizer = instantiate_registered_class(args=self._init_param_groups(), init=self.optimizer_init)
        scheduler = {
            "scheduler": instantiate_registered_class(args=optimizer, init=self.lr_scheduler_init),
            **self.pl_lrs_cfg,
        }
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        found_fts = [c for c in self.trainer.callbacks if isinstance(c, FinetuningScheduler)]
        if found_fts:
            self.finetuningscheduler_callback = found_fts[0]
        return super().configure_callbacks()


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
