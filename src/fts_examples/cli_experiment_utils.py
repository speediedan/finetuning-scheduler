import os
import sys
from collections import namedtuple
from typing import Any, Dict
from datetime import datetime

import torch
from torch.utils import collect_env
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.fabric.utilities import rank_zero_warn
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT

from finetuning_scheduler.types import FTSLRSchedulerTypeTuple
from fts_examples.model_parallel.torchtitan_llama import ModelCfg
from fts_examples.profiling import MemProfiler, MemProfilerCfg, ProfilerHooksMixin
from fts_examples.cfg_utils import LightningLRSCfg, OptimizerCfg, LRSchedulerCfg, ExperimentCfg


class CustLightningCLI(LightningCLI):
    """Customize the :class:`~lightning.pytorch.cli.LightningCLI` to ensure the
    :class:`~pytorch_lighting.core.LightningDataModule` and :class:`~lightning.pytorch.core.module.LightningModule`
    use the same Hugging Face model, SuperGLUE task and custom logging tag."""

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.logger.init_args.name", "model.init_args.experiment_tag")
        parser.link_arguments("data.init_args.model_name_or_path", "model.init_args.model_name_or_path")
        parser.link_arguments("data.init_args.task_name", "model.init_args.task_name")


def instantiate_class(init: dict[str, Any], args: Any | tuple[Any, ...] | None = None) -> Any:
    """Instantiates a class with the given args and init. Accepts class definitions with a "class_path".

    Args:
        init: Dict of the form {"class_path":..., "init_args":...}.
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
    else:
        raise MisconfigurationException("A class_path was not included in a configuration that requires one")
    if not shortcircuit_local:
        module = __import__(class_module, fromlist=[class_name])
        args_class = getattr(module, class_name)
    return args_class(**kwargs) if not args else args_class(*args, **kwargs)


# override PyTorch default, extending it to capture additional salient packages for reproducability
# https://github.com/pytorch/pytorch/blob/7c2489bdae5a96dc122c3bb7b42c18528bcfdc86/torch/utils/collect_env.py#L271
def get_pip_packages(run_lambda):
    """Returns `pip list` output."""
    # People generally have `pip` as `pip` or `pip3`
    # But here it is invoked as `python -mpip`
    def run_with_pip(pip):
        if collect_env.get_platform() == "win32":
            system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
            findstr_cmd = os.path.join(system_root, "System32", "findstr")
            grep_cmd = rf'{findstr_cmd} /R "numpy torch mypy transformers datasets"'
        else:
            grep_cmd = r'grep "torch\|numpy\|mypy\|transformers\|datasets"'
        return collect_env.run_and_read_all(run_lambda, pip + " list --format=freeze | " + grep_cmd)

    pip_version = "pip3" if sys.version[0] == "3" else "pip"
    out = run_with_pip(sys.executable + " -mpip")

    return pip_version, out


def get_env_info():
    run_lambda = collect_env.run
    pip_version, pip_list_output = get_pip_packages(run_lambda)

    version_str = torch.__version__
    debug_mode_str = str(torch.version.debug)
    cuda_available_str = str(torch.cuda.is_available())
    cuda_version_str = torch.version.cuda
    xpu_available_str = str(torch.xpu.is_available())
    if torch.xpu.is_available():
        xpu_available_str = (
            f"{xpu_available_str}\n"
            + f"XPU used to build PyTorch: {torch.version.xpu}\n"
            + f"Intel GPU driver version:\n{collect_env.get_intel_gpu_driver_version(run_lambda)}\n"
            + f"Intel GPU models onboard:\n{collect_env.get_intel_gpu_onboard(run_lambda)}\n"
            + f"Intel GPU models detected:\n{collect_env.get_intel_gpu_detected(run_lambda)}"
        )
    if (
        not hasattr(torch.version, "hip") or torch.version.hip is None
    ):  # cuda version
        hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"
    else:  # HIP version

        def get_version_or_na(cfg, prefix):
            _lst = [s.rsplit(None, 1)[-1] for s in cfg if prefix in s]
            return _lst[0] if _lst else "N/A"

        cfg = torch._C._show_config().split("\n")
        hip_runtime_version = get_version_or_na(cfg, "HIP Runtime")
        miopen_runtime_version = get_version_or_na(cfg, "MIOpen")
        cuda_version_str = "N/A"
        hip_compiled_version = torch.version.hip

    sys_version = sys.version.replace("\n", " ")

    systemenv_kwargs = {
        "torch_version": version_str,
        "is_debug_build": debug_mode_str,
        "python_version": f"{sys_version} ({sys.maxsize.bit_length() + 1}-bit runtime)",
        "python_platform": collect_env.get_python_platform(),
        "is_cuda_available": cuda_available_str,
        "cuda_compiled_version": cuda_version_str,
        "cuda_runtime_version": collect_env.get_running_cuda_version(run_lambda),
        "nvidia_gpu_models": collect_env.get_gpu_info(run_lambda),
        "nvidia_driver_version": collect_env.get_nvidia_driver_version(run_lambda),
        "cudnn_version": collect_env.get_cudnn_version(run_lambda),
        "hip_compiled_version": hip_compiled_version,
        "hip_runtime_version": hip_runtime_version,
        "miopen_runtime_version": miopen_runtime_version,
        "pip_version": pip_version,
        "pip_packages": pip_list_output,
        "conda_packages": collect_env.get_conda_packages(run_lambda),
        "os": collect_env.get_os(run_lambda),
        "libc_version": collect_env.get_libc_version(),
        "gcc_version": collect_env.get_gcc_version(run_lambda),
        "clang_version": collect_env.get_clang_version(run_lambda),
        "cmake_version": collect_env.get_cmake_version(run_lambda),
        "caching_allocator_config": collect_env.get_cachingallocator_config(),
        "is_xnnpack_available": collect_env.is_xnnpack_available(),
        "is_xpu_available": xpu_available_str,
        "cpu_info": collect_env.get_cpu_info(run_lambda),
    }
    # get_cuda_module_loading_config() initializes CUDA which we want to avoid so we bypass this inspection
    systemenv_kwargs["cuda_module_loading"] = "not inspected"
    return collect_env.SystemEnv(**systemenv_kwargs)


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
    sys_info = get_env_info()
    sys_dict = sys_info._asdict()
    # TODO: since we now use uv via the pip interface, we should consider adding uv pip package versions here if torch
    # does not start doing so soon
    pip_packages = sys_dict.get("pip_packages")
    if pip_packages:
        pip_dict = {name: ver for name, ver in [p.split("==", 1) for p in pip_packages.split("\n") if "==" in p]}
        sys_dict["pip_packages"] = pip_dict
    else:
        sys_dict["pip_packages"] = {}
    return sys_dict


class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, seq_length: int, dataset_length: int = 64):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.dataset_length = dataset_length
        self.tokens = torch.randint(
            self.vocab_size,
            size=(len(self), self.seq_length + 1),
            # Set a seed to make this toy dataset the same on each rank
            # Fabric will add a `DistributedSampler` to shard the data correctly
            generator=torch.Generator().manual_seed(42),
        )

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, item: int):
        return self.tokens[item]

class ExpHarness(ProfilerHooksMixin, L.LightningModule):
    def __init__(self, model_cfg: ModelCfg, exp_cfg: ExperimentCfg,
                 optimizer_cfg: OptimizerCfg | None = None,
                 lr_scheduler_cfg: LRSchedulerCfg | None = None,
                 lightning_lrs_cfg: LightningLRSCfg | None = None,
                 memprofiler_cfg: MemProfilerCfg | None = None,
                 *args, **kwargs):
        super().__init__(memprofiler_cfg=memprofiler_cfg, *args, **kwargs)
        self.init_hparams = {
            "model_cfg": model_cfg,
            "exp_cfg": exp_cfg,
            "optimizer_cfg": optimizer_cfg,
            "lr_scheduler_cfg": lr_scheduler_cfg,
            "lightning_lrs_cfg": lightning_lrs_cfg,
            "memprofiler_cfg": memprofiler_cfg,
            "loss_parallel": exp_cfg.loss_parallel,
            "experiment_id": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ExperimentCfg.experiment_tag}",
        }
        self.init_hparams["env_info"] = collect_env_info() if ExperimentCfg.log_env_details else None
        self.save_hyperparameters(self.init_hparams)

    def setup(self, stage):
        super().setup(stage)
        self.dataset = RandomTokenDataset(vocab_size=self.hparams.model_cfg.vocab_size, seq_length=128,
                                          dataset_length=self.hparams.exp_cfg.dataset_length)

    @MemProfiler.memprofilable
    def training_step(self, batch):
        inputs = batch[:, :-1]
        labels = batch[:, 1:]
        output = self.model(inputs)
        loss = self.loss_fn(output, labels)
        self.log("loss", loss.item(), prog_bar=False, sync_dist=True)
        return loss

    @MemProfiler.memprofilable
    def validation_step(self, batch: torch.Tensor) -> STEP_OUTPUT | None:
        inputs = batch[:, :-1]
        labels = batch[:, 1:]
        output = self.model(inputs)
        loss = self.loss_fn(output, labels)
        self.log("val_loss", loss.item(), prog_bar=False, sync_dist=True)
        return {"val_loss": loss}

    def get_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.hparams.exp_cfg.batch_size,
                          num_workers=self.hparams.exp_cfg.num_workers, *args, **kwargs)

    train_dataloader = get_dataloader
    val_dataloader = get_dataloader


class FTSExperimentCLI(LightningCLI):
    """Customize the :class:`~lightning.pytorch.cli.LightningCLI` to facilitate various FTS experiments."""

    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args((Optimizer,))
        parser.add_lr_scheduler_args(FTSLRSchedulerTypeTuple)
        for nested_key, datacls in [('lightning_lrs_cfg', LightningLRSCfg), ('memprofiler_cfg', MemProfilerCfg)]:
            self.add_exp_harness_args_to_parser(parser, nested_key, datacls)
        parser.link_arguments("trainer.logger.init_args.name", "model.init_args.experiment_tag")
        # we collect additional optional `lightning_lrs_cfg` arguments in case the user wants to override the
        # `LRSchedulerConfig` default values (requires the user provide a `configure_optimizers`` method, not supported
        # by LightningCLI's auto_configure_optimizers option)

    def add_exp_harness_args_to_parser(self, parser, nested_key, datacls):
        kwargs: dict[str, Any] = {"instantiate": False, "fail_untyped": False, 'required': False}
        parser.add_dataclass_arguments(datacls, nested_key, **kwargs)
        parser.link_arguments(nested_key, f"model.init_args.{nested_key}")

    def _add_configure_optimizers_method_to_model(self, subcommand: str | None) -> None:
        if self.auto_configure_optimizers:
            super()._add_configure_optimizers_method_to_model(subcommand)
            lightning_lrs_cfg = self._get(self.config_init[self.subcommand], "lightning_lrs_cfg")
            if lightning_lrs_cfg is not None and getattr(lightning_lrs_cfg, "_overridden", False):
                rank_zero_warn("It appears you are using LightningCLI's `auto_configure_optimizers` feature which does"
                    " not support providing a `lightning_lrs_cfg`. If you would like to override "
                    " Lightning's `LRSchedulerConfig` defaults with a `LightningLRSCfg`, please set"
                    " LightningCLI's `auto_configure_optimizers` to `False` and provide a "
                    " `configure_optimizers` method instead.")

    def before_instantiate_classes(self) -> None:
        # since we are using CLI's auto `configure_optimizers` option, the model doesn't require access to the
        # optimizer and lr_scheduler configurations directly. We make them optional in the experimental harness and
        # pass them in via this args so that we have the option to save them with other hyperparameters as it's useful
        # for various experiment logging visualizations.
        def convert_reserved_keys(d: dict[str, Any]) -> dict[str, Any]:
            """Transform configuration dataclass args to avoid using `class_path` or `init_args` keys as
            jsonargparse reserves those names."""
            return {
                "class_fqn" if k == "class_path" else "args" if k == "init_args" else k: v
                for k, v in d.items()
            }
        config_root = self.config[self.subcommand]
        config_root.model.init_args.optimizer_cfg = convert_reserved_keys(config_root.optimizer.as_dict())
        config_root.model.init_args.lr_scheduler_cfg = convert_reserved_keys(config_root.lr_scheduler.as_dict())
        return super().before_instantiate_classes()
