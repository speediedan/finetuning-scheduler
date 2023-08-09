import operator
import os
import sys
from collections import namedtuple
from typing import Any, Dict, Optional, Tuple, Union

import torch
from lightning.fabric.accelerators.cuda import is_cuda_available
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_13
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning_utilities.core.imports import compare_version
from torch.utils import collect_env

_TORCH_GREATER_EQUAL_1_12_1 = compare_version("torch", operator.ge, "1.12.1")
_JSONGARGPARSE_GREATER_EQUAL_4_23_1 = compare_version("jsonargparse", operator.ge, "4.23.1")


class CustLightningCLI(LightningCLI):
    """Customize the :class:`~lightning.pytorch.cli.LightningCLI` to ensure the
    :class:`~pytorch_lighting.core.LightningDataModule` and :class:`~lightning.pytorch.core.module.LightningModule`
    use the same Hugging Face model, SuperGLUE task and custom logging tag."""

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.logger.init_args.name", "model.init_args.experiment_tag")
        parser.link_arguments("data.init_args.model_name_or_path", "model.init_args.model_name_or_path")
        parser.link_arguments("data.init_args.task_name", "model.init_args.task_name")


if _JSONGARGPARSE_GREATER_EQUAL_4_23_1:
    PatchedFSDPStrategy = FSDPStrategy
else:
    # patch of FSDPStrategy to avoid https://github.com/omni-us/jsonargparse/issues/337 with `jsonargparse`` < 4.23.1
    class PatchedFSDPStrategy(FSDPStrategy):
        def __init__(
            self,
            activation_checkpointing_policy: Optional[Any] = None,
            auto_wrap_policy: Optional[Any] = None,
            cpu_offload: Optional[Any] = None,
            *args,
            **kwargs,
        ):
            super().__init__(
                activation_checkpointing_policy=activation_checkpointing_policy,
                auto_wrap_policy=auto_wrap_policy,
                cpu_offload=cpu_offload,
                *args,
                **kwargs,
            )


def instantiate_class(init: Dict[str, Any], args: Optional[Union[Any, Tuple[Any, ...]]] = None) -> Any:
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

    if collect_env.TORCH_AVAILABLE:
        version_str = torch.__version__
        debug_mode_str = str(torch.version.debug)
        cuda_available_str = str(is_cuda_available())
        cuda_version_str = torch.version.cuda
        if not hasattr(torch.version, "hip") or torch.version.hip is None:  # cuda version
            hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"
        else:  # HIP version
            cfg = torch._C._show_config().split("\n")
            hip_runtime_version = [s.rsplit(None, 1)[-1] for s in cfg if "HIP Runtime" in s][0]
            miopen_runtime_version = [s.rsplit(None, 1)[-1] for s in cfg if "MIOpen" in s][0]
            cuda_version_str = "N/A"
            hip_compiled_version = torch.version.hip
    else:
        version_str = debug_mode_str = cuda_available_str = cuda_version_str = "N/A"
        hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"

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
        "cpu_info": collect_env.get_cpu_info(run_lambda),
    }
    if _TORCH_GREATER_EQUAL_1_13:
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
    pip_dict = {name: ver for name, ver in [p.split("==") for p in sys_info._asdict()["pip_packages"].split("\n")]}
    sys_dict["pip_packages"] = pip_dict
    return sys_dict
