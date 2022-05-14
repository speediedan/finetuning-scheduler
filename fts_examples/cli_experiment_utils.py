import os
import sys
from collections import namedtuple
from typing import Any, Dict, Optional, Tuple, Union

from pytorch_lightning.utilities.cli import _Registry, CALLBACK_REGISTRY
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils import collect_env

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
