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
# Initially based on https://bit.ly/3J5oOk4
import os
import re
import sys
from typing import Optional, Set, Union, Dict
from packaging.version import Version
import importlib.metadata as metadata

import pytest
import torch
from lightning.fabric.accelerators.cuda import num_cuda_devices
from lightning.pytorch.strategies.deepspeed import _DEEPSPEED_AVAILABLE

from fts_examples.patching.dep_patch_shim import ExpPatch, _ACTIVE_PATCHES

EXTENDED_VER_PAT = re.compile(r"([0-9]+\.){2}[0-9]+")

def maybe_mark_exp(exp_patch_set: Set[ExpPatch], mark_if_false: Optional[Dict] = None):
    """This allows us to evaluate whether an experimental patch set that is conditionally required for a given test
    is required in the current execution context.

    If the experimental patch set is not required, we mark the
    test with the provided `mark_if_false` dictionary directive (or an empty dictionary).
    """

    exp_patch_set = {ep for ep in exp_patch_set if all(ep.value.condition)}
    if any(exp_patch_set):
        return {"exp_patch": exp_patch_set}
    else:
        return mark_if_false or {}

# RunIf aliases
RUNIF_MAP = {
    "alone": {"standalone": True},
    "bf16_alone": {"bf16_cuda": True, "standalone": True},
}


class RunIf:
    """RunIf wrapper for simple marking specific cases, fully compatible with pytest.mark::

    @RunIf(min_torch="0.0")
    @pytest.mark.parametrize("arg1", [1, 2.0])
    def test_wrapper(arg1):
        assert arg1 > 0.0
    """

    standalone_ctx = os.getenv("PL_RUN_STANDALONE_TESTS", "0")

    def __new__(
        self,
        *args,
        min_cuda_gpus: int = 0,
        min_torch: Optional[str] = None,
        max_torch: Optional[str] = None,
        min_python: Optional[str] = None,
        max_python: Optional[str] = None,
        bf16_cuda: bool = False,
        skip_windows: bool = False,
        skip_mac_os: bool = False,
        standalone: bool = False,
        deepspeed: bool = False,
        exp_patch: Optional[Union[ExpPatch, Set[ExpPatch]]] = None,
        **kwargs,
    ):
        """
        Args:
            *args: Any :class:`pytest.mark.skipif` arguments.
            min_cuda_gpus: Require this number of gpus and that the ``PL_RUN_CUDA_TESTS=1`` environment variable is set.
            min_torch: Require that PyTorch is greater or equal to this version.
            max_torch: Require that PyTorch is less than or equal to this version.
            min_python: Require that Python is greater or equal to this version.
            max_python: Require that Python is less than this version.
            bf16_cuda: Require that CUDA device supports bf16.
            skip_windows: Skip for Windows platform.
            skip_mac_os: Skip Mac OS platform.
            standalone: Mark the test as standalone, our CI will run it in a separate process.
                This requires that the ``PL_RUN_STANDALONE_TESTS=1`` environment variable is set.
            deepspeed: Require that microsoft/DeepSpeed is installed.
            exp_patch: Require that a given experimental patch is installed.
            **kwargs: Any :class:`pytest.mark.skipif` keyword arguments.
        """
        conditions = []
        reasons = []

        if min_cuda_gpus:
            cuda_device_fx = num_cuda_devices if RunIf.standalone_ctx == "1" else torch.cuda.device_count
            conditions.append(cuda_device_fx() < min_cuda_gpus)
            reasons.append(f"GPUs>={min_cuda_gpus}")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["min_cuda_gpus"] = True

        if min_torch:
            torch_version = metadata.distribution('torch').version
            extended_torch_ver = EXTENDED_VER_PAT.match(torch_version).group() or torch_version
            conditions.append(Version(extended_torch_ver) < Version(min_torch))
            reasons.append(f"torch>={min_torch}, {extended_torch_ver} installed.")

        if max_torch:
            torch_version = metadata.distribution('torch').version
            extended_torch_ver = EXTENDED_VER_PAT.match(torch_version).group() or torch_version
            conditions.append(Version(extended_torch_ver) > Version(max_torch))
            reasons.append(f"torch<={max_torch}, {extended_torch_ver} installed.")

        if min_python:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if max_python:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            conditions.append(Version(py_version) >= Version(max_python))
            reasons.append(f"python<{max_python}, {py_version} installed.")

        if bf16_cuda:
            try:
                cond = not (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            except (AssertionError, RuntimeError) as e:
                # AssertionError: Torch not compiled with CUDA enabled
                # RuntimeError: Found no NVIDIA driver on your system.
                is_unrelated = "Found no NVIDIA driver" not in str(e) or "Torch not compiled with CUDA" not in str(e)
                if is_unrelated:
                    raise e
                cond = True

            conditions.append(cond)
            reasons.append("CUDA device bf16")

        if skip_windows:
            conditions.append(sys.platform == "win32")
            reasons.append("unimplemented on Windows")

        if skip_mac_os:
            conditions.append(sys.platform == "darwin")
            reasons.append("unimplemented or temporarily bypassing these tests for MacOS")

        if standalone:
            env_flag = os.getenv("PL_RUN_STANDALONE_TESTS", "0")
            conditions.append(env_flag != "1")
            reasons.append("Standalone execution")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["standalone"] = True

        if deepspeed:
            conditions.append(not _DEEPSPEED_AVAILABLE)
            reasons.append("Deepspeed")

        if exp_patch:
            # since we want to ensure we separate all experimental test combinations from normal unpatched tests, we
            # gate experimental patches with both an environmental flag and the required subset of active patches
            env_flag = os.getenv("FTS_EXPERIMENTAL_PATCH_TESTS", "0")
            if env_exp_flag := (env_flag != "1"):
                conditions.append(env_exp_flag)
                reasons.append("Experimental tests not enabled via 'FTS_EXPERIMENTAL_PATCH_TESTS' env variable")
            else:
                if not isinstance(exp_patch, Set):
                    exp_patch = {exp_patch}
                conditions.append(not exp_patch.issubset(_ACTIVE_PATCHES))
                reasons.append(f"Required experimental patch configuration {exp_patch} is not active.")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["exp_patch"] = True

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            *args, condition=any(conditions), reason=f"Requires: [{' + '.join(reasons)}]", **kwargs
        )


@RunIf(min_torch="99")
def test_always_skip():
    exit(1)


@pytest.mark.parametrize("arg1", [0.5, 1.0, 2.0])
@RunIf(min_torch="0.0")
def test_wrapper(arg1: float):
    assert arg1 > 0.0
