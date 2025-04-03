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
# Initially based on: https://bit.ly/34f7gDv
"""Diagnose your system and show basic information.

This server mainly to get detail info for better bug reporting.
"""

import os
import platform
import sys

import torch
import tqdm

sys.path += [os.path.abspath(".."), os.path.abspath(".")]
import lightning.pytorch  # noqa: E402
from lightning.fabric.utilities.imports import _NUMPY_AVAILABLE  # noqa: E402

import finetuning_scheduler  # noqa: E402

LEVEL_OFFSET = "\t"
KEY_PADDING = 20

if _NUMPY_AVAILABLE:
    import numpy

def info_system():
    return {
        "OS": platform.system(),
        "architecture": platform.architecture(),
        "version": platform.version(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def info_cuda():
    return {
        "GPU": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "available": torch.cuda.is_available(),
        "version": torch.version.cuda,
    }


def info_packages():
    return {
        "numpy": numpy.__version__ if _NUMPY_AVAILABLE else 'N/A',
        "pyTorch_version": torch.__version__,
        "pyTorch_git_version": torch.version.git_version,
        "pyTorch_debug": torch.version.debug,
        "pytorch-lightning": lightning.pytorch.__version__,
        "finetuning-scheduler": finetuning_scheduler.__version__,
        "tqdm": tqdm.__version__,
    }


def nice_print(details, level=0):
    lines = []
    for k in sorted(details):
        key = f"* {k}:" if level == 0 else f"- {k}:"
        if isinstance(details[k], dict):
            lines += [level * LEVEL_OFFSET + key]
            lines += nice_print(details[k], level + 1)
        elif isinstance(details[k], (set, list, tuple)):
            lines += [level * LEVEL_OFFSET + key]
            lines += [(level + 1) * LEVEL_OFFSET + "- " + v for v in details[k]]
        else:
            template = "{:%is} {}" % KEY_PADDING
            key_val = template.format(key, details[k])
            lines += [(level * LEVEL_OFFSET) + key_val]
    return lines


def main():
    details = {"System": info_system(), "CUDA": info_cuda(), "Packages": info_packages()}
    lines = nice_print(details)
    text = os.linesep.join(lines)
    print(text)


if __name__ == "__main__":
    main()
