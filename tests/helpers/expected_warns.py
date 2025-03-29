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
"""Central location for warning definitions used across tests."""
from copy import copy
from itertools import chain

# warning inventory last evaluated w/ PT 2.7.0.dev20250201 and Lightning commit efe311cd (2.6.0dev)

WORKER_WARNING = "does not have many workers"

BASE_WARNINGS = [
    "GPU available but",
    "`max_epochs` was not",
    "The dirpath has changed from",
    "Conversion of an array with ndim > 0",  # still needed with python 3.9 and torch 2.4.0
    WORKER_WARNING,
]

def create_base_warns():
    return BASE_WARNINGS

def extend_warns(base_warns, additional_warns):
    """Create a new warning list by extending base warnings with additional ones."""
    warns = copy(base_warns)
    warns.extend(additional_warns)
    return warns

BASE_EXPECTED_WARNS = create_base_warns()

EXPECTED_DIRPATH_WARN = "is not empty."
EXPECTED_TRAINCHK_WARN = "could not find the monitored key in the returned"
EXPECTED_CKPT_WARNS = ["Be aware that when using `ckpt_path`, callbacks"]
OPTIMIZER_INIT_WARNS = ["currently depends upon", "No monitor metric specified"]

# Dynamo warnings
DYNAMO_PHASE_WARNING = "Final phase max_transition_epoch"
DYNAMO_TENSOR_WARNING = "TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled."
BASE_DYNAMO_EXPECTED_WARNS = [
    DYNAMO_PHASE_WARNING,
    # DYNAMO_TENSOR_WARNING
]

# FSDP specific warnings
additional_fsdp_warns = [
    "The number of training batches",
    "when logging on epoch level in distributed",
    "FSDP.state_dict_type",
    "of Tensor.pin_memory",
    "Tensor.is_pinned",
    "Deallocating Tensor ",
    "`_get_pg_default_device` will be deprecated"
]
FSDP_BASE_WARNS = extend_warns(BASE_EXPECTED_WARNS, additional_fsdp_warns)
FSDP_DYNAMO_EXPECTED_WARNS = [DYNAMO_PHASE_WARNING]

# Model parallel warnings
additional_model_parallel_warns = [
    "The number of training batches",
    "when logging on epoch level in distributed",
    #"You are using `torch.load` with `weights_only=False`",
    "of Tensor.pin_memory",
    "of Tensor.is_pinned"
]
MODEL_PARALLEL_BASE_WARNS = extend_warns(BASE_EXPECTED_WARNS, additional_model_parallel_warns)
MODEL_PARALLEL_DYNAMO_EXPECTED_WARNS = []

# Example warnings
EXAMPLE_BASE_WARNS = [
    "sentencepiece tokenizer that you are converting",
    "co_lnotab is deprecated, use co_lines instead.",
    #"`resume_download` is deprecated",
    #"distutils Version classes are deprecated",
    #"Please use torch.utils._pytree.register_pytree_node",
    #"We are importing from `pydantic`",
    #"`TorchScript` support for functional optimizers is",
    #"`is_compiling` is deprecated",
    #"You are using `torch.load` with `weights_only=False`",
    #"is multi-threaded, use of fork"
]

ALL_EXAMPLE_EXPECTED = [
    "Use timezone-aware",
    "is smaller than the logging interval",
    WORKER_WARNING,
]
additional_advanced_lrs_warns = ["Found an `init_pg_lrs` key"]
EXAMPLE_EXPECTED_WARNS = extend_warns(extend_warns(BASE_EXPECTED_WARNS, EXAMPLE_BASE_WARNS), ALL_EXAMPLE_EXPECTED)
MODEL_PARALLEL_EXAMPLE_WARNS = extend_warns(MODEL_PARALLEL_BASE_WARNS, ALL_EXAMPLE_EXPECTED)
ADV_EXAMPLE_EXPECTED_WARNS = extend_warns(EXAMPLE_EXPECTED_WARNS, additional_advanced_lrs_warns)

def print_warns(example_warns=False):
    """Print all relevant warnings either from example or non-example categories.

    Args:
        example_warns (bool): If True, print example warns, otherwise print non-example warns
    """
    if example_warns:
        # we only want to print warnings unique to the example context (since other contexts are managed by base lists)
        all_warns = set(chain(EXAMPLE_BASE_WARNS, ALL_EXAMPLE_EXPECTED, additional_advanced_lrs_warns))
    else:
        all_warns = set(chain(BASE_EXPECTED_WARNS, BASE_DYNAMO_EXPECTED_WARNS, FSDP_BASE_WARNS,
                              FSDP_DYNAMO_EXPECTED_WARNS, MODEL_PARALLEL_BASE_WARNS,
                              MODEL_PARALLEL_DYNAMO_EXPECTED_WARNS))

    for warn in sorted(all_warns):
        print(f'"{warn}"')
