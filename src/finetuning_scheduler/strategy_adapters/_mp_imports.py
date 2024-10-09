# TODO: replace local version once Lightning version available
# from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_5
from lightning_utilities.core.imports import compare_version
import operator
_TORCH_GREATER_EQUAL_2_5 = compare_version("torch", operator.ge, "2.5.0", use_base_version=True)

# ruff: noqa: F401
# we require torch 2.5 or higher for composable distributed API support so until torch 2.5.0 is the minimum version,
# supported, we conditionally import indirectly to avoid duplicating import logic in several different modules
if _TORCH_GREATER_EQUAL_2_5:
    from torch.distributed._composable import checkpoint
    from torch.distributed._composable.fsdp._fsdp_api import CPUOffloadPolicy
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Replicate, Shard
    from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
    from torch.distributed.tensor.experimental import implicit_replication
    from torch.distributed._composable.fsdp import FSDPModule, fully_shard
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (checkpoint_wrapper, offload_wrapper,
                                                                             ActivationWrapper)
    from torch.distributed.tensor.parallel import (ColwiseParallel, PrepareModuleInput, RowwiseParallel,
                                                   SequenceParallel, parallelize_module, loss_parallel)
else:
    for mp_obj in ["SDPBackend", "DeviceMesh", "DTensor", "Replicate", "Shard", "ColwiseParallel", "PrepareModuleInput",
                "RowwiseParallel", "SequenceParallel", "implicit_replication", "parallelize_module", "loss_parallel",
                "FSDPModule", "fully_shard", "checkpoint", "checkpoint_wrapper", "offload_wrapper", "ActivationWrapper",
                "CPUOffloadPolicy", "sdpa_kernel", "FSDPMemTracker"]:
        globals()[mp_obj] = None
