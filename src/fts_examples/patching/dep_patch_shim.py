import operator
import sys
import os
from enum import Enum
from typing import NamedTuple, Tuple, Callable
from fts_examples.patching._patch_utils import lwt_compare_version
from lightning.pytorch.cli import _JSONARGPARSE_SIGNATURES_AVAILABLE

class OSEnvToggle(NamedTuple):
    env_var_name: str
    default: str

class DependencyPatch(NamedTuple):
    """Ephemeral dependency patches to conditionally apply to the environment.

    To activate a given patch, all defined `condition` callables must return truthy and the `env_flag` must be set (or
    must default) to '1'
    """
    condition: Tuple[Callable]  # typically a tuple of `lwt_compare_version` to define version dependency
    env_flag: OSEnvToggle  # a tuple defining the environment variable based condition and its default if not set
    function: Callable
    patched_package: str
    description: str


def _dep_patch_repr(self):
    return f'Patch of {self.patched_package}: {self.description})'

DependencyPatch.__repr__ = _dep_patch_repr

# N.B. One needs to ensure they patch all relevant _calling module_ references to patch targets since we usually patch
# after those calling modules have already secured the original (unpatched) references.

def _patch_einsum_strategies():
    from fts_examples.patching.patched_einsum_strategies import gen_einsum_strategies

    # In this case fortunately, we only import/call `gen_einsum_strategies` from
    # `torch.distributed._tensor.ops.matrix_ops`, so only need to patch there.
    target_mod = 'torch.distributed.tensor._ops._matrix_ops'
    sys.modules.get(target_mod).__dict__['gen_einsum_strategies'] = gen_einsum_strategies

def _patch_unsupported_numpy_arrow_extractor():
    from fts_examples.patching.patched_numpyarrowextractor import NumpyArrowExtractor
    # since the TorchFormatter and NumpyFormatter classes are already defined we need to patch both definitions
    # to use our patched `NumpyArrowExtractor`
    for old_mod, stale_ref in zip(['torch_formatter', 'np_formatter'], ['TorchFormatter', 'NumpyFormatter']):
        target_mod = f'datasets.formatting.{old_mod}'
        sys.modules.get(target_mod).__dict__.get(stale_ref).numpy_arrow_extractor = NumpyArrowExtractor


def _patch_triton():
    from fts_examples.patching.patched_triton_jit_fn_init import _new_init
    target_mod = 'triton.runtime.jit'
    sys.modules.get(target_mod).__dict__.get('JITFunction').__init__ = _new_init

def _patch_lightning_jsonargparse():
    from fts_examples.patching.patched_lightning_jsonargparse import _updated_parse_known_args_patch
    target_mod = 'lightning.pytorch.cli'
    sys.modules.get(target_mod).__dict__.get('ArgumentParser')._parse_known_args = _updated_parse_known_args_patch

# TODO: remove once `2.6.0` is minimum
# required for `torch==2.5.x`, TBD wrt subsequent versions though appears fixed in torch `2.6.0` nightlies
einsum_strategies_patch = DependencyPatch(
    condition=(lwt_compare_version("torch", operator.le, "2.5.2"),
               lwt_compare_version("torch", operator.ge, "2.5.0"),),
    env_flag=OSEnvToggle("ENABLE_FTS_EINSUM_STRATEGY_PATCH", default="0"),
    function=_patch_einsum_strategies, patched_package='torch',
    description='Address trivial tp submesh limitation until PyTorch provides upstream fix')

# TODO: remove when min jsonargparse is `4.35.0` or another `2.5.0.postN` lightning release fixes
lightning_jsonargparse_patch = DependencyPatch(
    condition=(sys.version_info >= (3, 12, 8), lwt_compare_version("jsonargparse", operator.lt, "4.35.0"),
               _JSONARGPARSE_SIGNATURES_AVAILABLE),
               env_flag=OSEnvToggle("ENABLE_FTS_LIGHTNING_JSONARGPARSE_PATCH", default="1"),
               function=_patch_lightning_jsonargparse,
               patched_package='lightning',
               description=("For the edge case where `lightning` allows a `jsonargparse` version that doesn't support"
                            " python versions >= `3.12.8`.")
)

# TODO: remove once `datasets==2.21.0` is minimum
datasets_numpy_extractor_patch = DependencyPatch(
    condition=(lwt_compare_version("numpy", operator.ge, "2.0.0"),
               lwt_compare_version("datasets", operator.lt, "2.21.0")),
               env_flag=OSEnvToggle("ENABLE_FTS_NUMPY_EXTRACTOR_PATCH", default="1"),
               function=_patch_unsupported_numpy_arrow_extractor,
               patched_package='datasets',
               description='Adjust `NumpyArrowExtractor` to properly use `numpy` 2.0 copy semantics')

# TODO: remove once `torch 2.5.0` is minimum, only required for `torch==2.4.x`
triton_codgen_patch = DependencyPatch(
    condition=(lwt_compare_version("pytorch-triton", operator.eq, "3.0.0", "45fff310c8"),),
    env_flag=OSEnvToggle("ENABLE_FTS_TRITON_CODEGEN_PATCH", default="1"),
    function=_patch_triton, patched_package='pytorch-triton',
    description='Address `triton` #3564 until PyTorch pins the upstream fix')

class ExpPatch(Enum):
    EINSUM_STRATEGIES = einsum_strategies_patch
    NUMPY_EXTRACTOR = datasets_numpy_extractor_patch
    TRITON_CODEGEN = triton_codgen_patch
    LIGHTNING_JSONARGPARSE = lightning_jsonargparse_patch

_DEFINED_PATCHES = set(ExpPatch)
_ACTIVE_PATCHES = set()

for defined_patch in _DEFINED_PATCHES:
    if all(defined_patch.value.condition) and os.environ.get(defined_patch.value.env_flag.env_var_name,
                                                             defined_patch.value.env_flag.default) == "1":
        defined_patch.value.function()
        _ACTIVE_PATCHES.add(defined_patch)
