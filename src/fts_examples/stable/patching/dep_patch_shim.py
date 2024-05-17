import operator
import sys
from typing import NamedTuple, Tuple, Callable
from fts_examples.stable.patching._patch_utils import lwt_compare_version


class DependencyPatch(NamedTuple):
    """Ephemeral dependency patches to conditionally apply to the environment."""
    condition: Tuple[Callable]
    function: Callable
    patched_package: str
    description: str


def _dep_patch_repr(self):
    return f'Patch of {self.patched_package}: {self.description})'

DependencyPatch.__repr__ = _dep_patch_repr


def _patch_unsupported_numpy_arrow_extractor():
    from fts_examples.stable.patching.patched_numpyarrowextractor import NumpyArrowExtractor
    # since the TorchFormatter and NumpyFormatter classes are already defined we need to patch both definitions
    # to use our patched `NumpyArrowExtractor`
    for old_mod, stale_ref in zip(['torch_formatter', 'np_formatter'], ['TorchFormatter', 'NumpyFormatter']):
        target_mod = f'datasets.formatting.{old_mod}'
        sys.modules.get(target_mod).__dict__.get(stale_ref).numpy_arrow_extractor = NumpyArrowExtractor


def _patch_triton():
    from fts_examples.stable.patching.patched_triton_jit_fn_init import _new_init
    target_mod = 'triton.runtime.jit'
    sys.modules.get(target_mod).__dict__.get('JITFunction').__init__ = _new_init


datasets_numpy_extractor_patch = DependencyPatch((lwt_compare_version("numpy", operator.ge, "2.0.0"),
                                     lwt_compare_version("datasets", operator.le, "2.19.1")),
                                     _patch_unsupported_numpy_arrow_extractor, 'datasets',
                                     'Adjust `NumpyArrowExtractor` to properly use `numpy` 2.0 copy semantics')

triton_codgen_patch = DependencyPatch((lwt_compare_version("pytorch-triton", operator.eq, "3.0.0", "45fff310c8"),),
                                      _patch_triton, 'pytorch-triton',
                                     'Address `triton` #3564 until PyTorch pins the upstream fix')

_DEFINED_PATCHES = {datasets_numpy_extractor_patch, triton_codgen_patch}
_ACTIVE_PATCHES = set()

for defined_patch in _DEFINED_PATCHES:
    if all(defined_patch.condition):
        defined_patch.function()
        _ACTIVE_PATCHES.add(defined_patch)
