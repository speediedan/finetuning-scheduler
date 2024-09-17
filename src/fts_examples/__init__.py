from lightning_utilities.core.imports import module_available

_HF_AVAILABLE = module_available("transformers") and module_available("datasets")
_SP_AVAILABLE = module_available("sentencepiece")

from fts_examples.patching.dep_patch_shim import _ACTIVE_PATCHES  # noqa: E402, F401
