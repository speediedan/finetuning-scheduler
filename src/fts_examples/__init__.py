import importlib
from importlib.util import find_spec


def _module_available(module_path: str) -> bool:
    """Whether ``module_path`` can be imported (as ``lightning_utilities``' ``module_available`` decides, without
    depending on it, so the Lightning-free examples run without Lightning installed)."""
    try:
        if find_spec(module_path.split(".")[0]) is None:
            return False
        importlib.import_module(module_path)
    except ImportError:
        return False
    return True


_HF_AVAILABLE = _module_available("transformers") and _module_available("datasets")
_SP_AVAILABLE = _module_available("sentencepiece")

from fts_examples.patching.dep_patch_shim import _ACTIVE_PATCHES  # noqa: E402, F401
