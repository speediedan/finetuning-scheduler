from lightning_utilities.core.imports import module_available

_HF_AVAILABLE = module_available("transformers") and module_available("datasets")
_SP_AVAILABLE = module_available("sentencepiece")
