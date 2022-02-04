from pytorch_lightning.utilities import _module_available

_HF_AVAILABLE = _module_available("transformers") and _module_available("datasets")
_SP_AVAILABLE = _module_available("sentencepiece")
