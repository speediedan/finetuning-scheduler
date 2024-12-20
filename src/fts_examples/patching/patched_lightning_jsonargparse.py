from fts_examples.patching._patch_utils import _prepare_module_ctx
from lightning.pytorch.cli import LightningCLI  # noqa: F401

# we ignore these for the entire file since we're using our global namespace trickeration to patch
# ruff: noqa: F821
# pyright: reportUndefinedVariable=false

globals().update(_prepare_module_ctx('lightning.pytorch.cli', globals()))

def _updated_parse_known_args_patch(self: ArgumentParser, args: Any = None, namespace: Any = None,
                                    intermixed: bool = False) -> tuple[Any, Any]:
    namespace, args = super(ArgumentParser, self)._parse_known_args(args, namespace,
                                                                    intermixed=intermixed)  # type: ignore
    return namespace, args
