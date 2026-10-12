import os
if not os.getenv("SHOW_EXAMPLE_CPP_WARNS", "0") == "1":
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import warnings

import torch
import torch.nn.functional as F
from torch.distributed.tensor.parallel import loss_parallel
import lightning as L

from fts_examples.cli_experiment_utils import ExpHarness, FTSExperimentCLI, ExperimentCfg
# the TP plan lives in the Lightning-free `parallel_plans` module, shared with `torch_mp_example.py`; it remains
# importable from here
from fts_examples.model_parallel.parallel_plans import apply_tp_plan
from fts_examples.model_parallel.torchtitan_llama import ModelCfg, Transformer

# Lightning ModelParallel still uses `torch.load` with `weights_only=False`
warnings.filterwarnings("ignore", ".*uses the default pickle.*")


class ModParallelExample(ExpHarness, L.LightningModule):
    def __init__(self, model_cfg: ModelCfg, exp_cfg: ExperimentCfg, *args, **kwargs):
        # the ExpHarness mixin just saves our hparams and collects our env info so our experiments can be nicely logged
        super().__init__(model_cfg=model_cfg, exp_cfg=exp_cfg, *args, **kwargs)
        with torch.device("meta"):
            self.model = Transformer(self.hparams.model_cfg)

    def on_train_start(self) -> None:
        self.model.init_weights()
        super().on_train_start()

    def configure_model(self):
        if self.device_mesh["tensor_parallel"].size() > 1:
            # User-defined function that applies the a given TP plan if desired
            apply_tp_plan(self.model, device_mesh=self.device_mesh, loss_parallel=self.hparams.exp_cfg.loss_parallel)

        # Note FTS will apply any module name/pattern-based `fsdp_plan` directives after any preceding Tensor
        # Parallel (like above) or explicit `fully_shard` directives here in `LightningModule.configure_model`. FTS will
        # only apply `fully_shard` to a specified module if it was not already applied to that module, so auto
        # `fsdp_plan` and manual `fully_shard` directives can be composed without conflict.

    def loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.hparams.exp_cfg.loss_parallel:
            with loss_parallel():
                loss = F.cross_entropy(output.reshape(-1, output.size(-1)), target.reshape(-1))
        else:
            loss = F.cross_entropy(output.reshape(-1, output.size(-1)), target.reshape(-1))
        return loss

    def backward(self, *args, **kwargs):
        if self.hparams.exp_cfg.loss_parallel:
            with loss_parallel():
                super().backward(*args, **kwargs)
        else:
            super().backward(*args, **kwargs)


def cli_main() -> None:
    torch.set_float32_matmul_precision("high")
    assert torch.cuda.device_count() >= 2, "This example requires at least 2 GPUs"
    # every configuration of this example depends upon a shared set of defaults.
    default_config_file = os.path.join(os.path.dirname(__file__), "config", "defaults", "fts_mp_example_defaults.yaml")
    _ = FTSExperimentCLI(L.LightningModule, subclass_mode_model=True, save_config_kwargs={"overwrite": True},
        parser_kwargs={"fit": {"default_config_files": [default_config_file]}},
    )

if __name__ == "__main__":
    cli_main()
