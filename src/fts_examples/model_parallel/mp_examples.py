import os
if not os.getenv("SHOW_EXAMPLE_CPP_WARNS", "0") == "1":
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import warnings

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (ColwiseParallel, PrepareModuleInput, RowwiseParallel,
                                                SequenceParallel, parallelize_module, loss_parallel)
import lightning as L

from fts_examples.cli_experiment_utils import ExpHarness, FTSExperimentCLI, ExperimentCfg
from fts_examples.model_parallel.torchtitan_llama import ModelCfg, Transformer

# Lightning ModelParallel still uses `torch.load` with `weights_only=False`
warnings.filterwarnings("ignore", ".*uses the default pickle.*")


# modified version of https://bit.ly/torchtitan_transformer_tp_plan
def apply_tp_plan(model: Transformer, device_mesh: DeviceMesh, loss_parallel: bool) -> Transformer:
    """Apply parallelism."""
    tp_mesh = device_mesh["tensor_parallel"]

    # 1. Parallelize the pre-transformer block layers
    non_transformerblock_tp_plan = {
        # Parallelize the embedding and shard its outputs (which are the first transformer block's inputs)
        "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1),),
        "norm": SequenceParallel(),  # Parallelize the root norm layer over the sequence dim
    }
    model = parallelize_module(model, tp_mesh, non_transformerblock_tp_plan)

    # 2. Parallelize each transformer block
    for transformer_block in model.layers.values():
        plan = {
            "attention": PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
            ),
            "attention_norm": SequenceParallel(),
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),

        }

        parallelize_module(transformer_block, tp_mesh, plan)

    # 3. Parallelize the output layer
    output_parallelize_plan = (
        ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
            use_local_output=not loss_parallel,
        )
    )
    parallelize_module(model.output, tp_mesh, output_parallelize_plan)

    return model


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
