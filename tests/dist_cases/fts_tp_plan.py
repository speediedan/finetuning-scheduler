# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``parallelize_model`` cases (#150): a plan from configuration on the model-parallel toy transformer, the same
plan as ``gen_apply_transformer_tp_plan``, trained and compared with a single-process run (pool cases, gloo or
NCCL)."""
import os

import yaml
from torch.distributed.tensor import DTensor

from finetuning_scheduler.strategies.plan import parallelize_model
from tests.dist_cases.fts_tuning_scheduler_mp import _data, _model, _params, _train, _tuning_scheduler, _unmoved

TOY_TP_PLAN = yaml.safe_load(r"""
tok_embeddings: {style: rowwise, input_layouts: replicate, output_layouts: {shard: 1}}
pos_embeddings: {style: rowwise, input_layouts: replicate, output_layouts: {shard: 0}}
norm: {style: sequence}
'layers\.\d+\.attention$': {style: prepare_input, input_layouts: {shard: 1}, desired_input_layouts: replicate}
'layers\.\d+\.(attention_norm|ffn_norm)$': {style: sequence}
'layers\.\d+\.attention\.w[qkv]$': {style: colwise, use_local_output: false}
'layers\.\d+\.attention\.wo$': {style: rowwise, output_layouts: {shard: 1}}
'layers\.\d+\.feed_forward\.w1$': {style: colwise, input_layouts: {shard: 1}}
'layers\.\d+\.feed_forward\.w2$': {style: rowwise, output_layouts: {shard: 1}}
output:
  style: colwise
  input_layouts: {shard: 1}
  output_layouts: replicate
  loss_parallel: {output_layouts: {shard: -1}, use_local_output: false}
""")


def _planned(ctx, seed: int, tp_loss_parallel: bool):
    mesh = ctx.mesh((ctx.world_size,), ("tensor_parallel",))
    return parallelize_model(_model(seed, ctx.device, math_sdpa=True), mesh["tensor_parallel"], TOY_TP_PLAN,
                             loss_parallel=tp_loss_parallel)


def planned_tp(ctx, epochs: int = 3, steps: int = 2, batch: int = 4, seed: int = 7, tp_loss_parallel: bool = True,
               ckpt_dir: str | None = None, val: list | None = None) -> dict:
    """Train the configured plan through the schedule, each rank seeing the whole batch."""
    model = _planned(ctx, seed, tp_loss_parallel)
    initial = _params(model)
    fts = _tuning_scheduler(model, ckpt_dir, restore_best=ckpt_dir is not None)
    losses = []
    path = _train(fts, model, _data(epochs, steps, batch, seed), slice(None), ctx.device, epochs, val=val,
                  tp_loss_parallel=tp_loss_parallel, losses=losses)
    return {"path": path, "params": _params(model), "restored": fts.restored_count, "unmoved": _unmoved(model, initial),
            "losses": losses, "plain": sorted(n for n, p in model.named_parameters() if not isinstance(p, DTensor))}


def planned_tp_resume(ctx, ckpt_dir: str, epochs: int = 3, split: int = 2, steps: int = 2, batch: int = 4,
                      seed: int = 7, tp_loss_parallel: bool = True, val: list | None = None) -> dict:
    """Train ``split`` epochs, save through DCP, load into a fresh planned model and tuning scheduler, and
    finish."""
    data = _data(epochs, steps, batch, seed)
    first_model = _planned(ctx, seed, tp_loss_parallel)
    first = _tuning_scheduler(first_model, os.path.join(ckpt_dir, "best"), restore_best=True)
    path = _train(first, first_model, data, slice(None), ctx.device, split, val=val, tp_loss_parallel=tp_loss_parallel)
    saved = first.save(os.path.join(ckpt_dir, "resume"))
    model = _planned(ctx, seed + 100, tp_loss_parallel)  # different initial weights: the load must overwrite them
    fts = _tuning_scheduler(model, os.path.join(ckpt_dir, "best"), restore_best=True)
    fts.load(saved)
    path = _train(fts, model, data, slice(None), ctx.device, epochs, val=val, tp_loss_parallel=tp_loss_parallel,
                  start_path=path)
    return {"path": path, "params": _params(model), "saved_is_dir": os.path.isdir(saved)}


def example_plan_matches_code(ctx) -> dict:
    """The PyTorch example's configured plan leaves the torchtitan ``Transformer`` laid out as the Lightning
    example's ``apply_tp_plan`` does: the same placements and the same hooks on every module."""
    import torch

    from fts_examples.model_parallel.parallel_plans import apply_tp_plan
    from fts_examples.model_parallel.torchtitan_llama import ModelCfg, Transformer

    config = os.path.join(os.path.dirname(apply_tp_plan.__code__.co_filename), "config", "torch_tp_plan.yaml")
    with open(config) as f:
        plan = yaml.safe_load(f)["tp_plan"]
    mesh = ctx.mesh((ctx.world_size,), ("tensor_parallel",))

    def layout(model) -> dict:
        params = {n: tuple(map(str, p.placements)) for n, p in model.named_parameters() if isinstance(p, DTensor)}
        hooks = {n: (len(m._forward_pre_hooks), len(m._forward_hooks)) for n, m in model.named_modules()
                 if m._forward_pre_hooks or m._forward_hooks}
        return {"params": params, "hooks": hooks,
                "plain": sorted(n for n, p in model.named_parameters() if not isinstance(p, DTensor))}

    def build() -> Transformer:
        with torch.device("meta"):
            return Transformer(ModelCfg(dim=16, n_layers=2, n_heads=2, vocab_size=32, max_seq_len=8))

    return {loss_parallel: {
        "configured": layout(parallelize_model(build(), mesh["tensor_parallel"], plan, loss_parallel=loss_parallel)),
        "code": layout(apply_tp_plan(build(), device_mesh=mesh, loss_parallel=loss_parallel)),
    } for loss_parallel in (True, False)}
