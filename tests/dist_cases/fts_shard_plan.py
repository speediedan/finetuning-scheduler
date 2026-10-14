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
"""``shard_model`` cases (#55) on the model-parallel toy transformer and its fine-tuning schedule: run on the fake
process group (``tests/test_shard_model.py``), so they check the plan applied and the feedback, not numerics."""
import re

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.strategies.fsdp2 import FSDP2Adapter
from finetuning_scheduler.strategies.plan import shard_model
from tests.dist_cases.fts_controller_mp import SCHEDULE, _model


def phases(model: torch.nn.Module) -> dict[int, list[str]]:
    """The schedule's phases as the physical parameter names each thaws (first phase that names a parameter)."""
    names = [n for n, _ in model.named_parameters()]
    return {d: [n for n in names if any(re.match(p, n) for p in SCHEDULE[d]["params"])] for d in SCHEDULE}


def _layout(model: torch.nn.Module) -> dict:
    return {"fsdp_modules": sorted(n for n, m in model.named_modules() if isinstance(m, FSDPModule)),
            "plain": sorted(n for n, p in model.named_parameters() if not isinstance(p, DTensor))}


def shard(ctx, plan: dict, default_kwargs: dict | None = None, pre_shard: str | None = None) -> dict:
    """Shard the toy transformer with ``plan`` (the root last) and report its layout and the FSDP2 adapter's
    findings on the schedule (what the controller reports)."""
    model = _model(7, ctx.device)
    if pre_shard is not None:  # a module the caller shards itself before the plan runs
        from torch.distributed.fsdp import fully_shard

        fully_shard(model.get_submodule(pre_shard), mesh=ctx.mesh(), reshard_after_forward=False)
    try:
        shard_model(model, ctx.mesh(), plan, default_kwargs=default_kwargs)
    except FTSConfigurationError as err:
        return {"error": str(err)}
    feedback = FSDP2Adapter().validate(model, phases(model))
    return {**_layout(model), "feedback": [(f.level, f.message) for f in feedback]}


def example_plan(ctx, config: str) -> dict:
    """An example configuration's FSDP2 plan and schedule on the examples' model, built on the meta device: the
    layout and the FSDP2 adapter's findings, which the examples' warning checks would otherwise meet only on GPUs.

    The
    Lightning configurations (``fts_*``) hold the plan in the callback's ``strategy_adapter_cfg``; their
    ``LightningModule`` holds the transformer as ``model``, as the plain PyTorch example's ``ModParallelModule`` does.
    """
    import argparse

    import yaml

    from finetuning_scheduler.core.schedule import load_yaml_schedule, resolve_param_entries, shared_params_ref
    from fts_examples.model_parallel import torch_mp_example as ex

    if config.startswith("torch_"):
        cfg = ex.load_config(argparse.Namespace(config=ex.example_path(f"config/{config}"), max_epochs=None,
                                                output_dir=None))
        plan, default_kwargs, sched_path = cfg["fsdp_plan"], cfg["fsdp_default_kwargs"], cfg["ft_schedule"]
        model_cfg_path = cfg["model_cfg"]
    else:
        with open(ex.example_path(f"config/{config}")) as f:
            fts_args = yaml.safe_load(f)["trainer"]["callbacks+"][0]["init_args"]
        plan = fts_args["strategy_adapter_cfg"]["fsdp_plan"]
        default_kwargs = fts_args["strategy_adapter_cfg"].get("fsdp_default_kwargs", {})
        sched_path, model_cfg_path = fts_args["ft_schedule"], "config/defaults/torchtitan_llama.yaml"
    with open(ex.example_path(model_cfg_path)) as f:
        model_cfg = ex.ModelCfg(**yaml.safe_load(f))
    with torch.device("meta"):
        model = ex.ModParallelModule(model_cfg)
    # the Lightning adapter shards only what its plan names (here, the transformer last); `shard_model` would shard
    # the `LightningModule` stand-in too
    shard_model(model, ctx.mesh(), plan, default_kwargs=default_kwargs, shard_root=config.startswith("torch_"))
    names, shared = [n for n, _ in model.named_parameters()], shared_params_ref(model)
    schedule = load_yaml_schedule(ex.example_path(sched_path))
    sched_phases = {d: resolve_param_entries(schedule[d]["params"], d, names, shared) for d in schedule}
    feedback = FSDP2Adapter().validate(model, sched_phases)
    return {**_layout(model), "feedback": [(f.level, f.message) for f in feedback]}
