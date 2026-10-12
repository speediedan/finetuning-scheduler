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
r"""
Fine-Tuning Scheduler Phases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Framework-neutral freezing and thawing: freeze a module tree before training, thaw a phase's parameters, and record or
re-apply an optimizer's parameter groups by parameter name.

"""
from collections.abc import Callable, Iterable
from pprint import pformat as pfmt
from typing import Any

import torch
from torch.nn import Module, ModuleDict
from torch.nn.modules.batchnorm import _BatchNorm

from finetuning_scheduler.core.dist import rank_zero_debug


def reshard_fsdp_modules(module: Module) -> None:
    """Register every FSDP2 module's sharded parameters, before parameters are frozen, thawed, optimized or
    checkpointed.

    FSDP2 can leave a module's unsharded parameters registered between steps: the root module's after a forward-only
    pass (a validation or evaluation pass), and a root module's whose parameters are all frozen after its forward, since
    no backward hook reshards it. Thawing those would miss the sharded parameters the next all-gather registers again,
    which stay frozen.
    """
    if not torch.distributed.is_available():
        return
    from torch.distributed.fsdp import FSDPModule

    for submodule in module.modules():
        if isinstance(submodule, FSDPModule):
            submodule.reshard()


def thaw_params(module: Module, thaw_pl: list, translation_func: Callable | None = None,
                init_thaw: bool = False) -> tuple[list, list]:
    """Thaw (unfreeze) the named parameters of ``module`` listed in ``thaw_pl``.

    Args:
        module: The module whose parameters are selectively thawed.
        thaw_pl: The names of the parameters to thaw.
        translation_func: Optionally translates parameter names for log messages only.
        init_thaw: Whether this is the initial (phase 0) thaw, which only changes the log message.

    Returns:
        The names newly thawed by this call (in ``named_parameters`` order), and the names of every currently thawed
        parameter (those thawed before this call, then the newly thawed ones).
    """
    thawed_p_names = []
    curr_thawed = []
    for n, p in module.named_parameters():
        if not p.requires_grad and n in thaw_pl:
            p.requires_grad = True
            thawed_p_names.append(n)
        elif p.requires_grad:
            curr_thawed.append(n)
    if thawed_p_names:
        rank_zero_debug(
            f"{'Initializing with' if init_thaw else 'Thawed'} the following module parameters: "
            f"{pfmt(translation_func(thawed_p_names)) if translation_func else pfmt([n for n in thawed_p_names])}"
        )
    curr_thawed.extend(thawed_p_names)
    rank_zero_debug(
        f"The following module parameters are currently thawed: "
        f"{pfmt(translation_func(curr_thawed)) if translation_func else pfmt([n for n in curr_thawed])}"
    )
    return thawed_p_names, curr_thawed


def flatten_modules(modules: Module | Iterable[Module | Iterable]) -> list[Module]:
    """Flatten a module, or an iterable of modules, into its leaf modules plus any parent module that holds
    parameters directly."""
    if isinstance(modules, ModuleDict):
        modules = modules.values()
    if isinstance(modules, Iterable):
        flattened = []
        for m in modules:
            flattened.extend(flatten_modules(m))
        _modules = iter(flattened)
    else:
        _modules = modules.modules()
    return [m for m in _modules if not list(m.children()) or m._parameters]


def freeze_module_tree(modules: Module | Iterable[Module | Iterable], train_bn: bool = False,
                       bn_track_running_stats: bool = False) -> None:
    """Freeze every parameter of ``modules`` before a phased run.

    Args:
        modules: A module or an iterable of modules.
        train_bn: Leave ``BatchNorm`` layers trainable (and tracking running statistics).
        bn_track_running_stats: Keep frozen ``BatchNorm`` layers tracking running statistics.
    """
    for mod in flatten_modules(modules):
        if isinstance(mod, _BatchNorm):
            mod.track_running_stats = train_bn or bn_track_running_stats
        # recursion would visit parameters of parent modules more than once
        for param in mod.parameters(recurse=False):
            param.requires_grad = train_bn and isinstance(mod, _BatchNorm)


def map_param_groups(param_groups: list[dict[str, Any]], mapping: dict) -> list[dict[str, Any]]:
    """Copy ``param_groups`` with each entry of ``params`` replaced by ``mapping[entry]``.

    With a parameter-to-name mapping this records an optimizer's groups by name; with the inverse mapping it rebuilds
    them from the record.
    """
    output = []
    for g in param_groups:
        group_state = {k: v for k, v in g.items() if k != "params"}
        group_state["params"] = [mapping[p] for p in g["params"]]
        output.append(group_state)
    return output


def thaw_plan(schedule: dict, depth: int, best_ckpt_depth: int, depth_sync: bool,
              reinit_optim_cfg: dict | None) -> tuple[dict | None, list[tuple[int, dict]]]:
    """Select the scheduled phases to (re)visit when configuring the optimizer for ``depth``.

    Args:
        schedule: The fine-tuning schedule.
        depth: The target depth.
        best_ckpt_depth: The depth of the restored (best) checkpoint.
        depth_sync: Revisit every phase after the restored depth, rather than only ``depth``.
        reinit_optim_cfg: An implicit-mode optimizer reinitialization configuration, if any.

    Returns:
        The optimizer reinitialization configuration to apply (never for depth 0), and the ``(phase, config)`` pairs to
        visit in schedule order. A reinitialized optimizer revisits every phase from 0.
    """
    new_optimizer_cfg = (reinit_optim_cfg or schedule[depth].get("new_optimizer", None)) if depth > 0 else None
    restored_depth = -1 if new_optimizer_cfg else best_ckpt_depth
    if depth_sync or new_optimizer_cfg:
        return new_optimizer_cfg, [(d, tl) for d, tl in schedule.items() if d > restored_depth]
    return new_optimizer_cfg, [(depth, schedule[depth])]


def lr_reinit_cfg_for(phase: dict, depth: int, reinit_lr_cfg: dict | None) -> dict | None:
    """The lr scheduler reinitialization configuration that applies at ``depth`` (never at depth 0)."""
    return (reinit_lr_cfg or phase.get("new_lr_scheduler", None)) if depth > 0 else None


def depth_remaining(max_depth: int, curr_depth: int) -> int:
    """The number of scheduled phases after the current one (never negative)."""
    return max(max_depth - curr_depth, 0)
