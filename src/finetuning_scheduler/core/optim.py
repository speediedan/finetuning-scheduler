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
Fine-Tuning Scheduler Core Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Framework-neutral optimizer and lr scheduler support: reinitialization class resolution and sanity checks, parameter
group construction and growth, scheduler extension and lr state restoration.

"""
import os
import warnings
from collections.abc import Callable, Iterable, Sequence
from copy import copy
from functools import reduce
from pprint import pformat
from typing import Any, overload

import torch
from torch.nn import Module
from torch.optim import Optimizer

from finetuning_scheduler.core.dist import rank_zero_debug, rank_zero_warn
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.types import FTSLRSchedulerType, FTSLRSchedulerTypeTuple, ParamGroupAddable
from finetuning_scheduler.setup_tools import disable_always_warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*functional optimizers is deprecated.*")
with disable_always_warnings():
    from torch.distributed.optim import ZeroRedundancyOptimizer

SANITY_CHK_ITERABLE = (torch.nn.Parameter(torch.empty(1)),)

PHASE_0_DIVERGENCE_MSG = (
    "After executing the provided `configure_optimizers` method, the optimizer state differs from the configuration"
    " FinetuningScheduler expected at the beginning of scheduled fine-tuning (phase 0).\n"
)


class _MockOptimizer(Optimizer):
    """A parameterless stand-in optimizer used to sanity check lr scheduler configurations."""

    def __init__(self) -> None:
        super().__init__([torch.zeros(1)], {})

    def add_param_group(self, param_group: dict[Any, Any]) -> None:
        pass  # Do Nothing

    def load_state_dict(self, state_dict: dict[Any, Any]) -> None:
        pass  # Do Nothing

    def state_dict(self) -> dict[str, Any]:
        return {}  # Return Empty

    @overload
    def step(self, closure: None = ...) -> None:
        ...  # pragma: no cover

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...  # pragma: no cover

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if closure is not None:
            return closure()
        return None

    def zero_grad(self, set_to_none: bool | None = True) -> None:
        pass  # Do Nothing

    def __repr__(self) -> str:
        return "No Optimizer"


def import_reinit_class(reinit_cfg: dict) -> Any:
    """Import the class named by the ``class_path`` of a reinitialization configuration.

    Only the import is guarded, so an error raised while validating the imported class is never reported as an
    import failure.

    Args:
        reinit_cfg: The reinitialization configuration.

    Raises:
        FTSConfigurationError: If ``class_path`` is not fully qualified or the class cannot be imported.

    Returns:
        The imported class.
    """
    try:
        if "." not in reinit_cfg["class_path"]:
            raise ValueError("a fully qualified `class_path` is required (e.g. `torch.optim.SGD` rather than `SGD`)")
        class_module, class_name = reinit_cfg["class_path"].rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        reinit_class = getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as err:
        error_msg = (
            "Could not import specified reinitialization configuration class using class_path "
            f"({reinit_cfg['class_path']}). Received the following error while importing: {err}. Please validate "
            "specified `class_path` before resubmitting."
        )
        rank_zero_warn(error_msg)
        raise FTSConfigurationError(error_msg)
    return reinit_class


def check_supported_lr(lr_class: FTSLRSchedulerType, allow_untested: bool = False) -> None:
    """Check that ``lr_class`` is a supported lr scheduler, warning instead of raising if ``allow_untested``.

    Raises:
        FTSConfigurationError: If ``allow_untested`` is ``False`` and ``lr_class`` is not a subclass of a supported
            scheduler (``FTSLRSchedulerTypeTuple``).
    """
    if not issubclass(lr_class, FTSLRSchedulerTypeTuple):
        if not allow_untested:
            error_msg = (
                f"The provided lr scheduler ({lr_class}) is not currently supported by"
                " FinetuningScheduler. Please use a currently supported torch scheduler (or subclass thereof)"
                f" ({([i.__name__ for i in FTSLRSchedulerTypeTuple])}) or if you would like to attempt to use the"
                " currently specified scheduler, pass ``allow_untested=True`` to the FinetuningScheduler callback"
                " when adding it."
            )
            rank_zero_warn(error_msg)
            raise FTSConfigurationError(error_msg)
        else:
            warn_msg = "Allowing untested scheduler" f" '{type(lr_class)}' because ``allow_untested`` is ``True``."
            rank_zero_warn(warn_msg)


def check_supported_reinit_optimizer(optim_class: Any) -> None:
    """Check that ``optim_class`` is supported in the context of optimizer reinitialization.

    Raises:
        FTSConfigurationError: If ``optim_class`` is known to be unsupported for reinitialization.
    """
    if issubclass(optim_class, ZeroRedundancyOptimizer):
        error_msg = (
            f"The provided optimizer ({optim_class}) is not currently supported by FinetuningScheduler in the"
            " context of optimizer reinitialization. Please use a currently supported torch optimizer (or subclass"
            " thereof) from those provided in `torch.optim`: https://pytorch.org/docs/stable/optim.html#algorithms."
        )
        rank_zero_warn(error_msg)
        raise FTSConfigurationError(error_msg)


def check_optimizer_init(optimizer_class: Any, optimizer_init: dict) -> None:
    """Sanity check that ``optimizer_class`` can be instantiated with the configured ``init_args``.

    Raises:
        FTSConfigurationError: If the optimizer cannot be instantiated with the specified ``init_args``.
    """
    test_optimizer_init = copy(optimizer_init.get("init_args", {}))
    try:
        assert callable(optimizer_class)
        test_optimizer = optimizer_class(SANITY_CHK_ITERABLE, **test_optimizer_init)
    except Exception as err:
        error_msg = (
            "Could not configure the specified optimizer class using the `init_args` "
            f"({optimizer_init['init_args']}). Received the following error while sanity checking schedule "
            f"phases: {err}. Please validate specified `init_args` before resubmitting."
        )
        rank_zero_warn(error_msg)
        raise FTSConfigurationError(error_msg)
    assert isinstance(test_optimizer, ParamGroupAddable)


def check_lr_scheduler_init(lrs_class: Any, lr_scheduler_init: dict, is_implicit_mode: bool = False) -> None:
    """Sanity check that ``lrs_class`` can be instantiated with the configured ``init_args``.

    An ``optimizer`` key in ``init_args`` is removed in place with a warning, since the existing optimizer is always
    used.

    Raises:
        FTSConfigurationError: If ``min_lr`` is a list or tuple for an implicit mode ``ReduceLROnPlateau``
            reinitialization, or if the scheduler cannot be instantiated with the specified ``init_args``.
    """
    if lr_scheduler_init.get("init_args") and "optimizer" in lr_scheduler_init.get("init_args", {}).keys():
        warn_msg = (
            f"Found an `optimizer` key in the provided `lr_scheduler_init`: {lr_scheduler_init['init_args']} "
            f"Note that the existing optimizer and all associated parameter groups will be used when "
            "reinitializing the lr schedule using the specified scheduler so the provided `optimizer` key will "
            "have no effect."
        )
        rank_zero_warn(warn_msg)
        del lr_scheduler_init["init_args"]["optimizer"]
    min_lr_param = lr_scheduler_init["init_args"].get("min_lr")
    invalid_min_lr = (
        True if min_lr_param and (isinstance(min_lr_param, list) or isinstance(min_lr_param, tuple)) else False
    )
    reinit_rlrop = is_implicit_mode and issubclass(lrs_class, torch.optim.lr_scheduler.ReduceLROnPlateau)
    if reinit_rlrop and invalid_min_lr:
        raise FTSConfigurationError(
            "In the lr scheduler configuration passed via `reinit_lr_cfg` (i.e. implicit mode training)"
            " `min_lr` cannot be a list or tuple since the same lr scheduler configuration is intended to be"
            " reinitialized at every fine-tuning phase with implicit mode fine-tuning."
        )
    test_lr_init = copy(lr_scheduler_init.get("init_args", {}))
    if min_lr_param:
        del test_lr_init["min_lr"]  # our mock optimizer will not have any param groups
    try:
        assert callable(lrs_class)
        testlr = lrs_class(_MockOptimizer(), **test_lr_init)
    except Exception as err:
        error_msg = (
            "Could not configure the specified LR scheduler class using the `init_args` "
            f"({lr_scheduler_init['init_args']}). Received the following error while sanity checking schedule "
            f"phases: {err}. Please validate specified `init_args` before resubmitting."
        )
        rank_zero_warn(error_msg)
        raise FTSConfigurationError(error_msg)
    assert issubclass(type(testlr), FTSLRSchedulerTypeTuple)


def reinit_phase0_pgs(module: Module, thawed_pl: list, no_decay: list | None = None) -> list:
    """Reconstruct the parameter groups associated with phase 0 of the schedule.

    Args:
        module: The module whose parameters populate the groups.
        thawed_pl: The names of the parameters from which to construct the initial parameter groups.
        no_decay: Parameter name fragments that should always have ``weight_decay`` set to 0.

    Returns:
        A list of one or two new parameter groups (two when ``no_decay`` is used).
    """
    if no_decay:
        pgs = [
            {
                "params": [
                    p
                    for n, p in module.named_parameters()
                    if not any(nd in n for nd in no_decay) and n in thawed_pl and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in module.named_parameters()
                    if any(nd in n for nd in no_decay) and n in thawed_pl and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
    else:
        pgs = [{"params": [p for n, p in module.named_parameters() if n in thawed_pl and p.requires_grad]}]
    return pgs


def repartition_sharded_optim(optimizer: Any) -> None:
    """Repartition and reset a sharded optimizer state (e.g. ``ZeroRedundancyOptimizer``)."""
    # For optimizers that shard their states like (e.g. ZeroRedundancyOptimizer), one use case for this method is
    # to clear local optimizer partition caches and repartition to support restoring across multiple depths
    partition_params = (
        optimizer._partition_parameters if callable(optimizer._partition_parameters) else optimizer.partition_parameters
    )
    optimizer._clear_cache()
    optimizer.optim.param_groups = partition_params()[optimizer.rank]  # type: ignore[index]
    optimizer._sync_param_groups(optimizer.optim.param_groups, optimizer.param_groups)


def restore_latest_lr_state(
    optimizer: ParamGroupAddable, schedulers: Iterable[Any], curr_lr_state: dict, prev_optimizer_lrs: list
) -> None:
    """Restore the lr scheduler state and the most recent ``lr`` of every parameter group from previous depths.

    Args:
        optimizer: The (new) optimizer whose existing parameter groups should take the previous ``lr`` s.
        schedulers: The lr schedulers to restore ``curr_lr_state`` into.
        curr_lr_state: The lr scheduler state captured before groups were added to the new optimizer.
        prev_optimizer_lrs: The most recent ``lr`` s of the previous optimizer's parameter groups.
    """
    for scheduler in schedulers:
        scheduler.load_state_dict(curr_lr_state)
    for _, data in enumerate(zip(optimizer.param_groups, prev_optimizer_lrs)):
        param_group, lr = data
        param_group["lr"] = lr
    rank_zero_debug("Current LR state restored for previous depth parameter groups.")


def add_optimizer_groups(
    module: Module,
    optimizer: ParamGroupAddable,
    thawed_pl: list,
    no_decay: list | None = None,
    lr: float | None = None,
    apply_lambdas: bool = False,
    schedulers: Sequence[Any] | Callable[[], Sequence[Any]] = (),
) -> None:
    """Add the parameter groups for the next fine-tuning phase and extend each lr scheduler to cover them.

    Args:
        module: The module from which the parameters will be read.
        optimizer: The optimizer to which parameter groups will be added.
        thawed_pl: The thawed parameters that should be added to the new parameter group(s).
        no_decay: Parameter name fragments that should always have ``weight_decay`` set to 0.
        lr: The initial learning rate for the new group(s); the first group's ``lr`` if not specified.
        apply_lambdas: Whether to apply lr lambdas to the new groups.
        schedulers: The lr schedulers to extend, or a callable returning them (called only when there are
            parameters to add).
    """
    if len(thawed_pl) == 0:
        rank_zero_warn("No thawed parameters passed so no new optimizer groups will be added.")
    else:
        phase_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
        orig_lr_factor = phase_lr
        target_schedulers = schedulers() if callable(schedulers) else schedulers
        if target_schedulers:
            for scheduler in target_schedulers:
                if hasattr(scheduler, "lr_lambdas") and scheduler.lr_lambdas and apply_lambdas:
                    phase_lr = phase_lr * scheduler.lr_lambdas[-1](scheduler.last_epoch)
                added_pgs = 0
                added_pgs = add_groups(no_decay, optimizer, module, thawed_pl, phase_lr)
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.min_lrs.extend([scheduler.min_lrs[0]] * added_pgs)
                else:
                    scheduler.base_lrs.extend([orig_lr_factor] * added_pgs)
                    if hasattr(scheduler, "lr_lambdas") and scheduler.lr_lambdas:
                        # due to PyTorch lr scheduler state_dict peculiarities wrt lr_lambdas, lr_lambdas may
                        # already be pg-aligned (since lr_lambdas are only conditionally saved/restored) see:
                        # https://bit.ly/lr_lambda_state_dict_special_handling
                        lambdas_to_sync = max(len(scheduler.base_lrs) - len(scheduler.lr_lambdas), 0)
                        if lambdas_to_sync:
                            scheduler.lr_lambdas.extend([scheduler.lr_lambdas[-1]] * lambdas_to_sync)
        else:
            _ = add_groups(no_decay, optimizer, module, thawed_pl, phase_lr)


def add_groups(
    no_decay: list | None, optimizer: ParamGroupAddable, module: Module, thawed_pl: list, phase_lr: float
) -> int:
    """Add parameter group(s) holding the thawed parameters not already owned by an existing group.

    Args:
        no_decay: Parameter name fragments that should always have ``weight_decay`` set to 0.
        optimizer: The optimizer to which parameter groups will be added.
        module: The module from which the parameters will be read.
        thawed_pl: The thawed parameters that should be added to the new parameter group(s).
        phase_lr: The initial learning rate for the new parameter group(s).

    Returns:
        The number of parameter groups added; ``0`` if every candidate parameter already belongs to a group (as when
        resuming, where the restored optimizer state precedes the replay of the schedule phases).
    """
    # exclude parameters an existing group already owns; identity rather than equality since parameter tensors
    # are not hashable by value and `in` would trigger elementwise comparison
    existing_params = {id(p) for group in optimizer.param_groups for p in group.get("params", [])}
    candidates = [
        (n, p)
        for n, p in module.named_parameters()
        if n in thawed_pl and p.requires_grad and id(p) not in existing_params
    ]
    if not candidates:
        # nothing left to add: returning 0 keeps the caller from extending `base_lrs`/`min_lrs`/`lr_lambdas`
        # for groups that were never created
        return 0
    if no_decay:
        optimizer.add_param_group(
            {
                "params": [p for n, p in candidates if not any(nd in n for nd in no_decay)],
                "lr": phase_lr,
                "initial_lr": phase_lr,
            }
        )
        optimizer.add_param_group(
            {
                "params": [p for n, p in candidates if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": phase_lr,
                "initial_lr": phase_lr,
            }
        )
        added_pgs = 2
    else:
        optimizer.add_param_group(
            {
                "params": [p for _, p in candidates],
                "lr": phase_lr,
                "initial_lr": phase_lr,
            }
        )
        added_pgs = 1
    return added_pgs


def sync(objs: tuple, asets: tuple, agg_func: Callable = max) -> None:
    """Synchronize sets of object attributes using a given aggregation function.

    Dotted attribute paths are read through the nested objects they name but are never written: they are read-only
    views of state owned by another component. Only plain (non-dotted) attributes receive the aggregated value.

    Args:
        objs: The target objects to synchronize.
        asets: The attribute sets to synchronize.
        agg_func: The aggregation function used to synchronize each attribute set. Defaults to ``max``.
    """
    for attrs in asets:
        agg = reduce(agg_func, [reduce(getattr, a.split(sep="."), o) for o, a in zip(objs, attrs)])
        for o, a in zip(objs, attrs):
            if "." not in a:
                setattr(o, a, agg)


def grad_mismatch_feedback(w_msg: str, param_diff_summary: dict) -> str:
    """Append a summary of how the optimizer's thawed parameters diverge from those scheduled for phase 0.

    Args:
        w_msg: The initial warning message.
        param_diff_summary: The ``optim_params`` and ``phase_0_params`` name lists that differ.

    Returns:
        The warning message with the divergence summary appended.
    """
    w_msg += (
        " a differing set of trainable parameters. Please find below a summary of the differences between"
        " the currently thawed parameters in the optimizer and those scheduled to be optimized during fine-tuning"
        " phase 0: \n"
        "Currently thawed parameters included in the optimizer:\n"
        f"{pformat(param_diff_summary['optim_params'])}{os.linesep}"
        "Parameters expected to be thawed and optimized in phase 0:\n"
        f"{pformat(param_diff_summary['phase_0_params'])}{os.linesep}"
    )
    return w_msg
