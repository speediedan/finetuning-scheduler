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
Fine-Tuning Scheduler Model Parallel Strategy Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` that extends Fine-Tuning Scheduler's support
for PyTorch's distributed composable and Tensor Parallel APIs.

"""
from typing import Any  # Dict used for runtime isinstance() checks
from collections.abc import Callable, Sequence
from functools import wraps
import re
from dataclasses import dataclass, field

import torch
from torch.distributed._composable import checkpoint
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (checkpoint_wrapper, offload_wrapper,
                                                                         ActivationWrapper)
from lightning.fabric.utilities.enums import LightningEnum
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.fabric.utilities import rank_zero_warn, rank_zero_info
from lightning.pytorch.utilities.rank_zero import rank_zero_debug

from finetuning_scheduler.strategies.base import DistributedAdapter, select_distributed_adapter
from finetuning_scheduler.strategy_adapters.base import StrategyAdapter
from finetuning_scheduler.strategy_adapters._wrap_utils import _compose_ncac

# an `fsdp_plan` key with this prefix selects modules by class rather than by name
CLASS_SELECTOR = "class:"


class ActCkptEnum(LightningEnum):
    COMPOSABLE = "composable"
    WRAPPED = "wrapped"
    WRAPPED_OFFLOAD = "wrapped_offload"

@dataclass
class ActCkptCfg:
    mode: ActCkptEnum | str
    cfg: dict[str, Any] | None = field(default_factory=dict)
    _func: Callable | None = None

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            try:
                self.mode = ActCkptEnum(self.mode)
            except ValueError:
                rank_zero_warn(f"Unknown AC mode: {self.mode}. Defaulting to `{ActCkptEnum.COMPOSABLE.value}` mode.")
                self.mode = ActCkptEnum.COMPOSABLE
        if self.mode == ActCkptEnum.COMPOSABLE:
            self._func = checkpoint
        elif self.mode == ActCkptEnum.WRAPPED_OFFLOAD:
            self._func = offload_wrapper
        elif self.mode == ActCkptEnum.WRAPPED:
            self._func = checkpoint_wrapper


class ModelParallelStrategyAdapter(StrategyAdapter):
    r"""
    A :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` that extends
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` (FTS) to support flexible, multi-phase, scheduled fine-tuning
    with PyTorch's composable distributed (e.g. ``fully_shard``) and Tensor Parallelism APIs.
    FTS augments Lightning's Model Parallel strategy
    (:py:class:`~lightning.pytorch.strategies.model_parallel.ModelParallelStrategy`) by allowing users to apply
    the ``fully_shard`` API using module name/pattern-based configuration instead of manually inspecting modules and
    applying the API in ``LightningModule.configure_model`` (see
    :attr:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter.fsdp_plan`).

    See the :ref:`model-parallel-fine-tuning-examples` tutorial for a concrete example and additional guidance.

    .. note::
        ``fsdp_plan`` module name/pattern-based ``fully_shard`` directives are applied after any preceding Tensor
        Parallel or explicit ``fully_shard`` directives in ``LightningModule.configure_model``. FTS will only apply
        ``fully_shard`` to a specified module if it was not already applied to that module.

    .. note::
        In addition to all valid ``fully_shard`` API kwargs, ``fsdp_plan`` also supports a ``act_ckpt`` and
        ``cpu_offload_policy`` kwargs.

    For specified module/patterns (or ``fsdp_default_kwargs``), ``act_ckpt`` allows one
    to pass a string alias specifying the use of the desired activation checkpointing API (e.g. "composable",
    "wrapped", "wrapped_offload") as well as an optional ``Dict`` of activation checkpointing kwargs. The specified
    checkpointing APIs will be applied to the matching module(s) before ``fully_shard``.

    ``cpu_offload_policy`` is a convenience alias that will apply CPUOffloadPolicy to the matching module(s) along
    with any provided ``Dict`` of policy kwargs.

    A plan key may also select modules by class: ``class:TransformerBlock`` selects every module that is an instance
    of a class named ``TransformerBlock`` (as ``ModuleWrapPolicy({TransformerBlock})`` would), and
    ``class:my_pkg.blocks.TransformerBlock`` names the class unambiguously. ``ignored_params`` may name the parameters
    to leave unsharded, by parameter name or regex (or a list of them), so a plan written in YAML can use it.
    """

    # a built-in adapter: its own overrides are never naming overrides (see `StrategyAdapter.naming_overrides`)
    _fts_naming_overrides: frozenset[str] = frozenset()

    def __init__(self, fsdp_default_kwargs: dict | None = None, fsdp_plan: dict | None = None, *args: Any,
                 **kwargs: Any) -> None:
        """The only user-facing configuration for
        :class:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter` are
        :attr:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter.fsdp_plan` and
        :attr:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter.fsdp_default_kwargs`.

        Args:
            fsdp_plan (Dict | None): An optional dictionary of module names or regex pattern keys with associated
                ``fully_shard`` composable distributed API kwargs to apply to matching modules.

                - Allows users to apply the ``fully_shard`` API using module name/pattern-based configuration instead of
                  manually inspecting modules and applying the API in ``LightningModule.configure_model``.
                - ``fsdp_plan`` directives can also be composed with explicit ``fully_shard`` calls in
                  ``LightningModule.configure_model``, as the ``fsdp_plan`` directives will only invoke ``fully_shard``
                  on a specified module if it was not already applied to that module.
                - All valid ``fully_shard`` API kwargs are supported.
                - :attr:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter.fsdp_plan` directives are
                  applied in the order provided in the ``fsdp_plan`` dictionary.
                - A key ``class:<name>`` (or ``class:<module>.<qualname>``) selects every module that is an instance of
                  that class, as ``ModuleWrapPolicy`` does.

                Additionally, ``fsdp_plan`` supports ``act_ckpt`` and ``cpu_offload_policy`` kwargs, and an
                ``ignored_params`` given by parameter name or regex (or a list of them), resolved to the parameters
                they name when the plan is applied. For specified module/patterns (or ``fsdp_default_kwargs``):

                - ``act_ckpt`` (``Sequence`` [ ``str``, ``Dict`` | ``None`` ] | ``ActCkptCfg``): pass an alias
                  specifying the use of the desired activation checkpointing API (e.g. "composable", "wrapped",
                  "wrapped_offload") as well as an optional ``Dict`` of activation checkpointing kwargs. The specified
                  checkpointing APIs will be applied to the matching module(s) before ``fully_shard``.
                - ``cpu_offload_policy`` (``Dict`` [ ``Optional`` [ ``str`` , ``Any`` ]]) is a convience alias that will
                  apply ``CPUOffloadPolicy`` to the matching module(s) along with any provided Dict of policy kwargs.
                  Defaults to ``None``.

            fsdp_default_kwargs (Dict | None): An optional dictionary of default ``fully_shard`` API kwargs to apply
                to each matching module in ``fsdp_plan``. Module-name/pattern specific kwargs will take precedence over
                these. All kwargs valid for ``fsdp_plan`` above are supported. Defaults to ``None``.

        Attributes:
            fsdp_plan: An optional dictionary of module names or regex pattern keys with associated
                ``fully_shard`` composable distributed API kwargs to apply to matching modules.

                - Allows users to apply the ``fully_shard`` API using module name/pattern-based configuration instead of
                  manually inspecting modules and applying the API in ``LightningModule.configure_model``.
                - ``fsdp_plan`` directives can also be composed with explicit ``fully_shard`` calls in
                  ``LightningModule.configure_model``, as the ``fsdp_plan`` directives will only invoke ``fully_shard``
                  on a specified module if it was not already applied to that module.
                - All valid ``fully_shard`` API kwargs are supported.
                - :attr:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter.fsdp_plan` directives are
                  applied in the order provided in the ``fsdp_plan`` dictionary.
                - A key ``class:<name>`` (or ``class:<module>.<qualname>``) selects every module that is an instance of
                  that class, as ``ModuleWrapPolicy`` does.

                Additionally, ``fsdp_plan`` supports ``act_ckpt`` and ``cpu_offload_policy`` kwargs, and an
                ``ignored_params`` given by parameter name or regex (or a list of them), resolved to the parameters
                they name when the plan is applied. For specified module/patterns (or ``fsdp_default_kwargs``):

                - ``act_ckpt`` (``Sequence`` [ ``str``, ``Dict`` | ``None`` ] | ``ActCkptCfg``): pass an alias
                  specifying the use of the desired activation checkpointing API (e.g. "composable", "wrapped",
                  "wrapped_offload") as well as an optional ``Dict`` of activation checkpointing kwargs. The specified
                  checkpointing APIs will be applied to the matching module(s) before ``fully_shard``.
                - ``cpu_offload_policy`` (``Dict`` [ ``Optional`` [ ``str`` , ``Any`` ]]) is a convience alias that will
                  apply ``CPUOffloadPolicy`` to the matching module(s) along with any provided Dict of policy kwargs.

            fsdp_default_kwargs: An optional dictionary of default ``fully_shard`` API kwargs to apply to each matching
                module in ``fsdp_plan``. Module-name/pattern specific kwargs will take precedence over
                these. All kwargs valid for ``fsdp_plan`` above are supported.
        """
        super().__init__(*args, **kwargs)
        self.fsdp_default_kwargs = fsdp_default_kwargs or {}
        self.fsdp_plan = fsdp_plan or {}

    def on_before_init_fts(self) -> None:
        """In this hook executed immediately before
        :meth:`~finetuning_scheduler.fts_supporters.ScheduleImplMixin.init_fts`, to accommodate enhanced Model
        Parallel functionality, we:

        1. Validate the :attr:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter.fsdp_plan`
           configuration
        2. Configure FTS wrapping of the provided :py:class:`~lightning.pytorch.core.module.LightningModule`
           to either use the provided ``LightningModule.configure_model`` method (if present) or a provided
           ``fsdp_plan``.
        """
        self._validate_fsdp_plan()
        if is_overridden("configure_model", self.pl_module):
            rank_zero_debug("Overridden `LightningModule.configure_model` hook identified. Any name-driven distributed "
                            "API plans will be applied after the configure_model hook.")
            cm_func = self._wrapped_configure_model(self.pl_module.configure_model)
            setattr(self.pl_module, "configure_model", cm_func)
        else:
            setattr(self.pl_module, "configure_model", self._fts_auto_configure_model)

    _distributed_adapter: DistributedAdapter | None = None

    @property
    def distributed_adapter(self) -> DistributedAdapter:
        """The torch-native :class:`~finetuning_scheduler.strategies.base.DistributedAdapter` for the composed
        module's layout (FSDP2, tensor parallelism or unsharded), selected when the fit starts, after
        ``configure_model`` (until then, it reflects the module's current layout)."""
        return self._distributed_adapter or select_distributed_adapter(self.pl_module)

    def _validate_fsdp_fts_config(self) -> None:
        """Check each phase's physical parameters (compiled through an attached view, if any) against the composed
        module's layout, through its distributed adapter."""
        phases = {d: self.scheduled_physical(d) for d in self.fts_handle.ft_schedule}
        for feedback in self.distributed_adapter.validate(self.pl_module, phases):
            if feedback.level == "error":
                raise MisconfigurationException(feedback.message)
            (rank_zero_warn if feedback.level == "warning" else rank_zero_info)(feedback.message)

    def on_before_fts_fit_start(self) -> None:
        """In this hook executed immediately before the :class:`~finetuning_scheduler.fts.FinetuningScheduler`
        :meth:`~finetuning_scheduler.fts.FinetuningScheduler.on_fit_start` hook begins, we ensure the provided
        fine-tuning schedule and FSDP2 composed :py:class:`~lightning.pytorch.core.module.LightningModule` are
        appropriately aligned.

        If the fine-tuning schedule and composed modules yield parameter group configurations that may not be supported
        by some optimizer group operations, detailed feedback on potential remediation is provided to the user.
        """
        self._distributed_adapter = select_distributed_adapter(self.pl_module)
        self._validate_fsdp_fts_config()

    def _maybe_update_fsdp_plan(self, resolved_modules: dict, kwargs: dict, name: str) -> bool:
        # right now, the only explicitly disallowed types are the container types w/o a forward: `ModuleList``
        #  and `ModuleDict``
        # NB there are other ``fully_shard`` distributed API constraints that the user may need to validate, for
        # instance, shared parameters must belong to the same ``fully_shard`` instance
        # (https://bit.ly/fsdp_dapi_shared_constraint)
        if isinstance(self.pl_module.get_submodule(name), (torch.nn.ModuleList, torch.nn.ModuleDict)):
            rank_zero_warn(f"Provided FSDP module plan includes a directive to apply the FSDP API to ('{name}') which"
                           f" has the unsupported type '{type(self.pl_module.get_submodule(name))}'. This"
                           " directive will be pruned and ignored.")
            return False
        kwargs = ModelParallelStrategyAdapter._resolve_cfg_aliases(kwargs)  # default policy aliases already resolved
        resolved_modules[name] = {**self.fsdp_default_kwargs, **kwargs}
        return True

    @staticmethod
    def _resolve_cfg_aliases(config_dict: dict) -> dict:
        """Resolve any provided convenience FSDP default kwargs."""
        for k, v in list(config_dict.items()):
            if k == "act_ckpt":
                if not isinstance(v, ActCkptCfg):
                    config_dict["act_ckpt"] = ActCkptCfg(*v) if isinstance(v, Sequence) else ActCkptCfg(mode=v)
        return config_dict

    def _validate_fsdp_plan(self) -> None:
        """Expand any regex expressions specified in
        :attr:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter.fsdp_plan`.

        Raises:
            MisconfigurationException: If a specified module name or regex does not resolve to at least one named
                module.
        """
        if not self.fsdp_plan:
            return
        self._refuse_cpu_offload()
        # TODO: renable coverage below when upstream cpu offload issue addressed
        if self.fsdp_default_kwargs:  # pragma: no cover
            self.fsdp_default_kwargs = ModelParallelStrategyAdapter._resolve_cfg_aliases(self.fsdp_default_kwargs)
        named_modules = dict(self.pl_module.named_modules())
        resolved_modules: dict[str, dict] = {}
        for plan_n, kwargs in self.fsdp_plan.items():
            matched = ModelParallelStrategyAdapter._plan_key_modules(plan_n, named_modules)
            if not matched:
                what = "class" if plan_n.startswith(CLASS_SELECTOR) else "module or regex"
                raise MisconfigurationException(
                    f"The {what} '{plan_n}' specified in `fsdp_plan` did not match any named modules "
                    "in the provided model."
                )
            for n in matched:
                self._maybe_update_fsdp_plan(resolved_modules, kwargs, n)  # skips (with a warning) unsupported types
        self.fsdp_plan = resolved_modules

    def _refuse_cpu_offload(self) -> None:
        """Refuse a CPU offload policy in the plan, which this strategy cannot run.

        Lightning's ``ModelParallelStrategy`` moves the whole model to the GPU once ``configure_model`` has run, and a
        module sharded with ``CPUOffloadPolicy`` keeps its shards on the CPU, so the move fails (``_apply(): Couldn't
        swap ...``). CPU offload works with the PyTorch API (``FTSController``), where the training loop places the
        model itself.

        Raises:
            MisconfigurationException: If ``fsdp_default_kwargs`` or an ``fsdp_plan`` entry sets
                ``cpu_offload_policy``, or an ``offload_policy`` that offloads to the CPU.
        """
        entries = [("`fsdp_default_kwargs`", self.fsdp_default_kwargs or {})]
        entries += [(f"the `fsdp_plan` entry for '{k}'", v or {}) for k, v in self.fsdp_plan.items()]
        for where, kwargs in entries:
            if "cpu_offload_policy" in kwargs or isinstance(kwargs.get("offload_policy"), CPUOffloadPolicy):
                raise MisconfigurationException(
                    f"{where} asks for CPU offload, which Lightning's ModelParallelStrategy cannot run: it moves the "
                    "model to the GPU after `configure_model`, and FSDP2 keeps CPU-offloaded shards on the CPU. Use "
                    "the PyTorch API (`FTSController`) with `fully_shard(..., offload_policy=CPUOffloadPolicy())` for "
                    "CPU offload."
                )

    @staticmethod
    def _plan_key_modules(plan_n: str, named_modules: dict[str, torch.nn.Module]) -> list[str]:
        """The names of the modules an ``fsdp_plan`` key selects, in module order: a module name, a regex matched
        from the start of each module name, or ``class:<name>`` for every module that is an instance of a class
        with that name (``class:<module>.<qualname>`` to name it unambiguously), as ``ModuleWrapPolicy``
        selects."""
        if plan_n.startswith(CLASS_SELECTOR):
            ref = plan_n[len(CLASS_SELECTOR):].strip()

            def _is_instance(m: torch.nn.Module) -> bool:
                return any(ref in (c.__name__, f"{c.__module__}.{c.__qualname__}") for c in type(m).__mro__)

            return [n for n, m in named_modules.items() if _is_instance(m)]
        if plan_n in named_modules:
            return [plan_n]
        mpat = re.compile(fr"{plan_n}")
        return [n for n in named_modules if mpat.match(n)]

    def _fts_auto_configure_model(self) -> None:
        """A wrapper for configuration-driven composable distributed API configurations that may be composed with
        or substitute for explicit directives in a `LightningModule.configure_model` method.

        Currently, FTS only supports configuration-driven distributed API plans for `fully_shard`. See :attr:`fsdp_plan`
        for details.
        """
        # TODO: replace directly with `_apply_fsdp_plan` if foregoing `tp_auto_plan` implementation
        # self._apply_tp_auto_plan()
        self._apply_fsdp_plan()

    def _compose_or_warn(self, n: str) -> str | None:
        # NB PL does not currently support passing user-created meshes so we unconditionally pass the device mesh
        if not getattr(self.pl_module.get_submodule(n), '_is_fsdp_managed_module', False):
            if act_ckpt := self.fsdp_plan[n].pop('act_ckpt', None):
                self._apply_activation_checkpointing(act_ckpt,n)
            else:
                fully_shard(self.pl_module.get_submodule(n), mesh=self.pls_handle.device_mesh["data_parallel"],
                            **self._shard_kwargs(n))
        else:
            rank_zero_warn(
                f"Module '{n}' or one of its parents has already registered the ``fully_shard`` composable "
                f"distributed API. Applying the ``fully_shard`` API to '{n}' is not supported in this context."
                f"Please apply ``fully_shard`` to '{n}' before any of its parents if it is intended to have a "
                f"distinct `FSDPState`."
            )
            return n  # return the uncomposed fqn for subsequent handling

    def _apply_activation_checkpointing(self, act_ckpt: ActCkptCfg, n: str) -> None:
        assert isinstance(act_ckpt, ActCkptCfg)
        assert isinstance(act_ckpt.cfg, dict) and callable(act_ckpt._func)
        if act_ckpt.mode == ActCkptEnum.COMPOSABLE:
            act_ckpt._func(self.pl_module.get_submodule(n), **act_ckpt.cfg)
        elif act_ckpt.mode in (ActCkptEnum.WRAPPED, ActCkptEnum.WRAPPED_OFFLOAD):
            module_prefix, _, module_name = n.rpartition('.')
            outer_mod = self.pl_module if not module_prefix else self.pl_module.get_submodule(module_prefix)
            setattr(outer_mod, module_name, act_ckpt._func(self.pl_module.get_submodule(n), **act_ckpt.cfg))
            assert isinstance(self.pl_module.get_submodule(n), ActivationWrapper)
        fully_shard(self.pl_module.get_submodule(n), mesh=self.pls_handle.device_mesh["data_parallel"],
                    **self._shard_kwargs(n))

    def _shard_kwargs(self, n: str) -> dict[str, Any]:
        """The ``fully_shard`` kwargs for the plan's module ``n``, with an ``ignored_params`` given by name
        resolved to the parameters it names.

        ``ignored_params`` may be a parameter name or regex (matched from the start of each name, as schedule entries
        are), or a list of them, so a plan written in YAML can name the parameters ``fully_shard`` leaves unsharded.
        Names are the ``LightningModule``'s own parameter names, and are resolved when the plan is applied, since
        ``configure_model`` may create the parameters. A set of parameters is passed through unchanged.

        Raises:
            MisconfigurationException: If a name or regex in ``ignored_params`` matches no parameter.
        """
        kwargs = dict(self.fsdp_plan[n])
        names = kwargs.get("ignored_params")
        if names is None or isinstance(names, set):
            return kwargs
        named_params = dict(self.pl_module.named_parameters())
        resolved: set[torch.nn.Parameter] = set()
        for pattern in [names] if isinstance(names, str) else names:
            if pattern in named_params:
                matched = [named_params[pattern]]
            else:
                mpat = re.compile(fr"{pattern}")
                matched = [p for pn, p in named_params.items() if mpat.match(pn)]
            if not matched:
                raise MisconfigurationException(
                    f"The parameter name or regex '{pattern}' in the `ignored_params` of the `fsdp_plan` entry for "
                    f"'{n}' did not match any named parameter in the provided model."
                )
            resolved.update(matched)
        kwargs["ignored_params"] = resolved
        return kwargs

    def _any_noncomposable_AC(self) -> tuple[str, type] | None:
        for n, m in self.pl_module.named_modules():
            if isinstance(m, ActivationWrapper):
                for c in type(m).__mro__:
                    if issubclass(c, ActivationWrapper) and not issubclass(c, FSDPModule):
                        return n, c
        return None

    def _apply_fsdp_plan(self) -> None:
        """Apply the FSDP distributed API according to the name-driven plan provided by the user via
        :attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.fsdp_plan`. This method can be used in
        conjunction with a user-overridden `LightningModule.configure_model` method or without one. Any name-driven
        FSDP API plan directives will be applied after `configure_model` is executed.

        Note: If the provided `fsdp_plan` specifies a module that has already had ``fully_shard`` applied (e.g.
        explicitly via `configure_model`), that directive will be skipped with a warning.
        """
        auto_plan_keys = self.fsdp_plan.keys()
        named_modules = dict(self.pl_module.named_modules())
        has_fsdp_managed = False
        unsuccessful_compositions: list[str] = []
        for n in auto_plan_keys:
            if n in named_modules:
                unsuccessful_fqn = self._compose_or_warn(n)  # returns fqn if composition could not be performed
                if unsuccessful_fqn:
                    unsuccessful_compositions.append(unsuccessful_fqn)
            if not has_fsdp_managed:
                has_fsdp_managed = getattr(named_modules[n], '_is_fsdp_managed_module', False)
        for k in unsuccessful_compositions:
            del self.fsdp_plan[k]  # user has been warned, remove the unsuccessful directives from the active plan
        if non_comp_ac := self._any_noncomposable_AC():
            _compose_ncac(self.pl_module)
            rank_zero_warn(
                "Non-composable activation checkpoint (NCAC) APIs are being used (e.g. `{}` is `{}`). To better "
                "integrate these APIs with the ``fully_shard`` composable distributed API, FTS has composed the "
                "provided LightningModule with a simple adapter to filter out AC-specific parameter prefixes. This is "
                "an experimental feature and may be removed in a future release.\n"
                "If you don't need specific features of the used non-composable AC APIs, using the composable AC API "
                "is recommended instead, i.e. `act_ckpt=('composable', ...)`.".format(non_comp_ac[0], non_comp_ac[1])
            )

    def _wrapped_configure_model(self, cm_func: Callable) -> Callable:
        """If the user has overridden ``configure_model`` in their ``LightningModule``, wrap the user's explicit
        wrapping method with the required
        :class:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter` methods.

        Args:
            cm_func (Callable): The user's overridden ``LightningModule.configure_model`` method

        Returns:
            Callable: The user's overridden ``LightningModule.configure_model`` method wrapped with this
            adapter's internal implementation methods.
        """

        @wraps(cm_func)
        def wrapped_func() -> None:
            cm_func()
            self._post_cm_plan_apply()

        return wrapped_func

    def _post_cm_plan_apply(self) -> None:
        # TODO: determine if we want to provide tp_auto_plan functionality or require manual configure_model for it
        # self.tp_auto_plan()
        if self.fsdp_plan:
            self._apply_fsdp_plan()
