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
Fine-Tuning Scheduler Parallelism Plans
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An FSDP2 plan names the modules to apply ``fully_shard`` to and the keyword arguments for each, so a model can be
sharded from configuration (a YAML file, for instance) rather than by code. The same plan grammar serves the PyTorch
API (:func:`shard_model`) and the Lightning
:class:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter` (its ``fsdp_plan``).

A plan maps keys to ``fully_shard`` keyword arguments. A key is a module name, a regular expression matched from the
start of each module name, or ``class:<name>`` (``class:<module>.<qualname>`` to name the class unambiguously) for
every module that is an instance of that class. An entry's ``ignored_params`` may name parameters by name or regular
expression (or a list of them).

:func:`phase_feedback` reports the FSDP units (one per ``fully_shard`` call) that reshard after forward and whose
parameters thaw in different phases. From the earliest of those phases on, such a unit's backward pass all-gathers all
of its parameters again, those not yet trained included, so splitting it (or not resharding it after forward) saves that
communication. :class:`~finetuning_scheduler.TuningScheduler` and the Lightning
:class:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter` report them when they validate a schedule
against an FSDP2 model (:class:`~finetuning_scheduler.strategies.fsdp2.FSDP2Adapter`).

A tensor-parallel plan (:func:`parallelize_model`, PyTorch API only) uses the same keys, each mapped to one parallel
style and its options, so a plan can live in YAML:

.. code-block:: yaml

    tok_embeddings: {style: rowwise, input_layouts: replicate, output_layouts: {shard: 1}}
    'layers\.\d+\.attention$': {style: prepare_input, input_layouts: {shard: 1}, desired_input_layouts: replicate}
    'layers\.\d+\.attention\.w[qkv]$': {style: colwise, use_local_output: false}
    output:
      style: colwise
      input_layouts: {shard: 1}
      output_layouts: replicate
      loss_parallel: {output_layouts: {shard: -1}, use_local_output: false}

``style`` is ``colwise``, ``rowwise``, ``sequence``, ``prepare_input``, ``prepare_output`` or ``prepare_input_output``,
and the other options are that style's keyword arguments. A layout is ``replicate``, ``{shard: <dim>}`` or null, or a
list of those with one per argument. An entry's ``loss_parallel`` options override the others when the plan is applied
with ``loss_parallel=True``, for use with ``torch.distributed.tensor.parallel.loss_parallel``.

.. warning:: This is an experimental API and may change in a future release.
"""
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel, parallelize_module, ParallelStyle, PrepareModuleInput, PrepareModuleInputOutput,
    PrepareModuleOutput, RowwiseParallel, SequenceParallel,
)

from finetuning_scheduler.core.dist import rank_zero_warn
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.strategies.base import Feedback

# a plan key with this prefix selects modules by class rather than by name
CLASS_SELECTOR = "class:"

# the parallel styles a tensor-parallel plan may name
_TP_STYLES = {"colwise": ColwiseParallel, "rowwise": RowwiseParallel, "sequence": SequenceParallel,
              "prepare_input": PrepareModuleInput, "prepare_output": PrepareModuleOutput,
              "prepare_input_output": PrepareModuleInputOutput}

__all__ = ["CLASS_SELECTOR", "parallelize_model", "phase_feedback", "plan_key_modules", "resolve_fsdp_plan",
           "resolve_ignored_params", "resolve_tp_plan", "shard_model"]


def plan_key_modules(key: str, named_modules: Mapping[str, nn.Module]) -> list[str]:
    """The names of the modules a plan key selects, in module order: a module name, a regex matched from the start
    of each module name, or ``class:<name>`` for every module that is an instance of a class with that name
    (``class:<module>.<qualname>`` to name it unambiguously), the instances ``ModuleWrapPolicy`` selects."""
    if key.startswith(CLASS_SELECTOR):
        ref = key[len(CLASS_SELECTOR):].strip()

        def _is_instance(m: nn.Module) -> bool:
            return any(ref in (c.__name__, f"{c.__module__}.{c.__qualname__}") for c in type(m).__mro__)

        return [n for n, m in named_modules.items() if _is_instance(m)]
    if key in named_modules:
        return [key]
    mpat = re.compile(fr"{key}")
    return [n for n in named_modules if mpat.match(n)]


def resolve_fsdp_plan(model: nn.Module, plan: Mapping[str, Mapping[str, Any] | None],
                      default_kwargs: Mapping[str, Any] | None = None,
                      warn: Callable[[str], None] = rank_zero_warn) -> dict[str, dict[str, Any]]:
    """Expand a plan's keys into the modules they select, each with its ``fully_shard`` kwargs over
    ``default_kwargs``.

    Modules keep the order in which a key first selects them; a later key selecting the same module replaces its
    kwargs. A ``ModuleList`` or ``ModuleDict`` (no ``forward``, so ``fully_shard`` cannot drive it) is skipped with a
    warning.

    Raises:
        FTSConfigurationError: If a key selects no module.
    """
    named_modules = dict(model.named_modules())
    resolved: dict[str, dict[str, Any]] = {}
    for key, kwargs in plan.items():
        matched = plan_key_modules(key, named_modules)
        if not matched:
            what = "class" if key.startswith(CLASS_SELECTOR) else "module or regex"
            raise FTSConfigurationError(
                f"The {what} '{key}' specified in `fsdp_plan` did not match any named modules in the provided model."
            )
        for n in matched:
            if isinstance(named_modules[n], (nn.ModuleList, nn.ModuleDict)):
                warn(f"Provided FSDP module plan includes a directive to apply the FSDP API to ('{n}') which has the "
                     f"unsupported type '{type(named_modules[n])}'. This directive will be pruned and ignored.")
                continue
            resolved[n] = {**(default_kwargs or {}), **(kwargs or {})}
    return resolved


def resolve_ignored_params(model: nn.Module, n: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """``kwargs`` for the plan's module ``n`` with an ``ignored_params`` given by name resolved to the parameters
    it names.

    ``ignored_params`` may be a parameter name or regex (matched from the start of each of ``model``'s parameter names,
    as fine-tuning schedule entries are), or a list of them, resolved among the parameters of the module ``n``
    (``fully_shard`` ignores only those), every name of a tied parameter included. A collection of parameters is passed
    through as a set. FSDP2 neither shards nor reduces the gradients of ignored parameters, so a schedule that thaws
    one trains it independently on each rank unless something else synchronizes it.

    Raises:
        FTSConfigurationError: If ``ignored_params`` mixes names and parameters, or a name or regex matches no
            parameter of the module.
    """
    kwargs = dict(kwargs)
    names = kwargs.get("ignored_params")
    if names is None:
        return kwargs
    names = [names] if isinstance(names, str) else list(names)  # a generator would be consumed by the check below
    if all(isinstance(p, nn.Parameter) for p in names):
        kwargs["ignored_params"] = set(names)
        return kwargs
    if not all(isinstance(p, str) for p in names):
        raise FTSConfigurationError(
            f"The `ignored_params` of the `fsdp_plan` entry for '{n}' mixes parameter names and parameters; give "
            "either names (or regexes) or parameters."
        )
    own = set(model.get_submodule(n).parameters())
    named_params = {pn: p for pn, p in model.named_parameters(remove_duplicate=False) if p in own}
    resolved: set[nn.Parameter] = set()
    for pattern in names:
        if pattern in named_params:
            matched = [named_params[pattern]]
        else:
            mpat = re.compile(fr"{pattern}")
            matched = [p for pn, p in named_params.items() if mpat.match(pn)]
        if not matched:
            raise FTSConfigurationError(
                f"The parameter name or regex '{pattern}' in the `ignored_params` of the `fsdp_plan` entry for "
                f"'{n}' did not match any parameter of that module."
            )
        resolved.update(matched)
    kwargs["ignored_params"] = resolved
    return kwargs


# the_dtype-valued kwargs of `MixedPrecisionPolicy`, which a plan may give by name so a YAML plan can use them
_MP_DTYPES = ("param_dtype", "reduce_dtype", "output_dtype")
_ACT_CKPT_MODES = ("composable", "wrapped", "wrapped_offload")


def _resolve_mp_policy(n: str, kwargs: dict[str, Any]) -> None:
    """Resolve an ``mp_policy`` given as a mapping into a ``MixedPrecisionPolicy`` in place.

    The dtype-valued entries may be ``torch.dtype`` objects or their names (``"bfloat16"``), so a plan written in
    YAML can carry a mixed precision policy. A ``MixedPrecisionPolicy`` is passed through.
    """
    mp = kwargs.get("mp_policy")
    if mp is None or isinstance(mp, MixedPrecisionPolicy):
        return
    if not isinstance(mp, Mapping):
        raise FTSConfigurationError(
            f"The `mp_policy` of the `fsdp_plan` entry for '{n or '<root>'}' must be a mapping of"
            " `MixedPrecisionPolicy` kwargs (or a `MixedPrecisionPolicy`)."
        )
    mp = dict(mp)
    for k in _MP_DTYPES:
        if isinstance(mp.get(k), str):
            try:
                mp[k] = getattr(torch, mp[k])
            except AttributeError:
                raise FTSConfigurationError(
                    f"The `mp_policy` of the `fsdp_plan` entry for '{n or '<root>'}' names no torch dtype:"
                    f" {k}={mp[k]!r}."
                ) from None
    kwargs["mp_policy"] = MixedPrecisionPolicy(**mp)


def _apply_act_ckpt(model: nn.Module, n: str, act_ckpt: Any) -> None:
    """Apply the activation checkpointing a plan entry asks for to the module ``n``, before it is sharded.

    ``wrapped`` and ``wrapped_offload`` replace the module in its parent with a ``checkpoint_wrapper`` /
    ``offload_wrapper``, as the Lightning adapter does: the wrapped parameters then carry the wrapper's
    ``_checkpoint_wrapped_module.`` prefix, which a fine-tuning schedule must use to name them. The root has no
    parent to replace it in, so the wrapped modes refuse it.
    """
    if isinstance(act_ckpt, Mapping) or not act_ckpt:
        raise FTSConfigurationError(f"`act_ckpt` for '{n or '<root>'}' must be a mode or a [mode, kwargs] list.")
    mode, ac_kwargs = (act_ckpt, None) if isinstance(act_ckpt, str) else (act_ckpt[0], _second(act_ckpt))
    ac_kwargs = dict(ac_kwargs or {})
    if mode not in _ACT_CKPT_MODES:
        raise FTSConfigurationError(
            f"`act_ckpt` mode '{mode}' for '{n or '<root>'}' is not supported; use one of"
            f" {', '.join(repr(m) for m in _ACT_CKPT_MODES)}."
        )
    if mode == "composable":
        from torch.distributed._composable import checkpoint

        checkpoint(model.get_submodule(n), **ac_kwargs)
        return
    if not n:
        raise FTSConfigurationError(
            f"`act_ckpt` mode '{mode}' wraps a module in its parent, so it cannot wrap the root module itself;"
            " name the modules to checkpoint in the plan (or use 'composable')."
        )
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, offload_wrapper

    wrap = checkpoint_wrapper if mode == "wrapped" else offload_wrapper
    prefix, _, name = n.rpartition(".")
    parent = model.get_submodule(prefix) if prefix else model
    setattr(parent, name, wrap(model.get_submodule(n), **ac_kwargs))


def _apply(model: nn.Module, n: str, mesh: Any, kwargs: Mapping[str, Any]) -> None:
    """``fully_shard`` the module ``n`` with ``kwargs``, applying activation checkpointing first if asked and
    resolving the ``cpu_offload_policy`` and ``mp_policy`` aliases.

    ``ignored_params`` are resolved first, by the
    names the parameters have before any checkpoint wrapper renames them.
    """
    kwargs = resolve_ignored_params(model, n, kwargs)
    act_ckpt = kwargs.pop("act_ckpt", None)
    offload = kwargs.pop("cpu_offload_policy", None)
    if offload is not None:
        kwargs["offload_policy"] = CPUOffloadPolicy(**offload)
    _resolve_mp_policy(n, kwargs)
    if act_ckpt is not None:
        _apply_act_ckpt(model, n, act_ckpt)
    fully_shard(model.get_submodule(n), mesh=mesh, **kwargs)


def _second(seq: Sequence[Any]) -> Any:
    return seq[1] if len(seq) > 1 else None


def _sharded_ancestor(model: nn.Module, n: str) -> str | None:
    """The name of the nearest-to-root ancestor of the module ``n`` that is already sharded, if any."""
    parts = n.split(".") if n else []
    for i in range(len(parts)):
        if isinstance(model.get_submodule(".".join(parts[:i])), FSDPModule):
            return ".".join(parts[:i])
    return None


def _reshards_after_forward(unit: FSDPModule, is_outermost: bool) -> bool:
    """Whether ``unit`` frees its unsharded parameters after forward (so backward all-gathers them again).

    ``fully_shard`` without ``reshard_after_forward`` reshards every unit but the root (the outermost unit forward runs
    through), which it keeps unsharded for backward. Read from FSDP2's state, not a public API, so a unit whose state
    cannot be read is assumed to reshard.
    """
    try:
        state = unit._get_fsdp_state()
        # a unit has one parameter group per mesh (several with per-parameter meshes)
        groups = getattr(state, "_fsdp_param_groups", None)
        if groups is None:
            groups = [g] if (g := state._fsdp_param_group) is not None else []
        if not groups or (is_outermost and getattr(state, "_auto_reshard_after_forward", False)):
            return False
        return any(g.post_forward_mesh_info is not None for g in groups)
    except (AttributeError, AssertionError):
        return True


def _managed(p: nn.Parameter) -> bool:
    """Whether an FSDP unit manages ``p``: it shards its parameters into ``DTensor``s and leaves those it ignores
    plain."""
    return isinstance(p, DTensor)


def phase_feedback(model: nn.Module, phases: Mapping[int, Iterable[str]]) -> list[Feedback]:
    """Report the FSDP units of ``model`` that reshard after forward and whose parameters thaw in different phases.

    A unit is a module ``fully_shard`` was applied to, managing its parameters that no nested unit manages (parameters
    it ignores, left plain, are not among them). From the earliest phase that thaws one of its parameters, its backward
    pass all-gathers all of them again after forward freed them, those not yet trained included. A unit kept unsharded
    after forward (the root, by default) pays no such cost, nor does gradient reduction, which covers only parameters
    with gradients. ``phases`` maps each phase's depth to the physical parameter names it thaws
    (``model.named_parameters()`` names); a parameter in no phase is never trained.

    Args:
        model: The sharded model.
        phases: Each phase's physical parameter names.

    Returns:
        A ``warning`` per unit that reshards after forward and whose parameters thaw in more than one phase (or are
        partly never trained).
    """
    phase_of: dict[str, int] = {}
    for depth, names in sorted(phases.items()):
        for pn in names:
            phase_of.setdefault(pn, depth)
    name_of = {p: pn for pn, p in model.named_parameters()}
    units = [(n, m) for n, m in model.named_modules() if isinstance(m, FSDPModule)]
    outermost = {n for n, _ in units if not any(o != n and (o == "" or n.startswith(f"{o}.")) for o, _ in units)}
    claimed: set[nn.Parameter] = set()
    feedback: list[Feedback] = []
    for n, unit in reversed(units):  # innermost first, so each unit holds only what no nested unit claimed
        held = [p for p in unit.parameters() if p not in claimed and _managed(p)]
        claimed.update(held)
        depths = {phase_of.get(name_of[p]) for p in held}
        if len(depths) > 1 and _reshards_after_forward(unit, n in outermost):
            shown = ", ".join("never trained" if d is None else f"phase {d}"
                              for d in sorted(depths, key=lambda d: (d is None, d or 0)))
            feedback.append(Feedback("warning", (
                f"The FSDP unit '{n or '<root>'}' reshards after forward and holds parameters that thaw at different "
                f"points ({shown}), so from the earliest of those phases on, every backward pass all-gathers all of "
                "them again, those not yet trained included. Applying `fully_shard` to submodules whose parameters "
                "thaw together (or `reshard_after_forward=False` for this unit) avoids that.")))
    return feedback


def shard_model(model: nn.Module, mesh: Any, plan: Mapping[str, Mapping[str, Any] | None] | None = None, *,
                default_kwargs: Mapping[str, Any] | None = None, shard_root: bool = True) -> nn.Module:
    """Apply ``fully_shard`` to the modules an FSDP2 plan selects, then to ``model`` itself.

    Modules are sharded deepest first, so a plan may list a module before its children; entries at the same depth keep
    plan order (see :func:`resolve_fsdp_plan`). A module already sharded (by the caller, say) is left as it is. An
    ``ignored_params`` given by name belongs in a plan entry: in ``default_kwargs`` it would have to name a parameter
    of every planned module. With ``shard_root``, ``model`` is sharded last, so every parameter becomes a ``DTensor``:
    a parameter outside every FSDP unit would be neither sharded nor gradient-synchronized across data-parallel
    ranks. An entry's ``act_ckpt`` applies activation checkpointing to the module before it is sharded:
    ``"composable"`` (composable checkpointing, optionally with its kwargs), or ``"wrapped"`` / ``"wrapped_offload"``
    (a ``checkpoint_wrapper`` / ``offload_wrapper`` replacing the module in its parent, whose parameters then carry
    the wrapper's ``_checkpoint_wrapped_module.`` prefix, which a fine-tuning schedule must use to name them).
    ``cpu_offload_policy`` (a mapping of ``CPUOffloadPolicy`` kwargs) becomes its ``offload_policy``, and an
    ``mp_policy`` given as a mapping becomes a ``MixedPrecisionPolicy``, its dtypes by name or as ``torch.dtype``.

    Args:
        model: The model to shard (on the meta device or materialized), after any tensor-parallel plan.
        mesh: The data-parallel ``DeviceMesh``.
        plan: Plan keys mapped to ``fully_shard`` kwargs (or ``None``); see the module docstring for the key grammar.
        default_kwargs: Kwargs every planned module takes unless its entry overrides them; all but ``act_ckpt`` also
            apply to the root.
        shard_root: Shard ``model`` itself after the planned modules.

    Returns:
        ``model``, sharded.

    Raises:
        FTSConfigurationError: If a plan key selects no module, a planned module's ancestor was already sharded by
            the caller, an ``ignored_params`` name matches no parameter of its module, ``act_ckpt`` names an
            unsupported mode (or a wrapped mode for the root), or an ``mp_policy`` mapping names no torch dtype.
    """
    resolved = resolve_fsdp_plan(model, plan or {}, default_kwargs)
    # deepest first, so children shard before their parents however the plan lists them; the sort is stable, so
    # entries at one depth keep plan order
    for n, kwargs in sorted(resolved.items(), key=lambda kv: -kv[0].count(".")):
        if isinstance(model.get_submodule(n), FSDPModule):
            continue
        if (ancestor := _sharded_ancestor(model, n)) is not None:
            raise FTSConfigurationError(
                f"`shard_model` cannot shard '{n}': its ancestor '{ancestor or '<root>'}' is already sharded. Shard a "
                "module's children before any ancestor of theirs (the plan's own entries are applied deepest first)."
            )
        _apply(model, n, mesh, kwargs)
    if shard_root and not isinstance(model, FSDPModule):
        _apply(model, "", mesh, {k: v for k, v in (default_kwargs or {}).items() if k != "act_ckpt"})
    return model


def _tp_layout(n: str, value: Any) -> Any:
    """A placement from its plan form, recursing into the per-argument list a ``prepare_*`` style takes."""
    if value is None or isinstance(value, Placement):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_tp_layout(n, v) for v in value)
    if value == "replicate":
        return Replicate()
    if isinstance(value, Mapping) and set(value) == {"shard"} and isinstance(value["shard"], int):
        return Shard(value["shard"])
    raise FTSConfigurationError(
        f"The layout {value!r} in the `tp_plan` entry for '{n or '<root>'}' is not 'replicate', {{shard: <dim>}} or "
        "null, or a list of those."
    )


def _tp_style(n: str, entry: Any, loss_parallel: bool) -> ParallelStyle:
    """A new parallel style for the module ``n`` from its plan entry: a style may hold per-module state, so modules
    a key selects together never share one."""
    if not isinstance(entry, Mapping) or entry.get("style") not in _TP_STYLES:
        raise FTSConfigurationError(
            f"The `tp_plan` entry for '{n or '<root>'}' must be a mapping whose `style` is one of "
            f"{', '.join(repr(s) for s in _TP_STYLES)}; got {entry!r}."
        )
    options = {k: v for k, v in entry.items() if k not in ("style", "loss_parallel")}
    if loss_parallel:
        options.update(entry.get("loss_parallel") or {})
    for option, value in options.items():
        if option.endswith("_kwarg_layouts"):
            options[option] = {k: _tp_layout(n, v) for k, v in value.items()}
        elif option.endswith("_layouts"):
            options[option] = _tp_layout(n, value)
    try:
        return _TP_STYLES[entry["style"]](**options)
    except TypeError as err:
        raise FTSConfigurationError(
            f"The `tp_plan` entry for '{n or '<root>'}' has an option its style '{entry['style']}' does not take: {err}"
        ) from err


def resolve_tp_plan(model: nn.Module, plan: Mapping[str, Mapping[str, Any]], *, loss_parallel: bool = False,
                    warn: Callable[[str], None] = rank_zero_warn) -> dict[str, ParallelStyle]:
    """Expand a tensor-parallel plan's keys into the modules they select, each with a new parallel style.

    Modules keep the order in which a key first selects them; a later key selecting the same module replaces its style.
    A ``ModuleList`` or ``ModuleDict`` is skipped with a warning, as :func:`resolve_fsdp_plan` skips one. See the module
    docstring for the plan grammar.

    Args:
        model: The model the plan is for.
        plan: Plan keys mapped to parallel styles and their options.
        loss_parallel: Apply each entry's ``loss_parallel`` options.
        warn: Called with the warning for a skipped module.

    Returns:
        Each selected module's name mapped to its parallel style.

    Raises:
        FTSConfigurationError: If a key selects no module, an entry or layout is malformed or names an option its style
            does not take, or ``loss_parallel`` is set and no entry has ``loss_parallel`` options.
    """
    if loss_parallel and not any(isinstance(e, Mapping) and e.get("loss_parallel") for e in plan.values()):
        raise FTSConfigurationError(
            "`loss_parallel=True` needs a `tp_plan` entry with `loss_parallel` options: the module computing the "
            "logits must keep them sharded for `torch.distributed.tensor.parallel.loss_parallel`."
        )
    named_modules = dict(model.named_modules())
    resolved: dict[str, ParallelStyle] = {}
    for key, entry in plan.items():
        matched = plan_key_modules(key, named_modules)
        if not matched:
            what = "class" if key.startswith(CLASS_SELECTOR) else "module or regex"
            raise FTSConfigurationError(
                f"The {what} '{key}' specified in `tp_plan` did not match any named modules in the provided model."
            )
        for n in matched:
            if isinstance(named_modules[n], (nn.ModuleList, nn.ModuleDict)):
                warn(f"Provided TP module plan includes a directive to apply a parallel style to ('{n}') which has the "
                     f"unsupported type '{type(named_modules[n])}'. This directive will be pruned and ignored.")
                continue
            resolved[n] = _tp_style(n, entry, loss_parallel)
    return resolved


def parallelize_model(model: nn.Module, tp_mesh: Any, plan: Mapping[str, Mapping[str, Any]], *,
                      loss_parallel: bool = False) -> nn.Module:
    """Apply the parallel styles a tensor-parallel plan selects, with ``parallelize_module``.

    The model may be on the meta device or materialized, and parameter names do not change, so a fine-tuning schedule
    names the same parameters it would without the plan. For both tensor and data parallelism, apply this first over
    ``mesh["tensor_parallel"]`` and then :func:`shard_model` over ``mesh["data_parallel"]``; only a one-dimensional
    tensor-parallel mesh over all ranks is tested so far.

    Args:
        model: The model to parallelize.
        tp_mesh: A one-dimensional ``DeviceMesh``.
        plan: Plan keys mapped to parallel styles and their options; see the module docstring for the grammar.
        loss_parallel: Apply each entry's ``loss_parallel`` options, to keep the logits sharded for
            ``torch.distributed.tensor.parallel.loss_parallel``.

    Returns:
        ``model``, parallelized.

    Raises:
        FTSConfigurationError: If ``tp_mesh`` is not one-dimensional, or :func:`resolve_tp_plan` refuses the plan.
    """
    if tp_mesh.ndim != 1:
        raise FTSConfigurationError(
            f"`parallelize_model` needs a one-dimensional mesh, not one with {tp_mesh.ndim} dimensions: pass the "
            "tensor-parallel dimension (`mesh['tensor_parallel']`)."
        )
    return parallelize_module(model, tp_mesh, resolve_tp_plan(model, plan, loss_parallel=loss_parallel))
