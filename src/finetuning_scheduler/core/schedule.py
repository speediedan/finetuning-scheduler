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
Fine-Tuning Scheduler Core Schedules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Framework-neutral fine-tuning schedule loading, saving, generation, parsing and validation.

These functions operate on an explicit schedule dictionary (mutating it in place where noted) rather than on interface
state, so any training interface can use them.
"""
import itertools
import os
import pathlib
import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any

import yaml
from torch import nn
from venngram import AddressBook
from venngram.errors import VenngramError

from finetuning_scheduler.core._io import get_filesystem
from finetuning_scheduler.core.dist import rank_zero_info, rank_zero_only, rank_zero_warn
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.roles import parse_role

NON_CONTIGUOUS_KEYS_MSG = "The supplied schedule was found to have non-contiguous or non-zero-indexed keys "


class UniqueKeyLoader(yaml.SafeLoader):
    """Alters SafeLoader to enable duplicate key detection by the SafeConstructor."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        """Overrides the construct_mapping method of the SafeConstructor to raise a ValueError if duplicate keys
        are found.

        Inspired by and adapated from https://stackoverflow.com/a/63215043
        """
        mapping = []
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key not in mapping:
                mapping.append(key)
            else:
                raise ValueError(key)
        return super().construct_mapping(node, deep)


def find_shared_parameters(module: nn.Module) -> list[list[str]]:
    """Return the names of each set of parameters shared (tied) within ``module``, one list per shared tensor."""
    return _find_shared_parameters(module)


def _find_shared_parameters(module: nn.Module, tied_parameters: dict | None = None, prefix: str = "") -> list:
    if tied_parameters is None:
        tied_parameters = {}
    for name, param in module._parameters.items():
        param_prefix = prefix + ("." if prefix else "") + name
        if param is None:
            continue
        if param not in tied_parameters:
            tied_parameters[param] = []
        tied_parameters[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        _find_shared_parameters(m, tied_parameters, submodule_prefix)
    return [x for x in tied_parameters.values() if len(x) > 1]


@rank_zero_only
def save_schedule(schedule_name: str, layer_config: dict, dump_loc: str | os.PathLike) -> os.PathLike:
    """Save a fine-tuning schedule to ``dump_loc`` as YAML, on global rank zero only.

    Args:
        schedule_name: The file name of the schedule.
        layer_config: The schedule dictionary.
        dump_loc: The directory to which the schedule (.yaml) should be written.

    Returns:
        The path of the saved schedule (``None`` on other ranks).
    """
    dump_path = pathlib.Path(dump_loc)
    dump_path.mkdir(exist_ok=True, parents=True)
    ft_schedule_yaml = dump_path / schedule_name
    fs = get_filesystem(ft_schedule_yaml)
    with fs.open(ft_schedule_yaml, "w", newline="") as fp:
        yaml.dump(layer_config, fp)
    assert os.access(ft_schedule_yaml, os.F_OK)
    rank_zero_info(f"fine-tuning schedule dumped to {ft_schedule_yaml}.")
    return ft_schedule_yaml


def build_default_schedule(module: nn.Module) -> dict:
    """Build the default fine-tuning schedule for ``module``: parameters are thawed in pairs, starting from the
    last registered parameter.

    Args:
        module: The module for which to build a schedule.

    Returns:
        The schedule, mapping each phase to its ``params``.
    """
    # Note: This initial default fine-tuning schedule generation approach is intentionally simple/naive but is
    # effective for a surprising fraction of models. Future versions of this callback may use module
    # introspection to generate default schedules that better accommodate more complex structures and
    # specific architectures if the callback proves sufficiently useful.
    param_lists: list = []
    cur_group: list = []
    model_params = list(module.named_parameters())[::-1]
    for i, (n, _) in enumerate(model_params):
        if i % 2 == 0:
            cur_group = []
            cur_group.append(n)
        else:
            cur_group.append(n)
            param_lists.append(cur_group)
    if len(model_params) % 2 == 1:
        param_lists.append([model_params[-1][0]])
    layer_config = {}
    for i, param_l in enumerate(param_lists):
        layer_config[i] = {"params": param_l}
    return layer_config


def gen_ft_schedule(module: nn.Module, dump_loc: str | os.PathLike) -> os.PathLike | None:
    """Generate the default fine-tuning schedule for ``module`` (see :func:`build_default_schedule`) and save it to
    ``dump_loc``.

    Args:
        module: The module for which a fine-tuning schedule will be generated.
        dump_loc: The directory to which the generated schedule (.yaml) should be written.

    Returns:
        The path to the generated schedule.
    """
    rank_zero_info(f"Proceeding with dumping default fine-tuning schedule for {module.__class__.__name__}")
    layer_config = build_default_schedule(module)
    schedule_name = f"{module.__class__.__name__}_ft_schedule.yaml"
    assert dump_loc is not None
    return save_schedule(schedule_name, layer_config, dump_loc)


def load_yaml_schedule(schedule_yaml_file: str | os.PathLike) -> dict:
    """Load a schedule defined in a .yaml file and transform it into a dictionary.

    Args:
        schedule_yaml_file: The .yaml fine-tuning schedule file.

    Raises:
        FTSConfigurationError: If the specified schedule file is not found or contains duplicate keys.

    Returns:
        The dictionary representation of the fine-tuning schedule.
    """
    try:
        with open(schedule_yaml_file) as df:
            schedule_dict = yaml.load(df, Loader=UniqueKeyLoader)
    except FileNotFoundError as fnf:
        error_msg = (
            f"Could not find specified fine-tuning scheduling file '{schedule_yaml_file}': {fnf}."
            f"Please reconfigure and try again."
        )
        rank_zero_warn(error_msg)
        raise FTSConfigurationError(error_msg)
    except ValueError as dup_key:
        error_msg = (
            f"Duplicate key ({dup_key.args[0]}) found in supplied schedule: {schedule_yaml_file}'. Please validate "
            "schedule before resubmitting."
        )
        rank_zero_warn(error_msg)
        raise FTSConfigurationError(error_msg)
    return schedule_dict


ADDRESSES_KEY = "addresses"


def split_addresses(schedule: Mapping) -> tuple[dict, Any]:
    """Separate a schedule's top-level ``addresses`` key (its named sets) from its phases.

    Returns:
        The schedule without the key (a new dictionary), and the key's value (``None`` when absent).
    """
    phases = {k: v for k, v in schedule.items() if k != ADDRESSES_KEY}
    return phases, schedule.get(ADDRESSES_KEY)


def load_address_sets(raw: Any, base_dir: str | os.PathLike | None = None) -> dict[str, Any]:
    """The named sets a schedule's ``addresses`` key provides, for ``@<set>`` references.

    Args:
        raw: The key's value: a path to an address book (or a list of paths, whose sets must not share a name), or an
            inline book (a mapping with ``sets`` and optionally ``imports``; ``schema_version`` may be omitted).
        base_dir: The directory relative paths resolve against (the schedule file's directory, when it came from one).

    Raises:
        FTSConfigurationError: If a book cannot be loaded, two books define the same set, or ``raw`` has another type.
    """
    if raw is None:
        return {}
    base = pathlib.Path(base_dir) if base_dir is not None else pathlib.Path(".")
    try:
        if isinstance(raw, Mapping):
            return AddressBook.from_mapping(raw, source="the schedule's `addresses`", base_dir=base, inline=True).sets()
        paths = [raw] if isinstance(raw, (str, os.PathLike)) else raw
        if not isinstance(paths, list) or not all(isinstance(p, (str, os.PathLike)) for p in paths):
            raise FTSConfigurationError(
                f"A schedule's `addresses` is an address book's path, a list of paths, or an inline book (a mapping"
                f" with `sets`), got {raw!r}.")
        sets: dict[str, Any] = {}
        for path in paths:
            book = AddressBook.from_file(base / path)
            clash = sorted(set(sets) & set(book.sets()))
            if clash:
                raise FTSConfigurationError(f"The address books in `addresses` both define {clash}; import one of them "
                                            "under a namespace instead.")
            sets.update(book.sets())
        return sets
    except VenngramError as err:
        raise FTSConfigurationError(f"The schedule's `addresses` cannot be loaded: {err}") from err


def check_param_entries(schedule: Mapping[int, Mapping[str, Any]]) -> None:
    """Refuse a ``params`` entry that is not a string: an address written as a mapping belongs in the schedule's
    ``addresses`` sets, where an entry references it by ``@<set>``.

    Raises:
        FTSConfigurationError: Naming the phase and the entry.
    """
    for depth, phase in schedule.items():
        for entry in phase.get("params", []) or []:
            if not isinstance(entry, str):
                raise FTSConfigurationError(
                    f"Phase {depth}'s `params` entry {entry!r} is not a string. Write a parameter name, a regex or an "
                    "address in its string form (`role:adapter & layers:-4:`); an address in mapping form goes in the "
                    "schedule's `addresses` sets and is referenced as `@<set>`.")


def convert_phase_keys(schedule: dict) -> tuple[dict, bool]:
    """Return ``schedule`` with integer phase keys, converting them where possible.

    Args:
        schedule: The fine-tuning schedule.

    Raises:
        FTSConfigurationError: If a phase key is not convertible to an integer.

    Returns:
        The schedule with integer keys (a new dictionary), and whether any key was converted (in which case a warning
        naming the converted keys has been emitted).
    """
    try:
        orig_keys = set(schedule.keys())
        converted = {int(k): v for k, v in schedule.items()}
    except ValueError as value_err:
        raise FTSConfigurationError(
            "The supplied schedule was found to use one or more keys that were not convertible to integers. "
            f"The encountered error was: {value_err}"
        )
    key_diff = set(converted.keys()) ^ orig_keys
    if key_diff:
        rank_zero_warn(
            "Note, the specified fine-tuning schedule had non-integer keys implicitly converted to "
            f"integers. Key diff: {key_diff}"
        )
    return converted, bool(key_diff)


def ensure_contiguous_phase_keys(schedule: dict) -> bool:
    """Renumber the phases of ``schedule`` in place to be zero-based and contiguous, preserving their order.

    Args:
        schedule: The fine-tuning schedule, with integer phase keys.

    Returns:
        Whether the keys were already zero-based and contiguous (``False`` means the schedule was renumbered).
    """
    contiguous = len(schedule.keys()) == (max(schedule.keys()) + 1)
    if not contiguous:
        for i, k in enumerate(sorted(schedule.keys())):
            schedule[i] = schedule.pop(k)
    return contiguous


def validate_epoch_transitions(schedule: dict) -> None:
    """Ensure every phase of ``schedule`` specifies an epoch-driven transition (``max_transition_epoch``).

    Raises:
        FTSConfigurationError: If any phase lacks a ``max_transition_epoch``.
    """
    missing_transitions = [d for d in schedule.keys() if schedule[d]["max_transition_epoch"] < 0]
    if missing_transitions:
        raise FTSConfigurationError(
            f"epoch_transitions_only specified but some phases "
            f"({', '.join(str(d) for d in missing_transitions)}) are missing a "
            "max_transition_epoch. Please unset epoch_transitions_only or "
            "specify a max_transition_epoch for each phase."
        )


def parse_phase_lr(schedule: dict, depth: int, base_max_lr: float) -> None:
    """Parse and default the base learning rate of phase ``depth`` in place.

    Phases after the first default to ``base_max_lr``; a learning rate given for phase 0 is removed with a warning,
    because the initial optimizer configuration determines it.

    Raises:
        FTSConfigurationError: If the specified learning rate is not convertible to a float.
    """
    if depth > 0:
        schedule[depth].setdefault("lr", base_max_lr)
        try:
            float(schedule[depth]["lr"])
        except ValueError:
            raise FTSConfigurationError(
                f"The lr '{schedule[depth]['lr']}' in phase {depth} of the provided explicit schedule"
                " could not be cast to a float. Specified learning rates must be convertable to a float."
            )
    else:
        if schedule[depth].get("lr", None):
            rank_zero_warn(
                f"A lr for fine-tuning phase 0 has been specified ({schedule[0]['lr']}). This"
                " lr will be overridden by the lr specified via the initial optimizer configuration"
                " (typically in `configure_optimizers()`)."
            )
            del schedule[depth]["lr"]


def resolve_param_entries(entries: Iterable[str], depth: int, named_params: Collection[str],
                          shared_params: tuple, passthrough: Callable[[str], bool] | None = None) -> list[str]:
    """Resolve a phase's parameter names and regexes to fully qualified parameter names, without modifying the
    schedule.

    A name present in ``named_params`` resolves to itself; any other entry is a regex matched (``re.match``) against
    every name. Explicitly specified copies of shared parameters are pruned with a warning, since they are thawed with
    their registered source parameter.

    Args:
        entries: The phase's ``params`` entries.
        depth: The phase, for messages.
        named_params: The names entries resolve against.
        shared_params: The model's shared parameter names, as a tuple of (a list of per-tensor name lists, a set of
            all shared names).
        passthrough: Entries for which this returns ``True`` (``role:`` selectors, for example) are kept as they are.

    Returns:
        The resolved names, in entry order.

    Raises:
        FTSConfigurationError: If a specified parameter or regex matches no parameter.
    """
    resolved_params = []
    for p in entries:
        if passthrough is not None and passthrough(p):
            resolved_params.append(p)
            continue
        regex_params = []
        explicit_params = False
        if p in named_params:
            explicit_params = True
            resolved_params.append(p)
        else:
            ppat = re.compile(fr"{p}")
            regex_params = [n for n in named_params if ppat.match(n)]
            resolved_params.extend(regex_params)
        if not (regex_params or explicit_params):
            if p in shared_params[1]:
                pruning_param_msg = (
                    f"Pruning explicitly specified shared parameter {p} from provided schedule (it will be thawed "
                    f" when its registered source parameter {[pl[0] for pl in shared_params[0] if p in pl][0]} is"
                    " thawed."
                )
                rank_zero_warn(pruning_param_msg)
            else:
                raise FTSConfigurationError(
                    f"The parameter or regex '{p}' specified in phase {depth} of the "
                    "provided explicit schedule did not match any named parameter in the "
                    "model."
                )
    return resolved_params


def resolve_phase_params(schedule: dict, depth: int, named_params: Collection[str], shared_params: tuple,
                         passthrough: Callable[[str], bool] | None = None) -> None:
    """Expand the parameter names and regexes of phase ``depth`` in place to fully qualified parameter names (see
    :func:`resolve_param_entries`).

    Args:
        schedule: The fine-tuning schedule.
        depth: The phase to resolve.
        named_params: The names of the model's parameters.
        shared_params: The model's shared parameter names, as a tuple of (a list of per-tensor name lists, a set of
            all shared names).
        passthrough: Entries for which this returns ``True`` are kept as they are.

    Raises:
        FTSConfigurationError: If a specified parameter or regex matches no parameter.
    """
    schedule[depth]["params"] = resolve_param_entries(schedule[depth].get("params", []), depth, named_params,
                                                      shared_params, passthrough)


def shared_params_ref(module: nn.Module) -> tuple:
    """Return the shared-parameter reference :func:`resolve_phase_params` expects for ``module``."""
    model_shared_params = find_shared_parameters(module)
    return tuple((model_shared_params, set(itertools.chain(*model_shared_params))))


def validate_phases_disjoint(schedule: dict) -> None:
    """Validate that no parameter is specified in more than one phase of ``schedule``.

    Raises:
        FTSConfigurationError: Listing the parameters specified in more than one phase.
    """
    phase_lists = [schedule[d]["params"] for d in schedule.keys()]
    params = Counter(list(itertools.chain(*phase_lists)))
    unique_params = Counter(list(set().union(*phase_lists)))
    params.subtract(unique_params)
    dup_params = list(params.elements())
    if dup_params:
        raise FTSConfigurationError(
            f"Phases are not disjoint. The following parameters are specified in "
            f"multiple phases: {', '.join(dup_params)}"
        )


def validate_reinit_keys(target_sched: dict, target_key: str, depth: int | None = None) -> None:
    """Key validation common to all reinitialization configuration dictionaries.

    Args:
        target_sched: The reinitialization configuration, for an implicit mode schedule or one explicit mode phase.
        target_key: The expected reinitialization key for the current parsing context.
        depth: If parsing an explicit schedule, the current phase.

    Raises:
        FTSConfigurationError: If the expected reinitialization key is missing, or its configuration does not specify
            a ``class_path``.
    """
    if target_key not in target_sched.keys():
        phase_specific_msg = "" if not depth else f"for phase {depth}"
        key_specific_msg = "a lr scheduler" if target_key == "lr_scheduler_init" else "an optimizer"
        raise FTSConfigurationError(
            f"Specifying {key_specific_msg} configuration to reinitialize with requires a valid configuration "
            f"dictionary be provided via a `{target_key}` key but no such key was found " + phase_specific_msg + "."
        )
    if not target_sched[target_key].get("class_path"):
        phase_specific_msg = "the specified config." if not depth else f"the specified phase ({depth})."
        raise FTSConfigurationError(
            f"Specifying `{target_key}` requires at least a  `class_path` to be specified but this is not the case "
            "for " + phase_specific_msg
        )


def parse_reinit_pg_lrs(depth: int, init_pg_lrs: list) -> None:
    """Validate the per-parameter-group initial learning rates of an lr scheduler reinitialization.

    Raises:
        FTSConfigurationError: If any of the learning rates is not convertible to a float.
    """
    for lr in init_pg_lrs:
        try:
            float(lr)
        except ValueError:
            raise FTSConfigurationError(
                f"Not all of the lrs specified in `init_pg_lrs`: ({init_pg_lrs}) associated with phase {depth} of "
                "the provided explicit schedule could be cast to a float. Specified learning rates must be "
                "convertable to a float."
            )


def update_epoch_watermark(schedule: dict, depth: int, max_epoch_wm: int, epoch_transitions_only: bool) -> int:
    """Raise the schedule's epoch watermark to phase ``depth``'s ``max_transition_epoch``, reporting a phase whose
    ``max_transition_epoch`` does not exceed an earlier phase's (it may then run for a single epoch).

    Phase 0 does not participate: callers start at depth 1.

    Returns:
        The updated watermark.
    """
    curr_max_epoch = schedule[depth]["max_transition_epoch"]
    if 0 <= curr_max_epoch <= max_epoch_wm:
        es_addendum = " depending upon EarlyStopping criteria."
        rank_zero_info(
            f"Specified max_transition_epoch of depth {depth}"
            f"({schedule[depth]['max_transition_epoch']}) is less than or equal to a "
            f"previous max_transition_epoch ({max_epoch_wm}), depth may execute only a single "
            f"epoch{'.' if epoch_transitions_only else es_addendum}"
        )
    return max(max_epoch_wm, curr_max_epoch)


def collect_reinit_cfgs(schedule: dict, implicit_cfgs: dict[str, dict | None]) -> dict[str, dict]:
    """Gather each reinitialization directive: the implicit-mode configuration if given, otherwise the per-phase
    configurations keyed by depth.

    Args:
        schedule: The fine-tuning schedule.
        implicit_cfgs: Each reinitialization key (``new_optimizer``, ``new_lr_scheduler``) mapped to its implicit-mode
            configuration, or ``None``.
    """
    return {
        reinit_k: implicit or {d: schedule[d].get(reinit_k) for d in schedule.keys() if schedule[d].get(reinit_k)}
        for reinit_k, implicit in implicit_cfgs.items()
    }


def refuse_phase0_reinit(reinit_cfg: dict[str, dict]) -> None:
    """Refuse a reinitialization directive on the initial phase, whose configuration belongs to the model itself.

    Raises:
        FTSConfigurationError: If any directive targets phase 0.
    """
    for rk, rp in reinit_cfg.items():
        if isinstance(rp, dict) and 0 in rp.keys():
            raise FTSConfigurationError(
                f"You have specified a `{rk}` reinitialization directive for the initial training phase which is an "
                "invalid configuration. The initial configuration should be passed to your LightningModule."
            )


def check_schedule_vs_reinit_lr(explicit_schedule: bool, reinit_lr_cfg: dict | None) -> None:
    """Refuse an implicit-mode lr scheduler reinitialization configuration alongside an explicit schedule.

    Raises:
        FTSConfigurationError: If both are given.
    """
    if explicit_schedule and reinit_lr_cfg:
        error_msg = (
            "Specifying both `ft_schedule` and `reinit_lr_cfg` is an invalid configuration. `reinit_lr_cfg` "
            "specifies an lr scheduler configuration to reinitialize with at every new phase of an implicitly "
            "defined fine-tuning shedule whereas `ft_schedule` is an explicity defined schedule. To reinitialize "
            "a given lr scheduler configuration with an explicit fine-tuning schedule, please add the desired "
            "lr scheduler configurations to your explicit schedule using the `new_lr_scheduler` key of the "
            "relevant phases."
        )
        rank_zero_warn(error_msg)
        raise FTSConfigurationError(error_msg)


def resolve_max_depth(max_depth: int, num_phases: int) -> int:
    """The deepest phase to run: every phase when ``max_depth`` is ``-1``, otherwise ``max_depth`` capped at the
    final phase."""
    return num_phases - 1 if max_depth == -1 else min(max_depth, num_phases - 1)


def final_phase_max_epochs(schedule: dict, max_phase: int, max_epoch_wm: int, max_epochs: int) -> int | None:
    """When the final phase sets ``max_transition_epoch``, warn that it is overridden and return the run's maximum
    epoch count (the greater of ``max_epochs`` and the schedule's epoch watermark); otherwise ``None``."""
    if schedule[max_phase]["max_transition_epoch"] < 0:
        return None
    rank_zero_warn(
        "Final phase max_transition_epoch"
        f" ({schedule[max_phase]['max_transition_epoch']})"
        f" will be overidden by the greater of max_epochs ({max_epochs}) and"
        f" the maximum phase-specified epoch ({max_epoch_wm})."
    )
    return max(max_epoch_wm, max_epochs)


# phase keys the PyTorch API adds
DIRECT_API_PHASE_KEYS = ("max_transition_step", "phase_config", "pg_overrides", "promote")
# those only the PyTorch API interprets in this release (the Lightning callback refuses them)
LIGHTNING_UNSUPPORTED_PHASE_KEYS = ("max_transition_step", "promote")


def parse_phase_extensions(schedule: dict) -> None:
    """Validate and normalize, in place, the phase keys the PyTorch API adds.

    - ``max_transition_step``: a global optimizer step count at which the phase ends (default ``-1``, unset).
    - ``phase_config``: an opaque mapping handed to phase hooks and gates.
    - ``pg_overrides``: optimizer options (other than ``params`` and ``lr``, which have their own keys) applied to the
      parameter group the phase adds.
    - ``promote``: tensors to promote from the ``instrument`` role to a ``latent:<name>`` role when the phase starts,
      as ``{selector_or_name: "latent:<name>"}`` (not in phase 0).

    Raises:
        FTSConfigurationError: If a key has the wrong type.
    """
    for depth, phase in schedule.items():
        parse_promote(depth, phase.get("promote"))
        mts = phase.setdefault("max_transition_step", -1)
        if isinstance(mts, bool) or not isinstance(mts, int) or mts < -1:
            raise FTSConfigurationError(
                f"Phase {depth}'s `max_transition_step` must be an integer >= 0 (or -1, unset), got {mts!r}."
            )
    validate_phase_options(schedule)


def validate_phase_options(schedule: dict) -> None:
    """Validate each phase's ``phase_config`` and ``pg_overrides`` (see :func:`parse_phase_extensions`).

    Raises:
        FTSConfigurationError: If either has the wrong type, or ``pg_overrides`` sets ``params`` or ``lr``.
    """
    for depth, phase in schedule.items():
        if "phase_config" in phase and not isinstance(phase["phase_config"], Mapping):
            raise FTSConfigurationError(
                f"Phase {depth}'s `phase_config` must be a mapping, got {type(phase['phase_config']).__name__}."
            )
        overrides = phase.get("pg_overrides")
        if overrides is not None:
            if not isinstance(overrides, Mapping):
                raise FTSConfigurationError(
                    f"Phase {depth}'s `pg_overrides` must be a mapping of optimizer options, got "
                    f"{type(overrides).__name__}."
                )
            reserved = sorted({"params", "lr"} & set(overrides))
            if reserved:
                raise FTSConfigurationError(
                    f"Phase {depth}'s `pg_overrides` may not set {reserved}: set the phase's parameters with `params` "
                    "and its learning rate with `lr`."
                )


def parse_promote(depth: int, promote: Any) -> None:
    """Validate phase ``depth``'s ``promote`` mapping (``None`` when the phase has none).

    Raises:
        FTSConfigurationError: If it is not a non-empty mapping from names or selectors to ``latent:<name>`` roles, or
            phase 0 has one.
    """
    if promote is None:
        return
    if not isinstance(promote, Mapping) or not promote:
        raise FTSConfigurationError(
            f"Phase {depth}'s `promote` must be a non-empty mapping of selectors or names to `latent:<name>` roles, got"
            f" {promote!r}.")
    if depth == 0:
        raise FTSConfigurationError("Phase 0 may not `promote` tensors: a promotion happens when a later phase starts.")
    for source, target in promote.items():
        if not isinstance(source, str) or not source:
            raise FTSConfigurationError(f"Phase {depth}'s `promote` keys must be selectors or names, got {source!r}.")
        if not isinstance(target, str) or parse_role(target)[0] != "latent":
            raise FTSConfigurationError(
                f"Phase {depth} promotes {source!r} to {target!r}: tensors can be promoted only to a `latent:<name>`"
                " role.")


def refuse_direct_api_keys(schedule: dict, keys: Sequence[str] = LIGHTNING_UNSUPPORTED_PHASE_KEYS) -> None:
    """Refuse the phase keys only the PyTorch API interprets (``keys``), naming each phase that uses one.

    Raises:
        FTSConfigurationError: If any phase uses one.
    """
    used = {depth: sorted(k for k in keys if k in phase) for depth, phase in schedule.items()}
    used = {depth: keys for depth, keys in used.items() if keys}
    if used:
        detail = "; ".join(f"phase {depth}: {', '.join(keys)}" for depth, keys in used.items())
        raise FTSConfigurationError(
            f"The schedule uses keys only the PyTorch API (`TuningScheduler`) supports in this release ({detail})."
            " Remove them to use the schedule with the Lightning callback."
        )
