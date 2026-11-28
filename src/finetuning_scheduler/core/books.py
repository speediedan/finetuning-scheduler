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
Schedules as Address Books
^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`schedule_book` writes a schedule's phase selections as a `venngram <https://github.com/speediedan/venngram>`_
address book, one set per phase (``phase_0``, ``phase_1``, ...), so a tuning setup can be shared, composed and cited
like any other set: published with :func:`venngram.store.publish`, imported by another book, and checked against a
consumer's support record (:func:`finetuning_scheduler.core.compile.support_record`) with
``python -m venngram.conformance``.

Each phase's set is the union of its ``params`` entries, as written:

- an address stays an address, with every ``@<set>`` reference replaced by the set it names, so the book stands alone;
- a parameter name becomes a ``name`` term matching exactly that name, and a regex a ``name`` term matching what
  Fine-Tuning Scheduler matches (``re.match``, a prefix match, where venngram's ``name`` matches in full). Names are in
  the view's vocabulary, so a book with any declares ``requires.naming_scheme``.

A book declares the address book schema its contents need: 1 for plain selections, 2 for a set of gradients
(``of: grad``), 6 for one using ``axis``, 7 for layers relative to the depth, and so on (:mod:`venngram.book` lists
them). A phase of gradient addresses becomes a set of gradients. A phase mixing whole tensors and gradient addresses
becomes one set uniting a value part and a gradient part, each value entry stating ``of: value``, and a phase of several
gradient entries with unit sets one set whose disjuncts each give their own mask, united per tensor (venngram's
``mixed-sets`` and ``union-masks``, which Fine-Tuning Scheduler's support record declares). Either compiles back to the
phase's masks.

A phase's gradient transforms (``grad_transforms``) become a treatment of the book named after the phase's set
(``phase_<depth>``), so a schedule reading the book names it (``grad_transforms: phase_1``), in an address book of
schema 4 or later (the first that can hold a treatment's ``gradients``):

- the built-ins :class:`~finetuning_scheduler.core.grad_transforms.Scale`,
  :class:`~finetuning_scheduler.core.grad_transforms.Replace`, :class:`~finetuning_scheduler.core.grad_transforms.Add`,
  :class:`~finetuning_scheduler.core.grad_transforms.Project` and
  :class:`~finetuning_scheduler.core.grad_transforms.Reject` become venngram's ``scale``, ``replace``, ``add``,
  ``project`` and ``reject`` ops, in order, each on a set of the gradients its entry selects
  (``phase_<depth>_grad_<n>``, the entry stating ``of: grad``), with a span's directions inline as an operand (in the
  ``out`` basis). Every other transform (``Clamp``, or a callable of your own) is refused by name: venngram declares no
  op for it on gradients, and a callable is not portable;
- a treatment the phase names is copied, its target sets inlined as ``phase_<depth>_<set>`` and any sidecar operand
  read inline, so the book stands alone.

A treatment records the model it was written for (``model``, its ``id`` and ``revision``). A built-in acts along a
tensor's first axis unless told otherwise, where a venngram op on units or a span follows the view's component map, so
:meth:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler.export_book` compiles the book back and refuses an
export whose transforms would differ. A phase that clears its transforms (``grad_transforms: {}``) has no treatment: a
schedule reading the book clears them itself.

A book derived from address books fetched by reference records their exact references under ``derived_from``, and a
book derived from books that pin the models they were written against (``requires.base_model`` and
``requires.latent_models``) pins the same models (schema 8).

.. warning:: This is an experimental API and may change in a future release.
"""
import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from venngram import AddressBook, is_address
from venngram.address import And, Expr, Guarded, Minus, Or, Ref, Term, terms, to_mapping
from venngram.address import parse as parse_address
from venngram.errors import VenngramError
from venngram.operands import INLINE_LIMIT

from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.grad_transforms import Add, Project, Reject, Replace, Scale


def _inline(expr: Expr, sets: Mapping[str, Expr], path: tuple[str, ...] = ()) -> Expr:
    """``expr`` with every reference replaced, recursively, by the set it names."""
    if isinstance(expr, Ref):
        if expr.name not in sets:
            raise FTSConfigurationError(f"`@{expr.name}` names no set of the schedule's `addresses`.")
        if expr.name in path:  # the book refused cycles when it loaded; this guards an assembled mapping
            raise FTSConfigurationError(f"Set references form a cycle through `@{expr.name}`.")
        return _inline(sets[expr.name], sets, path + (expr.name,))
    if isinstance(expr, Guarded):
        return Guarded(_inline(expr.expr, sets, path), optional=expr.optional, expect=expr.expect)
    if isinstance(expr, Minus):
        return Minus(_inline(expr.left, sets, path), _inline(expr.right, sets, path))
    if isinstance(expr, (And, Or)):
        return type(expr)(tuple(_inline(i, sets, path) for i in expr.items))
    return expr


def _entry_expr(entry: str, names: Sequence[str] | None, sets: Mapping[str, Expr]) -> tuple[Expr, bool]:
    """A ``params`` entry as an address, and whether it was written as a name or regex."""
    if is_address(entry):
        return _inline(parse_address(entry), sets), False
    if names is not None and entry in names:
        return Term(name=re.escape(entry)), True
    return Term(name=f"(?:{entry}).*"), True


_VALUED_OPS = {Scale: ("scale", "factor"), Replace: ("replace", "value"), Add: ("add", "value")}
_SPAN_OPS = {Project: "project", Reject: "reject"}


def _of_grad(expr: Expr) -> Expr:
    """``expr`` addressing gradients: as written when a term states ``of``, else with ``of: grad``."""
    return expr if any(t.of is not None for t in terms(expr)) else And((expr, Term(of="grad")))


def _inline_values(values: Sequence[float], shape: Sequence[int], where: str) -> dict[str, Any]:
    """An operand's values inline, refused past venngram's inline limit."""
    if len(values) > INLINE_LIMIT:
        raise FTSConfigurationError(f"{where} has {len(values)} values, more than the {INLINE_LIMIT} an exported"
                                    " book holds inline; publish the treatment in a book of its own instead.")
    return {"values": [float(v) for v in values], "shape": [int(n) for n in shape]}


def _transforms_treatment(depth: int, transforms: Mapping[str, Any], names: Sequence[str] | None,
                          sets: Mapping[str, Expr], model: Mapping[str, Any] | None) -> tuple[dict, dict, bool]:
    """A phase's Python ``grad_transforms`` as a treatment body, the sets its ops target, and whether an entry was
    written as a name or regex."""
    if model is None:
        raise FTSConfigurationError(f"Phase {depth}'s gradient transforms export as a treatment, which records the"
                                    " model it was written for: pass `model={'id': ..., 'revision': ...}`.")
    targets: dict[str, Any] = {}
    operands: dict[str, Any] = {}
    ops: list[dict[str, Any]] = []
    uses_names = False
    for n, (entry, fns) in enumerate(transforms.items()):
        expr, named = _entry_expr(entry, names, sets)
        uses_names |= named
        target = f"phase_{depth}_grad_{n}"
        targets[target] = to_mapping(_of_grad(expr))
        for fn in list(fns) if isinstance(fns, (list, tuple)) else [fns]:
            if type(fn) in _VALUED_OPS:
                op, attr = _VALUED_OPS[type(fn)]
                ops.append({"op": op, "target": {"set": target}, "value": {"dose": float(getattr(fn, attr))}})
            elif type(fn) in _SPAN_OPS:
                d = torch.as_tensor(fn.directions, dtype=torch.float64)
                d = d if d.dim() == 1 else d.T  # a subspace stacks its directions on its leading axis
                operand = f"phase_{depth}_dir_{len(operands)}"
                operands[operand] = {**_inline_values(d.flatten().tolist(), d.shape,
                                                      f"Phase {depth}'s `{type(fn).__name__}` on {entry!r}"),
                                     "basis": "out"}
                ops.append({"op": _SPAN_OPS[type(fn)], "target": {"set": target}, "operand": operand,
                            "params": {"fraction": float(fn.fraction)}})
            else:
                raise FTSConfigurationError(
                    f"Phase {depth}'s `grad_transforms` entry {entry!r} applies {fn!r}, which an address book cannot"
                    " hold: venngram declares ops on gradients for `Scale`, `Replace`, `Add`, `Project` and `Reject`"
                    " only, and a callable is not portable.")
    body = {"model": dict(model), **({"operands": operands} if operands else {}), "gradients": ops}
    return body, targets, uses_names


def _named_treatment(depth: int, name: str, addresses: Any) -> tuple[dict, dict]:
    """The treatment ``name`` of the schedule's books as a body standing alone: its target sets inlined (as
    ``phase_<depth>_<set>``) and its sidecar operands read inline."""
    found = [b.treatments[name] for b in getattr(addresses, "books", ()) if name in b.treatments]
    if len(found) != 1:
        raise FTSConfigurationError(f"Phase {depth}'s `grad_transforms` names the treatment `{name}`, which"
                                    f" {'several' if found else 'none'} of the schedule's address books have.")
    treatment = found[0]
    body = treatment.to_mapping()
    targets: dict[str, Any] = {}
    for op in body.get("gradients", ()):  # each op targets one set of the book (venngram refuses anything else)
        renamed = f"phase_{depth}_{op['target']['set']}"
        targets[renamed] = to_mapping(_inline(Ref(op["target"]["set"]), addresses))
        op["target"] = {"set": renamed}
    for key, raw in list(body.get("operands", {}).items()):
        if "values" not in raw:
            values, shape = treatment.operands[key].load(treatment.base_dir or ".")
            body["operands"][key] = {**_inline_values(values, shape, f"Phase {depth}'s treatment `{name}` operand"
                                                                     f" `{key}`"),
                                     **{k: raw[k] for k in ("basis", "space") if k in raw}}
    return body, targets


def _merged_pins(books: Sequence[AddressBook]) -> dict[str, Any]:
    """The model pins of ``books`` and everything they import, merged into one book's ``requires``."""
    out: dict[tuple[str, ...], Any] = {}
    for book in books:
        pins = book.pins()
        found: dict[tuple[str, ...], Any] = {("latent_models", n): p for n, p in pins.get("latent_models", {}).items()}
        if "base_model" in pins:
            found[("base_model",)] = pins["base_model"]
        for key, pin in found.items():
            if out.setdefault(key, pin) != pin:
                raise FTSConfigurationError(f"The schedule's address books pin `{'.'.join(key)}` differently"
                                            f" ({out[key]} and {pin}), so one book cannot hold their sets.")
    merged: dict[str, Any] = {"base_model": dict(out[("base_model",)])} if ("base_model",) in out else {}
    latents = {key[1]: copy.deepcopy(pin) for key, pin in out.items() if key[0] == "latent_models"}
    return {**merged, **({"latent_models": latents} if latents else {})}


def schedule_book(schedule: Mapping[int, Mapping[str, Any]], *, name: str, addresses: Mapping[str, Any] | None = None,
                  logical_names: Sequence[str] | None = None, naming_scheme: str = "physical",
                  applies_to: Mapping[str, Any] | None = None, description: str | None = None,
                  model: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A schedule's phase selections as an address book document (a mapping, ready to dump as YAML).

    Args:
        schedule: The schedule (phase depth mapped to its configuration; only ``params`` is read), with its entries as
            written (logical names, regexes and addresses), not compiled.
        name: The book's name.
        addresses: The named sets the entries reference (see
            :func:`~finetuning_scheduler.core.schedule.load_address_sets`); their ``references`` are recorded under
            ``derived_from``, and the models their ``books`` pin under ``requires``.
        logical_names: The view's logical names: an entry among them is written as an exact name rather than a regex.
            ``None`` treats every name entry as a regex.
        naming_scheme: The scheme of the view the names belong to, declared as ``requires.naming_scheme`` when any
            entry is a name or regex.
        applies_to: The book's ``applies_to`` (for example ``{"architectures": ["Qwen3ForCausalLM"]}``).
        description: The book's description.
        model: The model a phase's Python gradient transforms were written for (``{"id": ..., "revision": ...}``),
            recorded by the treatment they become; a treatment the schedule names keeps its own.

    Raises:
        FTSConfigurationError: If a phase has no ``params``, an address references a set ``addresses`` lacks, two of
            the ``books`` pin one model differently, the book would not load, or a phase's gradient transforms cannot
            be exported: a transform venngram declares no op on gradients for, no ``model`` for Python transforms, or a
            span with more values than a book holds inline.
    """
    sets = dict(addresses or {})
    treatments: dict[str, Any] = {}
    treatment_sets: dict[str, Any] = {}
    out_sets: dict[str, Any] = {}
    uses_names = False
    for depth in sorted(schedule):
        entries = schedule[depth].get("params") or []
        if not entries:
            raise FTSConfigurationError(f"Phase {depth} has no `params` to export.")
        exprs = []
        for entry in entries:
            if not isinstance(entry, str):
                raise FTSConfigurationError(f"Phase {depth}'s `params` entry {entry!r} is not a string.")
            expr, named = _entry_expr(entry, logical_names, sets)
            exprs.append(expr)
            uses_names |= named
        # entries are inlined, so each term says whether it addresses gradients; beside a gradient entry, a value entry
        # states `of: value`, since terms that do not state `of` follow those that do
        if any(t.of == "grad" for e in exprs for t in terms(e)):
            exprs = [e if any(t.of is not None for t in terms(e)) else And((e, Term(of="value"))) for e in exprs]
        out_sets[f"phase_{depth}"] = to_mapping(exprs[0] if len(exprs) == 1 else Or(tuple(exprs)))
        stated = schedule[depth].get("grad_transforms")
        if isinstance(stated, str):
            treatments[f"phase_{depth}"], targets = _named_treatment(depth, stated, addresses)
        elif stated:
            treatments[f"phase_{depth}"], targets, named = _transforms_treatment(depth, stated, logical_names, sets,
                                                                                 model)
            uses_names |= named
        else:
            continue
        treatment_sets.update(targets)
    out_sets.update(treatment_sets)
    book: dict[str, Any] = {"name": name}
    if description is not None:
        book["description"] = description
    book["portability"] = "family"
    if applies_to:
        book["applies_to"] = dict(applies_to)
    requires = {**({"naming_scheme": naming_scheme} if uses_names else {}),
                **_merged_pins(getattr(addresses, "books", ()))}
    if requires:
        book["requires"] = requires
    references = list(getattr(addresses, "references", ()))
    if references:
        book["derived_from"] = references
    book["sets"] = out_sets
    if treatments:
        book["treatments"] = treatments
    try:  # an inline book without a schema takes the one its contents need
        needed = AddressBook.from_mapping(book, source="the exported book", inline=True).schema_version
    except VenngramError as err:
        raise FTSConfigurationError(f"The exported book does not load: {err}") from err
    return {"schema_version": needed, **book}


__all__ = ["schedule_book"]
