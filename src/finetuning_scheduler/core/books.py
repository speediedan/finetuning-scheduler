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

A book derived from address books fetched by reference records their exact references under ``derived_from``.

.. warning:: This is an experimental API and may change in a future release.
"""
import re
from collections.abc import Mapping, Sequence
from typing import Any

from venngram import is_address
from venngram.address import And, Expr, Guarded, Minus, Or, Ref, Term, to_mapping
from venngram.address import parse as parse_address

from finetuning_scheduler.core.exceptions import FTSConfigurationError


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


def schedule_book(schedule: Mapping[int, Mapping[str, Any]], *, name: str, addresses: Mapping[str, Any] | None = None,
                  logical_names: Sequence[str] | None = None, naming_scheme: str = "physical",
                  applies_to: Mapping[str, Any] | None = None, description: str | None = None) -> dict[str, Any]:
    """A schedule's phase selections as an address book document (a mapping, ready to dump as YAML).

    Args:
        schedule: The schedule (phase depth mapped to its configuration; only ``params`` is read), with its entries as
            written (logical names, regexes and addresses), not compiled.
        name: The book's name.
        addresses: The named sets the entries reference (see
            :func:`~finetuning_scheduler.core.schedule.load_address_sets`); their ``references`` are recorded under
            ``derived_from``.
        logical_names: The view's logical names: an entry among them is written as an exact name rather than a regex.
            ``None`` treats every name entry as a regex.
        naming_scheme: The scheme of the view the names belong to, declared as ``requires.naming_scheme`` when any
            entry is a name or regex.
        applies_to: The book's ``applies_to`` (for example ``{"architectures": ["Qwen3ForCausalLM"]}``).
        description: The book's description.

    Raises:
        FTSConfigurationError: If a phase has no ``params``, or an address references a set ``addresses`` lacks.
    """
    sets = dict(addresses or {})
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
        out_sets[f"phase_{depth}"] = to_mapping(exprs[0] if len(exprs) == 1 else Or(tuple(exprs)))
    book: dict[str, Any] = {"schema_version": 1, "name": name}
    if description is not None:
        book["description"] = description
    book["portability"] = "family"
    if applies_to:
        book["applies_to"] = dict(applies_to)
    if uses_names:
        book["requires"] = {"naming_scheme": naming_scheme}
    references = list(getattr(addresses, "references", ()))
    if references:
        book["derived_from"] = references
    book["sets"] = out_sets
    return book


__all__ = ["schedule_book"]
