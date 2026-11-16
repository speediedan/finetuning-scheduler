.. _tunable-views-advanced:

##################################################
Advanced Tunable Views
##################################################

.. warning::
   Tunable views are **experimental** in 2.15 (see the :ref:`API stability classifications<versioning>`): the protocol
   may still change.

Most schedules need only :ref:`tunable-views`: ``venngram.view(model)`` gives a model its view, and roles, selectors
and addresses say what each phase tunes. This page covers more advanced or extended usage patterns: writing and testing
a view of your own, promotion, sharing address books through venngram, and third-party strategy adapters that name
parameters themselves. venngram's `README <https://github.com/speediedan/venngram>`_ is the reference for the grammar, address
books and stores; this page says only how Fine-Tuning Scheduler uses them.

.. _tunable-views-writing:

Writing a View
**************
A view is usually composed from published maps by ``venngram.view(model)``, and the ``tunable-view-generator`` agent
skill drafts and verifies a map for an architecture that has none (see :ref:`tunable-views-finding-maps`). Write a view
of your own only when a schedule needs a vocabulary that differs from the model's own names, or behavior the data-driven
views do not have.

A view needs four members; the protocol defines a default for each of the rest, applied when a view does not define
it.

- ``root``: the module whose parameters the optimizer and checkpoints see (the ``LightningModule`` under Lightning).
- ``support()``: a :class:`~finetuning_scheduler.core.views.TunableSupport` record. It says whether the model can be
  fine-tuned through the view (``trainable=False`` with a ``reason`` refuses it), its naming scheme and the role kinds
  it uses. The naming scheme names the vocabulary of the view's logical names: ``"physical"`` when they are the root's
  own parameter names, as for every data-driven view, or another label (interpretune's TransformerLens view uses
  ``"tl"``) when they differ. A view with another scheme must define ``logical_named_parameters``; messages and the
  generated schedule then use its names, and an address book written in one vocabulary declares it
  (``requires: {naming_scheme: tl}``).
- ``to_physical(logical)``: each logical name mapped to its physical parameter names.
- ``to_logical(physical)``: physical names mapped back, for messages.

The optional members (:class:`~finetuning_scheduler.core.views.TunableViewExtras`) are ``logical_named_parameters``
(required when the naming scheme is not ``"physical"``), ``coupled``, ``depth_groups``, ``roles``, the checkpoint pair,
``promote`` and ``structure`` (where the root's tensors sit in its block structure, for addresses; the default finds the
root's block list). :class:`~finetuning_scheduler.core.views.ViewDefaults` implements every default, for implementers who
prefer to inherit. A view need not inherit from any class. Members are methods: a view whose ``roles`` (for example) is
a dictionary rather than a method returning one is refused.

A view that does not satisfy the protocol (one lacking a required member, or whose ``support()`` returns something
other than a :class:`~finetuning_scheduler.core.views.TunableSupport`) raises
:class:`~venngram.tuning.errors.TunableViewError`, as does a view's ``to_physical`` given a name it cannot resolve
(:class:`~venngram.tuning.errors.Unresolvable`, a subclass). A schedule or
configuration that is invalid, including one whose names a view cannot resolve, raises
:class:`~finetuning_scheduler.core.exceptions.FTSConfigurationError` (under Lightning, also a
``MisconfigurationException``).

.. code-block:: python

    from finetuning_scheduler import TunableSupport


    class BlockView:
        """Logical ``block<i>`` names over a sequential model, tuning each block's norm with it."""

        BLOCKS = {"block0": ("0.weight", "0.bias"), "head": ("2.weight", "2.bias")}
        NORMS = {"0.weight": ["1.weight", "1.bias"]}

        def __init__(self, root):
            self.root = root

        def support(self):
            return TunableSupport(trainable=True, naming_scheme="blocks")

        def to_physical(self, logical):
            return {n: self.BLOCKS[n] for n in logical}

        def to_logical(self, physical):
            rev = {p: n for n, ps in self.BLOCKS.items() for p in ps}
            return list(dict.fromkeys(rev.get(p, p) for p in physical))

        def logical_named_parameters(self):
            for n, ps in self.BLOCKS.items():
                yield n, self.root.get_parameter(ps[0])

        def coupled(self, physical):
            return [c for p in physical for c in self.NORMS.get(p, [])]

The schedule then uses the logical names:

.. code-block:: python

    fts = TuningScheduler(model, {0: {"params": ["head"]}, 1: {"params": ["block0"]}}, optimizer=opt, view=BlockView(model))

A coupled tensor joins the first phase that pulls it in.

Testing a View
**************
:class:`~venngram.tuning.conformance.ViewConformance` checks a view implementation against the
protocol: the name mapping, the checkpoint round trip, roles and promotion, which hold whatever consumes the view.
:class:`~finetuning_scheduler.testing.conformance.TunableViewConformance` adds the cases Fine-Tuning Scheduler relies on:
schedule compilation, thawing, an optimizer step and refusals. In both, which cases apply is decided by the view's own
support record, and a run in which no case applies fails, so a view cannot pass vacuously. It needs neither pytest nor
Lightning:

.. code-block:: python

    from finetuning_scheduler.testing.conformance import TunableViewConformance

    report = TunableViewConformance(lambda: MyView(build_model())).run(raise_on_failure=True)

``finetuning_scheduler.testing.fixtures`` provides four known-good fixture views to compare against.

.. _tunable-views-promotion:

Promotion
*********
A phase can promote ``artifact`` tensors to a ``latent`` role when it starts, so that phase, or a later one, trains
them (the PyTorch API only):

.. code-block:: python

    schedule = {
        0: {"params": ["role:adapter"], "max_transition_epoch": 2},
        1: {"params": ["role:latent:transcoder"], "promote": {"role:artifact": "latent:transcoder"}},
    }

A ``promote`` source is a selector or an address. The promotion happens after the phase's start hooks and before it
starts tuning. A promoted parameter that shares storage with another tensor (a tied weight) gets its own copy, and the other
name keeps the original tensor (with the role the view gives that name, ``base`` if none); a view can own the promotion
with its own ``promote``, which must keep each tensor's name. Promoted tensors are checkpointed from their promotion on,
and promotions are replayed before a saved state loads. Restoring a checkpoint saved before a promotion (a best
checkpoint, or the rollback point of a vetoed phase) puts the promoted tensors back to their values at promotion; those
values are kept on the CPU and saved with :meth:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler.save`.
Promotion is not yet supported on a sharded model, and a view that overrides its checkpoint state must define ``promote``.

.. _tunable-views-gradient-masks:

Training Part of a Tensor
*************************
An address of gradients (``of: grad``) trains part of each tensor it selects (the PyTorch API only): the tensor is
thawed, and its gradient is masked to the indices along its first axis (a weight's output rows, a bias's entries) that
the address's ``units`` keep. Here a model's middle ``Linear`` trains only output units 0 and 2, then unit 5 as well,
beside its head's weight trained whole:

.. code-block:: python

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    schedule = {
        0: {"params": ["module:2 & units:[0, 2] & of:grad"], "max_transition_epoch": 2},
        1: {"params": ["4.weight", "module:2 & units:[5] & of:grad"]},
    }
    fts = TuningScheduler(model, schedule, optimizer=partial(torch.optim.AdamW, lr=1e-3))
    fts.setup()
    fts.gradient_masks  # {'2.weight': (0, 2), '2.bias': (0, 2)}; from phase 1, (0, 2, 5) for each

The rules:

- **A phase's mask for a tensor is the union of its entries.** An entry that selects the tensor whole, or a mask that
  keeps every index, trains it unmasked. A later phase that selects a masked tensor again widens its mask (unit 5 above)
  or trains it whole.
- **A phase may mix whole tensors and gradient addresses**, as phase 1 does: each entry resolves on its own.
- **The masked indices stay put.** The mask is applied to each step's accumulated gradient, before clipping and the
  optimizer step, and the masked indices are also held through the step, so weight decay and momentum do not move
  them, for any optimizer. Masks follow the schedule through refreezing, promotion, optimizer reinitialization and
  resume; each phase logs its masked tensors, and
  :attr:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler.gradient_masks` gives the masks in effect.
- **Unit sets are written like any other field**, in an entry or in a named set, where an architecture-neutral
  component makes the same schedule apply to every model its view maps:

  .. code-block:: yaml

      addresses:
        sets:
          q_rows: {component: attn.q, layers: "-4:", units: [0, 17, 42], of: grad}
      0:
        params: ["@q_rows"]

``of: grad`` is refused by name in a ``refreeze`` or ``promote`` entry, on the Lightning callback (its strategy adapters
shard and checkpoint parameters themselves), and on a sharded model (FSDP2 sets a parameter's gradient directly, so the
mask's hook would not run). A mask is so far the only operation on the gradients a schedule addresses; scaling,
projecting or a function of your own are `#219 <https://github.com/speediedan/finetuning-scheduler/issues/219>`_.

Address Books Through venngram
******************************
venngram defines how address books are published, referenced and checked (`Address books
<https://github.com/speediedan/venngram#address-books>`_ in its README). Fine-Tuning Scheduler adds three things.

.. _tunable-views-books-by-reference:

**Books by reference.** A schedule can name a published book by its document reference
(``<store>:<ref>[@<revision>]:<path>[#<sha256>]``, see `Sharing documents
<https://github.com/speediedan/venngram#sharing-documents-manifests-and-stores>`_) anywhere it names a book:

.. code-block:: yaml

    addresses: hub-model:my-org/qwen3-sets@v1:books/lora.yaml

Each book fetched by reference is pinned when the schedule loads (``venngram.store.pin``), the pinned references are
logged and saved in every checkpoint (``address_references``), and resuming from a checkpoint whose books have changed
since is refused, naming the saved references so the schedule can cite them to resume exactly.

.. _tunable-views-support-record:

**What a book may ask.** Fine-Tuning Scheduler declares what it tunes as a venngram support record
(``finetuning_scheduler/core/support_record.yaml``, read by :func:`~finetuning_scheduler.core.compile.support_record`;
see `Support records <https://github.com/speediedan/venngram#support-records>`_). A set a schedule uses that asks for
more (an activation site, or units of a tensor) is refused at compile time; a set of the same book the schedule does not
use is not judged, so one book can serve a steering tool and a schedule. A book can be checked against the record
without Fine-Tuning Scheduler installed:

.. code-block:: bash

    python -m venngram.conformance hub-model:my-org/qwen3-sets@v1 --support support_record.yaml

.. _tunable-views-export-book:

**A schedule as a book.** :meth:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler.export_book` (or
:func:`~finetuning_scheduler.core.books.schedule_book`) writes a schedule's phase selections as an address book, one set
per phase (``phase_0``, ``phase_1``, ...), so a tuning setup can be published, imported and cited like any other set:

.. code-block:: python

    fts.export_book("books/my_schedule.yaml")

Names and regexes become ``name`` terms (a regex keeps Fine-Tuning Scheduler's prefix match), the book declares the
view's naming scheme, and books fetched by reference are recorded under ``derived_from``. A phase of gradient addresses
becomes a set of gradients, in an address book of schema 2. A venngram set is either values or gradients, and takes a
gradient mask from one conjunction, so two kinds of phase that train as written cannot yet be exported: one mixing
whole tensors and gradient addresses (phase 1 of the :ref:`example above<tunable-views-gradient-masks>`), and one of
several gradient entries with a unit set among them. Each is refused by name;
`venngram#91 <https://github.com/speediedan/venngram/issues/91>`_ would let one set hold either.

.. _tunable-views-refusals:

Refusals
********
An address that does not resolve is refused: it matches nothing, references a set no book
defines, names a layer outside the model, needs architecture knowledge the view does not supply (``component:`` needs a
vocabulary-backed structure provider, such as :class:`~venngram.maps.view.MappedView`'s), addresses an activation site
(``point:``), or addresses units of a tensor's value (``units:`` outside an ``of: grad`` address; units are trained in
place through a :ref:`gradient mask<tunable-views-gradient-masks>`, or tuned through a unit adapter of their own).
venngram's ``venngram-addressing`` agent skill (`Agent skill <https://github.com/speediedan/venngram#agent-skill>`_)
explains how to read each one.

.. _tunable-views-renaming-adapters:

Strategy Adapters That Name Parameters
**************************************
A third-party :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` that renames parameters (it overrides
``logical_param_translation``, ``get_named_params_for_schedule_validation`` or ``gen_ft_schedule``, usually together
with ``fts_optim_transform`` and ``validate_ft_sched``) keeps naming parameters itself, and takes precedence over an
attached view, with a ``FutureWarning`` naming both; addresses (``role:`` selectors among them) are refused under it.
Renaming through those methods is deprecated in favor of a view and will be unsupported no earlier than 2.17. Sharding
customization through ``StrategyAdapter`` subclasses is unaffected: an adapter that overrides only
``fts_optim_transform`` (calling ``super()`` first, which resolves an attached view's names) or only adds checks in
``validate_ft_sched`` schedules through views and selectors as the built-in adapters do.
