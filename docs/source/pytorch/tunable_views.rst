.. _tunable-views:

##################################################
Scheduling Through a Tunable View
##################################################

.. warning::
   Tunable views are **experimental** in 2.15 (see the :ref:`API stability classifications<versioning>`): the protocol
   may still change.

Overview
********
A fine-tuning schedule names parameters. By default those are the names ``named_parameters()`` reports, but a wrapped
model often has better names of its own: an interpretability library may call a block's weights ``blocks.3.attn.W_Q``
while the underlying module stores them, fused, as ``transformer.h.3.attn.c_attn.weight``.

A :class:`~finetuning_scheduler.core.views.TunableView` is the contract between the two: a schedule addresses a model
through the view's *logical* names, and Fine-Tuning Scheduler thaws, optimizes and checkpoints the *physical* tensors
they stand for. One logical name may stand for several physical tensors, and a view may thaw related tensors (a block's
normalization layer, for example) together with the ones a phase names.

The protocol is independent of Fine-Tuning Scheduler, which is one consumer of it. It lives in
`venngram <https://github.com/speediedan/venngram>`_ (``venngram.tuning``, with the role vocabulary in
``venngram.roles``), beside the address grammar schedules use, and Fine-Tuning Scheduler depends on it.
``finetuning_scheduler.tunable_view``, :mod:`finetuning_scheduler.core.views` and :mod:`finetuning_scheduler.core.roles`
re-export it for Fine-Tuning Scheduler's users. A view can therefore be written against the protocol alone.

Views work with both interfaces: :class:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler` (pass ``view=``)
and the Lightning callback (attach the view to the ``LightningModule``). A model without a view is scheduled through
:class:`~finetuning_scheduler.core.views.CanonicalView`, the identity view, so nothing changes for existing schedules.

Writing a View
**************
A view needs four members; the protocol defines a default for each of the rest, applied when a view does not define
it.

- ``root``: the module whose parameters the optimizer and checkpoints see (the ``LightningModule`` under Lightning).
- ``support()``: a :class:`~finetuning_scheduler.core.views.TunableSupport` record. It says whether the model can be
  fine-tuned through the view (``trainable=False`` with a ``reason`` refuses it), its naming scheme (``"physical"``
  when logical names are the root's own parameter names) and the role kinds it uses.
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
        """Logical ``block<i>`` names over a sequential model, thawing each block's norm with it."""

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

Fine-Tuning Scheduler compiles the schedule once, at setup, into the physical parameters each phase thaws
(:func:`~finetuning_scheduler.core.compile.compile_schedule`). A coupled tensor joins the first phase that pulls it in.
Compilation refuses a schedule whose phases share storage, a name the view cannot resolve, and a phase that schedules a
tensor whose role is never thawed, naming the input. A digest of the compiled schedule is saved with every checkpoint,
and resuming through a view that maps the schedule differently is refused.

Roles, Selectors and Addresses
******************************
A view may give tensors **roles** (``roles()``, physical parameter or buffer names to roles; anything unlisted is
``base``):

- ``adapter`` or ``adapter:<name>``: parameter-efficient adapters;
- ``intervention`` or ``intervention:<name>``: modules that intervene in the forward pass;
- ``latent:<name>``: learned latent dictionaries;
- ``fitted``: run-fitted state, never thawed or optimized but checkpointed;
- ``instrument``: published, re-loadable artifacts, never thawed, optimized or checkpointed.

A schedule can select tensors by role: ``role:adapter`` selects every adapter, ``role:adapter:char`` exactly that one.
Selectors work without a view too, where every parameter is ``base``, except under a Lightning strategy adapter that
names parameters itself (see `Views With the Lightning Callback`_).

.. code-block:: python

    schedule = {0: {"params": ["role:adapter"], "max_transition_epoch": 2}, 1: {"params": ["role:latent"]}}

**Addresses (experimental).** A role selector is the simplest *address*:

- **Fields.** An address is a record of fields that intersect: ``role``, ``layers``, ``module``, ``name``, ``leaf`` and
  others.
- **Algebra.** Addresses combine with ``&`` (intersection), ``-`` (difference) and ``|`` (union), with operators
  delimited by spaces.
- **Named sets.** ``@<set>`` references a set the schedule names.
- **Grammar.** The grammar, the named-set documents (*address books*) and their resolution are
  `venngram <https://github.com/speediedan/venngram>`_'s.

An entry is an address only when it starts with a field prefix (``role:``, ``layers:``, ``module:`` and the others) or
``@``, after any opening parentheses. Every parameter name and regex keeps its meaning.

Thawing a subset of a role has three answers, and the third is the one that needs no change to the view:

1. **Name the subsets as roles.** Give each group its own named role (``adapter:top``, ``adapter:mid``,
   ``adapter:low``) in the view's ``roles``. ``role:adapter`` still selects every adapter and ``role:adapter:top`` one
   group. A tensor has one role, so groups cannot overlap: the adapters of the top blocks and, separately, every
   ``q_proj`` adapter cannot both be roles.
2. **Fall back to a regex over names** (``model\.layers\.2[4-7]\..*lora_.*``). This gives up what the role gave: the
   schedule no longer reads as "the adapters", and it depends on the adapter's parameter names.
3. **Address the subset.** ``role:adapter & layers:-4:`` is the adapters of the top four blocks, in any model whose
   block list the view finds. Sets are selectors rather than roles, so they may overlap. For example,
   ``role:adapter & module:q_proj`` selects every ``q_proj`` adapter, including those in the top four blocks.

``layers`` takes an index, a list, ``odd``, ``even``, or a half-open slice whose negative indices count from the depth
(``-4:``, ``8:-4``). It counts the blocks of the root's block list, which a view reports through ``structure()``. The
default finds the root's one top-level ``nn.ModuleList`` and refuses, naming each, when there are several.

``module`` matches tensors under a module whose path ends with the pattern, on whole segments. ``name`` is a regex over
tensor names, quoted in an address (``name:"head\..*"``).

A schedule names its sets in a top-level ``addresses`` key. The key holds an inline book (a mapping with ``sets``, and
optionally ``imports``), the path of an address book relative to the schedule file, or a list of paths:

.. code-block:: yaml

    addresses:
      sets:
        top: {role: adapter, layers: "-4:"}
        rest: {role: adapter, exclude: "@top"}
    0:
      params: ["@top"]
      max_transition_epoch: 2
    1:
      params: ["@rest"]

**Refusals.** An address that does not resolve is refused, with every reason named. Examples:

- it matches nothing;
- it references a set no book defines;
- it names a layer outside the model;
- it needs architecture knowledge the view does not supply (``component:`` needs a vocabulary-backed structure provider,
  such as interpretune's);
- it addresses an activation site rather than tensors (``point:``);
- it addresses units of a tensor (``units:``), which are tuned through a unit adapter of their own.

Both interfaces accept addresses, in ``params`` and as ``promote`` sources. A ``params`` entry is a string. An address
written as a mapping goes in the ``addresses`` sets and is referenced by ``@<set>``.

Promotion
*********
A phase can promote ``instrument`` tensors to a ``latent`` role when it starts, so that phase, or a later one, trains
them (the PyTorch API only):

.. code-block:: python

    schedule = {
        0: {"params": ["role:adapter"], "max_transition_epoch": 2},
        1: {"params": ["role:latent:transcoder"], "promote": {"role:instrument": "latent:transcoder"}},
    }

The promotion happens after the phase's start hooks and before its thaw. A promoted parameter that shares storage with
another tensor (a tied weight) gets its own copy, and the other name keeps the original tensor (with the role the view
gives that name, ``base`` if none); a view can own the promotion with its own ``promote``, which must keep each tensor's
name. Promoted tensors are checkpointed from their promotion on, and promotions are replayed before a
saved state loads. Restoring a checkpoint saved before a promotion (a best checkpoint, or the rollback point of a vetoed
phase) puts the promoted tensors back to their values at promotion; those values are kept on the CPU and saved with
:meth:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler.save`. Promotion is not supported on a sharded model,
and a view that overrides its checkpoint state must define ``promote``.

Exporting by Role
*****************
:meth:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler.export_best` writes the best checkpoint's tensors
whose role matches, for example only the adapters, so a large frozen base and the optimizer state need not be kept:

.. code-block:: python

    fts.export_best(roles=("adapter",))  # best_export.pt beside the best checkpoint
    fts.export_best(write_peft, roles=("role:adapter:char",))  # write_peft(tensors, metadata, path): another format

The tensors come from the best checkpoint, not the live model, keyed by wrapper-free names (without segments such as
``torch.compile``'s ``_orig_mod.``); a sharded model's are read in full on global rank zero. ``fn`` receives
``path=None`` when no path is given.
``instrument`` tensors are never checkpointed, so they cannot be exported. The external artifacts a run depends on can
be recorded with ``TuningScheduler(artifacts={"lens": {"path": ..., "sha256": ..., "revision": ...}})``; they appear in
every checkpoint's and export's metadata.

Views With the Lightning Callback
*********************************
Attach the view to the ``LightningModule`` as ``tunable_view``, rooted at the module itself:

.. code-block:: python

    module = MyLightningModule()
    module.tunable_view = MyView(module)
    trainer = Trainer(callbacks=[FinetuningScheduler(ft_schedule="schedule.yaml")])
    trainer.fit(module)

The schedule keeps its logical names (in logs and in the saved schedule), the callback thaws and optimizes the physical
parameters, and a generated schedule uses the view's logical names. The FSDP (FSDP1) strategy accepts only a view
whose naming scheme is ``"physical"`` (its logical names are the module's own parameter names), applying its roles and
coupled tensors; use ``ModelParallelStrategy`` (FSDP2) to fine-tune through logical names.

A third-party :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` that renames parameters (it overrides
``logical_param_translation``, ``get_named_params_for_schedule_validation`` or ``gen_ft_schedule``, usually together
with ``fts_optim_transform`` and ``validate_ft_sched``) keeps naming parameters itself, and takes precedence over an
attached view, with a ``FutureWarning`` naming both; addresses (``role:`` selectors among them) are refused under it. Renaming through those
methods is deprecated in favor of a view and will be unsupported no earlier than 2.17. Sharding customization through
``StrategyAdapter`` subclasses is unaffected: an adapter that overrides only ``fts_optim_transform`` (calling
``super()`` first, which resolves an attached view's names) or only adds checks in ``validate_ft_sched`` schedules
through views and selectors as the built-in adapters do.

Testing a View
**************
:class:`~venngram.tuning.conformance.ViewConformance` checks a view implementation against the
protocol: the name mapping, the checkpoint round trip, roles and promotion, which hold whatever consumes the view.
:class:`~finetuning_scheduler.testing.conformance.TunableViewConformance` adds the cases Fine-Tuning Scheduler relies on:
schedule compilation, thawing, an optimizer step and refusals. In both, which cases apply is decided by the view's own support record, and a run in which no case applies fails, so a view cannot
pass vacuously. It needs neither pytest nor Lightning:

.. code-block:: python

    from finetuning_scheduler.testing.conformance import TunableViewConformance

    report = TunableViewConformance(lambda: MyView(build_model())).run(raise_on_failure=True)

``finetuning_scheduler.testing.fixtures`` provides four known-good fixture views to compare against.
