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

Views work with both hosts: :class:`~finetuning_scheduler.core.controller.FTSController` (pass ``view=``) and the
Lightning callback (attach the view to the ``LightningModule``). A model without a view is scheduled through
:class:`~finetuning_scheduler.core.views.CanonicalView`, the identity view, so nothing changes for existing schedules.

Writing a View
**************
A view needs four members; Fine-Tuning Scheduler supplies the rest when a view does not define them.

- ``root``: the module whose parameters the optimizer and checkpoints see (the ``LightningModule`` under Lightning).
- ``support()``: a :class:`~finetuning_scheduler.core.views.TunableSupport` record. It says whether the model can be
  fine-tuned through the view (``trainable=False`` with a ``reason`` refuses it), its naming scheme (``"physical"``
  when logical names are the root's own parameter names) and the role kinds it uses.
- ``to_physical(logical)``: each logical name mapped to its physical parameter names.
- ``to_logical(physical)``: physical names mapped back, for messages.

The optional members (:class:`~finetuning_scheduler.core.views.TunableViewExtras`) are ``logical_named_parameters``
(required when the naming scheme is not ``"physical"``), ``coupled``, ``depth_groups``, ``roles``, the checkpoint pair
and ``promote``. :class:`~finetuning_scheduler.core.views.ViewDefaults` implements every default, for implementers who
prefer to inherit. A view need not inherit from any Fine-Tuning Scheduler class.

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

    fts = FTSController(model, {0: {"params": ["head"]}, 1: {"params": ["block0"]}}, optimizer=opt, view=BlockView(model))

Fine-Tuning Scheduler compiles the schedule once, at setup, into the physical parameters each phase thaws
(:func:`~finetuning_scheduler.core.compile.compile_schedule`). A coupled tensor joins the first phase that pulls it in.
Compilation refuses a schedule whose phases share storage, a name the view cannot resolve, and a phase that schedules a
tensor whose role is never thawed, naming the input. A digest of the compiled schedule is saved with every checkpoint,
and resuming through a view that maps the schedule differently is refused.

Roles and Selectors
*******************
A view may give tensors **roles** (``roles()``, physical parameter or buffer names to roles; anything unlisted is
``base``):

- ``adapter`` or ``adapter:<name>``: parameter-efficient adapters;
- ``intervention`` or ``intervention:<name>``: modules that intervene in the forward pass;
- ``latent:<name>``: learned latent dictionaries;
- ``fitted``: run-fitted state, never thawed or optimized but checkpointed;
- ``instrument``: published, re-loadable artifacts, never thawed, optimized or checkpointed.

A schedule can select tensors by role: ``role:adapter`` selects every adapter, ``role:adapter:char`` exactly that one.
Selectors work without a view too, where every parameter is ``base``.

.. code-block:: python

    schedule = {0: {"params": ["role:adapter"], "max_transition_epoch": 2}, 1: {"params": ["role:latent"]}}

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
another tensor (a tied weight) gets its own copy; a view can own the promotion with its own ``promote``, which must keep
each tensor's name. Promoted tensors are checkpointed from their promotion on, and promotions are replayed before a
saved state loads. Restoring a checkpoint saved before a promotion (a best checkpoint, or the rollback point of a vetoed
phase) puts the promoted tensors back to their values at promotion; those values are kept on the CPU and saved with
:meth:`~finetuning_scheduler.core.controller.FTSController.save`. Promotion is not supported on a sharded model, and a
view that overrides its checkpoint state must define ``promote``.

Views With the Lightning Callback
*********************************
Attach the view to the ``LightningModule`` as ``fts_tunable_view``, rooted at the module itself:

.. code-block:: python

    module = MyLightningModule()
    module.fts_tunable_view = MyView(module)
    trainer = Trainer(callbacks=[FinetuningScheduler(ft_schedule="schedule.yaml")])
    trainer.fit(module)

The schedule keeps its logical names (in logs and in the saved schedule), the callback thaws and optimizes the physical
parameters, and a generated schedule uses the view's logical names. The FSDP (FSDP1) strategy does not support views;
use ``ModelParallelStrategy`` (FSDP2) instead.

A third-party :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` that overrides the parameter-naming
methods (``fts_optim_transform``, ``logical_param_translation``, ``get_named_params_for_schedule_validation``,
``validate_ft_sched``, ``gen_ft_schedule``) keeps naming parameters itself, and takes precedence over an attached
view, with a ``FutureWarning`` naming both. Overriding those methods is deprecated in favor of a view and will be
unsupported no earlier than 2.17; sharding customization through ``StrategyAdapter`` subclasses is unaffected.

Testing a View
**************
:class:`~finetuning_scheduler.testing.conformance.TunableViewConformance` checks a view implementation against the
contract: the name mapping, thawing, an optimizer step, the checkpoint round trip, refusals, roles and promotion. Which
cases apply is decided by the view's own support record, and a run in which no case applies fails, so a view cannot
pass vacuously. It needs neither pytest nor Lightning:

.. code-block:: python

    from finetuning_scheduler.testing.conformance import TunableViewConformance

    report = TunableViewConformance(lambda: MyView(build_model())).run(raise_on_failure=True)

``finetuning_scheduler.testing.fixtures`` provides four known-good fixture views to compare against.
