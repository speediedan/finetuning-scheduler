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
- **A phase may mix whole tensors and gradient addresses**, as phase 1 does, and so may one set: in
  ``(@whole & of:value) | (@rows & of:grad)`` the value part trains its tensors whole and the gradient part is masked.
  A union of gradient disjuncts takes each disjunct's mask, united per tensor.
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
mask's hook would not run). Other operations on the gradients a schedule addresses are
:ref:`gradient transforms<tunable-views-gradient-transforms>`.

.. _tunable-views-gradient-transforms:

Transforming Gradients
**********************
A phase's ``grad_transforms`` (the PyTorch API only) maps entries, written as ``params`` entries are, to transforms of
each step's accumulated gradient of the tensors they select: scale a component's gradient without splitting parameter
groups, keep its update out of a protected direction, bound it, or apply a function of your own.

.. code-block:: python

    from finetuning_scheduler.core.grad_transforms import Clamp, Reject, Scale

    concept = torch.randn(8)  # a direction in the middle layer's output space
    first = {"module:2 & leaf:weight": Reject(concept), "module:2 & units:[0, 1] & of:grad": Scale(0.1)}
    schedule = {
        0: {"params": ["module:2"], "max_transition_epoch": 2, "grad_transforms": first},
        1: {"params": ["4.weight"], "grad_transforms": {"module:2": [Scale(0.5), Clamp(-1.0, 1.0)]}},
    }

A transform is any callable ``transform(name, grad, units)`` returning the gradient's new value, where ``units`` are
the indices along its first axis that the entry's gradient unit set keeps (``None`` without one). The built-ins in
:mod:`~finetuning_scheduler.core.grad_transforms` (:class:`~finetuning_scheduler.core.grad_transforms.Scale`,
:class:`~finetuning_scheduler.core.grad_transforms.Replace`, :class:`~finetuning_scheduler.core.grad_transforms.Add`,
:class:`~finetuning_scheduler.core.grad_transforms.Clamp`, :class:`~finetuning_scheduler.core.grad_transforms.Project`
and :class:`~finetuning_scheduler.core.grad_transforms.Reject`) act on those rows when given. An entry selects tensors
only to transform them: it neither thaws nor masks them, and a tensor it selects must be trainable in the phase.

.. _tunable-views-treatments:

**Treatments.** ``grad_transforms`` may instead name a treatment of the schedule's address books. In venngram, a book's
sets select what to address, and its ``treatments`` say what is done to them for one model. Each treatment is a
*recipe* (``venngram.recipe.Recipe``; see `Large unit sets and treatments
<https://github.com/speediedan/venngram#large-unit-sets-and-treatments>`_ in venngram's README), stored in the book
under a name of its author's choosing. A recipe can hold several sections: ``stages`` act on activations,
``weights`` edit parameters, and ``gradients`` act on each step's parameter gradients. Fine-Tuning Scheduler applies
the ``gradients`` section, and a phase names the treatment by its key. Because the ops are data rather than callables, a
schedule read from YAML can name one:

.. code-block:: yaml

    addresses:
      sets:
        mlp_13: {component: mlp.out, layers: [13]}
        neuron_grads: {component: mlp.out, layers: [13], leaf: weight, units: [52, 101], of: grad}
        down_grad: {component: mlp.out, layers: [13], leaf: weight, of: grad}
      treatments:
        keep_neurons_out:
          model: {id: Qwen/Qwen3-4B, revision: main}
          operands:
            concept: {file: dirs.safetensors, key: concept, sha256: ..., shape: [2560], basis: unfolded}
          gradients:
            - {op: replace, target: {set: neuron_grads}, value: {dose: 0.0}}
            - {op: reject, target: {set: down_grad}, operand: concept}
    0:
      params: ["@mlp_13"]
      grad_transforms: keep_neurons_out

Read top to bottom:

- ``sets`` are ordinary named sets. ``mlp_13`` is layer 13's MLP output projection, which phase 0 trains.
  ``neuron_grads`` and ``down_grad`` address the gradient of that projection's weight (``of: grad``):
  ``neuron_grads`` covers only units 52 and 101, and ``down_grad`` covers all of it.
- ``keep_neurons_out`` is the treatment's name: a key under ``treatments``, and what ``grad_transforms`` names.
- ``model`` is the model the treatment was written for, since its units and directions are that model's.
- ``operands`` are the data its ops take, by name. Here ``concept`` is one direction in the layer's output space,
  read from a safetensors sidecar beside the book and checked against its hash and shape.
- ``gradients`` lists the ops, each on the set its ``target`` names. The first replaces units 52 and 101's gradients
  with zero (a dose of ``0.0``), so those two neurons' outgoing weights stay fixed while the rest of the projection
  trains. The second rejects ``concept`` from the projection's gradient, so no update moves its output along that
  direction.

So phase 0 trains layer 13's MLP output projection while keeping two neurons and one direction out of the update. That
is what the name ``keep_neurons_out`` records.

Its ops apply in order, each to the gradients of the trainable tensors its target set selects, as the built-ins:
``scale``, ``replace`` and ``add`` with a ``dose`` or per-unit ``doses``, and ``project`` and ``reject`` with a
``vector`` or ``subspace`` operand and a ``fraction``. Where the built-ins act along a tensor's first axis, a treatment
follows venngram's meaning: a target set with ``units`` acts along each tensor's unit axis (a down projection's input
side, for one), and a span's directions run along a weight's ``out`` side, both as the view's component map records
them (``venngram.view(model)``; a span uses the first axis without one). A treatment with ``stages`` or ``weights``,
another op, or a value taken ``from`` evidence or through a value transform is refused, as Fine-Tuning Scheduler's
support record declares (its ``steering`` section lists the ``gradient_ops`` it applies).

The rules:

- **Transforms apply in the order written**, an entry's list in order and the entries in the mapping's order, and
  before the tensor's mask, so no transform refills a masked row. Like a mask, they act before clipping and the
  optimizer step.
- **A phase's transforms carry on until a later phase states its own**, which replace them; ``grad_transforms: {}``
  clears them, and a ``refreeze`` drops a tensor's transforms with it.
- **They follow the schedule** through optimizer reinitialization, promotion and resume, and a state saved under other
  transforms is refused. Each phase that changes them logs the tensors transformed, and
  :attr:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler.gradient_transforms` gives the transforms in
  effect. A transform should keep no state of its own: a resumed run rebuilds the transforms from the schedule.

Transforms are refused on the Lightning callback and on a sharded model, as masks are, and a schedule with
them is not exported as an address book: a callable is not portable, and a treatment is already in the book that holds
it.

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
use is not judged, so one book can serve a steering tool and a schedule. The record's ``steering`` section declares
the ops on parameter gradients it applies from a treatment a phase names (``gradient_ops``), and a treatment that asks
for more is refused the same way. A book can be checked against the record without Fine-Tuning Scheduler installed:

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
becomes a set of gradients, in an address book of schema 2. Every phase is one set: a phase mixing whole tensors and
gradient addresses (phase 1 of the :ref:`example above<tunable-views-gradient-masks>`) becomes a union of a value part,
each value entry stating ``of: value``, and a gradient part, and a phase of several gradient entries with unit sets a
union whose disjuncts each give their own mask. Either compiles back to the phase's masks, and the support record
declares both readings (venngram's ``mixed-sets`` and ``union-masks``).

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
