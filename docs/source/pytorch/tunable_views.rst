.. _tunable-views:

##################################################
Scheduling Through a Tunable View
##################################################

.. warning::
   Tunable views are **experimental** in 2.15 (see the :ref:`API stability classifications<versioning>`): the protocol
   may still change.

.. tip::
   **You may not need a view.** A schedule of parameter names and regexes, or the generated default schedule, needs
   none, and stays the simplest choice for many uses, above all when you write or edit a schedule by hand. That way of
   scheduling remains fully supported.

   Views and addresses exist for what names do badly:

   - **interpretability experiments**, which tune, freeze and promote adapters, interventions, latent models and
     instruments by role, often through several wrappers of one model;
   - **composable, shareable model components**: named sets of parameters or units, defined once for a model family
     and reused, sliced and recombined;
   - **experiments defined once and run across models**: conditions written as addresses depend far less on any one
     model's parameter names, so the same definition can be compiled, compared and orchestrated over several models;
   - **agent-orchestrated experiments**, where a model description of roles, layers and components is easier to
     compose over than regexes. If an architecture has no component map, an agent can draft and verify one with the
     ``tunable-view-generator`` agent skill (in the repository's ``.agents/skills/``, usable from any agent harness
     that reads skills).

   :ref:`fts-evolution` explains the direction, with a table of which approach fits which use. The
   :doc:`LoRA placement notebook</examples/notebooks/tunable_views/lora_placement>` shows a view and addresses at work in an experiment, and
   :doc:`Addressing Across Architectures</examples/notebooks/tunable_views/cross_architecture_addresses>` shows one set of addresses on four model families.

Overview
********
**The same schedule** runs **unchanged** on a Llama, OPT or Gemma model, with PEFT's LoRA or the examples' own. A
schedule written through a tunable view says *what* to tune ("the MLP adapters of the top four blocks") rather than
which parameter names hold it, so it does not depend on one architecture or one adapter library. The same approach
builds schedules shared across interpretability packages: a package that gives the model it wraps a view reporting the
same roles and components runs the same schedule.

To enable this model addressing abstraction, Fine-Tuning Scheduler uses the
`venngram <https://github.com/speediedan/venngram>`_ protocol. Its addresses can share and compose references to any
subset of a model: from groups of layers to intersecting sets of individual neurons, and from latent-space features to
activation sites and attribution graphs. Schedules use the part of it that selects tensors to tune.

A schedule names parameters, by default with the names ``named_parameters()`` reports. A
:class:`~finetuning_scheduler.core.views.TunableView` tells Fine-Tuning Scheduler more about a model's tensors than
their names:

- **what each tensor is**: its :ref:`role<tunable-views-roles>` (the model's own weights, an adapter, an intervention,
  a latent model, ...);
- **where it sits**: its layer and an architecture-neutral component, such as ``attn.q`` or ``mlp.out``;
- optionally, **names of its own**: a wrapped model often has a vocabulary of its own (an interpretability library may
  call a block's query weights ``blocks.3.attn.W_Q`` while the model stores them inside one fused projection), and a
  view can let a schedule use it.

A schedule addresses the model through the view, and Fine-Tuning Scheduler tunes and checkpoints the *physical*
tensors that result. One logical name may stand for several physical tensors, and a view may tune related tensors (a block's normalization layer, for example) together with the ones a phase names.

The protocol is independent of Fine-Tuning Scheduler, which is one consumer of this nascent model addressing
protocol.

Views work with both FTS interfaces: :class:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler` (pass ``view=``)
and the Lightning callback (attach the view to the ``LightningModule``). A model without a view is scheduled through
:class:`~finetuning_scheduler.core.views.CanonicalView`, the identity view, so nothing changes for existing schedules.

Giving a Model a View
*********************
One call gives a model its view: :func:`venngram.view`. The view keeps the model's own parameter names (so names and
regexes still work through it), gives every tensor a role, and places it at its layer and architecture-neutral
component. It is a view of venngram's tunable view protocol (``venngram.tuning``, with the role vocabulary in
``venngram.roles``), which :mod:`finetuning_scheduler.core.views` and :mod:`finetuning_scheduler.core.roles`
re-export. This is the view the :doc:`LoRA placement notebook</examples/notebooks/tunable_views/lora_placement>`
checks with the conformance suite and trains, on Qwen3-0.6B with LoRA on all seven projections:

.. code-block:: python

    import torch
    import venngram
    from transformers import AutoModelForCausalLM

    from finetuning_scheduler import TuningScheduler
    from fts_examples.tunable_views.lora import add_lora

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    add_lora(model, ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), r=16)
    model.requires_grad_(False)

    # the MLP adapters first, then the attention adapters as well
    schedule = {
        0: {"params": ["role:adapter & component:mlp.*"], "max_transition_epoch": 1},
        1: {"params": ["role:adapter & component:attn.*"]},
    }
    fts = TuningScheduler(
        model, schedule, optimizer=lambda groups: torch.optim.AdamW(groups, lr=2e-4), view=venngram.view(model)
    )

Through this view, a LoRA tensor takes the component of the projection it adapts, and everything else is ``base``:

.. list-table::
   :header-rows: 1

   * - tensor
     - role
     - layer
     - component
   * - ``model.layers.27.self_attn.q_proj.lora_A.weight``
     - adapter
     - 27
     - ``attn.q``
   * - ``model.layers.27.mlp.down_proj.base.weight``
     - base
     - 27
     - ``mlp.out``
   * - ``model.layers.0.post_attention_layernorm.weight``
     - base
     - 0
     - ``ln2``
   * - ``model.embed_tokens.weight``
     - base
     - (none)
     - ``embed``

When only roles are wanted, :class:`~finetuning_scheduler.core.views.CanonicalView` with ``roles=`` gives a plain model
roles (and layers, from its block list) while keeping its own names.

Fine-Tuning Scheduler compiles the schedule once, at setup, into the physical parameters each phase tunes
(:func:`~finetuning_scheduler.core.compile.compile_schedule`). Compilation refuses a schedule whose phases share
storage, a name the view cannot resolve, and a phase that schedules a tensor whose role is never tuned. A digest of the
compiled schedule is saved with every checkpoint.

.. _tunable-views-finding-maps:

Where a View Comes From
=======================
``venngram.view(model)`` composes the view from published data: the *map* of the model's architecture and the map of
each adapter library whose modules the model carries (PEFT's LoRA, a spliced SAE or transcoder, a steering module).
Neither needs code, so an architecture or an adapter library is covered once its maps are published. For an
architecture venngram has no map for, the ``tunable-view-generator`` agent skill (in the repository's
``.agents/skills/``) drafts one from the model's module tree and verifies it, without downloading weights, and can
then register it: in the running process, or by publishing it for a registry to index. venngram's README describes the
maps, where they are looked up, offline use, and writing one
(`Maps: base and overlay <https://github.com/speediedan/venngram#maps-base-and-overlay>`_). A view of your own, for a
vocabulary or behavior no map gives, is covered in :ref:`tunable-views-writing`.

.. _tunable-views-roles:

Roles
*****
A role says what a tensor *is*, and so what Fine-Tuning Scheduler may do with it. Every tensor has exactly one:

.. list-table::
   :header-rows: 1
   :widths: 14 24 12 12 38

   * - role
     - what it is
     - trained
     - checkpointed
     - why it is a role of its own
   * - ``base``
     - the model's own weights: every tensor the view does not list
     - when selected
     - yes
     - the default; it takes no name, because a subset of the model's own weights is an address
       (``role:base & component:ln1``), not a role
   * - ``adapter``, ``adapter:<name>``
     - parameter-efficient adapters (LoRA and similar)
     - when selected
     - yes
     - added weights, which ``role:adapter`` and :ref:`export by role<tunable-views-export>` take apart from the
       model's own
   * - ``intervention``, ``intervention:<name>``
     - modules that intervene in the forward pass (steering, for example)
     - when selected
     - yes
     - trained like an adapter, but "the adapters", in a schedule or an export, must not take in steering modules,
       nor the reverse
   * - ``latent:<name>``
     - learned latent dictionaries (a spliced transcoder, for example)
     - when selected
     - yes
     - the only role :ref:`promotion<tunable-views-promotion>` produces; always named, because a model can carry
       several dictionaries, each scheduled on its own
   * - ``fitted``
     - state computed during the run without gradients (a calibrated statistic, for example)
     - never
     - yes
     - the only role checkpointed but never trained: a resumed run needs its values, and no optimizer may touch them
   * - ``instrument``
     - a published artifact loaded for analysis (a pretrained SAE, for example)
     - never (until promoted)
     - no
     - reloaded from its source rather than checkpointed; the only role that can be promoted

The trained roles behave alike in Fine-Tuning Scheduler; they are separate kinds rather than names of one kind so that
the common selector (``role:adapter``) means what it says when a model carries several kinds of added tensors, and so
that tools reading a view can tell them apart. ``fitted`` and ``instrument`` exist for interpretability work, where
interpretune's views use them.

Selectors and Addresses
***********************
A schedule selects tensors by role: ``role:adapter`` selects every adapter, ``role:adapter:math`` exactly the
adapter named ``math`` (one of several a model carries, say ``math`` and ``code``).
Selectors work without a view too, where every parameter is ``base``, except under a Lightning strategy adapter that
names parameters itself (see :ref:`tunable-views-renaming-adapters`).

.. code-block:: python

    schedule = {0: {"params": ["role:adapter"], "max_transition_epoch": 2}, 1: {"params": ["role:latent"]}}

**Addresses.** A role selector is one *field*; combining fields narrows the selection.
``role:adapter & layers:-4:`` is the adapters of the top four blocks, in any model whose block list the view finds:
``role:adapter`` and ``layers:-4:`` are fields, and ``&`` intersects them; the whole expression is an *address*. Addresses combine with ``&``
(intersection), ``-`` (difference) and ``|`` (union), operators delimited by spaces, and ``@<set>`` references a set
the schedule names. An address is a selection, not a role, so two may overlap: ``role:adapter & module:q_proj`` selects
every ``q_proj`` adapter, including those in the top four blocks.

The fields a schedule uses most:

- ``layers`` takes an index, a list, ``odd``, ``even``, or a half-open slice whose negative indices count from the
  depth (``-4:``, ``8:-4``). It counts the blocks of the model's block list.
- ``component`` matches architecture-neutral components (``component:mlp.*``), through a view built from a map.
- ``module`` matches tensors under a module whose path ends with the pattern, on whole segments.
- ``name`` is a regex over tensor names, quoted in an address (``name:"head\..*"``).

An entry is an address only when it starts with a field prefix (``role:``, ``layers:``, ``module:`` and the others) or
``@``, after any opening parentheses. Every parameter name and regex keeps its meaning. A ``params`` entry is a string;
an address written as a mapping goes in the schedule's ``addresses`` sets and is referenced by ``@<set>``.

A schedule names its sets in a top-level ``addresses`` key: an inline book (a mapping with ``sets``, and optionally
``imports``), the path of an address book relative to the schedule file, or a list of them:

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

The grammar and address books are `venngram <https://github.com/speediedan/venngram#addresses>`__'s, and venngram ships
an agent skill for them, ``venngram-addressing``. Books can also be published and named by reference, and a schedule can be exported as a book: see
:ref:`tunable-views-advanced`.

.. _tunable-views-export:

Exporting by Role
*****************
:meth:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler.export_best` writes the best checkpoint's tensors
whose role matches, for example only the adapters, so a large frozen base and the optimizer state need not be kept:

.. code-block:: python

    fts.export_best(roles=("adapter",))  # best_export.pt beside the best checkpoint
    fts.export_best(write_peft, roles=("role:adapter:math",))  # write_peft(tensors, metadata, path): another format

The tensors come from the best checkpoint, not the live model, keyed by wrapper-free names (without segments such as
``torch.compile``'s ``_orig_mod.``); a sharded model's are read in full on global rank zero. The external artifacts a run depends on can
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

The schedule keeps its logical names (in logs and in the saved schedule), the callback tunes the physical parameters,
and a generated schedule uses the view's logical names. A third-party strategy
adapter that renames parameters takes precedence over an attached view (see :ref:`tunable-views-renaming-adapters`).

Going Further
*************
:ref:`tunable-views-advanced` covers writing and testing a view of your own, promotion (an ``instrument`` becoming a
trainable ``latent``), address books by reference, what a book may ask of Fine-Tuning Scheduler, and exporting a
schedule as a book.
