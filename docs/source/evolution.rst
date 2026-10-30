.. _fts-evolution:

######################################
Why Fine-Tuning Scheduler Is Evolving
######################################

Fine-Tuning Scheduler was first released in 2022 to run multi-phase fine-tuning schedules over the transfer-learning
practice of the time: phased fine-tuning of one pretrained model on a downstream task, such as gradual unfreezing. A
schedule was a short list of parameter names or regexes per phase. That way of working is still fully supported, and
for many uses it is still the simplest (see :ref:`still-the-simplest`).

Usage patterns have broadened a great deal since then. Phased fine-tuning patterns such as gradual unfreezing now
play a smaller role in modern experimentation than adapters, interventions, latent models, individual units and
agent-driven experiments, all of which ask more of a schedule than a list of names. Fine-Tuning Scheduler's 2.15
release starts to evolve with them. This page explains why, what is new, and what stays the same. It is a first
statement of the direction and will be revised as the work proceeds.

What Changed
************

- **What is trained is rarely the whole model.** Most fine-tuning today trains adapters (LoRA and its relatives),
  inserted modules, or a small, deliberately chosen set of the base model's parameters. The interesting unit is a
  *role* a tensor plays (an adapter, an intervention, a latent dictionary) as often as a layer.
- **Interpretability research tunes and intervenes.** Steering vectors, sparse autoencoders, transcoders and
  replacement models are spliced into a model, trained or fitted, then frozen and used as instruments. Some work acts on
  individual neurons with no latent model at all. A schedule now has to say things like "the SAEs at the top four
  layers, after the adapters converge", which a list of parameter names says badly.
- **The same model is wrapped many ways.** One base model may be driven through Hugging Face, TransformerLens, nnsight
  or SAE Lens, each with its own parameter names. A schedule written against one wrapper's names does not carry to the
  next.
- **Experiments are shared and composed.** A useful set of parameters or units for a model family (its attention
  adapters, the neurons behind a behaviour) is worth publishing once and reusing, sliced and recombined, without each
  user rediscovering how a component is named in a given model version.
- **Agents increasingly run the experiments.** An agent orchestrating many tuning runs benefits from a model description
  it can reason over, roles, layers and components, more than from regexes over parameter names.
- **Training loops vary.** Many training loops are written directly in PyTorch, often distributed with FSDP2 or tensor
  parallelism, rather than in a framework.
- **A schedule is no longer only a fine-tuning schedule.** It may train adapters, fit or freeze a latent model, or
  promote an instrument, as steps of an experiment. These docs therefore say *schedule*, or *experiment schedule* where
  a qualifier helps, rather than *fine-tuning schedule*, which remains only in the names of existing APIs
  (``ft_schedule``, :class:`~finetuning_scheduler.fts.FinetuningScheduler`).

What 2.15 Begins
****************

Each change below is experimental in 2.15 (see the :ref:`API stability classifications<versioning>`). Each is a first
step, and each leaves existing schedules alone.

- **A PyTorch API beside the Lightning callback.**
  :class:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler` runs the same schedules in your own training
  loop (:ref:`plain-pytorch-loop`). Both interfaces share one scheduling core.
- **Tunable views.** A :ref:`tunable view<tunable-views>` maps the names a schedule uses (a wrapper's names, or a
  normalized vocabulary) to the tensors that are trained and checkpointed, and gives tensors roles. Every model without
  a view is scheduled through the identity view, so nothing changes for it.
- **Addresses.** An address selects tensors by what they are rather than what they are called:
  ``role:adapter & layers:-4:`` is the adapters of the top four blocks, on any model whose view reports its structure.
  Addresses combine by set algebra, and named sets live in address books, plain data files meant to be shared.
- **Experiments defined less by the model they run on.** Together, a view and addresses separate what an experiment
  does (its conditions, phases and the sets they train) from how a given model names its tensors. The definition is
  written once, largely model independent, and each model it runs on needs only a view. Orchestrating an experiment
  across models, by hand, by a sweep or by an agent, becomes a loop over models and conditions rather than a rewrite
  per model; :doc:`Addressing Across Architectures</examples/notebooks/tunable_views/cross_architecture_addresses>` shows one set of schedules compiling on four model
  families.
- **A neutral protocol package.** The tunable view protocol and the address grammar live in
  `venngram <https://github.com/speediedan/venngram>`_, which knows no training loop and no interpretability package.
  Fine-Tuning Scheduler is one consumer. `interpretune <https://github.com/speediedan/interpretune>`_ is another: it
  supplies the views and site resolvers that make one address mean the same thing across interpretability backends, so
  Fine-Tuning Scheduler can tune and interpretune can steer and analyse the same named slices of a model.

Interpretability-driven examples are becoming the reference workloads, beside the phased fine-tuning examples, because
they exercise what is new: tuning adapters, latent models and units, scheduled by role and address, through several
backends.

.. _still-the-simplest:

What Stays the Same, and When It Is Still the Simplest Choice
*************************************************************

Explicit schedules of parameter names and regexes, and the implicitly generated default schedule, keep their meaning and
remain supported on both interfaces. For many uses they are still the simplest option, especially when you write or edit
a schedule by hand:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - If you want to...
     - use
   * - thaw a plain model top-down, or a handful of named layers
     - the default schedule, or an explicit schedule of names and regexes
   * - write and edit a schedule by hand for one model
     - names and regexes
   * - train a subset of a role (some adapters, not all) across several wrappers of one model
     - a view and addresses
   * - tune or freeze latent models, interventions and instruments, and promote them between phases
     - a view, with roles
   * - share or reuse a named set of parameters or units across models of a family
     - addresses and address books
   * - have an agent compose many schedules over one model
     - a view and addresses (an agent can also :ref:`write the view<tunable-views>`)

For an architecture with a component map (venngram bundles maps for several common ones, and a map for any other is
a short data document), venngram's ``MappedView`` keeps the model's own names and adds roles, layers and
architecture-neutral components. The :doc:`LoRA placement notebook</examples/notebooks/tunable_views/lora_placement>` uses it to run several LoRA placements on
one model as address schedules, and :doc:`Addressing Across Architectures</examples/notebooks/tunable_views/cross_architecture_addresses>` resolves the same addresses
on four model families.

Experimental Status
*******************

venngram is a dependency of Fine-Tuning Scheduler, so tunable views and addresses are always available. They are
experimental: the first schedule in a process that uses an address warns once, saying so. Parameter names and regexes
never warn. Expect the protocol and the grammar to change between releases while they settle, and expect this page to
change with them. Feedback, in an issue or a discussion, shapes where it goes next.
