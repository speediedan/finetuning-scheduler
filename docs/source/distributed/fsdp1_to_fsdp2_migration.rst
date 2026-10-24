##################################
FSDP1 to FSDP2 Migration Guide
##################################

Overview
********

FSDP1 (PyTorch's wrapper-based ``FullyShardedDataParallel``, Lightning's
:py:class:`~lightning.pytorch.strategies.fsdp.FSDPStrategy`) flattens each wrapped module's parameters into one
``FlatParameter``. FSDP2 (the composable ``fully_shard`` API, driven by Lightning's
:py:class:`~lightning.pytorch.strategies.model_parallel.ModelParallelStrategy`) shards each parameter separately as a
``DTensor`` and composes with tensor parallelism. For scheduled fine-tuning, the per-parameter layout removes most of
the module-wrapping work FSDP1 needed to keep phases apart.

This guide maps an FTS FSDP1 configuration to its FSDP2 equivalent, covering strategy configuration, fine-tuning
schedules and checkpoints. It assumes the :doc:`FSDP1 guide <fsdp_scheduled_fine_tuning>` and the
:doc:`model-parallel guide <model_parallel_scheduled_fine_tuning>`, which documents ``fsdp_plan`` in full.

Strategy Configuration
**********************

Under FSDP1, the strategy wraps the model (``auto_wrap_policy``) and FTS's
:class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter` adds module-name overrides (``awp_overrides``).
Under FSDP2, FTS's :class:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter` applies ``fully_shard``
itself, from an ``fsdp_plan`` passed in ``strategy_adapter_cfg``, so module selection moves from the strategy to the
plan.

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - FSDP1 (``FSDPStrategy``)
      - FSDP2 (``ModelParallelStrategy`` with ``fsdp_plan``)
    * - ``strategy: FSDPStrategy``
      - ``strategy: ModelParallelStrategy`` with ``data_parallel_size`` (the number of devices) and
        ``tensor_parallel_size: 1``
    * - ``auto_wrap_policy: ModuleWrapPolicy({Cls, ...})``
      - an ``fsdp_plan`` key ``class:Cls`` per class, which selects the same instances, then the model itself last
    * - ``awp_overrides: [names or regexes]``
      - ``fsdp_plan`` name or regex keys, often unnecessary (see :ref:`fsdp1-to-fsdp2-schedules`)
    * - ``activation_checkpointing_policy: {Cls, ...}``
      - ``act_ckpt: ['composable']`` on the corresponding ``fsdp_plan`` entries (or in ``fsdp_default_kwargs``)
    * - ``sharding_strategy: FULL_SHARD`` / ``SHARD_GRAD_OP``
      - ``reshard_after_forward: True`` / ``False``
    * - ``use_orig_params``
      - nothing: FSDP2 always keeps the original parameters, as ``DTensor`` s
    * - mixed precision (FSDP1's ``MixedPrecision``, which Lightning derives from the Trainer's ``precision``)
      - ``fully_shard``'s ``mp_policy`` (a ``MixedPrecisionPolicy`` object, so given in Python, for example in
        ``fsdp_default_kwargs``)
    * - ``cpu_offload: True``
      - refused under ``ModelParallelStrategy`` (see the
        :ref:`CPU offload note<model-parallel-fsdp2-auto-plan-aliases>`); use the PyTorch API
        (:class:`~finetuning_scheduler.FTSController` with :func:`~finetuning_scheduler.strategies.plan.shard_model`,
        whose plan may set ``cpu_offload_policy``)
    * - ``state_dict_type: full`` / ``sharded``
      - ``save_distributed_checkpoint: False`` / ``True`` (the default)

``fsdp_plan`` entries are applied in order, so list a module's children before the module (bottom-up), and end with
the model itself, so every parameter is sharded: a parameter outside every FSDP unit would remain a plain tensor beside
the ``DTensor`` s in the optimizer's parameter groups.

Example
=======

The FSDP1 example ``fts_examples/config/advanced/fsdp/fts_fsdp_awp_overrides_profile.yaml`` configures:

.. code-block:: yaml

    trainer:
      callbacks:
      - class_path: finetuning_scheduler.FinetuningScheduler
        init_args:
          ft_schedule: ./config/RteBoolqModule_ft_schedule_deberta_base_fsdp.yaml
          max_depth: 2
          strategy_adapter_cfg:
            awp_overrides: ["model.pooler.dense", "model.classifier"]
      strategy:
        class_path: lightning.pytorch.strategies.FSDPStrategy
        init_args:
          activation_checkpointing_policy: !!set
            ? transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Layer
          auto_wrap_policy:
            class_path: torch.distributed.fsdp.wrap.ModuleWrapPolicy
            init_args:
              module_classes: !!set
                ? transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Layer
                ? transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Embeddings
                ? transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Encoder

Its FSDP2 equivalent, on the same two devices and with the same schedule:

.. code-block:: yaml

    trainer:
      callbacks:
      - class_path: finetuning_scheduler.FinetuningScheduler
        init_args:
          ft_schedule: ./config/RteBoolqModule_ft_schedule_deberta_base_fsdp.yaml
          max_depth: 2
          strategy_adapter_cfg:
            fsdp_plan:
              "class:DebertaV2Layer": {act_ckpt: ["composable"]}  # the activation checkpointing policy's class
              "class:DebertaV2Embeddings": {}
              "class:DebertaV2Encoder": {}  # after the layers it contains
              "model": {}  # the model itself, last
      strategy:
        class_path: lightning.pytorch.strategies.ModelParallelStrategy
        init_args:
          data_parallel_size: 2
          tensor_parallel_size: 1

The ``awp_overrides`` have no counterpart: the pooler and classifier stay in the model's own unit, which is the
outermost one (see below). Applied with :func:`~finetuning_scheduler.strategies.plan.shard_model` to a small
DeBERTa-v2 configuration, this plan runs forward and backward, and FTS reports no unit spanning phases for the example's
schedule. Adding ``model.pooler.dense`` and ``model.classifier`` entries before ``class:DebertaV2Encoder`` (the
literal translation of the overrides) works too.

.. _fsdp1-to-fsdp2-schedules:

Fine-Tuning Schedules
*********************

A fine-tuning schedule needs no change: it names the same parameters under FSDP2. What changes is which module
layouts a schedule allows.

**Under FSDP1** (with ``use_orig_params=False``), all the parameters flattened into one ``FlatParameter`` share
``requires_grad``, so the parameters of each FSDP unit must thaw in the same phase. FTS refuses a wrapping that
violates this before training starts, and ``awp_overrides`` exist to wrap separately the modules (a classifier, a
pooler) that thaw apart from their neighbours.

**Under FSDP2**, each parameter is sharded on its own, so a unit may hold parameters that thaw in different phases.
That costs communication, not correctness, and only for a unit that reshards after forward: once any of its parameters
is trainable, its backward pass all-gathers all of them again, those not yet trained included. When FTS validates the
schedule it warns about each such unit, naming the phases (see the
:ref:`units spanning phases note<model-parallel-fsdp2-auto-plan-aliases>`). The outermost unit is kept unsharded after
forward by default, so parameters left to it can thaw in any phase without that cost, which is why the example above
drops the overrides.

In practice:

- Keep a unit per block that thaws as a whole (``class:DebertaV2Layer`` above), as under FSDP1.
- Modules that FSDP1 needed wrapped separately only because they thaw in their own phase can usually stay in the
  outermost unit.
- A unit spanning phases inside the model (``DebertaV2Encoder`` holding all its layers, without the
  ``class:DebertaV2Layer`` entry) trains correctly, with a warning about the extra all-gathers. Give its
  earlier-thawing children their own units, or set ``reshard_after_forward: False`` on it.

The same plan format serves the PyTorch API: :func:`~finetuning_scheduler.strategies.plan.shard_model` applies it to a
plain PyTorch model for :class:`~finetuning_scheduler.FTSController`.

Checkpoints
***********

A run checkpointed under FSDP1 can resume under FSDP2 with its model, optimizer and fine-tuning schedule state: a full
checkpoint directly, a sharded one after consolidation with Lightning's ``consolidate_checkpoint`` utility. See
:ref:`model-parallel-fsdp1-resume` for the procedure, including how the FSDP1 run's best checkpoint is restored.

Under ``ModelParallelStrategy``, ``save_distributed_checkpoint`` (``True`` by default) writes a distributed checkpoint
directory; ``False`` writes a single file, as FSDP1's ``state_dict_type: full`` does.

.. note::

    FSDP1's restriction on resuming across ``use_orig_params`` settings has no FSDP2 counterpart, since FSDP2 has no
    such setting.
