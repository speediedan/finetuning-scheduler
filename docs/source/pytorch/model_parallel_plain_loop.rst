.. _model-parallel-plain-loop:

######################################################
Model-Parallel Scheduled Fine-Tuning in a Plain Loop
######################################################

.. warning::
   This example uses the **experimental** PyTorch API (see :ref:`plain-pytorch-loop`), which may still change.

Overview
********
``fts_examples/model_parallel/torch_mp_example.py`` is the plain PyTorch counterpart of the
:ref:`Lightning model-parallel examples<model-parallel-fine-tuning-examples>`. It fine-tunes the same ``torchtitan``
Llama model with the same three-phase schedule and the same FSDP2 and tensor-parallel (TP) plans, but the training loop
is its own and :class:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler` schedules it. It is a ``torchrun``
script and imports neither Lightning nor jsonargparse.

.. code-block:: bash

    cd ./fts_examples/model_parallel

    # FSDP2 over two data-parallel ranks
    torchrun --nproc_per_node=2 torch_mp_example.py --config config/torch_fsdp_plan.yaml

    # TP over two ranks, with loss parallelism
    torchrun --nproc_per_node=2 torch_mp_example.py --config config/torch_tp_plan.yaml

Each configuration is read with ``yaml.safe_load`` and merged over ``config/defaults/torch_mp_example_defaults.yaml``,
the counterpart of the Lightning example's shared defaults. ``--max-epochs`` and ``--output-dir`` override the
configuration, and ``--limit-train-batches`` and ``--limit-val-batches`` shorten each epoch. Checkpoints and a
``summary.json`` of each epoch's phase, trainable parameters and losses are written under
``<output_dir>/<run_name>``.

Mapping the Lightning Configuration
***********************************
Each piece of the Lightning example's configuration has a direct equivalent:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Lightning example
     - Plain loop (``torch_mp_example.py``)
   * - ``ModelParallelStrategy`` with ``data_parallel_size`` and ``tensor_parallel_size``
     - ``init_device_mesh("cuda", (dp, tp), mesh_dim_names=("data_parallel", "tensor_parallel"))``
   * - The ``LightningModule`` holding the model as ``self.model``
     - ``ModParallelModule``, which also holds it as ``model``, so parameter and module names
       (``model.output.weight``, ``model.layers.0``) are unchanged
   * - ``apply_tp_plan`` in ``LightningModule.configure_model``
     - ``apply_tp_plan`` from ``fts_examples/model_parallel/parallel_plans.py``, applied before the FSDP2 plan
   * - ``strategy_adapter_cfg``'s ``fsdp_plan`` and ``fsdp_default_kwargs``
     - :func:`~finetuning_scheduler.strategies.plan.shard_model` (``shard_model(model, mesh["data_parallel"],
       fsdp_plan, default_kwargs=fsdp_default_kwargs)``), which takes the same plan and then shards the root module
   * - Materialization by the strategy, then ``init_weights`` in ``on_train_start``
     - ``model.to_empty(device=...)``, then ``init_weights()``
   * - ``FinetuningScheduler(ft_schedule=..., max_depth=2)``
     - ``TuningScheduler(model, "config/defaults/llama_ft_schedule.yaml", max_depth=2, ...)``
   * - ``FTSEarlyStopping(monitor="val_loss", min_delta=0.001, patience=2)``
     - ``PlateauTransition(monitor="val_loss", min_delta=0.001, patience=2)``, composed with the schedule's
       ``max_transition_epoch`` keys
   * - ``FTSCheckpoint(monitor="val_loss", save_top_k=1)``
     - ``BestCheckpoint(dirpath, monitor="val_loss")``; the best state is restored at each phase boundary
   * - The ``optimizer`` and ``lr_scheduler`` configuration
     - Factories (``partial(AdamW, ...)``, ``partial(CosineAnnealingWarmRestarts, ...)``) stepped with
       ``fts.step_lr_schedulers("epoch")``
   * - The ``loss_parallel`` context in ``loss_fn`` and ``backward``
     - One ``loss_parallel()`` context around the loss and ``loss.backward()``, when ``tp`` > 1
   * - The data-parallel sampler Lightning adds
     - ``DistributedSampler(num_replicas=dp_mesh.size(), rank=dp_mesh.get_local_rank())``
   * - ``self.log("val_loss", ..., sync_dist=True)``
     - The validation loss averaged across ranks and passed to ``fts.epoch_end``

Building the Model
******************
The model is built on the meta device, so no rank allocates the full weights. The TP plan is applied first, then the
FSDP2 plan; only then are the sharded weights allocated and initialized:

.. literalinclude:: ../../../src/fts_examples/model_parallel/torch_mp_example.py
   :pyobject: build_model

``shard_model`` shards the modules the plan selects (here the output layer, the final norm and every
``TransformerBlock``, the last by a ``class:`` key, with composable activation checkpointing), then the root module.
Sharding the root matters in a plain loop:

- The tuning scheduler refuses ``DistributedDataParallel`` (see :ref:`plain-pytorch-loop`), so ``fully_shard`` is what
  synchronizes gradients across data-parallel ranks. A parameter outside every ``fully_shard`` call, such as the
  embedding or the final norm under the Lightning plan alone, would be trained on each rank's data alone, and nothing
  would report the divergence.
- With the root sharded, every parameter is a ``DTensor``, so no optimizer parameter group mixes ``DTensor`` and plain
  parameters, which some optimizer implementations do not support.

The final norm gets its own unit for a different reason. The example reshards every unit after forward, the root
included (``reshard_after_forward: True``). Left to the root's unit, the norm (phase 0) would sit beside the embeddings,
which the schedule thaws two phases later, and from phase 0 on every backward pass would all-gather the frozen
embeddings again. The tuning scheduler warns about any FSDP unit that reshards after forward and holds parameters
thawing in different phases when it validates the schedule against the sharded model.

The Training Loop
*****************
The loop calls the tuning scheduler at the points described in :ref:`plain-pytorch-loop`:

.. literalinclude:: ../../../src/fts_examples/model_parallel/torch_mp_example.py
   :pyobject: train

A few things to note:

- **Every rank reports the same metrics.** The validation loss is averaged across ranks before ``epoch_end``, so every
  rank reaches the same plateau decision (the tuning scheduler follows the greatest decision, with a warning, when ranks
  disagree).
- **Tensor-parallel ranks share their batches.** The sampler shards the data across the data-parallel dimension only.
- **The lr scheduler steps after** ``epoch_end``, as Lightning steps an epoch-interval scheduler after validation, so a
  best checkpoint holds the scheduler state of the epoch that produced it.
- **Checkpoints are sharded.** Every parameter is a ``DTensor``, so ``BestCheckpoint`` writes through
  ``torch.distributed.checkpoint``, and each rank restores its own shards at a phase boundary.

With this schedule, phase 0 (the output layer and final norm) and phase 1 (the last ``TransformerBlock``) each last one
epoch, ending at their ``max_transition_epoch``. Phase 2 thaws the remaining blocks and the embedding and runs until
``max_epochs``, or until the plateau criteria stop training. The two-GPU test ``test_model_parallel_examples_torch``
runs both configurations for three epochs, one per phase.
