.. testsetup:: *

    from lightning.pytorch.trainer.trainer import Trainer
    from finetuning_scheduler.fts import FinetuningScheduler

.. _finetuning_scheduler:

Introduction to the Fine-Tuning Scheduler
=========================================

Fine-Tuning Scheduler (FTS) accelerates and enhances foundation model experimentation with flexible, multi-phase
fine-tuning schedules. A schedule says which parameters to thaw in each phase; FTS thaws them, adds them to the
optimizer's parameter groups (optionally reinitializing the optimizer or lr scheduler) and decides when each phase ends.
Training with FTS is simple and confers a host of benefits:

* it dramatically increases fine-tuning flexibility
* expedites and facilitates exploration of model tuning dynamics
* enables marginal performance improvements of fine-tuned models

FTS runs a schedule in either of two hosts, which share one scheduling core and one schedule format:

* **your own PyTorch training loop**, through :class:`~finetuning_scheduler.core.controller.FTSController`
  (experimental in 2.15; it imports neither Lightning nor jsonargparse);
* **Lightning**, through the :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback (fully supported).

.. note::
   If you're exploring Fine-Tuning Scheduler, this is a great place to start! :ref:`basic-usage` below walks through
   both hosts, and there are tutorials for each: the :doc:`PyTorch API tutorial
   notebook<notebooks/superglue/fts_superglue_pytorch>`, the :doc:`Lightning tutorial
   notebook<notebooks/superglue/fts_superglue_lightning>` and, for those using the
   :py:class:`~lightning.pytorch.cli.LightningCLI`, a :ref:`CLI-based<scheduled-fine-tuning-superglue>` example.

Setup
*****
`uv <https://docs.astral.sh/uv/>`_ is the preferred installation approach for
Fine-Tuning Scheduler.

.. code-block:: bash

   # Install uv if you haven't already (one-time setup)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install Fine-Tuning Scheduler
   uv pip install finetuning-scheduler

Additional installation options (from source etc.) are discussed under "Additional installation options" in the
`README <https://github.com/speediedan/finetuning-scheduler/blob/main/README.md>`_

.. _motivation:

Motivation
**********
Fundamentally, Fine-Tuning Scheduler enables multi-phase, scheduled fine-tuning of foundation models. Gradual
unfreezing (i.e. thawing) can help maximize foundation model knowledge retention while allowing (typically upper layers
of) the model to optimally adapt to new tasks during transfer learning [#]_ [#]_ [#]_ .

FTS orchestrates the gradual unfreezing of models via a fine-tuning schedule that is either implicitly generated (the
default) or explicitly provided by the user (more computationally efficient). Fine-tuning phase transitions are driven
by epoch limits for each phase (``max_transition_epoch``), by criteria on a monitored metric (a plateau in validation
loss, say), or by a composition of the two. A training session completes when the final phase of the schedule has its
stopping criteria met.

.. _basic-usage:

Basic Usage
***********

In Your Own PyTorch Loop
========================
Build an :class:`~finetuning_scheduler.core.controller.FTSController` for your model, call
:meth:`~finetuning_scheduler.core.controller.FTSController.setup` once, iterate over
:meth:`~finetuning_scheduler.core.controller.FTSController.epochs`, and report each epoch's validation metrics with
:meth:`~finetuning_scheduler.core.controller.FTSController.epoch_end`. The controller decides when each phase starts and
which parameters train; the loop stays yours. Passing ``None`` as the schedule uses the
:ref:`default schedule<default schedule>`:

.. code-block:: python

    from functools import partial

    import torch
    from finetuning_scheduler import FTSController, PlateauTransition

    fts = FTSController(
        model,
        None,  # the default schedule, or a mapping, or the path to a schedule YAML file
        optimizer=partial(torch.optim.AdamW, lr=1e-3),
        transition=PlateauTransition(monitor="val_loss", patience=2, min_delta=0.01),
    )
    fts.setup()
    for epoch in fts.epochs(max_epochs=30):
        for x, y in train_loader:
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            fts.optimizer.step()  # always `fts.optimizer`: a phase can replace the optimizer
            fts.optimizer.zero_grad()
        fts.epoch_end({"val_loss": evaluate(model)})

:ref:`plain-pytorch-loop` covers the rest: transition policies, checkpoints and exact resume, phase hooks and gates,
and distributed training.

With Lightning
==============
Add the :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback to your ``Trainer``. Without a user-provided
schedule, it generates the :ref:`default schedule<default schedule>` and fine-tunes according to it, with default
:class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` and
:class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` callbacks monitoring ``val_loss``:

.. code-block:: python

    import lightning as L
    from finetuning_scheduler import FinetuningScheduler

    trainer = L.Trainer(callbacks=[FinetuningScheduler()])

:ref:`lightning-scheduled-fine-tuning` covers the rest: the callback's dependencies, early-stopping and epoch-driven
transitions, resuming, and the supported strategies and lr schedulers.

.. _default schedule:

The Default Fine-Tuning Schedule
********************************
The default schedule thaws parameters in pairs, starting from the last registered parameter (a naive, two-parameters
per level heuristic). It is produced by :func:`~finetuning_scheduler.core.schedule.gen_ft_schedule`, which writes it to
a YAML file named after the model's class with the suffix ``_ft_schedule.yaml``, and under Lightning by
:meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.gen_ft_schedule`, which strategy adapters can override.
Using the default/implicitly generated schedule will often be less computationally efficient than a user-defined
fine-tuning schedule but can often serve as a good baseline for subsequent explicit schedule refinement and will
marginally outperform many explicit schedules.


.. _specifying schedule:

Specifying a Fine-Tuning Schedule
*********************************

To specify a fine-tuning schedule, it's convenient to first generate the default schedule and then alter the
thawed/unfrozen parameter groups associated with each fine-tuning phase as desired. Fine-tuning phases are zero-indexed
and executed in ascending order. In addition to being zero-indexed, fine-tuning phase keys should be contiguous and
either integers or convertible to integers via ``int()``.

1. First, generate the default schedule.

.. code-block:: python

    # in your own loop: write it to a directory
    from finetuning_scheduler.core.schedule import gen_ft_schedule

    gen_ft_schedule(model, "schedules/")

    # with Lightning: write it to `FinetuningScheduler.log_dir` (`Trainer.log_dir` by default) without training
    trainer = L.Trainer(callbacks=[FinetuningScheduler(gen_ft_sched_only=True)])


2. Alter the schedule as desired.

.. container:: sbs-code

    .. rst-class:: sbs-hdr1

        Changing the generated schedule for this boring model...

    .. rst-class:: sbs-blk1

    .. code-block:: yaml
      :linenos:

        0:
            params:
            - layer.3.bias
            - layer.3.weight
        1:
            params:
            - layer.2.bias
            - layer.2.weight
        2:
            params:
            - layer.1.bias
            - layer.1.weight
        3:
            params:
            - layer.0.bias
            - layer.0.weight

    .. rst-class:: sbs-hdr2

        ... to have three fine-tuning phases instead of four:

    .. rst-class:: sbs-blk2

    .. code-block:: yaml
      :linenos:

        0:
            params:
            - layer.3.bias
            - layer.3.weight
        1:
            params:
            - layer.2.*
            - layer.1.bias
            - layer.1.weight
        2:
            params:
            - layer.0.*

3. Once the fine-tuning schedule has been altered as desired, pass it to commence scheduled training:

.. code-block:: python

    # in your own loop
    fts = FTSController(model, "/path/to/my/schedule/my_schedule.yaml", optimizer=..., transition=...)

    # with Lightning
    trainer = L.Trainer(callbacks=[FinetuningScheduler(ft_schedule="/path/to/my/schedule/my_schedule.yaml")])

.. tip::

    Use of regex expressions can be convenient for specifying more complex schedules. Also, a per-phase maximum
    learning rate (``lr``) and an epoch limit (``max_transition_epoch``) can be specified:

    .. code-block:: yaml
      :linenos:
      :emphasize-lines: 2, 5, 7, 12, 13, 15

       0:
         params: # the parameters for each phase definition can be fully specified
         - model.classifier.bias
         - model.classifier.weight
         max_transition_epoch: 3
       1:
         params: # or specified via a regex
         - model.albert.pooler.*
       2:
         params:
         - model.albert.encoder.*.ffn_output.*
         max_transition_epoch: 9
         lr: 1e-06 # per-phase maximum learning rates can be specified
       3:
         params: # both approaches to parameter specification can be used in the same phase
         - model.albert.encoder.*.(ffn\.|attention|full*).*
         - model.albert.encoder.embedding_hidden_mapping_in.bias
         - model.albert.encoder.embedding_hidden_mapping_in.weight
         - model.albert.embeddings.*

Phase Transitions and Resuming
******************************
A phase ends at its ``max_transition_epoch`` or when its transition criteria are met, whichever comes first. In your
own loop, the criteria are transition policies such as
:class:`~finetuning_scheduler.core.transitions.PlateauTransition` (see :ref:`plain-pytorch-loop`); under Lightning, they
are :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` criteria (see
:ref:`lightning-scheduled-fine-tuning`). Both hosts restore the best checkpoint at each phase boundary by default when
checkpointing is configured, and both resume a session at the depth it was saved at:
:meth:`~finetuning_scheduler.core.controller.FTSController.save` and
:meth:`~finetuning_scheduler.core.controller.FTSController.load` in your own loop, a checkpoint saved by
:class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` under Lightning.

Where to Go Next
****************

- **Your own PyTorch loop:** :ref:`plain-pytorch-loop`, the :ref:`model-parallel example<model-parallel-plain-loop>`
  (FSDP2 and tensor parallelism), and :doc:`tunable views<pytorch/tunable_views>` for addressing a model through
  logical parameter names.
- **Lightning:** :ref:`lightning-scheduled-fine-tuning`, :doc:`model-parallel scheduled
  fine-tuning<distributed/model_parallel_scheduled_fine_tuning>`, :doc:`FSDP1 scheduled
  fine-tuning<distributed/fsdp_scheduled_fine_tuning>` and the :doc:`FSDP1-to-FSDP2 migration
  guide<distributed/fsdp1_to_fsdp2_migration>`.
- **Either host:** :doc:`lr scheduler<advanced/lr_scheduler_reinitialization>` and
  :doc:`optimizer<advanced/optimizer_reinitialization>` reinitialization at phase transitions.

----------

Example: Scheduled Fine-Tuning For SuperGLUE
********************************************
The CLI-based example of scheduled fine-tuning on the SuperGLUE RTE and BoolQ tasks now has its own page:
:ref:`scheduled-fine-tuning-superglue`.

Footnotes
*********

.. [#] `Howard, J., & Ruder, S. (2018) <https://arxiv.org/pdf/1801.06146.pdf>`_. Fine-tuned Language Models for Text
 Classification. ArXiv, abs/1801.06146.
.. [#] `Chronopoulou, A., Baziotis, C., & Potamianos, A. (2019) <https://arxiv.org/pdf/1902.10547.pdf>`_. An
 embarrassingly simple approach for transfer learning from pretrained language models. arXiv preprint arXiv:1902.10547.
.. [#] `Peters, M. E., Ruder, S., & Smith, N. A. (2019) <https://arxiv.org/pdf/1903.05987.pdf>`_. To tune or not to
 tune? adapting pretrained representations to diverse tasks. arXiv preprint arXiv:1903.05987.

.. raw:: html

   <div style="display:none">

.. toctree::
   :name: Introduction
   :caption: Introduction

   self

.. toctree::
   :maxdepth: 1
   :name: PyTorch API
   :caption: PyTorch API (experimental)

   pytorch/plain_pytorch_loop
   PyTorch API tutorial notebook <notebooks/superglue/fts_superglue_pytorch>
   pytorch/model_parallel_plain_loop
   pytorch/tunable_views

.. toctree::
   :maxdepth: 1
   :name: Lightning
   :caption: Lightning

   lightning/scheduled_fine_tuning
   Notebook-based Fine-Tuning Scheduler tutorial <notebooks/superglue/fts_superglue_lightning>
   CLI-based Fine-Tuning Scheduler tutorial <examples/superglue_cli>

.. toctree::
   :maxdepth: 1
   :name: Enhanced Distributed Strategies
   :caption: Enhanced Distributed Strategies (Lightning)

   distributed/model_parallel_scheduled_fine_tuning
   distributed/fsdp_scheduled_fine_tuning
   distributed/fsdp1_to_fsdp2_migration

.. toctree::
   :maxdepth: 1
   :name: Advanced Usage
   :caption: Advanced Usage

   advanced/lr_scheduler_reinitialization
   advanced/optimizer_reinitialization

.. toctree::
   :maxdepth: 1
   :name: Configurable Profiling
   :caption: Configurable Profiling

   profiling/memprofiler_profiling

.. toctree::
   :maxdepth: 1
   :name: Plugins
   :caption: Plugins

   plugins/strategy_adapter_entry_points

.. toctree::
   :maxdepth: 2
   :name: Advanced Installation Options
   :caption: Advanced Installation Options

   install/dynamic_versioning

.. toctree::
   :maxdepth: 2
   :name: api
   :caption: APIs

   fts_api
   core_api
   memprofiler_api

.. toctree::
   :maxdepth: 1
   :name: Community
   :caption: Community

   generated/CODE_OF_CONDUCT.md
   generated/CONTRIBUTING.md
   versioning
   governance
   generated/CHANGELOG.md

.. raw:: html

   </div>

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
