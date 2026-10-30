.. testsetup:: *

    from lightning.pytorch.trainer.trainer import Trainer
    from finetuning_scheduler.fts import FinetuningScheduler

.. _lightning-scheduled-fine-tuning:

####################################
Scheduled Fine-Tuning with Lightning
####################################

The :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback runs a fine-tuning schedule under Lightning's
:py:class:`~lightning.pytorch.trainer.trainer.Trainer`: the ``Trainer`` runs the loop, and the callback thaws each
phase's parameters, grows the optimizer's parameter groups and decides when each phase ends. Schedules use the same
format as the :ref:`PyTorch API<plain-pytorch-loop>` (see :ref:`specifying schedule`).

Basic Usage
***********
If no fine-tuning schedule is user-provided, :class:`~finetuning_scheduler.fts.FinetuningScheduler` will generate a
:ref:`default schedule<index:The Default Fine-Tuning Schedule>` and proceed to fine-tune
according to the generated schedule, using default
:class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping`
and :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` callbacks with
``monitor=val_loss``.

.. code-block:: python

    import lightning as L
    from finetuning_scheduler import FinetuningScheduler

    trainer = L.Trainer(callbacks=[FinetuningScheduler()])

.. note::
    If not provided, FTS will instantiate its callback dependencies
    (:class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` and
    :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint`) with default configurations and ``monitor=val_loss``.
    If the user provides base versions of these dependencies (e.g.
    :py:class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping`,
    :py:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`) the provided configuration of
    those callbacks will be used to instantiate their FTS analogs instead.

Generating and Passing a Schedule
*********************************
To generate the default schedule without training, set
:paramref:`~finetuning_scheduler.fts.FinetuningScheduler.gen_ft_sched_only`. The schedule is written to
:paramref:`~finetuning_scheduler.fts.FinetuningScheduler.log_dir` (``Trainer.log_dir`` by default), named after your
:py:class:`~lightning.pytorch.core.module.LightningModule` subclass with the suffix ``_ft_schedule.yaml``:

.. code-block:: python

    import lightning as L
    from finetuning_scheduler import FinetuningScheduler

    trainer = L.Trainer(callbacks=[FinetuningScheduler(gen_ft_sched_only=True)])

Once you have altered it as desired (see :ref:`specifying schedule`), pass it to
:class:`~finetuning_scheduler.fts.FinetuningScheduler` to commence scheduled training:

.. code-block:: python

    import lightning as L
    from finetuning_scheduler import FinetuningScheduler

    trainer = L.Trainer(callbacks=[FinetuningScheduler(ft_schedule="/path/to/my/schedule/my_schedule.yaml")])

.. note::

    For each fine-tuning phase, :class:`~finetuning_scheduler.fts.FinetuningScheduler` will unfreeze/freeze parameters
    as directed in the explicitly specified or implicitly generated schedule. Prior to beginning the first phase of
    training (phase ``0``), FinetuningScheduler will inspect the optimizer to determine if the user has manually
    initialized the optimizer with parameters that are non-trainable or otherwise altered the parameter trainability
    states from that expected of the configured phase ``0``. By default, FTS ensures the optimizer configured in
    ``configure_optimizers`` will optimize the parameters (and only those parameters) scheduled to be optimized in phase
    ``0`` of the current fine-tuning schedule. This auto-configuration can be disabled if desired by setting
    :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.enforce_phase0_params` to ``False``.

.. note::

     When freezing ``torch.nn.modules.batchnorm._BatchNorm`` modules, Lightning by default disables
     ``BatchNorm.track_running_stats``. FTS overrides this behavior by default so that
     even frozen ``BatchNorm`` layers continue to have ``track_running_stats`` set to ``True``. To disable
     ``BatchNorm.track_running_stats`` when freezing ``torch.nn.modules.batchnorm._BatchNorm`` modules, one can set the
     FTS parameter :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.frozen_bn_track_running_stats` to ``False``.

EarlyStopping and Epoch-Driven Phase Transition Criteria
********************************************************

By default, :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` (a multi-phase extension of
:py:class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping`) and epoch-driven transition criteria are
composed. If a ``max_transition_epoch`` is specified for a given phase, the next finetuning phase will begin at that
epoch unless :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` criteria are met first. If
:paramref:`~finetuning_scheduler.fts.FinetuningScheduler.epoch_transitions_only` is ``True``,
:class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` will not be used and transitions will be exclusively
epoch-driven. A :class:`~finetuning_scheduler.fts.FinetuningScheduler` training session completes when the final phase
of the schedule has its stopping criteria met. See :ref:`Early Stopping<common/early_stopping:Early stopping>` for more
details on that callback's configuration.

For a practical end-to-end example of using
:class:`~finetuning_scheduler.fts.FinetuningScheduler` in implicit versus explicit modes,
see :ref:`scheduled fine-tuning for SuperGLUE<scheduled-fine-tuning-superglue>` or the
:doc:`notebook-based tutorial<../examples/notebooks/superglue/fts_superglue_lightning>`.

Resuming Scheduled Fine-Tuning Training Sessions
************************************************

Resumption of scheduled fine-tuning training is identical to the continuation of
:ref:`other training sessions<common/trainer:trainer>` with the caveat that the provided checkpoint must
have been saved by a :class:`~finetuning_scheduler.fts.FinetuningScheduler` session.
:class:`~finetuning_scheduler.fts.FinetuningScheduler` uses
:class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` (an extension of
:py:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`) to maintain schedule state with
special metadata.


.. code-block:: python

    import lightning as L
    from finetuning_scheduler import FinetuningScheduler

    trainer = L.Trainer(callbacks=[FinetuningScheduler()])
    trainer.ckpt_path = "some/path/to/my_checkpoint.ckpt"
    trainer.fit(...)

Training will resume at the depth/level of the provided checkpoint according the specified schedule. Schedules can be
altered between training sessions but schedule compatibility is left to the user for maximal flexibility. If executing a
user-defined schedule, typically the same schedule should be provided for the original and resumed training
sessions.


.. tip::

    By default (
    :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.restore_best` is ``True``),
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` will attempt to restore
    the best available checkpoint before fine-tuning depth transitions.

    .. code-block:: python

        trainer = Trainer(callbacks=[FinetuningScheduler()])
        trainer.ckpt_path = "some/path/to/my_kth_best_checkpoint.ckpt"
        trainer.fit(...)

    Note that similar to the behavior of
    :py:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`, when resuming training
    with a different :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` ``dirpath`` from the provided
    checkpoint, the new training session's checkpoint state will be re-initialized at the resumption depth with the
    provided checkpoint being set as the best checkpoint.

Fine-Tuning All The Way Down!
*****************************

There are plenty of options for customizing
:class:`~finetuning_scheduler.fts.FinetuningScheduler`'s behavior, see
:ref:`scheduled fine-tuning for SuperGLUE<scheduled-fine-tuning-superglue>` for examples of composing different
configurations.

.. _supported_strategies:

.. note::

   Currently, :class:`~finetuning_scheduler.fts.FinetuningScheduler` supports the following distributed strategies:

   .. hlist::
      :columns: 2

      * :py:class:`~lightning.pytorch.strategies.ddp.DDPStrategy`:``ddp``,
        ``ddp_find_unused_parameters_false``, ``ddp_find_unused_parameters_true``, ``ddp_spawn``, ``ddp_fork``,
        ``ddp_notebook``
      * :py:class:`~lightning.pytorch.strategies.fsdp.FSDPStrategy`:
        ``fsdp``, ``fsdp_cpu_offload``

.. _supported_lr_schedulers:

.. note::
   Currently, :class:`~finetuning_scheduler.fts.FinetuningScheduler` officially supports the following torch lr
   schedulers:

   .. hlist::
      :columns: 2

      * :py:class:`~torch.optim.lr_scheduler.StepLR`
      * :py:class:`~torch.optim.lr_scheduler.MultiStepLR`
      * :py:class:`~torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`
      * :py:class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`
      * :py:class:`~torch.optim.lr_scheduler.LambdaLR`
      * :py:class:`~torch.optim.lr_scheduler.ConstantLR`
      * :py:class:`~torch.optim.lr_scheduler.LinearLR`
      * :py:class:`~torch.optim.lr_scheduler.ExponentialLR`
      * :py:class:`~torch.optim.lr_scheduler.CosineAnnealingLR`
      * :py:class:`~torch.optim.lr_scheduler.MultiplicativeLR`

.. _supported_reinit_optimizers:

.. note::
   :class:`~finetuning_scheduler.fts.FinetuningScheduler` supports reinitializing all PyTorch optimizers (or subclasses
   thereof) `provided in torch.optim <https://pytorch.org/docs/stable/optim.html#algorithms>`_ in the context of all
   supported training strategies (including FSDP). Use of
   :py:class:`~torch.distributed.optim.ZeroRedundancyOptimizer` is also supported, but currently only
   outside the context of optimizer reinitialization.

.. tip::
    Custom or officially unsupported strategies and lr schedulers can be used by setting
    :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.allow_untested` to ``True``.

    Some officially unsupported strategies may work unaltered and are only unsupported due to
    the Fine-Tuning Scheduler project's lack of CI/testing resources for that strategy (e.g. ``single_tpu``). Most
    unsupported strategies and schedulers, however, are currently unsupported because they require varying degrees of
    modification to be compatible.

    For instance, with respect to strategies, ``deepspeed`` will require a
    :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` similar to the one written for ``FSDP``
    (:class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter`) to be written before support can be added,
    while ``tpu_spawn`` would require an override of the current broadcast method to include python objects.

    Regarding lr schedulers, :py:class:`~torch.optim.lr_scheduler.ChainedScheduler` and
    :py:class:`~torch.optim.lr_scheduler.SequentialLR` are examples of schedulers not currently supported
    due to the configuration complexity and semantic conflicts supporting them would introduce. If a supported torch lr
    scheduler does not meet your requirements, one can always subclass a supported lr scheduler and modify it as
    required (e.g. :py:class:`~torch.optim.lr_scheduler.LambdaLR` is especially useful for this). PRs are
    also always welcome!

.. seealso::
    - :py:class:`~lightning.pytorch.trainer.trainer.Trainer`
    - :py:class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping`
    - :py:class:`~lightning.pytorch.callbacks.finetuning.BaseFinetuning`
