.. testsetup:: *

    from lightning.pytorch.trainer.trainer import Trainer
    from finetuning_scheduler.fts import FinetuningScheduler

.. _finetuning_scheduler:

Introduction to the Fine-Tuning Scheduler
=========================================

The :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback accelerates and enhances
foundation model experimentation with flexible fine-tuning schedules. Training with the
:class:`~finetuning_scheduler.fts.FinetuningScheduler` (FTS) callback is simple and confers a host of benefits:

* it dramatically increases fine-tuning flexibility
* expedites and facilitates exploration of model tuning dynamics
* enables marginal performance improvements of fine-tuned models

.. note::
   If you're exploring using the :class:`~finetuning_scheduler.fts.FinetuningScheduler`, this is a great place
   to start!
   You may also find the `notebook-based tutorial <https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/finetuning-scheduler.html>`_
   useful and for those using the :external+pl:class:`~lightning.pytorch.cli.LightningCLI`, there is a
   :ref:`CLI-based<scheduled-fine-tuning-superglue>` example at the bottom of this introduction.

Setup
*****
Starting with version 2.10, `uv <https://docs.astral.sh/uv/>`_ is the preferred installation approach for
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
Fundamentally, the :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback enables
multi-phase, scheduled fine-tuning of foundation models. Gradual unfreezing (i.e. thawing) can help maximize
foundation model knowledge retention while allowing (typically upper layers of) the model to optimally adapt to new
tasks during transfer learning [#]_ [#]_ [#]_ .

:class:`~finetuning_scheduler.fts.FinetuningScheduler` orchestrates the gradual unfreezing
of models via a fine-tuning schedule that is either implicitly generated (the default) or explicitly provided by the user
(more computationally efficient). fine-tuning phase transitions are driven by
:class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` criteria (a multi-phase
extension of :external+pl:class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping`),
user-specified epoch transitions or a composition of the two (the default mode). A
:class:`~finetuning_scheduler.fts.FinetuningScheduler` training session completes when the
final phase of the schedule has its stopping criteria met. See
:ref:`Early Stopping<common/early_stopping:Early stopping>` for more details on that callback's configuration.

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
    :external+pl:class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping`,
    :external+pl:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`) the provided configuration of
    those callbacks will be used to instantiate their FTS analogs instead.

.. _default schedule:

The Default Fine-Tuning Schedule
********************************
Schedule definition is facilitated via
:meth:`~finetuning_scheduler.fts_supporters.ScheduleImplMixin.gen_ft_schedule` which dumps
a default fine-tuning schedule (by default using a naive, 2-parameters per level heuristic) which can be adjusted as
desired by the user and/or subsequently passed to the callback. Using the default/implicitly generated schedule will
often be less computationally efficient than a user-defined fine-tuning schedule but can often serve as a
good baseline for subsequent explicit schedule refinement and will marginally outperform many explicit schedules.


.. _specifying schedule:

Specifying a Fine-Tuning Schedule
*********************************

To specify a fine-tuning schedule, it's convenient to first generate the default schedule and then alter the
thawed/unfrozen parameter groups associated with each fine-tuning phase as desired. Fine-tuning phases are zero-indexed
and executed in ascending order. In addition to being zero-indexed, fine-tuning phase keys should be contiguous and
either integers or convertible to integers via ``int()``.

1. First, generate the default schedule (output to :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.log_dir`,
   defaults to ``Trainer.log_dir``). It will be named after your
   :external+pl:class:`~lightning.pytorch.core.module.LightningModule` subclass with the suffix ``_ft_schedule.yaml``.

.. code-block:: python

    import lightning as L
    from finetuning_scheduler import FinetuningScheduler

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

3. Once the fine-tuning schedule has been altered as desired, pass it to
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
     ``BatchNorm.track_running_stats``. Beginning with FTS ``2.4.0``, FTS overrides this behavior by default so that
     even frozen ``BatchNorm`` layers continue to have ``track_running_stats`` set to ``True``. To disable
     ``BatchNorm.track_running_stats`` when freezing ``torch.nn.modules.batchnorm._BatchNorm`` modules, one can set the
     FTS parameter :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.frozen_bn_track_running_stats` to ``False``.


EarlyStopping and Epoch-Driven Phase Transition Criteria
********************************************************

By default, :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` and epoch-driven
transition criteria are composed. If a ``max_transition_epoch`` is specified for a given phase, the next finetuning
phase will begin at that epoch unless
:class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` criteria are met first.
If :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.epoch_transitions_only` is
``True``, :class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` will not be used
and transitions will be exclusively epoch-driven.

.. tip::

    Use of regex expressions can be convenient for specifying more complex schedules. Also, a per-phase
    :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.base_max_lr` can be specified:

    .. code-block:: yaml
      :linenos:
      :emphasize-lines: 2, 7, 13, 15

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

For a practical end-to-end example of using
:class:`~finetuning_scheduler.fts.FinetuningScheduler` in implicit versus explicit modes,
see :ref:`scheduled fine-tuning for SuperGLUE<scheduled-fine-tuning-superglue>` below or the
`notebook-based tutorial <https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/finetuning-scheduler.html>`_.


Resuming Scheduled Fine-Tuning Training Sessions
************************************************

Resumption of scheduled fine-tuning training is identical to the continuation of
:ref:`other training sessions<common/trainer:trainer>` with the caveat that the provided checkpoint must
have been saved by a :class:`~finetuning_scheduler.fts.FinetuningScheduler` session.
:class:`~finetuning_scheduler.fts.FinetuningScheduler` uses
:class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` (an extension of
:external+pl:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`) to maintain schedule state with
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
    :external+pl:class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`, when resuming training
    with a different :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` ``dirpath`` from the provided
    checkpoint, the new training session's checkpoint state will be re-initialized at the resumption depth with the
    provided checkpoint being set as the best checkpoint.

Fine-Tuning All The Way Down!
*****************************

There are plenty of options for customizing
:class:`~finetuning_scheduler.fts.FinetuningScheduler`'s behavior, see
:ref:`scheduled fine-tuning for SuperGLUE<scheduled-fine-tuning-superglue>` below for examples of composing different
configurations.

.. _supported_strategies:

.. note::

   Currently, :class:`~finetuning_scheduler.fts.FinetuningScheduler` supports the following distributed strategies:

   .. hlist::
      :columns: 2

      * :external+pl:class:`~lightning.pytorch.strategies.ddp.DDPStrategy`:``ddp``,
        ``ddp_find_unused_parameters_false``, ``ddp_find_unused_parameters_true``, ``ddp_spawn``, ``ddp_fork``,
        ``ddp_notebook``
      * :external+pl:class:`~lightning.pytorch.strategies.fsdp.FSDPStrategy`:
        ``fsdp``, ``fsdp_cpu_offload``

.. _supported_lr_schedulers:

.. note::
   Currently, :class:`~finetuning_scheduler.fts.FinetuningScheduler` officially supports the following torch lr
   schedulers:

   .. hlist::
      :columns: 2

      * :external+torch:class:`~torch.optim.lr_scheduler.StepLR`
      * :external+torch:class:`~torch.optim.lr_scheduler.MultiStepLR`
      * :external+torch:class:`~torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`
      * :external+torch:class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`
      * :external+torch:class:`~torch.optim.lr_scheduler.LambdaLR`
      * :external+torch:class:`~torch.optim.lr_scheduler.ConstantLR`
      * :external+torch:class:`~torch.optim.lr_scheduler.LinearLR`
      * :external+torch:class:`~torch.optim.lr_scheduler.ExponentialLR`
      * :external+torch:class:`~torch.optim.lr_scheduler.CosineAnnealingLR`
      * :external+torch:class:`~torch.optim.lr_scheduler.MultiplicativeLR`

.. _supported_reinit_optimizers:

.. note::
   :class:`~finetuning_scheduler.fts.FinetuningScheduler` supports reinitializing all PyTorch optimizers (or subclasses
   thereof) `provided in torch.optim <https://pytorch.org/docs/stable/optim.html#algorithms>`_ in the context of all
   supported training strategies (including FSDP). Use of
   :external+torch:class:`~torch.distributed.optim.ZeroRedundancyOptimizer` is also supported, but currently only
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

    Regarding lr schedulers, :external+torch:class:`~torch.optim.lr_scheduler.ChainedScheduler` and
    :external+torch:class:`~torch.optim.lr_scheduler.SequentialLR` are examples of schedulers not currently supported
    due to the configuration complexity and semantic conflicts supporting them would introduce. If a supported torch lr
    scheduler does not meet your requirements, one can always subclass a supported lr scheduler and modify it as
    required (e.g. :external+torch:class:`~torch.optim.lr_scheduler.LambdaLR` is especially useful for this). PRs are
    also always welcome!

----------

.. _scheduled-fine-tuning-superglue:

Example: Scheduled Fine-Tuning For SuperGLUE
********************************************
A demonstration of the scheduled fine-tuning callback
:class:`~finetuning_scheduler.fts.FinetuningScheduler` using the
`RTE <https://huggingface.co/datasets/viewer/?dataset=super_glue&config=rte>`_ and
`BoolQ <https://github.com/google-research-datasets/boolean-questions>`_ tasks of the
`SuperGLUE <https://paperswithcode.com/dataset/superglue>`_ benchmark and the
:external+pl:class:`~lightning.pytorch.cli.LightningCLI` is available under ``./fts_examples``.

Since this CLI-based example requires a few additional packages (e.g. ``transformers``, ``sentencepiece``), you
should install them using the ``[examples]`` extra:

.. code-block:: bash

   uv pip install finetuning-scheduler['examples']

There are three different demo schedule configurations composed with shared defaults (./config/fts_defaults.yaml)
provided for the default 'rte' task. Note DDP (with auto-selected GPUs) is the default configuration so ensure you
adjust the configuration files referenced below as desired for other configurations.

Note there will likely be minor variations in training paths and performance as packages (e.g. ``transformers``,
``datasets``, ``finetuning-scheduler`` itself etc.) evolve. The precise package versions and salient environmental
configuration used in the building of this tutorial is available in the logs and checkpoints referenced below if you're
interested.

.. code-block:: bash

    # Generate a baseline without scheduled fine-tuning enabled:
    python fts_superglue.py fit --config config/nofts_baseline.yaml

    # Train with the default fine-tuning schedule:
    python fts_superglue.py fit --config config/fts_implicit.yaml

    # Train with a non-default fine-tuning schedule:
    python fts_superglue.py fit --config config/fts_explicit.yaml


All three training scenarios use identical configurations with the exception of the provided fine-tuning schedule. See
the table below for a characterization of the relative computational and performance tradeoffs associated with these
:class:`~finetuning_scheduler.fts.FinetuningScheduler` configurations.

:class:`~finetuning_scheduler.fts.FinetuningScheduler` expands the space of possible
fine-tuning schedules and the composition of more sophisticated schedules can yield marginal fine-tuning performance
gains. That stated, it should be emphasized the primary utility of
:class:`~finetuning_scheduler.fts.FinetuningScheduler` is to grant greater fine-tuning
flexibility for model exploration in research. For example, glancing at DeBERTa-v3's implicit training run, a critical
tuning transition point is immediately apparent:

.. raw:: html

    <div style="max-width:400px; width:50%; height:auto;">
        <img src="_static/images/fts/implicit_training_transition.png">
    </div>

Our val_loss begins a precipitous decline at step 3119 which corresponds to phase 17 in the schedule. Referring to our
schedule, in phase 17 we're beginning tuning the attention parameters of our 10th encoder layer (of 11). Interesting!
Though beyond the scope of this documentation, it might be worth investigating these dynamics further and
:class:`~finetuning_scheduler.fts.FinetuningScheduler` allows one to do just that quite
easily.

Full logs/schedules for all three scenarios
`are available <https://drive.google.com/file/d/1LrUcisRLHeJgh_BDOOD_GUBPp5iHAkoR/view?usp=sharing>`_
as well as the `checkpoints produced <https://drive.google.com/file/d/1t7myBgcqcZ9ax_IT9QVk-vFH_l_o5UXB/view?usp=sharing>`_
in the scenarios (caution, ~3.5GB).

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - | **Example Scenario**
     - | **nofts_baseline**
     - | **fts_implicit**
     - | **fts_explicit**
   * - | Fine-Tuning Schedule
     - None
     - Default
     - User-defined
   * - | RTE Accuracy
       | (``0.81``, ``0.84``, ``0.85``)
     -
        .. raw:: html

            <div style='width:150px;height:auto'>
                <img src="_static/images/fts/nofts_baseline_accuracy_deberta_base.png">
            </div>
     -
        .. raw:: html

            <div style='width:150px;height:auto'>
                <img src="_static/images/fts/fts_implicit_accuracy_deberta_base.png">
            </div>
     -
        .. raw:: html

            <div style='width:150px;height:auto'>
                <img src="_static/images/fts/fts_explicit_accuracy_deberta_base.png">
            </div>

Note that though this example is intended to capture a common usage scenario, substantial variation is expected among
use cases and models. In summary, :class:`~finetuning_scheduler.fts.FinetuningScheduler`
provides increased fine-tuning flexibility that can be useful in a variety of contexts from exploring model tuning
behavior to maximizing performance.

.. figure:: _static/images/fts/fts_explicit_loss_anim.gif
   :alt: FinetuningScheduler Explicit Loss Animation
   :width: 300

Footnotes
*********

.. [#] `Howard, J., & Ruder, S. (2018) <https://arxiv.org/pdf/1801.06146.pdf>`_. Fine-tuned Language Models for Text
 Classification. ArXiv, abs/1801.06146.
.. [#] `Chronopoulou, A., Baziotis, C., & Potamianos, A. (2019) <https://arxiv.org/pdf/1902.10547.pdf>`_. An
 embarrassingly simple approach for transfer learning from pretrained language models. arXiv preprint arXiv:1902.10547.
.. [#] `Peters, M. E., Ruder, S., & Smith, N. A. (2019) <https://arxiv.org/pdf/1903.05987.pdf>`_. To tune or not to
 tune? adapting pretrained representations to diverse tasks. arXiv preprint arXiv:1903.05987.

.. seealso::
    - :external+pl:class:`~lightning.pytorch.trainer.trainer.Trainer`
    - :external+pl:class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping`
    - :external+pl:class:`~lightning.pytorch.callbacks.finetuning.BaseFinetuning`

.. raw:: html

   <div style="display:none">

.. toctree::
   :name: Introduction
   :caption: Introduction

   self

.. toctree::
   :maxdepth: 1
   :name: Enhanced Distributed Strategies
   :caption: Enhanced Distributed Strategies

   distributed/model_parallel_scheduled_fine_tuning
   distributed/fsdp_scheduled_fine_tuning

.. toctree::
   :maxdepth: 1
   :name: Configurable Profiling
   :caption: Configurable Profiling

   profiling/memprofiler_profiling

.. toctree::
   :maxdepth: 1
   :name: Advanced Usage
   :caption: Advanced Usage

   advanced/lr_scheduler_reinitialization
   advanced/optimizer_reinitialization

.. toctree::
   :maxdepth: 1
   :name: Basic Examples
   :caption: Basic Examples

   Notebook-based Fine-Tuning Scheduler tutorial <https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/finetuning-scheduler.html>
   CLI-based Fine-Tuning Scheduler tutorial <https://finetuning-scheduler.readthedocs.io/en/stable/#example-scheduled-fine-tuning-for-superglue>

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
