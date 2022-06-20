.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer
    from finetuning_scheduler.fts import FinetuningScheduler

.. _finetuning_scheduler:

Introduction to the Finetuning Scheduler
========================================

The :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback accelerates and enhances
foundational model experimentation with flexible finetuning schedules. Training with the
:class:`~finetuning_scheduler.fts.FinetuningScheduler` callback is simple and confers a host of benefits:

* it dramatically increases finetuning flexibility
* expedites and facilitates exploration of model tuning dynamics
* enables marginal performance improvements of finetuned models

.. note::
   If you're exploring using the :class:`~finetuning_scheduler.fts.FinetuningScheduler`, this is a great place
   to start!
   You may also find the `notebook-based tutorial <https://lightning-ai.github.io/tutorials/notebooks/lightning_examples/finetuning-scheduler.html>`_
   useful and for those using the :doc:`LightningCLI<cli/lightning_cli>`, there is a
   :ref:`CLI-based<scheduled-finetuning-superglue>` example at the bottom of this introduction.

Setup
*****
Setup is straightforward, just install from PyPI!

.. code-block:: bash

   pip install finetuning-scheduler

Additional installation options (from source etc.) are discussed under "Additional installation options" in the
`README <https://github.com/speediedan/finetuning-scheduler/blob/main/README.md>`_

.. _motivation:

Motivation
**********
Fundamentally, the :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback enables
multi-phase, scheduled finetuning of foundational models. Gradual unfreezing (i.e. thawing) can help maximize
foundational model knowledge retention while allowing (typically upper layers of) the model to optimally adapt to new
tasks during transfer learning [#]_ [#]_ [#]_ .

:class:`~finetuning_scheduler.fts.FinetuningScheduler` orchestrates the gradual unfreezing
of models via a finetuning schedule that is either implicitly generated (the default) or explicitly provided by the user
(more computationally efficient). Finetuning phase transitions are driven by
:class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping` criteria (a multi-phase
extension of :external+pl:class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`),
user-specified epoch transitions or a composition of the two (the default mode). A
:class:`~finetuning_scheduler.fts.FinetuningScheduler` training session completes when the
final phase of the schedule has its stopping criteria met. See
:ref:`Early Stopping<common/early_stopping:Early stopping>` for more details on that callback's configuration.

Basic Usage
***********
If no finetuning schedule is user-provided, :class:`~finetuning_scheduler.fts.FinetuningScheduler` will generate a
:ref:`default schedule<index:The Default Finetuning Schedule>` and proceed to finetune
according to the generated schedule, using default
:class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping`
and :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` callbacks with
``monitor=val_loss``.

.. code-block:: python

    from pytorch_lightning import Trainer
    from finetuning_scheduler import FinetuningScheduler

    trainer = Trainer(callbacks=[FinetuningScheduler()])


.. _default schedule:

The Default Finetuning Schedule
*******************************
Schedule definition is facilitated via
:meth:`~finetuning_scheduler.fts_supporters.SchedulingMixin.gen_ft_schedule` which dumps
a default finetuning schedule (by default using a naive, 2-parameters per level heuristic) which can be adjusted as
desired by the user and/or subsequently passed to the callback. Using the default/implicitly generated schedule will
often be less computationally efficient than a user-defined finetuning schedule but can often serve as a
good baseline for subsequent explicit schedule refinement and will marginally outperform many explicit schedules.


.. _specifying schedule:

Specifying a Finetuning Schedule
********************************

To specify a finetuning schedule, it's convenient to first generate the default schedule and then alter the
thawed/unfrozen parameter groups associated with each finetuning phase as desired. Finetuning phases are zero-indexed
and executed in ascending order. In addition to being zero-indexed, finetuning phase keys should be contiguous and
either integers or convertible to integers via ``int()``.

1. First, generate the default schedule to ``Trainer.log_dir``. It will be named after your
   :external+pl:class:`~pytorch_lightning.core.module.LightningModule` subclass with the suffix
   ``_ft_schedule.yaml``.

.. code-block:: python

    from pytorch_lightning import Trainer
    from finetuning_scheduler import FinetuningScheduler

    trainer = Trainer(callbacks=[FinetuningScheduler(gen_ft_sched_only=True)])


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

        ... to have three finetuning phases instead of four:

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

3. Once the finetuning schedule has been altered as desired, pass it to
   :class:`~finetuning_scheduler.fts.FinetuningScheduler` to commence scheduled training:

.. code-block:: python

    from pytorch_lightning import Trainer
    from finetuning_scheduler import FinetuningScheduler

    trainer = Trainer(callbacks=[FinetuningScheduler(ft_schedule="/path/to/my/schedule/my_schedule.yaml")])

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
see :ref:`scheduled finetuning for SuperGLUE<scheduled-finetuning-superglue>` below or the
`notebook-based tutorial <https://lightning-ai.github.io/tutorials/notebooks/lightning_examples/finetuning-scheduler.html>`_.


Resuming Scheduled Finetuning Training Sessions
***********************************************

Resumption of scheduled finetuning training is identical to the continuation of
:ref:`other training sessions<common/trainer:trainer>` with the caveat that the provided checkpoint must
have been saved by a :class:`~finetuning_scheduler.fts.FinetuningScheduler` session.
:class:`~finetuning_scheduler.fts.FinetuningScheduler` uses
:class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` (an extension of
:external+pl:class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`) to maintain schedule state with
special metadata.


.. code-block:: python

    from pytorch_lightning import Trainer
    from finetuning_scheduler import FinetuningScheduler

    trainer = Trainer(callbacks=[FinetuningScheduler()], ckpt_path="some/path/to/my_checkpoint.ckpt")

Training will resume at the depth/level of the provided checkpoint according the specified schedule. Schedules can be
altered between training sessions but schedule compatibility is left to the user for maximal flexibility. If executing a
user-defined schedule, typically the same schedule should be provided for the original and resumed training
sessions.


.. tip::

    By default (
    :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.restore_best` is ``True``),
    :class:`~finetuning_scheduler.fts.FinetuningScheduler` will attempt to restore
    the best available checkpoint before finetuning depth transitions.

    .. code-block:: python

        trainer = Trainer(
            callbacks=[FinetuningScheduler()],
            ckpt_path="some/path/to/my_kth_best_checkpoint.ckpt",
        )

    Note that similar to the behavior of
    :external+pl:class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`,
    (`specifically this PR <https://github.com/Lightning-AI/lightning/pull/12045>`_), when resuming training
    with a different :class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` ``dirpath`` from the provided
    checkpoint, the new training session's checkpoint state will be re-initialized at the resumption depth with the
    provided checkpoint being set as the best checkpoint.

Finetuning all the way down!
****************************

There are plenty of options for customizing
:class:`~finetuning_scheduler.fts.FinetuningScheduler`'s behavior, see
:ref:`scheduled finetuning for SuperGLUE<scheduled-finetuning-superglue>` below for examples of composing different
configurations.


.. note::
   Currently, :class:`~finetuning_scheduler.fts.FinetuningScheduler` supports the following
   :external+pl:class:`~pytorch_lightning.strategies.Strategy` s:

   .. hlist::
      :columns: 3

      * :external+pl:class:`~pytorch_lightning.strategies.ddp.DDPStrategy`
      * :external+pl:class:`~pytorch_lightning.strategies.sharded.DDPShardedStrategy`
      * :external+pl:class:`~pytorch_lightning.strategies.ddp_spawn.DDPSpawnStrategy`
      * :external+pl:class:`~pytorch_lightning.strategies.sharded_spawn.DDPSpawnShardedStrategy`
      * :external+pl:class:`~pytorch_lightning.strategies.dp.DataParallelStrategy`

.. tip::
    Custom or officially unsupported strategies can be used by setting
    :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.allow_untested` to ``True``.

    Some officially unsupported strategies may work unaltered and are only unsupported due to
    the ``Finetuning Scheduler`` project's lack of CI/testing resources for that strategy (e.g.
    ``single_tpu``).

    Most unsupported strategies, however, are currently unsupported because they require varying degrees of modification
    to be compatible (e.g. ``deepspeed`` requires an ``add_param_group`` method, ``tpu_spawn`` an override of the
    current broadcast method to include python objects).

----------

.. _scheduled-finetuning-superglue:

Example: Scheduled Finetuning For SuperGLUE
*******************************************
A demonstration of the scheduled finetuning callback
:class:`~finetuning_scheduler.fts.FinetuningScheduler` using the
`RTE <https://huggingface.co/datasets/viewer/?dataset=super_glue&config=rte>`_ and
`BoolQ <https://github.com/google-research-datasets/boolean-questions>`_ tasks of the
`SuperGLUE <https://super.gluebenchmark.com/>`_ benchmark and the :doc:`LightningCLI<cli/lightning_cli>`
is available under ``./fts_examples/``.

Since this CLI-based example requires a few additional packages (e.g. ``transformers``, ``sentencepiece``), you
should install them using the ``[examples]`` extra:

.. code-block:: bash

   pip install finetuning-scheduler['examples']

There are three different demo schedule configurations composed with shared defaults (./config/fts_defaults.yaml)
provided for the default 'rte' task. Note DDP (with auto-selected GPUs) is the default configuration so ensure you
adjust the configuration files referenced below as desired for other configurations.

Note there will likely be minor variations in training paths and performance as packages (e.g. ``transformers``,
``datasets``, ``finetuning-scheduler`` itself etc.) evolve. The precise package versions and salient environmental
configuration used in the building of this tutorial is available in the tensorboard summaries, logs and checkpoints
referenced below if you're interested.

.. code-block:: bash

    # Generate a baseline without scheduled finetuning enabled:
    python fts_superglue.py fit --config config/nofts_baseline.yaml

    # Train with the default finetuning schedule:
    python fts_superglue.py fit --config config/fts_implicit.yaml

    # Train with a non-default finetuning schedule:
    python fts_superglue.py fit --config config/fts_explicit.yaml


All three training scenarios use identical configurations with the exception of the provided finetuning schedule. See
the |tensorboard_summ| and table below for a characterization of the relative computational and performance tradeoffs
associated with these :class:`~finetuning_scheduler.fts.FinetuningScheduler` configurations.

:class:`~finetuning_scheduler.fts.FinetuningScheduler` expands the space of possible
finetuning schedules and the composition of more sophisticated schedules can yield marginal finetuning performance
gains. That stated, it should be emphasized the primary utility of
:class:`~finetuning_scheduler.fts.FinetuningScheduler` is to grant greater finetuning
flexibility for model exploration in research. For example, glancing at DeBERTa-v3's implicit training run, a critical
tuning transition point is immediately apparent:

.. raw:: html

    <div style="max-width:400px; width:50%; height:auto;">
        <a target="_blank" rel="noopener noreferrer" href="https://tensorboard.dev/experiment/n7U8XhrzRbmvVzC4SQSpWw/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfZXhwbGljaXQiOmZhbHNlLCJub2Z0c19iYXNlbGluZSI6ZmFsc2UsImZ0c19pbXBsaWNpdCI6dHJ1ZX0%3D">
            <img alt="open tensorboard experiment" src="_static/images/fts/implicit_training_transition.png">
        </a>
    </div>

Our val_loss begins a precipitous decline at step 3119 which corresponds to phase 17 in the schedule. Referring to our
schedule, in phase 17 we're beginning tuning the attention parameters of our 10th encoder layer (of 11). Interesting!
Though beyond the scope of this documentation, it might be worth investigating these dynamics further and
:class:`~finetuning_scheduler.fts.FinetuningScheduler` allows one to do just that quite
easily.

In addition to the `tensorboard experiment summaries <https://tensorboard.dev/experiment/n7U8XhrzRbmvVzC4SQSpWw/>`_,
full logs/schedules for all three scenarios
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
   * - | Finetuning Schedule
     - None
     - Default
     - User-defined
   * - | RTE Accuracy
       | (``0.81``, ``0.84``, ``0.85``)
     -
        .. raw:: html

            <div style='width:150px;height:auto'>
                <a target="_blank" rel="noopener noreferrer" href="https://tensorboard.dev/experiment/n7U8XhrzRbmvVzC4SQSpWw/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfZXhwbGljaXQiOmZhbHNlLCJub2Z0c19iYXNlbGluZSI6dHJ1ZSwiZnRzX2ltcGxpY2l0IjpmYWxzZX0%3D">
                    <img alt="open tensorboard experiment" src="_static/images/fts/nofts_baseline_accuracy_deberta_base.png">
                </a>
            </div>
     -
        .. raw:: html

            <div style='width:150px;height:auto'>
                <a target="_blank" rel="noopener noreferrer" href="https://tensorboard.dev/experiment/n7U8XhrzRbmvVzC4SQSpWw/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfZXhwbGljaXQiOmZhbHNlLCJub2Z0c19iYXNlbGluZSI6ZmFsc2UsImZ0c19pbXBsaWNpdCI6dHJ1ZX0%3D">
                    <img alt="open tensorboard experiment" src="_static/images/fts/fts_implicit_accuracy_deberta_base.png">
                </a>
            </div>
     -
        .. raw:: html

            <div style='width:150px;height:auto'>
                <a target="_blank" rel="noopener noreferrer" href="https://tensorboard.dev/experiment/n7U8XhrzRbmvVzC4SQSpWw/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfZXhwbGljaXQiOnRydWUsIm5vZnRzX2Jhc2VsaW5lIjpmYWxzZSwiZnRzX2ltcGxpY2l0IjpmYWxzZX0%3D">
                    <img alt="open tensorboard experiment" src="_static/images/fts/fts_explicit_accuracy_deberta_base.png">
                </a>
            </div>

Note that though this example is intended to capture a common usage scenario, substantial variation is expected among
use cases and models. In summary, :class:`~finetuning_scheduler.fts.FinetuningScheduler`
provides increased finetuning flexibility that can be useful in a variety of contexts from exploring model tuning
behavior to maximizing performance.

.. figure:: _static/images/fts/fts_explicit_loss_anim.gif
   :alt: FinetuningScheduler Explicit Loss Animation
   :width: 300

.. note:: The :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback is currently in beta.

Footnotes
*********

.. [#] `Howard, J., & Ruder, S. (2018) <https://arxiv.org/pdf/1801.06146.pdf>`_. Fine-tuned Language Models for Text
 Classification. ArXiv, abs/1801.06146.
.. [#] `Chronopoulou, A., Baziotis, C., & Potamianos, A. (2019) <https://arxiv.org/pdf/1902.10547.pdf>`_. An
 embarrassingly simple approach for transfer learning from pretrained language models. arXiv preprint arXiv:1902.10547.
.. [#] `Peters, M. E., Ruder, S., & Smith, N. A. (2019) <https://arxiv.org/pdf/1903.05987.pdf>`_. To tune or not to
 tune? adapting pretrained representations to diverse tasks. arXiv preprint arXiv:1903.05987.

.. seealso::
    - :external+pl:class:`~pytorch_lightning.trainer.trainer.Trainer`
    - :external+pl:class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`
    - :external+pl:class:`~pytorch_lightning.callbacks.finetuning.BaseFinetuning`

.. |tensorboard_summ| raw:: html

            <a target="_blank" rel="noopener noreferrer" href="https://tensorboard.dev/experiment/Qy917MVDRlmkx31A895CzA/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfZXhwbGljaXQiOnRydWUsImZ0c19pbXBsaWNpdCI6dHJ1ZSwibm9mdHNfYmFzZWxpbmUiOnRydWV9">
            tensorboard experiment summaries
            </a>

.. raw:: html

   <div style="display:none">

.. toctree::
   :maxdepth: 2
   :name: api
   :caption: API

   fts_api

.. toctree::
   :maxdepth: 1
   :name: Advanced Usage
   :caption: Advanced Usage

   advanced/lr_scheduler_reinitialization

.. toctree::
   :maxdepth: 1
   :name: Examples
   :caption: Examples

   Notebook-based Finetuning Scheduler tutorial <https://lightning-ai.github.io/tutorials/notebooks/lightning_examples/finetuning-scheduler.html>
   CLI-based Finetuning Scheduler tutorial <https://finetuning-scheduler.readthedocs.io/en/latest/#example-scheduled-finetuning-for-superglue>

.. toctree::
   :maxdepth: 1
   :name: Community
   :caption: Community

   generated/CODE_OF_CONDUCT.md
   generated/CONTRIBUTING.md
   governance
   generated/CHANGELOG.md

.. raw:: html

   </div>

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
