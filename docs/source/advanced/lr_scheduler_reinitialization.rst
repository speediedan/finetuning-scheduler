#############################
LR Scheduler Reinitialization
#############################

**Audience**: Users looking to use pretrained models with Lightning.

----

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


.. _explicit lr reinitialization schedule:

The Default Finetuning Schedule
*******************************
Schedule definition is facilitated via
:meth:`~finetuning_scheduler.fts_supporters.SchedulingMixin.gen_ft_schedule` which dumps
a default finetuning schedule (by default using a naive, 2-parameters per level heuristic) which can be adjusted as
desired by the user and/or subsequently passed to the callback. Using the default/implicitly generated schedule will
often be less computationally efficient than a user-defined finetuning schedule but can often serve as a
good baseline for subsequent explicit schedule refinement and will marginally outperform many explicit schedules.


.. _implicit lr reinitialization schedule:

Specifying a Finetuning Schedule
********************************

To specify a finetuning schedule, it's convenient to first generate the default schedule and then alter the
thawed/unfrozen parameter groups associated with each finetuning phase as desired. Finetuning phases are zero-indexed
and executed in ascending order.

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
notebook-based tutorial (link will be added as soon as it is released on the PyTorch Lightning production documentation
site).

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

.. code-block:: bash

    # Generate a baseline without scheduled finetuning enabled:
    python fts_superglue.py fit --config config/nofts_baseline.yaml

    # Train with the default finetuning schedule:
    python fts_superglue.py fit --config config/fts_implicit.yaml

    # Train with a non-default finetuning schedule:
    python fts_superglue.py fit --config config/fts_explicit.yaml


:class:`~finetuning_scheduler.fts.FinetuningScheduler` expands the space of possible
finetuning schedules and the composition of more sophisticated schedules can yield marginal finetuning performance
gains. That stated, it should be emphasized the primary utility of
:class:`~finetuning_scheduler.fts.FinetuningScheduler` is to grant greater finetuning
flexibility for model exploration in research. For example, glancing at DeBERTa-v3's implicit training run, a critical
tuning transition point is immediately apparent:


Our val_loss begins a precipitous decline at step 3119 which corresponds to phase 17 in the schedule. Referring to our
schedule, in phase 17 we're beginning tuning the attention parameters of our 10th encoder layer (of 11). Interesting!
Though beyond the scope of this documentation, it might be worth investigating these dynamics further and
:class:`~finetuning_scheduler.fts.FinetuningScheduler` allows one to do just that quite
easily.


Note that though this example is intended to capture a common usage scenario, substantial variation is expected among
use cases and models. In summary, :class:`~finetuning_scheduler.fts.FinetuningScheduler`
provides increased finetuning flexibility that can be useful in a variety of contexts from exploring model tuning
behavior to maximizing performance.

.. figure:: ../_static/images/fts/fts_explicit_loss_anim.gif
   :alt: FinetuningScheduler Explicit Loss Animation
   :width: 300

.. note:: The :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback is currently in beta.
