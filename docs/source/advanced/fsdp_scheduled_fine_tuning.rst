##########################
FSDP Scheduled Fine-Tuning
##########################

Overview
********

:class:`~finetuning_scheduler.fts.FinetuningScheduler` (FTS) now supports flexible, multi-phase, scheduled fine-tuning
with the Fully Sharded Data Parallel (FSDP) strategy (
:external+pl:class:`~lightning.pytorch.strategies.fsdp.FSDPStrategy`). This tutorial
assumes a basic understanding of FSDP training, please see
`this PyTorch tutorial  <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`_ for a good introduction to
FSDP training.

As with standard FSDP usage, FSDP wrapping of a :external+pl:class:`~lightning.pytorch.core.module.LightningModule`
can be performed either by providing an ``auto_wrap_policy`` or (for maximal control) by overriding the
``configure_model`` method of :external+pl:class:`~lightning.pytorch.core.module.LightningModule` and
manually wrapping the module.

This tutorial walks through the configuration of an example multi-phase, scheduled FSDP fine-tuning training session and
largely uses the same code as the basic :ref:`scheduled fine-tuning for SuperGLUE<scheduled-fine-tuning-superglue>`
examples.

.. _fsdp-fine-tuning-example:

Example: Multi-Phase Scheduled Fine-Tuning with FSDP
****************************************************

Demonstration FTS FSDP training/profiling configurations and a DDP baseline for comparison are available under
``./fts_examples/stable/config/advanced/fsdp``.

Most of these FTS FSDP training examples have the same dependencies as the basic
:ref:`scheduled fine-tuning for SuperGLUE<scheduled-fine-tuning-superglue>` examples except PyTorch >= ``2.0`` is
required. Running the :ref:`basic example<basic-fsdp-fine-tuning-example>` requires PyTorch >= ``2.1.0``.

.. note::

    The examples below are not configured to execute a full training session but instead to generate the minimal
    meaningful profiling statistics for analysis and exposition (e.g. using only 2 batches, very limited epochs, etc.)

The demo schedule configurations are composed with the basic FTS example's shared defaults
(``./config/fts_defaults.yaml``) and can be executed as follows:

.. code-block:: bash

    cd ./fts_examples/stable

    # there is an open issue regarding superfluous profiler messages (still as of 2023.04.15)
    # setting the environmental variable below is a workaround to keep the example output clean:

    export TORCH_CPP_LOG_LEVEL=ERROR

    # Profiled demo of basic scheduled fine-tuning with FSDP (requires PyTorch >= 2.1.0)
    python fts_superglue.py fit --config config/advanced/fsdp/fts_fsdp_basic_profile.yaml

    # Profiled demo of FSDP scheduled fine-tuning using the ``awp_overrides`` option:
    python fts_superglue.py fit --config config/advanced/fsdp/fts_fsdp_awp_overrides_profile.yaml

    # Profiled demo of comparable DDP scheduled fine-tuning baseline:
    python fts_superglue.py fit --config config/advanced/fsdp/fts_ddp_fsdp_baseline_profile.yaml

    # Profiled demo of FSDP scheduled fine-tuning with CPU Offloading but full precision
    # (for reference, not reviewed in this tutorial)
    python fts_superglue.py fit --config config/advanced/fsdp/fts_fsdp_awp_overrides_offload_profile.yaml

.. _basic-fsdp-fine-tuning-example:

Basic Scheduled Fine-Tuning with FSDP
*************************************

Beginning with PyTorch version ``2.1.0``, the effective constraints FSDP imposed on fine-tuning schedules were substantially relaxed. As you'll see below,
scheduled fine-tuning with FSDP is pretty straightforward! All one need do:

1. Pass ``use_orig_params`` to the FSDP strategy configuration.
2. Provide a simple ``auto_wrap_policy`` configuration (not technically required but almost always desired).

For a given fine-tuning schedule:

.. code-block:: yaml
  :linenos:

  0:
    params:
    - model.classifier.*
    max_transition_epoch: 1
  1:
    params:
    - model.pooler.dense.*
    - model.deberta.encoder.layer.11.(output|attention|intermediate).*
    max_transition_epoch: 2
  2:
    params:
    - model.deberta.encoder.layer.([0-9]|10).(output|attention|intermediate).*
    - model.deberta.encoder.LayerNorm.bias
    - model.deberta.encoder.LayerNorm.weight
    - model.deberta.encoder.rel_embeddings.weight

We can just define an ``auto_wrap_policy`` for our DeBERTa-v3 module, directing FTS/FSDP to wrap the specified Transformer layers in separate FSDP modules:

.. code-block:: yaml
  :linenos:
  :emphasize-lines: 5-10

  strategy:
    class_path: lightning.pytorch.strategies.FSDPStrategy
    init_args:
      # other FSDP args as desired ...
      use_orig_params: True
      auto_wrap_policy:
        class_path: torch.distributed.fsdp.wrap.ModuleWrapPolicy
        init_args:
          module_classes: !!set
            ? transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Layer

That's it! Note that we set ``use_orig_params`` to ``True`` in line 5 as it allows for more flexible fine-tuning schedules with PyTorch >= ``2.1.0``.

In the next section, we'll cover some of the more advanced configuration options available for customizing scheduled fine-tuning with FSDP.

Advanced FSDP Wrapping For Scheduled Fine-Tuning
************************************************

There are a number of usage contexts that might motivate moving beyond the simple configuration above. For instance:

.. list-table:: Motivations for Advanced FSDP Wrapping
   :widths: 50 50
   :header-rows: 1

   * - Potential Use case
     - Relevant Features & Info
   * - Optimize resource utilization (whether memory, compute or network)
     - :ref:`activation checkpointing<activation-ckpt-and-cpu-offload>`, :ref:`cpu offload<activation-ckpt-and-cpu-offload>`, :attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides`
   * - More granular control over module wrapping policy w/o manually writing a "configure_model" method
     - :attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides`
   * - A desire to use FSDP in the default "use_orig_params=False" mode
     - `See PyTorch documentation for possible issues <https://pytorch.org/docs/master/fsdp.html?highlight=use_orig_params>`_
   * - if using a version of PyTorch < ``2.1.0``
     -

As with standard FSDP module wrapping, one can use an ``auto_wrap_policy`` to wrap a model for FSDP scheduled
fine-tuning. In the current FTS release, there is only one FTS-specific FSDP configuration enhancement to consider:
the :attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides` list.

:attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides` is an optional list of module names
that should be wrapped in separate FSDP instances, complementing the modules that would be individually wrapped by
``auto_wrap_policy`` provided in the
:external+pl:class:`~lightning.pytorch.strategies.fsdp.FSDPStrategy` strategy
configuration.

Starting with a defined ``auto_wrap_policy`` and providing module name-based complements/overrides as needed using
:attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides` is often the most expedient approach
to auto-wrapping models in alignment with a fine-tuning schedule.

We again start by defining a simple fine-tuning schedule that we would like to ensure our module wrapping supports:

.. code-block:: yaml
  :linenos:

  0:
    params:
    - model.classifier.*
    max_transition_epoch: 1
  1:
    params:
    - model.pooler.dense.*
    - model.deberta.encoder.layer.11.(output|attention|intermediate).*
    max_transition_epoch: 2
  2:
    params:
    - model.deberta.encoder.layer.([0-9]|10).(output|attention|intermediate).*
    - model.deberta.encoder.LayerNorm.bias
    - model.deberta.encoder.LayerNorm.weight
    - model.deberta.encoder.rel_embeddings.weight
    # excluding these parameters from the schedule to enhance the debugging demonstration
    #- model.deberta.embeddings.LayerNorm.bias
    #- model.deberta.embeddings.LayerNorm.weight
    #- model.deberta.embeddings.word_embeddings.weight

We define the ``auto_wrap_policy`` for our DeBERTa-v3 module as follows:

.. code-block:: yaml
  :linenos:
  :emphasize-lines: 5-11

  strategy:
    class_path: lightning.pytorch.strategies.FSDPStrategy
    init_args:
      # other FSDP args as desired ...
      auto_wrap_policy:
        class_path: torch.distributed.fsdp.wrap.ModuleWrapPolicy
        init_args:
          module_classes: !!set
            ? transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Layer
            ? transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Embeddings
            ? transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Encoder


We'll inspect the rationale for this policy below, but first, notice we have not referenced our ``classifier`` and
``pooler`` layers. Because we would like to thaw our ``classifier`` and ``pooler`` layers in separate phases from some
other layers, we need to separately wrap these layers as well. If we specified separate wrapping of all ``Linear``
layers however in our ``auto_wrap_policy``, we would end up unnecessarily (and in many cases problematically) separately
wrapping the many ``Linear`` layers within our currently FSDP wrapped modules (``DebertaV2Layer`` etc.).

To facilitate module wrapping in alignment with fine-tuning schedule phases, FTS provides the
:attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides` feature which allows users to provide
module name-based complements to a given ``auto_wrap_policy``.

In this case, simply listing the names of (or regex patterns matching) modules we would like to separately wrap allows
us to achieve FSDP wrapping that aligns with our fine-tuning schedule. FTS support for FSDP training is provided via a
:class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter`
(:class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter`). Configuration for FTS-extensions of strategies
like FSDP is passed to FTS via the
:attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.strategy_adapter_cfg` configuration dictionary.

So in our example, we can pass the :attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides`
configuration option to FTS like so:

.. code-block:: yaml
  :linenos:
  :emphasize-lines: 3, 7, 8

  # in ./fts_examples/stable/config/advanced/fsdp/fts_fsdp_awp_overrides_profile.yaml
  ...
    - class_path: finetuning_scheduler.FinetuningScheduler
    init_args:
      ft_schedule: ./config/RteBoolqModule_ft_schedule_deberta_base_fsdp.yaml
      max_depth: 2
      strategy_adapter_cfg:
        awp_overrides: ["model.pooler.dense", "model.classifier"]
  ...

.. _activation-ckpt-and-cpu-offload:

Finally, we configure the FSDP training strategy as desired per usual, for instance, specifying
``activation_checkpointing_policy`` and ``cpu_offload`` configurations in addition the ``auto_wrap_policy`` we defined above:

.. code-block:: yaml
  :linenos:
  :emphasize-lines: 6-8

  # in ./fts_examples/stable/config/advanced/fsdp/fts_fsdp_awp_overrides_profile.yaml
    ...
    strategy:
      class_path: lightning.pytorch.strategies.FSDPStrategy
      init_args:
        cpu_offload: false
        activation_checkpointing_policy: !!set
          ? transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Layer
        auto_wrap_policy:
          class_path: torch.distributed.fsdp.wrap.ModuleWrapPolicy
          init_args:
            module_classes: !!set
              ? transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Layer
              ? transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Embeddings
              ? transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Encoder

That's all there is to it! We've successfully defined our fine-tuning schedule and FSDP wrapped our model in a manner
that supports FSDP multi-phase scheduled fine-tuning.


Additional FSDP Wrapping and Debugging Guidance
***********************************************

In order to support multi-phase scheduled fine-tuning with FSDP in ``use_orig_params=False`` mode, FTS's key precondition
is that the defined fine-tuning schedule phases have disjoint sets of FSDP-flattened parameters (a ``FlatParameter`` is created when wrapping a set of
modules in a FSDP instance/unit). This constraint is derived from the fact that (for PyTorch < ``2.1.0`` or ``use_orig_params=False`` mode) the ``requires_grad`` attribute
must be the same for all parameters flattened into the same ``FlatParameter``. [#]_

FTS will attempt to validate that the module is wrapped in a manner that aligns with the defined fine-tuning
schedule phases prior to the start of training and provide detailed feedback for the user if a misalignment is
discovered.

For example, note that because we wanted to thaw some ``DebertaV2Layer`` s separately from others, we directed FSDP to
wrap ``DebertaV2Layer`` s in their own FSDP instances rather than just the entire ``DebertaV2Encoder``.

What happens if we just direct FSDP to wrap ``DebertaV2Layer`` s and not ``DebertaV2Encoder`` s and
``DebertaV2Embeddings`` as well?

FTS stops before beginning training and provides extensive context via this error message:

.. code-block:: bash

  "Fine-tuning schedule phases do not have disjoint FSDP-flattened parameter sets. Because the `requires_grad` attribute of FSDP-flattened parameters currently must be the same for all flattened parameters (for PyTorch < ``2.1.0`` or if in ``use_orig_params=False`` mode), fine-tuning schedules must avoid thawing parameters in the same FSDP-flattened parameter in different phases. Please ensure parameters associated with each phase are wrapped in separate phase-aligned FSDP instances.

  In this particular case, there are parameters not included in your fine-tuning schedule that span more than one fine-tuning phase. HINT: parameters associated with unwrapped modules will be included in the top-level (aka 'root') FSDP instance so ensuring all modules associated with fine-tuning scheduled parameters are wrapped separately from the top-level FSDP instance may avoid triggering this exception.

  The following logical parameters are associated with an FSDP-flattened parameter that spans more than one fine-tuning phase. The mapping of each logical parameter with the module name wrapped by its associated FSDP instance is provided below:

  {'model.deberta.embeddings.LayerNorm.bias': 'DebertaV2ForSequenceClassification',
   'model.deberta.embeddings.LayerNorm.weight': 'DebertaV2ForSequenceClassification',
   'model.deberta.embeddings.word_embeddings.weight': 'DebertaV2ForSequenceClassification',
   'model.deberta.encoder.LayerNorm.bias': 'DebertaV2ForSequenceClassification',
   'model.deberta.encoder.LayerNorm.weight': 'DebertaV2ForSequenceClassification',
   'model.deberta.encoder.rel_embeddings.weight': 'DebertaV2ForSequenceClassification'}"

This helps us understand that we have parameters that all belong to the same top-level FSDP instance (the instance
that wraps ``DebertaV2ForSequenceClassification``). By failing to specify separate wrapping of ``DebertaV2Encoder`` s,
parameters associated with that module fell to the top-level/root FSDP instance to be managed. While
``DebertaV2Embeddings`` parameters were not included in our schedule, they still must be wrapped by FSDP and so also are
included with ``DebertaV2Encoder`` parameters in the same top-level ``FlatParameter``. If training had been permitted
to proceed in this case, ``DebertaV2Embeddings`` parameters would have been thawed along with the ``DebertaV2Encoder``
parameters in phase ``2``, violating of our specified fine-tuning schedule.

To avoid violating the phase-wise disjointness constraint, we add ``DebertaV2Encoder`` to our ``auto_wrap_policy``.
While not technically required, we add ``DebertaV2Embeddings`` separately as well for future experimental flexibility.

As always, if needed, one can alternatively override ``configure_model`` and manually wrap a given
:external+pl:class:`~lightning.pytorch.core.module.LightningModule` to align with a desired fine-tuning schedule.

.. warning::

  :class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter` is in BETA and subject to change. The
  interface can bring breaking changes and new features with the next release of PyTorch.

.. note::

  The ``no_decay`` attribute that FTS supports on
  :external+pl:class:`~lightning.pytorch.core.module.LightningModule` with the base
  :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` is not currently supported in the context of
  FSDP fine-tuning.

.. note::

  Resuming across heterogeneous ``use_orig_params`` contexts with FTS is not currently supported (e.g.
  ``use_orig_params=True`` checkpoints need to be resumed with ``use_orig_params=True`` set)

.. note::

  With PyTorch versions < ``2.0``, optimizer state dicts are not currently saved/loaded when restoring checkpoints
  in the context of FSDP training. This comports with upstream Lightning behavior/limitations. Please use PyTorch >=
  ``2.0`` if restoring optimizer state from checkpoints (while FSDP training) is critical to your use case. For more
  regarding this version constraint, see `this issue <https://github.com/Lightning-AI/lightning/issues/18230>`_.

.. tip::

  If FSDP training with PyTorch >= ``2.1.0`` and ``use_orig_params=True``, ``DEBUG`` level logging will provide
  parameter shard allocation diagnostic info where relevant.

.. tip::

  If you want to extend FTS to use a custom, currently unsupported strategy or override current FTS behavior with a
  given training strategy, subclassing :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` is a way to do
  so.

Footnotes
*********

.. [#] As of PyTorch ``2.1.0``, ``FlatParameter`` s constructed in ``use_orig_params`` mode are allowed to contain
  original params with non-uniform ``requires_grad``.
