##########################
FSDP Scheduled Fine-Tuning
##########################

Overview
********

:class:`~finetuning_scheduler.fts.FinetuningScheduler` (FTS) now supports flexible, multi-phase, scheduled fine-tuning
with the Fully Sharded Data Parallel (FSDP) strategy (
:external+pl:class:`~pytorch_lightning.strategies.fully_sharded_native.DDPFullyShardedNativeStrategy`). This tutorial
assumes a basic understanding of FSDP training, please see
`this PyTorch tutorial  <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`_ for a good introduction to
FSDP training.

As with standard FSDP usage, FSDP wrapping of a :external+pl:class:`~pytorch_lightning.core.module.LightningModule`
can be performed either by providing an ``auto_wrap_policy`` or (for maximal control) by overriding the
``configure_sharded_model`` method of :external+pl:class:`~pytorch_lightning.core.module.LightningModule` and
manually wrapping the module.

This tutorial walks through the configuration of an example multi-phase, scheduled FSDP fine-tuning training session and
largely uses the same code as the basic :ref:`scheduled fine-tuning for SuperGLUE<scheduled-fine-tuning-superglue>`
examples.

.. _fsdp-fine-tuning-example:

Example: Multi-Phase Scheduled Fine-Tuning with FSDP
****************************************************

Demonstration FTS FSDP training/profiling configurations and a DDP baseline for comparison are available under
``./fts_examples/config/advanced/fsdp``.

This FTS FSDP training example has the same dependencies as the basic
:ref:`scheduled fine-tuning for SuperGLUE<scheduled-fine-tuning-superglue>` examples except PyTorch >= ``1.13`` is
required.

.. note::

    This version of :class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter` supports stable PyTorch
    releases >= 1.13. Support for PyTorch 2.0 is expected upon its release.

.. note::

    The examples below are not configured to execute a full training session but instead to generate the minimal
    meaningful profiling statistics for analysis and exposition (e.g. using only 2 batches, very limited epochs, etc.)

The demo schedule configurations are composed with the basic FTS example's shared defaults
(``./config/fts_defaults.yaml``) and can be executed as follows:

.. code-block:: bash

    cd ./fts_examples/

    # Profiled demo of FSDP scheduled fine-tuning using the ``awp_overrides`` option:
    python fts_superglue.py fit --config config/advanced/fsdp/fts_fsdp_awp_overrides_profile.yaml

    # Profiled demo of comparable DDP scheduled fine-tuning baseline:
    python fts_superglue.py fit --config config/advanced/fsdp/fts_ddp_fsdp_baseline_profile.yaml

    # Profiled demo of FSDP scheduled fine-tuning with CPU Offloading but full precision
    # (for reference, not reviewed in this tutorial)
    python fts_superglue.py fit --config config/advanced/fsdp/fts_fsdp_awp_overrides_offload_profile.yaml

FSDP Wrapping For Scheduled Fine-Tuning
***************************************

As with standard FSDP module wrapping, one can use an ``auto_wrap_policy`` to wrap a model for FSDP scheduled
fine-tuning. In the current FTS release, there is only one FTS-specific FSDP configuration enhancement to consider:
the :attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides` list.

:attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides` is an optional list of module names
that should be wrapped in separate FSDP instances, complementing the modules that would be individually wrapped by
``auto_wrap_policy`` provided in the
:external+pl:class:`~pytorch_lightning.strategies.fully_sharded_native.DDPFullyShardedNativeStrategy` strategy
configuration.

Starting with a provided ``auto_wrap_policy`` (e.g. in this example, ``transformer_auto_wrap_policy``) and providing
module name-based complements/overrides as needed using
:attr:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter.awp_overrides` is often the most expedient approach
to auto-wrapping models in alignment with a fine-tuning schedule.

We start by defining a simple fine-tuning schedule that we would like to ensure our module wrapping supports:

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

In this example (policy defined in ``./fts_examples/fts_fsdp_superglue.py``), we modify the base
``transformer_auto_wrap_policy`` to define the ``auto_wrap_policy`` for our DeBERTa-v3 module:

.. code-block:: python
  :linenos:
  :emphasize-lines: 2, 14

    # we use a non-partial formulation here for expository benefit
    deberta_transformer_layer_cls = {DebertaV2Layer, DebertaV2Embeddings, DebertaV2Encoder}


    def deberta_awp(
        module: torch.nn.Module,
        recurse: bool,
        unwrapped_params: int,
        transformer_layer_cls: Set[Type[torch.nn.Module]] = deberta_transformer_layer_cls,
    ) -> bool:
        if recurse:
            # always recurse
            return True
        else:
            # if not recursing, decide whether we should wrap for the leaf node or remainder
            return isinstance(module, tuple(transformer_layer_cls))


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

  # in ./fts_examples/config/advanced/fsdp/fts_fsdp_awp_overrides_profile.yaml
  ...
    - class_path: finetuning_scheduler.FinetuningScheduler
    init_args:
      ft_schedule: ./config/RteBoolqModule_ft_schedule_deberta_base_fsdp.yaml
      max_depth: 2
      strategy_adapter_cfg:
        awp_overrides: ["model.pooler.dense", "model.classifier"]
  ...

Finally, we configure the FSDP training strategy as desired per usual, for instance, specifying
``activation_checkpointing`` and ``cpu_offload`` configurations in addition the ``auto_wrap_policy`` we defined:

.. code-block:: yaml
  :linenos:
  :emphasize-lines: 6-9

  # in ./fts_examples/config/advanced/fsdp/fts_fsdp_awp_overrides_profile.yaml
    ...
    strategy:
      class_path: pytorch_lightning.strategies.DDPFullyShardedNativeStrategy
      init_args:
        cpu_offload: false
        activation_checkpointing:
        - transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Layer
        auto_wrap_policy: fts_examples.fts_fsdp_superglue.deberta_awp

That's all there is to it! We've successfully defined our fine-tuning schedule and FSDP wrapped our model in a manner
that supports FSDP multi-phase scheduled fine-tuning.


Additional FSDP Wrapping and Debugging Guidance
***********************************************

In order to support multi-phase scheduled fine-tuning with FSDP, FTS's key precondition is that the defined
fine-tuning schedule phases have disjoint sets of FSDP-flattened parameters (a ``FlatParameter`` is created when
wrapping a set of modules in a FSDP instance/unit). This constraint is derived from the fact that the
``requires_grad`` attribute currently must be the same for all parameters flattened into the same ``FlatParameter``.

FTS will attempt to validate that the module is wrapped in a manner that aligns with the defined fine-tuning
schedule phases prior to the start of training and provide detailed feedback for the user if a misalignment is
discovered.

For example, note that because we wanted to thaw some ``DebertaV2Layer`` s separately from others, we directed FSDP to
wrap ``DebertaV2Layer`` s in their own FSDP instances rather than just the entire ``DebertaV2Encoder``.

What happens if we just direct FSDP to wrap ``DebertaV2Layer`` s and not ``DebertaV2Encoder`` s and
``DebertaV2Embeddings`` as well?

FTS stops before beginning training and provides extensive context via this error message:

.. code-block:: bash

  "Fine-tuning schedule phases do not have disjoint FSDP-flattened parameter sets. Because the `requires_grad` attribute of FSDP-flattened parameters currently must be the same for all flattened parameters, fine-tuning schedules must avoid thawing parameters in the same FSDP-flattened parameter in different phases. Please ensure parameters associated with each phase are wrapped in separate phase-aligned FSDP instances.

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

As always, if needed, one can alternatively override ``configure_sharded_model`` and manually wrap a given
:external+pl:class:`~pytorch_lightning.core.module.LightningModule` to align with a desired fine-tuning schedule.

.. warning::

    :class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter` is in BETA and subject to change. The
    interface can bring breaking changes and new features with the next release of PyTorch.

.. note::

    The ``no_decay`` attribute that FTS supports on
    :external+pl:class:`~pytorch_lightning.core.module.LightningModule` with the base
    :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` is not currently supported in the context of
    FSDP fine-tuning.

.. tip::

  If you want to extend FTS to use a custom, currently unsupported strategy or override current FTS behavior with a
  given training strategy, subclassing :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` is a way to do
  so.
