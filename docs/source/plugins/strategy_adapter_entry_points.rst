.. _strategy_adapter_entry_points:

Strategy Adapter Entry Points
##############################

.. warning::
   This is an :ref:`experimental <versioning:API Stability Classifications>` feature which is
   still in development. The entry point API and plugin discovery mechanism may change in future
   releases.

Overview
********

Fine-Tuning Scheduler (FTS) supports custom strategy adapters via Python entry points, enabling third-party
packages to extend FTS with specialized adapters for custom training strategies, model architectures,
or parameter naming conventions (e.g. dynamic parameter views for latent space analysis).

This plugin mechanism allows packages like `Interpretune <https://github.com/speediedan/interpretune>`_
to provide adapters that integrate seamlessly with FTS without requiring modifications to the FTS
codebase itself.

**Important Concepts:**

- **Lightning Strategy Flags**: Built-in PyTorch Lightning strategy identifiers (e.g., ``single_device``, ``ddp``, ``fsdp``)
- **Strategy Adapters**: Classes that extend FTS functionality for specific strategies or use cases
- **Custom Strategy Adapter Mapping**: User-provided dictionary mapping Lightning strategy flags to adapter implementations

Note that custom strategy adapters are meant to **adapt existing Lightning strategies**, not create
wholly new ones. If you need a new strategy, register it with PyTorch Lightning first, then create
an adapter to extend FTS support for it.

Entry Point Specification
**************************

Entry Point Group
=================

Custom strategy adapters are registered under the ``finetuning_scheduler.strategy_adapters`` entry
point group.

Registration Format
===================

In your package's ``pyproject.toml``, register your adapter:

.. code-block:: toml

    [project.entry-points."finetuning_scheduler.strategy_adapters"]
    adapter_name = "package.module:AdapterClass"

The entry point name (``adapter_name``) will be lowercased and used to reference the adapter.
The value should follow Python's standard entry point format: ``module:attribute`` or
``module.submodule:attribute``.

Discovery and Loading
*********************

Entry points are discovered lazily during strategy setup (at the start of training). The discovery process:

1. Scans for all registered entry points in the ``finetuning_scheduler.strategy_adapters`` group
2. Attempts to load each adapter class
3. Adds successfully loaded adapters to the runtime ``STRATEGY_ADAPTERS`` mapping, keyed by the
   lowercased entry point name
4. Logs warnings for any adapters that fail to load (without preventing FTS initialization)

Usage Example
*************

Real-World Example: TransformerBridge Adapter
==============================================

The Interpretune package provides a ``TransformerBridgeStrategyAdapter`` that enables clean
TransformerLens-style parameter naming in FTS schedules. Here's how it's registered and used:

**Registration** (in Interpretune's ``pyproject.toml``):

.. code-block:: toml

    [project.entry-points."finetuning_scheduler.strategy_adapters"]
    transformerbridge = "interpretune.adapters.transformer_lens:TransformerBridgeStrategyAdapter"

**Usage** (in training configuration):

.. code-block:: python

    from finetuning_scheduler import FinetuningScheduler

    # Map Lightning strategy flags to the adapter
    # Multiple strategy flags can use the same adapter
    fts = FinetuningScheduler(
        custom_strategy_adapters={
            "single_device": "transformerbridge",  # Use entry point name
            "auto": "transformerbridge",  # Same adapter for auto strategy
            # Or use fully qualified paths:
            # "single_device": "interpretune.adapters.transformer_lens:TransformerBridgeStrategyAdapter",
            # "auto": "interpretune.adapters.transformer_lens.TransformerBridgeStrategyAdapter",
        },
        strategy_adapter_cfg={"use_tl_names": True},  # Adapter-specific config
    )

This allows fine-tuning schedules to use architecture-agnostic parameter names like ``blocks.9.attn.W_Q`` instead
of verbose and architecture-dependent canonical names, while FTS handles the necessary translations automatically.

**Key Points:**

- Strategy flags (``single_device``, ``auto``, etc.) refer to Lightning's built-in strategies
- The same adapter can be mapped to multiple strategy flags
- Three formats are supported for referencing adapters:

  1. Entry point name: ``\"transformerbridge\"``
  2. Colon-separated path: ``\"interpretune.adapters.transformer_lens:TransformerBridgeStrategyAdapter\"``
  3. Dot-separated path: ``\"interpretune.adapters.transformer_lens.TransformerBridgeStrategyAdapter\"``

Creating Custom Adapters
*************************

Base Requirements
=================

Custom strategy adapters should:

1. Inherit from :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter`
2. Implement required methods for your specific use case (see :doc:`/api/finetuning_scheduler.strategy_adapters`)
3. Follow the adapter lifecycle hooks (``:meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.connect``,
   ``:meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.on_before_init_fts``, etc.)

Override Points
===============

Strategy adapters can customize FTS behavior at multiple levels of abstraction accommodating a variety of use cases:

**Parameter Naming**
  Override :meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.get_named_params_for_schedule_validation`
  to provide custom parameter names while using default validation logic.

**Full Validation**
  Override :meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.validate_ft_sched`
  to completely customize schedule validation.

**Schedule Generation**
  Override :meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.gen_ft_schedule`
  to customize how default schedules are generated.

**Checkpoint Handling**
  Override ``:meth:`~finetuning_scheduler.strategy_adapters.StrategyAdapter.before_restore_model()``,
  ``lightning_module_state_dict()``, and ``load_model_state_dict()`` for custom checkpoint translation logic.

See :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter` for the complete API.

Best Practices
**************

Robust Loading
==============

Entry point loading is wrapped in exception handling to prevent adapter failures from breaking FTS
initialization. However, adapters should:

- Validate dependencies and raise clear errors during ``__init__()`` if requirements aren't met
- Use meaningful exception messages to help users diagnose configuration issues
- Document any required dependencies in your package documentation

Naming Conventions
==================

- Use descriptive, lowercase entry point names (e.g., ``transformerbridge``, ``custom_fsdp``)
- Avoid generic names that might conflict with other packages
- Consider prefixing with your package name for uniqueness (e.g., ``mypackage_adapter``)

Configuration
=============

Custom Adapter Mapping Format
------------------------------

The :paramref:`~finetuning_scheduler.fts.FinetuningScheduler.custom_strategy_adapters` parameter
accepts a dictionary mapping PyTorch Lightning strategy flags (canonical strategy names like
``"single_device"``, ``"auto"``, ``"ddp"``, etc.) to adapter references. The adapter reference
can be:

1. **An entry point name** (lowercased) registered under ``finetuning_scheduler.strategy_adapters``
2. **A fully qualified class path** in the format ``"module.path:ClassName"`` or
   ``"module.path.ClassName"``

This allows multiple strategy flags to be associated with the same adapter. For example:

.. code-block:: python

    from finetuning_scheduler import FinetuningScheduler

    # Multiple strategies can use the same registered plugin adapter
    fts = FinetuningScheduler(
        custom_strategy_adapters={
            "single_device": "transformerbridge",  # Plugin entry point name
            # We can use the same plugin for multiple strategies, here we use a fully qualified path format as well
            "auto": "interpretune.adapters.transformer_lens.TransformerBridgeStrategyAdapter",
        },
        strategy_adapter_cfg={
            "use_tl_names": True,  # Configuration passed to the adapter
        },
    )

**Native FTS Adapters**: FTS includes built-in adapters in the ``STRATEGY_ADAPTERS`` mapping
that are always available:

- ``"fsdp"`` - :class:`~finetuning_scheduler.strategy_adapters.FSDPStrategyAdapter`
- ``"modelparallelstrategy"`` - :class:`~finetuning_scheduler.strategy_adapters.ModelParallelStrategyAdapter`

These can be referenced directly without requiring plugin registration.

Adapter-Specific Configuration
-------------------------------

If your adapter accepts configuration, use the ``strategy_adapter_cfg`` parameter:

.. code-block:: python

    fts = FinetuningScheduler(
        custom_strategy_adapters={"target_strategy": "my_adapter"},
        strategy_adapter_cfg={
            "option1": value1,
            "option2": value2,
        },
    )

Testing
=======

Test your adapter with FTS by:

1. Creating test fixtures that instantiate FTS with your adapter
2. Verifying schedule validation works with your parameter naming
3. Testing checkpoint save/restore if you override those methods
4. Ensuring your adapter works with both explicit and implicit schedules

Future Directions
*****************

This plugin system may be extended in future releases to support:

- Versioned adapter APIs
- Additional extension points beyond strategy adapters

Community adapters and feedback on the plugin system are welcome! Please share your use cases
and suggestions on the `GitHub repository <https://github.com/speediedan/finetuning-scheduler/issues>`_.

See Also
********

- :doc:`/api/finetuning_scheduler.strategy_adapters`
- :class:`~finetuning_scheduler.strategy_adapters.StrategyAdapter`
- :ref:`versioning:API Stability Classifications`
