.. _dynamic_versioning:

Dynamic Versioning
==================

Overview
--------

The Dynamic Versioning system in Fine-Tuning Scheduler (FTS) enables constrained self-modification of
the ``finetuning-scheduler`` package itself. The initial application of this feature is to dynamically orchestrate
management of Lightning package imports, making it easy to switch between the unified ``lightning`` package and the
standalone ``pytorch-lightning`` package.

This feature is particularly useful for:

- Adapting existing code to work with different Lightning package import styles
- Testing compatibility with both unified and standalone Lightning
- Migrating projects from one import style to another

Toggling Lightning Import Styles
---------------------------------

After installing FTS, you can switch between unified and standalone imports using the ``toggle-lightning-mode``
command-line tool:

.. code-block:: bash

    # Switch to standalone imports (pytorch_lightning)
    toggle-lightning-mode --mode standalone

    # Switch back to unified imports (lightning.pytorch)
    toggle-lightning-mode --mode unified

The tool performs the following actions:

1. Checks if the requested package is installed
2. Scans all Python files in the FTS package
3. Updates import statements to match the requested style
4. Preserves functionality while changing only the import paths

.. note::
    You must have the target package installed before toggling. For example, to toggle to unified mode,
    the ``lightning`` package must be installed, and to toggle to standalone mode, the ``pytorch-lightning``
    package must be installed.

Implementation Details
----------------------

The dynamic versioning system:

- Uses pattern matching to identify Lightning imports
- Excludes certain files from modification to prevent self-modification
- Supports both source and installed package directories
- Automatically handles all import variations (direct imports, from imports, etc.)

The conversion operations are individually idempotent and mutually reversible, making it safe to run the toggle
command multiple times.

Using Lightning CI Commit Pinning
----------------------------------

For development and testing, FTS supports pinning to a specific Lightning commit via the ``USE_CI_COMMIT_PIN``
environment variable. This approach is similar to PyTorch's approach with Triton:

.. code-block:: bash

    # Enable Lightning CI commit pinning during installation
    export USE_CI_COMMIT_PIN="1"

    # Install FTS with pinned Lightning commit
    pip install finetuning-scheduler

    # Or for development installation
    git clone https://github.com/speediedan/finetuning-scheduler.git
    cd finetuning-scheduler
    python -m pip install -e ".[all]"

The specific Lightning commit is defined in ``requirements/ci/overrides.txt`` and is used by the CI system to
ensure consistent testing.
