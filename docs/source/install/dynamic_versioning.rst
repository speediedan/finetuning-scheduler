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

For development and testing, FTS supports pinning to a specific Lightning commit via the ``UV_OVERRIDE``
environment variable. This is handled automatically by the build scripts and CI workflows.

**Using the build script (recommended):**

.. code-block:: bash

    # The build_fts_env.sh script automatically sets up Lightning commit pinning
    # and optionally installs PyTorch nightly (if configured in requirements/ci/torch-nightly.txt)
    ./scripts/build_fts_env.sh --repo_home=${PWD} --target_env_name=fts_latest

**Manual installation with Lightning commit pin:**

.. code-block:: bash

    git clone https://github.com/speediedan/finetuning-scheduler.git
    cd finetuning-scheduler

    # Set UV_OVERRIDE to use the pinned Lightning commit
    export UV_OVERRIDE=${PWD}/requirements/ci/overrides.txt
    uv pip install -e ".[all]"

**Manual installation with PyTorch nightly:**

When ``torch-nightly.txt`` is configured, use a two-step installation approach:

.. code-block:: bash

    git clone https://github.com/speediedan/finetuning-scheduler.git
    cd finetuning-scheduler

    # Step 1: Install PyTorch nightly (adjust version and CUDA target as needed)
    uv pip install --prerelease=if-necessary-or-explicit torch==2.10.0.dev20251124 \
        --index-url https://download.pytorch.org/whl/nightly/cu128

    # Step 2: Install FTS with Lightning commit pin (torch already installed, will be skipped)
    export UV_OVERRIDE=${PWD}/requirements/ci/overrides.txt
    uv pip install -e ".[all]"

The nightly version is specified in ``requirements/ci/torch-nightly.txt`` and documented in
``requirements/ci/torch_override.txt`` for reference. The specific Lightning commit is defined in
``requirements/ci/overrides.txt``.
