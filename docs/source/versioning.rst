.. _versioning:

Fine-Tuning Scheduler Versioning
#################################

This document describes the versioning policy for Fine-Tuning Scheduler (FTS) and its relationship to upstream dependencies, particularly PyTorch and PyTorch Lightning.

Release Versioning
******************

Starting with the **2.9 minor release**, Fine-Tuning Scheduler is pivoting from tight Lightning version alignment to **core PyTorch version alignment**. This strategic change enables FTS to more flexibly integrate with the latest core PyTorch functionality that is increasingly important in research and reduces maintenance burden while continuing to support a stable API and robust Lightning integration.

Motivation for PyTorch Alignment
=================================

This versioning policy change is driven by:

1. **Evolving Lightning Release Cadence**: The Lightning project's release cadence has evolved, with minor releases increasingly driven by key upstream PyTorch deprecations and features rather than synchronized with PyTorch minor releases. See `Lightning Issue #21073 <https://github.com/Lightning-AI/pytorch-lightning/issues/21073>`_ and the associated `versioning documentation update PR #21107 <https://github.com/Lightning-AI/pytorch-lightning/pull/21107>`_ for details.

2. **PyTorch API Evolution**: There are substantial composable distributed API changes across recent PyTorch versions (e.g., 2.4 and 2.5) that require version-specific code paths. Aligning with PyTorch allows FTS to adopt new capabilities more rapidly while reducing technical debt from maintaining compatibility layers.

3. **Research Flexibility**: Aligning with PyTorch enables faster integration of cutting-edge research capabilities and best practices, which is particularly valuable for foundation model experimentation.

4. **Reduced Maintenance Burden**: Supporting a defined range of PyTorch versions provides clearer deprecation and breaking change opportunities.

PyTorch Compatibility Policy
=============================

Fine-Tuning Scheduler strives to officially support **at least the latest 4 PyTorch minor releases**. This provides a balance between API stability and the ability to leverage new PyTorch features while reducing maintenance overhead.

For example, when PyTorch 2.9 was released, FTS was guaranteed to support PyTorch >= 2.6 at a minimum. In practice, FTS may support additional earlier PyTorch versions (e.g. FTS 2.9 supports PyTorch 2.5), but support for the most recent 4 versions is the target commitment.

.. note::
   This is a target policy rather than a strict requirement, providing the project latitude to support additional versions when feasible.

Lightning Compatibility Policy
===============================

While FTS is pivoting to PyTorch version alignment, it continues to maintain compatibility with Lightning. This policy change is expected to be transparent to, or a net benefit for, the vast majority of Lightning ecosystem users.
FTS will continue to:

- Support the stable Lightning API
- Provide clear documentation of compatible Lightning versions
- Maintain backwards compatibility within major version series (2.x)

Compatibility Matrix
====================

The following table shows the compatibility between Fine-Tuning Scheduler, PyTorch, and Lightning versions:

.. list-table::
   :widths: 20 20 20 20
   :header-rows: 1

   * - FTS Version
     - Min PyTorch
     - Max PyTorch (tested)
     - Compatible Lightning (min)
   * - 2.10.x
     - 2.6.0
     - 2.10.0
     - >= 2.5.0
   * - 2.9.x
     - 2.5.0
     - 2.9.0
     - >= 2.5.0
   * - 2.5.x
     - 2.2.0
     - 2.8.0
     - >= 2.5.0
   * - 2.4.x
     - 2.1.0
     - 2.4.0
     - >= 2.4.0
   * - 2.3.x
     - 2.0.0
     - 2.3.0
     - >= 2.3.0
   * - 2.2.x
     - 1.13.0
     - 2.2.0
     - >= 2.2.0
   * - 2.1.x
     - 1.12.0
     - 2.1.0
     - >= 2.1.0
   * - 2.0.x
     - 1.11.0
     - 2.0.0
     - >= 2.0.0

.. note::
   Prior to version 2.9, FTS minor releases were paired with Lightning minor releases (e.g., FTS 2.0 → Lightning 2.0). Starting with 2.9, FTS versioning is decoupled from Lightning and aligned with PyTorch instead.

Version Numbering
*****************

Fine-Tuning Scheduler follows `Semantic Versioning <https://semver.org/>`_:

- **Major version** (e.g., 2.x.x → 3.x.x): Breaking API changes
- **Minor version** (e.g., 2.9.x → 2.10.x): New features, aligned with PyTorch minor releases when significant features are added
- **Patch version** (e.g., 2.9.0 → 2.9.1): Bug fixes and minor improvements

API Evolution Policy
********************

For API removal, renaming, or other forms of backward-incompatible changes, FTS follows this deprecation process:

1. A deprecation process is initiated at version X, producing warning messages at runtime and in the documentation.
2. Calls to the deprecated API remain unchanged in their function during the deprecation phase.
3. Two minor versions in the future at version X+2, the breaking change takes effect.

The "X+2" rule is a recommendation and not a strict requirement. Longer deprecation cycles may apply for some cases.

Breaking Changes
================

Since version 2.0.0, any breaking change (within supported PyTorch versions) will occur at ``(MAJOR+1).0.0`` (e.g., 3.0.0), unless it is fixing a critical bug. This provides strong API stability guarantees within major version series.

API Stability Classifications
==============================

New API and features are declared as:

- **Experimental**: Anything labeled as *experimental* or *beta* in the documentation is considered unstable and should not be used in production. The community is encouraged to test the feature and report issues directly on GitHub.

- **Stable**: Everything not specifically labeled as experimental should be considered stable. Reported issues will be treated with priority.

Testing and Continuous Integration
***********************************

Fine-Tuning Scheduler is rigorously tested across multiple CPUs, GPUs, and against the supported range of PyTorch and Python versions.

The project maintains:

- **Continuous Integration**: Testing on Linux, macOS, and Windows across supported Python versions
- **GPU Testing**: Multi-GPU testing on consumer hardware (RTX 4090, RTX 2070)
- **Distributed Training Testing**: FSDP and DDP testing with various PyTorch versions
- **Standalone Test Suite**: Comprehensive tests for distributed training scenarios

See the `README <https://github.com/speediedan/finetuning-scheduler>`_ for current build status details.

Upstream Dependency Management
*******************************

PyTorch Version Management
==========================

FTS testing includes:

- **Minimum supported PyTorch version**: Tested in CI to ensure compatibility
- **Latest PyTorch stable**: Tested to ensure forward compatibility
- **PyTorch nightly**: Periodically updated and monitored for upcoming changes (not guaranteed to be stable)

Lightning Version Management
============================

FTS maintains compatibility with Lightning by:

- Testing against the Lightning development branch to catch breaking changes early
- Supporting Lightning's stable API surface
- Providing clear documentation when Lightning-specific features are required

For development and CI, FTS may pin Lightning to a specific commit (similar to PyTorch's approach with Triton). This ensures reproducible builds while maintaining compatibility with released Lightning versions.

Release Cadence
***************

Starting with version 2.9:

- **Minor releases** are typically aligned with significant PyTorch minor releases that introduce features relevant to fine-tuning workflows
- **Patch releases** occur as needed for bug fixes and minor improvements
- The release cadence prioritizes stability and thorough testing over rapid iteration

This approach balances the need for timely access to PyTorch features with the stability requirements of production users.


See Also
********

- `Lightning Versioning Policy <https://lightning.ai/docs/pytorch/stable/versioning.html>`_
- `PyTorch Release Process <https://github.com/pytorch/pytorch/blob/main/RELEASE.md>`_
- :ref:`Fine-Tuning Scheduler Governance <governance>`
- :ref:`Contributing Guidelines <contributing>`
