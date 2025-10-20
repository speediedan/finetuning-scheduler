.. _governance:

Fine-Tuning Scheduler Governance
################################

This document describes governance processes we follow in developing the Fine-Tuning Scheduler.

Persons of Interest
*******************

.. _governance_bdfl:

BDFL
----
Role: All final decisions related to Fine-Tuning Scheduler.

- Dan Dale (`speediedan <https://github.com/speediedan>`_) (Fine-Tuning Scheduler author)

Releases
********

Release versioning and cadence are described in detail in the :ref:`versioning` document.

In summary, starting with version 2.9, Fine-Tuning Scheduler aligns with PyTorch minor releases rather than Lightning releases, providing greater flexibility for integrating the latest PyTorch functionality while maintaining robust Lightning compatibility.

Project Management and Decision Making
**************************************

TBD

API Evolution
*************

Fine-Tuning Scheduler's API evolution and deprecation policies are described in detail in the :ref:`versioning` document.

In brief:

- Backward-incompatible changes follow a deprecation process with warnings before taking effect (typically X+2 minor versions)
- APIs are classified as *experimental* (unstable, not for production) or *stable* (production-ready with priority support)

See :ref:`versioning` for complete details on API evolution, compatibility policies, and version numbering.
