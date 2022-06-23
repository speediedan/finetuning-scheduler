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

Release cadence TBD

Project Management and Decision Making
**************************************

TBD

API Evolution
*************

For API removal, renaming or other forms of backward-incompatible changes, the procedure is:

#. A deprecation process is initiated at version X, producing warning messages at runtime and in the documentation.
#. Calls to the deprecated API remain unchanged in their function during the deprecation phase.
#. Two minor versions in the future at version X+2 the breaking change takes effect.

The "X+2" rule is a recommendation and not a strict requirement. Longer deprecation cycles may apply for some cases.

New API and features are declared as:

- *Experimental*: Anything labelled as *experimental* or *beta* in the documentation is considered unstable and should
    not be used in production. The community is encouraged to test the feature and report issues directly on GitHub.
- *Stable*: Everything not specifically labelled as experimental should be considered stable. Reported issues will be
    treated with priority.
