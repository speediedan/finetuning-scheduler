PyTorch API (experimental)
=================================

The modules below make up the PyTorch API (see :ref:`plain-pytorch-loop`). They are experimental in 2.15.
The scheduling core's remaining modules are shared implementation for the PyTorch API and the Lightning integration and
are not documented as public API.

.. currentmodule:: finetuning_scheduler

.. autosummary::
    :toctree: api
    :nosignatures:

    core.tuning_scheduler
    core.transitions
    core.interface
    core.checkpoint
    core.executor
    core.views
    core.roles
    core.compile
    strategies.base
    strategies.unsharded
    strategies.dtensor
    strategies.fsdp2
    strategies.plan
    testing.conformance
    testing.fixtures
    core.schedule
    core.state
    core.rng
    core.config
    core.fingerprint
    core.exceptions
    lightning.interface
    lightning.tuning_scheduler_bridge

The tunable view protocol and the address grammar schedules use are
`venngram <https://github.com/speediedan/venngram>`_'s, which Fine-Tuning Scheduler depends on and re-exports as
``finetuning_scheduler.tunable_view`` (its submodules ``views``, ``roles``, ``errors`` and ``conformance`` are
venngram's own modules).

.. currentmodule:: venngram

.. autosummary::
    :toctree: api
    :nosignatures:

    address
    book
    resolve
    physical
    roles
    errors
    tuning.views
    tuning.errors
    tuning.conformance
