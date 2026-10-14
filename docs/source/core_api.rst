PyTorch API (experimental)
=================================

The modules below make up the PyTorch API (see :ref:`plain-pytorch-loop`). They are experimental in 2.15.
The scheduling core's remaining modules are shared implementation for the PyTorch API and the Lightning integration and
are not documented as public API.

.. currentmodule:: finetuning_scheduler

.. autosummary::
    :toctree: api
    :nosignatures:

    core.controller
    core.transitions
    core.host
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
    lightning.host
    lightning.controller_bridge
