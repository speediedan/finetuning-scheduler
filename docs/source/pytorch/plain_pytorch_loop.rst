.. _plain-pytorch-loop:

##################################################
Scheduled Fine-Tuning in a Plain PyTorch Loop
##################################################

.. warning::
   :class:`~finetuning_scheduler.core.controller.FTSController` is **experimental** in 2.15 (see the
   :ref:`API stability classifications<versioning>`): its API may still change. The Lightning callback
   (:class:`~finetuning_scheduler.fts.FinetuningScheduler`) is unaffected and remains fully supported.

Overview
********
:class:`~finetuning_scheduler.core.controller.FTSController` brings Fine-Tuning Scheduler's multi-phase schedules to a
training loop you write yourself, with no Lightning ``Trainer`` involved. It shares its scheduling logic with the
Lightning callback: the same schedule format, the same phase thawing and parameter-group growth, the same optimizer and
lr scheduler reinitialization and the same plateau criteria. A parity test suite runs scenarios on both and requires the
same phase path, trainable-parameter counts and learning rates.

The loop stays yours. The controller owns three things: which parameters are trainable, the optimizer's parameter
groups, and when the next phase starts. You call it at a few well-defined points:

- :meth:`~finetuning_scheduler.core.controller.FTSController.setup` once, before training;
- :meth:`~finetuning_scheduler.core.controller.FTSController.epochs` to iterate over epochs (or
  :meth:`~finetuning_scheduler.core.controller.FTSController.epoch_start` and
  :meth:`~finetuning_scheduler.core.controller.FTSController.epoch_end` if you drive epochs yourself);
- :meth:`~finetuning_scheduler.core.controller.FTSController.epoch_end` with the epoch's validation metrics;
- :meth:`~finetuning_scheduler.core.controller.FTSController.step_end` after each optimizer step, only when a phase
  should end at a step count rather than at an epoch boundary.

The PyTorch API does not import Lightning or jsonargparse.

A Minimal Loop
**************
The schedule below thaws the classifier head first, then the layers beneath it. Phase 0 runs for two epochs
(``max_transition_epoch``); later phases end when validation loss stops improving
(:class:`~finetuning_scheduler.core.transitions.PlateauTransition`). In the final phase, a plateau ends training.

.. code-block:: python

    from functools import partial

    import torch
    from torch import nn

    from finetuning_scheduler import FTSController, PlateauTransition
    from finetuning_scheduler.core.checkpoint import BestCheckpoint

    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

    schedule = {
        0: {"params": ["4.weight", "4.bias"], "max_transition_epoch": 2},
        1: {"params": ["2.weight", "2.bias"], "lr": 1e-3},
        2: {"params": ["0.weight", "0.bias"], "lr": 1e-4},
    }

    fts = FTSController(
        model,
        schedule,
        optimizer=partial(torch.optim.AdamW, lr=1e-2),
        transition=PlateauTransition(monitor="val_loss", patience=2, min_delta=0.01),
        checkpoint=BestCheckpoint("checkpoints", monitor="val_loss"),
    )
    fts.setup()
    for epoch in fts.epochs(max_epochs=30):
        for x, y in train_loader:
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            fts.optimizer.step()
            fts.optimizer.zero_grad()
        fts.epoch_end({"val_loss": evaluate(model)})

A few things to note:

- **Pass an optimizer instance or factory.** This example passes a factory
  (``partial(torch.optim.AdamW, lr=1e-2)``), which the controller calls with phase 0's parameter groups; it then adds
  groups as later phases thaw parameters. A factory is preferred, and phase 0's ``pg_overrides`` requires one. An
  optimizer instance also works: unless ``enforce_phase0_params=False``, it is reconfigured to optimize exactly phase
  0's parameters.
- **Use** ``fts.optimizer`` **in the loop**, not a reference taken before ``setup()``: a schedule can replace the
  optimizer at a phase transition (see :ref:`optimizer reinitialization<optim-reinit-overview>`), and the retired one
  refuses to step.
- **Phases advance at epoch boundaries.** A decision reported by ``epoch_end`` takes effect at the start of the next
  epoch, and at most one phase starts per epoch.
- **With a checkpoint policy, the best state is restored at each phase boundary** (``restore_best``, on by default
  when ``checkpoint`` is given), as the Lightning callback does: the next phase starts from the best model, optimizer
  and lr scheduler state of the phase that just ended.
- **The schedule** may be a mapping as above, a path to a schedule YAML file in the
  :ref:`usual format<motivation>`, or ``None``, which generates the default schedule (parameters thawed in pairs,
  starting from the last registered parameter).

Transitions
***********
A phase ends when any of these is due:

- the phase's ``max_transition_epoch`` (an epoch count) or ``max_transition_step`` (a global optimizer-step count,
  checked in ``step_end``);
- one of the ``transition`` policies decides so.

The policies are :class:`~finetuning_scheduler.core.transitions.PlateauTransition`,
:class:`~finetuning_scheduler.core.transitions.EpochTransition`,
:class:`~finetuning_scheduler.core.transitions.StepTransition`, and
:class:`~finetuning_scheduler.core.transitions.CallableTransition`, which wraps a function of the
:class:`~finetuning_scheduler.core.transitions.TransitionContext`. Pass several as a list, or compose them with
:class:`~finetuning_scheduler.core.transitions.AnyOf`. Each policy returns a
:class:`~finetuning_scheduler.core.transitions.Decision` (``CONTINUE``, ``ADVANCE``, ``STOP`` or ``VETO``). In a
distributed run, the decisions are agreed across ranks, and the greatest wins, so every rank advances together.

Without any policy, every phase but the final one must set ``max_transition_epoch`` or ``max_transition_step``.

Schedule Keys for the PyTorch API
*********************************
Three per-phase keys are interpreted only by the PyTorch API. The Lightning callback refuses them rather than ignoring
them:

- ``max_transition_step``: end the phase at a global optimizer-step count (call ``step_end()`` after each step);
- ``phase_config``: a mapping handed to phase hooks and gates for that phase (for example a per-phase loss weight).
  ``phase_config_type`` validates it (it is called with the mapping's items as keyword arguments, so a dataclass
  works);
- ``pg_overrides``: optimizer options applied to the parameter group the phase adds, for example
  ``{"weight_decay": 0.0}`` (``params`` and ``lr`` have their own keys).

Phase Hooks and Gates
*********************
:meth:`~finetuning_scheduler.core.controller.FTSController.add_phase_hook` runs a function when a phase starts or ends.
Hooks receive a :class:`~finetuning_scheduler.core.controller.PhaseEvent` and run with the training random streams
isolated, so a hook that samples data does not change the training run.

:meth:`~finetuning_scheduler.core.controller.FTSController.add_gate` registers a check that accepts or vetoes each
completed phase. A gate receives a :class:`~finetuning_scheduler.core.transitions.GateEvent` and returns a
:class:`~finetuning_scheduler.core.transitions.GateResult` (or a ``bool``). A veto on any rank stops training and
restores the best checkpoint as of the vetoed phase's start, when there is one (for phase 0, only with
``BestCheckpoint(..., save_initial=True)``):

.. code-block:: python

    from finetuning_scheduler import GateResult


    def no_regression(event):
        ok = event.best_score is not None and event.best_score.item() < 0.6
        return GateResult(accept=ok, reason=None if ok else f"phase {event.depth} best val_loss too high")


    fts.add_gate(no_regression)

Gate outcomes are kept in ``fts.gate_records`` and, after a veto, ``fts.stop_reason`` says why training stopped.

Checkpoints and Resuming
************************
:class:`~finetuning_scheduler.core.checkpoint.BestCheckpoint` keeps the ``save_top_k`` best checkpoints by a monitored
metric (one by default), optionally saving only the scheduled parameters (``include="scheduled"``), and pins the best at
each phase boundary as the rollback point for a gate's veto. Each checkpoint gets a new name and is never rewritten.
:meth:`~finetuning_scheduler.core.controller.FTSController.save` and
:meth:`~finetuning_scheduler.core.controller.FTSController.load` save and restore the full training state for an exact
resume: the model, optimizer and lr scheduler; the controller's own state; every random stream; and anything registered
with :meth:`~finetuning_scheduler.core.controller.FTSController.register_stateful` (a sampler or a data position, for
example):

.. code-block:: python

    fts.register_stateful("sampler", sampler)  # anything with state_dict() and load_state_dict()
    fts.save("run/last.pt")

    # later, in a fresh process, after building the same model, schedule and controller
    fts.setup()
    fts.load("run/last.pt")

A save keeps the checkpoints it refers to until the next save, so resuming from the latest save always finds them, even
when the run went on to write newer best checkpoints before it stopped. An older save can be resumed while its
checkpoints remain; if its best checkpoint or rollback point has been removed,
:meth:`~finetuning_scheduler.core.controller.FTSController.load` raises ``FileNotFoundError`` rather than restoring the
wrong state later.

Distributed Training
********************
Shard the model yourself with FSDP2 (``torch.distributed.fsdp.fully_shard``), tensor parallelism
(``torch.distributed.tensor.parallel.parallelize_module``) or both, before handing it to the controller, and call the
controller on every rank. Transition decisions and gate results are agreed across ranks. For sharded models,
checkpoints are written with ``torch.distributed.checkpoint``.

Apply ``fully_shard`` to the root module too, last, so FSDP2 manages every parameter: a parameter no FSDP module
manages has its gradients synchronized by nothing, so each data-parallel rank would train its own copy. The controller
warns about scheduled parameters left that way. The
:ref:`model-parallel example for the PyTorch API<model-parallel-plain-loop>` shards a Llama model this way, and also
runs it with tensor parallelism, mapping each part of the Lightning model-parallel example's configuration to its
plain-loop equivalent.

Two configurations are currently refused:

- ``DistributedDataParallel``, on the model or any submodule, and torch's composable ``replicate()``. DDP registers
  only the parameters that require grad when it is built (``replicate()`` builds it at the first forward, after the
  controller has frozen the model), so parameters thawed in later phases would never be gradient-synchronized. The
  Lightning callback, which owns its DDP wrapper, rebuilds it at each transition instead. With FSDP2, parameters
  thawed in later phases stay synchronized;
- ``ZeroRedundancyOptimizer``.

The Lightning callback's distributed support is unchanged; see
:doc:`model-parallel scheduled fine-tuning (FSDP2 and tensor parallelism)<../distributed/model_parallel_scheduled_fine_tuning>`
and :doc:`FSDP1 scheduled fine-tuning<../distributed/fsdp_scheduled_fine_tuning>`.

Relationship to the Lightning Callback
**************************************
The callback and the controller share one scheduling core (``finetuning_scheduler.core``), so features land in both
and the parity tests keep their behavior aligned. The differences are in what each host owns: under Lightning, the
``Trainer`` runs the loop, checkpoints through
:class:`~finetuning_scheduler.fts_supporters.FTSCheckpoint` and early-stops through
:class:`~finetuning_scheduler.fts_supporters.FTSEarlyStopping`; in a plain loop, you run the loop and the controller
provides the equivalents.

Lightning remains a base dependency of Fine-Tuning Scheduler until it becomes an optional dependency, no earlier than
2.17 (see :ref:`versioning`). Until then, a plain install keeps the Lightning integration working as before. Naming
``finetuning-scheduler[lightning]`` (or ``finetuning-scheduler[pytorch-lightning]`` for the standalone package) in a
Lightning-based project's requirements now means that change needs no further edit.
