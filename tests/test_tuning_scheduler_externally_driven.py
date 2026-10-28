# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An externally driven ``TuningScheduler`` (``attach``): the framework driving it owns the loop and phase
execution, the tuning scheduler sequences transitions (#85).

These need only PyTorch and the scheduling core (the ``core-no-lightning`` job runs them with ``--noconftest``).
"""
import pytest
import torch
from torch import nn

from finetuning_scheduler import TuningScheduler, TorchInterface
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.executor import TuningSchedulerExecutor, PhaseExecutor

SCHEDULE = {0: {"params": ["2.weight", "2.bias"]}, 1: {"params": ["1.weight", "1.bias"]},
            2: {"params": ["0.weight", "0.bias"]}}


class RecordingExecutor:
    def __init__(self, tuning_scheduler_ref):
        self.tuning_scheduler_ref, self.calls = tuning_scheduler_ref, []

    def configure_phase(self, depth, restored_depth, depth_sync, pre_state):
        self.calls.append(("configure_phase", depth, restored_depth, depth_sync, pre_state is not None))

    def resume_at(self, depth):
        self.calls.append(("resume_at", depth))

    def after_transition(self, depth):
        self.calls.append(("after_transition", depth))


class InterfaceRestoringPolicy:
    """A checkpoint policy whose interface restores the best checkpoint itself."""

    monitor = "loss"

    def __init__(self, best_depth, seen):
        self.best_depth_value, self.seen = best_depth, seen

    def restore(self):
        self.seen.append(("restore", self.tuning_scheduler.curr_depth))
        return self.best_depth_value


def _attached(checkpoint=None, max_depth=-1):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2))
    interface = TorchInterface(model)
    interface.set_optimizer(0, torch.optim.SGD(model[2].parameters(), lr=0.1))  # the interface built it
    fts = TuningScheduler(model, optimizer=torch.optim.SGD, interface=interface, checkpoint=checkpoint,
                          max_depth=max_depth)
    executor = RecordingExecutor(fts)
    fts.attach(executor, SCHEDULE)
    return fts, executor


def test_request_advance_sequences_the_transition_through_the_executor():
    fts, executor = _attached()
    events = []
    fts.add_phase_hook(lambda e: events.append(("end", e.depth)), when="end")
    fts.add_phase_hook(lambda e: events.append(("start", e.depth)))
    assert isinstance(executor, PhaseExecutor) and fts.curr_depth == 0
    fts.request_advance("the interface decided")
    assert fts.curr_depth == 1 and events == [("end", 0), ("start", 1)]
    assert executor.calls == [("configure_phase", 1, 0, False, True), ("after_transition", 1)]


def test_a_interface_restoring_policy_restores_before_the_end_hooks():
    """As for a self-driven tuning scheduler, the best state is restored before the ending phase's end hooks run,
    so they see it under either interface; the new phase is then configured from the restored depth."""
    seen = []
    policy = InterfaceRestoringPolicy(best_depth=0, seen=seen)
    fts, executor = _attached(checkpoint=policy)
    policy.tuning_scheduler = fts
    fts.add_phase_hook(lambda e: seen.append(("end", e.depth)), when="end")
    fts.request_advance()
    assert seen == [("restore", 0), ("end", 0)] and fts.restored_count == 1
    assert executor.calls[0] == ("configure_phase", 1, 0, True, True)


def test_resume_at_thaws_through_the_executor_without_configuring_the_optimizer():
    fts, executor = _attached()
    fts.resume_at(2)
    assert fts.curr_depth == 2 and executor.calls == [("resume_at", 2), ("after_transition", 2)]


def test_externally_driven_refusals():
    fts, _ = _attached(max_depth=1)
    for loop_method in (lambda: next(fts.epochs(2)), fts.epoch_start, fts.epoch_end, fts.step_end):
        with pytest.raises(FTSConfigurationError, match="externally driven tuning scheduler refuses it"):
            loop_method()
    for resume_method in (lambda: fts.save("x.pt"), lambda: fts.load("x.pt"), lambda: fts.load_state_dict({}),
                          fts.resume_state, lambda: fts.load_resume_state({})):
        with pytest.raises(FTSConfigurationError, match="driving it saves and restores the training state"):
            resume_method()
    with pytest.raises(FTSConfigurationError, match="Gates are not supported"):
        fts.add_gate(lambda e: True)
    for depth in (-1, 2):
        with pytest.raises(FTSConfigurationError, match=f"Cannot resume at phase {depth}"):
            fts.resume_at(depth)
    fts.request_advance()
    with pytest.raises(FTSConfigurationError, match="is the last phase"):
        fts.request_advance()
    with pytest.raises(FTSConfigurationError, match="already set up"):
        fts.attach(RecordingExecutor(fts), SCHEDULE)
    model = nn.Linear(2, 2)
    plain = TuningScheduler(model, {0: {"params": ["weight", "bias"]}}, optimizer=torch.optim.SGD)
    plain.setup()
    with pytest.raises(FTSConfigurationError, match="set it up with `attach\\(\\)`"):
        plain.request_advance()


def test_attach_refuses_registered_gates_and_policy_restore_is_externally_driven_only():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2))
    interface = TorchInterface(model)
    interface.set_optimizer(0, torch.optim.SGD(model[2].parameters(), lr=0.1))
    fts = TuningScheduler(model, optimizer=torch.optim.SGD, interface=interface)
    fts.add_gate(lambda e: True)
    with pytest.raises(FTSConfigurationError, match="Gates are not supported"):
        fts.attach(RecordingExecutor(fts), SCHEDULE)
    unrestorable = TuningScheduler(model, optimizer=torch.optim.SGD, interface=interface, restore_best=True)
    with pytest.raises(FTSConfigurationError, match="needs a `checkpoint` policy"):
        unrestorable.attach(RecordingExecutor(unrestorable), SCHEDULE)
    # a self-driven tuning scheduler ignores a policy's `restore` and restores its own checkpoint files
    seen = []
    policy = InterfaceRestoringPolicy(best_depth=0, seen=seen)
    policy.best_path = None
    driven = TuningScheduler(nn.Linear(2, 2), {0: {"params": ["weight", "bias"]}}, optimizer=torch.optim.SGD,
                             checkpoint=policy)
    assert driven._restore_best_state() is None and seen == []


def test_the_default_executor_resumes_by_thawing_each_phase():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2))
    schedule = {d: {**p, "max_transition_epoch": d + 1} for d, p in SCHEDULE.items()}
    fts = TuningScheduler(model, schedule, optimizer=torch.optim.SGD)
    fts.setup()
    TuningSchedulerExecutor(fts).resume_at(1)
    assert sorted(n for n, p in model.named_parameters() if p.requires_grad) == ["1.bias", "1.weight", "2.bias",
                                                                                  "2.weight"]
