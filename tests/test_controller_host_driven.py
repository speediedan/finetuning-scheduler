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
"""A host-driven ``FTSController`` (``attach``): the host owns the loop and phase execution, the controller
sequences transitions (#85).

These need only PyTorch and the scheduling core (the ``core-no-lightning`` job runs them with ``--noconftest``).
"""
import pytest
import torch
from torch import nn

from finetuning_scheduler import FTSController, TorchHost
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.executor import ControllerExecutor, PhaseExecutor

SCHEDULE = {0: {"params": ["2.weight", "2.bias"]}, 1: {"params": ["1.weight", "1.bias"]},
            2: {"params": ["0.weight", "0.bias"]}}


class RecordingExecutor:
    def __init__(self, controller_ref):
        self.controller_ref, self.calls = controller_ref, []

    def configure_phase(self, depth, restored_depth, depth_sync, pre_state):
        self.calls.append(("configure_phase", depth, restored_depth, depth_sync, pre_state is not None))

    def resume_at(self, depth):
        self.calls.append(("resume_at", depth))

    def after_transition(self, depth):
        self.calls.append(("after_transition", depth))


class HostRestoringPolicy:
    """A checkpoint policy whose host restores the best checkpoint itself."""

    monitor = "loss"

    def __init__(self, best_depth, seen):
        self.best_depth_value, self.seen = best_depth, seen

    def restore(self):
        self.seen.append(("restore", self.controller.curr_depth))
        return self.best_depth_value


def _attached(checkpoint=None, max_depth=-1):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2))
    host = TorchHost(model)
    host.set_optimizer(0, torch.optim.SGD(model[2].parameters(), lr=0.1))  # the host built it
    fts = FTSController(model, optimizer=torch.optim.SGD, host=host, checkpoint=checkpoint, max_depth=max_depth)
    executor = RecordingExecutor(fts)
    fts.attach(executor, SCHEDULE)
    return fts, executor


def test_request_advance_sequences_the_transition_through_the_executor():
    fts, executor = _attached()
    events = []
    fts.add_phase_hook(lambda e: events.append(("end", e.depth)), when="end")
    fts.add_phase_hook(lambda e: events.append(("start", e.depth)))
    assert isinstance(executor, PhaseExecutor) and fts.curr_depth == 0
    fts.request_advance("the host decided")
    assert fts.curr_depth == 1 and events == [("end", 0), ("start", 1)]
    assert executor.calls == [("configure_phase", 1, 0, False, True), ("after_transition", 1)]


def test_a_host_restoring_policy_restores_after_the_depth_advances():
    """As the host's own restore expects, the policy restores once the depth is the new phase's; the phase is then
    configured from the restored depth."""
    seen = []
    policy = HostRestoringPolicy(best_depth=0, seen=seen)
    fts, executor = _attached(checkpoint=policy)
    policy.controller = fts
    fts.request_advance()
    assert seen == [("restore", 1)] and fts.restored_count == 1
    assert executor.calls[0] == ("configure_phase", 1, 0, True, True)


def test_resume_at_thaws_through_the_executor_without_configuring_the_optimizer():
    fts, executor = _attached()
    fts.resume_at(2)
    assert fts.curr_depth == 2 and executor.calls == [("resume_at", 2), ("after_transition", 2)]


def test_host_driven_refusals():
    fts, _ = _attached(max_depth=1)
    for loop_method in (lambda: next(fts.epochs(2)), fts.epoch_start, fts.epoch_end, fts.step_end):
        with pytest.raises(FTSConfigurationError, match="host-driven controller refuses it"):
            loop_method()
    for resume_method in (lambda: fts.save("x.pt"), lambda: fts.load("x.pt"), lambda: fts.load_state_dict({})):
        with pytest.raises(FTSConfigurationError, match="the host saves and restores the training state"):
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
    plain = FTSController(model, {0: {"params": ["weight", "bias"]}}, optimizer=torch.optim.SGD)
    plain.setup()
    with pytest.raises(FTSConfigurationError, match="set it up with `attach\\(\\)`"):
        plain.request_advance()


def test_attach_refuses_registered_gates_and_policy_restore_is_host_driven_only():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2))
    host = TorchHost(model)
    host.set_optimizer(0, torch.optim.SGD(model[2].parameters(), lr=0.1))
    fts = FTSController(model, optimizer=torch.optim.SGD, host=host)
    fts.add_gate(lambda e: True)
    with pytest.raises(FTSConfigurationError, match="Gates are not supported"):
        fts.attach(RecordingExecutor(fts), SCHEDULE)
    # a controller-driven controller ignores a policy's `restore` and restores its own checkpoint files
    seen = []
    policy = HostRestoringPolicy(best_depth=0, seen=seen)
    policy.best_path = None
    driven = FTSController(nn.Linear(2, 2), {0: {"params": ["weight", "bias"]}}, optimizer=torch.optim.SGD,
                           checkpoint=policy)
    assert driven._restore_best_state() is None and seen == []


def test_the_default_executor_resumes_by_thawing_each_phase():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2))
    schedule = {d: {**p, "max_transition_epoch": d + 1} for d, p in SCHEDULE.items()}
    fts = FTSController(model, schedule, optimizer=torch.optim.SGD)
    fts.setup()
    ControllerExecutor(fts).resume_at(1)
    assert sorted(n for n, p in model.named_parameters() if p.requires_grad) == ["1.bias", "1.weight", "2.bias",
                                                                                  "2.weight"]
