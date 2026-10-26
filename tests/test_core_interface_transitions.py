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
"""Tests for the core's training interface seam and its transition policies (#51)."""
import warnings

import pytest
import torch
from torch import nn

from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.interface import LRStepConfig, TorchInterface, TrainingInterface
from finetuning_scheduler.core.transitions import (AnyOf, CallableTransition, Decision, EpochTransition, GateResult,
                                                   PlateauTransition, StepTransition, TransitionContext, as_decision)
from tests.test_phase_characterization import PLATEAU_CASES


def _ctx(**overrides):
    base = dict(depth=0, max_depth=2, is_final_phase=False, unit="epoch", epoch=0, global_step=0,
                phase_start_epoch=0, phase_start_step=0, max_transition_epoch=-1, max_transition_step=-1)
    return TransitionContext(**{**base, **overrides})


def test_torch_interface_without_process_group():
    module = nn.Linear(2, 2)
    interface = TorchInterface(module, log_dir="logs")
    assert isinstance(interface, TrainingInterface)
    assert interface.module is module and interface.log_dir == "logs"
    assert (interface.is_global_zero, interface.world_size) == (True, 1)
    assert interface.broadcast_object({"a": 1}) == {"a": 1}
    assert interface.all_gather_object(3) == [3]
    assert (interface.reduce_max(4), interface.reduce_any(False)) == (4, False)
    interface.barrier("noop")
    opt = torch.optim.SGD(module.parameters(), lr=0.1)
    interface.set_optimizer(0, opt)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1)
    interface.set_lr_scheduler(0, sched, LRStepConfig(interval="step"))
    assert interface.get_optimizers() == [opt]
    assert interface.get_lr_schedulers() == [(sched, LRStepConfig(interval="step"))]
    replacement = torch.optim.SGD(module.parameters(), lr=0.2)
    interface.set_optimizer(0, replacement)
    assert interface.get_optimizers() == [replacement]
    interface.advance_epoch(), interface.advance_step(3)
    assert (interface.current_epoch, interface.global_step) == (1, 3)
    interface.load_counters(5, 50)
    assert (interface.current_epoch, interface.global_step) == (5, 50)
    interface.set_max_epochs(7)
    assert interface.max_epochs == 7
    assert interface.rng_devices() == [] or all(d.type == "cuda" for d in interface.rng_devices())
    assert TorchInterface(module, rng_devices=[0, "cpu"]).rng_devices() == [torch.device("cuda", 0),
    torch.device("cpu")]


def test_torch_interface_collectives_single_process_group(tmp_path):
    """With a (single-process, gloo) group initialized, the collectives run through ``torch.distributed``."""
    dist = torch.distributed
    if not dist.is_available() or dist.is_initialized():
        pytest.skip("needs torch.distributed with no process group already initialized")
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        interface = TorchInterface(nn.Linear(2, 2))
        assert (interface.is_global_zero, interface.world_size) == (True, 1)
        assert interface.reduce_max(3) == 3 and interface.reduce_any(True) is True
        assert interface.broadcast_object([1, 2]) == [1, 2]
        assert interface.all_gather_object("x") == ["x"]
        interface.barrier()
    finally:
        dist.destroy_process_group()


def test_decision_order_and_normalization():
    assert Decision.CONTINUE < Decision.ADVANCE < Decision.STOP < Decision.VETO
    assert as_decision(True) == (Decision.ADVANCE, None)
    assert as_decision(False) == (Decision.CONTINUE, None)
    assert as_decision((Decision.VETO, "no")) == (Decision.VETO, "no")
    assert as_decision(2) == (Decision.STOP, None)


def test_epoch_and_step_transitions():
    assert as_decision(EpochTransition().evaluate(_ctx(max_transition_epoch=3, epoch=2)))[0] == Decision.CONTINUE
    assert as_decision(EpochTransition().evaluate(_ctx(max_transition_epoch=3, epoch=3)))[0] == Decision.ADVANCE
    final = _ctx(max_transition_epoch=3, epoch=9, is_final_phase=True)
    assert as_decision(EpochTransition().evaluate(final))[0] == Decision.CONTINUE
    step = _ctx(unit="step", max_transition_step=100, global_step=100)
    assert as_decision(StepTransition().evaluate(step))[0] == Decision.ADVANCE
    assert as_decision(StepTransition().evaluate(_ctx(unit="step", global_step=10**6)))[0] == Decision.CONTINUE


@pytest.mark.parametrize("case", list(PLATEAU_CASES), ids=list(PLATEAU_CASES))
def test_plateau_transition_matches_lightning(case):
    """``PlateauTransition`` reaches the same best score, patience count and outcome as ``FTSEarlyStopping`` on
    every characterized sequence (stop maps to ``STOP``; a completed phase to ``ADVANCE`` with patience reset).

    Only the stop reason's last sentence differs: it names no ``Trainer``.
    """
    kwargs, final_phase, values, expected = PLATEAU_CASES[case]
    policy = PlateauTransition("val_loss", **kwargs)
    for v, (should_stop, reason, best, wait, phase_complete) in zip(values, expected):
        decision, why = as_decision(policy.evaluate(_ctx(is_final_phase=final_phase,
                                                         metrics={"val_loss": torch.tensor(v)})))
        if should_stop:
            plain_loop_reason = reason.replace("Signaling Trainer to stop.", "Stopping training.")
            assert (decision, why) == (Decision.STOP, plain_loop_reason)
        elif phase_complete and decision == Decision.ADVANCE:
            policy.reset(0)  # the tuning scheduler resets policies at the phase boundary
        else:
            assert decision == Decision.CONTINUE and why == reason
        assert float(policy.best_score) == best
        assert policy.wait_count == wait


def test_plateau_transition_missing_metric_and_state():
    strict = PlateauTransition("val_loss")
    with pytest.raises(FTSConfigurationError, match="not reported"):
        strict.evaluate(_ctx())
    lenient = PlateauTransition("val_loss", strict=False)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert as_decision(lenient.evaluate(_ctx()))[0] == Decision.CONTINUE
        lenient.evaluate(_ctx())
    assert sum("not reported" in str(w.message) for w in record) == 1
    with pytest.raises(FTSConfigurationError, match="`mode` can be"):
        PlateauTransition("val_loss", mode="sideways")  # type: ignore[arg-type]
    policy = PlateauTransition("acc", mode="max", patience=2, min_delta=0.1)
    policy.evaluate(_ctx(metrics={"acc": torch.tensor(0.5)}))
    policy.evaluate(_ctx(metrics={"acc": torch.tensor(0.55)}))
    restored = PlateauTransition("acc", mode="max", patience=2, min_delta=0.1)
    restored.load_state_dict(policy.state_dict())
    assert (float(restored.best_score), restored.wait_count) == (0.5, 1)
    assert policy.config()["init_args"]["min_delta"] == 0.1


def test_any_of_and_callable():
    seen = []

    def vetoer(ctx):
        seen.append(ctx.unit)
        return Decision.VETO, "gate says no"

    composed = AnyOf(EpochTransition(), StepTransition(), CallableTransition(vetoer, granularity="step", name="v"))
    assert composed.granularity == frozenset({"epoch", "step"})
    assert composed.evaluate(_ctx(max_transition_epoch=0)) == (Decision.ADVANCE, "max_transition_epoch (0) reached")
    assert composed.evaluate(_ctx(unit="step")) == (Decision.VETO, "gate says no")
    assert seen == ["step"]
    assert composed.state_dict() == {"0": {}, "1": {}, "2": {}}
    composed.load_state_dict(composed.state_dict())
    composed.reset(1)
    assert [c["class_path"].rsplit(".", 1)[1] for c in composed.config()["policies"]] == [
        "EpochTransition", "StepTransition", "CallableTransition"]
    assert as_decision(CallableTransition(lambda ctx: True).evaluate(_ctx()))[0] == Decision.ADVANCE
    assert GateResult(accept=False, reason="worse").record == {}
