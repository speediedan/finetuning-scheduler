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
"""Tests for gradient transforms on the PyTorch API: a phase's ``grad_transforms`` transform the gradients its entries
select, before any mask, and follow the schedule (#219).

These need only PyTorch and the scheduling core (the ``core-no-lightning`` job runs them with ``--noconftest``).
"""
import logging
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import yaml

from finetuning_scheduler.core.compile import compile_schedule
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.executor import TuningSchedulerExecutor
from finetuning_scheduler.core.grad_transforms import Clamp, Project, Reject, Scale
from finetuning_scheduler.core.schedule import refuse_direct_api_keys
from finetuning_scheduler.strategies.dtensor import DTensorAdapter
from finetuning_scheduler import CanonicalView, TorchInterface, TuningScheduler
from tests.test_tuning_scheduler_grad_masks import _backward, _run, _train_epoch, _val, MID_ROWS
from tests.test_tuning_scheduler_stages import _assert_same, _model, _schedule

SGD = partial(torch.optim.SGD, lr=0.1)
MOMENTUM = partial(torch.optim.SGD, lr=0.1, momentum=0.9)
ADAMW = partial(torch.optim.AdamW, lr=0.05, weight_decay=0.5)
HALF = Scale(0.5)
# scale the middle layer's gradient and mask it to rows 0 and 2, then train the head with its first row's gradient
# scaled, keeping the middle layer's transforms, then clear every transform
TRANSFORMED = {
    0: {"params": [MID_ROWS], "max_transition_epoch": 2, "grad_transforms": {"module:1": HALF}},
    1: {"params": ["2.weight", "2.bias"], "max_transition_epoch": 4,
        "grad_transforms": {"module:1": HALF, "module:2 & leaf:weight & units:[0] & of:grad": Scale(0.0)}},
    2: {"params": ["0.weight"], "max_transition_epoch": 5},
    3: {"params": ["0.bias"], "grad_transforms": {}},
}


def _compiled(schedule, model=None):
    return compile_schedule(_schedule(schedule), CanonicalView(_model() if model is None else model))


def _fts(schedule=TRANSFORMED, model=None, optimizer=SGD, **kwargs):
    fts = TuningScheduler(_model() if model is None else model, _schedule(schedule), optimizer=optimizer, **kwargs)
    fts.setup()
    return fts


def _grads(schedule, epoch=0):
    """Each trainable tensor's gradient after one backward pass under ``schedule``'s phase 0."""
    fts = _fts(schedule)
    _backward(fts, epoch)
    return {n: p.grad.clone() for n, p in fts.model.named_parameters() if p.grad is not None}


# the built-ins

W = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.mark.parametrize("transform, units, expected", [
    (Scale(2.0), None, W * 2),
    (Scale(0.0), (1,), torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])),
    (Clamp(min=2.0, max=5.0), None, W.clamp(2.0, 5.0)),
    (Clamp(max=2.0), (0,), torch.tensor([[1.0, 2.0, 2.0], [4.0, 5.0, 6.0]])),
    # venngram's parameter reject vector: over-rejecting a direction on the output side
    (Reject(torch.tensor([1.0, 0.0]), fraction=6.0), None, torch.tensor([[-5.0, -10.0, -15.0], [4.0, 5.0, 6.0]])),
    (Reject(torch.tensor([3.0, 0.0])), None, torch.tensor([[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]])),  # any length
    (Project(torch.tensor([1.0, 0.0])), None, torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])),
    (Project(torch.tensor([1.0, 0.0]), fraction=0.5), None, torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.5, 3.0]])),
    (Project(torch.eye(2)), None, W),  # the whole space
], ids=["scale", "scale_units", "clamp", "clamp_units", "reject_over", "reject_unnormalized", "project",
        "project_half", "project_all"])
def test_built_in_transforms(transform, units, expected):
    out = transform("w", W.clone(), units)
    assert torch.allclose(out, expected) and out.shape == W.shape


def test_a_projection_and_its_rejection_sum_to_the_gradient():
    g = torch.randn(4, 3)
    directions = torch.randn(4, 2)
    assert torch.allclose(Project(directions)("w", g, None) + Reject(directions)("w", g, None), g, atol=1e-6)


@pytest.mark.parametrize("make, match", [
    (lambda: Clamp(), "needs a `min`, a `max` or both"),
    (lambda: Reject(torch.zeros(2, 2, 2)), "a direction or a matrix of directions"),
    (lambda: Project(torch.ones(2), fraction=-1.0), "must be >= 0"),
])
def test_malformed_built_ins_are_refused(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_a_span_of_the_wrong_length_is_named():
    with pytest.raises(ValueError, match="its directions have 3 entries, but the gradient has 2 rows"):
        Reject(torch.ones(3))("1.weight", W, None)


# compilation


def test_transforms_resolve_like_params_and_carry_until_restated():
    compiled = _compiled(TRANSFORMED)
    first, second, third, fourth = compiled.phases
    assert first.transforms == {"1.weight": ((HALF, None),), "1.bias": ((HALF, None),)}
    # an `of: grad` entry's unit set is passed to its transforms
    assert second.transforms["2.weight"] == ((Scale(0.0), (0,)),) and third.transforms is None
    assert fourth.transforms == {}
    assert compiled.transforms_at(2) == compiled.transforms_at(1) == second.transforms
    assert compiled.transforms_at(3) == {}
    with pytest.raises(KeyError, match="no phase 4"):
        compiled.transforms_at(4)


def test_a_refreeze_drops_a_tensors_transforms():
    schedule = {0: {"params": ["1.weight", "1.bias"], "max_transition_epoch": 1, "grad_transforms": {"module:1": HALF}},
                1: {"params": ["2.weight"], "refreeze": ["1.bias"]}}
    assert _compiled(schedule).transforms_at(1) == {"1.weight": ((HALF, None),)}


def test_an_entrys_transforms_apply_in_order_after_earlier_entries():
    clip = Clamp(-0.01, 0.01)
    (cp,) = _compiled({0: {"params": ["module:1"], "grad_transforms": {"1.weight": [HALF, clip],
                                                                        "module:1 & leaf:weight": Scale(3.0)}}}).phases
    assert cp.transforms["1.weight"] == ((HALF, None), (clip, None), (Scale(3.0), None))


@pytest.mark.parametrize("schedule, match", [
    ({0: {"params": ["module:1"], "grad_transforms": {"module:2": HALF}}}, r"not trainable in the phase: \['2.bias'"),
    ({0: {"params": ["module:1"], "max_transition_epoch": 1},
      1: {"params": ["2.weight"], "refreeze": ["1.weight"], "grad_transforms": {"1.weight": HALF}}},
     r"not trainable in the phase: \['1.weight'\]"),
], ids=["never_thawed", "refrozen_in_the_phase"])
def test_transforms_on_untrained_tensors_are_refused(schedule, match):
    with pytest.raises(FTSConfigurationError, match=match):
        _compiled(schedule)


def test_the_digest_sees_transforms_only_when_present():
    plain = {0: {"params": ["module:1"]}}
    assert _compiled(plain).digest == _compiled({0: {"params": ["module:1"]}}).digest
    digests = {_compiled({0: {**plain[0], "grad_transforms": t}}).digest
               for t in ({"module:1": HALF}, {"module:1": Scale(0.25)}, {"1.weight": HALF}, {})}
    assert len(digests) == 4 and _compiled(plain).digest not in digests


@pytest.mark.parametrize("transforms, match", [
    (["1.weight"], "must be a mapping"),
    ({"1.weight": "scale"}, "must be a callable or a non-empty list of callables"),
    ({"1.weight": []}, "must be a callable or a non-empty list of callables"),
    ({1: HALF}, "is not a string"),
])
def test_malformed_grad_transforms_are_refused(transforms, match):
    with pytest.raises(FTSConfigurationError, match=match):
        _fts({0: {"params": ["module:1"], "grad_transforms": transforms}})


# training


def test_a_transform_acts_on_the_accumulated_gradient_before_the_mask():
    plain = _grads({0: {"params": ["module:1"]}})
    scaled = _grads({0: {"params": ["module:1"], "grad_transforms": {"module:1": Scale(3.0)}}})
    assert all(torch.allclose(scaled[n], plain[n] * 3) for n in plain)
    # a transform that would refill a masked row runs before the mask, which zeroes it
    refill = _grads({0: {"params": [MID_ROWS], "grad_transforms": {"module:1": lambda n, g, u: g + 1.0}}})
    assert refill["1.weight"][[1, 3]].abs().sum() == 0
    assert torch.allclose(refill["1.weight"][[0, 2]], plain["1.weight"][[0, 2]] + 1.0)


def test_transforms_apply_in_the_order_the_schedule_states():
    plain = _grads({0: {"params": ["module:1"]}})["1.weight"]
    clamp_then_scale = _grads({0: {"params": ["module:1"], "grad_transforms": {
        "1.weight": [Clamp(-0.01, 0.01), Scale(10.0)]}}})["1.weight"]
    scale_then_clamp = _grads({0: {"params": ["module:1"], "grad_transforms": {
        "1.weight": [Scale(10.0), Clamp(-0.01, 0.01)]}}})["1.weight"]
    assert torch.allclose(clamp_then_scale, plain.clamp(-0.01, 0.01) * 10)
    assert torch.allclose(scale_then_clamp, (plain * 10).clamp(-0.01, 0.01))


def test_a_unit_set_entry_transforms_only_its_rows():
    plain = _grads({0: {"params": ["module:1"]}})["1.weight"]
    rows = _grads({0: {"params": ["module:1"], "grad_transforms": {
        "module:1 & leaf:weight & units:[1, 3] & of:grad": Scale(0.0)}}})["1.weight"]
    assert rows[[1, 3]].abs().sum() == 0 and torch.equal(rows[[0, 2]], plain[[0, 2]])


def test_a_transform_returning_another_shape_is_named():
    fts = _fts({0: {"params": ["module:1"], "grad_transforms": {"1.bias": lambda n, g, u: g[:2]}}})
    with pytest.raises(RuntimeError, match=r"on '1.bias' must return a tensor of the gradient's shape \(4,\)"):
        _backward(fts, 0)


def test_transforms_follow_the_schedule(caplog):
    seen = {}
    with caplog.at_level(logging.INFO, logger="finetuning_scheduler"):
        fts = _fts()
        for epoch in fts.epochs(7):
            seen[fts.curr_depth] = set(fts.gradient_transforms)
            _train_epoch(fts, epoch)
    assert seen == {0: {"1.weight", "1.bias"}, 1: {"1.weight", "1.bias", "2.weight"},
                    2: {"1.weight", "1.bias", "2.weight"}, 3: set()}
    assert fts.gradient_masks == {"1.bias": (0, 2), "1.weight": (0, 2)}  # clearing transforms keeps the masks
    reports = [r.getMessage() for r in caplog.records if "transforms the gradients" in r.getMessage()]
    assert reports == ["Phase 0 transforms the gradients of 2 tensors: 1.weight, 1.bias.",
                       "Phase 1 transforms the gradients of 3 tensors: 1.weight, 1.bias, 2.weight.",
                       "Phase 3 transforms the gradients of 0 tensors."]


def test_a_cleared_transform_no_longer_acts():
    fts = _fts()
    for epoch in fts.epochs(7):
        _train_epoch(fts, epoch)
        if fts.curr_depth == 3:
            break
    _backward(fts, 0)
    assert fts.model.get_parameter("2.weight").grad[0].abs().sum() > 0


def test_a_reinitialized_optimizer_keeps_the_transforms():
    schedule = {**_schedule(TRANSFORMED), 1: {**TRANSFORMED[1], "new_optimizer": {"optimizer_init": {
        "class_path": "torch.optim.SGD", "init_args": {"lr": 0.05}}}}}
    fts = _fts(schedule)
    before = fts.model.get_parameter("2.weight").detach().clone()
    _run(fts, 4)
    assert fts.curr_depth == 1 and "2.weight" in fts.gradient_transforms
    after = fts.model.get_parameter("2.weight").detach()
    assert torch.equal(after[0], before[0]) and not torch.equal(after[1], before[1])


# refusals


def test_the_lightning_callback_refuses_grad_transforms():
    with pytest.raises(FTSConfigurationError, match=r"phase 0: grad_transforms"):
        refuse_direct_api_keys({0: {"params": ["x"], "grad_transforms": {"x": HALF}}})


def test_an_externally_driven_tuning_scheduler_refuses_grad_transforms():
    model = _model()
    fts = TuningScheduler(model, optimizer=torch.optim.SGD, interface=TorchInterface(model))
    with pytest.raises(FTSConfigurationError, match=r"phase 0: grad_transforms"):
        fts.attach(TuningSchedulerExecutor(fts), {0: {"params": ["module:1"], "grad_transforms": {"module:1": HALF}}})


def test_a_sharded_model_refuses_gradient_transforms():
    schedule = {0: {"params": ["module:1"], "grad_transforms": {"module:1": HALF}}}
    with pytest.raises(FTSConfigurationError, match=r"Phases \[0\] transform the gradients .* of a sharded model"):
        _fts(schedule, distributed=DTensorAdapter())


# export

TOY = {"id": "toy", "revision": "0"}
# the middle layer's gradients doubled and its weight's partly rejected from a direction on its output side, then
# cleared
WHOLE = {
    0: {"params": ["module:1"], "max_transition_epoch": 1,
        "grad_transforms": {"module:1": Scale(2), "module:1 & leaf:weight": [Reject(torch.tensor([1.0, 0, 0, 0]),
                                                                                    fraction=0.5)]}},
    1: {"params": ["2.weight", "2.bias"], "grad_transforms": {}},
}


def test_python_transforms_export_as_a_treatment_per_phase(tmp_path):
    book = _fts(WHOLE).export_book(tmp_path / "book.yaml", model=TOY)
    assert book["schema_version"] == 4 and sorted(book["treatments"]) == ["phase_0"]
    assert book["sets"]["phase_0_grad_0"] == {"all": [{"module": "1"}, {"of": "grad"}]}
    assert book["treatments"]["phase_0"] == {
        "model": TOY,
        "operands": {"phase_0_dir_0": {"values": [1.0, 0.0, 0.0, 0.0], "shape": [4], "basis": "out"}},
        "gradients": [{"op": "scale", "target": {"set": "phase_0_grad_0"}, "value": {"dose": 2.0}},
                      {"op": "reject", "target": {"set": "phase_0_grad_1"}, "operand": "phase_0_dir_0",
                       "params": {"fraction": 0.5}}]}
    # a schedule reading the book names the treatment, and clears the transforms itself
    read = {"addresses": "book.yaml", 0: {"params": ["@phase_0"], "max_transition_epoch": 1,
                                          "grad_transforms": "phase_0"},
            1: {"params": ["@phase_1"], "grad_transforms": {}}}
    (tmp_path / "read.yaml").write_text(yaml.safe_dump(read), encoding="utf-8")
    fts = TuningScheduler(_model(), str(tmp_path / "read.yaml"), optimizer=SGD)
    fts.setup()
    _backward(fts, 0)
    got = {n: p.grad for n, p in fts.model.named_parameters() if p.grad is not None}
    want = _grads(WHOLE)
    assert sorted(got) == sorted(want) and all(torch.equal(got[n], want[n]) for n in want)


def test_an_export_takes_its_model_from_the_models_config():
    model = _model()
    model.config = SimpleNamespace(_name_or_path="org/toy", _commit_hash="abc123", architectures=None)
    book = _fts(WHOLE, model=model).export_book()
    assert book["treatments"]["phase_0"]["model"] == {"id": "org/toy", "revision": "abc123"}


@pytest.mark.parametrize("transforms, kwargs, match", [
    ({"module:1": Clamp(max=1.0)}, {"model": TOY}, r"applies Clamp\(.*\), which an address book cannot hold"),
    ({"module:1": lambda name, grad, units: grad}, {"model": TOY}, "a callable is not portable"),
    ({"module:1": Scale(2)}, {}, r"records the model it was written for: pass `model="),
    ({"module:1 & leaf:weight": Reject(torch.ones(5000))}, {"model": TOY}, "more than the 4096 an exported book"),
    ({"module:1 & units:[0] & of:grad": HALF}, {"model": TOY},
     "do not compile back to the schedule: .* only a view with a component map records"),
], ids=["clamp", "callable", "no_model", "too_long", "units_without_a_map"])
def test_an_export_refuses_what_a_book_cannot_hold(transforms, kwargs, match):
    schedule = {0: {"params": ["module:1"], "grad_transforms": transforms}}
    with pytest.raises(FTSConfigurationError, match=match):
        _fts(schedule).export_book(**kwargs)


# resume


def _make(tmp_path, name, optimizer, **kwargs):
    return _fts(optimizer=optimizer, lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma=0.5),
                **kwargs)


@pytest.mark.parametrize("split", [0, 1, 2, 4, 5])
@pytest.mark.parametrize("optimizer", [MOMENTUM, ADAMW], ids=["momentum", "adamw"])
def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path, split, optimizer):
    """Saved before, at or after the transforms change, carry on or are cleared, a resumed run reinstalls the
    transforms in effect and ends where an uninterrupted one does."""
    expected = _run(_make(tmp_path, "ref", optimizer), 7, _val)
    first = _make(tmp_path, "first", optimizer)
    for epoch in first.epochs(7):
        _train_epoch(first, epoch, _val(epoch))
        if epoch == split:
            saved = first.save(tmp_path / "s.pt")
            break
    resumed = _make(tmp_path, "resumed", optimizer, model=_model(seed=7))
    resumed.load(saved)
    assert resumed.curr_depth == first.curr_depth and resumed.gradient_transforms == first.gradient_transforms
    _assert_same(expected, _run(resumed, 7, _val, start=split + 1))


def test_a_state_with_other_transforms_is_refused(tmp_path):
    saved = _fts().save(tmp_path / "s.pt")
    other = {**_schedule(TRANSFORMED), 0: {**TRANSFORMED[0], "grad_transforms": {"module:1": Scale(0.25)}}}
    with pytest.raises(FTSConfigurationError, match="different fine-tuning schedule|different tensors"):
        _fts(other).load(saved)


def test_a_schedule_with_a_function_transform_saves_and_resumes(tmp_path):
    def halve(name, grad, units):
        return grad / 2

    schedule = {0: {"params": ["module:1"], "grad_transforms": {"module:1": halve}}}
    saved = _fts(schedule).save(tmp_path / "s.pt")
    resumed = _fts(schedule, model=_model(seed=7))
    resumed.load(saved)
    assert set(resumed.gradient_transforms) == {"1.weight", "1.bias"}
