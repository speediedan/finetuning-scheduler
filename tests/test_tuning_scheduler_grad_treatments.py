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
"""Tests for treatments as gradient transforms: a phase's ``grad_transforms`` naming a treatment of the schedule's
address books applies its ``gradients`` ops (venngram's ops on parameter gradients) as the built-in transforms (#219).

These need only PyTorch and the scheduling core, except the unit-axis cases, which build a tiny Llama with
``transformers`` for its component map.
"""
import copy

import pytest
import torch
import venngram
import yaml
from venngram.ops import op_cases

from finetuning_scheduler.core.compile import compile_schedule, support_record
from finetuning_scheduler.core.exceptions import FTSConfigurationError
from finetuning_scheduler.core.grad_transforms import (Add, Clamp, GRADIENT_OPS, op_transforms, Project, Reject,
                                                       Replace, Scale)
from finetuning_scheduler.core.schedule import load_address_sets, parse_grad_transforms
from finetuning_scheduler import CanonicalView, TuningScheduler
from tests.test_tuning_scheduler_grad_masks import _backward
from tests.test_tuning_scheduler_grad_transforms import _fts, SGD
from tests.test_tuning_scheduler_stages import _model

MODEL = {"id": "toy", "revision": "0"}
CONCEPT = [1.0, 0.0, 0.0, 0.0]
# the middle layer's gradients, scaled, then its weight's rejected from a direction on its output side
BOOK = {
    "sets": {"mid": {"module": "1"}, "mid_grads": {"module": "1", "of": "grad"},
             "mid_weight_grad": {"module": "1", "leaf": "weight", "of": "grad"},
             "mid_rows": {"module": "1", "units": [0], "of": "grad"}},
    "treatments": {"calm": {"model": MODEL, "operands": {"concept": {"values": CONCEPT, "shape": [4], "basis": "out"}},
                            "gradients": [{"op": "scale", "target": {"set": "mid_grads"}, "value": {"dose": 0.5}},
                                          {"op": "reject", "target": {"set": "mid_weight_grad"},
                                           "operand": "concept", "params": {"fraction": 0.5}}]}},
}
TREATED = {"addresses": BOOK, 0: {"params": ["@mid"], "max_transition_epoch": 1, "grad_transforms": "calm"},
           1: {"params": ["2.weight", "2.bias"]}}


def _book(**treatments):
    book = copy.deepcopy(BOOK)
    book["treatments"] = {name: {"model": MODEL, **body} for name, body in treatments.items()}
    return book


def _compiled(schedule, model=None):
    phases = {d: p for d, p in schedule.items() if d != "addresses"}
    return compile_schedule(phases, CanonicalView(_model() if model is None else model),
                            addresses=load_address_sets(schedule.get("addresses")))


# the built-ins along an axis

W = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.mark.parametrize("transform, units, expected", [
    (Scale(2.0, dim=1), (1,), torch.tensor([[1.0, 4.0, 3.0], [4.0, 10.0, 6.0]])),
    (Replace(0.0, dim=1), (0, 2), torch.tensor([[0.0, 2.0, 0.0], [0.0, 5.0, 0.0]])),
    (Replace(7.0), None, torch.full_like(W, 7.0)),
    (Add(1.0), (1,), torch.tensor([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])),
    (Clamp(max=4.0, dim=1), (2,), torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 4.0]])),
    (Reject(torch.tensor([1.0, 0.0, 0.0]), dim=1), None, torch.tensor([[0.0, 2.0, 3.0], [0.0, 5.0, 6.0]])),
    (Project(torch.tensor([0.0, 0.0, 1.0]), dim=1), None, torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 6.0]])),
], ids=["scale_dim1", "replace_dim1", "replace", "add", "clamp_dim1", "reject_dim1", "project_dim1"])
def test_built_ins_act_along_their_axis(transform, units, expected):
    out = transform("w", W.clone(), units)
    assert torch.allclose(out, expected) and out.shape == W.shape


@pytest.mark.parametrize("transform, match", [
    (Scale(2.0, dim=2), r"`Scale` on 'w': dimension 2 is not an axis of a gradient of shape \(2, 3\)"),
    (Reject(torch.ones(2), dim=1), "its directions have 2 entries, but the gradient has 3 entries along dimension 1"),
])
def test_an_axis_the_gradient_lacks_is_named(transform, match):
    with pytest.raises(ValueError, match=match):
        transform("w", W, (0,))


def _evaluate(case):
    """A venngram parameter conformance vector, read with the gradient as the tensor, through the built-ins."""
    i = case.inputs
    w = torch.tensor(i["w"], dtype=torch.float64).reshape([int(s) for s in i["shape"]])
    if case.op in ("project", "reject"):
        ((fn, units),) = op_transforms(case.op, directions=torch.tensor(i["u"]), fraction=i["fraction"][0])
        return fn("w", w, units)
    dim = int(i["dim"][0]) if "dim" in i else 0
    doses = {int(u): i["dose"][0] for u in i["units"]} if case.target == "param_units" else None
    out = w
    for fn, units in op_transforms(case.op, dim=dim, dose=i["dose"][0], doses=doses):
        out = fn("w", out, units)
    return out


GRADIENT_CASES = [c for t in ("param", "param_units") for c in op_cases(target=t)
                  if c.op in GRADIENT_OPS and "w" in c.expect and not {"v", "indices", "b"} & set(c.inputs)]


@pytest.mark.parametrize("case", GRADIENT_CASES, ids=[c.name for c in GRADIENT_CASES])
def test_the_built_ins_meet_venngrams_parameter_vectors(case):
    assert torch.allclose(_evaluate(case).flatten(), torch.tensor(case.expect["w"], dtype=torch.float64))


def test_the_vectors_cover_a_unit_axis_and_a_span():
    assert {"param-units-scale-along-the-first-dimension-of-a-transposed-weight",
            "param-reject-over-rejects-a-direction-on-the-output-side"} <= {c.name for c in GRADIENT_CASES}


def test_per_unit_doses_become_one_built_in_per_dose():
    assert op_transforms("scale", dim=1, doses={4: 0.5, 1: 0.0, 2: 0.5}) == [(Scale(0.0, dim=1), (1,)),
                                                                           (Scale(0.5, dim=1), (2, 4))]


def test_the_support_record_declares_the_ops_applied():
    steering = support_record().steering
    assert steering.gradient_ops == frozenset(GRADIENT_OPS) and steering.ops == frozenset()
    assert steering.targets == {"param", "param_units"} and steering.of == {"grad"}


# compilation


def test_a_treatment_compiles_to_its_ops_in_order():
    (first, second) = _compiled(TREATED).phases
    weight, bias = first.transforms["1.weight"], first.transforms["1.bias"]
    assert weight[0] == (Scale(0.5), None) and bias == ((Scale(0.5), None),)
    ((reject, units),) = weight[1:]
    assert isinstance(reject, Reject) and units is None and reject.fraction == 0.5 and reject.dim == 0
    assert torch.equal(reject.directions, torch.tensor(CONCEPT, dtype=torch.float64))
    assert second.transforms is None


def test_a_treatment_trains_as_its_built_ins_do():
    treated = _fts(TREATED)
    _backward(treated, 0)
    concept = torch.tensor(CONCEPT)
    built_ins = {**TREATED, 0: {**TREATED[0], "grad_transforms": {"@mid_grads": Scale(0.5),
                                                                  "@mid_weight_grad": Reject(concept, 0.5)}}}
    reference = _fts(built_ins)
    _backward(reference, 0)
    for (name, p), q in zip(treated.model.named_parameters(), reference.model.parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert torch.allclose(p.grad, q.grad), name


def test_a_subspace_operand_stacks_its_directions():
    book = _book(span={"operands": {"plane": {"values": [1.0, 0, 0, 0, 0, 1.0, 0, 0], "shape": [2, 4],
                                              "basis": "out"}},
                       "gradients": [{"op": "project", "target": {"set": "mid_weight_grad"}, "operand": "plane"}]})
    (cp,) = _compiled({"addresses": book, 0: {"params": ["@mid"], "grad_transforms": "span"}}).phases
    ((project, _),) = cp.transforms["1.weight"]
    assert torch.equal(project.directions, torch.eye(4, dtype=torch.float64)[:, :2])


def test_a_schedule_from_yaml_names_a_treatment(tmp_path):
    path = tmp_path / "schedule.yaml"
    path.write_text(yaml.safe_dump(TREATED, sort_keys=False))
    fts = TuningScheduler(_model(), str(path), optimizer=SGD)
    fts.setup()
    assert set(fts.gradient_transforms) == {"1.weight", "1.bias"}


@pytest.mark.parametrize("schedule, match", [
    ({"addresses": BOOK, 0: {"params": ["@mid"], "grad_transforms": "loud"}},
     r"names the treatment `loud`, which none of the schedule's address books .* has \(treatments: \['calm'\]\)"),
    ({"addresses": BOOK, 0: {"params": ["2.weight"], "grad_transforms": "calm"}},
     r"treatment `calm` op 0 \(`scale`, on `@mid_grads`\) selects tensors that are not trainable"),
    ({"addresses": _book(steer={"stages": [{"site": "blocks.1", "ops": [{"op": "add", "value": {"dose": 1.0}}]}]}),
      0: {"params": ["@mid"], "grad_transforms": "steer"}}, "has `stages`: Fine-Tuning Scheduler applies a treatment"),
    ({"addresses": _book(edit={"weights": [{"op": "scale", "target": {"set": "mid"}, "value": {"dose": 0.0}}]}),
      0: {"params": ["@mid"], "grad_transforms": "edit"}}, "has `weights`: Fine-Tuning Scheduler applies"),
    ({"addresses": _book(shift={"operands": {"c": {"values": CONCEPT, "shape": [4], "basis": "out"}},
                                "gradients": [{"op": "add", "target": {"set": "mid_grads"}, "operand": "c"}]}),
      0: {"params": ["@mid"], "grad_transforms": "shift"}}, "an operand, which only `project` and `reject` take"),
    ({"addresses": _book(rows={"gradients": [{"op": "scale", "target": {"set": "mid_rows"}, "value": {"dose": 0.0}}]}),
      0: {"params": ["@mid"], "grad_transforms": "rows"}}, "whose unit axis only a view with a component map"),
], ids=["unknown", "untrained", "stages", "weights", "operand", "units_without_a_map"])
def test_a_treatment_asking_for_more_is_refused(schedule, match):
    with pytest.raises(FTSConfigurationError, match=match):
        _compiled(schedule)


def test_a_treatment_name_must_be_a_non_empty_string():
    parse_grad_transforms(0, "calm")
    with pytest.raises(FTSConfigurationError, match="or the name of a treatment of the schedule's address books"):
        parse_grad_transforms(0, "")


def test_a_schedule_naming_a_treatment_is_not_exported(tmp_path):
    with pytest.raises(FTSConfigurationError, match="a treatment is already in the book that holds it"):
        _fts(TREATED).export_book(tmp_path / "book.yaml")


# the unit axis, through a component map


@pytest.fixture
def llama():
    """A tiny Llama's view, through its component map with the layouts measured on it (each tensor's unit axis)."""
    transformers = pytest.importorskip("transformers")
    from venngram.maps import component_map_for, from_document
    from venngram.maps.drafter import map_document
    from venngram.maps.verification import with_layouts

    config = transformers.LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=12, num_hidden_layers=2,
                                      num_attention_heads=2, num_key_value_heads=2)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config)
    cmap = from_document(with_layouts(map_document(component_map_for("LlamaForCausalLM")), model))
    return venngram.view(model, cmap)


DOWN = "model.layers.1.mlp.down_proj.weight"  # [hidden, intermediate]: its units (neurons) run along dimension 1
NEURONS = {"sets": {"mlp": {"component": "mlp.out", "layers": [1]},
                    "neurons": {"component": "mlp.out", "layers": [1], "leaf": "weight", "units": [2, 5], "of": "grad"},
                    "down": {"component": "mlp.out", "layers": [1], "leaf": "weight", "of": "grad"}},
           "treatments": {"quiet": {"model": MODEL, "operands": {"c": {"values": [1.0] + [0.0] * 7, "shape": [8],
                                                                       "basis": "unfolded"}},
                                    "gradients": [{"op": "scale", "target": {"set": "neurons"},
                                                   "value": {"doses": {1: {2: 0.5, 5: 0.0}}}},
                                                  {"op": "reject", "target": {"set": "down"}, "operand": "c"}]}}}


def test_units_act_along_the_unit_axis_and_spans_along_the_out_side(llama):
    compiled = compile_schedule({0: {"params": ["@mlp"], "grad_transforms": "quiet"}}, llama,
                                addresses=load_address_sets(NEURONS))
    (cp,) = compiled.phases
    scale_2, scale_5, (reject, units) = cp.transforms[DOWN]
    assert scale_2 == (Scale(0.5, dim=1), (2,)) and scale_5 == (Scale(0.0, dim=1), (5,))
    assert isinstance(reject, Reject) and reject.dim == 0 and units is None
    grad = torch.ones(8, 12)
    for fn, units in cp.transforms[DOWN]:
        grad = fn(DOWN, grad, units)
    assert torch.equal(grad[0], torch.zeros(12)) and torch.equal(grad[1:, 2], torch.full((7,), 0.5))
    assert torch.equal(grad[1:, 5], torch.zeros(7)) and torch.equal(grad[1:, 0], torch.ones(7))


def test_units_beyond_the_unit_axis_are_named(llama):
    book = copy.deepcopy(NEURONS)
    book["sets"]["neurons"]["units"] = [2, 12]
    book["treatments"]["quiet"]["gradients"][0]["value"] = {"dose": 0.0}
    with pytest.raises(FTSConfigurationError, match=r"names units \[12\] of `model.layers.1.mlp.down_proj.weight`, "
                                                    r"which has 12 along its unit axis \(dimension 1\)"):
        compile_schedule({0: {"params": ["@mlp"], "grad_transforms": "quiet"}}, llama,
                         addresses=load_address_sets(book))
