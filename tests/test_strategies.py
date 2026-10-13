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
"""The torch-native distributed adapters (``finetuning_scheduler.strategies``, #54): selection, validation and
state IO, single-process and on FSDP2 across two processes (gloo on CPU; NCCL on two GPUs)."""
import pytest
import torch
from torch import nn

from finetuning_scheduler.strategies import (DistributedAdapter, DTensorAdapter, Feedback, FSDP2Adapter,
                                             select_distributed_adapter, UnshardedAdapter)
from tests.dist_cases import fts_views
from tests.helpers.dist_pool import assert_ranks_agree, dist_pool_fixture
from tests.helpers.runif import RunIf
from tests.test_controller_distributed import _close

gloo_pool = dist_pool_fixture(world_size=2, backend="gloo")
nccl_pool = dist_pool_fixture(world_size=2, backend="nccl")
GLOO = RunIf(skip_windows=True, skip_mac_os=True)


def test_selection_and_protocol():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    adapter = select_distributed_adapter(model)
    assert isinstance(adapter, UnshardedAdapter) and not adapter.sharded
    explicit = DTensorAdapter()
    assert select_distributed_adapter(model, explicit) is explicit
    for cls in (UnshardedAdapter, DTensorAdapter, FSDP2Adapter):
        assert isinstance(cls(), DistributedAdapter)
    assert not DTensorAdapter().detect(model) and not FSDP2Adapter().detect(model)
    assert UnshardedAdapter().validate(model, {0: ["1.weight"]}) == []
    assert DTensorAdapter().validate(model, {0: ["1.weight"]}) == []  # no DTensor parameter, so nothing mixed


def test_unsharded_state_round_trip(tmp_path):
    """The unsharded adapter writes one file, leaves excluded keys out, and restores the rest (moving
    parameters)."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.randn(3, 4)).sum().backward()
    optimizer.step()
    adapter = UnshardedAdapter()
    model_sd, optim_sd = adapter.state_dict(model, optimizer, exclude=lambda keys: {"0.bias"})
    assert "0.bias" not in model_sd and optim_sd is not None
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    adapter.save({"model": model_sd, "optim": optim_sd}, str(tmp_path / "state.pt"))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    adapter.load_state(str(tmp_path / "state.pt"), model, optimizer, load_optimizer=True,
                       exclude=lambda keys: {"0.bias"})
    after = model.state_dict()
    assert all(torch.equal(after[k], saved[k]) for k in model_sd)
    assert not torch.equal(after["0.bias"], saved["0.bias"])  # never saved: left as it was before the load


def test_feedback_is_a_record():
    fb = Feedback("info", "mixed groups")
    assert (fb.level, fb.message) == ("info", "mixed groups")


@GLOO
@pytest.mark.parametrize("explicit", [False, True], ids=["detected", "explicit"])
def test_roles_view_on_fsdp2_matches_single_process(gloo_pool, tmp_path, explicit):
    """A view with roles schedules an FSDP2 model as it does a single-process one; frozen roles never train, and
    the sharded best checkpoint holds fitted tensors but not instrument ones."""
    results = gloo_pool.run(fts_views.roles, ckpt_dir=str(tmp_path / "sharded"), explicit=explicit)
    agreed = assert_ranks_agree([{k: v for k, v in r.items() if k != "params"} for r in results])
    ref = fts_views.reference(ckpt_dir=str(tmp_path / "ref"))
    assert agreed["adapter"] == "fsdp2" and ref["adapter"] == "unsharded"
    assert agreed["path"] == ref["path"] and agreed["path"][-1][1] == 1
    assert agreed["frozen_unchanged"]
    assert "model.2.bias" in agreed["saved_keys"] and not {"model.0.weight", "model.0.bias"} & set(
        agreed["saved_keys"])
    assert _close(results[0]["params"], ref["params"])


@GLOO
def test_sharded_promotion_is_refused(gloo_pool, tmp_path):
    results = gloo_pool.run(fts_views.promotion_refused, ckpt_dir=str(tmp_path / "promote"))
    assert all(r["refused"] for r in results)


@RunIf(min_cuda_gpus=2)
def test_roles_view_on_fsdp2_matches_single_process_nccl(nccl_pool, tmp_path):
    results = nccl_pool.run(fts_views.roles, ckpt_dir=str(tmp_path / "sharded"))
    agreed = assert_ranks_agree([{k: v for k, v in r.items() if k != "params"} for r in results])
    ref = fts_views.reference(ckpt_dir=str(tmp_path / "ref"))
    assert agreed["path"] == ref["path"] and agreed["frozen_unchanged"]
    assert _close(results[0]["params"], ref["params"], tol=1e-4)


@GLOO
def test_an_explicit_unsharded_adapter_on_fsdp2_is_refused(gloo_pool, tmp_path):
    results = gloo_pool.run(fts_views.explicit_unsharded_refused, ckpt_dir=str(tmp_path / "explicit"))
    assert all(r["refused"] for r in results)


@GLOO
def test_a_sharded_checkpoint_loads_into_an_unsharded_model(gloo_pool, tmp_path):
    """A DCP checkpoint written by FSDP2 ranks loads into a single-process model, as before the adapter split."""
    results = gloo_pool.run(fts_views.save_sharded, ckpt_dir=str(tmp_path / "sharded"))
    model = fts_views._model(107)  # different initial weights: the load must overwrite them
    fts = fts_views._controller(model, str(tmp_path / "plain"))
    assert fts._distributed.name == "unsharded"
    fts.load(results[0]["saved"])
    loaded = fts_views._params(model)
    instrument = {"0.weight", "0.bias"}  # never saved: they keep the fresh model's values
    assert _close({k: v for k, v in loaded.items() if k not in instrument},
                  {k: v for k, v in results[0]["params"].items() if k not in instrument}, tol=0.0)


@GLOO
def test_export_best_from_a_sharded_checkpoint(gloo_pool, tmp_path):
    """Rank zero exports the best checkpoint's adapter and fitted tensors in full from the DCP directory, as a
    single-process run exports them."""
    results = gloo_pool.run(fts_views.export_sharded, ckpt_dir=str(tmp_path / "sharded"))
    ref = fts_views.export_reference(ckpt_dir=str(tmp_path / "ref"))
    exported = next(r["exported"] for r in results if r["rank"] == 0)
    assert sorted(exported) == ["2.bias", "4.bias", "4.weight"]
    assert all(r["exported"] is None for r in results if r["rank"] != 0)
    assert _close(exported, ref["exported"])
