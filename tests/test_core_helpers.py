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
"""Tests for the core's copies of helpers it previously imported from Lightning (#50)."""
import pytest
import torch
from torch import nn

from finetuning_scheduler.core.optim import _MockOptimizer
from finetuning_scheduler.core.schedule import find_shared_parameters


class _TiedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 4)
        self.decoder.weight = self.encoder.weight  # tied
        self.register_parameter("unset", None)
        self.register_module("absent", None)
        self.inner = nn.Sequential(nn.Linear(4, 4))
        self.inner[0].bias = self.encoder.bias  # tied across nesting levels


def test_find_shared_parameters_matches_lightning():
    """The core's copy reports the same tied groups as Lightning's helper, and skips unset parameters and
    submodules."""
    lightning_utilities = pytest.importorskip("lightning.pytorch.utilities")
    module = _TiedModule()
    expected = [["encoder.weight", "decoder.weight"], ["encoder.bias", "inner.0.bias"]]
    assert find_shared_parameters(module) == expected
    assert find_shared_parameters(module) == lightning_utilities.find_shared_parameters(module)
    assert find_shared_parameters(nn.Linear(2, 2)) == []


def test_mock_optimizer():
    """The stand-in optimizer used for lr scheduler sanity checks accepts and ignores optimizer calls."""
    opt = _MockOptimizer()
    opt.add_param_group({"params": [torch.zeros(1)]})
    opt.load_state_dict({})
    opt.zero_grad()
    assert opt.state_dict() == {}
    assert opt.step() is None
    assert opt.step(lambda: 1.5) == 1.5
    assert repr(opt) == "No Optimizer"
    torch.optim.lr_scheduler.StepLR(opt, step_size=1)  # schedulers can be built against it
