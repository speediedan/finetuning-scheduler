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
import re
from copy import copy
from functools import partial
from typing import List, Optional, Tuple, NamedTuple, Dict, Iterable
from warnings import WarningMessage

import pytest

from lightning.pytorch import Callback, Trainer

from finetuning_scheduler import CallbackResolverMixin
from tests.helpers.runif import RunIf, RUNIF_MAP

fts_resolver = CallbackResolverMixin()

def get_fts(trainer: "Trainer") -> Callback:
    fts_resolver.connect_callback(trainer, reconnect=True)
    return fts_resolver.finetuningscheduler_callback

def default_fts_sanity_chk(trainer):
    finetuningscheduler_callback = get_fts(trainer)
    assert finetuningscheduler_callback.depth_remaining == 0
    #assert finetuningscheduler_callback.curr_depth == 2
    assert finetuningscheduler_callback.curr_depth == finetuningscheduler_callback.max_depth

def nones(num_n) -> Tuple:  # to help dedup config
    return (None,) * num_n

def multiwarn_check(
    rec_warns: List, expected_warns: List, expected_mode: bool = False
) -> List[Optional[WarningMessage]]:
    msg_search = lambda w1, w2: re.compile(w1).search(w2.message.args[0])  # noqa: E731
    if expected_mode:  # we're directed to check that multiple expected warns are obtained
        return [w_msg for w_msg in expected_warns if not any([msg_search(w_msg, w) for w in rec_warns])]
    else:  # by default we're checking that no unexpected warns are obtained
        return [w_msg for w_msg in rec_warns if not any([msg_search(w, w_msg) for w in expected_warns])]


unexpected_warns = partial(multiwarn_check, expected_mode=False)


unmatched_warns = partial(multiwarn_check, expected_mode=True)

class ExpectedResults(NamedTuple):
    expected_state: Optional[Dict] = None
    warns_expected: Optional[Tuple] = None
    exceptions_expected: Optional[Tuple] = None

class DeviceMeshSummary(NamedTuple):
    tensor_ndim: int
    mesh_ndim: int
    mesh_shape: Tuple
    mesh_dim_names: Tuple
    placement_summary: List[Optional[str | int]]

def pytest_param_factory(test_cfgs: Iterable):
    return [
        pytest.param(cfg, id=cfg.model_cfg_key,
                     marks=RunIf(**RUNIF_MAP[cfg.runif_alias]) if RUNIF_MAP.get(cfg.runif_alias, None) else tuple(),)
                     for cfg in test_cfgs ]

def fts_check_warns(recwarn, expected_warns: List, warns_expected: Optional[List] = None,
                          expected_warns_dynamo: Optional[List] = None, use_dynamo: bool = False):
    expected_warns = copy(expected_warns)
    if use_dynamo:
       expected_warns.extend(expected_warns_dynamo)
    if warns_expected:
        unmatched = unmatched_warns(rec_warns=recwarn.list, expected_warns=warns_expected)
        assert not unmatched
        expected_warns.extend(warns_expected)
    # ensure no unexpected warnings detected
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=expected_warns)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)
