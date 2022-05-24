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
import os.path
import re

import pytest
from packaging.version import Version
from pkg_resources import get_distribution

from fts_examples import _HF_AVAILABLE
from tests.helpers.runif import RunIf

ARGS_DEFAULT = (
    "--trainer.default_root_dir %(tmpdir)s "
    "--trainer.max_epochs 1 "
    "--trainer.limit_train_batches 2 "
    "--trainer.limit_val_batches 2 "
    "--trainer.limit_test_batches 2 "
    "--trainer.limit_predict_batches 2 "
    "--data.batch_size 32 "
)
ARGS_GPU = ARGS_DEFAULT + "--trainer.gpus 1 "
EXPECTED_WARNS = [
    "does not have many workers",
    "is smaller than the logging interval",
    "torch.distributed._sharded_tensor will be deprecated",
    "`np.object` is a deprecated alias",
    "`np.int` is a deprecated alias",
    "sentencepiece tokenizer that you are converting",
]
MIN_VERSION_WARNS = "1.8"
MAX_VERSION_WARNS = "1.11"
# torch version-specific warns will go here
EXPECTED_VERSION_WARNS = {MIN_VERSION_WARNS: [], MAX_VERSION_WARNS: []}
torch_version = get_distribution("torch").version
if Version(torch_version) < Version(MAX_VERSION_WARNS):
    EXPECTED_WARNS.extend(EXPECTED_VERSION_WARNS[MIN_VERSION_WARNS])
else:
    EXPECTED_WARNS.extend(EXPECTED_VERSION_WARNS[MAX_VERSION_WARNS])
ADV_EXPECTED_WARNS = EXPECTED_WARNS + ["Found an `init_pg_lrs` key"]


@pytest.mark.skipif(not _HF_AVAILABLE, reason="Hugging Face transformers and datasets packages required")
@RunIf(min_gpus=1, skip_windows=True)
@pytest.mark.parametrize(
    "config_file",
    ["nofts_baseline.yaml", "fts_explicit.yaml", "fts_implicit.yaml"],
    ids=["nofts_baseline", "fts_explicit", "fts_implicit"],
)
def test_examples_fts_superglue(monkeypatch, recwarn, tmpdir, config_file):
    from fts_examples.fts_superglue import cli_main

    example_script = os.path.join(os.path.dirname(__file__), "fts_superglue.py")
    config_loc = [os.path.join(os.path.dirname(__file__), "config", config_file)]
    cli_args = [
        f"--trainer.default_root_dir={tmpdir.strpath}",
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=2",
        "--trainer.gpus=1",
    ]
    monkeypatch.setattr("sys.argv", [example_script, "fit", "--config"] + config_loc + cli_args)
    cli_main()
    # ensure no unexpected warnings detected
    matched = [any([re.compile(w).search(w_msg.message.args[0]) for w in EXPECTED_WARNS]) for w_msg in recwarn.list]
    assert all(matched)


@pytest.mark.skipif(not _HF_AVAILABLE, reason="Hugging Face transformers and datasets packages required")
@RunIf(min_gpus=1, skip_windows=True)
@pytest.mark.parametrize(
    "config_file",
    [pytest.param("fts_explicit_reinit_lr.yaml", marks=RunIf(min_torch="1.10")), "fts_implicit_reinit_lr.yaml"],
    ids=["fts_explicit_reinit_lr", "fts_implicit_reinit_lr"],
)
def test_advanced_examples_fts_superglue(monkeypatch, recwarn, tmpdir, config_file):
    from fts_examples.fts_superglue import cli_main

    example_script = os.path.join(os.path.dirname(__file__), "fts_superglue.py")
    config_loc = [os.path.join(os.path.dirname(__file__), "config/advanced", config_file)]
    cli_args = [
        f"--trainer.default_root_dir={tmpdir.strpath}",
        "--trainer.max_epochs=10",
        "--trainer.limit_train_batches=2",
        "--trainer.gpus=1",
    ]
    monkeypatch.setattr("sys.argv", [example_script, "fit", "--config"] + config_loc + cli_args)
    cli_main()
    # ensure no unexpected warnings detected
    matched = [any([re.compile(w).search(w_msg.message.args[0]) for w in ADV_EXPECTED_WARNS]) for w_msg in recwarn.list]
    assert all(matched)
