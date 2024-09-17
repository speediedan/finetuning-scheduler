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
import subprocess
from unittest import mock

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint
from packaging.version import Version
from pkg_resources import get_distribution

from fts_examples import _HF_AVAILABLE
from tests.helpers.common import unexpected_warns
from tests.helpers.runif import RunIf, EXTENDED_VER_PAT

ARGS_DEFAULT = (
    "--trainer.default_root_dir %(tmpdir)s "
    "--trainer.max_epochs 1 "
    "--trainer.limit_train_batches 2 "
    "--trainer.limit_val_batches 2 "
    "--trainer.limit_test_batches 2 "
    "--trainer.limit_predict_batches 2 "
    "--data.batch_size 32 "
)
ARGS_GPU = ARGS_DEFAULT + "--trainer.devices 1 "
EXPECTED_WARNS = [
    "does not have many workers",
    "is smaller than the logging interval",
    "sentencepiece tokenizer that you are converting",
    "`resume_download` is deprecated",  # required because of upstream usage as of 2.2.2
    "distutils Version classes are deprecated",  # still required as of PyTorch/Lightning 2.2
    "Please use torch.utils._pytree.register_pytree_node",  # temp allow deprecated behavior of transformers
    "We are importing from `pydantic",  # temp pydantic import migration warning
    # allowing below until https://github.com/pytorch/pytorch/pull/123619 is resolved wrt `ZeroRedundancyOptimizer`
    "`TorchScript` support for functional optimizers is",  # required with pt 2.4 nightly 20240601
    "`is_compiling` is deprecated",  # required with pt 2.4 nightly 20240601 and `transformers` 4.41.2
    # required w/ PT 2.4 (until Lightning changes `weights_only` default value or offers a way to override it)
    "You are using `torch.load` with `weights_only=False`",
    # required for datasets <= 2.20.0 with python 3.12
    'co_lnotab is deprecated, use co_lines instead.',
    # required for google protobuf <= 3.20.1 with python 3.12
    "Use timezone-aware"

]
MIN_VERSION_WARNS = "2.0"
MAX_VERSION_WARNS = "2.4"
# torch version-specific warns go here
EXPECTED_VERSION_WARNS = {MIN_VERSION_WARNS: [],
                          MAX_VERSION_WARNS: [
                              'PairwiseParallel is deprecated and will be removed soon.',  # temp warning for pt 2.2
                              ]}
torch_version = get_distribution("torch").version
extended_torch_ver = EXTENDED_VER_PAT.match(torch_version).group() or torch_version
if Version(extended_torch_ver) < Version(MAX_VERSION_WARNS):
    EXPECTED_WARNS.extend(EXPECTED_VERSION_WARNS[MIN_VERSION_WARNS])
else:
    EXPECTED_WARNS.extend(EXPECTED_VERSION_WARNS[MAX_VERSION_WARNS])
ADV_EXPECTED_WARNS = EXPECTED_WARNS + ["Found an `init_pg_lrs` key"]


@pytest.mark.skipif(not _HF_AVAILABLE, reason="Hugging Face transformers and datasets packages required")
@RunIf(min_cuda_gpus=1, skip_windows=True)
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
        "--trainer.devices=1",
    ]
    monkeypatch.setattr("sys.argv", [example_script, "fit", "--config"] + config_loc + cli_args)
    with mock.patch.object(ModelCheckpoint, "_save_checkpoint"):  # do not save checkpoints for example tests
        cli_main()
    # ensure no unexpected warnings detected
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=EXPECTED_WARNS)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)


@pytest.mark.skipif(not _HF_AVAILABLE, reason="Hugging Face transformers and datasets packages required")
@RunIf(min_cuda_gpus=1, skip_windows=True)
@pytest.mark.parametrize(
    "config_file",
    [
        "reinit_lr/fts_explicit_reinit_lr.yaml",
        "reinit_lr/fts_implicit_reinit_lr.yaml",
        "reinit_optim_lr/fts_explicit_reinit_optim_lr.yaml",
        "reinit_optim_lr/fts_implicit_reinit_optim_lr.yaml",
    ],
    ids=[
        "fts_explicit_reinit_lr",
        "fts_implicit_reinit_lr",
        "fts_explicit_reinit_optim_lr",
        "fts_implicit_reinit_optim_lr",
    ],
)
def test_advanced_examples_fts_superglue(monkeypatch, recwarn, tmpdir, config_file):
    from fts_examples.fts_superglue import cli_main

    example_script = os.path.join(os.path.dirname(__file__), "fts_superglue.py")
    os.chdir(os.path.dirname(__file__))  # set cwd to that specified in the example
    config_loc = [os.path.join("config/advanced", config_file)]

    cli_args = [
        f"--trainer.default_root_dir={tmpdir.strpath}",
        "--trainer.max_epochs=10",
        "--trainer.limit_train_batches=2",
        "--trainer.devices=1",
    ]
    monkeypatch.setattr("sys.argv", [example_script, "fit", "--config"] + config_loc + cli_args)
    with mock.patch.object(ModelCheckpoint, "_save_checkpoint"):  # do not save checkpoints for example tests
        cli_main()
    # ensure no unexpected warnings detected
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=ADV_EXPECTED_WARNS)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)


@pytest.mark.skipif(not _HF_AVAILABLE, reason="Hugging Face transformers and datasets packages required")
@RunIf(min_cuda_gpus=1, skip_windows=True)
@pytest.mark.parametrize("nb_name", ["fts_superglue_nb"], ids=["fts_superglue_nb"])
def test_fts_superglue_nb(recwarn, nb_name):
    # simple sanity check that the notebook-based version of the example builds and executes successfully
    test_example_base = os.path.join(os.path.dirname(__file__), "ipynb_src")
    example_script = os.path.join(test_example_base, f"{nb_name}.py")
    command = ["python", "-m", "jupytext", "--set-formats", "ipynb,py:percent", example_script]
    cp = subprocess.run(command)
    assert cp.returncode == 0
    example_ipynb = os.path.join(test_example_base, f"{nb_name}.ipynb")
    assert os.path.exists(example_ipynb)
    command = ["python", "-m", "pytest", "--nbval", "-v", example_ipynb]
    cp = subprocess.run(command, capture_output=True)
    assert cp.returncode == 0
    generated_schedule = os.path.join(test_example_base, "RteBoolqModule_ft_schedule_deberta_base.yaml")
    for f in [example_ipynb, generated_schedule]:
        os.remove(f)
        assert not os.path.exists(f)
