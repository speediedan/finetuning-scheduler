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
from itertools import chain
import re

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from fts_examples import _HF_AVAILABLE

from tests.helpers.common import unexpected_warns
from tests.helpers.runif import RunIf
from tests.helpers.expected_warns import (EXAMPLE_EXPECTED_WARNS, MODEL_PARALLEL_EXAMPLE_WARNS,
                                          ADV_EXAMPLE_EXPECTED_WARNS)

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
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=EXAMPLE_EXPECTED_WARNS)
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
    unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=ADV_EXAMPLE_EXPECTED_WARNS)
    assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)


@RunIf(min_cuda_gpus=2, standalone=True)
@pytest.mark.parametrize(
    "config_files",
    [
        # ("fts_fsdp_auto_plan.yaml",), # TODO: debug mixed DTensor/Non-DTensor parameter group issue with PT 2.9+
        ("fts_tp_plan.yaml",),
        ("fts_fsdp_profiling.yaml", "profiling/memprofiler_demo.yaml"),
    ],
    ids=[
        # "fts_fsdp_auto_plan",
        "fts_tp_plan",
        "fts_fsdp_profiling",
    ],
)
def test_model_parallel_examples(tmpdir, config_files):
    os.environ["MKL_THREADING_LAYER"] = "GNU"  # see https://github.com/pytorch/pytorch/issues/37377
    mp_example_base = os.path.join(os.path.dirname(__file__), "model_parallel")
    example_script = os.path.join(mp_example_base, "mp_examples.py")
    os.chdir(mp_example_base)  # set cwd to that specified in the example
    config_locs = []
    for config_file in config_files:
        config_locs.append("--config")
        config_locs.append(os.path.join(mp_example_base, "config", config_file))

    cli_args = [
        f"--trainer.default_root_dir={tmpdir.strpath}", "--trainer.max_epochs=1",  "--trainer.limit_train_batches=2",
    ]
    command = ["python", example_script, "fit"] + config_locs + cli_args
    cp = subprocess.run(command, capture_output=True, text=True)
    stdout_lines, stderr_lines = cp.stdout.splitlines(), cp.stderr.splitlines()
    warn_lines = []
    for line in chain(stdout_lines, stderr_lines):
        if re.search(r"[^\s]+\.py:\d+", line):
            warn_lines.append(line)
    assert cp.returncode == 0
    unexpected = unexpected_warns(rec_warns=warn_lines, expected_warns=MODEL_PARALLEL_EXAMPLE_WARNS, raw_warns=True)
    assert not unexpected, unexpected

@pytest.mark.skipif(not _HF_AVAILABLE, reason="Hugging Face transformers and datasets packages required")
@RunIf(min_cuda_gpus=1, skip_windows=True)
@pytest.mark.parametrize("nb_name", ["fts_superglue_nb"], ids=["fts_superglue_nb"])
def test_fts_superglue_nb(nb_name):
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
