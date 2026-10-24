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
import json
import math
import os.path
import subprocess
import sys
from unittest import mock
from itertools import chain
from pathlib import Path
import re

import pytest
import yaml
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
        ("fts_fsdp_auto_plan.yaml",),
        ("fts_tp_plan.yaml",),
        ("fts_fsdp_profiling.yaml", "profiling/memprofiler_demo.yaml"),
    ],
    ids=[
        "fts_fsdp_auto_plan",
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
    if "profiling/memprofiler_demo.yaml" in config_files:
        _check_memprofiler_artifacts(Path(tmpdir.strpath))


# the accelerator allocator statistics the profiling guide documents, plus host memory and the default memory hooks
_MEMPROF_EXPECTED_KEYS = {"allocated_bytes.all.peak", "active_bytes.all.current", "reserved_bytes.all.peak",
                          "num_ooms", "rss", "vms", "npp_diff", "rss_diff"}


def _check_memprofiler_artifacts(root: Path) -> None:
    """Each rank of the profiling example records the documented accelerator statistics in every snapshot and
    writes its allocator history and ``FSDP2MemTracker`` reports, on CUDA and ROCm alike (both have device type
    ``cuda``)."""
    for rank in (0, 1):
        stats_file = next(root.rglob(f"rank_{rank}_memory_stats.yaml"))
        with open(stats_file) as f:
            stats = yaml.safe_load(f)
        assert set(stats) == {f"{rank}.training_step.2.{ctx}" for ctx in ("start", "end")}
        for snap_key, snap_stats in stats.items():
            assert not (missing := _MEMPROF_EXPECTED_KEYS - set(snap_stats)), (snap_key, missing)
        memprof_dir = stats_file.parent
        assert len(list(memprof_dir.glob(f"cuda_alloc_rank_{rank}.training_step.2.*.pickle"))) == 2
        assert (memprof_dir / f"fsdp_mem_rank_{rank}_training_step_2.log").exists()


@RunIf(min_cuda_gpus=2, standalone=True)
@pytest.mark.parametrize("config_file", ["torch_fsdp_plan.yaml", "torch_tp_plan.yaml"],
                         ids=["torch_fsdp_plan", "torch_tp_plan"])
def test_model_parallel_examples_torch(tmpdir, config_file):
    """The PyTorch API model-parallel example runs its schedule under ``torchrun`` without importing Lightning."""
    mp_example_base = os.path.join(os.path.dirname(__file__), "model_parallel")
    example_script = os.path.join(mp_example_base, "torch_mp_example.py")
    config = os.path.join(mp_example_base, "config", config_file)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", example_script,
               "--config", config, "--output-dir", tmpdir.strpath, "--max-epochs=3", "--limit-train-batches=2"]
    # `MKL_THREADING_LAYER`: see https://github.com/pytorch/pytorch/issues/37377; setting `OMP_NUM_THREADS` keeps
    # `torchrun` from warning that it set it
    env = {**os.environ, "MKL_THREADING_LAYER": "GNU", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
    cp = subprocess.run(command, capture_output=True, text=True, env=env)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    warn_lines = [line for line in chain(cp.stdout.splitlines(), cp.stderr.splitlines())
                  if re.search(r"[^\s]+\.py:\d+", line)]
    unexpected = unexpected_warns(rec_warns=warn_lines, expected_warns=MODEL_PARALLEL_EXAMPLE_WARNS, raw_warns=True)
    assert not unexpected, unexpected
    with open(os.path.join(tmpdir, config_file.removesuffix(".yaml"), "summary.json")) as f:
        summary = json.load(f)
    # one phase per epoch (each phase's `max_transition_epoch`), each thawing the scheduled parameters
    assert [(e["depth"], e["trainable_params"]) for e in summary["epochs"]] == [(0, 2), (1, 11), (2, 39)]
    assert summary["final_depth"] == 2
    assert all(math.isfinite(e["train_loss"]) and math.isfinite(e["val_loss"]) for e in summary["epochs"])
    assert summary["restored_count"] == 2  # the best checkpoint restored at both phase boundaries
    assert summary["param_types"] == ["DTensor"]  # nothing left unsharded
    assert summary["lightning_modules"] == []


NOTEBOOK_DEV_DIR = os.path.join(os.path.dirname(__file__), "notebooks", "dev")


def _run_notebook(rel_path: str, overrides: dict[str, str], workdir: str) -> str:
    """Execute a dev notebook in ``workdir`` with its ``parameters`` cell overridden; return its stream output."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(os.path.join(NOTEBOOK_DEV_DIR, rel_path), as_version=4)
    params = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(params) == 1, f"{rel_path} needs exactly one cell tagged 'parameters'"
    for name, value in overrides.items():
        params[0].source, n = re.subn(rf"^{name} = .*$", f"{name} = {value}", params[0].source, count=1, flags=re.M)
        assert n == 1, f"parameter {name} not found in {rel_path}'s parameters cell"
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": workdir}}).execute()
    return "".join(
        out.get("text", "") for cell in nb.cells if cell.cell_type == "code" for out in cell.get("outputs", [])
        if out.get("output_type") == "stream"
    )


@pytest.mark.skipif(not _HF_AVAILABLE, reason="Hugging Face transformers and datasets packages required")
@RunIf(min_cuda_gpus=1, skip_windows=True)
def test_fts_superglue_lightning_nb(tmpdir):
    # the Lightning tutorial notebook, shortened through its parameters cell
    _run_notebook(
        "superglue/fts_superglue_lightning.ipynb", {"MAX_EPOCHS": "1", "LIMIT_TRAIN_BATCHES": "2"}, tmpdir.strpath
    )
    assert os.path.exists(os.path.join(tmpdir, "RteBoolqModule_ft_schedule_deberta_base.yaml"))


@pytest.mark.skipif(not _HF_AVAILABLE, reason="Hugging Face transformers and datasets packages required")
@RunIf(min_cuda_gpus=1, skip_windows=True)
def test_fts_superglue_pytorch_nb(tmpdir):
    # the PyTorch API notebook, shortened through its parameters cell
    overrides = {"MAX_EPOCHS": "4", "TRAIN_SUBSET": "32", "VAL_SUBSET": "16"}
    stdout = _run_notebook("superglue/fts_superglue_pytorch.ipynb", overrides, tmpdir.strpath)
    for scenario in ("fts_explicit", "fts_implicit", "nofts_baseline"):
        assert re.search(rf"^{scenario}: finished at phase \d+ after \d+ epochs", stdout, flags=re.M), stdout
