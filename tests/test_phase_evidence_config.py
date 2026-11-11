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
"""CPU tests for this project's phase-evidence configuration (``[tool.gpu-epsilon.phase-evidence]`` in
``pyproject.toml``), which gpu-epsilon's writer (``python -m gpu_epsilon.phase_evidence``) reads for the GPU job's
phase manifests and reports.

The writer's own rules (collection, per-rank verdicts, coverage, classification, gating) are tested in gpu-epsilon,
against the module this project used to carry. These cases check that this project's table agrees with what reads it
and what it describes: the GPU job's phase steps, the variables the test selection reads, the persistent pool's
modules, the smoke scope and the reviewed exclusions.
"""
import json
import re
from pathlib import Path

import pytest
from gpu_epsilon import phase_evidence as pe

ROOT = Path(__file__).resolve().parents[1]
CONFIG = pe.EvidenceConfig.load(ROOT / "pyproject.toml")
PHASES = {p.name: p for p in CONFIG.phases}
CONTAINER = (ROOT / ".azure-pipelines" / "scripts" / "gpu_job_container.sh").read_text(encoding="utf-8")


def test_the_phases_are_the_gpu_jobs_phases():
    assert list(PHASES) == ["gpu-nonstandalone", "standalone", "examples", "examples-multigpu", "pool-reverse"]
    assert [n for n, p in PHASES.items() if p.release] == list(PHASES)[:4]
    assert [n for n, p in PHASES.items() if not p.measured] == ["examples", "pool-reverse"]
    # a measured phase's command must write coverage (the special-test runner runs every test under ``coverage run``)
    for name, phase in PHASES.items():
        measures = "coverage run" in phase.command or "gpu_epsilon.special_tests" in phase.command
        assert measures == phase.measured, name
    # the job collects every phase it runs, and runs a step for each
    collected = set(re.findall(r"--phase ([\w-]+)", CONTAINER))
    assert collected == set(PHASES)
    steps = {s for arm in re.findall(r"^      ([\w|-]+)\)$", CONTAINER, re.M) for s in arm.split("|")}
    assert set(PHASES) <= steps


def test_the_pool_reverse_leg_runs_the_pool_modules():
    phase = PHASES["pool-reverse"]
    modules = phase.resolved_paths(ROOT)
    assert modules and modules == tuple(sorted(modules))
    assert all(phase.paths_from["contains"] in (ROOT / m).read_text(encoding="utf-8") for m in modules)
    # this module must not join the reverse-order leg by naming the fixture itself
    assert "tests/test_phase_evidence_config.py" not in modules
    selector = CONFIG.selector(phase)
    assert selector.startswith("FTS_REVERSE_ORDER=1 PL_RUN_CUDA_TESTS=1; pytest ")
    assert all(m in selector for m in modules)


def test_the_smoke_scope_narrows_only_the_standalone_phase():
    name, scope = CONFIG.resolve_scope("standalone", "smoke")
    assert name == "smoke" and (ROOT / scope.select_file).is_file()
    assert CONFIG.selector(PHASES["standalone"], scope).endswith("; select_file=tests/.standalone_smoke")
    assert "FTS_STANDALONE_SCOPE" in CONTAINER
    with pytest.raises(ValueError, match="scope 'smoke' is not defined"):
        CONFIG.resolve_scope("examples", "smoke")


def test_every_selection_variable_the_tests_read_is_stripped_before_collecting():
    """An inherited selection variable must not change what a phase collects; ``PL_RUN_CUDA_TESTS`` gates skips
    only, so every collection sets it instead."""
    sources = "".join((ROOT / f).read_text(encoding="utf-8") for f in ("tests/conftest.py", "tests/helpers/runif.py"))
    read = set(re.findall(r'os\.getenv\("((?:PL|FTS)_\w+)"', sources))
    assert read and read - set(CONFIG.selection_vars) == {"PL_RUN_CUDA_TESTS"}
    assert CONFIG.collect_env == {"PL_RUN_CUDA_TESTS": "1"}


def test_run_env_records_every_rank(tmp_path):
    env = pe.run_env(CONFIG, "standalone", tmp_path)
    assert env["PL_RUN_STANDALONE_TESTS"] == "1" and env["PL_RUN_CUDA_TESTS"] == "1"
    assert env[pe.PHASE_RECORD_DIR_ENV] == str(tmp_path.resolve() / "records" / "standalone")
    assert env[pe.PHASE_RECORD_ROOT_ENV] == str(ROOT)
    assert env["PYTEST_ADDOPTS"].endswith(f"-p {pe.PLUGIN}")
    # a measured phase gets its own coverage data file; an unmeasured one none
    assert env["COVERAGE_FILE"] == str(pe.coverage_file(tmp_path, "standalone"))
    assert "COVERAGE_FILE" not in pe.run_env(CONFIG, "examples", tmp_path)


@pytest.mark.parametrize(
    "classification, scope, passes",
    [("clean", "full", True), ("unreviewed-exclusions", "smoke", False), ("empty", "smoke", False),
     ("failing", "full", False)],
)
def test_only_a_clean_phase_passes(classification, scope, passes):
    for job_class in ("ci-any-vendor", "ci-release"):
        assert pe.gate(CONFIG, classification, scope, job_class) is passes


def test_reviewed_exclusions_name_existing_tests():
    # every reviewed skip or xfail names a test that still exists (its module, function and parameter id) and gives a
    # reason, so a renamed test cannot leave a stale entry that makes some other skip look reviewed
    reviewed = json.loads((ROOT / "tests/gpu_ci/reviewed_exclusions.json").read_text(encoding="utf-8"))
    assert reviewed
    for nodeid, reason in reviewed.items():
        assert reason.strip(), nodeid
        module, _, test = nodeid.partition("::")
        name, _, param = test.partition("[")
        source = (ROOT / module).read_text(encoding="utf-8")
        assert f"def {name}(" in source, nodeid
        assert not param or f'"{param.rstrip("]")}"' in source, nodeid
