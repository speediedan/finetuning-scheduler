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
"""CPU tests for this project's special-test and coverage-phase configuration (``[tool.gpu-epsilon.special-tests]``
and ``[tool.gpu-epsilon.phases]`` in ``pyproject.toml``), which ``tests/special_tests.sh`` and
``scripts/gen_fts_coverage.sh`` run through gpu-epsilon's shared runner and phase driver.

The runner's and the driver's own rules are tested in gpu-epsilon. These cases check that this project's tables agree
with what reads them: the selection variables ``tests/conftest.py`` and ``RunIf`` select on, the coverage the GPU
phases expect, and the phase names the coverage script selects.
"""
import re
from pathlib import Path

from gpu_epsilon.phases import DriverConfig
from gpu_epsilon.special_tests import RunnerConfig, _parser, Run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = RunnerConfig.load(ROOT / "pyproject.toml")


def test_marks_set_the_variables_conftest_selects_on():
    conftest = (ROOT / "tests" / "conftest.py").read_text()
    assert set(CONFIG.marks) == {"standalone", "exp_patch"} and CONFIG.default_mark == "standalone"
    for spec in CONFIG.marks.values():
        for var in spec["env"]:
            assert f'os.getenv("{var}", "0") == "1"' in conftest
    assert CONFIG.env == {"PL_RUN_CUDA_TESTS": "1"}


def test_every_node_runs_under_coverage_of_the_package():
    # the measured GPU phases (tests/gpu_ci/phase_manifest.py) take their coverage from these runs
    assert CONFIG.exec[:3] == ("-m", "coverage", "run")
    assert ("--source", "src/finetuning_scheduler") in zip(CONFIG.exec, CONFIG.exec[1:])
    assert "--append" in CONFIG.exec


def test_the_experiments_list_and_its_default_mask(monkeypatch):
    monkeypatch.chdir(ROOT)
    names = (ROOT / CONFIG.experiments_list).read_text().split()
    args = _parser(CONFIG).parse_args(["--mark_type", "exp_patch"])
    assert Run(CONFIG, args).experiment_flags() == dict.fromkeys(names, False)
    env = Run(CONFIG, args).environment()
    assert env["FTS_EXPERIMENTAL_PATCH_TESTS"] == "1" and "PL_RUN_STANDALONE_TESTS" not in env


def test_the_coverage_phases_are_the_ones_the_coverage_script_selects():
    phases = DriverConfig.load(ROOT / "pyproject.toml")
    names = [p.name for p in phases.phases]
    assert names == ["main", "standalone", "examples", "examples-standalone", "exp_patch"]
    assert [p.name for p in phases.phases if not p.optional] == ["main", "standalone"]
    script = (ROOT / "scripts" / "gen_fts_coverage.sh").read_text()
    selected = re.search(r"selected_phases\(\)\{(.*?)\n\}", script, re.DOTALL).group(1)
    assert all(re.search(rf"[\",]{re.escape(name)}[\",]", selected) for name in names)
    # coverage is erased first and reported last, and the main suite runs the doctests too
    assert phases.setup == (("-m", "coverage", "erase"),) and phases.report == (("-m", "coverage", "report", "-m"),)
    main = phases.phases[0]
    assert main.run[:3] == ("-m", "coverage", "run") and "src/finetuning_scheduler" in main.run[-3:]
