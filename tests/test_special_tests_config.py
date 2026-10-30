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
"""CPU tests for this project's special-test configuration (``[tool.gpu-epsilon.special-tests]`` in
``pyproject.toml``), which ``tests/special_tests.sh`` runs through gpu-epsilon's shared runner.

The runner's own rules are tested in gpu-epsilon. These cases check that this project's table agrees with what reads
it: the selection variables ``tests/conftest.py`` and ``RunIf`` select on, and the coverage the GPU phases expect.
"""
from pathlib import Path

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
