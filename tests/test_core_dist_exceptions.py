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
"""Tests for the framework-neutral core's exception and rank-zero backends, and its import boundary (#50)."""
import ast
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

import pytest
from lightning.fabric.utilities import rank_zero_only
from lightning.pytorch.utilities.exceptions import MisconfigurationException

import finetuning_scheduler
from finetuning_scheduler.core import dist as core_dist
from finetuning_scheduler.core import exceptions as core_exceptions
from finetuning_scheduler.core.exceptions import (configuration_error_cls, FTSConfigurationError,
                                                  register_configuration_error)
from finetuning_scheduler.lightning import FTSMisconfigurationException, LIGHTNING_RANK_ZERO_BACKEND

THIS_FILE = os.path.abspath(__file__)
CORE_DIR = Path(finetuning_scheduler.__file__).parent / "core"
CORE_IMPORT_ALLOWLIST = {"torch", "yaml", "fsspec", "typing_extensions"}
# besides the core itself: the build-time helpers and the package metadata (version strings)
CORE_FTS_IMPORT_ALLOWLIST = ("finetuning_scheduler.core", "finetuning_scheduler.setup_tools",
                             "finetuning_scheduler.__about__")


@pytest.fixture
def default_backend():
    """Run a test against the core's default (Lightning-free) rank-zero backend."""
    core_dist.register_rank_zero_backend(core_dist.DEFAULT_BACKEND)
    try:
        yield core_dist.DEFAULT_BACKEND
    finally:
        core_dist.register_rank_zero_backend(LIGHTNING_RANK_ZERO_BACKEND)


@pytest.fixture
def unregistered_error():
    """Run a test with no host exception type registered."""
    orig = core_exceptions._HOST_ERROR_CLS
    core_exceptions._HOST_ERROR_CLS = None
    try:
        yield
    finally:
        core_exceptions._HOST_ERROR_CLS = orig


def _caller_warns(message: str) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        core_dist.rank_zero_warn(message)
    return list(record)


def test_lightning_registers_configuration_error():
    """Under Lightning, core configuration errors are also Lightning ``MisconfigurationException`` s."""
    assert configuration_error_cls() is FTSMisconfigurationException
    with pytest.raises(MisconfigurationException, match="bad schedule") as excinfo:
        raise FTSConfigurationError("bad schedule")
    err = excinfo.value
    assert type(err) is FTSMisconfigurationException
    assert isinstance(err, (FTSConfigurationError, ValueError))
    roundtrip = pickle.loads(pickle.dumps(err))
    assert type(roundtrip) is FTSMisconfigurationException and roundtrip.args == ("bad schedule",)


def test_configuration_error_without_host(unregistered_error):
    err = FTSConfigurationError("plain")
    assert type(err) is FTSConfigurationError and isinstance(err, ValueError)
    assert not isinstance(err, MisconfigurationException)
    assert configuration_error_cls() is FTSConfigurationError


def test_register_configuration_error_validates():
    with pytest.raises(TypeError, match="not a subclass"):
        register_configuration_error(MisconfigurationException)  # type: ignore[arg-type]
    register_configuration_error(FTSMisconfigurationException)  # idempotent
    assert configuration_error_cls() is FTSMisconfigurationException


def test_lightning_backend_is_registered():
    assert core_dist.rank_zero_backend() is LIGHTNING_RANK_ZERO_BACKEND


@pytest.mark.parametrize("rank, expected", [(0, 1), (1, 0)], ids=["rank0", "rank1"])
def test_lightning_backend_warn_rank_and_attribution(rank, expected):
    """Lightning's rank decides emission, and warnings are attributed to the caller, not to a helper."""
    rank_zero_only.rank = rank
    record = _caller_warns("lightning backend warning")
    assert len(record) == expected
    if expected:
        assert record[0].filename == THIS_FILE


def test_lightning_backend_follows_the_process_group():
    """With a process group initialized, its rank decides, not the rank Lightning read from the environment at
    import, which a launcher passing ranks to ``init_process_group`` never sets (#97)."""
    from tests.dist_cases._fake_pg import run_fake

    def case(ctx):
        record = _caller_warns("process group decides")
        return core_dist.global_rank(), [r.filename for r in record], core_dist.rank_zero_only(lambda: True)()

    rank_zero_only.rank = 0  # what Lightning reads on every rank of such a launch
    assert run_fake(case) == [(0, [THIS_FILE], True), (1, [], None)]


def test_lightning_backend_info_uses_lightning_logger():
    rz_logger = logging.getLogger("lightning.pytorch.utilities.rank_zero")
    records: list[logging.LogRecord] = []
    handler = logging.Handler(level=logging.DEBUG)
    handler.emit = records.append  # type: ignore[method-assign]
    rz_logger.addHandler(handler)
    orig_level = rz_logger.level
    rz_logger.setLevel(logging.DEBUG)
    try:
        rank_zero_only.rank = 0
        core_dist.rank_zero_info("core info message")
        core_dist.rank_zero_debug("core debug message")
    finally:
        rz_logger.removeHandler(handler)
        rz_logger.setLevel(orig_level)
    assert [r.getMessage() for r in records] == ["core info message", "core debug message"]
    assert all(r.pathname == THIS_FILE for r in records)


def test_rank_zero_only_reads_rank_at_call_time():
    @core_dist.rank_zero_only
    def answer() -> int:
        return 42

    rank_zero_only.rank = 1
    assert answer() is None
    rank_zero_only.rank = 0
    assert answer() == 42


@pytest.mark.parametrize("env_rank, expected", [(None, 1), ("0", 1), ("1", 0)], ids=["unset", "rank0", "rank1"])
def test_default_backend_warn(default_backend, monkeypatch, env_rank, expected):
    """Without Lightning, the rank comes from the launcher environment and warnings keep caller attribution."""
    for key in core_dist._RANK_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    if env_rank is not None:
        monkeypatch.setenv("RANK", env_rank)
    record = _caller_warns("default backend warning")
    assert len(record) == expected
    if expected:
        assert record[0].filename == THIS_FILE


def test_default_backend_logging(default_backend, monkeypatch, caplog):
    for key in core_dist._RANK_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    with caplog.at_level(logging.DEBUG, logger="finetuning_scheduler"):
        core_dist.rank_zero_info("default info")
        core_dist.rank_zero_debug("default debug")
        monkeypatch.setenv("RANK", "1")
        core_dist.rank_zero_info("suppressed on rank 1")
    messages = [(r.name, r.getMessage(), r.pathname) for r in caplog.records]
    assert messages == [
        ("finetuning_scheduler", "default info", THIS_FILE),
        ("finetuning_scheduler", "default debug", THIS_FILE),
    ]


def test_default_backend_rank_from_process_group(default_backend, tmp_path):
    """With an initialized process group, the default backend takes the rank from ``torch.distributed``."""
    torch_dist = pytest.importorskip("torch.distributed")
    if not torch_dist.is_available() or torch_dist.is_initialized():
        pytest.skip("needs torch.distributed with no process group already initialized")
    torch_dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        assert core_dist.global_rank() == 0
    finally:
        torch_dist.destroy_process_group()


def test_lightning_backend_requires_rank(monkeypatch):
    """Lightning's backend refuses to report a rank before Lightning has set one."""
    from finetuning_scheduler.lightning import _lightning_global_rank

    monkeypatch.delattr(rank_zero_only, "rank", raising=False)
    with pytest.raises(RuntimeError, match="needs to be set before use"):
        _lightning_global_rank()


def _imported_top_level_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


@pytest.mark.parametrize(
    "path", sorted(CORE_DIR.rglob("*.py")), ids=lambda p: str(p.relative_to(CORE_DIR.parent))
)
def test_core_import_allowlist(path):
    """Core modules import only the standard library, the allowlisted packages and the core itself (never
    Lightning), including imports guarded by ``TYPE_CHECKING``."""
    disallowed = {
        m
        for m in _imported_top_level_modules(path)
        if m not in sys.stdlib_module_names and m not in CORE_IMPORT_ALLOWLIST and m != "finetuning_scheduler"
    }
    assert not disallowed, f"{path.name} imports {disallowed}"
    fts_imports = {
        node.module
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("finetuning_scheduler")
    }
    assert all(m.startswith(CORE_FTS_IMPORT_ALLOWLIST) for m in fts_imports), fts_imports
