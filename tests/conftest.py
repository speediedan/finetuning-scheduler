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
# initially based on: https://bit.ly/3GDHDcI
import os
import signal
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

import pytest
import torch.distributed
from lightning.fabric.plugins.environments.lightning import find_free_network_port
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.pytorch.trainer.connectors.signal_connector import _SignalConnector

from tests import _PATH_DATASETS


@pytest.fixture(scope="session")
def datadir():
    return Path(_PATH_DATASETS)


@pytest.fixture(scope="function", autouse=True)
def preserve_global_rank_variable():
    """Ensures that the rank_zero_only.rank global variable gets reset in each test."""
    from lightning.fabric.utilities import rank_zero_only

    rank = getattr(rank_zero_only, "rank", None)
    yield
    if rank is not None:
        setattr(rank_zero_only, "rank", rank)


@pytest.fixture(scope="function", autouse=True)
def restore_env_variables():
    """Ensures that environment variables set during the test do not leak out."""
    env_backup = os.environ.copy()
    yield
    leaked_vars = os.environ.keys() - env_backup.keys()
    # restore environment as it was before running the test
    os.environ.clear()
    os.environ.update(env_backup)
    # these are currently known leakers - ideally these would not be allowed
    allowlist = {
        "CUBLAS_WORKSPACE_CONFIG",  # enabled with deterministic flag
        "CUDA_DEVICE_ORDER",
        "LOCAL_RANK",
        "NODE_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "PL_GLOBAL_SEED",
        "PL_SEED_WORKERS",
        "WANDB_MODE",
        "WANDB_REQUIRE_SERVICE",
        "WANDB_SERVICE",
        "RANK",  # set by DeepSpeed
        "POPLAR_ENGINE_OPTIONS",  # set by IPUStrategy
        "CUDA_MODULE_LOADING",
        "KMP_INIT_AT_FORK",
        "KMP_DUPLICATE_LIB_OK",
        "CRC32C_SW_MODE",  # leaked by tensorboardX
        "TRITON_CACHE_DIR",
        "OMP_NUM_THREADS",  # leaked by Lightning launchers
        "TORCHINDUCTOR_CACHE_DIR", # leaked by torch inductor
        "MLFLOW_TRACKING_URI",  # leaked by mlflow
        "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR",  # leaked by torch.compile
    }
    leaked_vars.difference_update(allowlist)
    assert not leaked_vars, f"test is leaking environment variable(s): {set(leaked_vars)}"


@pytest.fixture(scope="function", autouse=True)
def restore_signal_handlers():
    """Ensures that signal handlers get restored before the next test runs.

    This is a safety net for tests that don't run Trainer's teardown.
    """
    valid_signals = _SignalConnector._valid_signals()
    if not _IS_WINDOWS:
        # SIGKILL and SIGSTOP are not allowed to be modified by the user
        valid_signals -= {signal.SIGKILL, signal.SIGSTOP}
    handlers = {signum: signal.getsignal(signum) for signum in valid_signals}
    yield
    for signum, handler in handlers.items():
        if handler is not None:
            signal.signal(signum, handler)


@pytest.fixture(scope="function", autouse=True)
def teardown_process_group():
    """Ensures that the distributed process group gets closed before the next test runs."""
    yield
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.fixture(scope="function", autouse=True)
def reset_deterministic_algorithm():
    """Ensures that torch determinism settings are reset before the next test runs."""
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def caplog(caplog):
    """Workaround for https://github.com/pytest-dev/pytest/issues/3697.

    Setting ``filterwarnings`` with pytest breaks ``caplog`` when ``not logger.propagate``.
    """
    import logging

    lightning_logger = logging.getLogger("lightning.pytorch")
    propagate = lightning_logger.propagate
    lightning_logger.propagate = True
    yield caplog
    lightning_logger.propagate = propagate


@pytest.fixture
def tmpdir_server(tmpdir):
    Handler = partial(SimpleHTTPRequestHandler, directory=str(tmpdir))
    from http.server import ThreadingHTTPServer

    with ThreadingHTTPServer(("localhost", 0), Handler) as server:
        server_thread = threading.Thread(target=server.serve_forever)
        # Exit the server thread when the main thread terminates
        server_thread.daemon = True
        server_thread.start()
        yield server.server_address
        server.shutdown()


@pytest.fixture
def single_process_pg():
    """Initialize the default process group with only the current process for testing purposes.

    The process group is destroyed when the with block is exited.
    """
    if torch.distributed.is_initialized():
        raise RuntimeError("Can't use `single_process_pg` when the default process group is already initialized.")

    orig_environ = os.environ.copy()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_network_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.distributed.init_process_group("gloo")
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()
        os.environ.clear()
        os.environ.update(orig_environ)

def pytest_collection_modifyitems(items):
    # select special tests, all special tests run standalone
    # note standalone tests take precedence over experimental tests if both env vars are set
    # tests depending on experimental patches do not run in CI by default
    if os.getenv("PL_RUN_STANDALONE_TESTS", "0") == "1":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            # has `@RunIf(standalone=True)`
            if marker.name == "skipif" and marker.kwargs.get("standalone")
        ]
    elif os.getenv("FTS_EXPERIMENTAL_PATCH_TESTS", "0") == "1":
        items[:] = [
            item
            for item in items
            for marker in item.own_markers
            if marker.name == "skipif" and marker.kwargs.get("exp_patch")
        ]
