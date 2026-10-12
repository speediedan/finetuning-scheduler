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
"""A persistent pool of distributed workers that serves many test cases from one process group.

Launching processes per test costs far more than most distributed cases take to run. The pool starts ``world_size``
workers once per test module, each with one process group, and sends them cases from ``tests/dist_cases``. Tests
stay ordinary pytest functions: parametrize ids name the case, and assertions run in this process on the plain data
the ranks return. The workers' isolation rules are documented in ``tests/dist_cases/_worker.py``.

This is deliberately not ``torch.testing._internal``'s continuous harness. Importing that module changes global
backend flags and the multiprocessing start method for the whole session, and its 2.15 version lets a skip on one
rank hide a failure on another and skips the rest of a class after one error. Here a failure on any rank fails the
case, and an error, a timeout or a validity violation respawns the pool instead of poisoning the cases after it.

Usage::

    gloo_pool = dist_pool_fixture(world_size=2, backend="gloo")


    def test_sum(gloo_pool):
        assert gloo_pool.run(collectives.all_reduce_sum, n=4) == [[0, 3, 6, 9]] * 2
"""
import atexit
import multiprocessing
import queue
import sqlite3
import sys
import time
import warnings
from pathlib import Path
from typing import Any
from collections.abc import Callable

import pytest

from tests.dist_cases._worker import worker_main

_CASE_PACKAGE = "tests.dist_cases"
_FAILED = ("error", "violation", "init_error", "died", "timeout")
_RESPONSIVE = {"ok", "skip", "error", "violation"}  # replies from workers still serving requests
# Collection imports this module before any test runs, so this is the directory the run started in, whatever a test
# later does with ``os.chdir``. Relative coverage paths are resolved against it.
_INVOCATION_DIR = Path.cwd()


def case_ref(case: Callable | str) -> str:
    """The ``module:qualname`` a worker imports, which must name something in ``tests.dist_cases``."""
    ref = case if isinstance(case, str) else f"{case.__module__}:{case.__qualname__}"
    if not ref.startswith(f"{_CASE_PACKAGE}."):
        raise ValueError(f"pool cases must live in {_CASE_PACKAGE} (workers import no other test code): {ref}")
    return ref


class DistPool:
    """``world_size`` spawned workers sharing one process group, started on first use and reused until closed."""

    def __init__(
        self,
        world_size: int,
        backend: str,
        tmp_dir: Path,
        case_timeout: float = 120.0,
        start_timeout: float = 180.0,
        pg_timeout: float = 120.0,
        grace: float = 5.0,
    ) -> None:
        self.world_size, self.backend, self.tmp_dir = world_size, backend, Path(tmp_dir)
        self.case_timeout, self.start_timeout = case_timeout, start_timeout
        self.pg_timeout, self.grace = pg_timeout, grace
        self.spawn_s: list[float] = []  # wall time of each (re)spawn
        self.stats: list[dict] = []  # per case: wall time in this process, and each rank's case and overhead time
        self._procs: list = []
        self._requests: list = []
        self._responses: Any = None
        self._coverage, self._coverage_spec = _parent_coverage(self.tmp_dir / "coverage")
        atexit.register(self.kill)  # runs before multiprocessing's own exit hook, which would join live workers

    def _start(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        init_file = self.tmp_dir / f"pg_init_{len(self.spawn_s)}"  # a fresh rendezvous file for every spawn
        self._requests = [ctx.Queue() for _ in range(self.world_size)]
        self._responses = ctx.Queue()
        args = (self.world_size, self.backend, str(init_file))
        self._procs = [
            ctx.Process(
                target=worker_main,
                args=(rank, *args, self._requests[rank], self._responses, self.pg_timeout, self._coverage_spec),
                name=f"dist_pool_{self.backend}_rank{rank}",
            )
            for rank in range(self.world_size)
        ]
        start = time.perf_counter()
        for proc in self._procs:
            proc.start()
        replies = self._collect(time.monotonic() + self.start_timeout)
        if any(kind != "ready" for kind, *_ in replies.values()):
            self.kill()
            raise RuntimeError(f"the {self.backend} pool failed to start:\n{_report(replies)}")
        self.spawn_s.append(time.perf_counter() - start)

    def _collect(self, deadline: float) -> dict[int, tuple]:
        replies: dict[int, tuple] = {}
        while len(replies) < self.world_size and time.monotonic() < deadline:
            try:
                rank, *reply = self._responses.get(timeout=0.25)
                replies[rank] = tuple(reply)
                if reply[0] in _FAILED:  # the other ranks may be waiting in a collective that will never complete
                    deadline = min(deadline, time.monotonic() + self.grace)
            except queue.Empty:
                for rank, proc in enumerate(self._procs):
                    if rank not in replies and not proc.is_alive():
                        replies[rank] = ("died", f"worker exited with code {proc.exitcode}", {})
        for rank in range(self.world_size):
            replies.setdefault(rank, ("timeout", "no reply before the deadline (hung in a collective?)", {}))
        return replies

    def run(self, case: Callable | str, /, *, seed: int = 1234, timeout: float | None = None, **kwargs: Any) -> list:
        """Run ``case(ctx, **kwargs)`` on every rank and return the ranks' results, in rank order.

        Fails the calling test if any rank fails, times out or breaks a validity rule, skips it only if every rank
        skipped, and fails it if the ranks disagree about skipping.
        """
        ref = case_ref(case)
        if not self._procs:
            self._start()
        start = time.perf_counter()
        for requests in self._requests:
            requests.put((ref, kwargs, seed))
        replies = self._collect(time.monotonic() + (timeout or self.case_timeout))
        ranks = [replies[rank][2] for rank in range(self.world_size)]
        self.stats.append({"case": ref, "wall_s": time.perf_counter() - start, "ranks": ranks})
        kinds = {kind for kind, *_ in replies.values()}
        if kinds & set(_FAILED):
            # The next case gets fresh workers. Workers that all replied are still serving, so they are stopped
            # cleanly and keep the coverage they recorded for earlier cases; a hung or dead rank means killing them.
            self._stop(timeout=self.grace) if kinds <= _RESPONSIVE else self.kill()
            pytest.fail(f"{ref} failed:\n{_report(replies)}", pytrace=False)
        if kinds == {"skip"}:
            pytest.skip(replies[0][1])
        if "skip" in kinds:
            pytest.fail(f"{ref}: the ranks disagreed about skipping:\n{_report(replies)}", pytrace=False)
        return [replies[rank][1] for rank in range(self.world_size)]

    def pids(self) -> list[int | None]:
        return [proc.pid for proc in self._procs]

    def close(self, timeout: float = 30.0) -> None:
        """Stop the workers, merge the coverage they saved, and release the pool for good."""
        self._stop(timeout)
        atexit.unregister(self.kill)
        self._merge_coverage()

    def _stop(self, timeout: float) -> None:
        """Stop the workers cleanly (so they can destroy the group and save coverage), killing any that hang."""
        if self._procs:
            for requests in self._requests:
                requests.put(None)
            deadline = time.monotonic() + timeout
            for proc in self._procs:
                proc.join(max(0.0, deadline - time.monotonic()))
        self.kill()

    def _merge_coverage(self) -> None:
        """Merge the data the workers saved at exit into this process's coverage, which saves it with its own.

        The workers measure the same absolute paths as this process, so their data is added as it is: ``combine()``
        would remap it through any ``[paths]`` aliases, which it never applies to this process's own data. Under
        ``parallel = true`` the workers' files sit beside this run's own instead, for its ``coverage combine``.
        """
        if self._coverage is None or self._coverage.get_option("run:parallel"):
            return
        from coverage import CoverageData
        from coverage.exceptions import CoverageException, CoverageWarning

        data_dir = Path(self._coverage_spec["data_file"]).parent
        files = sorted(f for f in data_dir.glob(".coverage.*") if not f.name.endswith(("-journal", "-wal", "-shm")))
        if not files:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CoverageWarning)  # "No data was collected": this process may have none yet
            data = self._coverage.get_data()
        for f in files:
            try:
                worker_data = CoverageData(basename=str(f))
                worker_data.read()
                data.update(worker_data)
            except (CoverageException, sqlite3.Error) as err:  # a worker killed mid-save; the other files still merge
                warnings.warn(f"could not merge the pool worker's coverage data {f}: {err}")
                continue
            f.unlink()

    def kill(self) -> None:
        for proc in self._procs:
            if proc.is_alive():
                proc.terminate()
        for proc in self._procs:
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join(5)
        for q in [*self._requests, *([self._responses] if self._responses else [])]:
            q.close()
            q.cancel_join_thread()
        self._procs, self._requests, self._responses = [], [], None


def _parent_coverage(data_dir: Path) -> tuple[Any, dict | None]:
    """This process's running coverage, if any, and the settings that measure the workers the same way.

    Without this, code that runs only in the workers (every case) would be missing from a plain ``coverage run`` of the
    suite, since nothing measures a spawned process by default.
    """
    coverage = sys.modules.get("coverage")
    cov = coverage.Coverage.current() if coverage is not None else None
    if cov is None:
        return None, None
    spec = {opt: cov.get_option(f"run:{opt}") for opt in ("branch", "source", "source_pkgs", "include", "omit")}
    # a worker resolves relative paths against its own working directory, which is this process's at spawn time
    source = [str(_INVOCATION_DIR / s) if (_INVOCATION_DIR / s).is_dir() else s for s in spec["source"] or []]
    spec["source"] = source or None  # a source that is not a directory here names a package
    for opt in ("include", "omit"):
        spec[opt] = [p if p.startswith("*") else str(_INVOCATION_DIR / p) for p in spec[opt] or []] or None
    if cov.get_option("run:parallel"):  # beside this run's own data, for its ``coverage combine``
        spec["data_file"] = str(_INVOCATION_DIR / cov.get_option("run:data_file"))
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
        spec["data_file"] = str(data_dir / ".coverage")
    spec["config_file"] = cov.config.config_file or False
    return cov, spec


def _report(replies: dict[int, tuple]) -> str:
    return "\n".join(f"--- rank {rank}: {reply[0]}\n{reply[1]}" for rank, reply in sorted(replies.items()))


def assert_ranks_agree(values: list) -> Any:
    """Assert every rank returned the same value, and return it."""
    for rank, value in enumerate(values[1:], start=1):
        assert value == values[0], f"rank {rank} disagrees with rank 0: {value!r} != {values[0]!r}"
    return values[0]


def dist_pool_fixture(world_size: int = 2, backend: str = "gloo", **pool_kwargs: Any) -> Callable:
    """A module-scoped fixture providing a :class:`DistPool`; assign it to a module-level name to use it."""

    @pytest.fixture(scope="module")
    def _pool(tmp_path_factory: pytest.TempPathFactory) -> Any:
        pool = DistPool(world_size, backend, tmp_path_factory.mktemp(f"dist_pool_{backend}"), **pool_kwargs)
        yield pool
        pool.close()

    return _pool
