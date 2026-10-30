
#################
Profiling Example
#################

.. _model-parallel-configurable-profiling:

Profiling Composable Distributed Training
*****************************************

Composable distributed training (especially at scale) can make profiling resource utilization both more important and
challenging. To address this need, FTS includes a utility that enables quick, flexible orchestration of advanced
profiling combining multiple complementary PyTorch profilers.

The :class:`~finetuning_scheduler.profiling.memprofiler.MemProfiler` is a powerful memory profiling utility that synthesizes
numerous complementary profiling methods. As demonstrated in this example, the following profiling utilities are
integrated and simultaneously configured:

- ``FSDP2MemTracker``
- accelerator memory statistics and `allocator history snapshots
  <https://pytorch.org/docs/stable/torch_cuda_memory.html>`_
- host-level memory tracking
- custom memory hooks (e.g. for activation checkpoint memory tracking via ``saved_tensors_hooks`` etc.)

Collection does not assume CUDA: accelerator statistics come from the device-generic ``torch.accelerator`` API, and
allocator history from the current backend's own memory module, so the same configuration profiles CUDA, ROCm and
other accelerators (see :ref:`memprofiler-backend-support`).

The PyTorch Profiler (as well as an  extended version), can also be simultaneously configured to provide a remarkably
complete view of distributed training resource utilization.

.. note::

    While this example focuses on composable distributed training,
    :class:`~finetuning_scheduler.profiling.memprofiler.MemProfiler` can be used with any type of training session.

Profiling Artifacts
*******************

This example uses the following simple profiling configuration to enable extensive profiling of the default method
('training_step') using a default schedule (warmup 1 iteration, collect for 1 iteration) and produces the profiling
artifacts below.

For this ``model_parallel`` example, all artifacts should be nested in the tensorboard log dir under ``fts_examples``/
``model_parallel`` / ``lightning_logs`` / ``fts_fsdp_profiling`` / ``version_{n}``.

1. ``FSDP2MemTracker`` tracking for the collected iteration (per-rank), e.g.:
   ``./memprofiler/fsdp_mem_rank_0_training_step_2.log``, and the same statistics as one JSON report per rank, e.g.:
   ``./memprofiler/rank_0_fsdp_memory_report.json`` (see :ref:`memprofiler-fsdp2-reports`)
2. Consolidated memory snapshot collection entries, dumped to ``.yaml`` by default (can also be dumped to ``*.pickle``
   if preferred).

   - Snapshot entries are provided for each profiled method, 'start' and 'end' of each iteration for each
     of the enabled memory tracking types.
   - By default, keys associated with 'accelerator' (the accelerator allocator's statistics), 'cpu' (host-memory), and
     custom memory hooks are included.
   - The custom hooks are ``saved_tensors_hooks`` in this case, tracking non-parameter packed bytes which is a useful
     proxy for memory associated with activations: ``./memprofiler/rank_1_memory_stats.yaml``
3. Allocator history tracking, start and end of the collected iteration (per-rank), named for the accelerator's device
   type (``cuda`` on both CUDA and ROCm), e.g.: ``./memprofiler/cuda_alloc_rank_0.training_step.2.end.pickle``
4. PyTorch Profiler (extended version), according to the separately provided PyTorch Profiler configuration, e.g.:

   - ``./fit-fts_extended_pytorch_profiler-0[Strategy]ModelParallelStrategy.training_step.{epoch time}.pt.trace.json``
   - ``./fit-fts_extended_pytorch_profiler-0.txt``


Configuration
*************

.. code-block:: yaml

    trainer.profiler: pytorch_profiler.yaml  # enable normal PyTorch Profiler configuration
    # we enable MemProfiler collection for default functions ('training_step') and schedule
    # note any method decorated with @MemProfiler.memprofilable can be specified for collection
    memprofiler_cfg:
        enabled: true
        allocator_history: true  # enable accelerator memory snapshot/allocator history tracking
        track_fsdp_mem: true  # enable FSDP2MemTracker tracking
        fsdp_mem_track_module_depth: 3  # collect up to 3 levels of nested FSDP modules
        fsdp_mem_tracker_tabulate: true  # display FSDP2MemTracker stats in a table
        fsdp_mem_tracker_root_module: 'model'  # the root FSDP module for FSDP2MemTracker to track

.. tip::

    For more advanced usage examples (e.g. non-default schedules, custom collection functions, etc.), see the commented
    configuration options in ``fts_examples/model_parallel/config/profiling/memprofiler_demo.yaml``.

.. note::

    The configuration keys ``cuda_allocator_history``, ``collect_funcs.cuda`` and
    ``collect_funcs.cuda_allocator_history`` are deprecated in favor of ``allocator_history``,
    ``collect_funcs.accelerator`` and ``collect_funcs.allocator_history``. The old names still work, with a
    ``FutureWarning``, until no earlier than 2.17. A deprecated function set is merged into (added to) its
    replacement, and ``cuda_allocator_history`` sets ``allocator_history`` unless ``allocator_history`` is set too, in
    which case ``allocator_history`` wins.

.. _memprofiler-without-lightning:

Profiling Without Lightning
***************************

:class:`~finetuning_scheduler.profiling.memprofiler.MemProfiler` and its configuration live in
:mod:`finetuning_scheduler.profiling`, which imports no Lightning module, so a plain PyTorch training loop can drive it.
The module it profiles exposes its ``memprofiler_cfg`` and the ``memprofiler`` its profiled methods use; methods
decorated with :meth:`~finetuning_scheduler.profiling.memprofiler.MemProfiler.memprofilable` are profiled on the
configured schedule. The loop supplies what a trainer otherwise would, each as a value or as a callable read when it
is needed: the optimizer whose state ``FSDP2MemTracker`` attributes, the directory under which the artifacts are
written (a ``memprofiler`` directory, unless ``memprofiler_cfg.save_dir`` is set) and, optionally, the current training
phase (see :ref:`memprofiler-fsdp2-reports`).

.. code-block:: python

    import torch
    from finetuning_scheduler.profiling import MemProfiler, MemProfilerCfg, MemProfilerFuncs


    class Model(torch.nn.Module):
        def __init__(self, memprofiler_cfg: MemProfilerCfg):
            super().__init__()
            self.layer = torch.nn.Linear(16, 16)
            self.memprofiler_cfg = memprofiler_cfg
            self.memprofiler = None

        @MemProfiler.memprofilable
        def training_step(self, batch):
            return self.layer(batch).sum()


    model = Model(MemProfilerCfg(enabled=True, collect_funcs=MemProfilerFuncs(cpu={"training_step"})))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.memprofiler = MemProfiler()
    model.memprofiler.connect(model, optimizer=optimizer, log_dir="./logs")
    model.memprofiler.init_memprof_log_dir()
    for batch in batches:
        optimizer.zero_grad()
        model.training_step(batch).backward()
        optimizer.step()
    model.memprofiler.dump_memory_stats()

Host memory is tracked with ``psutil``, which is imported only when a profiler connects. The Lightning integration
(``fts_examples.profiling.ProfilerHooksMixin``) connects the same profiler with callables that read its trainer, and
reports the ``FinetuningScheduler`` callback's current depth as the phase.

.. _memprofiler-fsdp2-reports:

FSDP2 Reports
*************

Beside each call's text report, every rank writes ``rank_<n>_fsdp_memory_report.json`` (``dump_fsdp_mem_json``,
enabled by default). All values are bytes and modules are included to ``fsdp_mem_track_module_depth``:

.. code-block:: python

    {
        "version": 1,
        "rank": 0,
        "world_size": 2,
        "backend": "cuda",
        "units": "B",
        # measured in this process by `finetuning_scheduler.profiling.collector_availability()`
        "collectors": {
            "accelerator_memory_stats": True,
            "allocator_history": True,
            "fsdp_mem_tracker": True,
            "host_memory": True,
        },
        "entries": [
            {
                "fn_name": "training_step",
                "iter": 2,
                "phase": 0,
                # this call's peak, per device and category (e.g. "Sharded Param", "Activation", "Total")
                "peak": {"cuda:0": {"Sharded Param": 1048576, "Total": 52428800}},
                # each module's snapshots, per module state (e.g. "Peak-Forward", "Peak-Backward")
                "modules": {"FSDPModel": {"Peak-Backward": [{"cuda:0": {"Total": 52428800}}]}},
            }
        ],
    }

Each entry's ``peak`` is that call's own peak: the tracker resets its peak each time it is entered, so it is recorded
as each profiled call exits.

**Phases.** A fine-tuning phase transition changes which parameters train and can replace the optimizer, whose state the
tracker attributes. When the ``phase`` given to ``connect`` changes, the tracker is rebuilt with the optimizer then
current, and each entry records the phase its call ran in.

**Aggregating across ranks.** :func:`~finetuning_scheduler.profiling.fsdp_report.aggregate_fsdp_mem_reports` reads
every rank's report in a directory and, for each profiled call, reports each peak category (summed over the devices a
rank used) per rank with its maximum, minimum and mean, writing ``fsdp_memory_report_aggregate.json``. The maximum is
the figure that decides whether a configuration fits, since the most loaded rank runs out first. Set
``aggregate_fsdp_mem_reports`` to have rank 0 aggregate at the end of profiling, after every rank has written its report
(each rank waits at a barrier), or call the function afterwards. It refuses, naming what it found, reports whose
version it does not know, a missing or duplicated rank, and ranks that profiled different calls.

.. _memprofiler-backend-support:

Backend Support
***************

Which metrics each backend provides (accelerator statistics also include any key the backend's own
``torch.<device type>.memory_stats`` reports that the device-generic API does not, such as ``num_oom_rejections`` on
CUDA as of PyTorch 2.15):

.. list-table::
    :header-rows: 1
    :widths: 28 18 18 18 18

    * - Metric
      - CUDA
      - ROCm
      - XPU
      - Other accelerators
    * - accelerator statistics (``collect_funcs.accelerator``)
      - ``torch.accelerator.memory_stats``
      - ``torch.accelerator.memory_stats`` (the HIP caching allocator, with the same keys as CUDA)
      - expected (unverified): ``torch.accelerator.memory_stats``
      - where the backend implements ``torch.accelerator.memory_stats``
    * - allocator history (``allocator_history``)
      - ``torch.cuda.memory``
      - ``torch.cuda.memory`` (backed by HIP)
      - expected (unverified): ``torch.xpu.memory``
      - where ``torch.<device type>.memory`` provides it (no other backend does as of PyTorch 2.15)
    * - ``FSDP2MemTracker``
      - yes
      - yes
      - expected (unverified), without the caching allocator's block rounding
      - expected (unverified), without the caching allocator's block rounding
    * - host memory and memory hooks
      - yes
      - yes
      - yes (host-side, independent of the backend)
      - yes (host-side, independent of the backend)

A metric a backend does not provide is skipped with a warning naming the backend rather than failing: allocator history
is disabled when the configuration is created, and accelerator statistics on the first collection that the backend
refuses. Without any accelerator, only host memory, memory hooks and ``FSDP2MemTracker`` are collected. On ROCm builds of
PyTorch the accelerator's device type is ``cuda``, so ROCm artifacts carry ``cuda`` names. The extended PyTorch Profiler
likewise records the accelerator's activity (``ProfilerActivity.CUDA`` on both CUDA and ROCm, and
``ProfilerActivity.XPU`` on XPU where the PyTorch build's profiler records it, unverified) when the profiler supports
it.

The CUDA and ROCm columns are measured by the multi-GPU profiling example's test, which checks every snapshot's
statistics, allocator history snapshots and ``FSDP2MemTracker`` reports on each rank, on NVIDIA and AMD GPUs with
PyTorch 2.15 (a CUDA and a ROCm build). The XPU and other-accelerator columns follow
from the APIs PyTorch provides there and have not been run: an API's presence does not show that a given build
implements it.

Reviewing the Results
*********************

Running the example with the following command, we can explore the detailed resource utilization characterization we now
have available!

.. code-block:: bash

    cd ./fts_examples/model_parallel && \
    python mp_examples.py fit --config config/fts_fsdp_profiling.yaml --config config/profiling/memprofiler_demo.yaml

1. ``FSDP2MemTracker`` tracking data for rank 0:

.. figure:: ../_static/images/fts/fsdp_mem_tracker_report_table.png
   :alt: FSDP2 Mem Tracker Report
   :width: 100%

   FSDP2 Mem Tracker Report for ``training_step`` iteration 2, rank `0`

2. Consolidated accelerator memory snapshot, custom hook ('npp' ``saved_tensors_hooks`` in this case) and host-level memory
   summary  (rank ``0`` in this case):

.. code-block:: yaml

    0.training_step.2.end:
        #...
        # standard accelerator memory snapshot data
        active_bytes.all.allocated: 1948350976
        active_bytes.all.current: 184830976
        active_bytes.all.freed: 1763520000
        active_bytes.all.peak: 330579456
        #...
        active_bytes.small_pool.allocated: 204683776
        ...
        # measured 6.5 MiB of non-parameter packed bytes using our custom `saved_tensors_hooks``
        npp_diff: 6815744
        npp_post_forward: 6815744
        npp_pre_forward: 0
        num_ooms: 0
        #...
        # more standard accelerator memory snapshot data
        reserved_bytes.all.peak: 551550976
        #...
        # host-level memory snapshot data
        rss: 2174173184
        rss_diff: 393216
        rss_post_forward: 2173976576
        rss_pre_forward: 2173583360
        #...
        vms: 19357667328
    0.training_step.2.start:
        #...
        active_bytes.all.allocated: 1281527296
        #...

3. Allocator history visualization (generate by dragging the desired ``*.pickle`` file to
https://pytorch.org/memory_viz)

.. figure:: ../_static/images/fts/cuda_allocation_active_timeline.png
   :alt: Cuda Allocator History Active Timeline
   :width: 100%

   Cuda Allocator History Active Timeline for ``training_step`` iteration 2,  rank ``0``, start snapshot

4. PyTorch Profiler

.. figure:: ../_static/images/fts/pytorch_profiler_tboard.png
   :alt: PyTorch profiler tensorboard visualizations
   :width: 100%

   PyTorch profiler tensorboard visualizations

.. figure:: ../_static/images/fts/pytorch_profiler_txt_summary.png
   :alt: PyTorch profiler textual summary rank ``0``
   :width: 100%

   PyTorch profiler textual summary rank ``0``
