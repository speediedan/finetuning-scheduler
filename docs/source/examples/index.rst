.. _examples-overview:

Overview
########

End-to-end examples of scheduled fine-tuning, grouped by how they run: ``notebooks/`` holds the tutorial notebooks
(each rendered here from an executed run, and runnable in Colab or locally) and ``cli/`` the examples driven from the
command line. Each example names the interface it uses; the two interfaces share one schedule format.

Notebooks
*********

.. list-table::
   :header-rows: 1
   :widths: 30 18 52
   :class: fts-examples-table

   * - Example
     - Interface
     - What it shows
   * - :doc:`Scheduled fine-tuning on SuperGLUE<notebooks/superglue/fts_superglue_lightning>`
     - Lightning
     - Explicit, implicit (default) and baseline schedules for DeBERTa-v3 on RTE with the
       :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback.
   * - :doc:`Scheduled fine-tuning on SuperGLUE<notebooks/superglue/fts_superglue_pytorch>`
     - PyTorch (experimental)
     - The same experiment in a training loop you write yourself, with
       :class:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler`.

Command Line
************

.. list-table::
   :header-rows: 1
   :widths: 30 18 52
   :class: fts-examples-table

   * - Example
     - Interface
     - What it shows
   * - :ref:`Scheduled fine-tuning on SuperGLUE<scheduled-fine-tuning-superglue>`
     - Lightning (``LightningCLI``)
     - Explicit, implicit and baseline scenarios for RTE and BoolQ, configured entirely in YAML.
   * - Phased fine-tuning in an interpretune session (``fts_examples/interpretune_phased_tuning.py``)
     - PyTorch (experimental), in `interpretune <https://interpretune.org>`_
     - A small GPT-2 class model tuned in three phases by interpretune's core training loop, whose
       ``FTSCoreCallback`` drives a
       :class:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler`. The first phase selects a steering
       vector registered beside the model by its role (``role:intervention:steer``) through the tunable view
       interpretune attaches. Runs on a CPU in seconds, with no downloads; it needs an interpretune revision that
       is not yet released (the one in ``requirements/ci/interpretune.txt``).
