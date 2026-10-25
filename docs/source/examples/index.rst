.. _examples-overview:

Examples
########

End-to-end examples of scheduled fine-tuning, grouped by how they run: ``notebooks/`` holds the tutorial notebooks
(each rendered here from an executed run, and runnable in Colab or locally) and ``cli/`` the examples driven from the
command line. Each example names the interface it uses; the two interfaces share one schedule format.

Notebooks
*********

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

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
   :widths: 35 20 45

   * - Example
     - Interface
     - What it shows
   * - :ref:`Scheduled fine-tuning on SuperGLUE<scheduled-fine-tuning-superglue>`
     - Lightning (``LightningCLI``)
     - Explicit, implicit and baseline scenarios for RTE and BoolQ, configured entirely in YAML.
