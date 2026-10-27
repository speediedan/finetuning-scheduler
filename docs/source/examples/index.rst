.. _examples-overview:

Overview
########

End-to-end examples of scheduled fine-tuning, grouped by how they run: ``notebooks/`` holds the tutorial notebooks
(each rendered here from an executed run, and runnable in Colab or locally), ``cli/`` the examples driven from the
command line, and ``integrations/`` libraries built on Fine-Tuning Scheduler. Each example names the interface it uses;
the two interfaces share one schedule format.

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

Integrations
************

.. list-table::
   :header-rows: 1
   :widths: 30 18 52
   :class: fts-examples-table

   * - Example
     - Interface
     - What it shows
   * - :ref:`Scheduled fine-tuning in interpretune<example-interpretune>`
     - Lightning and PyTorch (experimental)
     - An AI interpretability research framework that declares a tunable view for each model composition, so
       schedules use the backend's own parameter names and select adapters and latents by role.
