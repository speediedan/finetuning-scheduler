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
   * - :doc:`Scheduled fine-tuning for math reasoning<notebooks/gsm8k/fts_gsm8k_lightning>`
     - Lightning
     - An explicit schedule (top decoder blocks, then LoRA adapters addressed by role, then the lower blocks) against
       a baseline for Qwen3-0.6B on GSM8K, measuring what the model retains as it tunes, with the
       :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback.
   * - :doc:`Scheduled fine-tuning for math reasoning<notebooks/gsm8k/fts_gsm8k_pytorch>`
     - PyTorch (experimental)
     - The same experiment in a training loop you write yourself, with
       :class:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler`, and a review of the runs' training
       trajectory.

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
