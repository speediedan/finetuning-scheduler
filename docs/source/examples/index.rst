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
   * - :doc:`Where should LoRA go? Probing LoRA Without Regret with addresses<notebooks/tunable_views/lora_placement>`
     - PyTorch (experimental)
     - The tunable views worked example: give Qwen3-0.6B one view with no view code (venngram's ``MappedView``:
       adapter roles, layers and architecture-neutral components from the model's component map), check it, and run
       attention-only, MLP-only, all-layer, MLP-plus-norms and phased LoRA as address schedules over the same model,
       probing a 2025 result on where LoRA adapters belong.

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
