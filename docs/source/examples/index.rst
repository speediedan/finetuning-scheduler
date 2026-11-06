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
     - Tunable views and addresses in an experiment: give Qwen3-0.6B a view (``venngram.view(model)``: adapter
       roles, layers and architecture-neutral components), check it, and run attention-only,
       MLP-only, all-layer, MLP-plus-norms and phased LoRA as address schedules over the same model, probing a 2025
       result on where LoRA adapters belong.
   * - :doc:`Addressing across architectures<notebooks/tunable_views/cross_architecture_addresses>`
     - PyTorch (experimental)
     - Portable addresses and schedules: the LoRA placement conditions, unchanged, compiled on four model families
       (Qwen3, Llama, OPT, GPT-2) built on the meta device from their configurations alone, showing what each address
       resolves to per family, the refusal of an address a family cannot provide, and each condition's parameter counts
       side by side: an experiment defined once and orchestrated across models. No training; runs on a CPU.
   * - :doc:`The selective pirate: Matryoshka attribution<notebooks/tunable_views/selective_pirate>`
     - PyTorch and Lightning
     - Row masks over a fixed LoRA delta, from the Matryoshka Attribution paper (2026): global randomized budgets, one
       run definition on both interfaces, score tensors thawed first by name regex and then by role, layer and component
       addresses, with matched one-phase, random and gradient controls. The shortened run trains a labeled smoke model.
       The :ref:`pretrained Qwen3-4B results <selective-pirate-results>` record what survives and what does not.
   * - :doc:`The goblin diplomat: Persona Hierarchy<notebooks/tunable_views/persona_hierarchy>`
     - PyTorch and Lightning
     - A bounded goblin-mention comparison from the Persona Hierarchy paper (2026): context-specific mentions with
       and without a default-context warm-up, one run definition on both interfaces, a first phase by name regex
       and a second by role, layer and component addresses, and an explicitly labeled phased neuron-mask extension
       with matched one-phase, random and gradient controls. The shortened run trains a labeled smoke model.
       The :ref:`pretrained Qwen3-4B results <persona-hierarchy-pilot>` record what survives and what does not.

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
