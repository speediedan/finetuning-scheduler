..  Licensed under the Apache License, Version 2.0 (the "License");
..  you may not use this file except in compliance with the License.
..  You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
..  Unless required by applicable law or agreed to in writing, software
..  distributed under the License is distributed on an "AS IS" BASIS,
..  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
..  See the License for the specific language governing permissions and
..  limitations under the License.

.. _example-interpretune:

Example: Scheduled Fine-Tuning In interpretune
**********************************************

.. warning::
   This example uses the experimental PyTorch API and tunable views (see the
   :ref:`API stability classifications<versioning>`).

.. note::
   interpretune is pre-MVP and evolving rapidly. This page shows the shape of the integration rather than a stable
   API; interpretune's own documentation, linked below, follows its current state.

`interpretune <https://interpretune.org>`_ is an AI interpretability research framework that strives to enable
collaborative model analysis and tuning across many existing interpretability frameworks (Hugging Face models,
TransformerLens, SAELens, nnsight and others), with arbitrary architectures in view through hub adapters and component
compositions. It uses Fine-Tuning Scheduler for flexible fine-tuning, and it shows what the
:ref:`tunable view<tunable-views>` contract is for: each interpretune module composition declares its own view and
attaches it where Fine-Tuning Scheduler finds it, so a schedule names a model's parameters the way the
interpretability backend does, and selects tensors by role.

Both interfaces
===============

An interpretune session drives Fine-Tuning Scheduler through either of its two interfaces:

- **Lightning.** A Lightning composition uses the :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback,
  as any ``LightningModule`` does.
- **An interpretune training loop.** The core loop drives
  :class:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler` at its boundaries through
  ``interpretune.tuning.core_loop.FTSCoreCallback``, which builds the tuning scheduler over the session's module when
  training starts:

.. code-block:: python

    from interpretune.config import SessionRunnerCfg
    from interpretune.tuning.core_loop import FTSCoreCallback

    # phase 0 trains the module's default PEFT adapter, selected by role; phase 1 also thaws the LM head
    schedule = {
        0: {"params": ["role:adapter:default"], "max_transition_epoch": 2},
        1: {"params": ["model.lm_head.weight"]},
    }
    runner_cfg = SessionRunnerCfg(it_session=session, max_epochs=4, callbacks=[FTSCoreCallback(schedule)])

The schedule is resolved through the module's view, so ``role:adapter:default`` reaches the adapter's tensors wherever
the backend keeps them. The tuning scheduler builds its optimizer and lr schedulers from the module's own factories
unless ``tuning_scheduler_kwargs`` passes them, and transition policies, checkpoint policies and ``max_depth`` pass
through the same argument.

A runnable example
==================

``fts_examples/interpretune_phased_tuning.py`` runs flexible fine-tuning through an interpretune training loop, on a CPU
in seconds and with no downloads. A small, randomly initialized GPT-2 class model is tuned in three phases by
``FTSCoreCallback``: the first phase thaws only a steering vector registered beside the model, selected by its role
(``role:intervention:steer``) through the tunable view interpretune attaches, and later phases add the model's blocks.

.. code-block:: bash

    python src/fts_examples/interpretune_phased_tuning.py

It needs an interpretune revision that is not yet released: the one pinned in ``requirements/ci/interpretune.txt``,
which Fine-Tuning Scheduler's CI installs to test it.

What each composition declares
==============================

A composition's view says whether it can be tuned and, if not, why. Compositions that declare nothing are scheduled
through the identity view, so they behave as a plain module would. Others declare their own logical names and roles: a
TransformerLens composition, for example, names its parameters in TransformerLens's own scheme, and an SAELens
composition with SAEs spliced in marks them as ``latent:sae``. Compositions whose parameters cannot yet be tuned safely,
such as a remote nnsight model, are refused, with the reason.

interpretune maintains the details, since they follow its backends:

- `Tuning <https://interpretune.org/en/latest/usage/fts_tunable_views.html>`_: the view each composition declares,
  flexible fine-tuning on the core loop (metrics, lr schedulers, resuming), and the conformance cases interpretune runs
  against each composition.
- `Tuning: core and HF <https://interpretune.org/en/latest/usage/fts_tuning_core_hf.html>`_ and the pages for each
  backend linked from there.
