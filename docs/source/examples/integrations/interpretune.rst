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

`interpretune <https://interpretune.org>`_ is a framework for interpretability analysis and tuning of transformer
models across several backends (Hugging Face models, TransformerLens, SAELens, nnsight and others). It uses
Fine-Tuning Scheduler for phased fine-tuning, and it shows what the :ref:`tunable view<tunable-views>` contract is for:
each interpretune module composition declares its own view and attaches it where Fine-Tuning Scheduler finds it, so a
schedule names a model's parameters the way the interpretability backend does, and selects tensors by role.

Both interfaces
===============

An interpretune session runs on one of two hosts, and each drives Fine-Tuning Scheduler through one of its two
interfaces:

- **Lightning.** A Lightning composition uses the :class:`~finetuning_scheduler.fts.FinetuningScheduler` callback,
  as any ``LightningModule`` does.
- **interpretune's own training loop.** The core loop drives
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

What each composition declares
==============================

A composition's view says whether it can be tuned and, if not, why. Compositions that declare nothing are scheduled
through the identity view, so they behave as a plain module would. Others declare their own logical names and roles: a
TransformerLens composition, for example, names its parameters in TransformerLens's own scheme, and an SAELens
composition with SAEs spliced in marks them as ``latent:sae``. Compositions whose parameters cannot be tuned safely,
such as a remote nnsight model, are refused, with the reason.

interpretune maintains the details, since they follow its backends:

- `Tuning <https://interpretune.org/en/latest/usage/fts_tunable_views.html>`_: the view each composition declares,
  phased fine-tuning on the core loop (metrics, lr schedulers, resuming), and the conformance cases interpretune runs
  against each composition.
- `Tuning: core and HF <https://interpretune.org/en/latest/usage/fts_tuning_core_hf.html>`_ and the pages for each
  backend linked from there.
