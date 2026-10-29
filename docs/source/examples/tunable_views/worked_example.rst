.. _tunable-views-worked-example:

#################################################
Worked Example: From No View to Address Schedules
#################################################

This example takes a model with no tunable view, gives it one, checks it, and then schedules LoRA tuning by address.
The same schedule file runs on two model families. It runs on CPU in seconds: the models are small and randomly
initialized from ``transformers`` configurations, so nothing is downloaded.

The source is in ``src/fts_examples/tunable_views/``. Run it with:

.. code-block:: bash

    python -m fts_examples.tunable_views.worked_example --family llama   # or --family opt

.. note::
   For one model and a schedule you write by hand, names and regexes (step 2) are usually all you need, and they stay
   fully supported. The view earns its keep from step 3 on: one schedule for several model families, sets defined
   once and shared, and a model description an agent can compose over. See :ref:`fts-evolution`.

1. The model
************
A Llama-style or OPT-style decoder with four blocks. LoRA adapters are added to the query and value projections, so
each target ``nn.Linear`` becomes a ``base`` layer plus a zero-initialized low-rank update:

.. literalinclude:: ../../../../src/fts_examples/tunable_views/decoder_view.py
   :language: python
   :pyobject: LoRALinear

The two families name the same parts differently. The Llama-style MLP is ``model.layers.3.mlp.up_proj``, the OPT-style
one ``model.decoder.layers.3.fc1``. The norm after attention is ``post_attention_layernorm`` in one and
``final_layer_norm`` in the other.

2. The classic schedule
***********************
Without a view, a schedule names parameters by name or regex. This one thaws the top two blocks' adapters, then the
rest, then the last block's MLP and its norm:

.. literalinclude:: ../../../../src/fts_examples/tunable_views/config/llama_regex_schedule.yaml
   :language: yaml

It is short and needs nothing else. But it encodes the model's depth (``[23]``, ``3``) and its module names, so an
OPT-style model, or a deeper Llama, needs a schedule of its own.

3. The view
***********
:class:`~finetuning_scheduler.core.views.CanonicalView` already places every tensor at its layer: the default structure
finds the model's one block list. So ``layers:-2:`` works on any decoder with no view class at all, and
``CanonicalView(model, roles=...)`` is often enough. Here the view adds two things:

- **roles**: every LoRA tensor is an ``adapter``, everything else ``base``;
- **components**: a small table per family maps module paths to architecture-neutral names (``attn.q``, ``mlp.up``,
  ``norm.mlp``, ...), reported through ``structure()``.

.. literalinclude:: ../../../../src/fts_examples/tunable_views/decoder_view.py
   :language: python
   :start-at: COMPONENTS: dict
   :end-before: class LoRALinear

.. literalinclude:: ../../../../src/fts_examples/tunable_views/decoder_view.py
   :language: python
   :pyobject: DecoderStructure

.. literalinclude:: ../../../../src/fts_examples/tunable_views/decoder_view.py
   :language: python
   :pyobject: DecoderView

The view keeps the model's own names (its naming scheme is ``"physical"``), so the classic schedule of step 2 still
works through it unchanged.

4. Checking the view
********************
:class:`~finetuning_scheduler.testing.conformance.TunableViewConformance` runs every protocol case and every case
Fine-Tuning Scheduler relies on (compiling a schedule, thawing, an optimizer step, refusals). It takes a factory, since
the cases change parameters:

.. code-block:: python

    from finetuning_scheduler.testing import TunableViewConformance

    report = TunableViewConformance(lambda: DecoderView(build_model("llama"), "llama")).run()
    assert report.ok, report.failures

5. The address schedule
***********************
The named sets live in an address book, a plain data file that can be shared and imported:

.. literalinclude:: ../../../../src/fts_examples/tunable_views/config/decoder_lora_sets.yaml
   :language: yaml

The schedule references them:

.. literalinclude:: ../../../../src/fts_examples/tunable_views/config/decoder_address_schedule.yaml
   :language: yaml

Through ``DecoderView``, this one file thaws the same parts of either family. On the Llama-style model it thaws
exactly what the regex schedule of step 2 does, which the example checks. On the OPT-style model, phase 2 adds
``fc1``, ``fc2`` and the block's ``final_layer_norm``, the same components under other names. The first schedule in a
process that uses an address warns once that addresses are experimental.

6. Running it
*************
``run()`` (in ``worked_example.py``) passes the schedule to
:class:`~finetuning_scheduler.core.tuning_scheduler.TuningScheduler` with ``view=``, and trains for three epochs on
random tokens, one phase per epoch. Its summary lists each epoch's phase and trainable parameters: 8 adapter tensors
in phase 0, all 16 in phase 1, and then the last block's MLP and norm as well.

With the Lightning callback, attach the view to the ``LightningModule`` as ``tunable_view`` instead (see
:ref:`tunable-views`), and pass the same schedule file.

Doing This for Your Own Model
*****************************
The steps are the same for any model: inspect its parameter tree, decide which roles and components a schedule needs
to name, write the view (often only ``CanonicalView`` with ``roles=``, plus a component table if you want
architecture-neutral components), check it with the conformance suite, then write the schedule's sets once.

An agent can do this for you. Point it at this page and the view's source, for example:

.. code-block:: text

    Write a finetuning-scheduler tunable view for <model>, following the worked example at
    src/fts_examples/tunable_views (docs: examples/tunable_views/worked_example). Give its LoRA tensors the `adapter`
    role and report components for its attention, MLP and norms. Check it with TunableViewConformance, then write an
    address book with the sets <...> and a schedule that uses them.

The ``tunable-view-generator`` agent skill packages these steps, with a script that inspects a model and one that
checks a view and the schedule compiled through it. It is in the repository's ``.agents/skills/``, usable from any
agent harness that reads skills, and its reference output is a view for GPT-2 made with it.
