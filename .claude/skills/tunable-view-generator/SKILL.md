---
name: tunable-view-generator
description: Write, check and use a finetuning-scheduler tunable view for a model that has none, so schedules can address it by role, layer and component (venngram addresses) rather than by parameter names and regexes. Use when an experiment needs a schedule over adapters, latent models or architecture-neutral components, when one schedule must run on several model families, or when asked to "make a view" or "generate a TunableView" for a model. Do NOT use when a schedule of names or regexes already says what is meant.
---

# Generating a tunable view

A tunable view (`finetuning_scheduler.core.views.TunableView`, protocol from `venngram.tuning`) tells finetuning-scheduler
what a model's tensors are: their roles (`adapter`, `latent:<name>`, `instrument`, ...), where they sit (layers), and
optionally what they are (architecture-neutral components such as `attn.q`, `mlp.down`). Schedules can then use
addresses such as `role:adapter & layers:-4:` or `component:mlp.* & layers:-1:`, which mean the same thing on every
model the view describes.

The reference for everything below is the worked example: `src/fts_examples/tunable_views/` (docs page
`docs/source/examples/tunable_views/worked_example.rst`). Read its `decoder_view.py` before writing a view.

## First: do you need one?

Ask before writing anything. Writing a view is more work than writing a regex.

- **No view needed** when a schedule of parameter names and regexes, or the generated default schedule, says what is
  meant: one model, a hand-written schedule, thawing layers top-down. Names and regexes stay supported and never warn.
- **No view class needed** when only roles or layers are wanted. `CanonicalView(model, roles={name: role})` keeps the
  model's names and adds roles, and the default `structure()` already places tensors at layers for any model with one
  top-level `nn.ModuleList`. That makes `role:adapter & layers:-2:` work.
- **A view class** is needed for architecture-neutral components (`component:`), for logical names that differ from the
  physical ones (a wrapper's vocabulary), for coupled tensors thawed together, for a model with several block lists, or
  for latent models and promotion.

## Steps

1. **Inspect the model.**

   ```bash
   python .agents/skills/tunable-view-generator/scripts/inspect_model.py --model pkg.module:make_model
   python .agents/skills/tunable-view-generator/scripts/inspect_model.py --hf-arch GPT2LMHeadModel
   ```

   It prints the block list (or why there is none, or several), one block's tensors with a first guess at each
   component, the tensors outside the blocks, and likely adapter tensors. **The guesses come from module names: confirm
   each against the model's `forward`** before trusting it. A fused projection (`c_attn`, `query_key_value`) is one
   tensor: name it as one component (`attn.qkv`), since a view cannot split a tensor.

2. **Choose the view's shape.** In order of preference:
   - `CanonicalView(model, roles=...)`: roles only.
   - A `CanonicalView` subclass overriding `structure()`: roles plus components, keeping the model's own names. This is
     the worked example's `DecoderView`. Reuse its `DecoderStructure`, giving it a table of `(module-path regex,
     component)` rows, first match wins.
   - A full view (`root`, `support()`, `to_physical()`, `to_logical()`, plus `logical_named_parameters()` when the
     naming scheme is not `"physical"`): only for a logical vocabulary that differs from the model's names. Inherit
     `ViewDefaults` for the optional members. See "Writing a View" in `docs/source/pytorch/tunable_views.rst`.

3. **Assign roles.** Adapters (LoRA `lora_A`/`lora_B`, IA3, prefix tensors) are `adapter` (or `adapter:<name>` for
   separately scheduled groups). Spliced latent models are `latent:<name>`. Fitted, frozen analysis tensors are
   `instrument`. Everything unlisted is `base`. Prefer one role plus addresses over many named sub-roles: a subset of
   a role is an address (`role:adapter & layers:-4:`), not a new role.

4. **Write the component table** (if components are wanted). Use the vocabulary the worked example uses: `attn.q`,
   `attn.k`, `attn.v`, `attn.qkv`, `attn.o`, `mlp.gate`, `mlp.up`, `mlp.down`, `norm.attn`, `norm.mlp`. Only tensors
   inside a block get a component.

5. **Check it.**

   ```bash
   python .agents/skills/tunable-view-generator/scripts/check_view.py --view pkg.module:make_view --schedule my_schedule.yaml
   ```

   `make_view` returns a fresh view over a fresh model each call. The script runs `TunableViewConformance` (every
   protocol case and every case finetuning-scheduler relies on), then compiles the schedule and prints each phase's
   tensors. **Read the printed tensors**: a schedule that compiles can still select the wrong ones. Iterate until
   conformance passes and every phase is what was meant.

6. **Write the schedule's sets once**, in an address book (`schema_version: 1`, `portability: family`, `sets:`), and
   reference them from the schedule (`addresses: <book>.yaml`, entries `@<set>`). See
   `src/fts_examples/tunable_views/config/` for both files.

7. **Use it.** PyTorch API: `TuningScheduler(model, schedule, optimizer=..., view=view)`. Lightning: attach it as
   `pl_module.tunable_view = view` (rooted at the `LightningModule`).

## Gotchas

- The first schedule in a process that uses an address warns once that addresses are experimental. That is expected.
- A view's members are methods: `roles` as a dictionary attribute, rather than a method returning one, is refused.
- A model with several top-level block lists (an encoder and a decoder, a vision tower beside a language model) makes
  the default structure refuse `layers`: pass `layers_pattern` to `PhysicalStructure`, or locate layers yourself.
- Do not put doses, learning rates or other run parameters in an address book. It names sets, nothing else.

## Reference output

`references/gpt2_view.py` is a view made with these steps for GPT-2, a model with no view of its own (fused `c_attn`,
`Conv1D` layers), and `references/gpt2_schedule.yaml` the schedule `check_view.py` compiles through it.
`src/fts_examples/test_examples.py` runs that check, so the reference stays correct.
