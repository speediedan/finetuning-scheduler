---
name: tunable-view-generator
description: Build and check the tunable view a finetuning-scheduler schedule needs to select tensors by role, layer, and component. Use for schedule compilation, adapter and artifact roles, or configuring a TunableView. For authoring the architecture's base map, use venngram-component-map.
---

# Finding or generating a tunable view

A tunable view (protocol in `venngram.tuning`) tells finetuning-scheduler what a model's tensors are: their roles
(`adapter`, `latent:<name>`, `artifact`, ...), their layers, and their architecture-neutral components (`attn.q`,
`mlp.out`, `ln2`, ...). Schedules can then use addresses such as `role:adapter & layers:-4:` or
`component:mlp.* & layers:-1:`, which mean the same tensors on every architecture the view describes.

For an architecture that has a verified base map, `venngram.view(model)` builds the view. Use the
[venngram-component-map skill](../venngram-component-map/SKILL.md) to find, draft, verify or propose the base map.

The LoRA placement notebook at `src/fts_examples/notebooks/dev/tunable_views/lora_placement.ipynb` runs
`venngram.view` on Qwen3-0.6B. The synthetic decoder and map under `references/` exercise the existing local
map helpers in the example suite.

## First: do you need one?

- **No view** when a schedule of parameter names and regexes, or the generated default schedule, says what is meant:
  one model, a hand-written schedule, thawing layers top-down. Names and regexes stay supported and never warn.
- **`CanonicalView(model, roles={name: role})`** when only roles or layers are wanted: it keeps the model's names and
  finds layers from its one top-level block list.
- **`venngram.view(model)`** (this skill) for components, or to run one schedule over several model families.
- **A view of your own** for a vocabulary that differs from the model's own names (a wrapper's names, say); see
  "Writing a View" in `docs/source/pytorch/tunable_views.rst`.

## Steps

1. **Build the view.** `venngram.view(model)` composes it from the architecture's base map and the *overlay map* of
   each library whose modules the model carries (an overlay gives that library's tensors their roles): the seed
   registry's `peft.lora` covers PEFT's LoRA names, so a LoRA model needs nothing more. For a library no registry
   covers, write and register an overlay map (venngram's README, "Maps: base and overlay"); never hard-code a library
   in the view.

   ```python
   import venngram
   from venngram.maps import load_map, register

   register(load_map("my_map.yaml"))      # only for an architecture venngram does not already know
   register(load_map("my_overlay.yaml"))  # only for an adapter library no registry covers
   view = venngram.view(model)
   ```

   Prefer one role plus addresses over many named sub-roles: a subset of a role is an address
   (`role:adapter & layers:-4:`), not a new role.

2. **Check the view with a schedule.** `make_view` returns a fresh view over a fresh model each call:

   ```bash
   python .agents/skills/tunable-view-generator/scripts/check_view.py --view pkg.module:make_view --schedule my_schedule.yaml
   ```

   It runs `TunableViewConformance`, compiles the schedule and prints each phase's tensors. **Read them**: a schedule
   that compiles can still select the wrong tensors.

3. **Write the schedule's sets once**, in an address book (`schema_version: 1`, `portability: family`, `sets:`), and
   reference them from the schedule (`addresses: <book>.yaml`, entries `@<set>`). See
   `src/fts_examples/tunable_views/config/`. venngram's `venngram-addressing` skill covers the grammar.

4. **Use it.** PyTorch API: `TuningScheduler(model, schedule, optimizer=..., view=view)`. Lightning: attach it as
   `pl_module.tunable_view = view` (rooted at the `LightningModule`).

## Gotchas

- The first schedule in a process that uses an address warns once that addresses are experimental. That is expected.
- A view's members are methods: `roles` as a dictionary attribute, rather than a method returning one, is refused.
- Do not put doses, learning rates or other run parameters in a map or an address book. They name structure and sets,
  nothing else.
