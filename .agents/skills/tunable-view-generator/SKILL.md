---
name: tunable-view-generator
description: Find, verify or create the tunable view a finetuning-scheduler schedule needs to address a model by role, layer and component (venngram addresses) rather than by parameter names and regexes. For most models the view is venngram's MappedView over a verified component map, so this looks up the architecture's map, verifies it deterministically, and drafts and verifies one when none exists. Use when an experiment needs a schedule over adapters, latent models or architecture-neutral components, when one schedule must run on several model families, or when asked to "make a view", "generate a TunableView" or "map a model". Do NOT use when a schedule of names or regexes already says what is meant.
---

# Finding or generating a tunable view

A tunable view (protocol in `venngram.tuning`) tells finetuning-scheduler what a model's tensors are: their roles
(`adapter`, `latent:<name>`, `instrument`, ...), their layers, and their architecture-neutral components (`attn.q`,
`mlp.out`, `ln2`, ...). Schedules can then use addresses such as `role:adapter & layers:-4:` or
`component:mlp.* & layers:-1:`, which mean the same tensors on every architecture the view describes.

For an architecture with a component map, venngram's `MappedView(model, role_rules=...)` is the view. So the work is
almost always: find the map, verify it, and only if there is none, draft one and verify that. A map is data (a short
YAML document), so it is safe to share.

References: the LoRA placement notebook `src/fts_examples/notebooks/dev/tunable_views/lora_placement.ipynb` (under
Examples in the docs) runs `MappedView` on Qwen3-0.6B with LoRA. `references/synthetic_decoder.py` is a small decoder
with deliberately unusual module names, so no map covers it, and `references/synthetic_map.yaml` is its map: drafted
by this skill, completed by hand where the draft could not place a module (those rows are marked), and verified.

## First: do you need one?

- **No view** when a schedule of parameter names and regexes, or the generated default schedule, says what is meant:
  one model, a hand-written schedule, thawing layers top-down. Names and regexes stay supported and never warn.
- **`CanonicalView(model, roles={name: role})`** when only roles or layers are wanted: it keeps the model's names and
  finds layers from its one top-level block list.
- **`MappedView`** (this skill) for components, or to run one schedule over several model families.
- **A view of your own** for a vocabulary that differs from the model's own names (a wrapper's names, say); see
  "Writing a View" in `docs/source/pytorch/tunable_views.rst`.

## Steps

1. **Look up and verify the map.** No weights are downloaded: a configuration builds the model on the meta device.

   ```bash
   python .agents/skills/tunable-view-generator/scripts/find_view.py --hf Qwen/Qwen3-0.6B
   python .agents/skills/tunable-view-generator/scripts/find_view.py --hf-arch Qwen3ForCausalLM
   ```

   It runs venngram's four checks: **exists** (every row's module is there), **partition** (every parameter has
   exactly one component), **shapes** (widths match the configuration, so swapped `attn.q`/`attn.k` fail under
   grouped-query attention) and **order** (components run in the expected order on a small random model, so swapped
   `ln1`/`ln2` fail). `verified` or `partial` (what the rules could not reach is printed) means use the map. Exit 2
   means there is no map: go to step 2.

2. **No map: draft one, then verify it.**

   ```bash
   python .agents/skills/tunable-view-generator/scripts/draft_map.py --hf-arch MyArchForCausalLM -o my_map.yaml
   python .agents/skills/tunable-view-generator/scripts/find_view.py --hf-arch MyArchForCausalLM --map my_map.yaml
   ```

   For a model that is not a `transformers` one, pass a factory instead (`--model path/to/file.py:make`); for
   `find_view.py` its module also supplies `CONFIG` (the widths) and `inputs(model)` (one forward, for the order
   check), as `references/synthetic_decoder.py` does:

   ```bash
   python .agents/skills/tunable-view-generator/scripts/find_view.py \
     --model .agents/skills/tunable-view-generator/references/synthetic_decoder.py:make \
     --map .agents/skills/tunable-view-generator/references/synthetic_map.yaml
   ```

   `draft_map.py` is a thin command line over venngram's drafter (`venngram.maps.drafter`), which names modules by
   rules mined from every map venngram knows, so it improves as maps are added, then by bundled rule documents. Each
   row it names carries a **confidence** (the share of known maps that agree on that name) and, where they disagreed,
   the **alternatives**; a module no rule names becomes a `draft.<path>` row marked `UNCONFIRMED`. Review the least
   certain rows first, give every unconfirmed row its component (or confirm it has none), and **treat every row as a
   guess**: read the model's `forward` for anything unusual. Fix what verification names, and re-run until it passes.
   `references/synthetic_decoder.py` is the extreme case: its names are novel, so only the block stack is drafted and
   every other row is named by hand (`references/synthetic_map.yaml`). Conventions:
   - use the core vocabulary (`embed`, `pos_embed`, `ln1`, `attn`, `attn.q`, `attn.k`, `attn.v`, `attn.qkv`, `attn.o`,
     `attn.q_norm`, `attn.k_norm`, `ln1_post`, `ln2`, `mlp`, `mlp.gate`, `mlp.in`, `mlp.out`, `ln2_post`, `ln_final`,
     `unembed`) wherever a module plays that part;
   - a fused tensor is one component (`attn.qkv`; `mlp.gate_in` for a fused gate and up projection): a view cannot
     split a tensor;
   - a map may state its own widths and orders in a `verification:` section where the core rules do not fit (venngram's
     `gpt2.yaml` does, for a null `n_inner`).

3. **Register the verified map, and share it.** A map is data, so a verified one is worth keeping and sharing:
   - **In this process:** `register(load_component_map("my_map.yaml"))`, then `MappedView(model)` finds it (registered
     maps take precedence over venngram's own).
   - **Keep the evidence:** `find_view.py ... --record my_arch.json` writes the verification record (see "Recording a
     trial" below). A shared map should travel with its record.
   - **Share it:** put the map (kind `component_map`) and its record (kind `verification_record`) in a directory and
     publish them with `venngram.store.publish(store, ref, root, documents=[(kind, path), ...])` to a local directory
     or a Hub repository or bucket; it writes a manifest listing each document's hash. Check the result the way a
     reader meets it: `python -m venngram.conformance <store>:<ref>`. Publishing to a remote store is an outward act:
     **an agent asks before it publishes**, and names the store and reference it will write.
   - **Propose it to a registry:** a registry indexes verified maps by content-checked reference, and is rebuilt by
     its crawler from a list of manifest locations and, when built with `--discover-hub`, every Hub repository
     tagged `venngram` (`python -m venngram.registry build --source <location> --discover-hub --out ./index`).
     Publish the map and its record to a Hub repository tagged `venngram`, or give the pinned reference
     (`venngram.store.pin(...)`) to whoever maintains the registry, so its next build carries a row for it. Check what
     a lookup serves with `python -m venngram.registry lookup <Architecture>`; `VENNGRAM_REGISTRIES` names the
     registries consulted, in order (by default venngram's seed registry).

4. **Build the view.** `role_rules` names the adapter library's tensors (`PEFT_LORA_RULES` covers PEFT's LoRA names):

   ```python
   from venngram.maps import MappedView, PEFT_LORA_RULES, load_component_map, register

   register(load_component_map("my_map.yaml"))  # only for a map venngram does not already know
   view = MappedView(model, role_rules=PEFT_LORA_RULES)
   ```

   Prefer one role plus addresses over many named sub-roles: a subset of a role is an address
   (`role:adapter & layers:-4:`), not a new role.

5. **Check the view with a schedule.** `make_view` returns a fresh view over a fresh model each call:

   ```bash
   python .agents/skills/tunable-view-generator/scripts/check_view.py --view pkg.module:make_view --schedule my_schedule.yaml
   ```

   It runs `TunableViewConformance`, compiles the schedule and prints each phase's tensors. **Read them**: a schedule
   that compiles can still select the wrong tensors.

6. **Write the schedule's sets once**, in an address book (`schema_version: 1`, `portability: family`, `sets:`), and
   reference them from the schedule (`addresses: <book>.yaml`, entries `@<set>`). See
   `src/fts_examples/tunable_views/config/`. venngram's `venngram-addressing` skill covers the grammar.

7. **Use it.** PyTorch API: `TuningScheduler(model, schedule, optimizer=..., view=view)`. Lightning: attach it as
   `pl_module.tunable_view = view` (rooted at the `LightningModule`).

## Recording a trial

When you map a new architecture, keep the evidence: `find_view.py --record <arch>.json` writes the verification record
(every check, the rule sets applied, the observed order, the configuration's fingerprint, venngram's version, the
date). A record lets a later run be compared against this one, and is what a registry entry is made from.

## Gotchas

- The first schedule in a process that uses an address warns once that addresses are experimental. That is expected.
- A view's members are methods: `roles` as a dictionary attribute, rather than a method returning one, is refused.
- A model with several top-level block lists (an encoder and a decoder, a vision tower beside a language model) needs
  its other stacks declared in the map (`stacks:`); `draft_map.py` drafts the primary stack and lists the others
  as `UNMAPPED STACK` for you to map.
- Do not put doses, learning rates or other run parameters in a map or an address book. They name structure and sets,
  nothing else.
