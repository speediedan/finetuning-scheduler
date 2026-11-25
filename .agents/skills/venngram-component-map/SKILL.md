---
name: venngram-component-map
description: Find, draft, verify, and prepare a venngram base component map for a model architecture. Use when adding an architecture to the registry or checking that a registry map describes a model's module and parameter layout.
---

# Author a venngram component map

## Find and verify a map

Use the exact model class and configuration revision. From the consumer's pinned environment, look up the
architecture with `venngram.maps.component_map_for(architecture, strict=True)` in a fresh process. Record the
map source and the index revision. An in-process registration can override the registry lookup.

Verify that map with the model's configuration. This command fetches only the configuration and builds the model
on the meta device. The order check uses a small model with random weights.

```bash
python -m venngram.maps.verification MODEL_ID --architecture MODEL_CLASS --record verification.yaml
```

Use `--map component-map.yaml` to verify a candidate file. To bind evidence to a particular Hub revision, fetch
its config at that revision with `AutoConfig.from_pretrained(model_id, revision=commit)` and call
`venngram.maps.verify(config, cmap, architecture=architecture)`. A custom model can pass its own model,
configuration and an `inputs(model)` forward to `verify`.

Read the status of each check. The verifier tests module existence, parameter partition, widths, weight layout
and forward order. It also tests declared head blocks and execution visits when present. The process exits zero
for a partial result, so check that the record says `status: verified` before proposing a verified row. If an
order check was skipped, supply a tiny configuration or a suitable forward. Run a planted omission or swapped
row as a control and retain the failure that demonstrates the relevant check ran.

## Draft a missing map

Build a representative module tree without downloading weights. Use `venngram.maps.drafter.draft_map(model)`
and `map_document(draft)` to get an editable YAML mapping. Review the forward and the parameter names with the
draft in hand. Resolve each `draft.*` row and each uncertain alternative. Keep fused projections as single
components and declare additional stacks explicitly. Use `optional` for a module's model-dependent layer
coverage. Use an explicit layer range or set for coverage fixed by the architecture.

Use `venngram.maps.verification.with_layouts(document, model)` to measure weight layouts and head sizes before
writing the candidate YAML. Verify the revised file and retain its report. Keep the exact bytes that were checked,
the config revision, the verifier version, and the verification record together.

## Prepare a registry contribution

Stage the two files as `component_map` and `verification_record` documents with
`venngram.store.publish("local", destination, source, documents=[(kind, path), ...])`. Use a fresh destination and
run `python -m venngram.conformance local:/absolute/path/to/destination`. A proposal identifies the exact document
paths, digests, configuration and verification status.

See [the registry runbook](https://github.com/speediedan/venngram/blob/main/registry/README.md) to inspect and
preview the index. Compare the proposed index against the currently published generation. Check that existing
rows survive and that the configured readers accept the candidate schema. Get explicit authorization naming the
Hub repository and reference before uploading documents or publishing an index. After an authorized publish,
re-read the immutable revision and verify that strict lookup returns the proposed map.
