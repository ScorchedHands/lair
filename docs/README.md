# LAIR Documentation

LAIR keeps two documentation trees. They answer different questions about the same build, and both are authoritative for their own audience.

| Tree | Question it answers | Who reads it | Entry point |
|---|---|---|---|
| [`docs/user/`](user/) | *What does LAIR do, and how do I run it?* Features as they behave today, every CLI verb and flag, every `lair.yaml` key with its default, constraints, and hot-reload mutation class, model families served, quant formats accepted, what is planned but not yet accepted. | Operators: anyone running a server or fetching a model pack. | [`docs/user/README.md`](user/README.md) — overview and operation; detail in [`reference/CLI.md`](user/reference/CLI.md) and [`reference/CONFIG.md`](user/reference/CONFIG.md). |
| [`docs/engine/`](engine/) | *How does the implemented engine actually work?* Module map and seams, each stage of the inference pipeline, scheduler and step composition, CPU kernel design, quantization handling, CUDA backend, KV cache and prefix reuse, residency tiers and budgets, model profiles, tokenizer/template/tool/sampling behavior, measurement harnesses, comparison with other engines, and the tricks/tweaks catalog. | Contributors, engine readers, and anyone deciding whether a default or a refusal is justified. | [`docs/engine/README.md`](engine/README.md) — index and reading order; detail pages under [`reference/`](engine/reference/). |

## Which tree a change belongs to

- A new CLI verb, flag, exit code, error shape, config key, default, constraint, or mutation class → `docs/user/` (its README plus the matching reference page).
- Anything that changes how the engine behaves internally — kernel selection, scheduling rule, KV/block semantics, residency or budget enforcement, backend capability gates, format coverage, monitoring payload → the matching page in [`docs/engine/reference/`](engine/reference/), and the user pages that describe the operator-visible consequence.
- A change an operator can see **and** a contributor must understand updates both trees: state the observable behavior in `docs/user/`, state the mechanism and its measured justification in `docs/engine/`. Cross-link rather than duplicating a paragraph across trees.

Both trees follow the same honesty rules — describe what this build does, label landed-but-not-default seams and refused configurations instead of omitting them, ground performance claims in committed artifacts under `bench/reports/` or `reports/<timestamp>/`, and never present an unimplemented knob as available. The per-subsystem mapping (which page every kind of change must update) lives in [`AGENTS.md`](../AGENTS.md); the build/test/bench workflow lives in [`DEVELOP.md`](../DEVELOP.md).

## Layout convention

Each tree keeps its index at `README.md` and its detail under `reference/`:

- `docs/user/README.md`, `docs/user/reference/CLI.md`, `docs/user/reference/CONFIG.md`
- `docs/engine/README.md`, `docs/engine/reference/<subsystem>.md` (15 subsystem pages: architecture, request-lifecycle, scheduling, monitoring-and-metrics, cpu-kernels, quantization, cuda-backend, kv-cache, residency-and-budgets, model-profiles, tokenizer-templating-tools, sampling-and-drafting, measurement-harnesses, comparisons, tricks-and-tweaks)

Repository entry point for everything: [`README.md`](../README.md).
