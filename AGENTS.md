# LAIR — Agent Notes

Repository-specific guidance for anyone (human or agent) working in this tree. Read before changing code, and especially before shipping a phase.

## Documentation is part of the deliverable

LAIR keeps two documentation trees, both authoritative, both maintained with the code:

| Tree | Audience | Contents |
|---|---|---|
| `docs/user/` | operators | What LAIR does today, how to run it, every CLI flag and config key, what is planned but not yet accepted. Entry point `docs/user/README.md`, detail in `docs/user/reference/CLI.md` and `docs/user/reference/CONFIG.md`. |
| `docs/engine/` | contributors and engine readers | How the implemented engine actually works: module map, each stage of the inference pipeline, scheduler and step composition, CPU kernel design, quantization handling, CUDA backend, KV cache and prefix reuse, residency tiers and budgets, model profiles, tokenizer/template/tool/sampling behavior, measurement harnesses, comparison with other engines, and the catalog of tricks/tweaks. Entry point `docs/engine/README.md`. |

Rules that apply to both trees:

- **Describe what this build does.** Engine pages document implemented behavior from the source in `src/`, not intent. Anything landed-but-not-default (internal seams, capability gates, refused configurations) is stated as such, with the refusal semantics — never silently described as working, and never quietly omitted.
- **Ground claims in code or measurement.** Cite code by path (`src/Lair.Kv/PrefixCache.cs`), behavior by observable effect, performance claims by a committed artifact under `bench/reports/` or `reports/<timestamp>/`. Do not cite `PLAN.md`, phase numbers, or MVP labels in engine prose: the plan is project bookkeeping, and its wording drifts.
- **Numbers carry their posture.** A throughput or cost number states model file, context length, thread quota/kernel selection, kernel/ISA path, and measurement tool. Undated numbers rot; put the date next to them or point at the report that holds it.
- **No invented knobs.** If a knob does not exist in this build, say so plainly rather than describing an aspirational surface. Planned-but-unaccepted keys belong only in the "Planned" sections of `docs/user/reference/`.

## Keeping docs current when code changes

Every change to engine behavior updates at least one page here — no exceptions for "small" performance or placement work:

| If you touch… | Update |
|---|---|
| routes, CLI verbs, flags, exit codes, error shapes | `docs/user/reference/CLI.md`, `docs/user/README.md` |
| config keys, defaults, constraints, hot-reload mutation class | `docs/user/reference/CONFIG.md`, `lair.example.yaml`, `docs/user/README.md` |
| scheduler, admission, batching, step composition, lanes, concurrency gates | `docs/engine/scheduling.md`, `docs/engine/request-lifecycle.md` |
| prefill kernels, GEMV/GEMM paths, ISA dispatch, fusion, grid modes, row widths | `docs/engine/cpu-kernels.md`, `docs/engine/tricks-and-tweaks.md` |
| quant format coverage (loader acceptance, kernel support) | `docs/engine/quantization.md`, plus the model-family section of `docs/user/README.md` and both backend pages |
| CUDA kernels, PTX binding, device step composition, KV mirroring, streams, capability gates | `docs/engine/cuda-backend.md` |
| block pools, KV formats, prefix cache, windowed KV | `docs/engine/kv-cache.md` |
| ledgers, tier chains, budget caps, host posture, NUMA placement, weight residency | `docs/engine/residency-and-budgets.md` |
| architecture profiles, rope scaling, softcap/windowing, MoE descriptors, new model families | `docs/engine/model-profiles.md`, plus the model-family section of `docs/user/README.md` |
| tokenizer profiles, chat templates, tool parsing, reasoning regions/budgets | `docs/engine/tokenizer-templating-tools.md` |
| sampler, penalties, logprobs, drafting seams | `docs/engine/sampling-and-drafting.md`, plus the sampling fields in `docs/user/reference/CONFIG.md` |
| monitor stream, metrics bus, accounting | `docs/engine/monitoring-and-metrics.md`, plus `/monitor/stream` and monitor rows in `docs/user/README.md` |
| bench harness verbs, parity gates, new measurement workflow | `docs/engine/measurement-harnesses.md` and `DEVELOP.md` |
| a decision that changes how the engine behaves relative to other engines | `docs/engine/comparisons.md`, plus an entry in `CHANGELOG.md` |

Also expected on each ship:

- `CHANGELOG.md` gets a versioned Keep-a-Changelog entry; never rewrite history.
- `README.md` (repo root) describes current state only — no roadmap, no phase status.
- `DEVELOP.md` describes the current build/test/bench workflow, not plan status.
- When a new measurement changes a documented default or cost claim, update that claim in the same change and reference the report directory produced by the run (`reports/<timestamp>/` JSON+HTML pair; narrative prose goes in `bench/reports/`).

If a page no longer matches the code because of an unrelated edit you are landing anyway, fix it in your change rather than leaving drift behind. If a page is genuinely missing coverage for something you implemented, add it — engine documentation gaps count as unfinished work.

## Engine documentation conventions

- One topic per file; cross-link instead of duplicating. Each page opens with a scope line and an "as of" date so readers can judge freshness.
- Prefer the shape: what this subsystem is for → how data actually flows through it (with real type names) → why it is built that way (measured reason, rejected alternative) → what knobs select it → what it refuses today.
- Diagrams in fenced `mermaid` blocks; keep them to flow rather than class detail. Tables are preferred over prose for format coverage and knob lists.
- Code snippets show the real API (`IBackend`, `IKvPool`, `PlacementPolicy`) with a path beside it, not paraphrase.

## Verification before you call anything done

```powershell
dotnet build Lair.sln
dotnet test Lair.sln                          # full suite
dotnet test Lair.sln -c Release               # required for latency/perf gates
dotnet test Lair.sln --filter Category=Parity  # golden parity vs pinned llama.cpp
```

Docs-only changes still need a link check: every relative link in `docs/` resolves, and the tables above name files that exist. Real-model suites require fixtures under `models/` (gitignored) and skip gracefully when absent — see `DEVELOP.md`.

## Git conventions in this repo

- Stage by explicit path; never `git add -A`. When a change touches only part of `PLAN.md`, stage just those hunks.
- Commit prefixes used here: `feat(...)`, `fix(...)`, `perf(...)`, `docs(...)`, with scope names like `plan`, `models`, `cpu`, `cuda`, `kv`, `reports`, `engine`.
- Phase work happens on a `phase/*` branch and merges into the default branch (`master`); benchmark evidence is committed in the same change as the kernel or configuration decision it validates.
