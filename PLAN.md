---
kilo-sensei: 3.0.0
---

# LAIR — Local AI Inference Runtime

## Context

LAIR is a C#/.NET inference runtime for local LLMs on x64 hardware, serving CPU/RAM-only machines and hybrid CPU/GPU machines from one engine. It adopts Strata's product surface — an OpenAI-compatible API, published GGUF model packs, disk-backed weights when they exceed RAM — and Colibri's residency approach: keep only the weights a token actually touches hot (routed experts, pageable PLE tables), gathering the rest from cold RAM or NVMe. Its differentiator is multi-tenant serving: concurrent requests batched into shared decode steps, multiple models resident concurrently against explicit per-device budgets, and per-model placement policies — compute split, prefill target, KV residency — all tunable live through a config file. Development happens on an i9-13900K / 64 GB / RTX 4090 box that exercises every code path; production targets include AVX2-only Xeons (E5-2683 v4, DDR4, 128 GB) with no GPU.

## Scope

**In (v1):**

- Dense GGUF inference (Llama/Qwen-class ≤3B) at Q4_K/Q8_0, AVX2 baseline with optional AVX-512 dispatch
- OpenAI-compatible HTTP API, supporting as much of the OpenAI spec as maps to local inference: `/v1/chat/completions` (text + image input, tool calls, logprobs, penalties, `logit_bias`, seed), `/v1/completions`, `/v1/embeddings` (text), `/v1/models`, SSE streaming with usage
- Vision input for chat models: image preprocessing + vision encoder + visual-token injection
- Concurrent requests batched into shared decode steps (continuous batching), with chunked prefill interleaved into the same batch steps
- Prompt caching: prefix KV reuse across requests; conversation caching: idle conversation state retained in RAM or persisted to disk and restored
- Multiple models resident and served concurrently against per-device budgets
- Per-model placement policy: device (`cpu` | `cuda` | `hybrid` with layer split), prefill target, KV residency (`dram` | `vram` | `split`), VRAM budget, CPU thread quota
- Config-file hot-reload for every runtime knob; no admin API
- CUDA backend with Strata-style hybrid CPU/GPU inference
- MoE residency for Qwen3.8-Flash-Next: router, expert gather, RAM hot tier, pageable PLE with indexed disk gather, MTP drafting
- Tiered residency: NVMe cold expert tier, VRAM hot expert tier
- Knobs-first principle: every placement/residency combination is implemented and testable; defaults reflect measured-optimal tuning; users may deliberately choose suboptimal combinations (e.g. lower speed for lower DRAM/VRAM footprint)
- Modular weight supply: kernels receive all weights through a swappable provider, so byte-reduction experiments (layer skipping, expert pruning, dynamic quant switching, early-exit vocab) plug in without kernel rewrites
- Observability: per-model tok/s, cache hit rates, tier utilization, PCIe transfer costs — structured logs plus a CLI monitor, and a real-time Web UI showing in-flight requests with per-request stats, engine/machine usage (CPU, memory, GPU, PCIe, disk I/O) portioned by model/role, CPU and GPU power draw where obtainable, CPU/GPU/RAM temperatures where exposed by the platform, cache usage with location breakdown, and the most recent X requests; monitor is read-only — no admin API

**Out (explicit non-goals for v1):**

- Model conversion/quantization tooling — LAIR consumes published GGUF quants (rationale: the ecosystem already produces them; Decision Log 2026-10-31)
- Anthropic-compatible API — backlog idea
- Vulkan/Metal/SYCL backends — backlog; LAIR commits to CPU + CUDA only
- Audio modality (OpenAI audio/realtime APIs) — not yet; backlog
- OpenAI platform surface (fine-tuning, assistants, batch, moderation, image generation) — decided against; platform features, not inference features (Decision Log 2026-10-31)
- Admin API surface — decided against; config-file hot-reload only (rationale: single-operator local runtime; Decision Log 2026-10-31)

## Architecture

```mermaid
graph TD
    C[Clients] -->|OpenAI-compatible HTTP + SSE| API[lair.api]
    API --> R[Model Router / Registry]
    R --> MI[ModelInstance x N]
    MI --> S[Scheduler: continuous batching, admission, placement]
    S --> E[Engine: layer executor]
    E --> B[IBackend kernels]
    B --> CPU[CpuBackend: AVX2 / AVX-512 dispatch]
    B --> CUDA[CudaBackend: dense + hybrid]
    E --> KV[KV/State Manager: paged blocks, device-backed pools]
    E --> SAMP[Sampler + MTP drafter]
    L[lair.loader: GGUF mmap] --> WA[Weight Arena: device-agnostic]
    WA --> CPU
    WA --> CUDA
    CFG[Watched config file] -->|hot-reload bus| S
    CFG -->|hot-reload bus| KV
    CFG -->|hot-reload bus| B
```

### Decisions

**Backend seam** — all compute behind `IBackend`; `CpuBackend` default, `CudaBackend` arrives MVP3 but the contract exists in MVP1.
- Rationale: dotLLM proves native C# kernels reach 66–88% of llama.cpp on decode; retrofitting a backend abstraction after the fact is costly.
- Rejected: wrapping llama.cpp wholesale (LAIR is a runtime, not bindings).
- Contract: op dispatch + tensor arena handles per device.

**Managed-first kernels + `[LibraryImport]` escape** — System.Runtime.Intrinsics everywhere; native microkernels only where measurement proves a wall.
- Rationale: dotLLM's outer-product tile needs ~23 YMM registers vs AVX2's 16 — RyuJIT spills kill tiling reuse; shims are the known fix.
- Rejected: pure-managed absolutism (ignores the register wall); native-core-first (loses the C# productivity thesis).

**Paged KV with device-backed pools** — fixed-size blocks allocated on demand from a CPU arena or a VRAM pool; runtime resize means changing pool budgets, never reallocating live sequences.
- Rationale: fits multi-tenant budgets and hot-reload; vLLM-proven.
- Rejected: contiguous per-sequence buffers (rigid budgets).
- Extension: blocks are reference-counted and copy-on-write, so a shared prompt prefix is prefilled once and reused across requests; idle conversations keep their block runs in RAM or snapshot them to the NVMe tier and restore on resume.

**Knobs-first, opinionated defaults** — every placement/residency combination (prefill device × decode split × KV residency) is implemented, tested, and user-selectable; defaults are the measured-optimal path.
- Rationale: users may trade inference speed for lower DRAM/VRAM usage; the engine must not hard-code the "optimal" combination.
- Contract: the bench harness documents the cost curve of each knob so defaults stay defensible.

**Continuous batching** — sequences share decode steps; a unified token batch is the only lever that raises arithmetic intensity on CPU decode.
- Rationale: batch-1 GEMV has zero weight reuse; batching converts GEMV toward GEMM.
- Rejected: one thread per request (bandwidth thrash, no weight sharing).
- Contract: prefill chunks interleave with decode tokens in the same batch step, so a long prompt never stalls decode latency for other sequences.

**Device-agnostic weight arena** — mmap'd GGUF tensors placed per `PlacementPolicy`; NUMA-aware first-touch; never mlock'd.
- Rationale: archive measurement shows mlock at −5%; page-cache residency is the legitimate alternative.
- Movement: weight slabs are relocatable at runtime — promotion/demotion across RAM/VRAM/NVMe tiers and migration across NUMA nodes while serving, guarded by slab reference counts and double-buffered staging so a slab in flight is never evicted mid-read.

**PCIe tax as measured cost** — the scheduler models transfer cost explicitly; KV follows decode device by default; `split` KV residency is validated by bench before it becomes a default anywhere.
- Rationale: attention reads all KV every token; bytes on the far side of PCIe are paid per token; llama.cpp's no-op-offload tax crippled hybrid runs in the archive.

**AVX2 baseline, AVX-512 optional** — both dev and production target boxes are AVX2-only (13900K and E5-2683 v4 lack AVX-512); AVX-512 is a dispatch path, never a design assumption.

**GGUF-first with GGUF-embedded tokenizer** — parse tokenizer + chat template from GGUF metadata; consume the published quant ecosystem.
- Rejected: HF tokenizer.json sidecar (extra files break the model-pack story).

**Config-file-only admin** — one watched config file pushes changes through a hot-reload bus to scheduler, KV pools, and backends.

**Read-only monitor** — the Web UI, CLI monitor, and logs all consume one metrics bus (internal engine counters + OS/driver stats) over a live stream (WebSocket/SSE); nothing on the monitor surface mutates state.
- Rationale: knobs-first tuning needs visibility; mutation stays confined to the config file so there is exactly one write path.
- Contract note: PCIe/DRAM "usage" is measured as engine counters (transfer bytes, tier hits) plus OS/driver readings for CPU/VRAM — hardware perf counters are a refine detail, not a promise. Disk I/O is both: OS counters for total device throughput, engine counters for engine-attributed traffic (expert tier streaming, PLE indexed reads, conversation snapshots) attributed per model/role. Power and temperature are best-effort: GPU via NVML (reliable for both); CPU via hwmon/RAPL sysfs on Linux and kernel-helper/third-party telemetry on Windows; RAM temperature only where the platform exposes it (server BMC/IPMI on the Xeon boxes, generally absent on consumer boards); per-model power is a utilization-weighted estimate, labeled as such.

**CPU-only deployment justified** — a CPU-only path avoids the PCIe offload tax entirely on GPU-less boxes; hybrid is opt-in on capable machines.

**Weight supply abstraction** — kernels never touch raw weight buffers; every weight access goes through an `IWeightSource` provider.
- Rationale: decode speed is bytes-per-token, so every future byte-reduction experiment (mixture-of-depths/layer skipping, expert pruning, dynamic quant switching, early-exit vocab, ternary formats) is a provider swap behind this seam, not kernel surgery.
- Rejected: kernels bound directly to mmap'd buffers (each experiment would require rewriting the hot path).

**OpenAI spec subset** — LAIR implements the OpenAI surface that maps to local inference, not the platform surface.
- In: chat completions (text + image, tool calls, logprobs, penalties, `logit_bias`, seed, streaming with usage), completions, text embeddings, models.
- Out (rationale: cloud-platform features with no local-inference meaning): fine-tuning, assistants, batch, moderation, image generation, audio/realtime.
- Contract: unmodified OpenAI SDK clients work against the supported endpoints.
- `/v1/models` extension: OpenAI baseline fields plus non-breaking metadata — `context_length`, modalities, `supported_parameters`, quant type, placement — following the Ollama/OpenRouter/llama.cpp ecosystem convention; coding tools depend on `context_length` and otherwise assume 8192.

**High lock-in (confirmed):** OpenAI-compatible wire contract · GGUF format commitment · `IBackend` contract · `PlacementPolicy` config schema.

### Modules

- `lair.host` — CLI + HTTP host, config watcher, lifecycle
- `lair.api` — OpenAI-compatible endpoints, SSE streaming, request lifecycle
- `lair.core` — engine, sequences, continuous batching
- `lair.models` — architecture definitions + GGUF tensor mapping (dense first; Flash-Next later)
- `lair.backends.cpu` — intrinsics kernels, fused dequant GEMV/GEMM, shim escape
- `lair.backends.cuda` — device kernels, hybrid layer-split execution
- `lair.kv` — paged state manager, device-backed block pools, budgets
- `lair.tokenizer` — GGUF-embedded BPE + chat templates
- `lair.sampling` — sampler stack + MTP drafter
- `lair.loader` — GGUF parse, mmap, placement into arenas
- `lair.monitor` — read-only Web UI (static SPA + live stats stream) and the metrics bus feeding it, CLI monitor, and logs
- `lair.vision` — image preprocessing, vision encoder, visual-token injection (MVP7)

### Seams

- `IBackend` — default `CpuBackend`
- `IGatherKernel` — default CPU gather (MoE phase)
- `IResidencyTier` — default RAM-only; NVMe cold tier and VRAM hot tier later
- `IConfigSource` — default watched config file
- `IModelLoader` — default GGUF mmap
- `IWeightSource` — default `MmapTensorSource`; byte-reduction experiments plug in as alternate providers

### Data model

Weights are GGUF tensor blobs, mmap'd and never transformed. Sequence state is a paged block table allocated from device-backed pools. The model registry is config-declared. Requests are transient. No relational store; structured reads come from GGUF metadata parsed at load.

### Key references

llama.cpp/ggml (fused dequant kernels, block quant formats, hybrid offload pitfalls) · ik_llama.cpp (CPU-first fused kernels, expert prefetch — the CPU bench ceiling and the reactive half of what LAIR makes proactive) · dotLLM (native C# engine patterns, `IBackend` shape, register-pressure lessons) · colibri (expert streaming off disk, hot-tier residency, pageable PLE, MTP drafting, multi-conversation shared decode) · Strata (hybrid CPU/GPU serving, model packs, OpenAI-compatible product surface) · vLLM (paged KV cache).

## Build / Deployment

- .NET 10; solution layout `src/`, `tests/`, `bench/`
- Publish: self-contained `win-x64` single executable; config file + model pack directory alongside
- CUDA native shims built via an `nvcc` MSBuild target (MVP3 onward); CPU-only builds carry no CUDA dependency
- Bench harness vs llama.cpp is a first-class artifact from MVP2 — it validates every knob's cost curve
- Dev box: i9-13900K / 64 GB DDR5 / RTX 4090 24 GB (AVX2, exercises all paths). Production target: dual/single Xeon E5-2683 v4, DDR4 quad-channel, 128 GB, AVX2, no GPU
- CI/CD and Native AOT packaging: backlog

## Phases

### MVP1 [v0.1.0] - Shipped

One dense GGUF model answers one OpenAI request, streamed, on CPU.

#### Design Notes

- **DN-M1-1** — Llama-family and Qwen2 dense (≤3B) are the MVP1 architectures, including GQA and q/k/v bias variants.
  - Rationale: both share one shape family (RMSNorm + RoPE + SwiGLU + GQA); every later model inherits GQA and bias handling.
  - Rejected: Llama-only (arch rewrite when Qwen2 arrives); MHA-only (rewrite in MVP5).
  - Contract: architecture registry in `lair.models`.
- **DN-M1-2** — Quant set is F16, Q8_0, Q4_K_M.
  - Rationale: F16 is the golden-parity reference format; Q8_0/Q4_K_M are the production paths.
  - Rejected: IQ formats (shuffle-heavy on AVX2, backlog); FP16-only (no production path).
  - Contract: format descriptor in `IWeightSource`.
- **DN-M1-3** — `IWeightSource` resolves at plan time; kernels hold raw `ReadOnlySpan<byte>` + format descriptor, never an interface call in the GEMV loop.
  - Rationale: hot-path indirection is paid per tile; mmap read-only spans keep weights OS-page-cache resident.
  - Rejected: runtime interface dispatch per tile; eager copy into private arenas (doubles memory).
  - Contract: `IWeightSource.Resolve(tensorId, rowRange) → (ReadOnlySpan<byte>, QuantFormat)`.
- **DN-M1-4** — Chat templates render through a mini Jinja-subset interpreter (conditionals, loops, common filters, special-token handling).
  - Rationale: correctness on arbitrary GGUF templates; llama.cpp's minja is the working precedent.
  - Rejected: hardcoded per-family templates (breaks on custom templates); deferring chat to MVP2 (breaks `/v1/chat/completions` in MVP1).
  - Contract: template render in `lair.api` request pipeline; `lair.tokenizer`.
- **DN-M1-5** — Config file is YAML (YamlDotNet).
  - Rationale: the hot-reload schema grows with PlacementPolicy; comments matter in server config.
  - Rejected: TOML (weaker for nested policy blocks); JSON (poor hand-editing).
  - Contract: `IConfigSource` implementation.
- **DN-M1-6** — MVP1 admits exactly one in-flight request; a second concurrent request gets `engine_overloaded`.
  - Rationale: the phase outcome is one request end-to-end; concurrency is MVP2's theme.
  - Rejected: speculative multi-request support (half-built concurrency, violates phase theme).
  - Contract: scheduler admission rule + error taxonomy.
- **DN-M1-7** — Errors use the OpenAI error shape `{error:{message,type,code}}` with a bounded code set: `invalid_request`, `model_not_found`, `context_length_exceeded`, `engine_overloaded`, `internal`.
  - Rationale: SDK clients handle errors uniformly; bounded codes keep behavior predictable.
  - Rejected: .NET exception passthrough (leaks internals, breaks SDK expectations).
  - Contract: `lair.api` middleware.
- **DN-M1-8** — Kestrel binds loopback by default; authentication deferred (backlog: server posture).
  - Rationale: local runtime default posture; exposure is an explicit later decision.
  - Contract: host binding config.
- **DN-M1-9** — Model loading is fail-fast: missing/corrupt/OOM model aborts startup with an actionable message; `/healthz` readiness reflects model state.
  - Rationale: no half-ready engine state to reason about.
  - Rejected: lazy load on first request (hides load failures behind request errors).
  - Contract: host lifecycle + `/healthz`.
- **DN-M1-10** — Every request carries `x-lair-request-id`; JSON structured logs correlate on it.
  - Rationale: correlation IDs must exist before multi-request phases make logs unreadable.
  - Contract: `lair.api`/`lair.host` logging.

- **DN-M1-11** — Loader/source split: `IModelLoader` parses the container into a logical tensor table (id, shape, format, offsets); retrieval strategy lives entirely in `IWeightSource` providers selected per model in config.
  - Rationale: experimental storage/retrieval modules (repacked sidecars, tiered streaming, read-time transforms) plug in as providers without touching loader or kernels — the byte-reduction experiment program depends on this split existing from day one.
  - Rejected: loader-owned retrieval (each experiment would fork the loader); kernel-direct file access (kills tiering).
  - Contract: `IModelLoader → LogicalTensorTable`; `IWeightSource` provider selected by config.

#### Implementation Items

- [✓] GGUF loader — mmap tensors, metadata, vocab (see DN-M1-3)
  - Verify: `dotnet test --filter Category=Loader` — 8/8 on synthetic GGUF v3 ✅; real-file load verified on SmolLM-135M-Instruct-F16.gguf via tokenizer integration ✅; quantized real-load runs at Parity against a pinned Q8_0 build (SmolLM dims are not block-divisible, so llama.cpp's own quantizer emits legacy fallback tensors there); full Q4_K_M family support tracked in Backlog.
- [✓] GGUF tokenizer — BPE + special tokens decode/encode
  - Verify: `dotnet test --filter Category=Tokenizer` — 18/18 ✅ (13 synthetic + 5 against the real SmolLM GGUF: ASCII/newline/emoji round-trips, control tokens).
- [✓] Chat templates — mini Jinja-subset interpreter
  - Verify: `dotnet test --filter Category=Template` — 20/20 ✅ golden renders (ChatML = real SmolLM `tokenizer.chat_template`, conditional if/else + `|tojson`, Phi-style), subset semantics (concat, `==`/`!=`, `not`, `tojson`/`trim`, `{%- -%}`), and rejections naming construct + position (`{% set %}`, `|upper`, macro, slices). The real Llama-3 template needs `{% set %}`/loop vars — out of subset (backlog).
- [✓] Dense kernels — fused dequant GEMV for Q4_K_M/Q8_0/F16, AVX2, row-block threading (see DN-M1-3)
  - Verify: `dotnet test --filter Category=Kernel` — 15/15 ✅ vs double-precision oracle incl. threaded bit-identity; bench Q4_K 4096×4096: AVX2 8.7× scalar, threaded ≈17 GB/s ✅.
- [✓] Layer executor — RMSNorm, RoPE, GQA attention, SwiGLU FFN, qkv bias (see DN-M1-1)
  - Verify: `dotnet test --filter Category=Layers` — 24/24 ✅ incl. hand-computed op units and real SmolLM-F16 forward smoke (finite deterministic logits); exact-token parity runs at Parity.
- [✓] Paged KV — CPU block pool, 16-token blocks, allocator, per-model budget from config
  - Verify: `dotnet test --filter Category=Kv` — 8/8 ✅ (capacity errors, alloc/release integrity, addressing across partial blocks, refcounts); multi-turn prefix intactness verified end-to-end at Manual Serve.
- [✓] Sampling — greedy, temperature, top-k/top-p, seed
  - Verify: `dotnet test --filter Category=Sampling` — 9/9 ✅ (seeded determinism, top-k/top-p intersection order, statistical match within 1%, banned/NaN handling).
- [✓] HTTP API — chat/completions, completions, models (with `context_length` metadata), SSE streaming, error taxonomy (see DN-M1-6, DN-M1-7)
  - Verify: `dotnet test --filter Category=Api` — 10/10 ✅ against a live server on the real SmolLM fixture: non-stream chat + completions shapes, SSE chunk stream ending `[DONE]`, `/v1/models` with `context_length`, `/healthz`, embeddings-404, error taxonomy (model_not_found / context_length_exceeded / engine_overloaded / invalid_request), `x-lair-request-id` on every response. OpenAI Python SDK probe runs under Manual Serve.
- [✓] Backend seam — `IBackend` + arenas + `IWeightSource` contracts, CPU-only implementation (see DN-M1-3, DN-M1-11)
  - Verify: `lair.core` compiles against contracts only — no project reference to `lair.backends.cpu` ✅; alternate `IWeightSource` provider end-to-end proof ran with the Parity item (`ArenaWeightSourceTests`: tensors copied into backend-allocated aligned blocks, token sequence identical to mmap-backed decode) ✅.
- [✓] Golden parity — logit-agreement harness vs pinned llama.cpp (server-mode reference), committed calibrated fixtures (see Decision Log 2026-11-01)
  - Verify: `dotnet test --filter Category=Parity` — 9/9 ✅ on SmolLM-135M F16 + Q8_0 vs pinned llama.cpp b12032 server references (4 prompts × 2 models, sha256-guarded): LAIR argmax lands in the reference top-5 for every prompt; calibrated distributions reproduce exactly. Harness work exposed and fixed four engine bugs: prefill double-forward of the last prompt token, BOS applied regardless of `add_bos_token`, SmolLM digit-run split cutting all non-digit text per-character, and reversed BPE merge priority order. Alternate-`IWeightSource` arena proof green — completes the Backend seam item's end-to-end proof too.
- [✓] Config load — YAML parse, validation, placement policy (single model, `cpu`)
  - Verify: `dotnet test --filter Category=Config` — 31/31 passing ✅; `lair serve -c lair.yaml` boot is exercised under the CLI item.
- [✓] CLI — `lair serve -c lair.yaml` and `lair chat` REPL
  - Verify: `dotnet test --filter Category=Cli` — 4/4 ✅ (piped stdin vs real SmolLM: banner + streamed reply, `/reset`, overflow retry then clean error line, cancellation → interrupted/0). Manual serve boot + piped chat run verified; quantized-model serving check runs at Manual Serve.

#### Tests

- [✓] Automated Loader: GGUF metadata, vocab, tensor offset parsing against fixtures — 8/8
- [✓] Automated Tokenizer: encode/decode round-trip vs llama.cpp semantics — 18/18, real-model cases executing live
- [✓] Automated Template: golden chat-template renders, three families
- [✓] Automated Kernel: fused dequant GEMV vs scalar reference, FP tolerance — 15/15 + DequantRow cases
- [✓] Automated Layers: per-layer activations vs hand-computed references + real-model smoke — 24/24
- [✓] Automated Kv: block allocator alloc/free/reuse and budget enforcement — 8/8
- [✓] Automated Sampling: determinism under greedy/seed; statistical top-k/p behavior — 9/9
- [✓] Automated Parity: per-prompt sha256 guard + LAIR argmax ∈ reference top-5 (pinned llama.cpp server) + reproduction of the fixture's calibrated LAIR distribution within |Δlogprob| ≤ 0.2, on SmolLM-135M-F16 and Q8_0 — 9/9
- [✓] Automated Config: YAML schema validation and error messages — 31/31
- [✓] Manual Serve: full serve→chat REPL session with streaming, abort, and reload-after-config-edit — done ✅ vs quantized Q8_0 model via YAML config: healthz/models/chat/completions probes, SSE chunk stream + mid-stream client abort with clean recovery on the next request (structured per-request log lines), piped `lair chat` REPL run against the same config, OpenAI Python SDK probe (`models.list`, non-stream chat, streaming) green, edited `max_context_tokens` 512→768 picked up on restart.

#### Open Items

- [X] **Jinja subset scope** — resolved by the Template implementation item: the interpreter supports conditionals, loops, variable access/indexing, concatenation, and the filters demanded by golden templates (`tojson`, `join`, `trim` as starting set); any unsupported construct fails with an explicit error naming the construct rather than rendering silently wrong.

### MVP2 [v0.2.0] - In Progress

Many requests at once, tuned, with live reconfiguration.

#### Design Notes

- **DN-M2-1** — Discrete batch steps: the scheduler runs fixed steps; each step executes one token per active decode sequence plus prefill chunks up to `max_batch_tokens`.
  - Rationale: step boundaries are the natural point for admission, cancellation, metrics, and hot-reload application.
  - Rejected: pipelined overlap across steps (complexity without CPU benefit).
  - Contract: scheduler in `lair.core`.
- **DN-M2-2** — Decode-priority scheduling; prefill chunks fill remaining batch slots, adaptive chunk size (default 512 tokens, config knob).
  - Rationale: inter-token latency is the perceived-quality metric; TTFT stays bounded by chunking. This is vLLM's proven pattern.
  - Rejected: prefill-priority (stalls ITL); strict FCFS (head-of-line blocking on long prompts).
  - Contract: scheduler step composition.
- **DN-M2-3** — Prefix cache: block table keyed by token-prefix hash, reference-counted copy-on-write blocks, LRU eviction within pool budget.
  - Rationale: shared prefixes (system prompts, multi-turn) are prefilled once; paged CoW makes sharing free.
  - Rejected: per-request prompt copies (memory blowup); hash-on-full-request only (no partial sharing).
  - Contract: `lair.kv` block table; invalidation on model reload, block-size change, or quant change.
- **DN-M2-4** — Cancellation at step boundaries: abort flag checked between steps; refcounted prefix blocks survive the abort of one sequence.
  - Rationale: mid-step abort would corrupt shared kernel state; step granularity is sub-100ms.
  - Contract: scheduler admission/cancel path.
- **DN-M2-5** — Hot-reload semantics table: live = `max_batch_tokens`, thread quota (next step), KV pool budget (grow immediate, shrink drains evictable blocks), per-model priority; new-requests-only = max context length; restart-only = block size, quant format, model path.
  - Rationale: every knob needs a declared mutation class or hot-reload becomes undefined behavior.
  - Contract: `IConfigSource` change-application rules.
- **DN-M2-6** — Core-set affinity: models pinned to disjoint NUMA-local core sets; requests inherit their model's set.
  - Rationale: partition unit is the model; per-request core splitting duplicates weight streaming inside shared batch steps.
  - Contract: scheduler + thread pool configuration.
- **DN-M2-7** — Metrics bus: engine counters (tok/s per model, step time, batch occupancy, cache hit rate, KV utilization) plus OS counters into an in-memory ring and JSON logs; CLI monitor consumes it now, MVP4's Web UI consumes the same bus later.
  - Contract: `lair.monitor` metrics bus.
- **DN-M2-8** — Tool calls: per-family tool schema injection into the chat template; `tool_calls` parsed from streamed output; tool execution stays client-side.
  - Rationale: coding-agent compatibility (Strata's whole tool story) requires the wire contract, not execution.
  - Contract: `lair.api` request/response pipeline.
- **DN-M2-9** — Latency budget: decode ITL p95 ≤ 400 ms for N=8 concurrent SmolLM-class requests on the dev machine; chunked prefill must hold short-request ITL within the same budget with an 8K-token prompt in flight.
  - Rationale: measured single-stream baseline is ~70–105 ms/token (SmolLM-135M Q8_0, full threads); a concrete ceiling makes batching regressions catchable without machine-specific absolutes.
  - Contract: Concurrency and Prefill test thresholds.

#### Implementation Items

- [✓] Continuous batching — shared decode steps across sequences (see DN-M2-1)
  - Verify: `dotnet test -c Release --filter Category=Concurrency` — 5/5 ✅ on real SmolLM-F16: N=8 concurrent outputs identical to sequential for greedy AND seeded; measured ITL p95 238 ms (budget 400); capacity exhaustion surfaces as `engine_overloaded`. Scheduler fixes found in testing: finished sequences must be excluded from the same step's decode window, and outcomes release KV blocks before publishing.
- [✓] Chunked prefill — prefill chunks interleaved with decode tokens in the same step (see DN-M2-2)
  - Verify: `dotnet test -c Release --filter Category=Prefill` — 1/1 ✅ ~407-token prompt prefills chunk-by-chunk while short requests stream past it at ITL p95 68 ms with outputs identical to a fresh-engine baseline. Tuning finding: per-step time ≈ chunk width × single-stream cost, so the DN-M2-2 default (512) overshoots the CPU budget — chunk widths of ~8 hold 400 ms at measured Release speed (~28 ms/token F16); wide chunks only pay off on faster compute.
- [✓] Prompt caching — reference-counted prefix blocks reused across requests (see DN-M2-3)
  - Verify: `dotnet test -c Release --filter Category=Cache` — 6/6 ✅ on real SmolLM-F16: shared 127-token prefix → second request skips 112 prefill tokens with byte-identical output vs cold run; CoW copy verified bit-safe for the original sequence; LRU eviction never touches blocks referenced by active sequences; refcount release ordering tested. Cache lifetime = pool lifetime (covers reload/quant/block-size invalidation this phase).
- [_] Cancellation — abort sequences mid-batch-step without stalling the rest of the batch (see DN-M2-4)
  - Verify: integration test — aborted request frees its blocks; other requests complete unchanged.
- [_] API completeness — tool calls, logprobs, penalties, `logit_bias`, seed (see DN-M2-8)
  - Verify: OpenAI SDK scripts — tool-call round trip, logprobs present and normalized, penalties alter sampling deterministically under seed.
- [_] Metrics bus — engine counters + OS/driver stats streamed to logs and CLI monitor (see DN-M2-7)
  - Verify: JSON log lines carry per-model counters; `lair monitor` command renders live stats during load.
- [_] Embeddings groundwork — encoder-only forward pass, no endpoint yet
  - Verify: unit test of encoder forward on a small fixture model produces stable pooled vectors.
- [_] Thread pools — sized to memory channels, per-model quotas, core-set affinity (see DN-M2-6)
  - Verify: `dotnet test --filter Category=Affinity` asserts worker threads land in configured core sets.
- [_] Hot-reload application path — config watcher pushes validated changes through the change-application bus by mutation class: live knobs apply at the next step boundary, new-requests-only knobs affect admission only, restart-only knobs are refused with an actionable message (see DN-M2-5)
  - Verify: `dotnet test --filter Category=HotReload` — a mid-serving edit to a live knob is observable in engine behavior without restart; editing a restart-only knob returns refusal naming the mutation class.

#### Tests

- [✓] Automated Concurrency: parallel-vs-sequential output equivalence — 5/5 (Release) incl. ITL budget and capacity refusal
- [✓] Automated Prefill: chunked prefill ITL protection — 1/1 (Release)
- [✓] Automated Cache: prefix hit, CoW integrity, LRU eviction — 6/6 (Release)
- [_] Automated Cancel: abort isolation and block reclamation
- [_] Automated HotReload: watcher detection, mutation-class enforcement, mid-serving application
- [_] Automated API: tool calls, logprobs, penalties, seed reproducibility via SDK scripts
- [_] Manual Monitor-lite: `lair monitor` live stats under mixed load

### MVP3 [v0.3.0] - Refined

CUDA joins the engine; hybrid CPU/GPU inference on the 4090.

#### Design Notes

- **DN-M3-1** — CUDA binding: nvcc-built microkernels via `[LibraryImport]` for fused dequant/GQA kernels; cuBLASLt for dense F16 GEMM paths.
  - Rationale: GPU kernels are CUDA regardless of host language; "managed-first" was always a CPU-kernel doctrine. Fused dequant-FMA needs full control.
  - Rejected: ILGPU (cannot realistically fuse dequant into FMA chains); managed CUDA abstraction layer (immature, adds a translation tax).
  - Contract: `lair.backends.cuda` build target (`nvcc` MSBuild) + `[LibraryImport]` surface.
- **DN-M3-2** — Contiguous layer split: first N transformer layers resident in VRAM, remainder on CPU; LM head defaults to CPU.
  - Rationale: exactly one PCIe activation crossing per token at the split boundary; activations are ≤32 KB. GPU LM head would spend scarce VRAM on the 248K-vocab head for marginal gain.
  - Rejected: interleaved layer split (crossing every layer); GPU LM head default.
  - Contract: `PlacementPolicy.gpu_layers`, engine executor split point.
- **DN-M3-3** — Device policy is fail-fast: `device: cuda` or `hybrid` requires the device; no silent CPU fallback.
  - Rationale: silent fallback hides misconfiguration and produces unmeasurable performance claims. An `auto` fallback mode is a backlog candidate, not a default.
  - Contract: host startup validation, consistent with DN-M1-9.
- **DN-M3-4** — KV device pools: separate VRAM and DRAM block pools; `split` residency = device-tagged blocks, attention executed as two passes (VRAM set, then DRAM set); default residency follows the decode device.
  - Rationale: attention reads all KV every token — placement follows the reader unless the user pays the measured split cost.
  - Contract: `lair.kv` device-backed pools.
- **DN-M3-5** — Prefill offload: `prefill: gpu` runs prompt processing on CUDA; produced KV lands per `kv_residency`; decode continues per `device`.
  - Contract: scheduler step composition per model policy.
- **DN-M3-6** — Pinned host staging buffers for activations crossing the split; transfers overlapped with compute where measurement justifies it.
  - Contract: weight arena / device arena staging.

#### Implementation Items

- [_] Binding spike — nvcc MSBuild target + cuBLASLt `[LibraryImport]` smoke
  - Verify: build produces the CUDA kernel binary; smoke GEMM matches CPU reference within FP tolerance.
- [_] CudaBackend — dense decode + prefill kernels behind `IBackend` (see DN-M3-1)
  - Verify: `dotnet test --filter Category=Parity` — same F16 model decodes token-identical on CPU, CUDA, and hybrid.
- [_] Prefill offload — GPU prefill, CPU/hybrid decode (see DN-M3-5)
  - Verify: bench records ≥2× prefill tok/s on the 4090 vs CPU for the reference model.
- [_] Layer split — `gpu_layers` placement policy (see DN-M3-2)
  - Verify: split-point sweep (0..N) on Qwen2-1.5B; parity holds at every split point; bench records decode tok/s curve.
- [_] VRAM residency — models fitting 24 GB resident on device
  - Verify: full-VRAM Q4_K_M 1.5B run; over-budget load rejected with ledger breakdown in the error.
- [_] KV device pools — `dram` | `vram` | `split` residency policies (see DN-M3-4)
  - Verify: parity across all three residencies; bench records the split-vs-follow cost curve.

#### Tests

- [_] Automated Parity-Hybrid: token match across cpu / cuda / hybrid configurations
- [_] Automated Budget: VRAM admission rejection with actionable error
- [_] Automated Kernel: CUDA fused dequant GEMV vs scalar reference
- [_] Manual Hybrid Serve: monitor shows GPU utilization and PCIe transfer counters during hybrid decode

### MVP4 [v0.4.0] - Refined

Many models at once, each with its own placement policy.

#### Design Notes

- **DN-M4-1** — Registry and hot swap: models declared in config with `type: chat|embed`; add loads in the background and serves when ready; remove drains in-flight requests, then frees pools.
  - Rationale: model changes ride the same hot-reload bus as every other knob; drain-then-free avoids mid-flight collapse.
  - Rejected: restart-to-change-model; immediate unload (kills in-flight requests).
  - Contract: registry in `lair.core` + `IConfigSource`.
- **DN-M4-2** — Global ledger admission: DRAM quota, VRAM pool, and core sets are accounted at load time; runtime KV exhaustion rejects new requests, never preempts running ones.
  - Rationale: preemption needs swap to disk — that machinery arrives in MVP6; until then the honest answer is refusal.
  - Rejected: best-effort oversubscription (thrash); preempt-to-fittest (needs MVP6 tiers).
  - Contract: ledger in scheduler; errors via `engine_overloaded` / `context_length_exceeded`.
- **DN-M4-3** — Embeddings: encoder-only forward, mean pooling default with CLS option and `normalize` flag; input batching reuses MVP2 batch steps.
  - Contract: `/v1/embeddings` OpenAI shape; registry `type: embed`.
- **DN-M4-4** — Monitor Web UI: lightweight static SPA (TypeScript, esbuild) served by Kestrel; WebSocket stats stream from the metrics bus; recent-request ring buffer (configurable X).
  - Rationale: the engine stays free of frontend runtime coupling; the metrics bus already exists.
  - Rejected: Blazor (runtime coupling to engine host); polling-only UI (fails the real-time requirement).
  - Contract: `lair.monitor` WS schema (additive-only, versioned field).
- **DN-M4-5** — Scheduler fairness: core-set partitioning is the isolation mechanism; FIFO within a model, per-model priority; no default core oversubscription (overcommit is a backlog knob).
  - Contract: scheduler admission order.

#### Implementation Items

- [_] Model registry — concurrent instances, add/remove via config reload (see DN-M4-1)
  - Verify: `dotnet test --filter Category=Registry` — model added and removed via config edit while another model serves; no request failures outside the removed model.
- [_] Device budgets — admission control against VRAM + DRAM pools (see DN-M4-2)
  - Verify: ledger test — over-committed config rejected at startup with per-device breakdown in the error.
- [_] Scheduler fairness — time-slicing, preemption, priority (see DN-M4-5)
  - Verify: two-model load test — per-model ITL stays within budget while the other model saturates its core set.
- [_] Embeddings endpoint — `/v1/embeddings`, encoder-only models (ModernBERT-class) (see DN-M4-3)
  - Verify: `dotnet test --filter Category=Embeddings` — pooled vectors match reference fixtures within FP tolerance.
- [_] Flagship scenario — GPU-heavy coding model + CPU-only embedding model side by side
  - Verify: manual — both models serve concurrent load; monitor shows per-model device portioning.
- [_] Monitor Web UI — real-time in-flight requests, per-model CPU/memory/GPU/PCIe/disk-I/O breakdown, cache usage by location, recent X request log (see DN-M4-4)
  - Verify: manual — browser shows live in-flight rows updating sub-second, per-model device bars, cache breakdown, and completed-request log.

#### Tests

- [_] Automated Registry: hot add/remove under load
- [_] Automated Budget: ledger rejection paths and quota accounting
- [_] Automated Fairness: cross-model ITL isolation
- [_] Automated Embeddings: pooling parity vs reference fixtures
- [_] Manual Monitor: real-time UI behaviors enumerated above

### MVP5 [v0.5.0] - Refined

Qwen3.8-Flash-Next runs: sparse MoE with residency management.

#### Design Notes

- **DN-M5-1** — Reality-check gate: no Flash-Next kernel work until llama.cpp/ik_llama.cpp measurements (tok/s, bytes/token, expert gather behavior) are recorded on the target GGUF.
  - Rationale: the archive's explicit recommendation; prevents building a C# engine against imagined numbers.
  - Contract: `bench/` measurement report committed before kernel items start.
- **DN-M5-2** — Quant scope: IQ3_XXS primary (Strata's recommended consumer quant), Q4_K_M as the measured baseline; other formats added only as residency experiments.
  - Rationale: smaller quants shrink the expert hot set, but IQ3 dequant is shuffle-heavy on AVX2 — the gate decides whether the residency win survives the kernel cost.
  - Rejected: FP8 native containers (breaks the GGUF-first commitment); low-bit-only with no baseline (unmeasurable claims).
  - Contract: `IWeightSource` format descriptors for IQ3_XXS/Q4_K_M expert slabs.
- **DN-M5-3** — Expert-aware batching: the residency cost function is the batch's union active-expert set, not per-sequence expert sets.
  - Rationale: shared decode steps (MVP2) make expert reads amortized across sequences — this is the engine-level MoE win.
  - Rejected: per-sequence expert scheduling (re-reads shared experts per sequence).
  - Contract: scheduler batch composition + `IGatherKernel`.
- **DN-M5-4** — Contiguous expert slabs via `IWeightSource` providers: experts stored as contiguous slabs, gathered through `IGatherKernel`, prefetched on top-K router pointers before the GEMV.
  - Rationale: ik_llama.cpp proves prefetch; slabs make gather a pointer chase plus prefetch, not a scatter.
  - Rejected: expert-major interleaved layout (kills slab prefetch).
  - Contract: `IGatherKernel`, `IWeightSource` slab provider.
- **DN-M5-5** — RAM hot tier: capacity/cost-model residency decides which experts stay hot — value = router hit frequency × gather cost saved, budget = RAM quota minus KV working set.
  - Rationale: prefetch is taken by ik_llama.cpp; capacity/cost-driven residency is the open niche.
  - Rejected: static top-N hot set (misses workload drift); pure LRU (ignores gather cost asymmetry).
  - Contract: `IResidencyTier` RAM implementation + tier stats on the metrics bus.
- **DN-M5-6** — PLE: n-gram table as an `IResidencyTier` pageable table with indexed gather (row-offset addressing); sidecar index built at load only if the GGUF tensor layout doesn't align rows.
  - Contract: `IResidencyTier` PLE provider; load-time index builder.
- **DN-M5-7** — MTP: draft-then-verify with accept-rate accounting; draft tokens verified in one batched forward; accept rates reported per model on the metrics bus.
  - Rationale: archive's top lever (12→25 tok/s on copy/code); verification batch amortizes expert reads.
  - Rejected: self/n-gram drafting at this phase (deferred to backlog); free-running draft without verification (quality risk).
  - Contract: `lair.sampling` drafter + scheduler verify step.

#### Implementation Items

- [_] Reality check — llama.cpp/ik_llama.cpp harness on target Flash-Next GGUF (see DN-M5-1)
  - Verify: committed bench report with tok/s, bytes/token, and expert-gather observations on the dev box and, if reachable, the Xeon box.
- [_] Router + gather — `IGatherKernel`, contiguous expert slabs (see DN-M5-4)
  - Verify: `dotnet test --filter Category=Moe` — gathered expert output matches dense reference math; `bench/` reports gather tok/s vs dense-equivalent.
- [_] RAM hot tier — capacity/cost-model expert residency (see DN-M5-5)
  - Verify: bench — hit-rate and tok/s improve over prefetch-only baseline on a replayed workload trace.
- [_] PLE — pageable n-gram table, huge-page backed; indexed disk gather (row-offset addressing; sidecar index if the quant format doesn't align rows) (see DN-M5-6)
  - Verify: `dotnet test --filter Category=Ple` — indexed gather matches in-memory reference; `bench/` reports disk-gather latency distribution.
- [_] GDN/QSA — recurrent state budgets, sparse attention semantics
  - Verify: layer-level parity fixtures vs llama.cpp qwen4exp reference activations for GDN and QSA layers.
- [_] MTP drafting — speculative decode with accept-rate accounting (see DN-M5-7)
  - Verify: parity — MTP output tokens match non-drafted greedy output; `bench/` reports accept rate and tok/s multiplier.
- [_] Flash-Next parity — full-model output match vs llama.cpp/ik reference
  - Verify: `dotnet test --filter Category=Parity` on Flash-Next GGUF (tolerance rules from the reality-check report).

#### Tests

- [_] Automated Moe: gather correctness vs dense reference math
- [_] Automated Ple: indexed gather correctness + latency distribution
- [_] Automated Layers: GDN/QSA activation parity fixtures
- [_] Automated MTP: drafted-vs-greedy output equivalence, accept-rate reporting
- [_] Automated Parity: Flash-Next end-to-end token match within documented tolerance
- [_] Manual Flash-Next Serve: full model serving on dev box with monitor showing expert tier hit rates

#### Open Items

- [_] **qwen4exp GGUF layout** — exact tensor naming/shaping for router, experts, PLE, GDN state, and MTP head in target GGUF files; resolved by a loader spike at the start of MVP5 build, feeding DN-M5-4/5/6 contracts.

### MVP6 [v0.6.0] - Tentative

Residency spans RAM, VRAM, and NVMe.

- NVMe cold tier — colibri-style expert streaming
- Conversation persistence — idle KV block runs snapshotted to disk, restored on resume
- VRAM hot tier — experts promoted onto the 4090
- Cross-tier prefetch — router-prediction-driven promotion
- Slab migration — promote/demote experts between RAM, VRAM, and NVMe while serving, refcounted and double-buffered
- Budget rebalancing — tier ratios adjustable at runtime

### MVP7 [v0.7.0] - Tentative

Vision input on chat models.

- Vision encoder — SigLIP/Qwen-VL-class tower loaded via `IModelLoader`
- Image pipeline — preprocessing, patch embedding, visual-token injection into chat
- Vision families — Qwen3.8-Flash-Next image input, Qwen-VL-class models
- Vision parity — golden tests against llama.cpp vision outputs

## Backlog

| Type | Item | Description |
|---|---|---|
| idea | Anthropic API compat | Strata exposes both surfaces; LAIR-native later |
| idea | AVX-512 dispatch path | +10–30% decode, 2–4× prefill on capable CPUs |
| idea | Native AOT packaging | ~50 ms vs ~500 ms startup |
| idea | Early-exit argmax | skip vocab rows; ~5–8% on 248K-vocab LM head |
| idea | IQ4_XS kernel | shuffle-heavy on AVX2 — measure before committing |
| idea | NUMA pinning | dual-Xeon production boxes |
| idea | Quant conversion tooling | only if the published-quant story breaks |
| idea | Vulkan/Metal backends | deferred; LAIR commits to CPU + CUDA |
| idea | Multimodal image input | Strata has it; LAIR later |
| idea | Embeddings model families | ModernBERT-class GGUF quants, pooling modes |
| question | License | MIT / Apache 2.0 / private — dotLLM is GPLv3: study ideas, never copy code |
| bug | 2DPC DIMM derate | production box derates 2400→2133 MT/s; more RAM ≠ speed |
| question | CUDA binding choice | ILGPU vs LibraryImport+cuBLASLt vs nvcc microkernels (MVP3 open item) |
| question | PLE index format | does the GGUF PLE tensor allow random row access, or is a repack sidecar index needed? (MVP5 open item) |
| idea | Conversation cache policy | TTL/LRU eviction rules; optional encryption of disk snapshots; cache invalidation on model/config change |
| idea | Client-keyed cache retention | tie prompt/conversation cache items to client identity — e.g. a sticky id issued in a response header — and use it for sliding retention (keep each client's most recent conversation cache item) instead of or alongside LRU; precondition: a reliable-enough client identification method (issued token header, optional API key/session, connection fingerprint); interacts with server-posture backlog |
| idea | Byte-reduction experiments | MoD/layer skipping, expert pruning/distillation, dynamic quant switching — each as an `IWeightSource` provider |
| idea | Dynamic layer rebalance | move whole layers between CPU and VRAM while serving (Strata-style), driven by measured device utilization |
| question | Server posture | default bind address, optional API key, LAN exposure rules |
| question | CPU power/temperature source on Windows | RAPL MSRs and CPU temp need kernel access — options: small signed helper service, HWiNFO shared-memory bridge, or omit on Windows and use Linux hwmon/RAPL for the Xeon boxes |
| idea | RAM temperature via IPMI | Xeon production box may expose DIMM temps over BMC/IPMI; absent on the dev box |
| idea | MTP on dense models | pull-forward candidate — MTP heads exist on some dense GGUFs; would move drafting from MVP5 toward MVP2/3 |
| idea | KV evict-and-recompute | retain prompt tokens, evict cold KV blocks, re-prefill on demand — a per-model speed/memory knob; precedent: llama.cpp `--cache-reuse`; recompute is compute-bound on CPU, so never a default |
| idea | Preemptive KV swap | preempt a running sequence's KV to the NVMe tier under memory pressure — depends on MVP6 tiers |
| idea | Prefill GEMM shim microkernel | tiled-matmul native microkernel via `[LibraryImport]` where the bench harness proves the managed tile wall |
| idea | Core-set overcommit | allow models to share/partially oversubscribe core sets as an explicit knob |
| idea | Cross-model interleaved batches | multi-model serving where batch steps alternate across models at configured ratios (e.g. 2:1) so all models share one compute/thread pool; per-model throughput degrades proportionally to share — explicit toggle with per-model ratios, never the default (DN-M2-6 core-set partitioning stays default); MVP4 scheduling candidate |
| idea | `auto` device fallback | silent CPU fallback when a configured CUDA device is absent — deliberately not a default (DN-M3-3) |
| bug | Q4_K_M family coverage gap | kernels implement only the Q4_K block format; real Q4_K_M files mix Q5_K/Q6_K tensors, and non-block-divisible dims force legacy formats (Q4_0/Q5_0…) — loader/kernel extension needed before serving published quants in production |

## Decision Log

- **2026-10-31 — Plan created.** Dense-first phasing; GGUF-first with GGUF-embedded tokenizer; managed-first kernels with `[LibraryImport]` escape hatch.
- **2026-10-31 — Admin surface is config-file hot-reload only.** No admin API; single-operator local runtime.
- **2026-10-31 — Paged KV cache.** Device-backed block pools; runtime resize by budget change.
- **2026-10-31 — OpenAI-compatible API confirmed** as the end-user surface, `/v1/embeddings` included at MVP4.
- **2026-10-31 — AVX2 baseline.** Confirmed by research: i9-13900K (Raptor Lake) and E5-2683 v4 (Broadwell-EP) both lack AVX-512; AVX-512 is optional dispatch only.
- **2026-10-31 — CUDA at MVP3 with the `IBackend` seam mandatory in MVP1.** Dev box (13900K/4090) exercises all paths; CPU-only Xeon deployment justified by the PCIe no-offload tax finding.
- **2026-10-31 — Per-model placement policy.** Device, `gpu_layers`, prefill target, KV residency (`dram`/`vram`/`split`), VRAM budget, thread quota — config-declared, hot-reloadable.
- **2026-10-31 — Knobs-first principle.** All placement/residency combinations are supported and bench-documented; defaults are measured-optimal; users may trade speed for memory footprint.
- **2026-10-31 — .NET 10 target with an accepted kernel gap.** Decode penalty vs C/C++ is ~12–34% (bandwidth-bound, proven by dotLLM at 66–88% of llama.cpp); prefill penalty 2–5× until the register-pressure wall is broken; worst case capped by `[LibraryImport]` microkernels at C-level cost. Orchestration, scheduling, and concurrency — LAIR's differentiators — favor C# over C and Python outright.
- **2026-10-31 — Performance thesis vs incumbents.** CPU dense decode: parity with llama.cpp is plausible (bandwidth-bound; dotLLM already 66–88%). CPU prefill: parity only via `[LibraryImport]` microkernels, not pure managed. GPU: match vLLM's serving features (batching, paged KV, prefix caching), not its CUDA kernels. Over-perform target: CPU MoE decode via capacity/cost-model residency — mainline llama.cpp is weak there (ik_llama.cpp +15–160%) and the niche is open.
- **2026-10-31 — License undecided.** Private repo for now; dotLLM is GPLv3 — its ideas may be studied, its code may not be copied into LAIR.
- **2026-10-31 — MVP2–MVP5 refined.** Batch-step scheduling with decode priority and chunked prefill; hot-reload knob mutation classes; CUDA via nvcc microkernels + cuBLASLt with contiguous layer split and fail-fast device policy; multi-model admission via global ledger with no runtime preemption until MVP6 (backlog: preemptive KV swap); Flash-Next quants IQ3_XXS primary with Q4_K_M baseline behind a reality-check gate; expert-aware batching as the MoE engine win; MTP-only drafting at MVP5.
- **2026-11-01 — Parity criterion is logit-distribution agreement, not exact token match.** Evidence from the pinned llama.cpp b12032 build: its own CLI and server modes produce different greedy (temp 0) continuations for identical prompts on SmolLM-135M-F16, and near-tie argmax flips (Δlogprob < 0.2) appear between any two independent fp pipelines. Cross-tool exact 48-token string equality is therefore unattainable; fixtures record position-0 top-5 logprobs from pinned server runs with a documented tolerance, and divergences inside the near-tie band are reported rather than failed.
- **2026-11-01 — Concrete latency budgets for batching gates (DN-M2-9).** Decode ITL p95 ≤ 400 ms at N=8 concurrent SmolLM-class requests; chunked prefill must hold short-request ITL within the same budget with an 8K-token prompt in flight. Unnumbered "within budget" verify lines are not gates; baselines measured on the dev box (single-stream ~70–105 ms/token).
- **2026-11-01 — Hot-reload application path added to MVP2.** Scope promises live reconfiguration and DN-M2-5 defines mutation semantics, but no implementation item owned the watcher/change-application bus; MVP2's outcome ("live reconfiguration") requires it in-phase rather than orphaned.
- **2026-11-01 — Cross-model interleaved batches recorded in Backlog.** User-proposed scheduling knob: batch steps alternate across models at configured ratios so models share one compute pool, with throughput degradation as the explicit, user-chosen cost; toggle + per-model ratios; MVP4 candidate; default stays core-set partitioning (DN-M2-6).
- **2026-11-01 — Client-keyed cache retention recorded in Backlog.** User-proposed: cache items tied to a client identity (e.g. sticky id returned in a response header) enable sliding retention — each client's most recent conversation cache item kept — instead of or alongside LRU; hinges on a reliable client-identification method, which itself depends on the server-posture backlog.
- **2026-11-01 — Latency gates are measured under Release configuration.** Debug JIT inflates kernels ~4.6× (bench Q4_K: 2.4 GB/s Debug vs 13 GB/s Release); absolute ITL budgets only reproduce in optimized builds. Correctness tests run in both configurations; performance acceptance runs `-c Release`.
- **2026-11-02 — Diffusion support and server-side MCP injection recorded in Backlog.** User-proposed: diffusion as a phase-sized addition (MVP8 candidate) requiring a Scope change since image generation is an explicit v1 non-goal; per-model config-declared MCP servers whose tools LAIR injects into requests and executes server-side, enabling cross-model composition (LLM → diffusion via MCP; embeddings engine as vector/RAG MCP). The injection contract deliberately cuts against DN-M2-8's client-side-tools rule — a future phase must reconcile the two explicitly.
