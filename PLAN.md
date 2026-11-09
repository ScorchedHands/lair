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
- Per-model placement policy: compute device (`cpu` | `cuda` | `hybrid` with layer split), prefill target, weight tier (`dram` | `vram` | `nvme`), KV residency (`dram` | `vram` | `split`), VRAM budget, CPU core/thread quota, prefill lane, and a per-model concurrency cap — compute device and weight tier are independent axes
- Config-file hot-reload for every runtime knob; no admin API
- CUDA backend with Strata-style hybrid CPU/GPU inference
- MoE residency for Qwen3.8-Flash-Next: router, expert gather, RAM hot tier, pageable PLE with indexed disk gather, MTP drafting
- Tiered residency: NVMe cold expert tier, VRAM hot expert tier
- PCIe-frac compute: DRAM-resident expert slabs executed on the GPU (Strata's `pcie-frac`) — GPU compute without spending VRAM on weights, streamed over PCIe with double-buffered prefetch
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
    C[Clients] -->|OpenAI-compatible HTTP + SSE| API[lair.http]
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

**Backend seam** — all compute behind `IBackend`; `CpuBackend` default, `CudaBackend` arrives MVP4 but the contract exists in MVP1.
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
- Contract: prefill chunks interleave with decode tokens in the same batch step by default, so a long prompt never stalls decode latency for other sequences; an opt-in dedicated prefill lane removes the sharing entirely (see MVP3.13).

**Request pipelines and device routes** — every request is an independent pipeline (transform → prefill → decode → finish) with its own quota, advanced independently up to the model's concurrency cap; a *step* is a merged batch over the pipelines one device route can serve, and a route is an ordered set of device segments each carrying its own compute device, weight tier, and quota.
- Rationale: the pipeline is the control plane and must not assume a device; merging is the execution plane's optimization for bandwidth-bound work and is optional where weight reuse is already free (prefill, GPU-resident decode). Keeping the two separate is what stops a CPU-decode performance decision from becoming the scheduler's identity.
- Rejected: one thread per request each traversing weights (bandwidth thrash, no weight sharing — unchanged); the current single global step loop as the only scheduling shape (cannot express a hybrid route or a dedicated prefill lane); device identity as the merge key (a hybrid model's sequences share one route and must still merge).
- Contract: a pipeline's token stream is bit-identical whether merged or run alone; the arbiter admits a step on route cost (CPU core quota and GPU stream/SM quota held for the step's duration), and lane priority applies per segment.

**Compute device and weight tier are separate axes** — a route segment states where the math runs and where the bytes live; GPU compute over DRAM-resident weights (PCIe-frac) is a first-class combination, not a fallback.
- Rationale: Strata's `pcie-frac` runs expert FFNs on the GPU while the expert slabs stay in DRAM, which is the mode that makes GPU compute useful when VRAM — not compute — is the scarce resource; conflating the axes in `device: hybrid` forecloses it, and any later accelerator is just another value on the same axis.
- Rejected: weight-offload-only hybrid (spends scarce VRAM on weights that KV wants); frac as a default anywhere (PCIe and DRAM bandwidth are comparable, so the win is situational and must be measured); streaming weights to GPU then copying them back per token.
- Contract: `PlacementPolicy` expresses compute device and weight tier per segment; frac is bench-gated before becoming a default, consistent with the PCIe-tax posture below.

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
- `lair.http` — serving layer: OpenAI-compatible endpoints, SSE streaming, request lifecycle, model router
- `lair.contracts` — contracts-only leaf: `IBackend`, `IWeightSource`, quant formats, tensor shapes (engine/sequences/continuous batching live in `lair.models` + `lair.kv`)
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
  - Contract: template render in `lair.http` request pipeline; `lair.tokenizer`.
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
  - Contract: `lair.http` middleware.
- **DN-M1-8** — Kestrel binds loopback by default; authentication deferred (backlog: server posture).
  - Rationale: local runtime default posture; exposure is an explicit later decision.
  - Contract: host binding config.
- **DN-M1-9** — Model loading is fail-fast: missing/corrupt/OOM model aborts startup with an actionable message; `/healthz` readiness reflects model state.
  - Rationale: no half-ready engine state to reason about.
  - Rejected: lazy load on first request (hides load failures behind request errors).
  - Contract: host lifecycle + `/healthz`.
- **DN-M1-10** — Every request carries `x-lair-request-id`; JSON structured logs correlate on it.
  - Rationale: correlation IDs must exist before multi-request phases make logs unreadable.
  - Contract: `lair.http`/`lair.host` logging.

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
  - Verify: `dotnet test --filter Category=Template` — 20/20 ✅ golden renders (ChatML = real SmolLM `tokenizer.chat_template`, conditional if/else + `|tojson`, Phi-style), subset semantics (concat, `==`/`!=`, `not`, `tojson`/`trim`, `{%- -%}`), and rejections naming construct + position (`{% set %}`, `|upper`, macro, slices). The real Llama-3 template needs `{% set %}`/loop vars — out of subset (pulled forward as MVP3.2).
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
  - Verify: `lair.contracts` compiles with zero project references — no reference to `lair.backends.cpu` ✅; alternate `IWeightSource` provider end-to-end proof ran with the Parity item (`ArenaWeightSourceTests`: tensors copied into backend-allocated aligned blocks, token sequence identical to mmap-backed decode) ✅.
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

### MVP2 [v0.2.0] - Shipped

Many requests at once, tuned, with live reconfiguration.

#### Design Notes

- **DN-M2-1** — Discrete batch steps: the scheduler runs fixed steps; each step executes one token per active decode sequence plus prefill chunks up to `max_batch_tokens`.
  - Rationale: step boundaries are the natural point for admission, cancellation, metrics, and hot-reload application.
  - Rejected: pipelined overlap across steps (complexity without CPU benefit).
  - Contract: scheduler in `lair.models`.
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
- **DN-M2-7** — Metrics bus: engine counters (tok/s per model, step time, batch occupancy, cache hit rate, KV utilization) plus OS counters into an in-memory ring and JSON logs; CLI monitor consumes it now, MVP3's Web UI consumes the same bus later.
  - Contract: `lair.monitor` metrics bus.
- **DN-M2-8** — Tool calls: per-family tool schema injection into the chat template; `tool_calls` parsed from streamed output; tool execution stays client-side.
  - Rationale: coding-agent compatibility (Strata's whole tool story) requires the wire contract, not execution.
  - Contract: `lair.http` request/response pipeline.
- **DN-M2-9** — Latency budget: decode ITL p95 ≤ 400 ms for N=8 concurrent SmolLM-class requests on the dev machine; chunked prefill must hold short-request ITL within the same budget with an 8K-token prompt in flight.
  - Rationale: measured single-stream baseline is ~70–105 ms/token (SmolLM-135M Q8_0, full threads); a concrete ceiling makes batching regressions catchable without machine-specific absolutes.
  - Contract: Concurrency and Prefill test thresholds.
- **DN-M2-10** — Tool-call detection (MVP2 scope): the wire contract is complete even for models without tool-trained templates. Schemas are injected into the rendered prompt per template family; a call is recognized ONLY when decoded output contains a fenced ```json block whose object matches `{name, arguments}` against a declared tool name (`arguments` JSON-encoded string). Recognized calls appear in `message.tool_calls` / streaming deltas and are stripped from visible text. Execution stays client-side. A model that never emits the contract simply returns ordinary text — valid behavior, tested at parser level, not demanded of small models.
  - Rationale: tool compliance is a property of the model, not the runtime; the runtime guarantees injection, detection, and wire shape.
  - Contract: `lair.http` tool pipeline + template injection.
- **DN-M2-11** — Hot-reload KV budget semantics under pinned arenas: the pool's physical arena is reserved once at startup; `kv_blocks` hot-edits operate as a soft cap within that reservation — grow raises usage up to reserved capacity immediately, shrink drains blocks above the new cap through prefix-cache eviction (never touches active sequences) and tightens admission. Physical re-reservation belongs to the tiered-pools phase, not MVP2.
  - Rationale: mid-serving arena reallocation over live block tables is exactly what step-boundary safety exists to avoid; a soft cap delivers the observable hot-reload behavior without engine risk.
  - Contract: BlockPool soft-cap + PrefixCache eviction interplay at reload apply time.

#### Implementation Items

- [✓] Continuous batching — shared decode steps across sequences (see DN-M2-1)
  - Verify: `dotnet test -c Release --filter Category=Concurrency` — 5/5 ✅ on real SmolLM-F16: N=8 concurrent outputs identical to sequential for greedy AND seeded; measured ITL p95 238 ms (budget 400); capacity exhaustion surfaces as `engine_overloaded`. Scheduler fixes found in testing: finished sequences must be excluded from the same step's decode window, and outcomes release KV blocks before publishing.
- [✓] Chunked prefill — prefill chunks interleaved with decode tokens in the same step (see DN-M2-2)
  - Verify: `dotnet test -c Release --filter Category=Prefill` — 1/1 ✅ ~407-token prompt prefills chunk-by-chunk while short requests stream past it at ITL p95 68 ms with outputs identical to a fresh-engine baseline. Tuning finding: per-step time ≈ chunk width × single-stream cost, so the DN-M2-2 default (512) overshoots the CPU budget — chunk widths of ~8 hold 400 ms at measured Release speed (~28 ms/token F16); wide chunks only pay off on faster compute.
- [✓] Prompt caching — reference-counted prefix blocks reused across requests (see DN-M2-3)
  - Verify: `dotnet test -c Release --filter Category=Cache` — 6/6 ✅ on real SmolLM-F16: shared 127-token prefix → second request skips 112 prefill tokens with byte-identical output vs cold run; CoW copy verified bit-safe for the original sequence; LRU eviction never touches blocks referenced by active sequences; refcount release ordering tested. Cache lifetime = pool lifetime (covers reload/quant/block-size invalidation this phase).
- [✓] Cancellation — abort sequences mid-batch-step without stalling the rest of the batch (see DN-M2-4)
  - Verify: `dotnet test -c Release --filter Category=Cancel` — 6/6 ✅ concurrent peers bit-identical to a no-cancel baseline; every block of an aborted sequence reclaimed (mid-decode, mid-prefill-chunk, and at-admission cases); stop literals now end the sequence early via a linked CTS (gap found: first-token stops previously no-op'd). HTTP client-abort contract covered live (`RequestAborted` already wired through both API paths; new test proves server stays healthy after mid-stream hang-up).
- [✓] API completeness — tool calls, logprobs, penalties, `logit_bias`, seed (see DN-M2-8, DN-M2-10)
  - Verify: `dotnet test -c Release --filter Category=Api` — 49/49 ✅ incl. 14 sampling-param cases (penalties alter seeded output deterministically; logit_bias flips the first greedy token, numeric AND token-string keys; logprobs normalized with chosen token included) and 24 tool cases (schema injection proven by prompt growth, `tool_choice: none` inertness, strict fenced-JSON detector incl. pretty→compact arguments, malformed-tools 400s naming fields, SSE complete-call deltas + `[DONE]`). Model-side call emission intentionally not demanded of the fixture (DN-M2-10). OpenAI SDK round-trip ✅ at ship: chat, sampling params, streaming deltas, completions, tools, models.list all parse through the official Python SDK.
- [✓] Metrics bus — engine counters + OS/driver stats streamed to logs and CLI monitor (see DN-M2-7)
  - Verify: `dotnet test -c Release --filter Category=Metrics` — 5/5 ✅ step-boundary counters (tok/s, step time, occupancy, prefill chunks/tokens, cache hits + skipped tokens, KV utilization, finish-reason tallies) ride the scheduler regardless of consumers; ring-bounded snapshots published only while active (`metrics_snapshot={…}` JSON lines alongside unchanged request logs); `/monitor/stream` SSE live-verified; `lair monitor [-u URL] [--interval ms]` renders a live table, clean Ctrl+C. Process working-set + CPU% included; hardware telemetry deferred per DN-M2-7 contract note.
- [✓] Embeddings groundwork — encoder-only forward pass, no endpoint yet
  - Verify: `dotnet test -c Release --filter Category=Embeddings` — 8/8 ✅ two-layer synthetic GGUF (tied head lets tests replay logits from Encode rows) + real SmolLM-F16 cases; sequential one-token encodes match incremental decode bit-for-bit while full-window encoding legitimately diverges at earlier positions — bidirectional wiring proven distinct from causal at op and run level. Mean/CLS pooling with L2 normalize (double accumulation). Endpoint deferred to the embeddings phase per DN-M5-3.
- [✓] Thread pools — sized to memory channels, per-model quotas, core-set affinity (see DN-M2-6)
  - Verify: `dotnet test --filter Category=Affinity` — 4/4 ✅ pinned workers land inside the configured core set across ≥100 sampled job executions (`SetThreadAffinityMask`, re-applied on wake; documented no-op + startup warning off-Windows); physical-core auto-sizing (SMT siblings excluded via `GetLogicalProcessorInformationEx`; i9-13900K: 24 workers vs old logical cap 16) with ProcessorCount fallback; `placement.thread_quota` caps effective workers (output byte-identical under quota); cross-model core-set overlap rejected at config load naming both models.
- [✓] Hot-reload application path — config watcher pushes validated changes through the change-application bus by mutation class: live knobs apply at the next step boundary, new-requests-only knobs affect admission only, restart-only knobs are refused with an actionable message (see DN-M2-5)
  - Verify: `dotnet test --filter Category=HotReload` — 6/6 ✅ on real SmolLM-F16: debounced watcher (atomic-rename settle) → typed bus (`ConfigMutationTable.Diff`); live `max_batch_tokens` effective at next chunk issue, quota applied at next kernel call, `kv_blocks` soft-cap grow immediate / shrink drains evictable cache via `TrimToBudget` with admission windows so mid-run shrinks never strand admitted sequences; new-requests-only context edits recompute the advertised window only (in-flight byte-compare identical); restart-only refusal names field + class (`engine.block_size requires restart (restart-only): 16 -> 32. ...`); broken YAML rejected pre-application, runtime untouched. `config_reload_*` log lines + `LastReload` summary exposed.

#### Tests

- [✓] Automated Concurrency: parallel-vs-sequential output equivalence — 5/5 (Release) incl. ITL budget and capacity refusal
- [✓] Automated Prefill: chunked prefill ITL protection — 1/1 (Release)
- [✓] Automated Cache: prefix hit, CoW integrity, LRU eviction — 6/6 (Release)
- [✓] Automated Cancel: abort isolation and block reclamation — 6/6 (Release) incl. HTTP client-abort health
- [✓] Automated HotReload: watcher detection, mutation-class enforcement, mid-serving application — 6/6 (Release)
- [✓] Automated API: tool calls, logprobs, penalties, seed reproducibility via SDK scripts — 49/49 (Release); SDK round-trip ✅ at ship
- [✓] Manual Monitor-lite: `lair monitor` live stats under mixed load — verified live: table rendered against serve under shared-prefix chats + long completion; KV used rose 3→55/96 with `active=1` during decode, cache-hit % climbed 50→100% on prefix reuse, tok/s + step-ms updated per tick (Debug-JIT rates), clean exit.

### MVP3 [v0.3.0] - Shipped

Operators see the engine live in a browser and can test it with a built-in chat UI — no external client needed.

#### Design Notes

- **DN-M3-1** — Monitor Web UI: lightweight static SPA (TypeScript, esbuild) served by Kestrel at `/ui`; live stats over SSE reusing `/monitor/stream` with `Last-Event-ID` replay from the snapshot ring.
  - Rationale: the engine stays free of frontend runtime coupling; the metrics bus already streams SSE, and the monitor is read-only — bidirectional transport buys nothing.
  - Rejected: WebSocket/SignalR (bidirectionality unused while the monitor is read-only; new framing/hub machinery); Blazor (runtime coupling to engine host); polling-only UI (fails the real-time requirement).
  - Contract: `lair.monitor` event schema over SSE (additive-only, versioned field), including per-request tok/s sample series for in-flight/recent request sparklines.
  - Carried forward from the multi-model phase design (formerly DN-M4-4) when the UI was pulled ahead of CUDA.
- **DN-M3-2** — Chat test UI: a hand-rolled OpenAI-compatible SSE client; shadcn primitives available as copy-in components but none ended up required (the mockup's DOM with token aliases sufficed); the Chat tab's model selector lists LLM (chat-type) models only.
  - Rationale: for a test client, a chat-runtime library's machinery (thread persistence, branching, cloud hooks) is dead weight; hand-rolling the SSE consumption doubles as a conformance check of LAIR's own API. License was never the differentiator — assistant-ui and shadcn are both MIT (verified 2026-11-02).
  - Rejected: assistant-ui (MIT, but pulls the Vercel AI SDK runtime tree for features this surface won't use); fully hand-rolled (shadcn primitives provide accessibility and composition for free).
  - Contract: chat surface consumes only public API endpoints (`/v1/chat/completions` SSE, `/v1/models`); no privileged engine access — the UI is a client like any other.
- **DN-M3-3** — UI information architecture: "Configured models" is the primary panel — one row per model with per-model stats (decode tok/s, seqs/step, KV used/cap, cache hit, workers, VRAM reserved) and expandable rows revealing config + state (context/block size, kv budget, core set/quota, prefill·decode·residency, cache blocks by location per model, finish-reason tallies, expert tier counts when MoE exists, model RAM). Request rows (in-flight + recent) carry tok/s + inline sparkline. GPU/PCIe/power/temperature tiles render as placeholders until the CUDA phase.
  - Rationale: stats belong where they're attributable; engine-level KPI cards stay few, model detail lives on the model rows (owner review 2026-11-02).
  - Contract: event schema fields are additive — later phases add tiles without reshaping existing ones; scheduler exposes per-sequence token-timestamp series.
- **DN-M3-4** — Tabbed shell, shared stylesheet, locked dark developer-tool theme: one SPA, two tabs (Monitor, Chat), one design system; the approved mockup is the verbatim spec for production code.
  - Rationale: one design system keeps the operator surface coherent and lets the chat tab double as a live test client for the monitor's data.
  - Contract: `mockup/mvp3/` holds the approved mockups (`mvp3-tabbed-shell.html` rev B + `mvp3-shared.css`); production SPA derives from them verbatim.
- **DN-M3-5** — Live data semantics: SSE reconnect replays from the 300-snapshot ring via `Last-Event-ID`; live tail coalesces to latest-per-subscriber with a bounded drop policy for slow consumers; recent-request ring bounded by a config knob (`ui.recent_requests`, live mutation class); UI state is volatile — no persistence.
  - Rationale: unbounded per-subscriber channels are a memory-growth risk under slow clients; replay must survive refresh/reconnect without engine work.
  - Contract: EngineMetricsBus subscriber channels gain latest-only coalescing + bounded queue; `ui.recent_requests` hot-reloads live.

Security (accepted): SPA is static files on the existing Kestrel endpoint (loopback default; exposure governed by the server-posture backlog); monitor is read-only; chat uses only public endpoints with the established error taxonomy; no secrets in or behind the UI. Data (accepted): no new persistence — rings are in-memory and volatile; recent list capped at X, no pagination needed at this scale. Contracts (accepted): versioned additive event schema; SSE reconnect semantics via `Last-Event-ID`; idempotency N/A for read-only streams. Observability (accepted): request rows key on the existing `x-lair-request-id`; UI status badge reads `/healthz`; server logs gain connect/disconnect lines for stats subscribers. Resilience (accepted): EventSource auto-reconnect + ring replay; engine restart shows a disconnected state and recovers; quiet-while-prefilling streams show elapsed time rather than stalling. Compliance: N/A — single-operator local runtime; RBAC deferred to server posture. Deployment (accepted): SPA built by pinned npm toolchain, embedded as static assets in the Host build; rollback = app revert; no feature flags. Scale (accepted): fan-out bounded by coalescing + drop policy; snapshot cadence ~2 s while active is trivial load; rate limiting governed by server-posture backlog.

#### Implementation Items

- [✓] Tabbed shell + shared stylesheet mockups — Monitor tab and Chat tab in one design system (`mockup/mvp3/`)
  - Verify: manual — approved 2026-11-02 (rev B): configured-models panel, expandable rows, sparklines, slimmed KPI cards, dark theme locked.
- [✓] SPA build pipeline — esbuild + shadcn init with pinned npm dependencies; built assets embedded in the Host's static file serving at `/ui`
  - Verify: `dotnet test --filter Category=WebUi` — 1/1 ✅ `GET /ui` + `/ui/index.html` return 200 `text/html` shell with bundled `app.css`/`app.js`/`app.js.map` served; `/ui/chat` falls back to the shell; non-UI 404s preserved; embedding runs from clean state, incremental builds skip npm when web sources are unchanged, `-p:SkipWebUi=true` builds clean without Node. Pinned: react 19.3.1, esbuild 0.28.2, typescript 7.0.2, tailwindcss 4.3.3 (+CLI), npm 11 lockfile committed. Mockup rev B reproduced verbatim as React shell (jsdom-verified 18/18 interactivity checks: tabs, row expand/collapse); design tokens aliased to shadcn CSS-variable names via Tailwind v4 `@theme inline`.
- [✓] Event schema + stats endpoint — versioned SSE event schema (per-model snapshots, per-request tok/s series, request ring bounded by `ui.recent_requests`) with `Last-Event-ID` replay and latest-only subscriber coalescing
  - Verify: `dotnet test --filter Category=MonitorUi` — 7/7 ✅ v1 events (`snapshot`, `request_start/update/complete`, id-less `resync`) on bus-scoped incrementing frame ids; per-request tok/s series from a zero-alloc 64-slot sample ring driven at the existing step-boundary hook (updates at token 1 then every 8th); `ui.recent_requests` ring (default 25, validated 1..1000) hot-reloads live — shrink narrows visibility, grow re-seats the retained tail; Last-Event-ID replays retained snapshots oldest-first with resync fallback for unknown/stale/caught-up ids; per-subscriber snapshot coalescing (capacity-1 latest-only) + bounded 200 request queue with drop counters on snapshots. `Category=Metrics` 5/5 unchanged — additive contract held, CLI monitor untouched.
  - Verify: automated (`Category=MonitorUi`) — a test client receives live snapshots during load with versioned field names, replays missed events after reconnect via `Last-Event-ID`, and the request ring never exceeds X.
- [✓] Monitor SPA — implements the mockup's Monitor tab verbatim: KPI cards, configured-models table with expandable config/state rows, in-flight + recent tables with tok/s sparklines, placeholder tiles (see DN-M3-1, DN-M3-3, DN-M3-4)
  - Verify: manual live serve ✅ — `/monitor/stream` carried snapshots (incl. 10 additive per-model fields), request lifecycle frames with `samples[]`, Last-Event-ID replay (3 snapshots + resync), client-abort surfaced as cancelled; jsdom harness against the built bundle asserts KPI updates, model rows expand on live data, in-flight row lifecycle, sparkline points from samples (7 for 8 samples), badge transitions across reconnect/back-to-live. Targeted: MonitorUi 7/7 + WebUi 1/1. Honest deltas vs mockup statics: step-time sub-line is avg-over-window (no server p95 today); tool_calls tallies absent until fence-derived calls reach the stream; GPU/PCIe/power remain placeholders.
  - Verify: manual — browser shows live in-flight rows updating sub-second under load; model row expansion reveals config + per-model cache blocks and finish tallies; sparklines render for streaming requests; GPU/PCIe tiles show placeholders.
- [✓] Chat tab — hand-rolled OpenAI-compatible SSE client (no copy-in shadcn primitives proved necessary: the mockup DOM with token aliases covers it): LLM-only model selector, streamed multi-turn replies, mid-stream abort, tool-call rendering (see DN-M3-2, DN-M3-4)
  - Verify: manual live serve ✅ — scripted curl SSE capture matched the client's parse shapes end-to-end (role chunk → content pieces → complete-call tool chunks → finish_reason chunk → `[DONE]`); real-bundle harness 11/11 incl. multi-turn replay accepted, mid-stream cut keeping partial text with working next send, pre-header Stop resolving as flow control; jsdom smoke (`npm run smoke:chat`) 29/29 covering selector filtering via new additive `"type"` field on `/v1/models`, delta rendering incl. half-frame buffering, tool card assembly for both complete-call and index-keyed incremental shapes, inline error bubble with usable composer. No shadcn primitives pulled — mockup verbatim rule outranks; no MessageBody change needed (tool cards display-only). Targeted: WebUi 1 + MonitorUi 7 + extended models assertion green.
  - Verify: manual — multi-turn streamed chat against the serving engine; Stop aborts mid-stream and the next request works; a parser-produced tool call renders as a tool card.

#### Tests

- [✓] Automated MonitorUi: versioned snapshot stream, `Last-Event-ID` replay, bounded request ring, subscriber coalescing — 7/7 (targeted)
- [✓] Manual Monitor: configured-models panel first with expandable rows; per-model stats; in-flight + recent rows with tok/s sparklines; placeholder tiles; sub-second updates — verified live ✅ against serve: `/ui` shell 200 with assets (`app.css`, `app.js`) and SPA fallback; stream carried request_start×2/update×3/complete×2/snapshot×2 all id-stamped with token sample arrays during chat+completion load; Last-Event-ID reconnect replayed snapshot + resync; jsdom harness vs built bundle 38/38 (KPI updates, live model-row expansion, in-flight lifecycle, sparkline points from samples, badge offline→reconnecting→live).
- [✓] Manual Chat UI: LLM-only model selector, streamed multi-turn chat, mid-stream abort, tool-call card rendering against the live server — verified live ✅: curl SSE capture matched client parse shapes (role → content deltas → finish chunk → `[DONE]`); real-bundle harness 11/11 on live serve incl. multi-turn replay accepted and genuine mid-stream cut with working next send; jsdom smoke 29/29 (embed model excluded via additive `/v1/models type`, tool cards in both wire shapes, inline error bubble keeps composer usable).

### MVP3.1 [v0.3.1] - Shipped

Published Q4_K_M models load and serve on this machine.

#### Design Notes

- **DN-M31-1** — Quant family coverage: implement loader acceptance + scalar/AVX2 fused-dequant GEMV kernels for Q5_0, Q5_K, and Q6_K, mirroring llama.cpp `ggml-quants.c` reference math, so published mixed-family quant files serve as-is (Q4_K_M puts attn k/v at Q5_0 and `output.weight` at Q6_K; Q4_K_S uses Q5_K).
  - Rationale: consuming published GGUF quants is a Scope commitment; the repo already mirrors llama.cpp quant math for Q4_K/Q8_0 — same pattern extends.
  - Rejected: offline requantization (breaks the GGUF-first commitment); rejection-with-a-nice-error (already exists; leaves the user's own model unusable).
  - Contract: `GgmlType`/`TensorShape` row strides (Q5_0 22 B/32, Q5_K 176 B/256, Q6_K 210 B/256 — llama.cpp `block_q6_k` is 210 B, no padding) + `CpuBackend.DequantRow`/`Gemv`/`GemvBatch` paths.

#### Implementation Items

- [✓] Q5_0 kernel — loader acceptance + scalar oracle + AVX2 fused dequant + batch path (see DN-M31-1)
  - Verify: `dotnet test --filter Category=Kernel` — synthetic-block scalar-vs-AVX2 and 1T-vs-4T bit-identity, `GemvBatch` equals solo `Gemv`. ✅ Kernel|Loader 77/77 Release.
- [✓] Q6_K kernel — same pattern (see DN-M31-1)
  - Verify: same Category=Kernel oracle pattern. ✅ included in the 77/77 batch; type id 14, block 210 B per llama.cpp.
- [✓] Q5_K kernel — same pattern (see DN-M31-1)
  - Verify: same Category=Kernel oracle pattern. ✅ included in the 77/77 batch; type id 13 per llama.cpp.
- [✓] End-to-end Q4_K_M serve — load `models/smollm-135m-instruct-q4_k_m.gguf` (real file on dev box), decode under the engine, serve it
  - Verify: automated — real-fixture test (skip-if-absent) asserts load, finite logits, thread-invariant argmax over an 8-token decode; manual — `lair serve` with the q4_k_m config, chat via API and the `/ui` Chat tab, Monitor tab shows the model live. ✅ Automated: `QuantFamilyFixtureTests` exact quant inventory (Q5_0 x166 tensors incl. all attn_q/k/output + ffn_gate/up; Q6_K/Q4_K split on ffn_down; Q8_0 token_embd) + thread-invariant decode; parity fixture `parity-smollm-q4km.json` pinned to llama.cpp b12032 (sha-pinned model file). Manual: serve live — root 301→/ui, `/v1/models` lists the model, non-stream + streaming chat 200 with streamed deltas, monitor stream carried request_start/update×3/complete/snapshot naming the model.

#### Tests

- [✓] Automated Kernel: Q5_0/Q5_K/Q6_K scalar-vs-AVX2 bit-identity + batch equivalence + real-fixture decode sanity ✅ Category=Kernel|Loader 77→92 with Parity; Release full 320/320
- [✓] Manual Q4_K_M Serve: q4_k_m model served; chat via API + `/ui`; monitor live ✅ verified live on :8392 (see item evidence)

### MVP3.2 [v0.3.2] - Shipped

Published Llama-family chat models serve chat end-to-end on this machine, rendering their own GGUF chat template through a managed full-minja-parity template library — starting with `models/Llama-3.2-1B-Instruct-Q8_0.gguf`.

#### Design Notes

- **DN-M32-1** — The blocker is the template interpreter, not the engine: the Llama-3.2 Q8_0 file satisfies every engine requirement already (architecture `llama` in the supported set, `tokenizer.ggml.pre = llama-bpe`, full-head RoPE with `rope.dimension_count == head_dim`, Q8_0 kernel), and loads cleanly under pinned b12032; LAIR refuses it at startup because `ChatTemplateRenderer` rejects its first construct — `{%- set tools = custom_tools %}` ("expected a tag close", template offset 37). Implement a managed `Lair.Templating.Minja` library with full minja parity for minja’s supported subset, verified against pinned llama.cpp b12032 `/apply-template`.
  - Rationale: Scope commits to dense Llama/Qwen-class ≤3B GGUFs (2026-10-31), and MVP1 recorded this exact gap as out-of-subset; general GGUF compatibility should target the llama.cpp/minja behavior that published templates are validated against.
  - Rejected: narrowed subset (leaves the same failure mode for other published templates); native minja interop (conflicts with LAIR's managed-first, self-contained packaging posture); .NET Jinja libraries (Jinja2-compatible is not minja-compatible without byte-exact proof); vendor llama.cpp/minja (imports a C++ dependency and native build burden); full Python Jinja2 parity (broader than GGUF chat templates and harder to verify against llama.cpp).
  - Contract: `Lair.Templating.Minja` exposes parser/evaluator/chat-template APIs and is verified against prompts rendered by pinned llama.cpp b12032 for identical message sets.
- **DN-M32-2** — LAIR adopts minja’s exact limitations as the MVP3.2 template contract: full expression syntax; statements, variable sections, and comments with whitespace elision; `if`/`elif`/`else`; `for` with `recursive`, `if`, `loop.*`, `break`, `continue`, and destructuring; `set` with namespaces and destructuring; `macro`/`call`/`filter`; minja’s extensible filter/test/function collection; no `{% raw %}`, `{% block %}`, `{% include %}`, `{% extends %}`, no tuples, no ternary-less `if` expressions.
  - Rationale: minja is the GGUF compatibility oracle; broader Jinja features would create behavior LAIR cannot byte-verify against llama.cpp.
  - Rejected: broader Jinja2 target (unverifiable against the GGUF oracle); minja plus LAIR-specific extensions (fragments the compatibility story).
  - Contract: parser errors name unsupported construct + position and fail model startup before serving.
- **DN-M32-3** — `Lair.Templating.Minja` uses a custom `MinjaValue` model: null/none, booleans, numbers, strings, lists, ordered maps, Python-like truthiness, no distinction between `none` and `undefined`, and minja-compatible JSON dump semantics for maps/strings.
  - Rationale: minja byte-exactness depends on Python-like value semantics and its own JSON dump behavior, not ordinary .NET JSON serialization.
  - Rejected: direct `System.Text.Json` values (ordering/dump semantics are not minja-compatible); dynamic/object dictionaries (undefined semantics and dump behavior drift).
  - Contract: `MinjaValue` is the public context/value currency for templates, chat inputs, filters, macros, and capability probes.
- **DN-M32-4** — Scoping follows minja/llama.cpp behavior proven by oracle goldens, including `{% set %}` visibility, namespace objects, re-binding loop inputs, macro/call/filter scoping, and destructuring.
  - Rationale: Llama-3.2 re-binds `messages` through `{% set %}`; fidelity requires matching the engine that published templates are tested against.
  - Rejected: Python Jinja2 block scoping (may differ from minja); simple global assignments (may pass one template but hides future divergence).
  - Contract: scoping tests are derived from minja behavior and pinned llama.cpp `/apply-template` goldens.
- **DN-M32-5** — `raise_exception` is implemented as a minja-compatible function/filter behavior. When reached during chat rendering, it surfaces as an OpenAI `invalid_request` 400 with the template-provided message.
  - Rationale: full minja parity requires `raise_exception`; load-time rejection would reject templates that are valid for the request shape actually used.
  - Rejected: load-time model rejection (too broad); ignoring raises (silently wrong output).
  - Contract: `MinjaRenderException` carries a bounded template error taxonomy; `lair.http` maps it to `invalid_request`.
- **DN-M32-6** — The existing `ChatTemplateRenderer` is replaced by a `MinjaChatTemplate` adapter over `Lair.Templating.Minja`; no fallback mini-Jinja path remains.
  - Rationale: one template engine keeps GGUF behavior observable and prevents silent divergence between old and new renderers.
  - Rejected: keep `ChatTemplateRenderer` as a compatibility facade (preserves the old semantics boundary); fallback path (hides template compatibility failures).
  - Contract: model load parses the GGUF chat template once and either stores a `MinjaChatTemplate` or fails startup with an actionable template error.
- **DN-M32-7** — Template tool precedence: when a model template declares or renders tool support, LAIR does not inject its own fenced-tool block for that request; LAIR injection is used only when the template lacks tool support or an explicit config override requests it. Capability detection is part of the template library output.
  - Rationale: published GGUF templates increasingly own tool formatting; LAIR’s MVP2 fenced-JSON injection is a fallback, not the canonical rendering path for templates that already know how to render tools.
  - Rejected: always injecting LAIR’s tool block (duplicates or corrupts native template tool sections); requiring model-specific hardcoded template handlers (breaks the GGUF-first template story).
  - Contract: `TemplateCapabilities` reports tool/system/tool-call/tool-response/typed-content facts; `lair.http` chooses injection behavior from those facts plus request/config intent.
- **DN-M32-8** — The chat-template wrapper implements minja chat context and polyfills: `messages`, `tools`, `add_generation_prompt`, `bos_token`, `eos_token`, `date_string`, system-role normalization, tool-call/tool-response normalization, typed-content handling, object-argument handling, and extra context.
  - Rationale: GGUF compatibility requires minja’s chat-template layer, not just raw Jinja rendering.
  - Rejected: raw Jinja context only (leaves HF chat quirks to LAIR callers); LAIR-specific facade only (prevents reusable minja parity).
  - Contract: `MinjaChatTemplate.Render(ChatTemplateInput)` returns rendered prompt plus capability facts.
- **DN-M32-9** — BOS/EOS handling avoids double BOS: the adapter renders the template exactly, then suppresses tokenizer BOS only when the rendered prompt already starts with the model BOS token.
  - Rationale: Llama templates emit `bos_token` while LAIR’s llama-bpe tokenizer path may add BOS; both must not appear.
  - Rejected: always add BOS (duplicates Llama BOS); strip template BOS blindly (breaks templates that intentionally emit BOS).
  - Contract: adapter compares rendered prefix, tokenizer BOS metadata, and `add_bos_token` policy before encoding.
- **DN-M32-10** — Untrusted GGUF templates get hard parse/render limits: max template size, parser/eval depth, loop iteration count, output size, and render timeout. Limits have library defaults in MVP3.2; no config knob is introduced yet.
  - Rationale: a full template interpreter is a third-party input attack surface and resource-exhaustion surface.
  - Rejected: no limits (unsafe); deferring limits (MVP3.2 is the first real GGUF template execution path).
  - Contract: `MinjaLimits` are enforced inside `Lair.Templating.Minja` and surface as template parse/render errors.
- **DN-M32-11** — Parsed templates are cached per loaded model; rendering is per request, but parsing happens once at model load.
  - Rationale: model load should fail fast on bad templates and avoid repeated parser work per request.
  - Rejected: parse per request (waste and late failures); lazy first-request parse (hides startup template errors).
  - Contract: `GenerationService` stores the parsed `MinjaChatTemplate` for the model instance lifetime.
- **DN-M32-12** — Template failures are observable but prompt injection remains out of scope: startup parse failures log model path + construct/position; request-time render failures log model name, request id, and bounded template error code/message; LAIR does not sanitize user messages against template-injection tricks in MVP3.2.
  - Rationale: operators need actionable template diagnostics, while message-content injection is a model/security policy problem larger than this phase.
  - Rejected: silent template fallback (hides GGUF incompatibility); full prompt-injection sanitization in MVP3.2 (scope creep without a defined policy).
  - Contract: structured logs use existing `x-lair-request-id` for request-time failures and model-load context for startup failures.
- **DN-M32-13** — Minja-compatible expression values include inline list and map/object literal expressions, not only context-provided lists/maps; templates rely on literals such as `['user', 'assistant']` for membership containers and tool-call metadata. Parenthesized tuple literals remain unsupported because minja treats tuples separately from lists.
  - Rationale: filters work proved that real llama-family templates use inline list containers heavily, so LAIR cannot claim full GGUF template parity until the parser/evaluator can parse them directly rather than requiring every container to be injected by LAIR context.
  - Rejected: context-only containers (breaks published-template parity); treating `(1, 2)` as a supported tuple literal (contradicts minja’s no-tuples limitation).
  - Contract: parser/evaluator support JSON-like list literals `[expr, ...]` and ordered-map literals `{key_expr: value_expr, ...}`; syntax goldens cover them, evaluator preserves insertion order for maps/lists, and comprehension forms are deferred unless oracle evidence requires them.

#### Implementation Items

- [✓] Minja library project — create `src/Lair.Templating.Minja` as a managed template engine with no dependency on `lair.http`, `lair.models`, `lair.tokenizer`, or native libraries (see DN-M32-1, DN-M32-2)
  - Verify: `dotnet build src/Lair.Templating.Minja/Lair.Templating.Minja.csproj` — succeeded with 0 warnings/0 errors; `dotnet list src/Lair.Templating.Minja/Lair.Templating.Minja.csproj reference` shows no LAIR runtime project references. Added to solution under `src`.
- [✓] Lexer/parser/AST — implement minja tokenization, two-phase parsing, comments, whitespace elision, statements, expressions, and unsupported-construct rejection (see DN-M32-2)
  - Verify: `dotnet test --filter Category=MinjaSyntax -c Release` — 50/50 passed after adding inline list/map literal goldens. Parenthesized tuple literals remain rejected as a deliberate divergence from minja’s tuple-as-list behavior until oracle parity requires changing it.
- [✓] `MinjaValue` and evaluator — implement ordered maps, Python-like truthiness, none/undefined equivalence, list/map/string operations, membership, boolean operators, slices, arithmetic/concatenation, and minja-compatible JSON dump (see DN-M32-3)
  - Verify: `dotnet test --filter Category=MinjaEval -c Release` — 103/103 passed; full suite 465/465 green. Missing keys, out-of-range indexes, and undefined attributes evaluate as none; maps cannot be sliced; statements/filters/tests/calls currently raise bounded unsupported errors.
- [✓] Minja statements — implement `if`/`elif`/`else`, `for` with `loop.*`, `recursive`, `if`, `break`, `continue`, destructuring, `set` with namespaces/destructuring, `macro`, `call`, and `filter` with minja-compatible scoping (see DN-M32-4)
  - Verify: `dotnet test --filter Category=MinjaStatements -c Release` — 49/49 passed; full suite green on rerun at 514/514 after one known concurrency timing flake. Loop targets are restored when loops end, recursive flattening is supported via the loop callable, and `loop.cycle()` remains unsupported until oracle evidence requires it.
- [✓] Expression literals — parse and evaluate inline list/map/object literal expressions while keeping parenthesized tuple rejection (see DN-M32-13)
  - Verify: `dotnet test --filter Category=MinjaSyntax -c Release` and `dotnet test --filter Category=MinjaEval -c Release` — 50/50 and 121/121 passed, including list/map literals, nested literals, inline membership containers, insertion-order JSON dump, and non-string map-key rejection. Oracle risks remain: minja accepts primitive non-string dict keys and may treat `(1, 2)` as a list; LAIR currently rejects both.
- [✓] Minja filters/tests/functions — implement minja’s supported filter/test/function collection, including `raise_exception`, `tojson`, `trim`, `length`, `items`, `dictsort`, `join`, `range`, `select`/`reject` variants, `namespace`, `escape`, and chat-template helpers needed by oracle tests (see DN-M32-5)
  - Verify: `dotnet test --filter Category=MinjaFilters -c Release` — 78/78 passed after adding `tojson(indent=N)` support. Built-ins resolve filters/tests/functions through one namespace; `raise_exception` throws `MinjaRenderException` with the template message and offset for later HTTP `invalid_request` mapping. Unknown built-ins remain actionable errors.
- [✓] Chat-template wrapper — implement `MinjaChatTemplate`, `ChatTemplateInput`, `TemplateCapabilities`, chat context normalization, polyfills, BOS/EOS handling, and capability probes (see DN-M32-7, DN-M32-8, DN-M32-9)
  - Verify: `dotnet test --filter Category=MinjaChat -c Release` — 27/27 passed. Wrapper exposes `Prompt`, `Capabilities`, and `StartsWithBos`; capability detection uses dummy-render probes, and polyfills cover typed content, tool-call argument conversion, tool-response fallback, system-role fallback, `strftime_now`, `date_string`, extra context, and `tojson(indent=N)`.
- [✓] Security limits — enforce library defaults for template size, parser/eval depth, loop iterations, output size, and render timeout (see DN-M32-10)
  - Verify: `dotnet test --filter Category=MinjaLimits -c Release` — 36/36 passed. Parse guards fail as `MinjaSyntaxException` (`template size`, `parser depth` with offset); sequence length, loop iterations, output size, evaluator/call recursion depth, the 30 s render deadline and caller cancellation fail as `RenderLimitsException` naming limit + construct + offset; over-budget templates stop in bounded wall time (no unbounded allocation or hang) and just-inside goldens stay byte-exact.
- [✓] LAIR renderer adapter — `GenerationService` parses each model's GGUF chat template once at load into a cached `MinjaChatTemplate` and fails startup with model path + construct + offset on invalid templates; request renders map LAIR chat/tool inputs through `ChatTemplateInput` with the request `CancellationToken`, log model name + `x-lair-request-id` + bounded detail, and surface failures as `invalid_request`; tokenizer BOS is suppressed only when the rendered prompt already starts with the model BOS; load-time `TemplateCapabilities` drive tool precedence — a tool-capable template renders tools itself and LAIR's fenced block is skipped unless `force_fenced_tools` overrides. The legacy `ChatTemplateRenderer` is deleted; no fallback remains. (see DN-M32-6, DN-M32-11, DN-M32-12)
  - Verify: `dotnet test --filter Category=Template -c Release` — 16/16 (four legacy rejection goldens retired with the renderer); `dotnet test --filter Category=TemplateDiagnostics -c Release` — 11/11 including startup/render diagnostics, parse-once caching, cancellation wiring, and tool precedence.
- [✓] Llama-family template parity — LAIR's render of a pinned real chat_template matches llama.cpp b12032's own rendering byte-for-byte across system/user/assistant/tool multi-turn sets with and without `add_generation_prompt`
  - Verify: `dotnet test --filter Category=TemplateParity -c Release` — 14/14 against `tests/fixtures/parity-templates-llama32.json` (12 byte-exact renders + sha256 guards + one documented reference refusal), regenerated by `scripts/generate-template-parity.ps1`; ChatML/Phi goldens stay green.
- [✓] End-to-end Llama-3.2 serve — real file loads, chat routes through its own template, multi-turn conversation stays coherent via API and `/ui`
  - Verify: `dotnet test --filter Category=Serve -c Release` — 2/2 (skip-if-absent: real load + pinned byte-exact prompt + finite deterministic coherent continuation ≤32 tokens). Manual serve verified live on :8181 (see Tests evidence).

#### Tests

- [✓] Automated MinjaSyntax: supported minja constructs, unsupported non-minja constructs, whitespace/comment handling, and parse errors naming construct + position. ✅ `tests/Lair.Tests/MinjaSyntaxTests.cs` — 50 cases passed under `Category=MinjaSyntax`, including list/map literal goldens and tuple rejection.
- [✓] Automated MinjaEval: value model, truthiness, membership, indexing, slicing, concatenation, maps/lists, and minja-compatible JSON dump. ✅ `tests/Lair.Tests/MinjaEvalTests.cs` — 121 cases passed under `Category=MinjaEval`.
- [✓] Automated MinjaStatements: `if`, `for`, `loop.*`, `break`, `continue`, `set`, namespaces, destructuring, `macro`, `call`, `filter`, and scoping goldens. ✅ `tests/Lair.Tests/MinjaStatementsTests.cs` — 49 cases passed under `Category=MinjaStatements`.
- [✓] Automated MinjaFilters: minja filter/test/function collection, including `raise_exception` render-time `invalid_request` mapping. ✅ `tests/Lair.Tests/MinjaFiltersTests.cs` — 78 cases passed under `Category=MinjaFilters`.
- [✓] Automated MinjaChat: chat-template inputs, polyfills, capabilities, BOS/EOS behavior, tool precedence facts, and `add_generation_prompt` variants. ✅ `tests/Lair.Tests/MinjaChatTests.cs` — 27 cases passed under `Category=MinjaChat`.
- [✓] Automated MinjaLimits: template size, parser/eval depth, loop iteration, output size, and render timeout guards. ✅ `tests/Lair.Tests/MinjaLimitsTests.cs` — 36 cases passed under `Category=MinjaLimits`, including non-configurable defaults, cancellation-guard behavior, and just-inside goldens that prove limits do not alter semantics.
- [✓] Automated Template regression: existing ChatML/Phi goldens remain green after replacing `ChatTemplateRenderer`. ✅ `tests/Lair.Tests/ChatTemplateTests.cs` — 16 cases passed under `Category=Template`; four legacy rejection cases were retired with the renderer because the minja parser accepts those constructs now.
- [✓] Automated TemplateDiagnostics: invalid startup templates log model path + construct/position; request-time template render failures log model name, request id, and bounded error code. ✅ `tests/Lair.Tests/TemplateDiagnosticsTests.cs` — 11 cases passed under `Category=TemplateDiagnostics`, including parse-once caching, cancellation wiring, and capability-driven tool precedence.
- [✓] Automated TemplateParity: pinned llama.cpp `/apply-template` byte-exact goldens for Llama-family templates. ✅ `tests/Lair.Tests/TemplateParityTests.cs` + `tests/fixtures/parity-templates-llama32.json` — 14 cases passed under `Category=TemplateParity` (12 byte-exact renders, sha256 guards, documented b12032 trailing-tool-call refusal divergence).
- [✓] Manual Llama-3.2 Serve: model served; multi-turn chat via API and `/ui`; monitor live. ✅ verified live on :8181 — 3-turn exchange coherent (`2 + 2 = 4` / `3 + 3  = 6` / `4 + 4  = 8`), model's own chat_template rendered with single leading BOS, `x-lair-request-id` correlation on every turn, all requests finished `eos`, peak working set 1,588 MB under the 4 GB budget (metrics-reported 1,566 MB steady).

### MVP3.3 [v0.3.3] - Shipped

GGUF tokenizer variants load through load-time tokenizer profiles instead of hardcoded pre-tokenizer rules.

#### Design Notes

- **DN-M33-1** — Tokenizer compatibility is a profile/registry seam selected at model load. The shared BPE machinery remains central; GGUF metadata selects profile behavior for `tokenizer.ggml.model`, `tokenizer.ggml.pre`, special/control token handling, byte fallback, and BOS/EOS/EOT metadata. Variants are internal implementations, not separate runtime assemblies or per-token dispatch.
  - Rationale: published GGUF tokenizers share BPE machinery but differ in pre-tokenization, control-token behavior, and metadata quirks. Selecting a profile at load keeps the hot tokenizer path concrete while making new families additive.
  - Rejected: one hardcoded `GgufTokenizer` switch forever (blocks new families); one library per tokenizer variant (fragmentation without packaging benefit); per-token plugin dispatch (interface tax in tokenization).
  - Contract: `lair.tokenizer` exposes a tokenizer-profile registry selected from GGUF metadata; profiles produce concrete tokenizer state before serving.
- **DN-M33-2** — Profile taxonomy is keyed by `tokenizer.ggml.pre`. MVP3.3 formalizes the three behaviors LAIR already implements (`llama-bpe`, `gpt2`, `smollm`) as load-time profiles; `tokenizer.ggml.model` stays validated as `gpt2` BPE. A missing `pre` key keeps the `llama-bpe` default; an unknown `pre` or model stays a load-time `GgufException` naming the offending value and the supported profile list. Additional profiles (`none`, `nova2`, `llama-code`, decoupled, …) land only when a real fixture plus a `/tokenize` oracle justifies them.
  - Rationale: the three shipped profiles cover both real fixtures on this machine (SmolLM declares `smollm`, Llama-3.2 declares `llama-bpe`); a speculative profile would ship behavior with no oracle to verify it against.
  - Rejected: keying profiles by architecture (pre-tokenization is a tokenizer property, not an architecture one); silent fallback for an unknown `pre` (hides GGUF incompatibility — same posture as DN-M32-2's construct rejections); shipping `none`/`nova2` now (no fixture and no oracle evidence).
  - Contract: `GgufTokenizer.FromGguf` resolves exactly one profile from `tokenizer.ggml.pre` at construction, and the registry is the single source of the rejection message's supported-profile list.
- **DN-M33-3** — A profile is a load-time factory, not a runtime interface: it resolves the pre-split strategy (chosen once at construction), the merge-lookup shortcut (`ignore_merges`), decode space cleaning, and BOS defaults into concrete tokenizer state before serving. The per-call `switch` on the pre-rule inside `PreSplit` disappears; the hot BPE path keeps its current concrete shape.
  - Rationale: profile flags are already constructor-resolved, but split dispatch still switches per pre-token chunk; resolving the strategy once keeps the hot path identical while making profiles additive.
  - Rejected: per-token profile dispatch (interface tax in tokenization); splitting `GgufTokenizer` into per-family classes (duplicates the shared BPE/merge machinery); a public profile plugin API (no third-party tokenizer story in MVP3).
  - Contract: `lair.tokenizer` keeps `GgufTokenizer.FromGguf`/`Encode`/`Decode` as the public surface; profiles stay internal.
- **DN-M33-4** — Tokenizer metadata coverage follows what llama.cpp actually reads: `tokenizer.ggml.byte_fallback`, `byte_to_token`, `byte_to_utf8`, `add_space_prefix`, `eot_token_id`, `utf8_flush_between_controls`. A present `byte_to_token` replaces the hardcoded `<0xNN>` vocab scan; `add_space_prefix` is read and exposed, and the pinned b12032 `/tokenize` oracle shows the gpt2/smollm pre-tokenizer keeps the folded leading space even when the key is false (SmolLM declares `false`; the oracle encodes `" world!"` as `[905, 17]`), so no supported profile strips it in the pre-split; `eot_token_id` is exposed and used as the stop/EOS signal when the model declares one. BOS/EOS fallback ids become profile-scoped defaults (llama-bpe keeps 11) with the applied default logged, instead of a hardcoded constant.
  - Rationale: byte handling and special-id fallbacks are currently hardcoded GPT-2/llama assumptions, while GGUF families declare these keys and llama.cpp honors them — parity requires reading them, and where the pinned oracle shows a key has no pre-split effect for a supported profile, matching the oracle wins over inferring behavior from the key.
  - Rejected: keeping the hardcoded byte maps (breaks on models whose byte pieces are not the GPT-2 visual map); requiring every key to be present (rejects valid older GGUFs); changing out-of-range id handling (out-of-range stays `-1` as today); stripping the folded leading space when `add_space_prefix=false` (b12032 `/tokenize` on the `smollm` fixture keeps it — stripping diverges from the oracle and drops characters on round-trip).
  - Contract: `GgufMetadata` gains typed accessors for these keys; `GgufTokenizer` consumes them at construction; `GenerationService` consumes `EotId` where EOS/stop decisions are made.
- **DN-M33-5** — Tokenizer parity gets a committed oracle harness: `scripts/generate-tokenizer-parity.ps1` spawns pinned llama.cpp b12032 `llama-server`, captures `/tokenize` id sequences for the real fixtures (SmolLM = `smollm`, Llama-3.2 = `llama-bpe`) into `tests/fixtures/parity-tokenizer-*.json` with model sha256 guards, and `Category=Tokenizer` asserts LAIR's `Encode` matches id-for-id. The 2026-11-04 manual proof becomes reproducible.
  - Rationale: profile and metadata changes touch the hot tokenizer path, and the 2026-11-04 RoPE investigation showed tokenizer divergence is expensive to diagnose without a pinned harness.
  - Rejected: synthetic-only coverage (synthetic GGUFs cannot prove llama.cpp agreement); the CLI `llama-tokenize` tool instead of the server endpoint (the pinned asset's server mode is already the harness Parity and TemplateParity use).
  - Contract: fixture shape mirrors `parity-templates-llama32.json` (guards + case list); parity cases skip cleanly when a model file is absent.

#### Implementation Items

- [✓] Tokenizer profile registry — move `tokenizer.ggml.pre` selection behind an internal profile registry (llama-bpe, gpt2, smollm), resolve the pre-split strategy once at construction instead of switching per call, and keep unsupported `pre`/`model` rejections naming the value plus the supported profile list (see DN-M33-2, DN-M33-3)
  - Verify: `dotnet test --filter Category=Tokenizer -c Release` — 23/23 passed: the 18 existing goldens unchanged plus profile-selection facts for all three pre keys, the missing-key llama-bpe default, and a registry-sourced rejection list. `PreRule` and the constructor are now internal; `FromGguf` is the only public entry.
- [✓] GGUF tokenizer metadata coverage — read `byte_fallback`, `byte_to_token`, `byte_to_utf8`, `add_space_prefix`, `eot_token_id`, and `utf8_flush_between_controls`; honor `byte_to_token` over the `<0xNN>` scan, read and expose `add_space_prefix` with no pre-split effect for the supported profiles (oracle-verified per DN-M33-4), expose `EotId` and use it for stop/EOS decisions, and make BOS/EOS fallbacks profile-scoped with the applied default logged (see DN-M33-4)
  - Verify: `dotnet test --filter Category=Tokenizer -c Release` — 40/40 passed: synthetic GGUF cases per key (present/absent/invalid), profile-scoped BOS/EOS defaults with `tokenizer_default` log lines, `EotId` stop-signal facts, and real-fixture facts (SmolLM `smollm` ids, round-trip, and `add_space_prefix=false` exposed without stripping; Llama-3.2 `llama-bpe` ids and `add_bos_token`). Full suite 735/735.
- [✓] Tokenizer oracle harness — add `scripts/generate-tokenizer-parity.ps1` against pinned b12032 `/tokenize` and commit `tests/fixtures/parity-tokenizer-*.json` for both real fixtures, asserted id-for-id by `Category=Tokenizer` (see DN-M33-5)
  - Verify: `pwsh -File scripts/generate-tokenizer-parity.ps1` regenerates both fixtures from the pinned build (SmolLM `pre=smollm`, Llama-3.2 `pre=llama-bpe`; `add_special=false`/`parse_special=false`); `dotnet test --filter Category=Tokenizer -c Release` — 42/42 passed, including 12 plain-text cases per fixture asserted id-for-id against the committed goldens. The harness immediately paid for itself: it exposed a pre-split classification defect (UTF-16 `char` classification treated astral code points as lone surrogates, so a leading space was not folded before emoji runs; the llama-bpe symbol fold additionally required the space itself to be a symbol). Both splits now classify by Unicode code point in `TokenizerProfile` and match the reference id-for-id.

#### Tests

- [✓] Automated Tokenizer: profile selection, per-key metadata coverage, and pinned `/tokenize` oracle parity for every supported profile. Verify: `dotnet test --filter Category=Tokenizer -c Release` — 42/42 passed; full suite 737/737.

### MVP3.4 [v0.3.4] - Shipped

Dense GGUF architecture families load through a load-time profile registry that produces a concrete model plan, instead of hardcoded `TransformerModel` assumptions.

#### Design Notes

- **DN-M34-1** — Architecture support is resolved at model load/plan time. A profile maps GGUF metadata and tensor names to a concrete execution plan: tensor map, norm behavior, activation function, attention bias behavior, RoPE strategy, KV shape, and layer graph. Hot-path execution consumes the resolved plan, never an architecture name.
  - Rationale: Qwen3, Gemma, Phi, Mistral, and later MoE variants share dense transformer machinery but differ in tensor naming, norm/activation details, attention variants, and positional encoding. Adapters keep `TransformerModel` from accumulating architecture-specific branches.
  - Rejected: hardcoded architecture branches in `TransformerModel`; runtime architecture plugins during decode; separate libraries per architecture family without packaging need; llama.cpp's `switch (arch)` in `load_hparams` (same hardcoded-branch posture, and llama.cpp already owns the word "adapter" for LoRA).
  - Contract: `lair.models` exposes a profile registry and produces a concrete model plan consumed by the engine.
- **DN-M34-2** — Resolution posture is hybrid: the family profile supplies tensor-name aliases and feature defaults, declared metadata keys override those values, and an unknown `general.architecture` falls back to a generic llama-shaped profile **only** when the tensor table matches exactly. Otherwise the load fails naming the missing or unexpected tensors and keys.
  - Rationale: family-first alone rejects llama-shaped GGUFs published under unknown or renamed architecture strings; feature-first alone makes tensor presence the contract and lets a missing optional tensor silently change the math. The hybrid keeps actionable failures while widening acceptance.
  - Rejected: family-first-only (unnecessarily rejects compatible files); feature-first-only (silent mis-detection, no actionable errors); accepting an unknown architecture with a partial tensor match (produces wrong output rather than a clear failure).
  - Contract: `lair.models` resolves profiles from `general.architecture` with a named generic fallback selected only on exact tensor-table match; the registry is the single source of the rejection message's supported-profile list.
- **DN-M34-3** — The plan is a resolved immutable value set, not a graph object and not a set of interfaces: tensor map (names, aliases, optional entries), norm strategy (kind, weighting, placement, epsilon source), activation strategy, bias policy, head/KV geometry, RoPE strategy id plus parameters, cap/window values, and a sparse-FFN descriptor. Strategies are enum/struct values resolved once at load, so the forward path stays concrete and dispatch-free.
  - Rationale: mirrors the shipped tokenizer profile pattern (DN-M33-3) — profiles are load-time factories. LAIR executes forward imperatively, so a ggml-style per-family build graph would be a foreign shape that adds allocation and indirection without benefit.
  - Rejected: per-token `INorm`/`IActivation` interface dispatch in the hot path; a per-family computation-graph builder (llama.cpp-shaped); growing `ModelDesc` in place (it is a flat scalar record whose unsupported cases are rejections, not values).
  - Contract: the plan replaces `ModelDesc` as the load product consumed by `TransformerModel.Open` and by `BlockPool` construction.
- **DN-M34-4** — Metadata is the contract, not tensor presence. Every feature flag the pinned reference honors for a family is read and either honored or rejected at load: `attention.use_bias`, `attention.head_length`/`key_length`/`value_length`, `attention.relax_norms`, activation keys, `rope.type` and `rope.dimension_count`, and cap keys. Today bias behavior is decided purely by tensor presence and `rope.scaling.*` is a blanket rejection; both change here. A family profile supplies the feature defaults for its family (per-head QK-norm for `qwen3` and `gemma2`), and a declared key overrides the default in either direction; a default with no tensors is an actionable load failure.
  - Rationale: the pinned llama.cpp build is the parity oracle (DN-M33-5). Parity requires reading the same keys and failing visibly where LAIR has no implementation yet. The pinned Qwen3 GGUF carries `blk.N.attn_q_norm.weight`/`attn_k_norm.weight` and applies them without declaring `attention.use_qk_norm`, so a declared-only rule would load that file with the norm silently missing — the exact failure mode this note exists to prevent.
  - Rejected: honoring tensor presence only (current bias behavior mis-states intent and silently accepts a model that declares no bias); ignoring unrecognized keys (silently wrong math); implementing every declared feature before any family is proven against the oracle; resolving QK-norm from a declared key alone (the reference applies it on tensor presence for these families).
  - Contract: `lair.loader` exposes typed metadata accessors for these keys; plan resolution reports an unsupported declared key as a load failure naming key, value, and family.
- **DN-M34-5** — Family order is fixture-driven under the 4 GB peak-working-set model budget: `llama` and `qwen2` are re-expressed as profiles with byte-identical behavior, then `qwen3` (per-head QK-norm, `head_length`, declared bias), then Gemma-2 (GELU-tanh FFN, attention and final-logit softcaps, `head_length`). Phi is deferred until RoPE scaling exists (MVP3.7) because its metadata declares it, and Gemma-3 stays deferred to MVP3.9 for its per-layer windowed/full attention interleave.
  - Rationale: each added family must prove a primitive the previous families did not (QK-norm and head-length independence; then a different activation, two caps, and a declared attention window) and must be verifiable against the pinned oracle on a file this box can actually load.
  - Rejected: adding families that repeat already-proven primitives; Phi-3 in this phase (needs `phi3.rope.scaling.*`, which is MVP3.7 work); Mistral in this phase (its distinguishing feature is window-local KV reads and interleave, deferred to MVP3.9); families with no published quant loadable under the ceiling.
  - Contract: each family lands with a pinned model file (sha256 recorded in its parity fixture) plus synthetic layer fixtures; `Category=Parity` and `Category=Layers` are the gates.
- **DN-M34-6** — MoE is representable in MVP3.4 and executable in MVP3.9. The plan carries a sparse-FFN descriptor (expert count, per-expert tensor map, shared expert, router tensors) and the loader recognizes MoE metadata and tensors, but a MoE model fails at load with an actionable message naming sparse execution as deferred.
  - Rationale: retrofitting a sparse-FFN descriptor into a plan that has already shipped three families is expensive; executing MoE before its gather/residency design is measurable is worse. MVP6 keeps Flash-Next residency, prefetch, PLE, and MTP.
  - Rejected: executing MoE in this phase with all experts resident (no gather design, no fixture under the RAM budget); leaving MoE invisible in the plan (retrofit later); pulling Flash-Next residency forward into MVP3.
  - Contract: the plan's sparse-FFN descriptor is present and validated in MVP3.4; `IGatherKernel` and expert residency land in MVP3.9/MVP6.
- **DN-M34-7** — A declared attention window is honored as a score mask with full KV retention. `<arch>.attention.sliding_window` becomes a plan value; each attention path excludes positions further back than the window, while the KV cache still stores every position. The KV-layout half of windowed attention — bounded KV reads, per-layer windowed/full interleave, and cache-reuse accounting — stays in MVP3.9.
  - Rationale: the pinned Gemma-2 file declares `gemma2.attention.sliding_window = 4096` and the pinned oracle applies that mask, so loading Gemma-2 in this phase without the mask would silently mis-state the math, which DN-M34-4 forbids. The pinned reference itself keeps full KV and masks scores, so the mask is the part that carries parity; the bounded-read optimization changes memory behavior, not output.
  - Rejected: rejecting Gemma-2 for declaring a window (blocks a family this phase exists to prove); loading it and ignoring the declaration (silently wrong beyond the window); pulling bounded KV reads and interleave into this phase (a KV-block design concern with no parity requirement at fixture context sizes).
   - Contract: `lair.models` resolves the window into the plan and applies it in every attention path; `lair.kv` layout consequences land in MVP3.9. Measured against the pinned reference on the Gemma-2 fixture, its masking is backward-only — a position at sequence start has zero forward visibility even when the whole sequence fits the declared window — so LAIR's symmetric range for the bidirectional path is a convention with no oracle counterpart yet (MVP3.9 open item).
- **DN-M34-8** — Residual-norm placement and QK-norm weighting are resolved plan values, not engine assumptions. The plan carries where block norms apply (pre-block only, or pre-block plus an extra norm on each sublayer output before the residual add), and per-head QK-norm resolves to one of three modes — off, weighted from tensors, or unweighted — because the pinned Gemma-2 file ships no `attn_q_norm`/`attn_k_norm` tensors while its reference applies a per-head normalization.
  - Rationale: today's forward graph is fixed RMSNorm-before-sublayer with weights read from tensors. That shape cannot express Gemma-2, which normalizes each attention and feed-forward output again before the residual add using `blk.N.post_attention_norm.weight` and `blk.N.post_ffw_norm.weight`, and it cannot express a normalization that has no weight tensor at all. Both are structural facts about the family's graph rather than new scalar operations, so they belong in the plan next to norm kind and activation.
  - Rejected: hardcoding post-norm behavior for Gemma inside `TransformerModel` (the architecture-branch posture DN-M34-1 rejects); treating missing QK-norm tensors as a load failure for every family (Gemma-2 legitimately has none); an optional per-role norm interface list resolved at runtime.
  - Contract: the plan carries residual-norm style and QK-norm mode; `TransformerModel` branches on them concretely, exactly as it already does for norm kind, activation, caps, window, and RoPE pairing.

#### Implementation Items

- [✓] Model plan and profile registry — replace `ModelDesc`'s supported-architecture list with a profile registry that produces the resolved plan (tensor map with aliases and optional entries, norm/activation/bias/geometry/RoPE strategies), and re-express `llama` and `qwen2` as profiles with no behavior change (see DN-M34-1, DN-M34-3).
  - Verify: `dotnet test --filter Category=Loader|Layers -c Release` — Loader 19 (10 new plan-resolution cases), Layers 23 unchanged. `dotnet test --filter Category=Parity|Serve -c Release` — 30/30 unchanged against the SmolLM and Llama-3.2 goldens. Full suite 747/747. `ModelDesc` survives as a thin view over the plan because 12 test call sites and ~12 test files read `model.Desc.*`.
- [✓] Generic llama-shaped fallback profile — accept an unknown `general.architecture` only when the tensor table matches the llama-shaped map exactly; otherwise reject naming the missing and unexpected tensors (see DN-M34-2).
  - Verify: `dotnet test --filter Category=Loader -c Release` — fallback resolves with profile identity `llama-generic` and a forward bit-identical to the same tensors declared as `llama`; a renamed/removed required tensor and an unnameable extra tensor (`blk.0.attn_qkv.weight`) each reject naming the tensor and the attempted fallback; a missing declared-prefix key names the key; the `model_profile` log line appears only when the fallback is applied. An empty/whitespace architecture rejects outright and never reaches the fallback. Full suite 754/754.
- [✓] Norm and activation strategies — resolve norm kind (RMSNorm, LayerNorm) with weighting and epsilon source, and activation kind (SwiGLU/SiLU, GELU, GELU-tanh, GEGLU) in the plan, replacing the hardcoded RMSNorm+SwiGLU pair (see DN-M34-3, DN-M34-4).
  - Verify: `dotnet test --filter Category=Layers -c Release` — 47/47: LayerNorm against a hand-computed vector (mean/population-variance/epsilon/weight/bias, zero-variance case), GELU against a series-computed erf oracle and a continued-fraction erfc oracle, GELU-tanh against the direct formula, GEGLU composition, and a forward-level `LayerNorm`+`GeGlu` run (decode, batch decode, encode) matching a double-precision scalar reference forward. `Category=Parity|Serve` stays 30/30 with `llama`/`qwen2` still resolving `RmsNorm`/`SwiGLU`.
- [✓] Geometry, bias policy, and QK-norm — honor `attention.use_bias` and declared norm-bias flags instead of tensor presence, support `attention.head_length` where head dim ≠ `embedding_length / head_count`, and resolve per-head QK-norm tensors (see DN-M34-4).
  - Verify: `dotnet test --filter Category=Loader -c Release` — 32/32: declared `use_bias`/`norm_bias`/`use_qk_norm` resolution with actionable missing-tensor failures and a `bias_ignored` log line when declared false with tensors present. `dotnet test --filter Category=Layers -c Release` — 52/52: Gemma-shaped head-dim independence against a double-precision scalar forward (decode/batch/encode), QK-norm on/off proving the logits change at position 1 and that the norm lands before RoPE, norm-bias through LayerNorm end to end, and declared-false-with-tensors bit-identical to the tensor-free model. `Category=Parity|Serve` stays 30/30. Full suite 789/789.
- [✓] Logit caps — attention-score softcap and final-logit softcap as plan values applied in the attention and output-head paths (see DN-M34-3).
  - Verify: `dotnet test --filter Category=Layers -c Release` — 77/77: softcap scalar goldens against a double-precision `cap * tanh(x / cap)` reference (near-identity, exact `±cap`, saturating magnitudes), a synthetic model whose attention-capped logits differ from uncapped and match a scalar reference forward through decode/batch/`Encode`, a final-capped model with logits provably inside `±cap`, batched vs single-step bit-identical under both caps, and the tied-embedding path honoring the cap. `Category=Loader` 54/54: absent keys resolve disabled, declared values resolve exactly, non-positive/NaN/infinite/out-of-float-range values reject naming key and value; `llama`/`qwen2`/`llama-generic` stay uncapped and `Category=Parity|Serve` stays 30/30. Full suite 836/836. `Encode` returns hidden states, so the final-logit cap applies only in the two paths that produce logits.
- [✓] Sliding-window score masking — resolve `<arch>.attention.sliding_window` into the plan and exclude positions beyond the window from every attention path, with full KV retention (see DN-M34-7).
  - Verify: `dotnet test --filter Category=Loader -c Release` — 77/77: absent resolves to unbounded, a declared value resolves exactly, zero/negative/non-integer/out-of-int-range declarations fail naming key and value, and `llama`/`qwen2`/`qwen3`/`llama-generic` all report the window disabled. `dotnet test --filter Category=Layers -c Release` — 86/86: a small-window model matches a masked double-precision scalar reference at an out-of-window query position and demonstrably differs from the unmasked one, in-window positions are provably inert, batched decode equals single-token bit-for-bit, a window at or beyond sequence length is bit-identical to no window through both decode and `Encode`, masking composes with a softcap against references doing each alone, and masked-out positions stay readable in KV. `Category=Parity|Serve` stays 34/34; full suite 873/873.
- [✓] Vocab-size fallback — resolve the vocabulary from `<arch>.vocab_size` when declared, otherwise from the `tokenizer.ggml.tokens` length as the pinned reference does, failing only when neither is available.
  - Verify: `dotnet test --filter Category=Loader -c Release` — a file with no `<arch>.vocab_size` resolves the token-array length and logs one `vocab_fallback` line; a declared key wins and a disagreement reports `vocab_disagree`; a file carrying neither fails naming both keys. The pinned Qwen3 fixture is exactly this case (no `qwen3.vocab_size`, 151,936 from the token array).
- [✓] Qwen3 dense family — a `qwen3` profile wiring QK-norm, `head_length`, and declared bias behavior, with synthetic layer fixtures and real-model parity (see DN-M34-5).
  - Verify: `dotnet test --filter Category=Loader -c Release` — 68/68: `qwen3` resolves RmsNorm/SwiGLU/GQA 16·8/head 128 over hidden 1024, theta 1e6, epsilon 1e-6, QK-norm on by profile default with a declared `attention.use_qk_norm=false` able to flip it and missing norm vectors failing actionably. `dotnet test --filter Category=Layers -c Release` — 80/80 including the synthetic qwen3 forward against a double-precision scalar reference through decode, batch decode, and encode, plus a rotate-half RoPE golden that differs from adjacent pairing on the same vector. `Category=Parity|Serve` 34/34 with the new Qwen3 fixture green and the SmolLM/Llama goldens unchanged; full suite 858/858.
  - Cause found while proving parity: LAIR applied llama-family adjacent-dimension RoPE pairing `(2i, 2i+1)` to qwen3, whose GGUF stores q/k in rotate-half layout `(i, i + head_dim/2)`. Logprob drift was 1.1–2.3 nats and the `def fibonacci(n):` argmax moved from `\n` back to the reference's `'   '`. The pinned b12032 oracle is unambiguous here — LAIR now reproduces its leaders to ~0.001 nats (`" Paris"` at -0.4327 against the reference's -0.4339, and the full top-5 order on prompt 2) — so RoPE pairing becomes a per-family plan value: `RopeStrategy.StandardFullHead` for llama/qwen2/generic, `RopeStrategy.RotateHalfFullHead` for qwen3.
- [✓] Residual-norm placement and QK-norm mode — resolve where block norms apply and whether per-head QK-norm is off, weighted from tensors, or unweighted, applying both concretely in the forward paths (see DN-M34-8).
  - Verify: `dotnet test --filter Category=Layers -c Release` — 92/92: a `PostSublayerOutputs` model matches a double-precision scalar reference that normalizes each sublayer output before the residual add and differs from the same weights under `PreOnly` through decode, batched decode, and `Encode`; each post vector is proven to move logits; unweighted per-head QK-norm is bit-identical to the weighted form with an all-ones vector and differs from off. `Category=Loader` 84/84: every profile resolves `PreOnly` with its current QK mode, a declared `attention.use_qk_norm=false` forces off, post-norm tensors under a pre-only profile fail naming them, and a post-placement file missing one fails naming it. `Category=Parity|Serve` stays 34/34; full suite 886/886.
  - Resolution rule as shipped: absent declaration takes the profile mode; declared false forces off; declared true means weighted-from-tensors unless the profile already resolves unweighted. That keeps the existing "declared true demands the vectors" behavior for `llama`/`qwen2`/generic, which the first wording would have broken.
- [✓] Gemma-2 dense family — a `gemma2` profile wiring the block graph the pinned reference actually runs, with synthetic layer fixtures and oracle-measured forward parity (see DN-M34-5, DN-M34-8).
  - Verify: `dotnet test --filter Category=Loader -c Release` — 93/93: `gemma2` resolves RmsNorm block norms, `GeGluExact`, rotate-half RoPE, post-sublayer output placement, the sqrt(hidden) input-embedding gain, QK-norm off with a declared true demanding the vectors, bias declared-only with present-but-unconsumed vectors reported through the ignore path, caps 50/30, window 4096, head dim 256 over hidden 2304 with GQA 8/4. `dotnet test --filter Category=Layers -c Release` — 94/94: the forward matches a double-precision scalar reference of the reference graph through decode, batched decode, and `Encode`, with each feature pinned by a counterfactual reference that demonstrably differs (no embedding gain, QK-normed, tanh GELU, pre-only placement, uncapped scores, uncapped logits). `Category=Parity|Serve` stays 34/34; full suite 897/897.
  - What the oracle overturned: the family was designed as LayerNorm with GELU-tanh, dense attention biases, and an unweighted per-head QK norm. Measured against the pinned reference it is RMSNorm everywhere including the output norm, exact GELU in the gate, no per-head QK normalization at all, an input-embedding gain of `sqrt(hidden)` that the conversion leaves out of the weights, and attention biases that its forward does not consume. Each choice ships with the measured delta that forced it (0.18–0.53 nats against the reference's own top-5), not with the design guess.
  - Real-model parity fixture is blocked, not failed: `tokenizer.ggml.model=llama` is a SentencePiece vocabulary and LAIR supports only GGUF BPE profiles, so `scripts/generate-parity.ps1` cannot calibrate this file. The forward was proven by driving LAIR's engine with reference-supplied token ids instead. Serving Gemma-2 needs the MVP3.11 SPM tokenizer profile.
- [✓] Window interleave — the pinned reference applies the Gemma-2 window to alternating layers (local on even indices, full on odd), while LAIR applies a declared window to every layer, which masks attention in layers the reference leaves unmasked beyond the window length. Resolve a per-family interleave pattern in the plan and apply it in the score mask only; bounded KV reads stay deferred (see DN-M34-7).
  - Verify: `dotnet test --filter Category=Layers -c Release` — 99/99: an alternating multi-layer model matches a double-precision scalar reference that masks only even-index layers and differs from masking every layer, `AllLayers` stays bit-identical to the explicit all-layers plan, batched decode equals single-token under alternation, the `Encode` golden holds across chunks, and an alternating pattern with no declared window is bit-identical to having no pattern. `Category=Loader` 95/95: `gemma2` resolves `AlternatingLocalFirst` with or without a declared window while `llama`/`qwen2`/`qwen3`/generic resolve `AllLayers`. `Category=Parity|Serve` stays 34/34; full suite 904/904.
- [✓] Family fixture set — pin one Qwen3 GGUF loadable under the 4 GB peak-working-set budget, record its sha256 in the new parity fixture, and regenerate goldens with `scripts/generate-parity.ps1` (see DN-M34-5). The Gemma-2 half is deferred: its SentencePiece vocabulary blocks the harness, so its pinning belongs to MVP3.11.
  - Verify: `pwsh -File scripts/generate-parity.ps1 -ModelPath models/Qwen3-0.6B-Q8_0.gguf -Out tests/fixtures/parity-qwen3.json` writes a guarded fixture and `dotnet test --filter Category=Parity -c Release` passes it.
  - Pinned: `models/Qwen3-0.6B-Q8_0.gguf` (609,800 KB, sha256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`, from `Qwen/Qwen3-0.6B-GGUF`) loads under the pinned b12032 loader and reports `qwen3` (28 layers, 1024 hidden, 16/8 heads, 128 key/value length). `models/gemma-2-2b-it.Q4_K_M.gguf` (1,629,000 KB, sha256 `ff7c35808d3b2fc5c4f955ca931efdec71c0808b9ec89a9878cb771608787058`, from `mradermacher/gemma-2-2b-it-GGUF`) is kept only as the oracle-measurement file for the forward graph — it reports `gemma2` (26 layers, 2304 hidden, 8/4 heads, 256 key/value length, caps 50/30, window 4096) and is out of budget.
  - Status: `tests/fixtures/parity-qwen3.json` is generated, sha256-guarded, and green, and `models/Qwen3-0.6B-Q8_0.gguf` at 610 MB sits inside the revised band. The Gemma-2 fixture cannot be generated: its `tokenizer.ggml.model=llama` vocabulary is SentencePiece and the harness calibrates through LAIR's tokenizer, which supports BPE only.
- [✓] MoE descriptor without execution — the plan carries the sparse-FFN descriptor and the loader recognizes MoE metadata and tensor names, failing at load with an actionable deferred-execution message (see DN-M34-6).
  - Verify: `dotnet test --filter Category=Loader -c Release` — 115/115: a synthetic MoE file with the reference's own names (`blk.N.ffn_gate_inp`, `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, `ffn_gate_shexp`/`ffn_gate_inp_shexp`, `ffn_up_shexp`, `ffn_down_shexp`) resolves its descriptor — expert count, routed-per-token, shared-expert presence, routing weight scale and normalization — with invalid `expert_count`/`expert_used_count`/`expert_weights_scale` declarations failing naming key and value; `qwen2moe` and `qwen3moe` are registered profiles carrying the sparse shape and each fails at engine open with `architecture '<arch>' declares a sparse-FFN block with N experts routing K per token: sparse FFN execution is not implemented by the forward engine`, asserted to contain no missing-tensor wording; dense families keep a null descriptor; an unregistered architecture carrying MoE tensor names still fails through the tensor-table message. `Category=Layers` 99/99 and `Category=Parity|Serve` 34/34 unchanged; full suite 924/924.

#### Tests

- [✓] Automated Model Plan: profile selection per architecture, tensor-map and alias resolution, optional-tensor handling, generic fallback accept and near-miss rejection, unsupported declared-key diagnostics. Verify: `dotnet test --filter Category=Loader -c Release` — 54/54: profile selection and tensor-map/alias/optional resolution, registry-sourced rejection, generic fallback accept plus near-miss and empty-architecture rejection, declared `use_bias`/`norm_bias`/`use_qk_norm` resolution with actionable missing-tensor failures, and cap/rope declared-key diagnostics naming key and value.
- [✓] Automated Layer Strategies: LayerNorm, GELU, GELU-tanh, GEGLU, QK-norm, head-length independence, and both logit caps against scalar references, with existing RMSNorm/SwiGLU regression unchanged. Verify: `dotnet test --filter Category=Layers -c Release` — 77/77 across all five items, with `Category=Parity|Serve` unchanged at 30/30.
- [✓] Automated Window Mask: declared-window resolution and rejection diagnostics, out-of-window positions contributing nothing against a masked scalar reference, window-at-or-beyond-length equivalence through decode and encode, batch/single equality, composition with softcap, and KV retention under masking. Verify: `dotnet test --filter Category=Loader|Layers -c Release` — Loader 77, Layers 86.
- [✓] Automated Qwen3 Parity: dense Qwen3 output parity vs pinned llama.cpp within documented tolerance. Verify: `dotnet test --filter Category=Parity -c Release` — 4 prompts against `tests/fixtures/parity-qwen3.json` (`models/Qwen3-0.6B-Q8_0.gguf`, sha256 recorded in the fixture), passing inside the quantized-file band with LAIR's calibration reproduced exactly.
- [✓] Manual Family Serve: `models/Qwen3-0.6B-Q8_0.gguf` served on `:8181`; two-turn exchange coherent (`2 + 2` → assistant `4` → `multiply by 3` answered `12`, `finish_reason: eos`), `/ui/` serving (HTTP 200), resolved architecture and quant read from `/v1/models` (`architecture: qwen3`, `quant: Q8_0`), and the monitor stream reported the working set, KV capacity/usage, devices and thread quota. Verify: live two-turn exchange plus `/monitor/stream` snapshot frames. Serving profile that satisfies the model budget: `max_context_tokens: 512`, `kv_blocks: 32`, `thread_quota: 4` — peak 1,000 MB, steady 977 MB. The default `4096`/`256`/`8` profile measured 1,726–1,740 MB, so the budget is a profile constraint, not just a file-size one. Gemma-2 is excluded until its tokenizer exists (MVP3.11).

#### Open Items

- [✓] **Fixture budget conformance** — `models/Llama-3.2-1B-Instruct-Q8_0.gguf` (1,260 MB) is replaced by `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` (770 MB, sha256 `3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1`, from `unsloth/Llama-3.2-1B-Instruct-GGUF`); the serve, tokenizer-parity and template-parity references, the `scripts/generate-template-parity.ps1` default, and `DEVELOP.md` all point at it, and both llama goldens were regenerated from it.
  - Verify: `Category=Tokenizer` 43/43, `Category=TemplateParity` 14/14, `Category=Parity|Serve` 34/34, full suite 924/924. The regenerated goldens are byte-identical except `model_file`, `model_sha256`, `generated_utc`, and the fixture-pinned `date_string` (`06 Nov 2026` → `08 Nov 2026`) that the 12 `expected_prompt` values embed — quant-independence of ids and template rendering is proven by the diff, not assumed.
- [X] **Serve working set under the revised ceiling** — measured: a 770 MB-weight fixture peaks at 1,102–1,174 MB (≈360–395 MB of runtime overhead per engine: worker compute buffers, JIT, vocab tables), and Qwen3-0.6B (610 MB weights) peaks at 1,726 MB on the default `4096`/`256`/`8` profile.
  - Decision: the owner restored a 4 GB peak-working-set ceiling, which both fixtures clear with no re-pin and no profile change. The measurements stay recorded because they define what a fixture costs — a loaded engine adds roughly 360–395 MB on top of weights and KV, so the ceiling is a working-set limit rather than a file-size limit.
- [X] **Gemma-2 serving** — real-model parity and live serve for the pinned Gemma-2 fixture need a SentencePiece tokenizer profile.
  - Decision: it lands as MVP3.11, whose fixtures and gates are now written there (DN-M311-3). It does not merge into an earlier tokenizer phase: MVP3.3 is shipped, and the vocabulary-type seam MVP3.11 adds is a different axis from the `pre` profiles that phase built.
- [X] **Window interleave** — whether the alternating local/full pattern belongs in the plan as a family property or as metadata once a declaration exists for it.
  - Decision: it ships as a family property (`WindowInterleave` resolved by the profile), which is what MVP3.4 built and measured. If a GGUF ever declares the pattern, the declared key overrides the profile default in either direction — the same posture DN-M34-4 already defines for every other feature default. The KV-read half is MVP3.9's item.

### MVP3.5 [v0.3.5] - In Progress

Published GGUF quant families are accepted through a load-time quant descriptor/kernel registry while CPU kernels remain concrete and performance-sensitive.

#### Design Notes

- **DN-M35-1** — Quant compatibility is split into a load-time descriptor/registry and concrete kernel implementations. The registry owns GGML type acceptance, block geometry, row strides, and loader diagnostics; kernels stay concrete in `lair.backends.cpu` or backend-specific code paths.
  - Rationale: quant formats are performance-critical and must preserve bit-identity/parity behavior. A registry makes loader support extensible without introducing dynamic interface dispatch in GEMV/GEMM loops.
  - Rejected: per-row `IQuantKernel` dispatch in hot paths; separate libraries per quant format without packaging need; loader acceptance without kernel parity gates.
  - Contract: `lair.loader` and `lair.contracts` resolve quant descriptors at load; backends execute concrete kernels for the selected formats.
- **DN-M35-2** — The registry entry is a resolved descriptor value, not a table of strings: GGML type id, format name, `blck_size`, `type_size`, and the sub-block/scale layout the kernel consumes. Acceptance is registry membership; an unsupported id fails naming the id and its GGML name. Stride validation is computed from the descriptor against the tensor's element count and byte length at load.
  - Rationale: today acceptance is a `switch` in `GgmlType` whose block geometry lives only in comments, so a new format means editing the switch and a stride problem surfaces mid-read rather than at load. A descriptor makes a format one registry entry plus one kernel and makes every geometry failure actionable before weights are read.
  - Rejected: config-driven format tables (no compile-time pairing to a kernel, so a typo becomes a runtime crash); deriving geometry from file size (silent mis-parse of a truncated file); accepting unknown ids with a best-effort dequant (produces wrong output instead of a named failure).
  - Contract: `GgmlType`'s switch becomes `GgmlTypeRegistry.TryResolve(uint id, out QuantDescriptor)`; `QuantDescriptor { TypeId, Name, Format, BlockSize, TypeSize }`; `QuantFormat` stays the enum the kernels switch on.
- **DN-M35-3** — Format priority is fixture-driven: `Q3_K` and `Q2_K` enter this phase, nothing else. `Q4_0`/`Q5_0`/`Q5_1`/`Q8_1`/`Q8_K`, the IQ family, and TQ formats stay out until a published fixture demands them.
   - Rationale: these are the formats the later phases' fixtures actually need — MVP3.9's MoE fixture is `OLMoE-1B-7B-0924-Instruct` at `Q2_K` (2,444 MB, loads at roughly 2.8 GB against the 4 GB ceiling) and MVP3.11's windowed-serve fixture is `gemma-3-1b-it` at `Q3_K_S` (657 MB). Every format added carries a kernel, an oracle parity gate, and a real fixture, so formats no target model ships in are dead weight.
  - Rejected: `IQ4_NL`/`IQ4_XS`/`IQ3_XXS` (they get their own phase at MVP3.8, where the fixture and measurement gates for them live); `Q4_0` (superseded by `Q4_K` for every model LAIR serves); `Q8_K` (no published GGUF ships it as a standard file); implementing every GGML type for completeness (untestable formats with no demand).
  - Contract: `QuantFormat` gains `Q2_K` and `Q3_K`; GGML ids 10 and 11 become accepted; `Category=Kernel` and `Category=Parity` gate both.
- **DN-M35-4** — `Q2_K` and `Q3_K` are super-block formats with per-sub-block scales, so their definition is the pinned reference's layout, not a LAIR-authored scalar golden. Parity is asserted against pinned b12032 on a real file in the quantized band. The two formats are asymmetric with each other: `Q2_K` is 16 sub-blocks of 16 with a 4-bit scale *and* a 4-bit min per sub-block packed in one byte and a value of `d*scale*q - dmin*min`, while `Q3_K` is 16 sub-blocks of 16 with a 6-bit scale packed across 12 bytes, a high-bit/hmask split of the 3-bit quants, and a value of `d*(sc-32)*(q-4)` with no min at all.
  - Rationale: these are the first formats LAIR sees where scales are per-sub-block inside a 256-element super-block, and the packing is fiddly enough that a hand-written golden that disagrees with the reference is worthless — the reference defines the format. Layout read directly from `b12032`'s `ggml-common.h` (`block_q2_K`, `block_q3_K` static_asserts: 84 and 110 bytes) and `ggml-quants.c` (`dequantize_row_q2_K`, `dequantize_row_q3_K`).
  - Rejected: treating `Q3_K` as min-based like `Q4_K` (wrong output, silently); accepting a close-enough dequant (destroys the parity contract); synthetic-only gating (passes LAIR's own misunderstanding).
  - Contract: kernels reproduce the reference layout (`Q2_K`: 16 packed scale/min bytes, 64 quant bytes, `d`/`dmin`; `Q3_K`: 32 hmask bytes, 64 quant bytes, 12 six-bit scale bytes, `d`); `Category=Kernel` asserts synthetic-block scalar-vs-AVX2 bit-identity and `Category=Parity` asserts real-fixture decode parity.

#### Implementation Items

- [✓] Quant descriptor registry — move GGML type acceptance, block geometry, and row-stride validation behind a descriptor registry while preserving current F16/Q8_0/Q4_K/Q5_0/Q6_K behavior bit-identically (see DN-M35-1, DN-M35-2).
  - Verify: `dotnet test --filter Category=Loader|Kernel -c Release` — 197/197: existing quant geometry and kernel cases unchanged; unsupported GGML ids (Q4_0, Q2_K, IQ2_XXS) reject naming the id and its GGML name; a tensor whose fastest dimension is not a multiple of the descriptor's `BlockSize`, or whose byte length does not match `TypeSize` arithmetic (including running past the end of the file), fails at load naming tensor, id, and expected geometry.
- [✓] `Q3_K` format — descriptor plus scalar and AVX2 kernels reproducing the reference super-block layout (see DN-M35-3, DN-M35-4).
  - Verify: `dotnet test --filter Category=Kernel -c Release` — synthetic two-super-block images with varied scale/min values including zero scales and max-value scales, scalar-vs-AVX2 bit-identity on dequant across the full sub-scale packing and both hmask states, batched-vs-single decode equivalence, and dequant against a double-precision reference implementing `d * (sc - 32) * ((low2 | hbit<<2) - 4)` with fp16 round-trip on `d`.
- [✓] `Q2_K` format — descriptor plus scalar and AVX2 kernels reproducing the reference super-block layout (see DN-M35-3, DN-M35-4).
  - Verify: `dotnet test --filter Category=Kernel -c Release` — same gate, covering the 4-bit scale/4-bit min packing and the asymmetric `d*scale*q - dmin*min` that distinguishes it from `Q4_K`.
- [_] Small-K-quant real fixtures — pin `Q3_K_S` and `Q2_K` copies of the Llama-3.2 fixture family and generate parity goldens from them, so both new formats are gated against the pinned reference on real files rather than synthetic blocks only (see DN-M35-3, DN-M35-4).
  - Verify: `dotnet test --filter Category=Parity -c Release` green on fixtures generated from `models/Llama-3.2-1B-Instruct-Q3_K_S.gguf` (612 MB) and `models/Llama-3.2-1B-Instruct-Q2_K.gguf` (554 MB); `Category=Serve` green on the existing Q4_K_M fixture with its peak working set recorded against the 4 GB ceiling.

#### Tests

- [✓] Automated Quant Registry: GGML id acceptance, descriptor geometry, stride and byte-length validation, unsupported-id diagnostics naming id and name. Verify: `dotnet test --filter Category=Loader -c Release` — 129/129.
- [✓] Automated Quant Kernel: `Q2_K` and `Q3_K` synthetic-block scalar-vs-AVX2 bit-identity, batch equivalence, and dequant against a double-precision reference. Verify: `dotnet test --filter Category=Kernel -c Release` — 94/94.
- [_] Automated Quant Parity: real-fixture decode parity for the re-pinned sub-1 GB Llama-3.2 quant against pinned b12032 inside the documented quantized band. Verify: `dotnet test --filter Category=Parity -c Release`.
- [_] Manual Budget Serve: serve the re-pinned Llama fixture and confirm the monitor-reported working set is under the ceiling. Verify: live two-turn exchange on `:8181` with a tight profile (`max_context_tokens`, `kv_blocks`, `thread_quota`), plus a `/monitor/stream` snapshot frame read.

### MVP3.6 [v0.3.6] - Refined

Tool-call and model output formats are handled by load-time parser registries instead of the static fenced-JSON detector.

#### Design Notes

- **DN-M36-1** — Output parsing is a registry selected at model load from architecture, chat-template capabilities, and explicit config overrides. Built-in parsers may implement fenced JSON plus native model-family tool formats; selection happens before serving, not per-token in the decode loop.
  - Rationale: general GGUF compatibility requires recognizing more than LAIR's MVP2 fenced-JSON fallback. Model families emit different tool/reasoning markup, and parser behavior must be attributable to the loaded model rather than inferred from raw text.
  - Rejected: a single static detector for all models; per-token parser dispatch during decode; separate libraries per parser family without packaging need.
  - Contract: `lair.http`/`lair.models` request pipeline consumes a resolved output-parser plan and preserves OpenAI wire shape.
- **DN-M36-2** — Selection keys are the chat-template capability probes plus a config override, never the architecture name alone: `models[].tool_parser: auto | template | fenced | none`, default `auto` resolving to today's precedence rule. The resolved choice is logged at load and reported by `/v1/models`.
  - Rationale: precedence is already capability-driven (`TemplateCapabilities` feeding `ShouldInjectFencedTools` in `GenerationService.cs:825-832`), so the registry formalizes what exists. Architecture name is a weak key because a fine-tune can change tool markup without changing `general.architecture`, and none of the pinned fixtures carries a GGUF metadata key that states a tool format.
  - Rejected: architecture-name-only selection (wrong for fine-tunes); inventing a GGUF tool-format key read (no such key exists on the fixtures LAIR actually loads); sniffing raw text to decide (the posture DN-M36-1 rejects).
  - Contract: load resolves an `OutputParserPlan { ToolParser, ReasoningRegion }`; `lair.yaml` gains `models[].tool_parser` with a new-requests-only hot-reload class.
- **DN-M36-3** — Reasoning-region recognition belongs to this phase; stripping is a separate config choice. The recognizer resolves the open/close markup of a reasoning region (the `<think>` … `</think>` shape Qwen3 emits today) and reports boundaries to the decode loop; `models[].reasoning: keep | strip` decides visibility, defaulting to `keep`.
  - Rationale: nothing in `src` recognizes reasoning markup today — Qwen3's `<think>` block streams raw into response content, which is why a 128-token request can spend its whole budget on reasoning and return no answer. MVP3.10's reasoning ceiling is unimplementable without a resolved region, and a default of `keep` keeps this phase's change observable rather than silently altering output.
  - Rejected: stripping reasoning unconditionally (destroys the evidence the budget gates need); guessing region boundaries heuristically per token; a standalone reasoning library.
  - Contract: `OutputParserPlan.ReasoningRegion` exposes boundaries to the decode loop and to MVP3.10; default wire content is byte-identical to today.
- **DN-M36-4** — Parser families in scope are exactly three: the existing fenced-JSON detector moved behind the registry, template-native tool rendering (the path the pinned Llama-3.2 template already exercises), and reasoning-region recognition. No model-specific native tool parser lands without a fixture emitting it.
  - Rationale: the pinned fixtures and templates LAIR already renders cover these three and nothing else; adding parsers with no emitting fixture repeats the demand rule DN-M35-3 applies to quant formats.
  - Rejected (as of this refinement): a Hermes-style `<tool_call>` XML parser (no pinned fixture emits it), a Gemma tool parser (its template renders no tools today), multi-format auto-detection (ambiguous precedence, no fixture to arbitrate).
  - Contract: built-in parsers implement one `IToolOutputParser` shape resolved by the registry; `Category=ToolCalls` goldens per parser.

#### Implementation Items

- [_] Tool-call parser registry — define the parser interface and the load-time selection path, and move fenced-JSON detection behind a built-in parser (see DN-M36-1, DN-M36-2).
  - Verify: `dotnet test --filter Category=Api -c Release` — existing tool wire and streaming-delta cases green unchanged; a load-time log line names the resolved parser for the loaded model.
- [_] `tool_parser` config key — `auto | template | fenced | none` per model with load-time validation, a new-requests-only hot-reload class, and `/v1/models` reporting the resolved choice.
  - Verify: `dotnet test --filter Category=Config|Api -c Release` — invalid values fail naming the model and key; `none` produces byte-identical content with no `tool_calls`; the hot-reload path applies to new requests only.
- [_] Template-native tool parser — recognize calls produced by the model's own template rendering and emit them as OpenAI `tool_calls` deltas (see DN-M36-4).
  - Verify: `dotnet test --filter Category=ToolCalls -c Release` — goldens for the pinned Llama-3.2 template's tool format, with the fenced path proven inactive for the same model.
- [_] Reasoning-region recognizer and `reasoning: keep | strip` — resolve region boundaries from the loaded model's markup and apply the visibility choice (see DN-M36-3).
  - Verify: `dotnet test --filter Category=ToolCalls|Serve -c Release` — region boundaries detected on the Qwen3 serve fixture's `<think>` markup; `strip` removes the region from visible content while `keep` is byte-identical to today; finish reasons still report `length` versus `eos` unchanged.

#### Tests

- [_] Automated ToolCalls: parser selection per model, fenced and template-native goldens, streaming-delta compatibility, and `tool_parser: none` inertness. Verify: `dotnet test --filter Category=ToolCalls|Api -c Release`.
- [_] Automated Reasoning Region: boundary detection on pinned-style markup, `keep` versus `strip` content equality, and config validation failures. Verify: `dotnet test --filter Category=ToolCalls -c Release`.
- [_] Manual Parser Proof: serve the pinned Qwen3 fixture with tools declared and confirm the response separates reasoning from visible answer text, then flip `reasoning: strip` in the live config and confirm the next request differs while the one in flight does not. Verify: live two-request sequence plus the hot-reload log line.

### MVP3.7 [v0.3.7] - Refined

RoPE scaling and positional-encoding variants load through RoPE strategy adapters, including YaRN/LongRoPE/partial rotary where GGUF metadata declares them.

#### Design Notes

- **DN-M37-1** — Positional encoding is a strategy selected at model load. Standard full-head RoPE remains the default; rope-scaling variants are adapters that resolve scaling tables, attention scaling behavior, partial rotary dimensions, and model-specific metadata before decode.
  - Rationale: YaRN and related rope-scaling schemes are not just theta adjustments; they affect rotary table construction, attention scaling, and sometimes partial rotary handling. They should not be folded into MVP3.2 or hidden inside architecture branches.
  - Rejected: treating rope scaling as a metadata-only tweak; implementing YaRN before template/tokenizer compatibility is stable; per-token rope strategy dispatch during decode.
  - Contract: `lair.models` resolves a concrete RoPE plan at load and feeds the layer executor through table-driven operations.
- **DN-M37-2** — Scaling variants become plan values resolved from the declared metadata, and the blanket rejection goes away: today every `<arch>.rope.scaling.*` key fails at load (`ModelPlans.cs:101-108`) and every `rope.dimension_count` that differs from the head dimension fails as unsupported partial rotary (`ModelPlans.cs:176-184`). Both become resolved values — a precomputed rotary table, an attention-logit scaling factor, and a rotary slice width — with unsupported *variant names* still failing actionably.
  - Rationale: the rejection is what stands between LAIR and the Phi family, and the fix is the same shape as every other MVP3.4 resolution: read the keys, resolve them once, fail on what is genuinely unimplemented rather than on the category.
  - Rejected: keeping the blanket rejection and special-casing one model; computing scaled rotary values inside the decode loop (allocation and dispatch in the hot path); resolving scaling from file names or architecture guesses instead of declared keys.
  - Contract: `ModelPlan` carries `RopeScaling { Variant, Multiplier, OriginalContextLength, AttentionFactor, CorrectionScale }` plus `RotarySlice`; `Category=Loader` asserts each declared key resolves or fails naming key and value.
- **DN-M37-3** — YaRN is the only variant implemented in this phase, gated on `bartowski/Phi-3-mini-4k-instruct-GGUF` at `Q2_K` (1,351 MB, loads near 1.75 GB against the 4 GB ceiling) plus double-precision scalar goldens for the rotary table and the attention-factor application.
  - Rationale: Phi-3-mini is the smallest published model that actually declares rope scaling, its tensors are llama-shaped so the existing profile machinery carries it, and `Q2_K` is a format MVP3.5 lands — so the fixture needs no new format work. LongRoPE, linear, and dynamic scaling stay out because no fixture declares them.
  - Rejected: gating YaRN on synthetic GGUFs only (proves LAIR against itself); using Phi-3's IQ2/IQ1 quants (worse quality makes parity tolerance meaningless); implementing three scaling variants with one fixture.
  - Contract: a `phi3` profile registers the family's tensor map and its declared scaling defaults; `Category=Layers` asserts the scaled table differs from unscaled RoPE and matches a scalar reference; `Category=Parity` asserts real-model parity inside the quantized band.

#### Implementation Items

- [_] RoPE strategy adapter — resolve declared scaling keys and partial-rotary slices into plan values, and keep unsupported variant names actionable (see DN-M37-2).
  - Verify: `dotnet test --filter Category=Loader -c Release` — standard RoPE cases unchanged; a declared `rope.scaling.name` of an unimplemented variant fails naming the variant; `rope.dimension_count` below the head dimension resolves to a rotary slice instead of rejecting.
- [_] YaRN table and attention scaling — build the scaled rotary table and apply the attention factor in both attention paths (see DN-M37-3).
  - Verify: `dotnet test --filter Category=Layers -c Release` — scalar-reference goldens for the table at positions beyond the original context length, a scaled model provably differing from the unscaled one, and batched decode equal to single-token.
- [_] Phi-3 family profile and parity — register `phi3` on the profile registry and prove the family against the pinned reference on the `Q2_K` fixture (see DN-M37-3).
  - Verify: `dotnet test --filter Category=Parity -c Release` green on a fixture generated from `models/Phi-3-mini-4k-instruct-Q2_K.gguf`, with the fixture's sha256 recorded and the RoPE pairing resolved by measurement rather than assumption.

#### Tests

- [_] Automated RoPE Strategies: standard RoPE regression, declared-key resolution and variant rejection, partial rotary slice behavior, YaRN table and attention-factor goldens. Verify: `dotnet test --filter Category=Loader|Layers -c Release`.
- [_] Automated Phi-3 Parity: real-model output parity for the pinned Phi-3-mini `Q2_K` fixture against pinned b12032 inside the documented band. Verify: `dotnet test --filter Category=Parity -c Release`.
- [_] Manual Long-Context Serve: serve the Phi-3 fixture and confirm output stays coherent past the model's original context length, where unscaled RoPE would degrade. Verify: live three-turn exchange with a long synthetic prompt plus a monitor reading showing the resolved scaling values.

### MVP3.8 [v0.3.8] - Refined

Dense IQ quantization loads, decodes, and serves on the CPU engine at two tiers — `IQ4_NL`/`IQ4_XS` for quality under the budget and `IQ3_XXS` as the owner's target quant — landed before MoE execution so the sparse path can reuse them.

#### Design Notes

- **DN-M38-1** — Three registry entries on the MVP3.5 descriptor seam, not a generic IQ kernel: ggml ids 20 `IQ4_NL`, 23 `IQ4_XS`, and 18 `IQ3_XXS`, each with block geometry verified against `ggml.h` and its own scalar-oracle plus AVX2 dequant/GEMV/GemvBatch paths in the shape DN-M31-1 established.
  - Rationale: the owner's runs target IQ3_XXS, MVP6's expert slabs (DN-M6-2) assume that format, and `IQ4_XS` is already published for a fixture family at budget size (`unsloth/Llama-3.2-1B-Instruct-GGUF` ships `IQ4_XS` at 709 MB, inside the ceiling and better quality per byte than the 770 MB Q4_K_M currently pinned). Landing both tiers together shares one fixture set, one oracle harness, and one registry change instead of repeating them in a later phase.
  - Rejected (preliminary): `IQ2_XXS`/`IQ2_XS`/`IQ2_S`, `IQ1_S`/`IQ1_M`, and `TQ1_0`/`TQ2_0` — sub-3-bit quality with the hardest packing in the family and no fixture demand anywhere in this plan; a single "generic IQ" kernel (the formats do not share block layout or table structure); leaving dense coverage in the MVP5 series, which puts an unbuilt dependency behind MoE execution.
  - Contract: `GgmlTypeRegistry`/`QuantDescriptor` accept ids 18, 20, and 23; `QuantFormat` gains `IQ4_NL`, `IQ4_XS`, and `IQ3_XXS`; `TensorShape` row stride and `CpuBackend` dequant/GEMV/GemvBatch paths follow DN-M31-1.
- **DN-M38-2** *(tentative)* — Every format in this phase is defined by the pinned reference's own dequant tables, not by a LAIR-authored golden: `IQ4_NL`/`IQ4_XS` through the reference's 4-bit IQ path and `IQ3_XXS` through its 3-bit one. Each gate is scalar-vs-AVX2 bit-identity plus real-fixture parity against pinned b12032, and each format must also record measured GEMV cost against the Q4_K_M baseline before it can be recommended rather than merely supported.
  - Rationale: IQ formats dequant through lookup tables whose exact values the reference owns, so a hand-written golden would only prove LAIR agrees with itself; and the family's known weakness is shuffle-heavy unpacking on AVX2, which has to be measured per format rather than assumed from the K-quant results.
  - Rejected (preliminary): synthetic-only gating; accepting "close enough" decode differences; shipping a recommended default quant from paper sizes instead of measured tok/s; porting llama.cpp's table structures without measuring the managed AVX2 path.
  - Contract: `Category=Kernel` asserts per-format scalar-vs-AVX2 and 1T-vs-4T bit-identity with `GemvBatch` equal to solo `Gemv`; `Category=Parity` asserts decode parity on real fixtures inside the quantized band; bench records each format's GEMV cost against Q4_K_M.
- **DN-M38-3** — Kernel strategy is decided per format family rather than shared: `IQ4_NL` and `IQ4_XS` get fused dequant-GEMV on AVX2 (their unpacking is arithmetic, no lookup tables), while `IQ3_XXS` dequantizes through the reference's asymmetric sub-block tables into a staging buffer before the fused GEMV, and the bench gate decides whether that staging survives.
  - Rationale: the 4-bit pair is the cheapest IQ work LAIR has ever done and the safest proof of the registry seam; `IQ3_XXS` is where table-driven unpacking could genuinely cost more than it saves, which is exactly the measurement DN-M38-2 demands rather than a guess.
  - Rejected: one shared IQ kernel shape (the 3-bit path needs tables the 4-bit path does not); porting llama.cpp's table structures wholesale into managed code before measuring; shipping `IQ3_XXS` as a recommendation on paper sizes alone.
  - Contract: `CpuBackend` keeps one fused-GEMV entry shape per format; staging-buffer dequant is permitted only where a format's tables require it, and bench records its cost.

#### Implementation Items

- [_] IQ registry entries and geometry — ids 18 `IQ3_XXS`, 20 `IQ4_NL`, and 23 `IQ4_XS` accepted through the MVP3.5 descriptor registry with correct block size, type size, and row stride, and per-tensor type resolution proven inside a single file (see DN-M38-1).
  - Verify: `dotnet test --filter Category=Loader -c Release` — synthetic tensors for all three ids accepted with correct stride; a truncated or mis-strided tensor fails naming the tensor, the id, and expected geometry; a file whose tensors carry different GGML ids resolves each independently; an out-of-scope id such as 22 `IQ2_S` still rejects naming its name and id, proving the scope boundary.
- [_] `IQ4_NL` and `IQ4_XS` kernels — scalar oracle plus AVX2 fused dequant and batch path per the settled strategy (see DN-M38-1, DN-M38-3).
  - Verify: `dotnet test --filter Category=Kernel -c Release` — synthetic-block scalar-vs-AVX2 and 1T-vs-4T bit-identity for each format, `GemvBatch` equals solo `Gemv`.
- [_] `IQ3_XXS` kernel — scalar oracle mirroring the reference tables, plus the AVX2 staging-then-fused-GEMV path (see DN-M38-1, DN-M38-3).
  - Verify: `dotnet test --filter Category=Kernel -c Release` — same gate as the 4-bit pair on a full super-block including its sub-scale packing and asymmetric table values.
- [_] IQ fixture set and end-to-end serve — pin `IQ4_XS` at 709 MB (`bartowski/Llama-3.2-1B-Instruct-GGUF`) and `UD-IQ3_XXS` at 549 MB (`unsloth/Llama-3.2-1B-Instruct-GGUF`), load, decode, and serve both (see DN-M38-2).
  - Verify: automated real-fixture decode sanity (skip-if-absent) plus manual `lair serve` chat through the API and `/ui`, with `/v1/models` reporting `quant: IQ4_XS` and `quant: IQ3_XXS`; the `UD-` file's mixed per-layer types load through the same registry path.
- [_] Format cost measurement and documented recommendation — GEMV and decode throughput for each new format against the `Q4_K_M` baseline, with the measured winner recorded in the user-facing docs (see DN-M38-2).
  - Verify: `dotnet run --project bench/Lair.Bench -- <args>` records a per-format GEMV and decode tok/s table on the same fixture model at equal context, and `README.md` names the recommended quant only for the format that actually wins.

#### Tests

- [_] Automated Registry: acceptance for ids 18/20/23, geometry and stride validation, mixed-type file resolution, out-of-scope id rejection naming name and id. Verify: `dotnet test --filter Category=Loader -c Release`.
- [_] Automated Kernel: per-format scalar-vs-AVX2 bit-identity, batch equivalence, and real-fixture decode sanity. Verify: `dotnet test --filter Category=Kernel|Parity -c Release`.
- [_] Manual IQ Serve: both tiers served; chat via API and `/ui`; monitor live with the quant reported.
- [_] Manual Quant Recommendation: read the bench table and confirm the documented recommendation matches the measured fastest format rather than the smallest file.

### MVP3.9 [v0.3.9] - Refined

Sparse and windowed architectures execute: dense-resident MoE decoding and sliding-window/local attention, both through the MVP3.4 model plan.

#### Design Notes

- **DN-M39-1** — MoE executes with all experts resident in RAM: router scoring, top-k selection with renormalization, shared expert, and per-expert gate/up/down GEMVs accumulated into the residual. Gather and residency optimization stay in MVP6, where the Flash-Next measurement gate and tier design live.
  - Rationale: proving sparse-FFN math against a dense-equivalent reference needs no residency machinery, and doing it before MVP6 turns `IGatherKernel` from a new concept into an optimization of a proven path.
  - Rejected: expert streaming or hot/cold tiers here (that is MVP6's measured design); MoE-only gather kernels before a correctness baseline.
  - Contract: `lair.models` executes the sparse-FFN descriptor resolved by the MVP3.4 plan; `Category=Moe` correctness goldens.
- **DN-M39-2** — The KV-layout half of windowed attention: bounded KV reads, per-layer windowed/full interleave, and prompt-cache reuse under windowed reads. The score-mask half — honoring a declared window by excluding out-of-range positions while keeping full KV — already ships in MVP3.4 (DN-M34-7); this note covers what changes once KV stops storing every position.
  - Rationale: Mistral and Gemma-3 are unusable without bounded reads and interleave, and the window interacts with the paged KV block design (DN-M2-4) rather than being an attention-loop constant.
  - Rejected: treating the window as an attention-kernel parameter only (ignores block reuse and KV budget accounting).
  - Contract: plan carries per-layer window and interleave pattern; `lair.kv` block selection honors it.
- **DN-M39-3** — The MoE fixture is `bartowski/OLMoE-1B-7B-0924-Instruct-GGUF` at `Q2_K` (2,444 MB, loads near 2.8 GB against the 4 GB ceiling), reached through a new `olmoe` profile whose tensor naming is measured against the pinned reference rather than assumed. Routing math is gated by dense-equivalent synthetic goldens first and real-model parity second.
  - Rationale: it is the smallest published MoE GGUF that loads under the ceiling in a format MVP3.5 lands — its `IQ2_M` file (2,220 MB) is smaller but needs an out-of-scope IQ2 format, and no published `qwen2moe` or `qwen3moe` file is small enough to load at all. OLMoE has no shared expert, so the descriptor's shared-expert slot must be provably optional rather than assumed present.
  - Rejected: a `qwen2moe`/`qwen3moe` fixture (none published under the ceiling); synthetic-only MoE gating (leaves the family's tensor naming and router semantics unproven); assuming the qwen-moe tensor names for OLMoE (the reference names experts per-expert, not as one stacked tensor).
  - Contract: `ModelProfiles` gains `olmoe`; `Category=Moe` asserts router/top-k/renormalize against a dense-equivalent reference and `Category=Parity` asserts decode parity on the pinned `Q2_K` fixture inside the quantized band.
- **DN-M39-4** — Windowed-KV correctness is proven on the already-pinned Gemma-2 fixture driven with reference-supplied token ids, the technique MVP3.4 used for the forward graph; the small Gemma-3 file (`unsloth/gemma-3-1b-it-GGUF` `Q3_K_S`, 657 MB) becomes the *served* fixture only after MVP3.11's SPM profile exists.
  - Rationale: bounded KV reads and cache accounting are a memory-behavior change measurable on a file already in the repo, and blocking it behind a tokenizer phase would invert the dependency. Gemma-3-1B is the right serving fixture because it is the only windowed model published small enough to serve comfortably, but its SentencePiece vocabulary is exactly what MVP3.11 has to build first.
  - Rejected: waiting on Gemma-3 for all windowed-KV work (delays a memory change behind tokenization); synthetic-only gating (block selection and cache reuse are precisely what synthetic logits will not catch); using a Mistral file (no published Mistral is loadable under the ceiling).
  - Contract: `Category=Layers|Cache` asserts windowed logits match the full-KV reference while block selection reads only the in-window range across alternating layers; `Category=Serve` for the Gemma-3 fixture lands in MVP3.11.
- **DN-M39-5** — Encoder parity follows the reference's backward-only masking for a windowed model: LAIR's `Encode` path keeps its symmetric range only where no window is declared, and applies the same backward-only window the causal path uses when one is.
  - Rationale: the pinned reference exposes only backward-only masking on a windowed file, so a symmetric encoder range would produce embeddings no oracle can corroborate; avoiding an unprovable convention is cheaper than unwinding it in an embedding fixture.
  - Rejected: keeping the symmetric range for windowed models (unprovable against any reference); dropping embeddings for windowed models (a feature loss with no measurement behind it).
  - Contract: `Encode` honors `LayerAttentionWindow` backward-only when a window is declared; `Category=Embeddings` asserts windowed-model embeddings against a reference-supplied-token-id run.

#### Implementation Items

- [_] MoE execution — router scoring, top-k with renormalize, optional shared expert, and per-expert FFN accumulation into the residual (see DN-M39-1, DN-M39-3).
  - Verify: `dotnet test --filter Category=Moe -c Release` — sparse-FFN output matches a dense-equivalent reference across router tie-breaking, renormalization, and a model with no shared expert.
- [_] `olmoe` profile and MoE fixture parity — register the family, resolve its expert tensor naming from the pinned file, and prove decode parity (see DN-M39-3).
  - Verify: `dotnet test --filter Category=Loader|Parity -c Release` — the pinned `Q2_K` OLMoE file resolves its descriptor with the reference's own tensor names and reports `olmoe`; parity green inside the quantized band with the fixture sha256 recorded.
- [_] Windowed KV layout — bounded KV reads, per-layer windowed/full interleave, and correct block selection and prompt-cache reuse on top of the shipped score mask (see DN-M39-2, DN-M39-4).
  - Verify: `dotnet test --filter Category=Layers|Cache -c Release` — windowed logits match the full-KV reference while block selection reads only the in-window range, cache reuse stays correct across windowed and full layers, and a full-attention layer's reads are provably unbounded.
- [_] Windowed KV budget accounting — KV block pools and the monitor report the reduced read footprint so the memory behavior is observable rather than implied (see DN-M39-2).
  - Verify: `dotnet test --filter Category=Cache|Metrics -c Release` — a windowed model's reported KV usage differs from the same model unwindowed, and the monitor snapshot carries the numbers.

- [_] Windowed encoder masking — `Encode` applies the declared window backward-only, matching the reference's only observable behavior (see DN-M39-5).
  - Verify: `dotnet test --filter Category=Embeddings|Layers -c Release` — a windowed model's embeddings differ from the same model with the window removed and match a backward-only scalar reference; a model with no declared window stays bit-identical to today's symmetric path.

#### Tests

- [_] Automated Moe: router/top-k/shared-expert correctness vs dense-equivalent math, plus real-model parity on the pinned OLMoE `Q2_K` fixture. Verify: `dotnet test --filter Category=Moe|Parity -c Release`.
- [_] Automated Windowed KV: interleave patterns, KV block selection bounds, cache-reuse correctness across windowed and full layers, and backward-only encoder masking. Verify: `dotnet test --filter Category=Layers|Cache|Embeddings -c Release`.
- [_] Manual MoE Serve: serve the pinned OLMoE fixture and confirm coherent multi-turn output with the monitor showing expert weights resident and no gather machinery involved. Verify: live two-turn exchange on `:8181` plus a `/monitor/stream` snapshot frame read.

### MVP3.10 [v0.3.10] - Refined

Token budgets are configured per model: a reasoning budget and a response budget, each overridable per request.

#### Design Notes

- **DN-M310-1** — Two budgets per model, `max_reasoning_tokens` and `max_response_tokens`, set per model in `lair.yaml` and overridable per request. The scheduler reserves against both, the decode loop stops at the reasoning ceiling while a recognized reasoning region is open, and the visible response is bounded by the response budget.
  - Rationale: today one `max_tokens` (default 128 at `ApiApp.cs:245`/`ApiApp.cs:282`, clamped at `GenerationService.cs:500`) covers every emitted token, so a thinking model can spend the whole budget on reasoning and return nothing visible — measured live this phase, where a 128-token Qwen3 request finished `length` with no answer. Budgets belong to the loaded model because the ceiling is a property of the model's output style, not of each request.
  - Rejected: a single combined budget (visible output starves); per-token heuristics that guess reasoning without a resolved parser; a server-wide setting with no per-model override.
  - Contract: `lair.yaml` model entries and `lair.http` request fields resolve into one budget pair consumed by `lair.models` scheduling; the OpenAI wire shape stays compatible with documented extension fields.
- **DN-M310-2** — Budgets ride on dedicated request fields, `max_reasoning_tokens` and `max_response_tokens`, never on `max_tokens`. Absent model configuration means today's behavior exactly: no reasoning budget, and `max_tokens` bounds the whole generation.
  - Rationale: `max_tokens` is already consumed by OpenAI clients with a fixed meaning, so overloading it breaks clients for no gain, and a default of "no budget" keeps this phase observable as an opt-in feature rather than a silent behavior change for every existing user.
  - Rejected: overloading `max_tokens` (client-visible semantic break); making budgets mandatory (changes serving behavior for models with no reasoning region); a single `thinking_budget` field with no response ceiling.
  - Contract: `ApiContracts` request bodies gain two optional ints validated against the model ceiling; `YamlConfigLoader` model keys gain `max_reasoning_tokens` and `max_response_tokens` with a new-requests-only hot-reload class.
- **DN-M310-3** — The reasoning ceiling closes the region rather than truncating mid-region: when the reasoning budget is reached with a region open, the recognizer closes it and decoding continues into the response budget. Which ceiling fired is reported, so a truncated answer is attributable.
  - Rationale: truncating inside a reasoning region leaves an unclosed block in visible content and no answer at all, which is the failure this phase exists to prevent. Reporting which ceiling fired turns `finish_reason: length` into an actionable reading instead of a guess.
  - Rejected: hard stop mid-region (unterminated markup, no answer); silently extending the budget when reasoning runs long (defeats the point of a budget); reporting only `length` as today.
  - Contract: decode honors the region close at the ceiling; the monitor and response metadata distinguish `reasoning` versus `response` budget exhaustion while `finish_reason` stays `length`.
- **DN-M310-4** — Sequencing is fixed by dependency, not convention: this phase follows MVP3.6 because a reasoning ceiling needs `OutputParserPlan.ReasoningRegion` resolved at load (DN-M36-3), and it follows MVP3.9 in plan order only because that keeps execution work ahead of serving policy.
  - Rationale: without a resolved region the ceiling is a guess, which DN-M36-1 and DN-M36-3 both refuse.
  - Rejected: implementing budgets before the parser registry (guessed regions); making budgets part of MVP3.6 (different concern: parsing recognizes markup, budgets bound work).
  - Contract: scheduler KV reservation math consumes both budgets from the resolved model plan; `Category=Scheduling|Api` gates the math.

#### Implementation Items

- [_] Per-model budget configuration — model-level `max_reasoning_tokens` and `max_response_tokens` defaults with load-time validation, absent meaning unbudgeted (see DN-M310-1, DN-M310-2).
  - Verify: `dotnet test --filter Category=Config -c Release` — model entries resolve both budgets, invalid or negative values fail at load naming the model and the key, and the hot-reload class reports them as new-requests-only.
- [_] Request overrides and wire fields — `max_reasoning_tokens` and `max_response_tokens` on chat and completion bodies, clamped to the model ceiling (see DN-M310-2).
  - Verify: `dotnet test --filter Category=Api -c Release` — an override above the model ceiling clamps and reports the clamp; an override on a model with no resolved reasoning region is refused naming the model.
- [_] Budget enforcement in scheduling and decode — KV reservation against both budgets, region close at the reasoning ceiling, response ceiling on visible tokens (see DN-M310-1, DN-M310-3).
  - Verify: `dotnet test --filter Category=Scheduling -c Release` — reservation math, reasoning-ceiling region close, response-ceiling stop, and batched decode equal to single-token under both ceilings.
- [_] Budget attribution in responses and monitor — which ceiling fired is reported without changing `finish_reason` values (see DN-M310-3).
  - Verify: `dotnet test --filter Category=Api|Metrics -c Release` — a reasoning-capped exchange reports the reasoning budget, a response-capped one reports the response budget, and `/monitor/stream` carries the attribution field.

#### Tests

- [_] Automated Token Budgets: per-model defaults, request overrides and clamping, reasoning region close at the ceiling, response ceiling behavior, and KV reservation math. Verify: `dotnet test --filter Category=Config|Api|Scheduling -c Release`.
- [_] Automated Budget Attribution: monitor and response reporting distinguishing which ceiling fired. Verify: `dotnet test --filter Category=Api|Metrics -c Release`.
- [_] Manual Budget Proof: serve the pinned Qwen3 fixture with `max_reasoning_tokens: 128` and `max_response_tokens: 64` and confirm the answer appears where today a 128-token request returns only reasoning, then hot-edit the budgets and confirm the next request differs while the one in flight does not. Verify: live two-request sequence plus monitor readings.

### MVP3.11 [v0.3.11] - Refined

SentencePiece (SPM) vocabularies load through a tokenizer profile, unlocking the published families that ship them.

#### Design Notes

- **DN-M311-1** — Selection splits on a second axis: vocabulary type from `tokenizer.ggml.model`, which is orthogonal to the pre-tokenizer key the existing profiles are keyed on. `lair.tokenizer` gains a vocabulary-type seam with two implementations — the current BPE pipeline (pre-split, byte-encode, merges) and SPM — while `TokenizerProfiles` stays exactly what it is for `pre`.
  - Rationale: today the registry maps `tokenizer.ggml.pre` to a factory of BPE merge state (`TokenizerProfiles.cs:46-112`), and SPM has no merge stage at all — it encodes by greedy longest-match over a piece vocabulary. Registering SPM as another `pre` value would mis-key a different concept and force merge machinery to run where there is nothing to merge.
  - Rejected: SPM as another `pre` profile (wrong axis, assumes merges); offline conversion of SPM vocabularies to BPE (needs a conversion tool and produces a vocabulary the reference no longer agrees with); a separate tokenizer library per vocabulary type (two places to keep in sync with the oracle).
  - Contract: `GgufTokenizer` resolves a vocabulary implementation from `tokenizer.ggml.model` at load, defaulting to the BPE implementation when the key is absent, and an unknown value fails naming the value and the supported list.
- **DN-M311-2** — SPM encode is greedy longest-match over the piece vocabulary preferring the higher `tokenizer.ggml.scores` entry on ties, with `▁` treated as the word-start marker, byte fallback through the `<0xNN>` byte pieces the vocabulary already carries, and special-token ids taken from `tokenizer.ggml.token_type`. Decode joins pieces and maps `▁` back to a space except at the start of a sequence.
  - Rationale: those are the keys the pinned Gemma-2 file actually carries, and they are the same metadata-coverage posture MVP3.3 settled for BPE — read what exists, apply defaults visibly, fail on what is missing.
  - Rejected: greedy shortest-match or first-match strategies (the reference scores pieces for a reason); synthesizing byte fallback from UTF-8 escapes instead of the vocabulary's byte pieces (diverges from the reference on unrepresentable text); ignoring `token_type` and treating every piece as ordinary text (special tokens then leak into content).
  - Contract: `Category=Tokenizer` asserts id-for-id `/tokenize` goldens against pinned b12032 on an SPM fixture, and `Category=Serve` asserts a two-turn exchange through the model's own chat template.
- **DN-M311-3** — Fixtures are the already-pinned Gemma-2 file at `Q4_K_M` (1,629 MB, loads near 2.0 GB against the 4 GB ceiling) for tokenize parity, parity goldens, and serve, plus a new `unsloth/gemma-3-1b-it-GGUF` `Q3_K_S` fixture (657 MB) as the second SPM file and MVP3.9's deferred windowed-serve fixture.
  - Rationale: Gemma-2 is already in the repo with a recorded sha256 and a forward graph already proven against reference-supplied token ids, so unblocking it needs no new download; Gemma-3-1B is the smallest published windowed model, so it is the only file that can prove MVP3.9's bounded-KV behavior end to end.
  - Rejected: re-pinning Gemma-2 at a smaller quant (no gate gains anything from it); using Gemma-1 (no longer a parity target anywhere in this plan); proving SPM on synthetic vocabularies only (the whole phase exists because a real file was blocked).
  - Contract: both fixtures sha256-pinned in their fixtures; `Category=Tokenizer|Serve` green on both; MVP3.9's `Category=Serve` windowed gate lands here rather than there.

#### Implementation Items

- [_] Vocabulary-type seam and SPM profile — resolve `tokenizer.ggml.model` at load, implement SPM piece selection with scores, byte fallback, `▁` handling, and special-token ids, and keep the BPE path byte-identical (see DN-M311-1, DN-M311-2).
  - Verify: `dotnet test --filter Category=Tokenizer -c Release` — synthetic SPM encode/decode round-trip goldens, tie-breaking on `scores`, byte-fallback on unrepresentable text, and existing BPE profile cases unchanged; an unknown `tokenizer.ggml.model` value fails naming the value and the supported list.
- [_] Load and serve coverage — the model load path selects the SPM implementation without a default guess, and the chat template renders against it end to end (see DN-M311-2).
  - Verify: `dotnet test --filter Category=Tokenizer|Serve -c Release` — the pinned Gemma-2 file loads, tokenizes, and completes a two-turn exchange.
- [_] Gemma-2 parity unblocked — generate the deferred MVP3.4 parity fixture from the pinned file and hold it inside the quantized band (see DN-M311-3).
  - Verify: `dotnet test --filter Category=Parity -c Release` green on `tests/fixtures/parity-gemma2.json` with the fixture sha256 recorded.
- [_] Gemma-3-1B windowed serve — pin the `Q3_K_S` fixture and land MVP3.9's deferred windowed serving gate (see DN-M311-3, DN-M39-4).
  - Verify: `dotnet test --filter Category=Serve|Cache -c Release` — the fixture serves a two-turn exchange past its local-attention window length with block selection staying bounded, and `Category=Loader` reports `gemma3` with its per-layer window pattern resolved.

#### Tests

- [_] Automated SPM Tokenizer: piece selection, score tie-breaking, byte fallback, special ids, round-trip stability, and `/tokenize` oracle parity on a real SPM fixture. Verify: `dotnet test --filter Category=Tokenizer -c Release`.
- [_] Gemma-2 Parity Unblocked: the deferred MVP3.4 fixture generates and passes inside the documented quantized band. Verify: `dotnet test --filter Category=Parity -c Release`.
- [_] Manual SPM Serve: serve the pinned Gemma-2 fixture and the Gemma-3-1B fixture, confirm coherent multi-turn output on both, and confirm `/v1/models` reports their architectures and quants. Verify: two live two-turn exchanges on `:8181` plus monitor readings.

### MVP3.12 [v0.3.12] - Refined

Prefill forwards batched rows instead of one token at a time, and the measurement decides whether a GEMM path is worth building. Scheduler-side work — step budgeting and prefill lanes — deliberately lives in MVP3.13, so the scheduler is rewritten once rather than twice.

#### Design Notes

- **DN-M312-1** — Batched prefill reuses the batch shape already shipped: chunk positions forward in sub-batches of up to `MaxBatchSequences` (16) rows through `GemvBatch`, with causal-within-chunk attention over the growing KV exactly as the embedding path already performs it. A 512-token chunk becomes 32 weight traversals instead of 512. The phase changes no scheduling behavior: chunk width, decode priority, and step composition are untouched.
  - Rationale: the engine already carries 16-row batch buffers and the encode path already proves causal-within-chunk attention against a growing KV; prefill is the only hot path not using them, and prompt cost is weight-read bound. Keeping the change engine-local is also what keeps it testable before the arbiter exists.
  - Rejected: GEMM-first (register-pressure wall plus native kernel work before measuring what batched GEMV delivers); widening the batch cap past 16 in this phase (buffer memory and register pressure, unmeasured); retaining per-token prefill as a safety path (batched decode and encode already prove bit-identical batched equivalence); doing the step-time budget and prefill lane here (both are arbiter decisions over MVP3.13's pipelines, and editing `GenerationScheduler` twice buys no gate anything).
  - Contract: `TransformerModel.Prefill` takes a row window and returns logits for the final row only; `Category=Prefill` asserts chunked-prefill logits and resulting KV bytes bit-identical to per-token prefill.
- **DN-M312-2** — The placement vocabulary stays device-neutral from the first commit: prefill carries the same shape as decode — compute device, weight tier, lane, quota — so CPU prefill and GPU prefill are the same design expressed as values, and neither is privileged.
  - Rationale: the device split must be data, not a code path, or the runtime grows a second scheduler; MVP4 lands the CUDA side against this seam instead of inventing a prefill design of its own.
  - Rejected: a CPU-only prefill design with GPU deferred to a separate mechanism; inventing `prefill: hybrid` layer semantics here (layer split is DN-M4-2's decision); per-device scheduler branches.
  - Contract: `PlacementPolicy.prefill` is read as a route request, and the CPU implementation rejects an unsupported device value naming it rather than silently falling back, consistent with DN-M4-3.
- **DN-M312-3** — GEMM prefill is a measured follow-up, not this phase's delivery: a `[LibraryImport]` GEMM spike opens only if the bench shows batched GEMV leaving ≥30% headroom against the pinned reference on the same fixture.
  - Rationale: the plan already states CPU prefill parity needs microkernels, but the register-pressure wall is the unknown; measuring batched GEMV first costs less than building a GEMM path speculatively.
  - Rejected: shipping GEMM in this phase (unmeasured cost); leaving it in Backlog with no trigger (it then never gets decided).
  - Contract: bench records batched-GEMV prefill tok/s against the llama.cpp reference, and the 30% threshold is the recorded gate for opening the GEMM item.

#### Implementation Items

- [_] Batched prefill in the engine — row-window forward through `GemvBatch` with causal-within-chunk attention over the growing KV (see DN-M312-1).
  - Verify: `dotnet test --filter Category=Prefill -c Release` — chunked-prefill logits, KV bytes, and the first sampled token are identical to per-token prefill across the pinned fixtures, and `Category=Parity` stays green.
- [_] Prefill bench record — batched-GEMV prefill tok/s against the pinned reference, before and after batching, with the 30% headroom threshold evaluated (see DN-M312-3).
  - Verify: `bench/` report committed with prefill tok/s for both paths and the GEMM decision written into the Decision Log. *(bench deferred per the 2026-11-02 owner decision — the gate stays open at ship time as a recorded deferred gate, and the GEMM spike cannot be opened until it closes.)*

#### Tests

- [_] Automated Prefill Identity: batched prefill bit-identical to per-token prefill in logits, KV bytes, and sampled tokens. Verify: `dotnet test --filter Category=Prefill|Parity -c Release`.
- [_] Manual Prefill Speed: serve the pinned fixture and confirm a long prompt reaches first output token measurably earlier than before batching, with decode tok/s for a second model unchanged. Verify: live monitor readings for both models plus the bench record.

### MVP3.13 [v0.3.13] - Refined

Requests become independent pipelines advanced by a route-aware arbiter, so concurrency, device assignment, and batching are configured per model rather than fixed by the scheduler's shape.

#### Design Notes

- **DN-M313-1** — Control plane and execution plane are separated: `RequestPipeline` owns a request's state machine (admit → transform → prefill → decode → finish), its sampler, and its quota; the step executor decides how a pipeline's step math runs. Pipelines are logical state machines advanced by the arbiter, not OS threads.
  - Rationale: the pipeline model is device-neutral and matches how GPU serving already works, while CPU decode still needs merged batches for weight reuse; conflating them is what currently makes one global step loop the scheduler's whole identity.
  - Rejected: one OS thread per request (bandwidth thrash, no weight sharing); keeping the single loop and bolting a lane onto it (cannot express a hybrid route); making batching a request-level option (batching is a device-cost decision, not a user knob).
  - Contract: `GenerationScheduler` exposes pipeline admission and step composition as distinct seams; a pipeline's token stream is bit-identical merged or alone (`Category=Scheduling` identity assertions carried over from MVP2).
- **DN-M313-2** — The merge key is route identity, not device identity: pipelines merge into a step when they share a model plan, share that model's route, have compatible KV residency for the segment, fit the segment's batch cap, and leave quota headroom for the pipelines already in the step.
  - Rationale: a hybrid model's sequences all ride one route (layer split and expert tier map are plan values), so they must merge even though the step crosses devices; merging on device alone would split them and pay the boundary tax per sequence.
  - Rejected: per-device merge groups (splits a hybrid model's batch); merging across models (no shared weights to reuse, and it breaks per-model quotas); merging regardless of KV residency (a `vram` sequence cannot ride a `dram` batch).
  - Contract: `Category=Scheduling|Parity-Hybrid` asserts a hybrid route's merged step is token-identical to the same sequences stepped alone, and that a split point changes speed, not results.
- **DN-M313-3** — A route is an ordered list of device segments, and each segment carries compute device, weight tier, batch cap, and quota; admission cost is route cost.
  - Rationale: a hybrid step holds a CPU quota and a GPU quota simultaneously for its duration, so admission on a single device cost is wrong; the route is also the seam a later accelerator plugs into without redesign.
  - Rejected: a single `device` label carrying the whole model (cannot express frac or a hybrid route); per-sequence device choice (routing is per token, not per sequence); implicit device selection.
  - Contract: `PlacementPolicy` resolves to a route value set in the model plan; the arbiter reports per-segment quota use and the monitor attributes utilization per segment.
- **DN-M313-4** — Admission is a three-way check owned by the arbiter: KV block capacity (MVP2 math), the resolved token budgets (MVP3.10's reservation), and the model's concurrency slot with its per-segment quota. Quota is per pipeline within `models[].max_concurrent_requests`: CPU segments take core-set shares, GPU segments take stream priority and SM share, and the sum a step needs is what admission checks.
  - Rationale: the user-facing requirement is "parallel up to a limit for that model" on every device class, and a request that wins a concurrency slot but cannot fit its KV or its budgets is a rejection nobody can act on. Making the arbiter the single admission point is what keeps three independently-built limits from each assuming the others; MVP3.10's reservation math and MVP5's device ledger become inputs rather than competing gates.
  - Rejected: unbounded concurrency with best-effort thread pools (unmeasurable ITL claims); per-request dedicated cores on CPU decode (multiplies bandwidth traffic); a global concurrency number shared across models (breaks per-model budgets); checking the cap in the HTTP layer and the budgets in the scheduler (three rejections with three different shapes).
  - Contract: `models[].max_concurrent_requests` and per-segment quota keys validate at load and hot-reload as new-request-only; admission rejection names which of the three limits blocked and the model it belongs to; `Category=Scheduling|Api` gates the three-way reporting.
- **DN-M313-5** — MVP3.13 lands the arbiter and route vocabulary on CPU only, and declares the CUDA seam rather than implementing it: a route naming a GPU segment is rejected naming the missing backend until MVP4 lands it.
  - Rationale: GPU concurrency is unserviceable without the pipeline and route seams, so they must exist before CUDA work starts — but building a GPU path with no device present is untestable, and DN-M4-3 already forbids silent fallback.
  - Rejected: doing this rearchitecture inside MVP4 (CPU-only deployment loses the benefit); shipping speculative GPU code paths unmeasured.
  - Contract: `Category=Scheduling` green on CPU routes; a GPU-segment route on a device-less box fails with the backend named, and MVP4's parity gate exercises the same route values.
- **DN-M313-6** — A route resolves to a **backend set**, not one backend: each segment names an `IBackend` implementation and its weight tier, and a step calls them in route order over the *same* merged batch. Two distinct concurrency modes follow from it — segment-parallel (one batch, several backends along the layer/tensor order, as in a layer split or frac) and request-parallel (several batches on several backends at once, as in two models on different devices).
  - Rationale: model-level concurrency already has this shape — each model instance binds its own `IBackend` at `TransformerModel.Open` and its pool at `BlockPool` — but both accept exactly one backend today, so a hybrid or frac step requires the engine to hold one per segment before MVP4 can name a split point. Keeping the batch whole across segments is what makes the crossing cheap: the handoff carries `batch width × hidden` once, not per sequence.
  - Rejected: hybrid emulated inside `CpuBackend` (device branching inside the CPU kernel path); splitting one merged batch's rows across backends for the same segment (destroys the weight reuse batching exists to create); a per-sequence backend choice (routing is per token, not per sequence); one backend process per device pair (duplicates weights and plans).
  - Contract: `TransformerModel` and `BlockPool` resolve a backend set from the route at load; activation handoff is a backend-allocated staging buffer; `Category=Scheduling` asserts a multi-backend step is token-identical to the single-backend step over the same fixtures.
- **DN-M313-7** — A step-time budget sizes each prefill step: the arbiter chooses how many *tokens* a step forwards from measured per-row cost so the step stays inside `engine.prefill_step_budget_ms`, hot-reloadable, floor 1 token, ceiling `max_batch_tokens`, executed as sub-batches of at most `MaxBatchSequences` rows. Chunk width and row width stay separate numbers in config and in the stats.
  - Rationale: MVP3.12's batching makes a chunk shorter in total but longer per step, so isolation is a measured bound rather than a fixed chunk width — it converts many uncapped stalls into a few capped ones and leaves decode-first priority untouched. Chunk width is a scheduling number and row width is a kernel number; conflating them makes one knob serve two jobs. It is an arbiter decision because the thing being bounded is a step the arbiter composes.
  - Rejected: fixed chunk width (starves decode or prefill); splitting one chunk across a step boundary (breaks the "identical to running alone" guarantee); a static row count (cost is model- and device-dependent); letting the budget raise row width past the batch cap (buffer memory and register pressure, unmeasured).
  - Contract: arbiter stats report tokens per chunk, sub-batch rows, and measured prefill step time; `Category=Prefill|Concurrency` asserts decode ITL p95 holds with an 8K-token prompt in flight.
- **DN-M313-8** — Dedicated prefill lane is opt-in per model: `prefill_lane: shared | dedicated` (default `shared`), where `dedicated` runs a pipeline's prefill on its own lane with its own activation buffers and its own quota, so decode steps never contain prefill work. Weights stay shared and read-only.
  - Rationale: a 30K-token prompt on shared cores lengthens a step however the budget is tuned; spatial isolation is the only mechanism that removes it, and its cost is bounded — activation buffers per lane, not duplicated weights. It needs pipelines to exist first, which is why it lands here rather than in MVP3.12: a lane is a property of a pipeline's progress, not of the engine.
  - Rejected: always-on dedicated lanes (buffer cost for models that never prefill long); a separate engine instance per lane (duplicate weights, breaks the shared-plan posture); treating the budget as sufficient for this case.
  - Contract: `PlacementPolicy` gains `prefill_lane` and a prefill quota (CPU core set, CUDA stream priority); lane buffers count in the model's load-time footprint and the monitor working-set reading; `Category=Concurrency|Affinity` asserts a dedicated-lane model's decode ITL is unaffected by its own in-flight prefill.

#### Implementation Items

- [_] Request pipeline control plane — pipeline state machine, per-pipeline sampler and quota, and admission independent of step composition (see DN-M313-1).
  - Verify: `dotnet test --filter Category=Scheduling -c Release` — pipelines advance independently up to the cap, and each pipeline's token stream is identical to the same sequence run alone.
- [_] Route value set in the model plan — ordered device segments with compute device, weight tier, batch cap, and quota, resolved at load (see DN-M313-3).
  - Verify: `dotnet test --filter Category=Loader|Config -c Release` — routes resolve for `cpu`, `cuda`, and `hybrid` model entries; an unresolvable segment fails naming the model and the segment.
- [_] Route-aware arbiter — merge predicate on route identity, KV-residency compatibility, batch cap, and quota headroom (see DN-M313-2).
  - Verify: `dotnet test --filter Category=Scheduling|Concurrency -c Release` — merged steps equal unmerged results, hybrid-shaped routes merge rather than split, and a step that cannot fit is refused with the blocking quota named.
- [_] Concurrency cap and per-segment quota keys — `max_concurrent_requests` plus CPU core-set and GPU stream-priority fields with load-time validation and hot-reload classes (see DN-M313-4).
  - Verify: `dotnet test --filter Category=Config|Api -c Release` — the cap admits and rejects the expected request counts, and a rejection names which of KV capacity, token budget, or quota blocked and the model it belongs to.
- [_] Step-time budget and adaptive token count — `engine.prefill_step_budget_ms` sizing tokens per step from measured per-row cost, executed as sub-batches within the batch cap, with floor and ceiling validation (see DN-M313-7).
  - Verify: `dotnet test --filter Category=Prefill|Concurrency -c Release` — a chosen budget produces token counts inside the floor/ceiling with sub-batch rows never exceeding the cap, the hot-reload class reports it as next-chunk, and ITL p95 holds with an 8K-token prompt in flight.
- [_] Dedicated prefill lane and quota — `prefill_lane` with its own activation buffers, core set, and stream-priority field, plus load-time footprint accounting (see DN-M313-8).
  - Verify: `dotnet test --filter Category=Concurrency|Affinity|Config -c Release` — a dedicated lane leaves decode ITL unchanged under its own prefill load, core-set collision rules extend to the new set, and an invalid lane value fails at load naming the model.
- [_] Quota and lane reporting — per-segment quota use, lane state, and admission rejections surfaced on `/v1/models` and `/monitor/stream` (see DN-M313-3, DN-M313-4).
  - Verify: `dotnet test --filter Category=Api|Metrics -c Release` — the monitor snapshot carries per-segment quota and lane state per model, and `/v1/models` reports the configured cap and route.
- [_] CUDA seam declaration — GPU-segment routes rejected naming the missing backend, with the seam exercised by a synthetic route in tests (see DN-M313-5).
  - Verify: `dotnet test --filter Category=Scheduling|Budget -c Release` — a GPU-segment route on a device-less host fails naming `lair.backends.cuda`, and MVP4 inherits the same route values unchanged.
- [_] Backend set from the route — engine and KV pool hold one `IBackend` per segment instead of one per model, with a staging buffer at each handoff (see DN-M313-6).
  - Verify: `dotnet test --filter Category=Scheduling|Loader -c Release` — a two-segment route executes a merged step over the same batch and produces token-identical output to the single-backend step; the handoff buffer is backend-allocated, and `Category=Parity` stays green.

#### Tests

- [_] Automated Pipeline Identity: independent pipeline advancement with token streams bit-identical merged and unmerged. Verify: `dotnet test --filter Category=Scheduling -c Release`.
- [_] Automated Route Admission: route resolution, merge predicate, three-way admission reporting, and concurrency cap behavior. Verify: `dotnet test --filter Category=Loader|Config|Scheduling|Concurrency -c Release`.
- [_] Automated Prefill Isolation: step-time budget bounds prefill steps and decode ITL p95 holds with a long prompt in flight. Verify: `dotnet test --filter Category=Prefill|Concurrency -c Release`.
- [_] Automated Lane Affinity: dedicated-lane isolation, quota validation, core-set collision handling, and per-segment reporting on the monitor stream. Verify: `dotnet test --filter Category=Concurrency|Affinity|Config|Metrics -c Release`.
- [_] Automated CPU/GPU Neutrality: GPU-segment routes rejected naming the backend, with no CPU-only assumption reachable from the scheduler. Verify: `dotnet test --filter Category=Scheduling|Budget -c Release`.
- [_] Manual Concurrency Proof (CPU half): serve one model on a CPU route with a concurrency cap and submit requests past it, confirming admitted requests advance at their configured quota and rejections name the blocking limit. Verify: live monitor readings per model and per segment.
- [_] Manual Long-Prompt Proof: serve two models concurrently, submit a 30K-token prompt on one, and confirm the other model's decode latency is unchanged on a dedicated lane and degraded-but-bounded on a shared lane. Verify: live monitor readings for both models plus a prefill tok/s record.
- [_] Manual Split-Route Proof: the same concurrency and long-prompt proof on a CPU+GPU-split route. *(deferred to MVP4 — the route cannot exist before the CUDA backend lands; recorded as a deferred gate at ship time, not closed here.)*

### MVP4 [v0.4.0] - Refined

CUDA joins the engine; hybrid CPU/GPU inference on the 4090, including the PCIe-frac seam where GPU compute runs over DRAM-resident weights.

#### Design Notes

- **DN-M4-1** — CUDA binding: nvcc-built microkernels via `[LibraryImport]` for fused dequant/GQA kernels; cuBLASLt for dense F16 GEMM paths.
  - Rationale: GPU kernels are CUDA regardless of host language; "managed-first" was always a CPU-kernel doctrine. Fused dequant-FMA needs full control.
  - Rejected: ILGPU (cannot realistically fuse dequant into FMA chains); managed CUDA abstraction layer (immature, adds a translation tax).
  - Contract: `lair.backends.cuda` build target (`nvcc` MSBuild) + `[LibraryImport]` surface.
- **DN-M4-2** — Contiguous layer split: first N transformer layers resident in VRAM, remainder on CPU; LM head defaults to CPU.
  - Rationale: exactly one PCIe activation crossing per token at the split boundary; activations are ≤32 KB. GPU LM head would spend scarce VRAM on the 248K-vocab head for marginal gain.
  - Rejected: interleaved layer split (crossing every layer); GPU LM head default.
  - Contract: `PlacementPolicy.gpu_layers`, engine executor split point. This is one route shape, not the only one — a segment may also compute on CUDA over DRAM-resident weights (see DN-M4-7).
- **DN-M4-3** — Device policy is fail-fast: `device: cuda` or `hybrid` requires the device; no silent CPU fallback.
  - Rationale: silent fallback hides misconfiguration and produces unmeasurable performance claims. An `auto` fallback mode is a backlog candidate, not a default.
  - Contract: host startup validation, consistent with DN-M1-9.
- **DN-M4-4** — KV device pools: separate VRAM and DRAM block pools; `split` residency = device-tagged blocks, attention executed as two passes (VRAM set, then DRAM set); default residency follows the decode device.
  - Rationale: attention reads all KV every token — placement follows the reader unless the user pays the measured split cost.
  - Contract: `lair.kv` device-backed pools.
- **DN-M4-5** — Prefill offload: `prefill: gpu` runs prompt processing on CUDA; produced KV lands per `kv_residency`; decode continues per `device`.
  - Contract: scheduler step composition per model policy.
- **DN-M4-6** — Pinned host staging buffers for activations crossing the split; transfers overlapped with compute where measurement justifies it.
  - Contract: weight arena / device arena staging.
- **DN-M4-7** — PCIe-frac execution seam: a route segment may compute on CUDA while its weights stay in DRAM, streamed through pinned double-buffered staging so transfer overlaps compute; VRAM holds activations and KV only. MVP4 proves the seam on a dense model resident entirely in RAM; expert-shaped frac lands with the gather path in MVP6.
  - Rationale: Strata's `pcie-frac` is the mode where GPU compute helps when VRAM rather than compute is scarce, and MoE makes it pay off because only the batch's union active-expert set streams rather than the whole weight set. The seam has to exist before MoE residency is designed, or MVP6 invents its own device story.
  - Rejected: treating VRAM offload as the only hybrid MoE path (spends the budget frac exists to avoid); unpipelined frac (PCIe latency serialized against compute — the archive's offload tax in a new costume); a frac-only hybrid mode (layer split stays the default hybrid shape until measured otherwise).
  - Contract: an `IWeightSource` slab provider feeding CUDA compute over host-resident blocks; bench records frac tok/s and bytes/token against CPU-resident and VRAM-resident variants of the same model before any default changes. The MVP4 dense-model measurement is a seam proof with an expected loss — PCIe bandwidth is lower than this box's DRAM bandwidth, so a bandwidth-bound dense decode should not beat the CPU route, and a loss there is not a failure. The win is only claimed for expert-shaped frac, where the bytes crossed are the batch's active slabs rather than the whole weight set (DN-M6-8).

#### Implementation Items

- [_] Binding spike — nvcc MSBuild target + cuBLASLt `[LibraryImport]` smoke
  - Verify: build produces the CUDA kernel binary; smoke GEMM matches CPU reference within FP tolerance.
- [_] CudaBackend — dense decode + prefill kernels behind `IBackend` (see DN-M4-1)
  - Verify: `dotnet test --filter Category=Parity` — same F16 model decodes token-identical on CPU, CUDA, and hybrid.
- [_] Prefill offload — GPU prefill, CPU/hybrid decode (see DN-M4-5)
  - Verify: bench records ≥2× prefill tok/s on the 4090 vs CPU for the reference model. *(bench deferred — see Open Items)*
- [_] Layer split — `gpu_layers` placement policy (see DN-M4-2)
  - Verify: split-point sweep (0..N) on Qwen2-1.5B; parity holds at every split point; bench records decode tok/s curve. *(parity gate active; bench deferred)*
- [_] Frac spike — a CUDA segment computing over DRAM-resident weights on a model fully resident in RAM (see DN-M4-7, DN-M313-6)
  - Verify: `dotnet test --filter Category=Parity|Scheduling` — the frac route decodes token-identical to the CPU-resident route with the model's weights never copied into VRAM. Exit criterion is identity plus a recorded measurement, not a speed claim: `bench/` logs frac vs CPU-resident vs VRAM-resident tok/s and bytes/token, and an expected loss on a bandwidth-bound dense model does not fail the item. *(bench gate active; the residency decision is MVP6's, see DN-M6-8)*
- [_] VRAM residency — models fitting 24 GB resident on device
  - Verify: full-VRAM Q4_K_M 1.5B run; over-budget load rejected with ledger breakdown in the error.
- [_] KV device pools — `dram` | `vram` | `split` residency policies (see DN-M4-4)
  - Verify: parity across all three residencies; bench records the split-vs-follow cost curve. *(parity gate active; bench deferred)*

#### Tests

- [_] Automated Parity-Hybrid: token match across cpu / cuda / hybrid configurations
- [_] Automated Budget: VRAM admission rejection with actionable error
- [_] Automated Frac Identity: a CUDA segment over DRAM-resident weights decodes token-identical to the CPU-resident route with no weight copy into VRAM. Verify: `dotnet test --filter Category=Parity|Scheduling -c Release`.
- [_] Automated Kernel: CUDA fused dequant GEMV vs scalar reference
- [_] Manual Hybrid Serve: monitor shows GPU utilization and PCIe transfer counters during hybrid decode *(GPU tiles arrive with MVP3's UI; counters require the CUDA phase)*

#### Open Items

- [_] **Windows CUDA dev toolchain incomplete** — registry lists CUDA Development/Toolkit 13.0 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`, but `nvcc.exe` is absent from the whole toolkit tree (only runtime DLLs + `cuda_runtime.h` are present; `cublasLt.h`/`nvrtc.h` missing). WSL Ubuntu-22.04 has nvcc 12.4 but cannot target a Windows .NET host. Resolve with a repair/reinstall of the CUDA 13.x developer components (or an explicit binding-strategy change via plan-update) before `/plan-build` starts.
- [_] **Bench verification deferred** — owner decision 2026-11-02: benchmarks are not run in the current work window; bench-only verify lines stay pending until requested. The 4090 currently runs other workloads at ~95% utilization.

### MVP5 [v0.5.0] - Refined

Many models at once, each with its own placement policy.

#### Design Notes

- **DN-M5-1** — Registry and hot swap: models declared in config with `type: chat|embed`; add loads in the background and serves when ready; remove drains in-flight requests, then frees pools.
  - Rationale: model changes ride the same hot-reload bus as every other knob; drain-then-free avoids mid-flight collapse.
  - Rejected: restart-to-change-model; immediate unload (kills in-flight requests).
  - Contract: registry in `lair.http` + `IConfigSource`.
- **DN-M5-2** — Global ledger admission: DRAM quota, VRAM pool, and core sets are accounted at load time; runtime KV exhaustion rejects new requests, never preempts running ones.
  - Rationale: preemption needs swap to disk — that machinery arrives in MVP7; until then the honest answer is refusal.
  - Rejected: best-effort oversubscription (thrash); preempt-to-fittest (needs MVP7 tiers).
  - Contract: ledger in scheduler; errors via `engine_overloaded` / `context_length_exceeded`.
- **DN-M5-3** — Embeddings: encoder-only forward, mean pooling default with CLS option and `normalize` flag; input batching reuses MVP2 batch steps.
  - Contract: `/v1/embeddings` OpenAI shape; registry `type: embed`.
- **DN-M5-4** — Scheduler fairness: core-set partitioning is the isolation mechanism; FIFO within a model, per-model priority; no default core oversubscription (overcommit is a backlog knob).
  - Contract: scheduler admission order.

#### Implementation Items

- [_] Model registry — concurrent instances, add/remove via config reload (see DN-M5-1)
  - Verify: `dotnet test --filter Category=Registry` — model added and removed via config edit while another model serves; no request failures outside the removed model.
- [_] Device budgets — admission control against VRAM + DRAM pools (see DN-M5-2)
  - Verify: ledger test — over-committed config rejected at startup with per-device breakdown in the error.
- [_] Scheduler fairness — time-slicing, preemption, priority (see DN-M5-4)
  - Verify: two-model load test — per-model ITL stays within budget while the other model saturates its core set.
- [_] Embeddings endpoint — `/v1/embeddings`, encoder-only models (ModernBERT-class) (see DN-M5-3)
  - Verify: `dotnet test --filter Category=Embeddings` — pooled vectors match reference fixtures within FP tolerance.
- [_] Flagship scenario — GPU-heavy coding model + CPU-only embedding model side by side
  - Verify: manual — both models serve concurrent load; monitor shows per-model device portioning.

#### Tests

- [_] Automated Registry: hot add/remove under load
- [_] Automated Budget: ledger rejection paths and quota accounting
- [_] Automated Fairness: cross-model ITL isolation
- [_] Automated Embeddings: pooling parity vs reference fixtures

### MVP6 [v0.6.0] - Refined

Qwen3.8-Flash-Next runs: sparse MoE with residency management.

#### Design Notes

- **DN-M6-1** — Reality-check gate: no Flash-Next kernel work until llama.cpp/ik_llama.cpp measurements (tok/s, bytes/token, expert gather behavior) are recorded on the target GGUF.
  - Rationale: the archive's explicit recommendation; prevents building a C# engine against imagined numbers.
  - Contract: `bench/` measurement report committed before kernel items start.
- **DN-M6-2** — Quant scope: IQ3_XXS primary (Strata's recommended consumer quant), Q4_K_M as the measured baseline; other formats added only as residency experiments.
  - Rationale: smaller quants shrink the expert hot set, but IQ3 dequant is shuffle-heavy on AVX2 — the gate decides whether the residency win survives the kernel cost.
  - Rejected: FP8 native containers (breaks the GGUF-first commitment); low-bit-only with no baseline (unmeasurable claims).
  - Contract: `IWeightSource` format descriptors for IQ3_XXS/Q4_K_M expert slabs. Dense IQ kernels land first in MVP3.8 and are reused by the gather path.
- **DN-M6-3** — Expert-aware batching: the residency cost function is the batch's union active-expert set, not per-sequence expert sets.
  - Rationale: shared decode steps (MVP2) make expert reads amortized across sequences — this is the engine-level MoE win.
  - Rejected: per-sequence expert scheduling (re-reads shared experts per sequence).
  - Contract: scheduler batch composition + `IGatherKernel`. The same union set is what a PCIe-frac segment streams, so it is also the device-axis cost: the union decides which tiers a step touches (see DN-M6-8).
- **DN-M6-4** — Contiguous expert slabs via `IWeightSource` providers: experts stored as contiguous slabs, gathered through `IGatherKernel`, prefetched on top-K router pointers before the GEMV.
  - Rationale: ik_llama.cpp proves prefetch; slabs make gather a pointer chase plus prefetch, not a scatter.
  - Rejected: expert-major interleaved layout (kills slab prefetch).
  - Contract: `IGatherKernel`, `IWeightSource` slab provider.
- **DN-M6-5** — RAM hot tier: capacity/cost-model residency decides which experts stay hot — value = router hit frequency × gather cost saved, budget = RAM quota minus KV working set.
  - Rationale: prefetch is taken by ik_llama.cpp; capacity/cost-driven residency is the open niche.
  - Rejected: static top-N hot set (misses workload drift); pure LRU (ignores gather cost asymmetry).
  - Contract: `IResidencyTier` RAM implementation + tier stats on the metrics bus.
- **DN-M6-6** — PLE: n-gram table as an `IResidencyTier` pageable table with indexed gather (row-offset addressing); sidecar index built at load only if the GGUF tensor layout doesn't align rows.
  - Contract: `IResidencyTier` PLE provider; load-time index builder.
- **DN-M6-7** — MTP: draft-then-verify with accept-rate accounting; draft tokens verified in one batched forward; accept rates reported per model on the metrics bus.
  - Rationale: archive's top lever (12→25 tok/s on copy/code); verification batch amortizes expert reads.
  - Rejected: self/n-gram drafting at this phase (deferred to backlog); free-running draft without verification (quality risk).
  - Contract: `lair.sampling` drafter + scheduler verify step.
- **DN-M6-8** — Frac expert residency: DRAM-resident expert slabs computed on CUDA through the PCIe-frac seam (DN-M4-7), streamed per the batch's union active-expert set with double-buffered prefetch so transfer overlaps compute. It is a residency option alongside the VRAM hot tier, chosen per model, not a replacement for it.
  - Rationale: the 4090 cannot hold Flash-Next's expert set and its KV at once, and the mode that spends GPU compute without spending VRAM on weights is the one Strata ships; gather makes it viable because only the slabs a step actually routes to cross PCIe.
  - Rejected: VRAM hot tier as the only hybrid MoE option (spends the budget frac exists to avoid); streaming whole expert sets per step (union-set streaming is the entire point); unpipelined transfer (PCIe latency serialized against compute).
  - Contract: `IResidencyTier` DRAM tier feeding the CUDA gather path through the route's segment; bench records frac vs VRAM-hot vs CPU-resident on the same fixture before either becomes a default.

#### Implementation Items

- [_] Reality check — llama.cpp/ik_llama.cpp harness on target Flash-Next GGUF (see DN-M6-1)
  - Verify: committed bench report with tok/s, bytes/token, and expert-gather observations on the dev box and, if reachable, the Xeon box.
- [_] Router + gather — `IGatherKernel`, contiguous expert slabs (see DN-M6-4)
  - Verify: `dotnet test --filter Category=Moe` — gathered expert output matches dense reference math; `bench/` reports gather tok/s vs dense-equivalent.
- [_] RAM hot tier — capacity/cost-model expert residency (see DN-M6-5)
  - Verify: bench — hit-rate and tok/s improve over prefetch-only baseline on a replayed workload trace.
- [_] Frac expert gather — DRAM-resident slabs computed on CUDA through the route's segment (see DN-M6-8, DN-M4-7)
  - Verify: `dotnet test --filter Category=Moe` — frac gather output matches the dense-equivalent reference; `bench/` records frac vs VRAM-hot vs CPU-resident tok/s and bytes/token on the same fixture. *(bench gate active — residency default changes only on measured evidence)*
- [_] PLE — pageable n-gram table, huge-page backed; indexed disk gather (row-offset addressing; sidecar index if the quant format doesn't align rows) (see DN-M6-6)
  - Verify: `dotnet test --filter Category=Ple` — indexed gather matches in-memory reference; `bench/` reports disk-gather latency distribution.
- [_] GDN/QSA — recurrent state budgets, sparse attention semantics
  - Verify: layer-level parity fixtures vs llama.cpp qwen4exp reference activations for GDN and QSA layers.
- [_] MTP drafting — speculative decode with accept-rate accounting (see DN-M6-7)
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

- [_] **qwen4exp GGUF layout** — exact tensor naming/shaping for router, experts, PLE, GDN state, and MTP head in target GGUF files; resolved by a loader spike at the start of MVP6 build, feeding DN-M6-4/5/6 contracts.
- [_] **Frac viability decision** — whether PCIe-frac expert residency (DN-M6-8) is worth building at all depends on Flash-Next's real expert geometry measured under DN-M6-1: if the hot expert set plus KV fits VRAM comfortably, the VRAM hot tier is the default and frac stays a seam; if it does not fit, frac is the mode that decides. MVP4's dense-model measurement (DN-M4-7) is a seam proof and does not settle this. Resolve before the MVP6 residency items are built.

### MVP7 [v0.7.0] - Tentative

Residency spans RAM, VRAM, and NVMe.

- NVMe cold tier — colibri-style expert streaming
- Conversation persistence — idle KV block runs snapshotted to disk, restored on resume
- VRAM hot tier — experts promoted onto the 4090
- Cross-tier prefetch — router-prediction-driven promotion
- Slab migration — promote/demote experts between RAM, VRAM, and NVMe while serving, refcounted and double-buffered
- Budget rebalancing — tier ratios adjustable at runtime

### MVP8 [v0.8.0] - Tentative

Vision input on chat models.

- Vision encoder — SigLIP/Qwen-VL-class tower loaded via `IModelLoader`
- Image pipeline — preprocessing, patch embedding, visual-token injection into chat
- Vision families — Qwen3.8-Flash-Next image input, Qwen-VL-class models
- Vision parity — golden tests against llama.cpp vision outputs

## Backlog

| Type | Item | Description |
|---|---|---|
| idea | Anthropic API compat | Strata exposes both surfaces; LAIR-native later |
| idea | Additional pre-tokenizer profiles | `none`/`nova2`/`llama-code`/decoupled profiles land only with a real fixture + `/tokenize` oracle (DN-M33-2) |
| idea | AVX-512 dispatch path | +10–30% decode, 2–4× prefill on capable CPUs |
| idea | Native AOT packaging | ~50 ms vs ~500 ms startup |
| idea | Early-exit argmax | skip vocab rows; ~5–8% on 248K-vocab LM head |
| idea | NUMA pinning | dual-Xeon production boxes |
| idea | Quant conversion tooling | only if the published-quant story breaks |
| idea | Vulkan/Metal backends | deferred; LAIR commits to CPU + CUDA |
| idea | Multimodal image input | Strata has it; LAIR later |
| idea | Embeddings model families | ModernBERT-class GGUF quants, pooling modes |
| question | License | MIT / Apache 2.0 / private — dotLLM is GPLv3: study ideas, never copy code |
| bug | 2DPC DIMM derate | production box derates 2400→2133 MT/s; more RAM ≠ speed |
| question | CUDA binding choice | resolved for planning: nvcc microkernels + cuBLASLt (DN-M4-1); revisit only if the toolchain open item forces a strategy change |
| question | PLE index format | does the GGUF PLE tensor allow random row access, or is a repack sidecar index needed? (MVP6 open item) |
| idea | Conversation cache policy | TTL/LRU eviction rules; optional encryption of disk snapshots; cache invalidation on model/config change |
| idea | Client-keyed cache retention | tie prompt/conversation cache items to client identity — e.g. a sticky id issued in a response header — and use it for sliding retention (keep each client's most recent conversation cache item) instead of or alongside LRU; precondition: a reliable-enough client identification method (issued token header, optional API key/session, connection fingerprint); interacts with server-posture backlog |
| idea | Byte-reduction experiments | MoD/layer skipping, expert pruning/distillation, dynamic quant switching — each as an `IWeightSource` provider |
| idea | Dynamic layer rebalance | move whole layers between CPU and VRAM while serving (Strata-style), driven by measured device utilization |
| question | Server posture | default bind address, optional API key, LAN exposure rules |
| question | CPU power/temperature source on Windows | RAPL MSRs and CPU temp need kernel access — options: small signed helper service, HWiNFO shared-memory bridge, or omit on Windows and use Linux hwmon/RAPL for the Xeon boxes |
| idea | RAM temperature via IPMI | Xeon production box may expose DIMM temps over BMC/IPMI; absent on the dev box |
| idea | MTP on dense models | pull-forward candidate — MTP heads exist on some dense GGUFs; would move drafting from MVP6 toward MVP2/4 |
| idea | KV evict-and-recompute | retain prompt tokens, evict cold KV blocks, re-prefill on demand — a per-model speed/memory knob; precedent: llama.cpp `--cache-reuse`; recompute is compute-bound on CPU, so never a default |
| idea | Preemptive KV swap | preempt a running sequence's KV to the NVMe tier under memory pressure — depends on MVP7 tiers |
| idea | Prefill GEMM shim microkernel | tiled-matmul native microkernel via `[LibraryImport]` where the bench harness proves the managed tile wall |
| idea | Core-set overcommit | allow models to share/partially oversubscribe core sets as an explicit knob |
| idea | Cross-model interleaved batches | multi-model serving where batch steps alternate across models at configured ratios (e.g. 2:1) so all models share one compute/thread pool; per-model throughput degrades proportionally to share — explicit toggle with per-model ratios, never the default (DN-M2-6 core-set partitioning stays default); MVP5 scheduling candidate |
| idea | Diffusion model support | UNet/DiT executor family (bidirectional attention, no KV cache), Conv2D/GroupNorm kernels, VAE encode/decode, noise-scheduler stack, CLIP/T5 conditioning; GGUF already hosts SD-class containers and the loader/backend/quant investment transfers; continuous batching fits denoise steps well; needs a Scope change (image generation is a current explicit non-goal) — proposed MVP9 candidate |
| idea | Server-side MCP injection | per-model, config-declared MCP servers whose tool definitions LAIR injects into chat requests regardless of client; injected tools execute server-side (contrast DN-M2-8: client-supplied tools stay client-side); enables cross-model composition — an LLM calling a diffusion model through its MCP endpoint, or the embeddings engine exposed as a vector/RAG MCP; interacts with tool-call API work and server posture |
| idea | `auto` device fallback | silent CPU fallback when a configured CUDA device is absent — deliberately not a default (DN-M4-3) |

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
- **2026-11-02 — Tool-call detection contract fixed (DN-M2-10).** MVP2 guarantees injection, a strict fenced-JSON detector, and `tool_calls` wire shape; actual call emission is model-dependent and not demanded of small fixture models — parser-level tests stand in for model compliance.
- **2026-11-02 — Hot-reload KV budget = soft cap within pinned arena (DN-M2-11).** Physical arenas are reserved at startup; `kv_blocks` hot-edits move the usage ceiling inside the reservation (grow immediate, shrink drains evictable cache); physical re-reservation deferred to the tiered-pools phase.
- **2026-11-02 — MVP3 inserted for the Web UI; CUDA and later phases shift down one.** New MVP3 [v0.3.0] (Tentative) ships the Monitor Web UI (design carried from the multi-model phase, formerly DN-M4-4 → DN-M3-1) plus a Chat test UI in a tabbed shell with a shared stylesheet (`mockup/mvp3/` mockups precede production code). CUDA becomes MVP4 [v0.4.0], multi-model MVP5 [v0.5.0], Flash-Next MVP6 [v0.6.0], residency MVP7 [v0.7.0], vision MVP8 [v0.8.0]. Rationale: the UI needs no GPU, exercises the shipped CPU engine, and the owner wants to test that engine hands-on now. Chat component stack (assistant-ui vs shadcn/ui vs hand-rolled — license, streaming ergonomics, build integration) is a refine-phase decision.
- **2026-11-02 — Windows CUDA toolchain found incomplete; MVP4 blocked until repaired.** Registry lists CUDA Development/Toolkit 13.0, but `nvcc.exe` is absent from the toolkit tree and `cublasLt.h`/`nvrtc.h` are missing (runtime DLLs + `cuda_runtime.h` present). WSL Ubuntu-22.04 has nvcc 12.4 but cannot target a Windows .NET host. Recorded as an MVP4 Open Item; repair/reinstall of the developer components (or a binding-strategy change via plan-update) required before `/plan-build` MVP4.
- **2026-11-02 — MVP3 refined: SSE transport, shadcn + hand-rolled chat, mockup rev B locked as spec.** Live stats reuse `/monitor/stream` SSE with `Last-Event-ID` replay from the snapshot ring (WebSocket/SignalR rejected — read-only monitor needs no bidirectionality); chat tab uses shadcn primitives with a hand-rolled OpenAI-compatible SSE client (assistant-ui rejected as runtime-library weight for a test client; both are MIT — the license difference assumed earlier does not exist); tabbed shell mockup (`mockup/mvp3/`, rev B) approved verbatim as the production spec with the dark theme locked; per-request tok/s series added to the event schema contract; `ui.recent_requests` config knob joins the live mutation class.
- **2026-11-02 — Benchmarks deferred by owner decision.** Perf/bench verify lines stay pending until requested; the 4090 currently runs other workloads at ~95% utilization. Correctness gates remain active.
- **2026-11-03 — Q4_K_M family coverage pulled forward as MVP3.1 [v0.3.1].** Owner hit the gap directly with `models/smollm-135m-instruct-q4_k_m.gguf` (attn k/v are Q5_0, output is Q6_K); consuming published quants is a Scope commitment and "unsupported ggml type" on a user's own model is unusable. Backlog row removed; kernels mirror the established llama.cpp `ggml-quants.c` pattern (DN-M31-1).
- **2026-11-03 — Repo layout conventions settled by owner.** SPA moves to ASP.NET convention: `src/Lair.Host/ClientApp/` with standard static-asset serving and root redirect (`/` -> `/ui`) replacing the root-level `web/`. The `Lair.Api` library + `Lair.Host` executable split stays (testable surface vs CLI composition root). Solution file and Directory.Build.props stay at repo root.
- **2026-11-03 — Project renames: `Lair.Api` → `Lair.Http`, `Lair.Core` → `Lair.Contracts`.** Owner-driven role naming: "Api" reads as a console app serving the API, but the project is the HTTP serving layer; "Core" held contracts only. `Lair.Http` names the serving adapter (endpoints, SSE streaming, request lifecycle, model router); `Lair.Contracts` names the contracts-only leaf (`IBackend`, `IWeightSource`, quant formats, tensor shapes). Mechanical rename of project dirs, csproj files, namespaces, and solution entries.
- **2026-11-03 — MVP5.1 [v0.5.1] inserted ahead of Flash-Next: dense IQ3_XXS coverage.** Owner's target quant for their own runs is IQ3_XXS; dense loader + kernels must land before MVP6 MoE work, whose expert slabs assume the same format (DN-M6-2). Tentative: the kernel strategy (fused dequant-GEMV vs llama.cpp dot-vec tables) is an Open Item for `/plan-refine`, not a mechanical K-quant mirror.
- **2026-11-03 — q4_k_m cross-engine decode gate: membership parity; no quant layout defect exists.** An initial llama.cpp CLI comparison on the real Q4_K_M fixture showed different greedy continuations, prompting a full audit: probes of b12032 `dequantize_row_q5_0/q4_K/q6_k` (independent transcription + real GGUF bytes) proved LAIR kernels byte-exact; float64 exact-arithmetic forwards showed q4-style weight noise flips a near tie — LAIR's argmax and the reference pick are mutual top-5 members with both sides' calibrated logprobs reproduced. Strict cross-engine token equality on low-bit formats is rejected as chasing llama.cpp approximation noise (consistent with the existing parity design); `tests/fixtures/parity-smollm-q4km.json` pins model sha256 + b12032 server reference mode.
- **2026-11-04 — Chat garbling root-caused to RoPE pair convention; parity gates widened beyond first-step distributions.** Every LAIR path carrying ChatML markers garbled from the first token on all quants while llama.cpp b12032 stayed coherent. The tokenizer was proven innocent (`GgufTokenizer.Encode` equals `/tokenize` id-for-id, including control-type specials mid-text), and KV paging/chunked prefill/prefix-cache were disproven by identical K/V bytes and identical greedy chains across pools differing only in block geometry. `LayerOps.ApplyRopeToHead` rotated half-split pairs `(i, i + headDim/2)` instead of the GGUF adjacent convention `(2i, 2i + 1)`: logits agreed with llama.cpp at absolute position 0 (≤0.02 nats) but diverged 1.4–2.8 nats from position ≥1, and greedy decoding amplified that into newline spam past ~17 tokens — so the defect looked length-dependent and tokenizer-adjacent while neither was causal. A from-scratch float64 forward of the same GGUF reproduced pre-fix LAIR exactly under half-pairs and llama.cpp exactly under adjacent pairs, confirming llama.cpp is not the outlier. Gates widened: Parity adds a cross-tool logit band (±0.05 nats F16, ±1.0 quantized) against pinned b12032 values, and `ContinuationParityTests` pins exact greedy continuations for every F16 fixture prompt; the 2026-11-01 membership criterion still governs the quantized files, where near-tie flips remain documented. Fixture `lair_logprob` calibration re-recorded from post-fix code with all llama.cpp-derived reference values left byte-identical.
- **2026-11-04 — Llama-family chat template support pulled forward as MVP3.2 [v0.3.2].** Owner hit the gap while looking for a model capable of holding a short conversation: `models/Llama-3.2-1B-Instruct-Q8_0.gguf` loads under pinned b12032 but LAIR refuses to serve it, failing at startup in `ChatTemplateRenderer` on its first `{% set %}` (offset 37); the template also needs `is defined`, `none` tests, slices, extra filters, comments, membership/boolean operators. Engine-side compatibility is already satisfied (`llama` arch, `pre = llama-bpe`, full-head RoPE, Q8_0 kernel), so Scope's dense Llama-class ≤3B commitment is unmet purely at the template layer — recorded as out-of-subset in MVP1's chat-template item and now promoted to its own Tentative phase pending `/plan-refine mvp3.2`.
- **2026-11-04 — GGUF compatibility phase family added as MVP3.3–MVP3.7.** Owner agreed to implement MVP3.2 as a managed `Lair.Templating.Minja` library and defer tokenizer profiles, architecture adapters, tool-call parser registry, quant descriptor/kernel registry, and RoPE strategy adapters to subsequent Tentative MVP3.x phases. Variant implementations are resolved at model load/plan time, not during runtime, and do not get separate assemblies unless packaging requires it.
- **2026-11-04 — MVP3.2 refined as full minja parity.** MVP3.2 now targets a managed `Lair.Templating.Minja` library with full minja parity for minja’s supported subset, custom `MinjaValue` semantics, minja-compatible scoping, chat-template polyfills/capability detection, hard template resource limits, and byte-exact validation against pinned llama.cpp `/apply-template`. The existing `ChatTemplateRenderer` is replaced rather than kept as a fallback.
- **2026-11-05 — Minja expression literals added to MVP3.2 scope.** Filters work showed inline list/map literal expressions are required for llama-family template parity (`['user', 'assistant']`, tool-call metadata), so the parser/evaluator must support JSON-like lists and ordered maps while parenthesized tuple literals remain unsupported; comprehension forms stay deferred until oracle evidence requires them.
- **2026-11-06 — Minja parity fixes landed from byte-exact Llama-3.2 goldens.** Capture against pinned b12032 `/apply-template` exposed three engine divergences, all fixed to match the oracle: lowercase `true`/`false` parse as boolean literals (the Llama-3.2 `tools_in_user_message` branch depends on them); the parser strips exactly one trailing newline from the template source (Jinja `keep_trailing_newline=False`) because GGUF templates end with `\n` and the reference does not render it; macro call arguments and defaults evaluate in the **calling** scope, verified against b12032 `common/jinja/runtime.cpp` (`call_expression::execute_impl` gathers arguments from the caller's context), which unblocks recursive helpers like `countdown(n - 1)`. b12032 additionally parents macro bodies to the caller (dynamic scoping); not adopted — no golden or shipped test distinguishes it, and LAIR keeps its pinned closure design.
- **2026-11-06 — Reference refusal divergence documented, not adopted.** b12032 `/apply-template` answers 400 for conversations ending in an assistant tool-call message; LAIR renders them. `tests/fixtures/parity-templates-llama32.json` records the refusal as `reference_refusal` and `TemplateParityTests` asserts both sides, so the divergence is visible instead of silently matched.
- **2026-11-06 — Legacy `ChatTemplateRenderer` retired with its rejection goldens.** The adapter replaces the mini-Jinja renderer (DN-M32-6) and the class is deleted; four `Category=Template` cases that asserted rejections of `{% set %}`, `|upper`, macros, and slices were retired because the minja parser accepts those constructs now — `Category=Template` goes 20→16 with no behavior regression for shipped ChatML/Phi goldens.
- **2026-11-06 — Tool precedence wired with a model-scoped override knob.** `TemplateCapabilities.SupportsTools` now decides the tool surface per DN-M32-7: a tool-capable template renders declared tools itself and LAIR's fenced block is skipped; a toolless template keeps LAIR's injection byte-identically. `models[].force_fenced_tools` (bool, default false, `NewRequestsOnly` hot-reload class) is the explicit config override DN-M32-7 anticipated. Fenced-JSON detection remains the only detector; native (non-fenced) tool-call parsing stays owned by the tool-call parser registry phase.
- **2026-11-06 — MVP3.3 refined: tokenizer profile taxonomy settled.** Profiles are keyed by `tokenizer.ggml.pre` and MVP3.3 formalizes only the three behaviors already implemented (`llama-bpe`, `gpt2`, `smollm`) — the two real fixtures on this machine declare exactly those. A profile is a load-time factory (split strategy resolved once, no hot-path switch, public `GgufTokenizer` surface unchanged), unknown `pre`/`model` stays an actionable load rejection naming the value and the supported list, metadata coverage follows llama.cpp's actual keys (`byte_fallback`, `byte_to_token`, `byte_to_utf8`, `add_space_prefix`, `eot_token_id`, `utf8_flush_between_controls`) with profile-scoped BOS/EOS defaults, and a committed `/tokenize` parity harness replaces the ad-hoc 2026-11-04 proof. Speculative profiles move to Backlog until fixture + oracle evidence exists.
- **2026-11-06 — `add_space_prefix` is read, not stripped — oracle overrides the inferred behavior.** SmolLM-135M-Instruct-F16 declares `tokenizer.ggml.add_space_prefix=false`, and the in-flight metadata implementation initially stripped the folded leading space from every pre-token; that dropped the final pre-token's tail (`"Hello, world!"` encoded without `"!"`) and broke round-trip. A direct b12032 `/tokenize` capture on the same fixture settles it: `" world!"` → `[905, 17]` and `"Hello, world!"` → `[19556, 28, 905, 17]` — the reference keeps the space. LAIR reads and exposes the key but does not alter the pre-split for the supported profiles. After the strip is removed, LAIR encodes `"Hello, world!"` as `[19556, 28, 905, 17]` — id-for-id with the reference — so the `smollm` merge path already reaches the reference's `905` (`" world"`) piece and the divergence was purely the strip.
- **2026-11-06 — Pre-splits classify by Unicode code point, settled by the new `/tokenize` goldens.** The committed oracle harness (`Category=Tokenizer`) failed on astral text: UTF-16 `char` classification treated an emoji's surrogate pair as undefined, so a leading space was not folded into the run (`"a 😀"` → `81,216,10813,242,218` against the reference's `81,40303,218`), and the llama-bpe symbol fold additionally required the space itself to be a symbol, which made space+symbol folding unreachable for `llama-bpe`. Both splits now classify and advance by code point (`CharUnicodeInfo.GetUnicodeCategory(string, int)` plus a surrogate-pair length check — deliberately not grapheme clusters, which could straddle an offset boundary and desynchronize the UTF-16 slicing). Merge logic, byte maps, and decode were untouched: the defect was pre-split only.
- **2026-11-07 — MVP3.4 refined: hybrid family-plus-metadata resolution, plan-not-switch.** Architecture compatibility resolves at load through a profile registry that produces an immutable plan (tensor map with aliases, norm/activation/bias/geometry/RoPE strategies, caps, sparse-FFN descriptor) — the shipped tokenizer-profile pattern, not llama.cpp's `switch (arch)`. An unknown `general.architecture` is accepted only on an exact llama-shaped tensor-table match; anything else fails naming the tensors. Metadata becomes the contract: `attention.use_bias`, `head_length`, activation and cap keys are honored or rejected at load, replacing today's tensor-presence bias rule and the blanket `rope.scaling.*` rejection. Family order is fixture-driven under the 4 GB budget: `llama`/`qwen2` re-expressed, then `qwen3` (QK-norm, head-length), then `gemma2` (GELU-tanh, two softcaps); Phi waits for MVP3.7's RoPE scaling and Gemma-3/Mistral wait for MVP3.8's windowed attention.
- **2026-11-07 — MoE execution moves to a new MVP3.8; MVP6 keeps residency.** MVP3.4 makes sparse FFN representable in the plan and rejects MoE at load with an actionable deferred message. MVP3.8 executes dense-resident MoE (router, top-k with renormalize, shared expert, per-expert FFN) and sliding-window/local attention — the two architecture shapes that need new execution patterns rather than new scalar ops. MVP6 keeps Flash-Next residency, prefetch, PLE, MTP, and GDN/QSA, so `IGatherKernel` arrives there as an optimization of an already-proven sparse path. Sequencing 3.4 → 3.5 → 3.6 → 3.7 → 3.8 keeps MoE dependent on the plan (3.4), the quant formats MoE GGUFs ship in (3.6), and the rope variants those models declare (3.7).
- **2026-11-07 — MVP3.4 family fixtures settled.** Qwen3-0.6B (F16 preferred, Q4_K_M fallback) and Gemma-2-2B-it Q4_K_M: both fit the 4 GB model budget (Gemma-2-2B F16 at ~4.4 GB does not), and the pinned b12032 `llama.dll` carries the `qwen3`, `gemma2`, `gemma3`, `phi3`, and `mistral` architecture names — verified by scanning the shipped binaries — so the oracle can serve every family MVP3.4 through MVP3.8 targets. Gemma-2 stays at Q4_K_M, which puts it under the quantized-parity tolerance rule rather than the F16 band.
- **2026-11-07 — MVP3.5 and MVP3.6 swapped: quant formats before output parsers.** The quant descriptor/kernel registry now runs as MVP3.5 and the tool-call parser registry as MVP3.6. Quant formats are the binding constraint on the families MVP3.4 adds (published Qwen3 and Gemma-2 files are consumed at Q4_K_M, and MVP3.8's MoE models ship in IQ-class formats), while output-format parsing is a serving-layer concern that no other MVP3 phase depends on. Design content is unchanged; only the phase numbers and DN ids moved (`DN-M35-1` is now the quant seam, `DN-M36-1` the parser seam).
- **2026-11-07 — Sliding-window score masking moves into MVP3.4; the KV-layout half stays in MVP3.8.** The pinned Gemma-2 fixture declares `gemma2.attention.sliding_window = 4096` and the pinned b12032 oracle applies that mask, so honoring the metadata contract (DN-M34-4) means LAIR must mask scores before Gemma-2 can be claimed as supported. The mask is the part that carries parity — the reference itself keeps full KV — so MVP3.4 resolves the window into the plan and masks out-of-range positions, while bounded KV reads, per-layer windowed/full interleave, and cache-reuse accounting remain MVP3.8 (DN-M34-7, DN-M38-2). Rejected instead: rejecting Gemma-2 for declaring a window, or loading it and ignoring the declaration.
- **2026-11-07 — MVP3.9 added: per-model reasoning and response token budgets.** No phase covered token budgets and nothing is implemented beyond a single per-request `max_tokens` (default 128, clamped to the context). MVP3.9 makes `max_reasoning_tokens` and `max_response_tokens` per-model settings with per-request overrides, enforced by the scheduler and the decode loop. Its reasoning ceiling depends on a resolved reasoning region, so it is sequenced after the MVP3.6 parser registry.
- **2026-11-07 — MVP3.4 family fixtures pinned as downloaded.** `models/Qwen3-0.6B-Q8_0.gguf` (609,800 KB, sha256 `9465e63a…3bb031`) from `Qwen/Qwen3-0.6B-GGUF`, and `models/gemma-2-2b-it.Q4_K_M.gguf` (1,629,000 KB, sha256 `ff7c3580…8787058`) from `mradermacher/gemma-2-2b-it-GGUF`. Qwen3 lands at Q8_0 rather than F16 because the upstream GGUF repo publishes only Q8_0 for 0.6B, and the Gemma-2 mirror is ungated where the `google/`, `unsloth/`, and `ggml-org/` Gemma-2 GGUF repos require license acceptance. Both load under the pinned b12032 loader and report the expected architecture and geometry.
- **2026-11-07 — Family profiles carry feature defaults; declared keys override them.** Loading the pinned Qwen3 fixture showed the two places where a declared-only rule diverges from the oracle: the file ships `blk.N.attn_q_norm.weight`/`attn_k_norm.weight` and the reference applies them without declaring `qwen3.attention.use_qk_norm`, and it declares no `qwen3.vocab_size` at all while the reference derives 151,936 from the token array. So QK-norm becomes a profile default for `qwen3`/`gemma2` that a declared key can flip either way, and vocabulary resolution falls back to `tokenizer.ggml.tokens` length with a logged fallback. Tensor presence alone still does not decide anything: a profile default with missing tensors is an actionable failure.
- **2026-11-08 — RoPE pairing convention becomes a per-family plan value; qwen3 uses rotate-half.** Proving Qwen3 parity against the pinned oracle exposed it: LAIR paired rotary dimensions adjacently `(2i, 2i+1)`, which is correct for the llama GGUF layout (its converter pre-permutes q/k), but the Qwen3 file stores them in rotate-half layout `(i, i + head_dim/2)`. The mismatch was not noise — logprobs drifted 1.1–2.3 nats and `def fibonacci(n):` picked `\n` where the reference picks `'   '`. With pairing split into `StandardFullHead` (llama/qwen2/generic) and `RotateHalfFullHead` (qwen3), LAIR reproduces the reference leaders to about 0.001 nats, which is what makes the layout claim evidence rather than inference. Two consequences: MVP3.7's RoPE work must treat pairing as a strategy axis alongside scaling, and each new family has to prove its pairing against the oracle before it can be claimed — Gemma-2's pairing is unproven until its parity fixture runs.
- **2026-11-08 — `qwen2` becomes a tokenizer pre-tokenizer profile.** The pinned Qwen3 file declares `tokenizer.ggml.model=gpt2` with `tokenizer.ggml.pre=qwen2`, and the shipped registry rejected that key. llama.cpp resolves `pre=qwen2` to the same byte-level GPT-2 split-and-merge pipeline, so LAIR adds a `qwen2` profile mirroring `gpt2` (same contractions, no merge shortcut, whole digit runs, BOS 1 / EOS 2 defaults) rather than defaulting with a log line. The unsupported-profile test moved to a genuinely unknown key (`nova2`) so the registry-sourced rejection message stays asserted.
- **2026-11-08 — The reference masks windowed attention backward-only; LAIR's symmetric encoder range is now an open item.** Measured with the pinned b12032 server on the Gemma-2 fixture (`--embeddings`, per-position embeddings, sequence inside the 4096 window): position 0 has exactly zero forward visibility — its embedding is bit-identical whether later tokens are present or not — while positions 150 and 300 change by tens of units. So a symmetric `[i-w+1, i+w]` rule cannot be what the reference does for this file. LAIR's causal paths now mask backward-only against scalar references; its `Encode` path keeps a symmetric range because nothing in MVP3 yet compares encoder output against the oracle for a windowed model. That gap is recorded as an MVP3.8 open item rather than assumed settled, and it is exactly the kind of convention that has to be proven before embedding parity claims exist.
- **2026-11-08 — QK-norm resolution: a declared true demands weighted tensors unless the family is known-unweighted.** The first wording ("absent or true keeps the profile's mode") would have made `llama`/`qwen2`/generic files that declare `attention.use_qk_norm=true` resolve to off, silently discarding a declaration the reference honors. Shipped rule: absent → profile mode; declared false → off; declared true → weighted-from-tensors unless the profile already resolves unweighted. Two related choices: tensor-side enforcement stays in the engine's open path (where QK-norm tensor requirements already live) rather than in plan resolution, which never inspects the tensor table for a registered family; and an unweighted family whose file still carries per-head vectors logs `qk_norm_vectors_unused` instead of failing, matching the existing ignored-tensor precedent.
- **2026-11-08 — Gemma-2's forward shape is what the reference does, not what the architecture write-up says.** The family was designed as LayerNorm block norms with GELU-tanh gating, dense attention biases, and an unweighted per-head QK norm. Measured against pinned b12032 on the pinned fixture, it is RMSNorm at every norm including the output norm, exact GELU in the gate, no per-head QK normalization at all, an input-embedding gain of `sqrt(hidden)` that the conversion leaves out of the weights, and q/k/v bias vectors its forward does not consume. LAIR ships the measured shape, with each choice pinned by a counterfactual reference that demonstrably differs and by 0.18–0.53 nat deltas against the reference's own top-5. The lesson is procedural as much as architectural: family profiles are hypotheses, and the oracle is what settles them.
- **2026-11-08 — SentencePiece tokenization is a real compatibility gap; MVP3.10 added for it.** The pinned Gemma-2 file declares `tokenizer.ggml.model=llama`, a SentencePiece vocabulary, and LAIR supports only GGUF BPE profiles. The consequence is concrete: the file loads into the engine and its forward proves out against reference-supplied token ids, but `scripts/generate-parity.ps1` cannot calibrate it and it cannot be served, so MVP3.4 ships Gemma-2 as forward-graph support with the parity fixture and live serve deferred. MVP3.10 adds SPM as a tokenizer profile rather than a parallel tokenizer, which also unblocks Gemma-3 and older Mistral/Phi files.
- **2026-11-08 — The reference windows Gemma-2 on alternating layers, not on every layer.** Its Gemma-2 graph applies the 4096 window to even layer indices and leaves odd layers full; LAIR currently applies a declared window to every layer, which masks attention in layers the reference leaves unmasked once a sequence exceeds the window. It is invisible at fixture context sizes and wrong beyond them, so it is recorded as an MVP3.4 item for the score-mask half and stays part of MVP3.8 for the KV-read half.
- **2026-11-08 — Model test budget moves from 4 GB to 512 MB–1 GB.** Owner revised the ceiling for model-loaded test work. Consequences are concrete rather than cosmetic: `models/Llama-3.2-1B-Instruct-Q8_0.gguf` (1,260 MB) and `models/gemma-2-2b-it.Q4_K_M.gguf` (1,629 MB) are now out of band, `models/Qwen3-0.6B-Q8_0.gguf` (610 MB) is inside it, and the SmolLM fixtures (101–258 MB) stay valid as small-fixture cases. The Llama-3.2 fixture has to be re-pinned at a sub-1 GB quant with its template and tokenizer goldens regenerated, and its live-serve working set re-measured against the new ceiling. Gemma-2 has no published quant under 1 GB (smallest is Q2_K at 1,173 MB, then Q3_K_S 1,298 MB), so the family stays engine-only — which is already its status, blocked on the SentencePiece tokenizer. MoE still has no published file under any ceiling, so MVP3.8's MoE gate stays synthetic-until-exception.
- **2026-11-08 — Llama-3.2 fixture re-pinned to Q4_K_M; the ceiling is a working-set ceiling, not only a file-size ceiling.** The Q8_0 file was deleted and everything that referenced it (serve fixture, tokenizer probe, both llama goldens, the template-generator default, `DEVELOP.md`) was repointed at the 770 MB Q4_K_M file. Regenerating both goldens from the new file changed only `model_file`, `model_sha256`, `generated_utc` and the fixture-pinned `date_string` — every tokenizer id and every rendered prompt body is byte-identical, which confirms the quant cannot affect those gates. The finding worth keeping: the file fits the band but the live serve peaks at 1,135–1,174 MB, so a fixture that satisfies the budget has to be chosen against measured working set, not file size. Every published quant small enough to fit at runtime (`Q2_K` 554 MB, `Q3_K_S` 612 MB) is a format the engine cannot yet load, so the fix is sequenced behind MVP3.5's quant registry rather than solved by shrinking context or re-tuning a gate. Gemma-2's smallest published quant is Q2_K at 1,173 MB, so the family stays engine-only regardless of the tokenizer gap.
- **2026-11-08 — The model budget is a serving-profile ceiling, not only a file-size ceiling.** Live serve measurements settle it: Qwen3-0.6B Q8_0 (610 MB weights) peaks at 1,726 MB on the default `4096`/`256`/`8` profile and at 1,000 MB on `512`/`32`/`4`, while Llama-3.2 Q4_K_M (770 MB weights) measures 1,102 MB even with the tight profile. Each engine carries roughly 360–395 MB of runtime overhead beyond weights and KV, so a fixture under the 1 GB ceiling needs about 640 MB of weights or less. Two consequences: the family-serve gate now records its compliant profile rather than a bare "served", and the llama fixture's compliance is genuinely blocked on MVP3.5 — the smaller published quants that fit are formats the engine cannot load yet. Rejected instead: shrinking the ceiling per-fixture, or counting a 770 MB file as budget-compliant because the file is small.
- **2026-11-08 — MVP3.4 ships with two gates deferred, not closed.** The phase is Shipped on the strength of 924/924 and a live Qwen3 family serve, but it does not carry everything it originally promised. Gemma-2's real-model parity fixture and live serve are deferred to MVP3.10 because the file's SentencePiece vocabulary cannot be calibrated or tokenized — the family ships engine-only, proven against reference-supplied token ids. The Llama-3.2 serve fixture stays over the 1 GB ceiling until MVP3.5 adds the quant formats small enough to fit. Both are recorded as open items rather than quietly dropped, and the Gemma-2 parity test item moved out of MVP3.4's Tests into MVP3.11, which already owned it.
- **2026-11-09 — Dense IQ coverage moves from the MVP5 series into MVP3.8; MoE, token budgets, and SPM shift one slot.** The phase that was MVP5.1 becomes new MVP3.8 covering `IQ4_NL`, `IQ4_XS`, and `IQ3_XXS`; dense-resident MoE plus windowed KV becomes MVP3.9, per-model token budgets become MVP3.10, and the SentencePiece tokenizer profile becomes MVP3.11. Two things drive it: MVP6's gather kernel (DN-M6-2) reuses the dense IQ kernels, so a format phase cannot sit after the execution phase that depends on it; and both IQ tiers have published fixtures loadable on this box, so they can be gated in MVP3 discipline rather than deferred. The format set stops at 3-bit: `IQ2_*`, `IQ1_*`, and TQ formats stay out because their packing is the hardest in the family, their quality is the worst, and no fixture in this plan demands them. Each format carries its own kernel gate plus a measured GEMV cost against Q4_K_M before it can be recommended as a default — the Backlog's "measure before committing" caveat became an item rather than disappearing.
- **2026-11-09 — Model budget returns to 4 GB, measured as peak working set, and it changes what each pending phase can prove.** Owner restored the ceiling after the 512 MB–1 GB revision made three gates unprovable. The number is a working-set limit, not a file-size limit, because a loaded engine costs about 360–395 MB beyond weights and KV. Consequences are concrete: MVP3.4's blocked serve fixture closes with no re-pin (1,102–1,174 MB measured); MVP3.7 gets a real YaRN fixture (`Phi-3-mini-4k-instruct` `Q2_K`, 1,351 MB → ~1.75 GB); MVP3.9 gets a real MoE fixture (`OLMoE-1B-7B-0924-Instruct` `Q2_K`, 2,444 MB → ~2.8 GB) where the smaller `IQ2_M` file would have needed an out-of-scope format; MVP3.11 unblocks Gemma-2 parity and serve on the already-pinned file (1,629 MB → ~2.0 GB) and adds `gemma-3-1b` `Q3_K_S` (657 MB) as the windowed serving fixture. MVP3.5's `Q2_K`/`Q3_K` formats are load-bearing for that MoE and Gemma fixture selection rather than nice-to-have.
- **2026-11-09 — MVP3.6 through MVP3.11 refined on measured code and fixture facts, not on the tentative sketches.** Five decisions carry the phases. Parser selection keys on chat-template capability plus a `tool_parser` config override, never architecture name, because fine-tunes change tool markup without changing `general.architecture` and no pinned fixture carries a tool-format metadata key. Reasoning-region recognition moves into MVP3.6 because nothing in `src` recognizes reasoning markup today and MVP3.10's reasoning ceiling is unimplementable without a resolved region. RoPE scaling stops being a blanket rejection: `rope.scaling.*` and `rope.dimension_count` become plan values with unimplemented variant names still failing. IQ kernels are decided per format family — fused GEMV for the 4-bit pair, table-driven staging for `IQ3_XXS` — with a bench gate deciding whether the staging survives. SPM splits on a new axis: `tokenizer.ggml.model` selects a vocabulary implementation, while `tokenizer.ggml.pre` profiles stay as MVP3.3 built them, because SPM has no merge stage to key a pre-profile against.
- **2026-11-08 — MVP3.5 refined: the quant registry is a descriptor registry, and its first two formats are chosen by the budget.** `GgmlType`'s switch becomes a registry resolving type ids to descriptors (`TypeId`, `Name`, `Format`, `BlockSize`, `TypeSize`) with geometry failures at load, and `Q3_K` plus `Q2_K` are the only formats entering the phase. The choice is not a completeness argument: they are the published formats that close MVP3.4's blocked fixture, and the only ones small enough to fit a 1 GB working set. `Q2_K`/`Q3_K` are also the first super-block formats with asymmetric min-based sub-block scales, so their definition is the pinned reference's `dqK` layout and a LAIR-authored scalar golden cannot gate them. IQ, TQ, `Q4_0`, `Q5_0`/`Q5_1`, `Q8_1`, and `Q8_K` stay out until a published fixture demands one. The Quant priority list open item is resolved by this decision.
- **2026-11-09 — MVP3.12 adds batched prefill under a step-time budget, with an opt-in dedicated prefill lane.** Prefill is the only hot path not using the batch machinery the engine already ships: `Prefill` forwards one token at a time while `GemvBatch` already traverses each weight once for 16 rows and the encode path already does causal-within-chunk attention over a growing KV. Batched prefill makes a chunk shorter in total but longer per step, so isolation is a measured step-time budget (`engine.prefill_step_budget_ms`) rather than a fixed chunk width, and a 30K-token prompt on shared cores still needs spatial isolation — hence `prefill_lane: shared | dedicated`, costing activation buffers rather than duplicated weights. Rejected instead: GEMM-first prefill (the register-pressure wall is unmeasured, so a GEMM spike is a bench-gated follow-up with a 30%-headroom trigger), widening the batch cap past 16, and splitting one chunk across a step boundary (it breaks the "identical to running alone" guarantee).
- **2026-11-09 — Requests become independent pipelines advanced by a route-aware arbiter; new MVP3.13 sits before MVP4.** The owner's model — each request owning its own transform/prefill/decode loop up to a per-model concurrency limit — is adopted as the control plane, while merged batch steps stay the execution plane's optimization, because on CPU a per-request weight traversal is exactly the rejected bandwidth-thrash shape: a single GEMV already saturates DRAM bandwidth, so N pipelines means N× weight traffic. Splitting the two layers is what makes the model device-neutral: prefill and GPU-resident decode get per-pipeline lanes for free, CPU decode keeps merging. The merge key becomes route identity rather than device identity, so a hybrid model's sequences still merge and pay the boundary tax once per step instead of once per sequence. Rejected instead: one OS thread per request, bolting a lane onto the existing single global step loop, and merging across models (no shared weights, and it breaks per-model quotas).
- **2026-11-09 — Compute device and weight tier become independent axes, and Strata's `pcie-frac` enters Scope.** `device: cpu | cuda | hybrid` conflates where the math runs with where the bytes live, which forecloses the mode where GPU compute helps while VRAM is the scarce resource: expert slabs resident in DRAM computed on CUDA, streamed per the batch's union active-expert set with double-buffered prefetch. MVP4 carries the seam (DN-M4-7, proved on a dense model fully resident in RAM) and MVP6 carries the expert-shaped version (DN-M6-8) beside the VRAM hot tier rather than replacing it. Rejected instead: VRAM offload as the only hybrid MoE path, frac as a default anywhere (PCIe and DRAM bandwidth are comparable, so the win is situational and must be measured), and unpipelined frac (PCIe latency serialized against compute — the archive's offload tax in a new costume).
- **2026-11-09 — A route resolves to a backend set, and a merged batch crosses segments whole.** `TransformerModel` and `BlockPool` accept exactly one `IBackend` today, so a hybrid or frac step needs the engine holding one backend per segment before MVP4 can name a split point; model-level concurrency already has the multi-backend shape because each model instance binds its own. The batch is not divided across backends — each segment runs the whole merged batch for its own layers, which is what makes the handoff `batch width × hidden` once instead of per sequence. Rejected instead: hybrid emulated inside `CpuBackend` (device branching in the CPU kernel path), splitting one batch's rows across backends for the same segment (destroys the weight reuse batching exists to create), and a per-sequence backend choice (routing is per token).
- **2026-11-09 — MVP3.12 is engine-only, and the step-time budget and prefill lane move into MVP3.13.** Both moved items edit `GenerationScheduler`, which MVP3.13 replaces outright: a lane is a property of a pipeline's progress and a token budget bounds a step the arbiter composes, so neither is expressible before pipelines exist. Building them in MVP3.12 would rewrite the scheduler twice for no additional gate. MVP3.12 therefore ships batched prefill, its bench record, and the GEMM decision trigger only.
- **2026-11-09 — Admission becomes a single three-way check owned by the arbiter.** A request must fit KV block capacity, the resolved token budgets (MVP3.10), and the model's concurrency slot with its per-segment quota, and a rejection names which of the three blocked. Splitting these across the HTTP layer, the scheduler, and the ledger produced three rejections with three different shapes that no user could act on; MVP3.10's reservation math and MVP5's device ledger become inputs to one admission point rather than competing gates.
- **2026-11-09 — MVP4's frac work is a spike with an identity exit criterion, and the residency decision is MVP6's.** PCIe bandwidth is lower than this box's DRAM bandwidth, so a bandwidth-bound dense decode is not expected to beat the CPU route; the MVP4 item proves the route shape and records the measurement, and an expected loss does not fail it. Whether frac is built at all is settled by Flash-Next's measured expert geometry under DN-M6-1, recorded as an MVP6 open item rather than assumed by this plan.
