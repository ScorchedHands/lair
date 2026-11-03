# Changelog

All notable changes to LAIR are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-11-03

### Added

- Q4_K_M family quant coverage: loader acceptance + scalar/AVX2 fused-dequant GEMV kernels for Q5_0, Q5_K, and Q6_K (llama.cpp `ggml-quants.c` parity), so published mixed-family files — attention at Q5_0, upgraded FFN downs at Q6_K — load and serve as-is. Real-fixture decode sanity tests plus a golden parity fixture pinned to llama.cpp b12032 on a real Q4_K_M model (membership gate: mutual top-5 calibrated logprobs; strict cross-engine token equality deliberately rejected).
- `GET /` redirects permanently (`301`) to `/ui/`.

### Changed

- The Web UI client project moved from root-level `web/` to `src/Lair.Host/ClientApp/` (ASP.NET Core SPA convention); build embedding, docs, and example config updated accordingly.
- Project renames: `Lair.Api` → `Lair.Http` (serving layer) and `Lair.Core` → `Lair.Contracts` (contracts-only leaf), namespaces included; no behavior change.

## [0.3.0] - 2026-11-02

### Added

- Web UI at `/ui` (served by the API host): a tabbed dark-themed operator surface built from the approved mockup — Monitor tab with live KPI cards, configured-models table (rows expand to show config and state: context/block size, KV budget, core set/quota, workers, residency, cache blocks, finish-reason tallies), in-flight request rows with per-request throughput sparklines, recent-request ring (`ui.recent_requests`, hot-reloadable), GPU/PCIe/power tiles reserved for the CUDA phase; Chat tab is a built-in test client: LLM-only model selector, streamed multi-turn replies, mid-stream Stop, tool-call cards.
- Versioned monitor event schema v1 on `/monitor/stream`: incrementing frame ids, `snapshot` / `request_start|update|complete` lifecycle events with per-request token-sample series (drives the sparklines), `resync` state dumps; `Last-Event-ID` reconnect replay from the snapshot buffer; latest-only coalescing and bounded subscriber queues with drop counters.
- `/v1/models` entries now carry a model `type` field.

### Changed

- Frontend toolchain under version control: pinned esbuild/Tailwind v4/React build in `web/`, embedded into the Host build automatically (`-p:SkipWebUi=true` opts out).

## [0.2.0] - 2026-11-02

### Added

- Continuous batching: shared decode steps across up to 16 sequences with decode-priority scheduling; concurrent requests run side by side (503 `engine_overloaded` only on capacity exhaustion). Verified output-identical to sequential execution for greedy and seeded runs.
- Chunked prefill: long prompts prefill in `max_batch_tokens`-bounded chunks interleaved with decode tokens, keeping short-request inter-token latency within budget while a multi-hundred-token prompt streams in.
- Prefix caching: reference-counted copy-on-write KV blocks reused across requests sharing a prompt prefix (LRU eviction, collision-guarded hash-chain keys); cached-prompt-token counts reported in usage.
- Cancellation: mid-generation aborts (client hang-up, stop literals, explicit cancel) reclaim every KV block of the aborted sequence without disturbing concurrent peers.
- API completeness: `tools` / `tool_choice` with schema injection and a fenced-JSON tool-call detector surfaced as OpenAI `tool_calls` (JSON + SSE deltas), `logprobs` / `top_logprobs`, `presence_penalty`, `frequency_penalty`, `logit_bias` (numeric ids or token strings), `seed`.
- Metrics: step-boundary engine counters (tok/s, step time, batch occupancy, prefill chunks, cache hits, KV utilization, finish-reason tallies, process working-set/CPU%) published as `metrics_snapshot` log lines and streamed over `GET /monitor/stream` (SSE); `lair monitor [-u URL] [--interval ms]` CLI renders a live per-model stats table.
- Config hot-reload: the serving process watches its config file; batch width, thread quota, and KV budget apply live, context length applies to new requests, restart-only fields (block size, model path, quant geometry) are refused with an actionable message; invalid edits never touch the running engine.
- Thread placement: worker threads pinned to a model's configured `core_set` (Windows), per-model `thread_quota`, and automatic sizing from physical cores (SMT siblings excluded).
- Encoder groundwork: bidirectional (non-causal) forward pass and mean/CLS pooling helpers for a future embeddings endpoint.

### Changed

- Default engine thread count now derives from physical cores instead of capping at 16.
- Concurrent requests no longer receive `503` by default; single-flight serving was replaced by batched execution.

## [0.1.0] - 2026-11-01

### Added

- GGUF loader: mmap'd tensors, full metadata table, vocabulary — F16 / Q8_0 / Q4_K quant families.
- GGUF-embedded tokenizer: BPE encode/decode with special tokens and digit-run handling; SmolLM-compatible merge ordering.
- Chat-template interpreter for the Jinja subset used by real templates (ChatML, Phi-style): conditionals, loops, `tojson`/`join`/`trim`; unsupported constructs fail loudly.
- AVX2 fused-dequant GEMV kernels for Q4_K_M / Q8_0 / F16 with row-block threading; bit-identical across thread counts.
- Layer executor: RMSNorm, RoPE, grouped-query attention, SwiGLU FFN, qkv bias (Llama/Qwen-class dense models).
- Paged KV cache: 16-token block pool with allocator and per-model budget from config.
- Sampling: greedy, temperature, top-k / top-p, seeded determinism.
- OpenAI-compatible HTTP API on loopback by default: `POST /v1/chat/completions`, `POST /v1/completions` (JSON + SSE streaming with usage), `GET /v1/models` (with `context_length` and quant metadata), `GET /healthz`; OpenAI-shaped error taxonomy (`invalid_request`, `model_not_found`, `context_length_exceeded`, `engine_overloaded`, `internal`) and `x-lair-request-id` on every response.
- Backend seam: `IBackend` + weight arenas + swappable `IWeightSource` providers; alternate provider verified end-to-end (arena-copied weights decode identically to mmap).
- Parity harness against a pinned llama.cpp server build with committed calibrated fixtures for SmolLM-135M-Instruct F16 and Q8_0 (`scripts/generate-parity.ps1`, `Category=Parity` tests).
- YAML config file (server, engine, single model + placement) with strict validation and actionable errors.
- CLI: `lair serve -c lair.yaml` and `lair chat` REPL — streamed replies, `/reset`, overlong-turn trim-and-retry, Ctrl+C interrupt; structured per-request logs.
