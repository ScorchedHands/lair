# Changelog

All notable changes to LAIR are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.6] - 2026-11-09

### Added

- Tool-call output parsing resolved per model at load through a parser plan (`OutputParserRegistry` / `IToolOutputParser`): the fenced-JSON detector behind the plan, a template-native reader for the call shape a tool-capable chat template instructs its own model to emit (`{name, parameters}` objects, arguments compacted onto the wire), and an inert `none` plan that never reads decoded text. `models[].tool_parser: auto | template | fenced | none` (default `auto`, preserving the existing capability precedence) with load-time validation, a new-requests-only hot-reload class, a `tool_parser model='<name>' choice=<choice>` load line, and the resolved choice reported by `/v1/models`.
- Reasoning-region recognition from the markup a generation emits (`<think>` … `</think>`): closed regions, several regions in document order, markup inside a region treated as content, a close with no open left as ordinary text, and an open that never closes recorded to the end of generation. Recognition also runs over streamed output piece by piece, holding a boundary marker split across pieces until the next piece settles it, so no part of a region reaches the wire. `models[].reasoning: keep | strip` (default `keep`, new-requests-only class, `reasoning model='<name>' policy=<policy>` load line, reported by `/v1/models`); under `strip` the region leaves visible content while `GenerationResult.ReasoningRegions` records its span, and prompt tokens, completion tokens, and finish reasons are unchanged.
- `Category=ToolCalls` goldens for both parser families plus reasoning recognition and streaming-piece cases; a pinned-Qwen3 serve suite (`Category=Serve`) proving `keep` carries the model's own thinking block, `strip` removes it at identical token counts, and a policy flip landing mid-generation leaves the admitted sequence on the policy it started with.

### Changed

- A sequence runs on the output-parser plan it was admitted on: the plan is captured when generation starts and travels on the result, so a `tool_parser` or `reasoning` reload re-plans later requests only. Streaming content filters follow the policy on the result rather than whatever is current when generation ends.
- `/v1/models` now also reports the resolved reasoning policy alongside `tool_parser`, `architecture`, and `quant`.

## [0.3.5] - 2026-11-09

### Added

- `Q2_K` and `Q3_K` quant support end to end on CPU: loader acceptance plus scalar and AVX2 fused-dequant GEMV kernels in single and batched decode (llama.cpp `ggml-common.h` block geometry from the pinned b12032 tag — 256 elements per block, 84 bytes for `Q2_K`, 110 for `Q3_K`), so published low-bit K-quant files load and serve as-is.
- `tests/fixtures/parity-llama32-q3ks.json` and `tests/fixtures/parity-llama32-q2k.json`: pinned b12032 output parity for `models/Llama-3.2-1B-Instruct-Q3_K_S.gguf` and `models/Llama-3.2-1B-Instruct-Q2_K.gguf` (`Category=Parity`), regenerated with `scripts/generate-parity.ps1`.

### Changed

- GGML type acceptance resolved at load through a descriptor registry (`GgmlTypeRegistry.TryResolve` / `QuantDescriptor`) instead of a hardcoded switch: membership in the registry is the acceptance rule, and block geometry plus row-stride/byte-length validation are computed from the descriptor. Currently accepted tensor types: F32, F16, Q8_0, Q4_K, Q5_0, Q5_K, Q6_K, Q2_K, Q3_K.
- Parity cross-tool logprob band applies only to candidates within 6 nats of the reference's own top-1, where a candidate carries sampling weight; deeper tail candidates are gated on membership in LAIR's calibrated window and reproduction of LAIR's calibrated value instead of absolute agreement, since sub-nat logit differences legitimately reorder the deep tail. Argmax-membership and reproduction gates are unchanged.
- `ConcurrencyTests` and `ParityTests` share one serialized xunit collection (`model-loaded`), so real-model suites never load engines beside the latency-budget suite — a second engine loading at once starves the inter-token budgets that suite measures.

## [0.3.4] - 2026-11-08

### Added

- Architecture support resolved at load through a profile registry: `llama`, `qwen2`, `qwen3`, `gemma2`, `qwen2moe`, and `qwen3moe` profiles plus a generic llama-shaped fallback that accepts an unknown `general.architecture` only when the tensor table matches exactly, otherwise failing and naming the missing or unexpected tensors. Profiles produce an immutable model plan (tensor map with aliases, norm/activation/bias/geometry/RoPE strategies, caps, window, sparse-FFN descriptor) that the forward path consumes instead of architecture names.
- `qwen3` and `gemma2` dense families: per-head QK-norm, `attention.head_length` independence, GELU-exact GEGLU gating, attention-score and final-logit softcaps (`attn_logit_softcapping`, `final_logit_softcapping`), a `sqrt(hidden)` input-embedding gain, and post-sublayer output norms (`blk.N.post_attention_norm.weight`, `blk.N.post_ffw_norm.weight`). Each shape was measured against pinned llama.cpp b12032 rather than assumed from the architecture write-up.
- Sliding-window attention: `<arch>.attention.sliding_window` resolves into the plan and masks out-of-window scores in every attention path while the KV cache keeps every position, with per-family interleave (`AlternatingLocalFirst` for `gemma2`, `AllLayers` elsewhere).
- Vocabulary resolution falls back to the `tokenizer.ggml.tokens` length when `<arch>.vocab_size` is absent, reporting `vocab_fallback` and `vocab_disagree`.
- Sparse-FFN descriptors: MoE metadata (`expert_count`, `expert_used_count`, `expert_shared_count`, `expert_weights_scale`, `expert_weights_norm`) and the reference's expert tensor names resolve into the plan, and an MoE model fails at engine open with a message naming sparse-FFN execution as not yet implemented.
- `tests/fixtures/parity-qwen3.json`: pinned b12032 output parity for `models/Qwen3-0.6B-Q8_0.gguf` (`Category=Parity`), regenerated with `scripts/generate-parity.ps1`.

### Changed

- Attention bias behavior follows the declaration (`attention.use_bias`, `attention.use_norm_bias`) instead of tensor presence; declared-false with tensors present logs `bias_ignored` and loads as if the tensors were absent.
- RoPE dimension pairing is a per-family plan value: adjacent `(2i, 2i+1)` for `llama`/`qwen2`/generic, rotate-half `(i, i + head_dim/2)` for `qwen3` and `gemma2`.
- The Llama-3.2 test fixture is now `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` (770 MB) and its template and tokenizer goldens were regenerated from it; the model test budget is 512 MB–1 GB of working set.

### Fixed

- `qwen3` logits no longer drift from the reference: LAIR applied llama-family adjacent-dimension RoPE to a rotate-half vocabulary, which moved argmax choices and cost 1.1–2.3 nats per token. Parity against pinned b12032 now holds within ~0.001 nats on the fixture prompts.

## [0.3.3] - 2026-11-06

### Added

- Tokenizer profiles keyed by `tokenizer.ggml.pre`: `llama-bpe`, `gpt2`, and `smollm` are resolved once at model load into concrete tokenizer state (pre-split strategy, merge-lookup shortcut, decode space cleaning, BOS/EOS defaults). An unknown `pre` or model fails the load naming the offending value and the supported profile list.
- GGUF tokenizer metadata coverage matching what llama.cpp reads: `tokenizer.ggml.byte_fallback`, `byte_to_token` (replaces the `<0xNN>` vocab scan when declared), `byte_to_utf8`, `add_space_prefix` (read and exposed), `eot_token_id`, and `utf8_flush_between_controls`. BOS/EOS fallbacks are profile-scoped and each applied default logs a `tokenizer_default` line.
- Tokenizer parity harness: `scripts/generate-tokenizer-parity.ps1` captures pinned llama.cpp b12032 `/tokenize` id sequences for both real fixtures into `tests/fixtures/parity-tokenizer-smollm.json` and `tests/fixtures/parity-tokenizer-llama32.json` (model sha256 + `pre` guards), and `Category=Tokenizer` asserts LAIR's `Encode` matches id-for-id.

### Changed

- A model that declares `tokenizer.ggml.eot_token_id` now stops on that id instead of its EOS id.

### Fixed

- Pre-tokenizer classification works on Unicode code points, not UTF-16 code units: astral characters (emoji) no longer break the leading-space fold, and the llama-bpe symbol fold no longer requires the space itself to be a symbol. Both were caught by the new `/tokenize` goldens.

## [0.3.2] - 2026-11-06

### Added

- `Lair.Templating.Minja`: a managed template engine at full minja parity for minja's supported subset — statements (`if`/`elif`/`else`, `for` with `loop.*`/`recursive`/`if`/`break`/`continue`/destructuring, `set` with namespaces and destructuring, `macro`/`call`/`filter`), full expression syntax including inline list/map literals, slices, membership and boolean operators, minja's filter/test/function collection, and a chat-template wrapper with dummy-render capability probes, chat polyfills, and BOS handling. Templates are parsed once at model load and cached for the model's lifetime.
- Hard template resource limits with library defaults and no config knobs: template source size, parser frames, evaluator/call depth, sequence length, loop iterations per render, rendered output size, a 30 s render deadline, and cooperative cancellation — parse guards fail as syntax errors, render guards as bounded actionable render errors.
- Llama-family chat templates serve end-to-end: `Llama-3.2-1B-Instruct-Q8_0.gguf` renders its own GGUF chat template, with byte-exact golden parity against pinned llama.cpp b12032 `/apply-template` (`Category=TemplateParity`, regenerated with `scripts/generate-template-parity.ps1`) and real-model serve tests (`Category=Serve`).
- `models[].force_fenced_tools` (bool, default false, hot-reload new-requests-only): forces LAIR's fenced tool injection even when the model's template renders tools itself.

### Changed

- Chat tool handling picks one surface from template capability: a tool-capable template renders declared tools natively and LAIR's fenced block is skipped; a toolless template keeps LAIR's schema injection and fenced-JSON detector exactly as before.
- Template failures are actionable: model startup fails with model path + offending construct + offset; request-time render failures log model name + `x-lair-request-id` + bounded detail and map to the OpenAI `invalid_request` error shape.

### Fixed

- Minja semantics proven against the pinned oracle: lowercase `true`/`false` parse as boolean literals; the parser strips exactly one trailing newline from the template source; macro call arguments and defaults evaluate in the calling scope, so recursive macro helpers work.

### Removed

- The legacy mini-Jinja `ChatTemplateRenderer` (with `TemplateSyntaxException`/`TemplateRenderException`) is deleted; there is no fallback template engine.

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
