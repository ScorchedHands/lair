# Changelog

All notable changes to LAIR are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.14] - 2026-11-15

### Added

- Request pipeline control plane: each request advances as an independent pipeline through transform, prefill, decode, and finish, with its own sampler, quota, and admission path independent from merged step composition. Token streams remain identical whether the pipeline is merged with a batch step or run alone.
- Model execution routes: model plans resolve ordered device segments carrying compute device, weight tier, batch cap, and quota. CPU routes collapse to one segment; synthetic split routes resolve backend spans, staging arenas, and admission refusals without inventing public pseudo-device knobs.
- Admission and quota reporting: optional per-model `max_concurrent_requests`; rejections name whether KV capacity, token budget, or quota blocked the request; `/v1/models` reports concurrency posture and resolved route segments; `/monitor/stream` carries per-segment quota, lane state, activation bytes, and a bounded recent admission-refusal ring.
- Step-time prefill budget: optional `engine.prefill_step_budget_ms` sizes prompt-window token counts from measured per-row cost, clamps the chosen token count, keeps sub-batch rows inside the engine's row cap, and reports chosen tokens, sub-batched forwards, estimated cost, and last step time through scheduler diagnostics.
- Dedicated prefill lanes: `placement.prefill_lane` accepts `shared` or `dedicated`. Dedicated posture sub-allocates a prefill worker share from the model's own claim, partitions pinned core ids disjointly, gives each lane its own activation buffers, and overlaps one prefill forward with decode while KV/cache mutations stay control-plane-reserved.
- Per-resource budget chains: `weights`, `kv`, and `prefill_buffers` resolve independent caps and tier priority chains, with terminal uncapped disk origins for read-side resources, host-capability reporting for unopened tiers, and presence-tracked legacy `kv_blocks` aliasing.
- Device ledger: opened models register weight and KV claims in config declaration order against host-scoped tier accounting. KV and prefill claims observe existing pool/scheduler admission, later claims spill or demote rather than preempting earlier claims, and exhausted admission reports model/resource/tier plus a breakdown under `budget_exhausted` / `engine_overloaded_error`.
- Prefill working-buffer clamp: resolved `prefill_buffers` ceilings can narrow effective prompt-window row width while preserving token identity, reporting requested width, effective width, working-set bytes, clamp description, and clamped chunk counts separately from ordinary chunk/sub-batch stats.
- Compute capacity ledger: top-level `tiers:` declares `dram_mb`, `dram_cores`, optional `nvme_path`, and per-node overrides; model worker claims are granted first-come against reported logical-processor capacity, oversubscribed later claims are clamped to the remainder, and monitor facts report requested versus granted worker counts, overcommit posture, and shared-core models.
- NUMA placement derivation: `models[].placement.numa_spread` accepts `auto`, `local`, or `interleave`; placement maps are derived deterministically from reported topology, claim order, least-loaded node, and lowest node index; per-node allocations are reported, and pinned-set conflicts fail naming the model and conflict.

### Changed

- Scheduler admission now composes steps by route identity, KV-residency compatibility, batch cap, and quota headroom rather than assuming one global CPU step shape.
- GPU-shaped route segments fail naming the missing `lair.backends.cuda` seam instead of silently falling back to CPU execution.
- Dedicated-lane execution uses two CPU backends over one model/weight source rather than separate engine instances, preserving one weight source and one resident KV arena while providing disjoint worker pools and per-lane activation buffers.
- Prefill diagnostics distinguish requested row width from clamped effective width and report lane buffer bytes beside step-time facts.

## [0.3.13] - 2026-11-14

### Added

- Prefill kernel selection: `engine.prefill_kernel` accepts `gemv`, `tiled`, and `gemm`. `tiled` reuses each K/V slot across the rows of a prompt sub-batch while keeping every row's online-softmax order unchanged; `gemm` uses managed unsafe blocked matmul over dequantized weight tiles for dense files whose matrices are all `Q8_0`, `Q4_K`, or `IQ4_XS`. Kernel selection is hot-reloaded as new-requests-only, reports through `/monitor/stream`, and refuses unsupported formats at model open naming them rather than falling back.
- Prefill row width: prompt windows now resolve a measured default of 32 rows independently from decode's batch cap, with candidate widths tested bit-identically across `16`, `32`, and `64` before the engine resolves one. Decode remains capped at 16 sequences.
- Blocked GEMM identity gates: raw-byte comparisons cover representative tensor shapes for `Q8_0`, `Q4_K`, and `IQ4_XS`, including threaded, single-threaded, scalar paths, unsupported-format refusal, model-level prefill logits/KV/decode continuation equality across `gemv`, `tiled`, and `gemm`.
- Bench kernel axis: `prefill-profile` accepts `--kernel gemv,tiled,gemm`, opens one engine per requested kernel × row width, records refusals instead of falling back for unsupported formats, and prints process peak working set. Committed comparison reports cover Qwen3 Q8_0 and Llama-3.2 Q8_0 with raw tables plus a live `gemv` vs `gemm` rollout proof on Qwen3 Q8_0.

### Changed

- Prompt-window routing now has three selectable CPU routes: default batched GEMV, row-tiled causal attention, and dense blocked matmul for proven-format files. Existing parity fixtures remain raw-byte identical across the runnable kernel combinations; live `tiled` prefill measured a 6.5% combined throughput gain on Llama-3.2 IQ4_XS with lower peak working set.
- Concurrency budget tests now run through the row-tiled prefill route, so decode inter-token budgets are checked while that prompt-window kernel is active.

## [0.3.12] - 2026-11-13

### Changed

- Prefill forwards batched rows: a prompt window of `L` rows advances in `ceil(L / 16)` weight traversals through the engine's 16-row batch buffers with causal-within-chunk attention over the growing KV, instead of one token per traversal. Chunk width, decode priority, and step composition are unchanged, and every prompt window produces bit-identical logits, KV bytes, sampled tokens, and subsequent decode steps to the per-token route it replaces. Measured on `Llama-3.2-1B-Instruct-IQ4_XS` (1751-token prompt, 8 threads): 169.3 s / 173.2 s before, 122.4 s / 129.7 s after; `Qwen3-0.6B-Q8_0` (928-token prompt): 64.3 s / 66.6 s before, 43.0 s / 43.7 s after, with its short-request decode unchanged.

### Added

- `Lair.Bench prefill-profile <gguf> [--threads N] [--lengths …] [--reps N] [--per-token] [--out <path>]`: times prompt windows through the batched-prefill path (and optionally the single-token forward) over a fresh KV per measurement, then fits measured cost as a weight term plus an attention term and reports the crossover length where attention becomes the larger term. A run whose costs stop rising with the causal span prints the split as unusable instead of fitting it.
- Committed prefill cost profile (`bench/reports/prefill-profile-2026-11-12.md` with its raw tables): `Llama-3.2-1B-Instruct-IQ4_XS` fits 252–345 ms per weight traversal (1.5–2.1 GB/s of the 496 MB a traversal streams) against 52–57 µs per position pair, crossing over at 555–827 tokens; `Qwen3-0.6B-Q8_0` fits 127 ms per traversal (3.7 GB/s) against 61 µs per position pair, crossing over at 266. At 2048 tokens on Llama the attention term is 110 s against 44 s of weight reads.

### Removed

- The per-token prefill route is deleted rather than kept as a fallback; batched prefill is bit-identical to it under `Category=Prefill` (16/16), with `Category=Parity` 49/49, `Category=Concurrency` 5/5, and `Category=Serve|Cache` 22/22 unchanged.

## [0.3.11] - 2026-11-12

### Added

- SentencePiece vocabularies load and serve: `tokenizer.ggml.model` selects the vocabulary implementation (`llama`/`spm` → SPM, `gpt2`/`llama-bpe` → BPE), orthogonal to the unchanged `tokenizer.ggml.pre` profiles. The SPM path encodes by greedy longest-match over piece vocabularies with score tie-breaking and `<0xNN>` byte fallback, honors word-boundary spelling (U+2581) per file — Gemma-2 declares no space prefix at sequence start, Phi-3 does — and pins `Decode(Encode(text)) == text`. Gemma-2, Gemma-3, and Phi-3 files load through this seam; BPE families are bit-identical.
- Gemma-3 windowed serving: the profile resolves its hybrid pattern (four windowed layers per group of six) as `SixLayerGroupsLocalFirst` with a second rotary table when local/full bases differ, so a 512-token sliding window serves past itself while KV reads stay bounded — measured live at `metrics_kv_windowed_layers = 22`, read footprint 1728 blocks against 4928 skipped.
- Phi-3 profile: fused attention arrives as one `blk.N.attn_qkv.weight` buffer addressed by head-region slices, the gate lives in halves of `ffn_up` with no separate gate tensor (an unexpected `ffn_gate.weight` is ignored and reported), rotary applies rotate-half over the declared 96-dimension slice, and an untied Q6_K output head resolves. Parity golden inside the documented band; worst candidate drift 0.4190 nats.
- New pinned gates: Gemma-2 serve (`Category=Serve|Cache` 22/22), Gemma-3 plan + windowed-serve tests, Phi-3 loader/tokenize cases (`Category=Loader` 231/231, `Category=Tokenizer` 63/63 including id-for-id `/tokenize` goldens for both SPM fixtures with round-trip decode), and a Phi-3 logit fixture inside the band (`Category=Parity` 49/49).

### Fixed

- The advertised quant on `/v1/models` is now measured by stored bytes rather than tensor count, with ties broken deterministically: Gemma-2 reports `Q4_K`, Phi-3 `Q2_K`, and Gemma-3 `Q8_0` because its token embedding (320,864,256 B) outweighs the IQ4_NL weight group; a count-based rule had reported `F32` for that file from 157 norm vectors. Guarded by synthetic and real-fixture cases (`Category=Api`).

## [0.3.10] - 2026-11-12

### Added

- Per-model token budgets: `max_reasoning_tokens` and `max_response_tokens` config keys (absent means unbudgeted, values validated at load naming the model and key), request-side overrides on chat and completion bodies clamped to the model's declared ceiling with the clamp reported in `token_budgets`, and enforcement through scheduling and decode — KV admission reserves the resolved reasoning-plus-response sum, an open reasoning region closes at its ceiling, and the response ceiling bounds visible output. Budget keys hot-reload as new-requests-only changes.
- Budget attribution: `budget_exhausted` names which ceiling ended a run (`"reasoning"` or `"response"`) on chat and completion bodies, on the closing chunk of a stream, and on `/monitor/stream` request lifecycle frames. `finish_reason` values are unchanged.
- Reasoning-marker vocabulary gate at model load: a declared reasoning ceiling binds only when the loaded tokenizer delivers `<think>` and `</think>` as whole tokens. A model declaring `max_reasoning_tokens` on a vocabulary that splits or lacks either marker fails at open naming the model, the key, and the marker; a hot-reloaded ceiling on such a model is refused the same way; response-side accounting on such a model behaves exactly as an unbudgeted region model. Load reports the bound pair as `reasoning_region model='…' open_token_id=… close_token_id=…`.

## [0.3.9] - 2026-11-12

### Added

- Dense-resident MoE execution through the model plan: router softmax/top-k selection with optional renormalization/weights scale, shared-expert support including gated selector semantics when a file ships it, and per-expert gate/up/down accumulation into the residual. The pinned `OLMoE-1B-7B-0924-Instruct-Q2_K.gguf` fixture loads, decodes routed steps through its mixed-tier expert stacks, and serves under the 4 GB model budget; `/v1/models` reports `sparse_ffn_strategy: dense_resident`.
- Sparse-FFN layout representability for later residency work: layouts/ranks/shared-basis tensor roles resolve into the plan, but only `DenseResident` executes in this release and non-dense selections are refused rather than silently downgraded.
- Windowed KV execution accounting: bounded causal/range reads select only the blocks covering a windowed layer's positions; per-layer AllLayers/AlternatingLocalFirst patterns resolve through the model plan; monitor snapshots report `metrics_kv_windowed_layers`, `metrics_kv_read_footprint_blocks`, and `metrics_kv_skipped_read_blocks`.
- Windowed encoder masking: declared positive windows apply backward-only in embeddings, keeping unbounded models bit-identical.

### Fixed

- Olmo chat prompt tokenization matches pinned b12032 `/tokenize` id-for-id for exact raw no-trailing-newline and rendered ChatML prompts by resolving an olmo-specific byte-level split with math-symbol symbol runs and whole-pre-token lookup; non-Olmo BPE profiles are unchanged.
- Expert tensor slices now derive their byte size from each layer's own quant tier rather than one role-wide value, so converted files that mix tiers inside an expert role load correctly. OLMoE attention normalizes whole projected Q/K streams with hidden-sized vectors (`QkNormMode.StreamWeighted`) matching the reference graph behavior.

## [0.3.8] - 2026-11-11

### Added

- Dense IQ quant formats load, decode and serve on the CPU engine: `IQ4_NL`, `IQ4_XS`, `IQ3_XXS`, `IQ3_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `Q2_0` and `BF16` (ggml ids 16/17/18/20/21/22/23/30/42), each with block geometry read from the pinned reference, a literal scalar oracle and an AVX2 fused dequant/GEMV/batch path — 214 kernel cases and 175 loader cases green.
- Two pinned Llama-3.2-1B IQ fixtures for parity and serving: `Llama-3.2-1B-Instruct-IQ4_XS.gguf` (743,141,504 B) and `Llama-3.2-1B-Instruct-UD-IQ3_XXS.gguf` (575,381,536 B), whose mixed per-layer tiers resolve through one registry path and serve with `/v1/models` reporting `quant: IQ4_XS` / `quant: IQ3_XXS`.
- `Lair.Bench gemv-file <gguf> [--threads N] [--reps R] [--max-rows M]`: GEMV timed over real mmap'd tensor rows, reporting per-tensor geometry, first-pass versus warm-best cost, and a private-copy pass that separates page residency from kernel cost.
- Measured quant guidance: `IQ4_XS` decodes at 11.6 tok/s against the `Q4_K_M` baseline's 10.3–10.4 while being the smaller file, so `README.md` names it as the pick with its accuracy caveat stated; sub-4-bit i-quants measure 1.07–1.09 ns/element (about 6.5× the baseline) and the mixed-tier `UD-IQ3_XXS` file serves at 3.9 tok/s.
- A loader gate over the owner's `qwen4exp` model shards: both headers parse and all 1,224 declared tensors resolve with pinned geometry, skipping cleanly when those files are absent.
- Kernel goldens written from the reference's own byte layout (`QuantLayoutTests.cs`), so a format that both LAIR's reader and a test writer misread cannot pass silently.

### Fixed

- `Q5_K` read the fifth bit from a plane advanced per nibble window instead of the single fixed plane the format defines, corrupting groups ≥ 2 in every `Q5_K` tensor since that format landed. Measured against a requantized reference: max |Δ| 0.0666, 1,535 divergent elements, ~770 of 2048 columns wrong in `token_embd`. Corrected at all five kernel sites and in the test writers, and now guarded by reference-encoded block bytes.

### Changed

- `CpuBackend` split into per-format partial files (`CpuBackend+<Format>.cs`, shared IQ plumbing in `CpuBackend+IQ.cs`); no behavior change.
- The parity harness's chat-header prompt carries an unambiguous user turn. The pinned reference's own top-1 token for the previous header flips between quantizations of the same model (`'Hello'` at `Q4_K_M`, `'<'` at `UD-IQ3_XXS`, `'What'` at `Q8_0`), so absolute cross-tool logprobs there were not a stable gate. Bands, windows and tolerances are unchanged and all nine fixtures were regenerated under the pinned harness.

## [0.3.7] - 2026-11-09

### Added

- Declared rope scaling loads instead of being rejected: `<arch>.rope.scaling.type` resolves `none` and `yarn` into a plan carrying the multiplier, original context length, attention factor, correction scale and both beta bounds — any other variant name, and any scaling operand declared without a variant, fails naming the key and its value — and `<arch>.rope.dimension_count` resolves a partial rotary slice instead of rejecting an unsupported partial rotation. YaRN builds its scaled `cos`/`sin` tables at open (correction band from the beta bounds, ramp mixing the interpolated angle toward the unscaled one) and the variant's attention factor is applied exactly once in the score path across single-token, batched, and encode forwards. The math is ported from the pinned b12032 reference (`ggml_rope_cache_init`, `rope_yarn`, `rope_yarn_ramp`, `ggml_rope_yarn_corr_dims`, and the `get_mscale` attention-factor block).
- `rope_scaling model='<name>' architecture='<arch>' variant=… multiplier=… original_context_length=… attention_factor=… rotary_dimensions=…` on the serve log whenever a model's positional plan is not the plain full-head rotation; an unscaled model stays silent.
- `scripts/patch-gguf-rope-scaling.ps1`, which writes a rope-scaling-patched copy of a GGUF — header only, with tensor info and every data byte verified identical and the run idempotent — so a variant can be declared on a file the ecosystem ships without one. `tests/fixtures/parity-qwen3-yarn.json` is generated from its output and holds inside the documented band.
- `Category=Loader` cases for declared-key resolution and variant rejection, `Category=Layers` goldens for the scaled tables, the slice behavior in both pair layouts, and the Open-time announcement line.

### Changed

- The serve host prints its load-time resolution lines on the console (profile and tensor resolution, `tool_parser`, `reasoning`, and the rope plan). The engine logger previously had no provider, so those lines were written and discarded.

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
