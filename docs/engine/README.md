# LAIR Engine Documentation

Technical documentation of the implemented engine: how each subsystem works, why it is built that way, what it refuses today, and which measurement justifies its defaults. Operator-facing material — commands, flags, config keys, feature overview — lives in [`docs/user/README.md`](../user/README.md) and [`docs/user/reference/`](../user/reference/). The two trees are described side by side in [docs/README.md](../README.md).

As of 2026-11-27. Each page carries its own "as of" date; when a page and the code disagree, the code is authoritative and the page is a bug — [`AGENTS.md`](../../AGENTS.md) lists which page every kind of change must update.

## Layout

This index stays at `docs/engine/README.md`; every detail page lives under [reference/](reference/). Start with [architecture.md](reference/architecture.md) for the module map and seams, then [request-lifecycle.md](reference/request-lifecycle.md) for one request traced end to end through the pipeline. The remaining pages go deeper into each subsystem.

| Page | Covers |
|---|---|
| [architecture.md](reference/architecture.md) | Projects and their responsibilities, `IBackend` / `IWeightSource` / `IKvPool` seams, how a request becomes kernel calls, landed-but-not-default internal seams |
| [request-lifecycle.md](reference/request-lifecycle.md) | One inference pipeline from HTTP bytes to the closing chunk: admission, templating, tokenization, prefix lookup, chunked prefill, decode steps, sampling, output parsing, finish; encoder-only (embeddings) route |
| [scheduling.md](reference/scheduling.md) | Pipelines vs merged steps, routes and segments, arbiter cost model, concurrency caps and queueing, fairness, prefill lanes, token budgets, hot-reload of scheduling knobs |
| [monitoring-and-metrics.md](reference/monitoring-and-metrics.md) | Metrics bus, engine snapshot fields, request lifecycle events, ledger occupancy, accelerator accounting (GPU busy / PCIe bytes / VRAM), what the monitor refuses to mutate |
| [cpu-kernels.md](reference/cpu-kernels.md) | ISA dispatch (`AVX-VNNI`/`AVX2`), GEMV vs GEMM vs tiled prefill, grid modes and row widths, fused pair/triple dots, shared-input dot paths, gather kernels, thread affinity and NUMA first-touch, kernel tuning knobs |
| [quantization.md](reference/quantization.md) | GGUF block-quant layout as LAIR reads it, per-family decode strategy (K-quants, i-quants, sub-4-bit), integer-dot selection, scale/block handling, mixed-tier files, format coverage per backend and what refuses |
| [cuda-backend.md](reference/cuda-backend.md) | Driver-API binding with embedded PTX, weight GEMV/GEMM kernels per format, attention kernels, KV mirroring, device-step composition, streams and multi-model device compute, capability gates and refusal semantics, transfer accounting |
| [kv-cache.md](reference/kv-cache.md) | Paged 16-token blocks, `BlockPool` / `SequenceKv`, refcounted copy-on-write prefix sharing and the prefix cache, quantized KV format pairs and their attention paths, windowed/sliding-window KV selection, residency modes |
| [residency-and-budgets.md](reference/residency-and-budgets.md) | Compute capacity ledger and device ledger, per-resource budget chains, tier vocabulary (`dram`/`vram`/`disk`) and what may move today, weight arenas and mmap placement, host posture detection, NUMA spread, prefill buffer clamps, `budget_exhausted` semantics |
| [model-profiles.md](reference/model-profiles.md) | Profile registry (`llama`, `qwen2`, `qwen3`, `gemma2`, `gemma3`, `phi3`, `olmoe`, generic match), tensor map resolution, rope-scaling plans, softcap and sliding-window/interleave layer behavior, sparse-FFN descriptors and dense-resident MoE execution, GDN/QSA layers, the synthetic Qwen4Exp engine seam |
| [tokenizer-templating-tools.md](reference/tokenizer-templating-tools.md) | GGUF tokenizer profiles (BPE families and SentencePiece longest-match with byte fallback), special-token handling, minja-parity template engine (parser, evaluator, builtins, limits), tool-parser resolution and fenced-block fallback, reasoning regions and token budgets |
| [sampling-and-drafting.md](reference/sampling-and-drafting.md) | Sampler stage order and transforms (`temperature`/`top_p`/`top_k`, penalties, `logit_bias`, seed, logprobs/top-logprobs accounting), finish reasons, the MTP drafting seam and why it is not public yet |
| [measurement-harnesses.md](reference/measurement-harnesses.md) | What each bench verb measures and how to read its output (`gemv-file`, `prefill-profile`, `kv-quant`, `cuda-gemv`/`cuda-gemm`/`cuda-prefill`), parity fixtures and their gates, lane scripts, the `reports/<timestamp>/` artifact convention |
| [comparisons.md](reference/comparisons.md) | Where LAIR's design differs from llama.cpp / ik_llama.cpp, dotLLM, vLLM, colibri, Strata — per subsystem, with the measured reason for each divergence and what those engines do better |
| [tricks-and-tweaks.md](reference/tricks-and-tweaks.md) | Catalog of individual engine tricks: register-pressure decisions, integer-dot selection, chain-group widths, matmul route selection per quant family, fused QKV triples, near-tie argmax handling, prefetch and hot-half experiments — each with the wall it removes and its evidence |

## Conventions used across these pages

- Behavior is described as this build behaves. Internal seams that are not reachable from the public config say so explicitly.
- Code paths are cited by file (`src/Lair.Kv/PrefixCache.cs`); performance claims point at a committed report or `bench/reports/` summary rather than restating numbers without posture.
- Format coverage tables list what the **loader accepts**, what each **backend can execute**, and what is **refused at model open** — three different facts that are easy to conflate.

Repository entry points: [README.md](../../README.md) (project overview), [DEVELOP.md](../../DEVELOP.md) (build/test/bench workflow).
