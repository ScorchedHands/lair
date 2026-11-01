# Changelog

All notable changes to LAIR are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
