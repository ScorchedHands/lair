# LAIR — Local AI Inference Runtime

A C#/.NET inference runtime that serves GGUF LLMs locally behind an OpenAI-compatible HTTP API, with a built-in chat REPL and a live metrics monitor. Dense models across the `llama`, `qwen2`, `qwen3`, `gemma2`, `gemma3`, and `phi3` architecture families — resolved at load through a profile registry, with unknown architectures accepted on an exact llama-shaped tensor match — CPU-first AVX2 kernels, continuous batching with chunked prefill, paged KV cache with prefix reuse, config hot-reload, GGUF-embedded tokenizer with llama.cpp-parity profiles (`llama-bpe`/`gpt2`/`smollm`) plus a SentencePiece vocabulary path selected by `tokenizer.ggml.model`, and chat templates rendered from GGUF metadata by a managed minja-parity template engine. SentencePiece vocabularies (Gemma-2, Gemma-3, Phi-3) encode by greedy longest-match over piece vocabularies with byte fallback and serve behind the same API; routed MoE models execute densely with selected experts resident in RAM, report their sparse strategy through `/v1/models`, and windowed attention honors per-layer windows through bounded KV block selection, monitor-visible read-footprint accounting, and backward-only encoder masking.

## Requirements

- .NET SDK 10.0.400+ (x64)
- A GGUF model file, e.g. from Hugging Face — see Quantization below for the supported tensor types and which one to pick

## Quantization

Supported tensor types: F32, F16, BF16, Q8_0, Q4_K, Q5_0, Q5_K, Q6_K, Q3_K, Q2_K, IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, Q2_0.

For dense Llama-class models under a 4 GB working-set budget on CPU, `IQ4_XS` is the measured pick: it decoded fastest of the fixtures tested — 11.6 tok/s against 10.3–10.4 for `Q4_K_M`, at context 4096 with thread quota 8 — and it is smaller than the baseline (743 MB against 808 MB). Quality is a separate question: the repo's parity fixtures pin `smollm` models only, so IQ4_XS is a throughput pick here rather than an accuracy-verified one. Sub-Q4 i-quants (`IQ3_XXS`, `IQ3_S`, `IQ2_*`) cut file size further but decode about 2.6× slower than `Q4_K_M` on this runtime, because their kernels cost roughly 6.5× more per element — they buy bytes you then pay for in tokens.

## Quick start

```powershell
# 1. Build once; this produces src\Lair.Host\bin\Release\net10.0\lair.exe on Windows (or ./lair on Linux/macOS).
dotnet build Lair.sln -c Release

# 2. Copy the example config and point it at your model
copy lair.example.yaml lair.yaml
# edit lair.yaml: model path, name, context, kv_blocks, port

$LAIR = ".\src\Lair.Host\bin\Release\net10.0\lair.exe"

# 3. Serve the OpenAI-compatible API (binds 127.0.0.1 by default)
& $LAIR serve -c lair.yaml

# 4. Or talk to the model directly in a terminal REPL
& $LAIR chat -c lair.yaml
```

## API

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | JSON or SSE streaming (`"stream": true`); supports `tools`/`tool_choice`, `logprobs`/`top_logprobs`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `seed` |
| `POST /v1/completions` | legacy text completions, JSON or SSE |
| `GET /v1/models` | model list including `context_length` and quant metadata |
| `GET /monitor/stream` | SSE stream of engine metrics snapshots and request lifecycle events (versioned schema; CUDA-backed serves also publish GPU-busy, PCIe transfer totals, persistent device VRAM bytes) |
| `GET /healthz` | liveness + model readiness |

Any OpenAI SDK works by pointing `base_url` at your configured API root, e.g. `http://127.0.0.1:8080/v1`. Errors use the OpenAI error shape with codes `invalid_request`, `model_not_found`, `context_length_exceeded`, `engine_overloaded`, `internal`. Every response carries `x-lair-request-id`.

Requests run concurrently through shared batch steps; `503 engine_overloaded` surfaces only when engine capacity is exhausted. Chat requests that declare tools get exactly one tool surface, resolved when the model loads: a model whose chat template renders tools itself gets them through the template and its own call shape is read out of the decoded text, a toolless template gets LAIR's schema injection plus the fenced-JSON detector, and `tool_parser: auto | template | fenced | none` pins the choice (`force_fenced_tools: true` still forces the injected form). A model that emits reasoning markup (`<think>` … `</think>`) streams it into visible content by default; `reasoning: strip` withholds recognized regions from visible output while their boundaries stay recorded, and a flip applies to requests admitted after it. Recognized calls appear as `tool_calls` (execution is client-side), and `/v1/models` reports the resolved parser and reasoning policy.

Model token budgets bound a generation on both surfaces: `max_reasoning_tokens` and `max_response_tokens` are optional per-model defaults and optional request fields, a request value clamps to the model's declared ceiling with the clamp reported in `token_budgets`, a reasoning region closes at its ceiling, the response ceiling bounds visible output, and whichever ceiling ended the run is reported as `budget_exhausted` on the response body, on the closing chunk of a stream, and on the monitor's request frames. A reasoning ceiling binds only a model whose tokenizer carries `<think>` and `</think>` as whole tokens; a model declaring `max_reasoning_tokens` on a vocabulary that cannot is refused at load naming the model, the key, and the marker.

## REPL

`/exit` or `/quit` to leave, `/reset` to clear the conversation. Ctrl+C interrupts the current generation. Overlong turns are reported as a single-line error after an automatic trim-and-retry.

## Monitor UI

Point a browser at your server's `/ui`, e.g. `http://127.0.0.1:8080/ui` when default settings are used: a tabbed web UI with live engine stats — configured models (expandable for config and state), in-flight requests, per-request throughput sparklines, recent request log — plus a built-in chat client for exercising the model directly from the browser (streaming replies, mid-stream stop, tool-call display). GPU utilization and PCIe transfer cards appear for CUDA-backed serves, while CPU-only rows stay reserved placeholders.

## Monitor CLI

```powershell
& $LAIR monitor            # defaults to http://127.0.0.1:8080; pass -u when your server uses another bind/port
& $LAIR monitor -u http://127.0.0.1:8181 --interval 1000
```

Live table per model: tok/s, step time, batch occupancy, cache-hit rate, KV utilization, active sequences. When a CUDA-backed engine is serving, rows append `gpu <kernel busy %>`, runtime PCIe copy bytes/calls (`pcie … over N copies`), and persistent device weight residency (`persistent-vram … MiB`). CPU-only serves omit those columns; the same facts arrive on `/monitor/stream` as optional additive fields.

## Model acquisition

A model entry may declare a published source:

```yaml
models:
  - name: manual-fetch-smollm
    path: models/SmolLM-135M-Instruct.Q8_0.gguf
    source:
      repo: QuantFactory/SmolLM-135M-Instruct-GGUF
      revision: d36054e030c66b4be24b0c65513ece348db06ba5
      file: SmolLM-135M-Instruct.Q8_0.gguf
```

`source:` pins a repo, a revision, and one or more GGUF files. A branch revision is resolved once at fetch and recorded as a commit sha in `models/manifest.json`; load verifies the manifest and fetches the pinned artifact only when it is absent. Hashes are computed by LAIR after download, and a `.part` file is never loadable.

```powershell
& $LAIR model fetch QuantFactory/SmolLM-135M-Instruct-GGUF -c lair.yaml
& $LAIR model check manual-fetch-smollm -c lair.yaml
& $LAIR model list -c lair.yaml
```

`model check` is report-only: it prints candidate deltas against the hub's current HEAD without downloading, repinning, or changing the load path. `/monitor/stream` carries the same candidate as optional `metrics_update_candidate` when a served model has a source pin and a candidate is found. Gated repos authenticate through `LAIR_HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN` fallback); the token is sent only to `huggingface.co` and is never written to logs, errors, or the manifest.

## Configuration

One YAML file (`lair.example.yaml` shows every knob): server bind/port, engine `threads`, `max_batch_tokens`, optional `prefill_kernel`, optional `prefill_step_budget_ms`, block size, optional top-level `host:` capacity posture (`dram_mb`, `vram_mb`, `cpu_cores`, presence-only `disk_path`, and per-node overrides under `numa.nodes`), exactly one model with path/name/type/context/KV budget, optional budgets chains for weights/KV/prefill buffers using tier caps like `<tier>_mb` or KV `<tier>_blocks`, optional token/tool/reasoning controls, optional `source:` acquisition, placement policy (`device: auto|cpu`, `kv_residency`, `prefill: auto|cpu`, `gpu_layers`, `prefill_lane`, core pinning/quota/overcommit, NUMA spread), and UI ring size. See [docs/reference/CONFIG.md](docs/reference/CONFIG.md) for the full current schema.

A running server watches the config file and applies edits automatically: batch width, thread quota, system-memory KV budget (`kv_blocks` or `budgets.kv.caps.dram_blocks`), step-time prefill budget, and UI ring size take effect live; context length, token budgets, tool/reasoning policy, kernel selection, NUMA spread, and `prefill_buffers` ceilings apply to new requests (a kernel edit re-points the engine's prompt-window route from its next window); server bind/port, model identity/path/source pins, host capacity totals/node overrides, weight caps, device-backed KV caps/residency modes, placement core sets/device/prefill/gpu layers/core overcommit, and budget chain orders are refused with a restart-only message. Invalid edits are rejected without touching the running engine. Prompt windows resolve a measured default of 32 rows independently from `max_batch_tokens`; `prefill_kernel` selects `gemv`, `tiled`, or `gemm`, and `gemm` is accepted only when every dense weight matrix in the served file has a blocked matmul format (`Q8_0`, `Q4_K`, `IQ4_XS`) — otherwise model open refuses naming those formats. `prefill_step_budget_ms` sizes prompt-window token counts from measured per-row cost, `prefill_buffers` caps can narrow the effective prompt-window row width while reporting requested versus effective width, and host posture plus placement/budget keys drive capacity, worker-grant, residency, prefill, and monitor reporting through the monitor stream. See [docs/reference/CONFIG.md](docs/reference/CONFIG.md) for hot-reload classes per key.

At load the server prints what each model will actually run: the profile and tensor decisions, the `tool_parser` choice, the `reasoning` policy, the prefill kernel selection (`prefill_kernel model='…' kernel=gemv`), the marker ids a declared reasoning budget binds to (`reasoning_region model='…' open_token_id=151667 close_token_id=151668`), and the positional plan when the file declares rope scaling or a partial rotary slice (`rope_scaling model='…' variant=yarn multiplier=2 original_context_length=512 …`).
