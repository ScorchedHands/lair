# LAIR — Local AI Inference Runtime

A C#/.NET inference runtime that serves GGUF LLMs locally behind an OpenAI-compatible HTTP API, with a built-in chat REPL and a live metrics monitor. Dense models across the `llama`, `qwen2`, `qwen3`, `gemma2`, `gemma3`, `phi3`, and `olmoe` architecture families — resolved at load through a profile registry, with unknown architectures accepted on an exact llama-shaped tensor match — CPU-first AVX2/AVX-VNNI dispatch, optional CUDA-backed hybrid execution for supported tensor formats through `placement.gpu_layers`, continuous batching with chunked prefill, paged KV cache with prefix reuse, config hot-reload, GGUF-embedded tokenizer with llama.cpp-parity profiles (`llama-bpe`/`gpt2`/`smollm`) plus a SentencePiece vocabulary path selected by `tokenizer.ggml.model`, and chat templates rendered from GGUF metadata by a managed minja-parity template engine. SentencePiece vocabularies (Gemma-2, Gemma-3, Phi-3) encode by greedy longest-match over piece vocabularies with byte fallback and serve behind the same API; routed MoE models execute densely with selected experts resident in RAM, report their sparse strategy through `/v1/models`, and windowed attention honors per-layer windows through bounded KV block selection, monitor-visible read-footprint accounting, and backward-only encoder masking.

User documentation lives in [`docs/README.md`](docs/README.md) (feature and operation overview, plus what is planned but not yet available), with the full [CLI reference](docs/reference/CLI.md) and [`lair.yaml` configuration reference](docs/reference/CONFIG.md) behind it.

## Requirements

- A `lair` executable. Build it from a checkout with the .NET 10 SDK (10.0.400+, x64):

  ```powershell
  dotnet publish src/Lair.Host -c Release -r win-x64 --self-contained -p:PublishSingleFile=true -o release
  ```

  `release\lair.exe` carries the .NET runtime and serves its Web UI from the `wwwroot` directory beside it. Building, testing, and running from a checkout is covered in [DEVELOP.md](DEVELOP.md).
- A GGUF model file, e.g. from Hugging Face — see Quantization below for the supported tensor types and which one to pick
- Optional hybrid CUDA execution needs a CUDA-capable NVIDIA driver/runtime and a model whose opened `placement.gpu_layers` span uses the currently covered CUDA kernel formats (`F32`, `F16`, `BF16`, `Q4_K`, `Q6_K`)

## Quantization

Supported tensor types: F32, F16, BF16, Q8_0, Q4_K, Q5_0, Q5_K, Q6_K, Q3_K, Q2_K, IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, Q2_0.

For dense Llama-class models under a 4 GB working-set budget on CPU, `IQ4_XS` remains the measured pick: it decoded fastest of the fixtures tested — ≈16.5 tok/s monitor-windowed decode against ≈12.4 for `Q4_K_M`, at context 4096 with thread quota 8 — while staying under the ceiling (≈1,344 MB peak working set). Quality is a separate question: the repo's parity fixtures pin `smollm` models only, so IQ4_XS is a throughput pick here rather than an accuracy-verified one. Sub-Q4 i-quants are no longer a measured ≈6.5× per-element tax on this build: settled GEMV samples put them at roughly 0.25–0.39 ns/element against `Q4_K`'s ≈0.164, about 1.5–2.3× rather than 6.5× (`IQ3_XXS` sits modestly above the intended 2× band), and a mixed-tier `UD-IQ3_XXS` fixture now peaks around 13.0 tok/s instead of ≈3.9; it is still slower than either dense baseline here, so sub-Q4 files remain a size tradeoff rather than a throughput recommendation.

## Quick start

```powershell
# 1. Copy the example config and point it at your model
copy lair.example.yaml lair.yaml
# edit lair.yaml: model path, name, context, kv_blocks, port

# 2. Serve the OpenAI-compatible API (binds 127.0.0.1; the example config serves on port 8181)
lair serve -c lair.yaml

# 3. Or talk to the model directly in a terminal REPL
lair chat -c lair.yaml
```

The loader defaults are `server.bind: 127.0.0.1` and `server.port: 8080`, but `lair.example.yaml` ships with `port: 8181` — every URL below follows whatever `server.port` your config declares. `monitor` defaults to `http://127.0.0.1:8080`, so point it with `-u` when the server runs on a different port.

## API

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | JSON or SSE streaming (`"stream": true`); supports `tools`/`tool_choice`, `logprobs`/`top_logprobs`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `seed` |
| `POST /v1/completions` | legacy text completions, JSON or SSE |
| `GET /v1/models` | model list including `context_length` and quant metadata |
| `GET /monitor/stream` | SSE stream of engine metrics snapshots and request lifecycle events (versioned schema; integer-dot CPU serves report resolved `metrics_cpu_isa` and `metrics_cpu_gather`; CUDA-backed serves also publish GPU-busy, PCIe transfer totals, persistent device VRAM bytes) |
| `GET /healthz` | liveness + model readiness |

Any OpenAI SDK works by pointing `base_url` at `http://127.0.0.1:8080/v1` (or your configured port). Errors use the OpenAI error shape with codes `invalid_request`, `model_not_found`, `context_length_exceeded`, `engine_overloaded`, `budget_exhausted` (a declared budget chain exhausted at its last copy-holding tier), `client_disconnected` (the client aborted), and `internal`. Every response carries `x-lair-request-id`.

Requests run concurrently through shared batch steps; `503 engine_overloaded` surfaces only when engine capacity is exhausted. Chat requests that declare tools get exactly one tool surface, resolved when the model loads: a model whose chat template renders tools itself gets them through the template and its own call shape is read out of the decoded text, a toolless template gets LAIR's schema injection plus the fenced-JSON detector, and `tool_parser: auto | template | fenced | none` pins the choice (`force_fenced_tools: true` still forces the injected form). A model that emits reasoning markup (`<think>` … `</think>`) streams it into visible content by default; `reasoning: strip` withholds recognized regions from visible output while their boundaries stay recorded, and a flip applies to requests admitted after it. Recognized calls appear as `tool_calls` (execution is client-side), and `/v1/models` reports the resolved parser and reasoning policy.

Model token budgets bound a generation on both surfaces: `max_reasoning_tokens` and `max_response_tokens` are optional per-model defaults and optional request fields, a request value clamps to the model's declared ceiling with the clamp reported in `token_budgets`, a reasoning region closes at its ceiling, the response ceiling bounds visible output, and whichever ceiling ended the run is reported as `budget_exhausted` on the response body, on the closing chunk of a stream, and on the monitor's request frames. A reasoning ceiling binds only a model whose tokenizer carries `<think>` and `</think>` as whole tokens; a model declaring `max_reasoning_tokens` on a vocabulary that cannot is refused at load naming the model, the key, and the marker.

## REPL

`/exit` or `/quit` to leave, `/reset` to clear the conversation. Ctrl+C aborts the generation in flight and ends the REPL (it prints `interrupted` and exits 0). Overlong turns are reported as a single-line error after an automatic trim-and-retry. The `chat` REPL opens its own engine rather than attaching to a running `serve`, and caps each reply at 256 tokens.

## Monitor UI

Point a browser at `http://127.0.0.1:8181/ui` (or your configured port; `/` redirects to `/ui/`) while the server runs: a tabbed web UI with live engine stats — configured models (expandable for config and state), in-flight requests, per-request throughput sparklines, recent request log — plus a built-in chat client for exercising the model directly from the browser (streaming replies, mid-stream stop, tool-call display). GPU utilization and PCIe transfer cards appear for CUDA-backed serves, while CPU-only rows stay reserved placeholders.

## Monitor CLI

```powershell
lair monitor --interval 0        # default http://127.0.0.1:8080, every snapshot
lair monitor -u http://127.0.0.1:8181 --interval 1000
```

`monitor` needs at least one option in this build — a bare `monitor` prints the usage block and exits 1. `-u` takes an absolute URL of the serving server; `--interval` is the minimum milliseconds between rendered rows (0 renders every snapshot; the server emits one every 2 seconds).

Live table per model: tok/s, step time, batch occupancy, cache-hit rate, KV utilization, active sequences. Integer-dot rows add `cpu isa=<scalar|avx2|vnni> gather=<scalar|gather>`. For CUDA-backed serves, the row tail adds `gpu <pct>  pcie <MiB> over <copies>  persistent-vram <MiB>` when GPU utilization, PCIe transfers, or persistent device VRAM are nonzero.

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
lair model fetch QuantFactory/SmolLM-135M-Instruct-GGUF -c lair.yaml
lair model check manual-fetch-smollm -c lair.yaml
lair model list -c lair.yaml
```

`model check` is report-only: it prints candidate deltas against the hub's current HEAD without downloading, repinning, or changing the load path. `/monitor/stream` carries the same candidate as optional `metrics_update_candidate` when a served model has a source pin and a candidate is found. Gated repos authenticate through `LAIR_HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN` fallback); the token is sent only to `huggingface.co` and is never written to logs, errors, or the manifest.

## Configuration

One YAML file (`lair.example.yaml` shows every knob): server bind/port, engine `threads`, `max_batch_tokens`, optional `prefill_kernel`, optional `prefill_step_budget_ms`, block size, optional host posture (`host.dram_mb`, `host.vram_mb`, `host.cpu_cores`, optional `host.disk_path`, and NUMA node overrides), exactly one model with path, name, type, `max_context_tokens`, `kv_blocks`, optional `max_concurrent_requests`, optional `force_fenced_tools`, optional `tool_parser` and `reasoning`, optional token budgets (`max_reasoning_tokens`, `max_response_tokens`), optional `source:` model acquisition, optional placement policy (`device`, `kv_residency`, `prefill`, `prefill_lane`, `core_set`, `thread_quota`, `core_overcommit`, `numa_spread`, `gpu_layers`), optional per-resource `budgets:` tier chains and caps, and the UI ring size `ui.recent_requests`. Every key, its default, its constraints, and its hot-reload mutation class: [docs/reference/CONFIG.md](docs/reference/CONFIG.md).

A running server watches the config file and applies edits automatically: batch width, thread quota, system-memory KV ceilings, and the step-time prefill budget take effect live; context length, token budgets, tool/reasoning policy fields, `prefill_kernel`, and `numa_spread` apply to new requests (prompt-window buffer ceilings classify for later requests but are refused live because their physical clamp is established at open); block size, model path, quant geometry, host capacity totals/node overrides, weight caps, device-backed KV caps/residency modes, placement core sets/device/prefill/`prefill_lane`/`gpu_layers`/core overcommit, and budget chain orders are refused with a message telling you to restart. Invalid edits are rejected without touching the running engine. Prompt windows resolve a measured default of 32 rows independently from `max_batch_tokens`; `prefill_kernel` selects `gemv`, `tiled`, or `gemm`, and `gemm` is accepted only when every dense weight matrix in the served file has a blocked matmul format (`Q8_0`, `Q4_K`, `IQ4_XS`) — otherwise model open refuses naming those formats. `prefill_step_budget_ms` sizes prompt-window token counts from measured per-row cost, `prefill_buffers` caps can narrow the effective prompt-window row width while reporting requested versus effective width, and host posture plus per-model placement/budget keys drive capacity, worker-grant, residency, prefill, hybrid CUDA, and monitor reporting through the monitor stream.

At load the server prints what each model will actually run: the profile and tensor decisions, the `tool_parser` choice, the `reasoning` policy, the prefill kernel selection (`prefill_kernel model='…' kernel=gemv`), the marker ids a declared reasoning budget binds to (`reasoning_region model='…' open_token_id=151667 close_token_id=151668`), and the positional plan when the file declares rope scaling or a partial rotary slice (`rope_scaling model='…' variant=yarn multiplier=2 original_context_length=512 …`).
