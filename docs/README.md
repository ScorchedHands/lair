# LAIR — User Documentation

LAIR (Local AI Inference Runtime) is a C#/.NET runtime that serves a GGUF LLM on your own machine behind an OpenAI-compatible HTTP API, with a terminal chat REPL, a live CLI monitor, and a browser monitor. It **consumes published GGUF quant files as they are** — it does not convert, requantize, or reformat models. By default it binds loopback only (`127.0.0.1`) and serves **one model per process**. Administration is the config file: there is no admin API and no CLI command that mutates a running server; a live server changes only through its watched `lair.yaml`.

This page is the base overview. The authoritative detail lives in [reference/CLI.md](reference/CLI.md) and [reference/CONFIG.md](reference/CONFIG.md).

## Requirements

- **A `lair` executable.** A release build produces a self-contained single-file executable that carries the .NET runtime, so running LAIR needs nothing else installed. Building it needs the **.NET 10 SDK** (every project targets `net10.0`), x64, plus Node 22 / npm 11 only when the Web UI is bundled (`-p:SkipWebUi=true` builds without it):

  ```powershell
  dotnet publish src/Lair.Host -c Release -r win-x64 --self-contained -p:PublishSingleFile=true -o release
  ```

  `release\lair.exe` ships with `wwwroot\` beside it — the Web UI bundle the host serves at `/ui`. A checkout can also be run without publishing (`dotnet build Lair.sln` then `src\Lair.Host\bin\Release\net10.0\lair.exe`). A prebuilt release artifact is not published yet: no publish profile or CI packaging step is committed.
- **A GGUF model file** on disk, from the published quant ecosystem (for example a Hugging Face GGUF repo), or fetched by LAIR from a pinned `source:` declaration.
- CPU kernels use AVX2 when the CPU has it and fall back to scalar paths otherwise.
- Optional hybrid CUDA execution needs a CUDA-capable NVIDIA driver/runtime and a model whose opened `placement.gpu_layers` span uses the currently covered CUDA kernel formats (`F32`, `F16`, `BF16`, `Q4_K`, `Q6_K`).

## Quick start

From a directory holding the `lair` executable and a config file:

```powershell
# 1. Copy the example config (it ships in the repository, not in a release directory) and point it at your model
copy lair.example.yaml lair.yaml
# edit lair.yaml: model name, path, max_context_tokens, kv_blocks, port

# 2. Serve the OpenAI-compatible API
lair serve -c lair.yaml

# 3. Or talk to the model directly in a terminal REPL
lair chat -c lair.yaml
```

Port precision matters here. The config loader's defaults are `server.bind: 127.0.0.1` and `server.port: 8080`, but `lair.example.yaml` ships with `port: 8181` — so a copied example serves on `http://127.0.0.1:8181`, not `8080`. The `monitor` command defaults to `http://127.0.0.1:8080`, so point it explicitly when the server runs on the example's port:

```powershell
lair monitor -u http://127.0.0.1:8181 --interval 1000
```

A bare `monitor` with no flags is a usage error in this build (it prints the usage block and exits `1`): pass `-u` or `--interval`. `serve` and `chat` require `-c <path>`; there is no default config path and no `--config` long form.

In a second terminal, verify the server answers and make one request:

```powershell
curl.exe http://127.0.0.1:8181/healthz

$body = '{"model":"smollm-135m-instruct-q8_0","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}'
Invoke-RestMethod -Uri http://127.0.0.1:8181/v1/chat/completions -Method Post -ContentType application/json -Body $body
```

Config paths are read relative to the working directory you launch `lair` from, and `model fetch`/`check`/`list` without `-c` work on a `models` directory beside the executable — so a release keeps its model packs next to `lair.exe`. Full command semantics, flags, output formats, and exit codes: [reference/CLI.md](reference/CLI.md).

## What LAIR does not do

- It does not convert, requantize, or export models: a GGUF you point at it is the model it runs.
- It does not serve more than one model per process, and it does not serve embeddings — `models:` accepts exactly one entry and `type` accepts `chat` only.
- It does not offer an admin API or any CLI command that changes a running server; the watched config file is the only write path.
- It does not authenticate requests or multi-tenant: exposure is governed by `server.bind` alone. `placement.device` and `placement.prefill` accept `auto`/`cpu` only; whole-model accelerator placement is not accepted through those keys. Hybrid accelerator execution is requested through `placement.gpu_layers` and only for the CUDA kernel formats supported today.

## What you get today

### OpenAI-compatible HTTP API

`serve` exposes exactly these routes; anything else is a 404 (`/v1/embeddings` is **not** served in this build):

| Route | Method | Notes |
|---|---|---|
| `/v1/chat/completions` | POST | JSON or SSE (`"stream": true`); `tools`/`tool_choice`, `logprobs`/`top_logprobs`, `presence_penalty`/`frequency_penalty`, `logit_bias`, `seed`, `stop`, `temperature`/`top_p`/`top_k`, `max_tokens` (default 128), `max_reasoning_tokens`/`max_response_tokens` |
| `/v1/completions` | POST | Legacy text completions, JSON or SSE, same sampling fields |
| `/v1/models` | GET | Advertises `context_length`, the byte-dominant `quant`, `architecture`, model `type`, the resolved `tool_parser` and `reasoning` policy, sparse-FFN strategy, the concurrency-cap posture, and the resolved route segments |
| `/monitor/stream` | GET | SSE engine snapshots (every 2 s) plus request lifecycle events; consumed by `monitor` and the Web UI; CUDA-backed serves also publish GPU-busy, PCIe transfer totals, and persistent device VRAM bytes |
| `/healthz` | GET | `{"status":"ok","model_ready":true}` readiness probe |
| `/ui` | GET | Web UI (tabbed monitor + browser chat client); `GET /` redirects permanently to `/ui/` |

Errors use the OpenAI error shape `{error:{message,type,code}}`:

| Code | HTTP | Raised when |
|---|---|---|
| `invalid_request` | 400 | Malformed body, out-of-range sampling knobs, a chat template that fails to render |
| `model_not_found` | 404 | The request names a model this engine does not serve |
| `context_length_exceeded` | 400 | The prompt alone does not fit `max_context_tokens` |
| `engine_overloaded` | 503 | KV pool cannot back the prompt plus token budget, or every `max_concurrent_requests` slot is held |
| `budget_exhausted` | 503 | The host ledger exhausted a declared budget chain at its last copy-holding tier |
| `client_disconnected` | 499 | The client aborted the request |
| `internal` | 500 | Unhandled engine fault |

Every response — including errors — carries an `x-lair-request-id` header, and monitor lifecycle rows key on the same id. Requests run concurrently through shared batch steps; streaming ends with a `finish_reason` chunk and `[DONE]`. Serve posture and hot-reload behavior of the listener: [reference/CLI.md](reference/CLI.md).

### Chat REPL, CLI monitor, Web UI

- `chat -c lair.yaml` opens its own engine (it does not attach to a running `serve`) and streams replies token by token, capped at 256 tokens per reply; `/exit`, `/quit`, and `/reset` control the session, and overlong turns are trimmed and retried once.
- `monitor -u <URL> [--interval ms]` renders one table row per snapshot: `tok/s`, `step ms`, `seqs/step`, `cache hit`, `KV used/cap`, `active`; for CUDA-backed serves, the row tail adds GPU utilization, PCIe transfer bytes/copies, and persistent device VRAM when those facts are nonzero.
- `GET /ui` while a server runs gives the tabbed browser monitor (models, in-flight requests, per-request throughput, recent-request ring) plus a chat client that exercises the API directly; GPU utilization and PCIe transfer cards appear for CUDA-backed serves.

Details, table semantics, and known defects: [reference/CLI.md](reference/CLI.md).

### Continuous batching, chunked prefill, paged KV with prefix reuse

Requests are not one-request-at-a-time. The scheduler advances many sequences in shared decode steps — one pass over every dense weight matrix serves up to 16 sequences per step — and prompts are prefilled in chunks rather than whole: `engine.max_batch_tokens` (default 512) bounds the largest chunk, and `engine.prefill_step_budget_ms` lets the scheduler size each future chunk from measured step cost. Prompt windows run in sub-batches of a resolved **row** count (measured default 32, engine ceiling 64); no config key sets that number today. `engine.prefill_kernel` selects `gemv` (batched row window), `tiled` (row-tiled causal attention), or `gemm` (blocked matmul over dense weights); a `gemm` selection a served file's weight formats cannot run is refused at model open naming them rather than falling back. KV state is paged in fixed 16-token blocks, and requests sharing a prompt prefix reuse the cached blocks instead of re-prefilling them — the monitor's `cache hit` column is that share. Operator effect: shared weight traversals amortize cost across concurrent requests, chunked prefill keeps a long prompt from starving short ones, and prefix reuse cuts prefill work on repeated conversations. When `placement.gpu_layers` selects a supported layer span, dense forwards for those opened segments run through the CUDA backend while the remaining segments stay CPU-backed. Knobs: [reference/CONFIG.md](reference/CONFIG.md).

### Config hot reload

`serve` watches the file named by `-c`, coalesces edits, revalidates them through the normal loader, and applies each difference by class. A broken edit is dropped whole; the running engine keeps every current value.

| Class | What a live edit does |
|---|---|
| live | Takes effect inside the running engine at the next step boundary (e.g. `max_batch_tokens`, `prefill_step_budget_ms`, `kv_blocks`, `thread_quota`, `ui.recent_requests`) |
| new-requests | Applies to requests admitted after the edit; in-flight sequences keep what they were admitted with (e.g. `max_context_tokens`, `prefill_kernel`, `tool_parser`, `reasoning`, token budgets, `max_concurrent_requests`, `numa_spread`; prompt-window buffer ceilings classify here but are refused live because their physical clamp is established at open) |
| restart-only | Refused with an actionable log line; the old value stays in force (e.g. `server.bind`/`server.port`, `engine.threads`, `block_size`, model `name`/`path`/`type`, `source:`, weight caps, `core_set`, `prefill_lane`, `host:` totals/node overrides, `placement.device`, `placement.kv_residency`, `placement.prefill`, `placement.gpu_layers`, `placement.core_overcommit`, budget chain orders) |

The complete per-key classification: [reference/CONFIG.md](reference/CONFIG.md).

### Model acquisition from Hugging Face

A model entry may declare `source:` — a repo, a revision (commit sha or branch), and one or more GGUF files with optional `sha256`/`size`. With `source:` present, `path` may name a file that does not exist yet: load verifies the pack's `manifest.json` with no network call when the artifact is present, fetches exactly the pinned revision when it is absent, and refuses on any disagreement rather than substituting. Downloads land as `.part` siblings that are never loadable; hashes are computed by LAIR after download, never trusted from the hub. `lair model fetch` lands and pins a declared file set (idempotent, per-artifact locked), `model check` is report-only — it prints candidate deltas against the hub's HEAD without downloading or repinning — and `model list` prints what each pack pins. Gated repositories authenticate through `LAIR_HF_TOKEN` (fallback `HUGGING_FACE_HUB_TOKEN`), read from the environment only and never logged. Commands: [reference/CLI.md](reference/CLI.md); schema and load-time behavior: [reference/CONFIG.md](reference/CONFIG.md).

### Token budgets, reasoning regions, tool parsing

`max_reasoning_tokens` and `max_response_tokens` are optional per-model ceilings, each also overridable per request; a request override above a declared ceiling is clamped, with the clamp reported in `token_budgets`, and whichever ceiling ends a run is reported as `budget_exhausted` on the response body, the closing stream chunk, and monitor request frames. A reasoning budget binds only a model whose tokenizer carries `<think>`/`</think>` as whole tokens; a model declaring the budget on such a vocabulary is refused at open. `reasoning: keep` streams a recognized reasoning region through as visible content; `reasoning: strip` withholds it from visible content while its boundaries stay recorded. Tool output is resolved once per model at load: `tool_parser: auto` follows the loaded chat template's own capabilities with LAIR's fenced block as fallback, `template` makes the model's own rendering authoritative, `fenced` forces LAIR's block, `none` leaves content byte-for-byte as generated with no `tool_calls`; `force_fenced_tools: true` forces the injected form. Recognized calls arrive as OpenAI-shaped `tool_calls` — execution is client-side. Semantics and mutation classes: [reference/CONFIG.md](reference/CONFIG.md).

### Model families and quantization

Architecture profiles resolved at load: `llama`, `qwen2`, `qwen3`, `gemma2`, `gemma3`, `phi3`, and `olmoe` (sparse-FFN models executed densely with selected experts resident, reported through `/v1/models`), plus a generic fallback that accepts an unregistered architecture only when its tensor table matches the llama-shaped map exactly. SentencePiece vocabularies (Gemma-2, Gemma-3, Phi-3) and BPE profiles are selected by GGUF metadata; chat templates render from GGUF metadata through a minja-parity template engine; declared rope scaling (YaRN and friends), sliding-window/local attention, and MoE descriptors resolve through load-time profiles.

Tensor types the loader accepts and the CPU engine decodes today: `F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_0`, `Q5_K`, `Q6_K`, `Q3_K`, `Q2_K`, `IQ4_NL`, `IQ4_XS`, `IQ3_S`, `IQ3_XXS`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, `Q2_0`. (`IQ1_*`, `TQ*`, `MXFP4`/`NVFP4`, and FP8 containers are not accepted.) Mixed-tier files resolve per tensor, so a file like a `UD-IQ3_XXS` quant loads through the same registry. The blocked-matmul prefill kernel exists only for `Q8_0`, `Q4_K`, and `IQ4_XS`: a `gemm` selection over any other dense format refuses at model open. CUDA hybrid execution covers `F32`, `F16`, `BF16`, `Q4_K`, and `Q6_K` tensors in the opened `placement.gpu_layers` span; an opened span with an unsupported CUDA format refuses rather than falling back to CPU execution.

Measured on this runtime (context 4096, `thread_quota: 8`): `IQ4_XS` decoded 11.6 tok/s against 10.3–10.4 for `Q4_K_M` while being smaller (743 MB against 808 MB), so `IQ4_XS` is the throughput pick under a 4 GB working-set budget. Sub-4-bit i-quants cost roughly 6.5× the `Q4_K` GEMV cost per element (≈1.07–1.09 ns/element against ≈0.163–0.171) and a mixed-tier `IQ3_XXS` file decoded 3.9 tok/s — about 2.6× slower than `Q4_K_M`: they buy bytes you then pay for in tokens. Quality is a separate question from these numbers: the repo's parity fixtures pin `smollm` models only, and a 2-bit-class file has been measured serving coherent-but-wrong arithmetic on a 1B model, so sub-3-bit files are load/parity fixtures here, not serving recommendations.

### Placement and capacity vocabulary

What this build enforces: `core_set` pinning (applied on Windows only; a no-op elsewhere), `thread_quota`, `prefill_lane: shared | dedicated` (a dedicated lane sub-allocates its own worker share and activation buffers), `numa_spread: auto | local | interleave` as a preference resolved against reported topology, `host:` totals accepted with a warning against what the host reports (`dram_mb`, `vram_mb`, `cpu_cores`, optional `disk_path`, and NUMA node overrides), and per-resource `budgets:` chains — `kv` at `dram` is the live KV cap, `kv` at `vram` bounds blocks a device-backed pool holds when a CUDA-backed split opens one, `prefill_buffers` caps narrow the effective prompt-window row width at engine open, `weights` caps record the load-time residency decision. A chain naming a tier no opened pool backs is refused at model open naming what is missing: vocabulary is accepted, capability is reported, never silently dropped. `placement.device` and `placement.prefill` accept `auto`/`cpu` only; `placement.kv_residency` accepts `auto`, `dram`, `vram`, and `split`; `placement.gpu_layers` opens a contiguous CUDA-backed layer split when the selected span uses CUDA-supported formats and a CUDA-capable host is available. Authoritative detail: [reference/CONFIG.md](reference/CONFIG.md).

## Upcoming

None of the following works in this build. Status labels describe project direction, not availability; the "Planned" sections of [reference/CLI.md](reference/CLI.md) and [reference/CONFIG.md](reference/CONFIG.md) list the config keys and CLI shapes that are **not accepted today**.

| Area | Status | What an operator would get |
|---|---|---|
| Whole-model CUDA execution and full CUDA quant coverage | **in progress** | `device: cuda`/`hybrid`, whole-model CUDA execution, CUDA kernels across all loader-accepted quant formats, and GPU compute paths beyond the current `gpu_layers` hybrid split — hybrid layer-split execution through `gpu_layers` is available today for `F32`, `F16`, `BF16`, `Q4_K`, and `Q6_K` |
| ISA dispatch and i-quant kernel rewrite | planned | Sub-4-bit i-quants decoding at `Q4_K`/`IQ4_XS` cost instead of the measured ≈6.5× penalty, with ISA-level dispatch selection |
| Quantized KV format pair | planned | Configurable KV storage formats (`fp32`/`fp16`/`int8`/`k8v4`/`q4`) with in-kernel quantized-dot attention — a larger effective context from the same RAM |
| Embeddings endpoint and multi-model serving (`type: embed`) | planned | `/v1/embeddings` for encoder-only models, and several models resident in one process, each with its own placement policy |
| Sparse-MoE residency management, PLE, MTP drafting | planned | Router-driven expert gather with hot/cold tiers, pageable n-gram tables with their own budgets, speculative drafting with accept-rate reporting. Dense-resident MoE serves today; gather, tiered expert residency, PLE, and drafting do not |
| Residency spanning RAM, VRAM, and NVMe | planned, design still tentative | Weight slabs and KV snapshots demoted across tiers while serving, with live budget rebalancing |
| Vision input on chat models | planned, design still tentative | Image preprocessing and visual-token injection into chat requests; no config shape has been settled |

## Documentation map

- [docs/README.md](README.md) — this page: platform overview and basic operation.
- [reference/CLI.md](reference/CLI.md) — every command, flag, output, exit code; planned CLI surface.
- [reference/CONFIG.md](reference/CONFIG.md) — every config key, defaults, mutation classes; planned configuration axes.
- [../README.md](../README.md) — repository entry point.
- [../DEVELOP.md](../DEVELOP.md) — build, test harnesses, parity gates, contributor detail.
