# LAIR — User Documentation

LAIR (Local AI Inference Runtime) is a C#/.NET runtime that serves GGUF chat and embedding models on your own machine behind an OpenAI-compatible HTTP API, with a terminal chat REPL, a live CLI monitor, and a browser monitor. It **consumes published GGUF quant files as they are** — it does not convert, requantize, or reformat models. By default it binds loopback only (`127.0.0.1`) and serves **multiple declared model entries per process**, each with its own placement policy. Administration is the config file: there is no admin API and no CLI command that mutates a running server; a live server changes only through its watched `lair.yaml`.

This page is the base overview. The authoritative detail lives in [reference/CLI.md](reference/CLI.md) and [reference/CONFIG.md](reference/CONFIG.md).

## Requirements

- **A `lair` executable.** A release build produces a self-contained single-file executable that carries the .NET runtime, so running LAIR needs nothing else installed. Building it needs the **.NET 10 SDK** (every project targets `net10.0`), x64, plus Node 22 / npm 11 only when the Web UI is bundled (`-p:SkipWebUi=true` builds without it):

  ```powershell
  dotnet publish src/Lair.Host -c Release -r win-x64 --self-contained -p:PublishSingleFile=true -o release
  ```

  `release\lair.exe` ships with `wwwroot\` beside it — the Web UI bundle the host serves at `/ui`. A checkout can also be run without publishing (`dotnet build Lair.sln` then `src\Lair.Host\bin\Release\net10.0\lair.exe`). A prebuilt release artifact is not published yet: no publish profile or CI packaging step is committed.
- **A GGUF model file** on disk, from the published quant ecosystem (for example a Hugging Face GGUF repo), or fetched by LAIR from a pinned `source:` declaration.
- CPU integer-dot kernels select `AVX-VNNI` or `AVX2` through engine-open ISA dispatch, falling back to scalar paths only when neither vector path is available.
- Optional hybrid CUDA execution needs a CUDA-capable NVIDIA driver/runtime and a model whose opened `placement.gpu_layers` span uses the currently covered CUDA weight-GEMV kernel formats (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_0`, `Q5_K`, `Q6_K`, `Q3_K`, `Q2_K`, `IQ4_NL`, `IQ4_XS`, `IQ3_S`, `IQ3_XXS`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, and `Q2_0`).

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
- It does not overlap GPU compute kernels across models resident on the same CUDA device; multi-model serving partitions capacity and admission only.
- It does not offer an admin API or any CLI command that changes a running server; the watched config file is the only write path.
- It does not authenticate requests or multi-tenant: exposure is governed by `server.bind` alone. `placement.device` and `placement.prefill` accept `auto`/`cpu` only; whole-model accelerator placement is not accepted through those keys. Hybrid accelerator execution is requested through `placement.gpu_layers` and only for the CUDA kernel formats supported today.

## What you get today

### OpenAI-compatible HTTP API

`serve` exposes chat, completion, embeddings, model-listing, health, monitor, and UI routes; anything else is a 404:

| Route | Method | Notes |
|---|---|---|
| `/v1/chat/completions` | POST | JSON or SSE (`"stream": true`); `tools`/`tool_choice`, `logprobs`/`top_logprobs`, `presence_penalty`/`frequency_penalty`, `logit_bias`, `seed`, `stop`, `temperature`/`top_p`/`top_k`, `max_tokens` (default 128), `max_reasoning_tokens`/`max_response_tokens` |
| `/v1/completions` | POST | Legacy text completions, JSON or SSE, same sampling fields |
| `/v1/embeddings` | POST | OpenAI embeddings for encoder-only models; accepts a string or array of inputs and answers one vector per input at its request index |
| `/v1/models` | GET | Advertises `context_length`, the byte-dominant `quant`, `architecture`, model `type`, the resolved `tool_parser` and `reasoning` policy, sparse-FFN strategy, concurrency/queue posture, encoder `pooling`/`normalize` posture when present, and the resolved route segments |
| `/monitor/stream` | GET | SSE engine snapshots (every 2 s) plus request lifecycle events; consumed by `monitor` and the Web UI; integer-dot CPU serves report resolved `metrics_cpu_isa` and `metrics_cpu_gather`; encoder-only engines publish active/completed encode counters and encoder throughput; CUDA-backed serves also publish GPU-busy, PCIe transfer totals, and persistent device VRAM bytes |
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

Requests are not one-request-at-a-time. The scheduler advances many sequences in shared decode steps — one pass over every dense weight matrix serves up to 16 sequences per step — and prompts are prefilled in chunks rather than whole: `engine.max_batch_tokens` (default 512) bounds the largest chunk, and `engine.prefill_step_budget_ms` lets the scheduler size each future chunk from measured step cost. Prompt windows run in sub-batches of a resolved **row** count (measured default 32, engine ceiling 64); no config key sets that number today. `engine.prefill_kernel` selects `gemv` (batched row window), `tiled` (row-tiled causal attention), or `gemm` (blocked matmul over dense weights); a `gemm` selection a served file's weight formats cannot run is refused at model open naming them rather than falling back. KV state is paged in fixed 16-token blocks, and requests sharing a prompt prefix reuse the cached blocks instead of re-prefilling them — the monitor's `cache hit` column is that share. Operator effect: shared weight traversals amortize cost across concurrent requests, chunked prefill keeps a long prompt from starving short ones, and prefix reuse cuts prefill work on repeated conversations. When `placement.gpu_layers` selects a supported layer span, weight-GEMV work for those opened segments runs through the CUDA backend while the remaining segments stay CPU-backed; CUDA dense-matmul coverage now spans many loader-accepted formats in the opened span; CUDA device-step prefill composition with device-resident KV is landed as a non-default internal seam with mixed measured payoff, so default hybrid serving still uses host-side attention on its public path. Knobs: [reference/CONFIG.md](reference/CONFIG.md).

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

Tensor types the loader accepts and the CPU engine decodes today: `F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_0`, `Q5_K`, `Q6_K`, `Q3_K`, `Q2_K`, `IQ4_NL`, `IQ4_XS`, `IQ3_S`, `IQ3_XXS`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, `Q2_0`. (`IQ1_*`, `TQ*`, `MXFP4`/`NVFP4`, and FP8 containers are not accepted.) Mixed-tier files resolve per tensor, so a file like a `UD-IQ3_XXS` quant loads through the same registry. On CPU-only routes the blocked-matmul prefill kernel remains limited to `Q8_0`, `Q4_K`, `IQ4_XS`, and `Q6_K`; CUDA hybrid spans add measured GEMM formats, while unsupported dense formats refuse naming them. CUDA hybrid weight-GEMV covers every loader-accepted GGUF tensor type LAIR loads today (`F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_0`, `Q5_K`, `Q6_K`, `Q3_K`, `Q2_K`, `IQ4_NL`, `IQ4_XS`, `IQ3_S`, `IQ3_XXS`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, and `Q2_0`) in the opened `placement.gpu_layers` span; an opened span whose backend reports no GEMV kernel for a selected format refuses rather than falling back to CPU execution; CUDA blocked matmul now covers measured dense/quantized formats while sub-4-bit i-quants remain refused.

Measured on this runtime (context 4096, `thread_quota: 8`): `IQ4_XS` remains the throughput pick — monitor-windowed decode peaks at ≈16.5 tok/s against ≈12.4 for `Q4_K_M`, with peak working sets of ≈1344 MB and ≈1431 MB respectively, so it is faster while staying under a 4 GB ceiling. Sub-4-bit GEMV costs are no longer the old ≈6.5× penalty: settled samples put them at roughly 0.25–0.39 ns/element against `Q4_K`'s ≈0.164 (about 1.5–2.3×, with `IQ3_XXS` modestly above the intended 2× band), and a mixed-tier `UD-IQ3_XXS` file now peaks around 13 tok/s rather than ≈3.9 — still slower than either dense baseline here because live throughput is fixture-level output behavior, not a per-format kernel measurement. The runs record their actual auto-resolved grid mode; this build has no public encoding override. Quality remains separate: parity fixtures pin `smollm` models only, and a 2-bit-class file has been measured serving coherent-but-wrong arithmetic on a 1B model, so sub-3-bit files are load/parity fixtures here, not serving recommendations.

### Placement and capacity vocabulary

What this build enforces: `core_set` pinning (applied on Windows only; a no-op elsewhere), `thread_quota`, `prefill_lane: shared | dedicated` (a dedicated lane sub-allocates its own worker share and activation buffers), `numa_spread: auto | local | interleave` as a preference resolved against reported topology, `host:` totals accepted with a warning against what the host reports (`dram_mb`, `vram_mb`, `cpu_cores`, optional `disk_path`, and NUMA node overrides), and per-resource `budgets:` chains — `kv` at `dram` is the live KV cap, `kv` at `vram` bounds blocks a device-backed pool holds when a CUDA-backed split opens one, `prefill_buffers` caps narrow the effective prompt-window row width at engine open, `weights` caps record the load-time residency decision. A chain naming a tier no opened pool backs is refused at model open naming what is missing: vocabulary is accepted, capability is reported, never silently dropped. `placement.device` and `placement.prefill` accept `auto`/`cpu` only; `placement.kv_residency` accepts `auto`, `dram`, `vram`, and `split`; `placement.gpu_layers` opens a contiguous CUDA-backed layer split when the selected span uses CUDA-supported formats and a CUDA-capable host is available. Authoritative detail: [reference/CONFIG.md](reference/CONFIG.md).

### KV cache format posture

The default KV cache posture remains `f32/f32`. CPU host-resident quantized KV pairs are available through `models[].kv.quant.{k,v}` for `q8_0/q8_0` — also spelled `q8`, `int8`, or `k8v8` on the K side — and `q8_0/q4_0`, the named asymmetric posture `k8v4`. Those pairs serve only with `engine.prefill_kernel: gemv`, the shared prefill lane, host `dram` KV residency, and no `placement.gpu_layers`; other quantized pairs, device-backed KV postures, and `q4_0/q4_0` refuse at model open rather than silently falling back to FP32. Device-side quantized KV attention is not available in this build and lands with the CUDA transformer-step work.

## Upcoming

Some of the following is available only in part where the row says so; status labels describe project direction, not full availability. The "Planned" sections of [reference/CLI.md](reference/CLI.md) and [reference/CONFIG.md](reference/CONFIG.md) list the config keys and CLI shapes that are **not accepted today**.

| Area | Status | What an operator would get |
|---|---|---|
| Whole-model CUDA execution and device-side transformer work | **in progress** | weight-GEMV kernels across all loader-accepted quant formats are available today through hybrid `gpu_layers` layer-split execution; CUDA hybrid execution now covers measured CUDA GEMM formats for dense/quantized tensors; explicit default whole-model `device: cuda` serving remains unshipped; CUDA dense prefill composition with on-device attention/KV is landed as a non-default internal seam only. |
| Quantized KV CUDA attention | in progress | CPU host-resident quantized KV pairs (`q8_0/q8_0` and `q8_0/q4_0`) are available today; CUDA per-combo quantized KV attention is landed as an internal/tested seam for those pairs, while public device-backed quantized KV serving remains unshipped. |
| Embeddings endpoint and multi-model serving (`type: embed`) | shipped | `/v1/embeddings` for encoder-only models, several `chat`/`embed` model entries resident in one process with per-model placement policy; GPU compute kernels across multiple CUDA-resident models still serialize through the single primary context. |
| Sparse-MoE residency management, PLE, MTP drafting | planned | Router-driven expert gather with hot/cold tiers, pageable n-gram tables with their own budgets, speculative drafting with accept-rate reporting. Dense-resident MoE serves today; gather, tiered expert residency, PLE, and drafting do not |
| Residency spanning RAM, VRAM, and NVMe | planned, design still tentative | Weight slabs and KV snapshots demoted across tiers while serving, with live budget rebalancing |
| Vision input on chat models | planned, design still tentative | Image preprocessing and visual-token injection into chat requests; no config shape has been settled |

## Documentation map

- [docs/user/README.md](README.md) — this page: platform overview and basic operation for operators.
- [reference/CLI.md](reference/CLI.md) — every command, flag, output, exit code; planned CLI surface.
- [reference/CONFIG.md](reference/CONFIG.md) — every config key, defaults, mutation classes; planned configuration axes.
- [../engine/README.md](../engine/README.md) — engine documentation: how the implemented engine works internally (pipeline stages, kernels and backends, KV cache, residency and budgets, model profiles, comparisons with other engines).
- [../README.md](../README.md) — repository entry point.
- [../DEVELOP.md](../DEVELOP.md) — build, test harnesses, parity gates, contributor detail.
