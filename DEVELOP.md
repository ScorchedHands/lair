# DEVELOP

## Build

```powershell
dotnet build Lair.sln
```

Target framework: net10.0. Solution layout: `src/Lair.*` projects (Contracts, Loader, Backends.Cpu, Backends.Cuda, Kv, Tokenizer, Templating.Minja, Sampling, Models, Http, Host), `tests/Lair.Tests`, `bench/Lair.Bench`. The continuous-batch scheduler lives in `src/Lair.Models/Scheduling/`; the metrics bus and monitor stream in `src/Lair.Http/Monitoring/`; accelerator accounting for CUDA-backed serves lives in `src/Lair.Http/Monitoring/AcceleratorAccounting.cs` + `src/Lair.Backends.Cuda/CudaBackend+Accounting.cs`; the config watcher and change-application bus in `src/Lair.Host/ConfigWatcher.cs` + `src/Lair.Http/Config/ConfigReload.cs`. `src/Lair.Templating.Minja` is the managed minja-parity template engine (parser/evaluator/chat wrapper/limits) with no dependency on the HTTP, model, or tokenizer projects.

## Web UI build (`src/Lair.Host/ClientApp/`)

The `/ui` SPA is a TypeScript/React app built with esbuild + Tailwind v4 (pinned deps, committed lockfile; Node 22 / npm 11), living in `src/Lair.Host/ClientApp/` per the ASP.NET Core SPA convention. `dotnet build` auto-runs `npm install`/`npm run build` when client sources changed and stages `ClientApp/dist` into the Host's web root at `wwwroot/ui`, served under `/ui` (root `/` redirects there); `-p:SkipWebUi=true` skips it entirely (no Node needed). Dev loop: `cd src/Lair.Host/ClientApp && npm run watch`. UI smoke harnesses (jsdom): `npm run smoke` (monitor tab), `npm run smoke:chat` (chat tab) — run after bundle-affecting changes.

## Test

```powershell
dotnet test Lair.sln                          # full suite
dotnet test Lair.sln -c Release               # full suite, optimized (required for perf gates)
dotnet test Lair.sln --filter Category=Parity # golden parity vs pinned llama.cpp
dotnet test Lair.sln --filter Category=Kernel # kernel oracle comparisons
```

Test categories: `Loader`, `Moe`, `Tokenizer`, `Template`, `TemplateDiagnostics`, `TemplateParity`, `Serve`, `MinjaSyntax`, `MinjaEval`, `MinjaStatements`, `MinjaFilters`, `MinjaChat`, `MinjaLimits`, `Kernel`, `Layers`, `Kv`, `Sampling`, `Api`, `Cli`, `Config`, `ToolCalls`, `Parity`, `Calib`, `Concurrency`, `Prefill`, `Cache`, `Cancel`, `HotReload`, `Metrics`, `Scheduling`, `Budget`, `Embeddings`, `Affinity`, `MonitorUi`, `WebUi`, `Cuda`, `MonitorCounters`, `Parity-Hybrid`.

Model acquisition gates run through the hub transport seam with local or mocked stand-ins (`Category=Loader|Config|Api|Metrics|Cli`); no automated test requires live network.

Performance gates (inter-token latency budgets in `ConcurrencyTests`/`PrefillTests`) are only meaningful under `-c Release` — Debug JIT inflates kernels ~4.6× and the absolute budgets cannot reproduce there. Correctness suites pass in both configurations. The real-model suites (`ConcurrencyTests`, `ParityTests`, and the pinned Qwen3 reasoning serve suite) share one serialized xunit collection, `model-loaded`, so two engines never load side by side — a second engine loading alongside the latency-budget suite starves its inter-token budgets.

Model fixtures live in `models/` (gitignored). Tests locate them via `ModelFixture.Resolve()` / `ModelFixture.ResolveRepoRelative(path)` and skip gracefully when a file is absent — so on a fresh clone the real-model cases (including Parity) no-op until models are placed. The model budget is a 4 GB peak working set per loaded engine: a loaded engine adds roughly 360–395 MB of runtime overhead on top of weights and KV, so keep fixture weights under about 3.6 GB and serve profiles as tight as the case needs (`max_context_tokens`, `kv_blocks`, `thread_quota`). Never run two model-loaded processes at once.

## Parity harness

Parity fixtures (`tests/fixtures/parity-smollm-*.json`) record reference continuations + top-5 logprobs from a pinned llama.cpp **server** build (tag `b12032`, asset `llama-b12032-bin-win-cpu-x64.zip`) and LAIR's own calibrated top-16 logprob distribution per reference token. Regenerate with:

```powershell
# 1. Extract the pinned build into tools/llama.cpp/b12032/ (gitignored)
#    download: https://github.com/ggml-org/llama.cpp/releases/download/b12032/llama-b12032-bin-win-cpu-x64.zip

# 2. Regenerate a fixture (spawns llama-server, runs the Calib probe, embeds calibration)
pwsh -File scripts/generate-parity.ps1 -Out tests/fixtures/parity-smollm-f16.json
pwsh -File scripts/generate-parity.ps1 -ModelPath models/smollm-135m-instruct-q8_0.gguf -Out tests/fixtures/parity-smollm-q8.json
```

The YaRN fixture runs on a rope-scaling-patched copy of the pinned Qwen3 file, produced by a header-only patcher (tensor data verified byte-identical, run idempotent):

```powershell
pwsh -File scripts/patch-gguf-rope-scaling.ps1
pwsh -File scripts/generate-parity.ps1 -ModelPath models/Qwen3-0.6B-Q8_0-YaRN.gguf -Out tests/fixtures/parity-qwen3-yarn.json
```

The SentencePiece families' logit fixtures run on the pinned files directly:

```powershell
pwsh -File scripts/generate-parity.ps1 -ModelPath models/gemma-2-2b-it.Q4_K_M.gguf -Out tests/fixtures/parity-gemma2.json
pwsh -File scripts/generate-parity.ps1 -ModelPath models/Phi-3-mini-4k-instruct-Q2_K.gguf -Out tests/fixtures/parity-phi3.json
```

Parity gate: model-file sha256 guard; LAIR argmax must appear in the reference top-5; LAIR's calibrated distribution must reproduce within |Δlogprob| ≤ 0.2 (calibrated-missing tokens must stay outside LAIR's top-16). The cross-tool logprob band applies only to candidates within 6 nats of the reference's own top-1; deeper tail candidates are gated on membership in LAIR's calibrated window and reproduction of LAIR's calibrated value instead of absolute agreement. Cross-tool exact string equality is not asserted — llama.cpp CLI and server modes disagree on identical greedy runs (see the pinned `reference_mode` field in each fixture).

Template parity is a separate gate: `tests/fixtures/parity-templates-llama32.json` pins byte-exact `/apply-template` renders for `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` (model + chat_template sha256 guards, pinned `date_string`), asserted by `Category=TemplateParity`. Regenerate with:

```powershell
pwsh -File scripts/generate-template-parity.ps1
```

Tokenizer parity is a third gate: `tests/fixtures/parity-tokenizer-smollm.json`, `tests/fixtures/parity-tokenizer-llama32.json`, `tests/fixtures/parity-tokenizer-gemma2.json`, and `tests/fixtures/parity-tokenizer-phi3.json` pin `/tokenize` id sequences (`add_special=false`, `parse_special=false`) for the real fixtures, with model sha256 guards (and a `tokenizer.ggml.pre` guard on the BPE fixtures), asserted id-for-id by `Category=Tokenizer`; the SentencePiece fixtures also assert `Decode(Encode(content)) == content`. Regenerate with:

```powershell
pwsh -File scripts/generate-tokenizer-parity.ps1
pwsh -File scripts/generate-tokenizer-parity.ps1 -ModelPath models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -Out tests/fixtures/parity-tokenizer-llama32.json
pwsh -File scripts/generate-tokenizer-parity.ps1 -ModelPath models/gemma-2-2b-it.Q4_K_M.gguf -Out tests/fixtures/parity-tokenizer-gemma2.json
pwsh -File scripts/generate-tokenizer-parity.ps1 -ModelPath models/Phi-3-mini-4k-instruct-Q2_K.gguf -Out tests/fixtures/parity-tokenizer-phi3.json
```

`CalibrationProbeTests` (Category `Calib`) are silent unless `LAIR_CALIB_MODEL` and `LAIR_CALIB_OUT` are set; the generator script uses them to embed calibration data.

## Bench

```powershell
dotnet run --project bench/Lair.Bench -- <args>
```

Kernel microbenchmarks (fused dequant GEMV, blocked matmul, and scalar oracle comparisons, thread scaling). `gemv` runs the synthetic Q4_K control; `gguf-dump <path>` prints a file's tensor table; `gemv-file <gguf path> [--threads N] [--reps R] [--max-rows M] [--grid scalar|gather|compare] [--prefetch on|off|compare]` times real GGUF bytes through the mmap weight source (defaults: quota 8, reps 150, max 1024 rows per tensor) and prints both mapping-timed numbers and a private-copy pass over the identical bytes. `prefill-profile <gguf path> [--threads N] [--lengths L1,L2,…] [--widths W1,W2,…] [--kernel gemv,tiled,gemm] [--reps R] [--per-token] [--out <path>]` times prompt windows through the selected prefill route over a fresh KV per measurement — `--per-token` adds the single-token forward for comparison — and prints a table splitting measured cost into a weight term (ms per weight traversal, implied GB/s) and an attention term (µs per position pair) with the crossover length where attention overtakes the weight term. `--widths` opens one engine per candidate resolved row count and reports sub-batch activation bytes; `--kernel` compares runnable prefill kernels or prints a refusal line naming unsupported dense formats instead of falling back. Each run closes with its process peak working set, so committed per-kernel peaks come from separate single-kernel processes. `--out` writes the same table to a file. Runs whose costs stop rising with the causal span print the split as unusable. Quantized KV and CUDA GEMV gates use `kv-spike`, `kv-micro`, `q4-parity`, `kv-quant`, and `cuda-gemv`; `cuda-gemv` sweeps deterministic pinned synthetic GEMV fixtures across supported formats, printing CPU wall arms, CUDA launch wall, device kernel-busy microseconds, ratios, and equivalence sanity.

### Measured GEMV cost

`gemv-file` defaults to quota 8, reps150, and max rows1024; the current sub-4-bit re-pin used explicit `--threads 8 --reps 200 --max-rows 256`. Raw reports are committed under `bench/reports/gemv-mvp5-*-reps200-*.txt`, with the summary in `bench/reports/sub4bit-cost-repin-2026-11-17.md`. Against the Q4_K_M resident single-thread baseline (`335.8 ns/GEMV`, `.164 ns/elt`), settled UD samples land at roughly `.247` for `IQ2_S` (the group summary is inflated by a first-measured warm-up sample to `.270`), `.320` for `IQ2_XS`, `.338` for `IQ3_S`, and `.383` for `IQ3_XXS`: about 1.5–2.3× the baseline rather than the old ≈6.5× penalty, with `IQ3_XXS` modestly above the intended band on this posture.

### Measured live decode throughput

Measured through the documented serve surface: one server at a time, each killed before the next runs and none left running. Config for every model: `max_context_tokens: 4096` with `kv_blocks: 256` (`block_size: 16`), engine `threads: 0` with `placement.thread_quota: 8`; startup logs confirm `compute_grant ... requested=8 granted=8`. Fixed greedy prompt set, three prompts per model, repeated twice after one untimed warm request (`temperature: 0`, `max_tokens: 256`). Completion-token counts differ across quantized models under identical prompts, so client-aggregated tok/s is confounded by output length; server-windowed `metrics_tok_s` is the primary decode-rate comparison. Peak working set is read from `metrics_process_working_set_mb`.

| Model file | Monitor peak tok/s | Client aggregate run1 / run2 | Peak working set MB |
|---|---:|---|---:|
| `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 12.43 | 8.153 / 8.818 (`337` tokens) | 1430.9 |
| `models/Llama-3.2-1B-Instruct-IQ4_XS.gguf` | 16.46 | 14.491 / 15.377 (`239` tokens) | 1344.2 |
| `models/Llama-3.2-1B-Instruct-UD-IQ3_XXS.gguf` | 12.97 | 10.794 / 11.811 (`204` tokens) | 1222.0 |

IQ4_XS remains the throughput pick: it is fastest by monitor-windowed decode while staying well under the 4 GB working-set ceiling. The mixed-tier UD file has a smaller peak working set than either dense baseline but still decodes slower on this runtime, so sub-Q4 files remain a size tradeoff rather than a throughput recommendation.

## Manual smoke

```powershell
copy lair.example.yaml lair.yaml   # point at a real model
dotnet run --project src/Lair.Host -- serve -c lair.yaml
curl.exe http://127.0.0.1:8181/healthz   # the example config sets server.port: 8181
dotnet run --project src/Lair.Host -- chat -c lair.yaml
dotnet run --project src/Lair.Host -- monitor -u http://127.0.0.1:8181 --interval 1000   # live engine stats table from the serving server; monitor needs at least one option
```

For an optional hybrid CUDA proof, point a config at a model whose opened `placement.gpu_layers` span uses supported formats (`F32`, `F16`, `BF16`, `Q4_K`, or `Q6_K`) and run the same serve/monitor commands; nonzero GPU utilization, PCIe transfer bytes/copies, or persistent device VRAM appear as a tail on monitor rows.

Editing `lair.yaml` while `serve` runs hot-reloads live knobs (batch width, thread quota, system-memory KV ceilings) and new-request-only knobs; host capacity totals/node overrides, device-backed KV caps/residency modes, placement posture including `gpu_layers`, weight caps, budget chain orders, block size, model identity, prefill lane/core-set changes are refused with a log line.
