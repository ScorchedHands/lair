# DEVELOP

## Build

```powershell
dotnet build Lair.sln
```

Target framework: net10.0. Solution layout: `src/Lair.*` projects (Contracts, Loader, Backends.Cpu, Kv, Tokenizer, Templating.Minja, Sampling, Models, Http, Host), `tests/Lair.Tests`, `bench/Lair.Bench`. The continuous-batch scheduler lives in `src/Lair.Models/Scheduling/`; the metrics bus and monitor stream in `src/Lair.Http/Monitoring/`; the config watcher and change-application bus in `src/Lair.Host/ConfigWatcher.cs` + `src/Lair.Http/Config/ConfigReload.cs`. `src/Lair.Templating.Minja` is the managed minja-parity template engine (parser/evaluator/chat wrapper/limits) with no dependency on the HTTP, model, or tokenizer projects.

## Web UI build (`src/Lair.Host/ClientApp/`)

The `/ui` SPA is a TypeScript/React app built with esbuild + Tailwind v4 (pinned deps, committed lockfile; Node 22 / npm 11), living in `src/Lair.Host/ClientApp/` per the ASP.NET Core SPA convention. `dotnet build` auto-runs `npm install`/`npm run build` when client sources changed and stages `ClientApp/dist` into the Host's web root at `wwwroot/ui`, served under `/ui` (root `/` redirects there); `-p:SkipWebUi=true` skips it entirely (no Node needed). Dev loop: `cd src/Lair.Host/ClientApp && npm run watch`. UI smoke harnesses (jsdom): `npm run smoke` (monitor tab), `npm run smoke:chat` (chat tab) — run after bundle-affecting changes.

## Test

```powershell
dotnet test Lair.sln                          # full suite
dotnet test Lair.sln -c Release               # full suite, optimized (required for perf gates)
dotnet test Lair.sln --filter Category=Parity # golden parity vs pinned llama.cpp
dotnet test Lair.sln --filter Category=Kernel # kernel oracle comparisons
```

Test categories: `Loader`, `Tokenizer`, `Template`, `TemplateDiagnostics`, `TemplateParity`, `Serve`, `MinjaSyntax`, `MinjaEval`, `MinjaStatements`, `MinjaFilters`, `MinjaChat`, `MinjaLimits`, `Kernel`, `Layers`, `Kv`, `Sampling`, `Api`, `Cli`, `Config`, `Parity`, `Calib`, `Concurrency`, `Prefill`, `Cache`, `Cancel`, `HotReload`, `Metrics`, `Embeddings`, `Affinity`, `MonitorUi`, `WebUi`.

Performance gates (inter-token latency budgets in `ConcurrencyTests`/`PrefillTests`) are only meaningful under `-c Release` — Debug JIT inflates kernels ~4.6× and the absolute budgets cannot reproduce there. Correctness suites pass in both configurations.

Model fixtures live in `models/` (gitignored). Tests locate them via `ModelFixture.Resolve()` / `ModelFixture.ResolveRepoRelative(path)` and skip gracefully when a file is absent — so on a fresh clone the real-model cases (including Parity) no-op until models are placed. The model budget is 512 MB–1 GB of working set: keep fixture weights small and serve profiles tight (`max_context_tokens`, `kv_blocks`, `thread_quota`), since each loaded engine adds roughly 360–395 MB of runtime overhead on top of weights and KV. Never run two model-loaded processes at once.

## Parity harness

Parity fixtures (`tests/fixtures/parity-smollm-*.json`) record reference continuations + top-5 logprobs from a pinned llama.cpp **server** build (tag `b12032`, asset `llama-b12032-bin-win-cpu-x64.zip`) and LAIR's own calibrated top-16 logprob distribution per reference token. Regenerate with:

```powershell
# 1. Extract the pinned build into tools/llama.cpp/b12032/ (gitignored)
#    download: https://github.com/ggml-org/llama.cpp/releases/download/b12032/llama-b12032-bin-win-cpu-x64.zip

# 2. Regenerate a fixture (spawns llama-server, runs the Calib probe, embeds calibration)
pwsh -File scripts/generate-parity.ps1 -Out tests/fixtures/parity-smollm-f16.json
pwsh -File scripts/generate-parity.ps1 -ModelPath models/smollm-135m-instruct-q8_0.gguf -Out tests/fixtures/parity-smollm-q8.json
```

Parity gate: model-file sha256 guard; LAIR argmax must appear in the reference top-5; LAIR's calibrated distribution must reproduce within |Δlogprob| ≤ 0.2 (calibrated-missing tokens must stay outside LAIR's top-16). Cross-tool exact string equality is not asserted — llama.cpp CLI and server modes disagree on identical greedy runs (see the pinned `reference_mode` field in each fixture).

Template parity is a separate gate: `tests/fixtures/parity-templates-llama32.json` pins byte-exact `/apply-template` renders for `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` (model + chat_template sha256 guards, pinned `date_string`), asserted by `Category=TemplateParity`. Regenerate with:

```powershell
pwsh -File scripts/generate-template-parity.ps1
```

Tokenizer parity is a third gate: `tests/fixtures/parity-tokenizer-smollm.json` and `tests/fixtures/parity-tokenizer-llama32.json` pin `/tokenize` id sequences (`add_special=false`, `parse_special=false`) for both real fixtures, with model sha256 and `tokenizer.ggml.pre` guards, asserted id-for-id by `Category=Tokenizer`. Regenerate with:

```powershell
pwsh -File scripts/generate-tokenizer-parity.ps1
pwsh -File scripts/generate-tokenizer-parity.ps1 -ModelPath models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -Out tests/fixtures/parity-tokenizer-llama32.json
```

`CalibrationProbeTests` (Category `Calib`) are silent unless `LAIR_CALIB_MODEL` and `LAIR_CALIB_OUT` are set; the generator script uses them to embed calibration data.

## Bench

```powershell
dotnet run --project bench/Lair.Bench -- <args>
```

Kernel microbenchmarks (fused dequant GEMV vs scalar oracle, thread scaling).

## Manual smoke

```powershell
copy lair.example.yaml lair.yaml   # point at a real model
dotnet run --project src/Lair.Host -- serve -c lair.yaml
curl http://127.0.0.1:8080/healthz
dotnet run --project src/Lair.Host -- chat -c lair.yaml
dotnet run --project src/Lair.Host -- monitor   # live engine stats table from the serving server
```

Editing `lair.yaml` while `serve` runs hot-reloads live knobs (batch width, thread quota, KV budget) and new-request-only knobs; restart-only fields are refused with a log line.
