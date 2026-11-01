# DEVELOP

## Build

```powershell
dotnet build Lair.sln
```

Target framework: net10.0. Solution layout: `src/Lair.*` projects (Core, Loader, Backends.Cpu, Kv, Tokenizer, Sampling, Models, Api, Host), `tests/Lair.Tests`, `bench/Lair.Bench`.

## Test

```powershell
dotnet test Lair.sln                          # full suite
dotnet test Lair.sln --filter Category=Parity # golden parity vs pinned llama.cpp
dotnet test Lair.sln --filter Category=Kernel # kernel oracle comparisons
```

Test categories: `Loader`, `Tokenizer`, `Template`, `Kernel`, `Layers`, `Kv`, `Sampling`, `Api`, `Cli`, `Config`, `Parity`, `Calib`.

Model fixtures live in `models/` (gitignored). Tests locate them via `ModelFixture.Resolve()` / `ModelFixture.ResolveRepoRelative(path)` and skip gracefully when a file is absent — so on a fresh clone the real-model cases (including Parity) no-op until models are placed.

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
```
