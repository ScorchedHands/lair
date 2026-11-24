# Small Model CPU Baseline Report

Report prefix convention: `optimize-small-model-cpu-baseline-2026-11-24T21-09-07Z`

## Host

- OS: Microsoft Windows 11 Pro
- CPU: 13th Gen Intel Core i9-13900K, 24 physical cores / 32 logical processors reported by WMI; `[Environment]::ProcessorCount = 32`.
- Total physical memory: `68423139328` bytes.
- Benchmark thread posture used for these baselines: `threads=8`.

## Model

- File: `C:\Users\Chuck\AppData\Local\Temp\kilo\optimize-small-cpu\models\qwen2.5-3b-instruct-q4_k_m.gguf`
- SHA/identity source: official Hugging Face file list for `Qwen/Qwen2.5-3B-Instruct-GGUF`, path `qwen2.5-3b-instruct-q4_k_m.gguf`.
- File bytes on disk: `2104932768`
- llama.cpp reported model facts: `qwen2 3B Q4_K - Medium`, size `1.95 GiB`, params `3.40 B`.
- LAIR geometry: `arch=qwen2 layers=36 hidden=2048 heads=16 kvHeads=2 headDim=128 vocab=151936`.

## llama.cpp Reference Baseline

Binary extraction root used locally:
`C:\Users\Chuck\AppData\Local\Temp\kilo\optimize-small-cpu\tools\llama-b12359-bin-win-cpu-x64`

Command shape:
```text
llama-bench.exe -m <model> -ngl 0 -t 8 -p 128,256 -n 32,64 -r 3
```

Result table captured in `llama.cpp.txt`:

| model | size | params | backend | threads | test | t/s |
|---|---:|---:|---|---:|---:|---:|
| qwen2 3B Q4_K - Medium | 1.95 GiB | 3.40 B | CPU | 8 | pp128 | 188.02 ± 7.27 |
| qwen2 3B Q4_K - Medium | 1.95 GiB | 3.40 B | CPU | 8 | pp256 | 195.40 ± 5.16 |
| qwen2 3B Q4_K - Medium | 1.95 GiB | 3.40 B | CPU | 8 | tg32 | 20.31 ± 1.18 |
| qwen2 3B Q4_K - Medium | 1.95 GiB | 3.40 B | CPU | 8 | tg64 | 21.32 ± 0.14 |

llama.cpp build reported: `b1b770d90 (12359)`.
Loaded backend during run: `ggml-cpu-alderlake.dll`; RPC backend was loaded by the binary but not used for this CPU measurement.

## LAIR Prefill Profile Baseline

Command shape:
```text
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 8 --lengths 128,256 --kernel gemv,tiled --reps 3
```

Raw output captured in `lair-prefill-profile.txt`.

Key facts:

- LAIR resolved AVX2 posture.
- Traversal bytes reported by the profile: `traversalBytes=1668685824`, `outputHeadBytes=255252480`.
- Process peak working set across this LAIR prefill-profile process run: `peakWorkingSetMb=1966.2`.

Kernel results:

| kernel | length | subBatches | batchedMs | batchedTok/s | weightMs | attentionMs | impliedWeightGB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemv | 128 | 4 | 9385.7 | 13.64 | 9065.5 | 320.2 | 0.74 |
| gemv | 256 | 8 | 19406.8 | 13.19 | 18131.0 | 1275.8 | 0.74 |
| tiled | 128 | 4 | 8477.7 | 15.10 | 8164.4 | 313.3 | 0.82 |
| tiled | 256 | 8 | 17577.3 | 14.56 | 16328.8 | 1248.5 | 0.82 |

Initial comparison: llama.cpp prompt processing is roughly `12x` faster than LAIR at these short bounded windows on this machine/thread posture. The LAIR profile identifies the weight traversal term as dominant for Qwen2.5-3B prefill, not attention.

## LAIR Decode Baseline Through Production CPU KV Seam

Commands:
```text
dotnet run --project bench/Lair.Bench -c Release --no-build -- kv-quant <model> --threads 8 --context 256 --decode 32 --windows 3 --modes baseline
dotnet run --project bench/Lair.Bench -c Release --no-build -- kv-quant <model> --threads 8 --context 512 --decode 64 --windows 3 --modes baseline
```

Raw outputs captured in `lair-kv-quant-context256.txt` and `lair-kv-quant-context512.txt`.

Context 256 / decode 32:

- Baseline storage: `f32/f32`.
- Attention route: production span<float> kernel.
- Prefill setup outside timed windows: `prefillSetupMs=19293.5` for a 256-token prompt.
- Reserved KV footprint: `reservedKvMiB=31.50`.
- Warm window tok/s: `6.133`.
- Timed decode windows median: `6.312 t/s`, min `6.308`, max `6.380`.
- Process peak working set: `peakWorkingSetMb=1986.6`.

Context 512 / decode 64:

- Baseline storage: `f32/f32`.
- Attention route: production span<float> kernel.
- Prefill setup outside timed windows: `prefillSetupMs=41988.8` for a 512-token prompt.
- Reserved KV footprint: `reservedKvMiB=58.50`.
- Warm window tok/s: `5.697`.
- Timed decode windows median: `5.568 t/s`, min `5.447`, max `5.690`.
- Process peak working set: `peakWorkingSetMb=2014.1`.

Initial comparison: llama.cpp generation throughput is roughly `3x` faster than LAIR's production CPU decode seam for this small model/thread posture. The `kv-quant` harness forces a fixed continuation token per step; it measures the decode pipeline cost, not full generated-token parity.

## Initial Interpretation

The first cross-backend optimization target should be prefill/prompt processing on Qwen2.5-3B: LAIR is already closer to llama.cpp for decode than for prefill in this baseline set. The current LAIR `prefill-profile` fit says short-window prefill cost is dominated by weight traversal, so the next focused step should compare LAIR CPU dense GEMV/GemvBatch/tiled staging against llama.cpp's batched prompt processing path before adding engine-specific stage profiling.
