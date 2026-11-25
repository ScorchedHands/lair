# Dense CPU traversal optimization report 2026-11-25T01-23-00Z

Model: `C:\Users\Chuck\AppData\Local\Temp\kilo\optimize-small-cpu\models\qwen2.5-3b-instruct-q4_k_m.gguf`
Lane: local CPU lane, `threads=32`, Qwen2.5-3B Q4_K_M, no CUDA benchmarking.

## llama.cpp reference rerun (`rerun-llama-bench-t32.txt`)
```text
pp128 = 265.35 ± 6.62 tok/s
pp256 = 164.06 ± 112.40 tok/s
tg32 = 13.79 ± 1.14 tok/s
tg64 = 11.22 ± 4.15 tok/s
```

The first llama.cpp run in `llama-bench-t32.txt` was noisy (`pp256=134.75 ± 110.79`, `tg32=8.10 ± 5.86`), so the rerun table is the cleaner paired reference.

## LAIR unprofiled prefill after optimization (`lair-prefill-unprofiled.txt`)
Command posture:
```text
prefill-profile --threads 32 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 3
```

Results:
```text
gemv length128 = 66.00 tok/s, length256 = 47.15 tok/s
tiled length128 = 48.30 tok/s, length256 = 41.96 tok/s
gemv weightFit = 579.08 ms/traversal, impliedWeightGB/s = 2.88
tiled weightFit = 1123.18 ms/traversal, impliedWeightGB/s = 1.49
```

Previous committed unprofiled baseline at the same posture was:
```text
gemv length128 = 25.31 tok/s, length256 = 22.76 tok/s
tiled length128 = 25.18 tok/s, length256 = 22.55 tok/s
weightFit ≈ 2244 ms/traversal, impliedWeightGB/s ≈ 0.74
```

This is a major local prefill improvement, roughly 2.5-3x versus the committed baseline and enough to move the dense weight traversal from about 0.74 GB/s to about 2.88 GB/s for the best gemv posture.

## LAIR production decode baseline
First run:
```text
prefillSetupMs = 5777.2, median decode = 5.299 tok/s, spread = 1.50%
```

Rerun:
```text
prefillSetupMs = 7474.1, median decode = 5.668 tok/s, spread = 13.07%
```

Decode remains noisy and far below llama.cpp `tg32` in this session. The landed optimization affects batched dense traversal paths; the single-token `DecodeNext` path is untouched, so production decode parity is not solved by this commit.

## Validation
Main re-ran:
```text
dotnet build -c Release
dotnet build bench/Lair.Bench/Lair.Bench.csproj -c Release
dotnet test tests/Lair.Tests -c Release --no-build --filter "FullyQualifiedName~DenseWeightFusionTests|Category=Kernel&FullyQualifiedName~BatchKernelTests"
dotnet test tests/Lair.Tests -c Release --no-build --filter "(Category=Prefill|Engine|Kv|Attention|Kernel|Serve)&Category!=Cuda" -> 682/682 passed
```

## Interpretation
The dense CPU traversal optimization is a real local win for batched prefill paths and dense batched FF traversal. It is not yet llama.cpp parity, especially for decode. The next CPU optimization target is the single-token dense decode GEMV traversal path, because production decode remains dominated by untouched single-vector weight traversal and remains around 5-6 tok/s versus llama.cpp 11-14 tok/s in this noisy session.
