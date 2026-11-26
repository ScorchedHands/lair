# Prefill gate/up fused pair batch dot 2026-11-25T13-45-35Z

Model: `C:\Users\Chuck\AppData\Local\Temp\kilo\optimize-small-cpu\models\qwen2.5-3b-instruct-q4_k_m.gguf`
Lane: local CPU lane, `threads=32`, `rowWidth=64`, Qwen2.5-3B Q4_K_M, no CUDA benchmarking.
Baseline arm: the worktree's pre-item code (attention-kernel vectorization + substage profiling intact, fused pair = materialize A → batch dot A → materialize B → batch dot B). Candidate arm: the new pair batch dot.

What changed: `CpuBackend+FusedPair.cs` gains a pair batch dot route. One row pair materializes both rows side by side in a per-thread pair scratch, then one traversal of the activation batch updates both matrices' accumulator chains per chunk: each group's input vector is loaded once and fed to the A and B FMAs, instead of the batch being walked once per matrix. Chain grouping comes from a new posture-resolved `CpuKernelTuning.PairChainGroup` (auto = half the resolved chain group, capped at `PairChainGroupMax = ChainGroupMax / 2 = 4`: four A chains + four B chains + two weight chunks fit AVX2's sixteen YMM registers; a pinned narrower chain group pairs at half its width and runs through a stack-staged pair kernel). The pair route keeps the existing single-row fused route as fallback when a pair of staged rows would no longer fit the scratch ceiling (`2 x cols > max_scratch_cols`). `HSum`, chunk order, and chain grouping semantics are untouched; grouping only chooses which batch vectors share one pass, so every route lands bit-identical to two independent `GemvBatch` calls.

Files: `src/Lair.Backends.Cpu/CpuKernelTuning.cs`, `src/Lair.Backends.Cpu/CpuBackend+FusedPair.cs`, `tests/Lair.Tests/DenseWeightFusionTests.cs` (+9 cases), `tests/Lair.Tests/CpuTuningTests.cs` (+6 cases).

Raw outputs: `ab-stage-profile.txt` (12 alternating single-rep stage-profile runs: baseline vs new, gemv and tiled, lengths 128/256, 3 rounds), `ab-throughput.txt` (12 alternating unprofiled `--reps 3` runs, same shapes), `probe-new-gemv.txt` (first-pass new-arm observation).

## Method

This host is very noisy between runs (the attention report recorded ±40% substage swings for unchanged code), so the committed reports are reference only. The real comparison is an alternating A/B: the pre-item code and the new code were built as two bench binaries and run interleaved, 3 rounds, stage-profile and unprofiled throughput. Medians below.

## feed_forward_gate_up_ms medians (same-session A/B, 3 rounds)

```text
gemv  length128  707.2 -> 502.2   (-29.0%)
gemv  length256  1399.6 -> 1048.2 (-25.1%)
tiled length128  619.1 -> 428.4   (-30.8%)
tiled length256  1367.7 -> 956.9  (-30.0%)
```

Every new sample beats every baseline sample at the same shape and kernel (e.g. gemv 256: baseline min 1355.0 > new max 1071.5). The gate/up ratio vs `feed_forward_down_ms` moves from about 1.45-1.5x (the gap the substage report flagged) to about 1.0-1.1x: the batched pair now gets the per-matrix efficiency the down pass gets.

Reference from the committed substage report (pre-attention-vectorization): gemv `679.8`/`1508.5`, tiled `769.3`/`1323.1`. The baseline re-measurements here (`622-733`/`1355-1499`/`617-639`/`1318-1437`) sit in the same band.

## Window total_ms medians (stage-profile)

```text
gemv  length128  2063.8 -> 1945.0  (-5.8%)
gemv  length256  4531.3 -> 4548.2  (+0.4%, inside run noise)
tiled length128  1593.1 -> 1410.5  (-11.5%)
tiled length256  4246.3 -> 3723.6  (-12.3%)
```

## Throughput medians (unprofiled, --reps 3, batchedTok/s)

```text
gemv  length128  77.8 -> 87.5  (+12.5%)
gemv  length256  65.0 -> 77.3  (+18.9%)
tiled length128  63.2 -> 83.7  (+32.4%)
tiled length256  73.7 -> 68.7  (-6.8%, inside run noise: new samples 60.1-77.2 vs baseline 55.3-76.2 overlap fully)
```

Committed reference (attention-kernel report medians): gemv `63.66`/`56.66`, tiled `58.48`/`52.83` tok/s. This session's baseline arm already runs above those medians (host state differs), so only the same-session A/B columns are direction-credible.

## Commands

```text
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel gemv --reps 1 --stage-profile
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel tiled --reps 1 --stage-profile
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel gemv --reps 3
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel tiled --reps 3
```

The A/B files were produced by running the two binaries alternately with the same arguments (`basebin` = HEAD fused-pair code, `newbin` = item code, built under the temp bench-compare dirs, not committed).

## Interpretation

- The pair batch dot is a real and consistent win on the gate/up substage: −25% to −31% across all four shapes, with every new sample beating every baseline sample. Halving the activation-batch walks per row pair is the mechanism; the weight traversal itself is unchanged.
- gate/up now costs about the same as down at every shape, closing the 1.4-1.5x traversal gap the substage report called out. gate/up remains the largest weight traversal (24-27% of the profiled window at 256, 22-26% at 128), so the remaining weight-traversal target is the down pass and the pair's own weight side (each row pair still reads both packed rows once — the win was on the activation side only).
- Window totals follow at 128 (gemv −5.8%, tiled −11.5%) and at tiled 256 (−12.3%); gemv 256 lands inside noise because the attention kernel substage dominates there and swings run-to-run.
- Throughput medians improve on three of four shapes; tiled 256 is noise-indeterminate. The stage-profile gate/up columns are the credible signal on this host.

## Validation

```text
dotnet build Lair.sln -c Release --nologo -> 0 warnings, 0 errors
dotnet test tests/Lair.Tests -c Release --no-build --filter "FullyQualifiedName~DenseWeightFusionTests|FullyQualifiedName~CpuTuningTests|Category=Kernel|BatchKernel" -> 546 total: 545 passed, 1 failed (a CUDA-tagged Kernel test, see below)
dotnet test ... --filter "FullyQualifiedName~DenseWeightFusionTests|FullyQualifiedName~CpuTuningTests|(Category=Kernel|BatchKernel)&Category!=Cuda" -> 403/403 passed
dotnet test tests/Lair.Tests -c Release --no-build --filter "(Category=Prefill|Engine|Kv|Attention|Kernel|Serve|Config|HotReload)&Category!=Cuda" -> 932/932 passed, 0 failed, 0 skipped (5m36s)
```

The one failure in the CUDA-inclusive filter is `CudaBlockQuantGemvTests.CudaQ2_0_Gemv_HostMappedPosture_RunsFromRegisteredHostBytes` (trait `Kernel` + `Cuda`): it passes in isolation and is pre-existing CUDA-lane band flakiness, unrelated to this item (CUDA tests are out of scope here and every required gate excludes `Category=Cuda`).

New tests prove: pair route vs independent `GemvBatch` calls bit-identical for Q4_K/Q4_K, Q4_K/Q6_K, Q6_K/Q6_K at ragged counts (8/9/16/32); pair route vs single-row fused fallback bit-identical (pair scratch ceiling pinned between one row and a pair of rows); pinned narrower pair groups (chain group 6 -> pair 3, 2 -> 1) bit-identical through the stack-staged pair kernel; threaded pair route bit-identical to the single-threaded pair route under forced tiny row blocks; `PairChainGroup` resolution follows the resolved chain group and vector posture and stays inside the pair register budget.
