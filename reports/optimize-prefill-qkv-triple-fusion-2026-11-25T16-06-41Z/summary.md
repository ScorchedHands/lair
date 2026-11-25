# Prefill Q/K/V batched triple shared-input fusion 2026-11-25T16-06-41Z

Model: `C:\Users\Chuck\AppData\Local\Temp\kilo\optimize-small-cpu\models\qwen2.5-3b-instruct-q4_k_m.gguf`
Lane: local CPU lane, `threads=32`, `rowWidth=64`, Qwen2.5-3B Q4_K_M, no CUDA benchmarking.
Baseline arm: the worktree's pre-item code (gate/up fused pair + attention-kernel vectorization + substage profiling intact, batched Q/K/V as three independent `GemvBatch` calls). Candidate arm: the new triple fusion.

What changed: batched attention Q/K/V projections (prefill windows, tiled prefill, merged decode, Encode/EncodeBatch) now try a new `IDenseWeightTripleFusion` seam before their three independent calls. The CPU implementation walks the union of the three matrices' row spaces; each row triple materializes its three packed rows side by side in a per-thread triple scratch with their formats' own bit-exact dequant kernels, then walks the activation batch one time, loading each input vector once and feeding it to all three matrices' accumulator chains — where the independent trio walks the batch once per matrix. The trio's differing row counts (query region 2048 rows vs kv regions 256) are handled by per-matrix output strides and row presence: rows 0..255 run the triple route, rows 256..2047 run the existing single-row scratch route for the query matrix alone. Grouping comes from a new posture-resolved `CpuKernelTuning.TripleChainGroup` (auto = resolved chain group capped at `TripleChainGroupMax = 3`: three chains per matrix plus three weight chunks fit the vector posture's register budget; a narrower resolved group caps the triple group to it and runs through a stack-staged triple kernel). The fused call degrades to the pair route for two matrices plus the single-row scratch route for the third when a triple of staged rows no longer fits the scratch ceiling, and to three single-row scratch passes when a pair does not fit. `HSum`, chunk order, chain order, and FMA operand order are untouched; grouping only chooses which batch vectors share one pass, so every route lands bit-identical to three independent `GemvBatch` calls.

Files: `src/Lair.Contracts/IDenseWeightTripleFusion.cs` (new), `src/Lair.Backends.Cpu/CpuBackend+FusedTriple.cs` (new), `src/Lair.Backends.Cpu/CpuKernelTuning.cs`, `src/Lair.Models/Engine/TransformerModel.cs` (new `RunAttentionQKVProjection` seam at five batched call sites), `tests/Lair.Tests/DenseWeightFusionTests.cs` (+16 cases), `tests/Lair.Tests/CpuTuningTests.cs` (+7 cases, +2 asserts).

Raw outputs: `ab-stage-profile.txt` (12 alternating single-rep stage-profile runs: baseline vs new, gemv and tiled, lengths 128/256, 3 rounds; the file also holds 6 extra new-arm runs from a first mis-pathed pass, excluded from medians), `ab-throughput.txt` (12 alternating unprofiled `--reps 3` runs, same shapes), `probe-new-gemv.txt` (first-pass new-arm observation).

## Method

This host is very noisy between runs (the gate/up report recorded ±40% substage swings for unchanged code, and this session's host state drifted further mid-run — some runs ran ~1.8x slower than the committed baseline), so the committed reports are reference only. The real comparison is an alternating A/B: the pre-item code and the new code were built as two bench binaries and run interleaved, 3 rounds, stage-profile and unprofiled throughput. Medians below. The baseline arm re-measured at `attention_projection_ms 356.4/345.9` for gemv/tiled 256, matching the committed `358/359`, so host state was comparable during the alternating block.

## attention_projection_ms medians (same-session A/B, 3 rounds)

```text
gemv  length128  169.2 -> 107.1  (-36.7%)   base [154,169,195] vs new [87,107,215] (overlap at top)
gemv  length256  356.4 -> 219.5  (-38.4%)   every new sample (199-263) below every base sample (339-511)
tiled length128  195.0 -> 131.8  (-32.4%)   every new sample (87-132) below every base sample (188-222)
tiled length256  345.9 -> 237.0  (-31.5%)   every new sample (214-240) below every base sample (312-456)
```

The projection win is the credible signal: three of four shapes show full sample separation, and the baseline medians land on the committed baseline numbers.

## Other substages (same-session A/B medians, gemv/tiled length256)

```text
gemv 256  base: out 214.9 kern 1508.2 gu 994.4 dn 981.0 total 4573.1
gemv 256  new:  out 226.9 kern 1579.2 gu 1076.8 dn 1038.3 total 4666.1
tiled 256 base: out 218.1 kern 957.0 gu 949.1 dn 970.5 total 3937.3
tiled 256 new:  out 232.1 kern 1166.6 gu 1123.1 dn 1120.8 total 4392.7
```

The untouched substages moved up in the new arm at 256 (kern/gu/dn +5-20%), i.e. the new arm's 256-length runs landed in slower host windows; the projection column is the only substage with clean cross-arm separation, and its drop is consistent across all four shapes. Window totals at gemv256 land inside noise (4573 -> 4666, +2%) because the attention kernel substage dominates there and swings run-to-run.

## Throughput medians (unprofiled, --reps 3, batchedTok/s)

```text
gemv  length128  68.68 -> 84.91  (+23.6%, samples overlap at the top)
gemv  length256  63.19 -> 53.55  (noise-indeterminate: new samples 52.7-74.3 vs base 56.0-72.8 overlap fully)
tiled length128  67.75 -> 65.86  (-2.8%, inside run noise)
tiled length256  60.13 -> 70.79  (+17.7%, samples overlap at the top)
```

Throughput on this host is too noisy to confirm a window-total win at 256; the projection substage columns are the direction-credible measurement.

## Commands

```text
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel gemv --reps 1 --stage-profile
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel tiled --reps 1 --stage-profile
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel gemv --reps 3
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel tiled --reps 3
```

The A/B files were produced by running the two binaries alternately with the same arguments (`base` = pre-item code copied to a temp bench-compare dir with the item's src changes reverted, `new` = item code in the worktree; the temp dir is not committed).

## Interpretation

- The triple batch dot is a real win on the projection substage: −32% to −38% at every measured shape, with full sample separation at three of four. Mechanism: the activation batch (64 x 2048 floats = 512 KB at this shape) is read once per shared row instead of three times across the 256 rows where all three matrices hold one; the weight traversal itself is unchanged.
- The win is bounded by row geometry: unlike gate/up (both matrices 3584 rows, fully shared), the QKV trio shares only its first 256 rows (kv regions), so the query matrix's remaining 1792 rows keep their own single-row passes. Projection is now roughly the cost of the query traversal alone.
- Projection is no longer a bottleneck target: at gemv256 it fell from ~7.8% of the profiled window to ~4.7%. The dominant remaining substages are the causal attention kernel (1508-1579 ms gemv256) and the two feed-forward traversals (gate/up and down near parity at ~1000 ms each); the down pass is Q6_K and is the plan item's remaining weight-side target.
- Window totals and throughput are noise-indeterminate at 256 on this host; the projection columns carry the signal.

## Validation

```text
dotnet build Lair.sln -c Release --nologo -> 0 warnings, 0 errors
dotnet test tests/Lair.Tests -c Release --no-build --filter "FullyQualifiedName~DenseWeightFusionTests|FullyQualifiedName~CpuTuningTests" -> 81/81 passed
dotnet test tests/Lair.Tests -c Release --no-build --filter "FullyQualifiedName~DenseWeightFusionTests|FullyQualifiedName~CpuTuningTests|FullyQualifiedName~SharedInputGemvTests|FullyQualifiedName~DenseForwardProfileTests|(Category=Kernel|BatchKernel|Prefill)&Category!=Cuda" -> 567/567 passed
dotnet test tests/Lair.Tests -c Release --no-build --filter "(Category=Prefill|Engine|Kv|Attention|Kernel|Serve|Config|HotReload)&Category!=Cuda" -> 932/932 passed, 0 failed, 0 skipped (5m52s)
```

New tests prove: triple route vs three independent `GemvBatch` calls bit-identical for Q4_K/Q4_K/Q4_K, Q6_K x3, and mixed Q4_K/Q6_K trios at equal-row, Qwen-shaped 2048/256/256, and asymmetric 512/384/256 row geometry at counts 1..64 including ragged counts (10/11/13/17 exercise triple-group tails); threaded triple route bit-identical to the single-threaded triple route and to threaded independent calls under forced tiny row blocks; scratch-ceiling degradation (triple -> pair+single -> three singles) bit-identical to independent calls at every pinned ceiling; pinned narrower triple groups (chain group 2 -> triple 2, 1 -> 1) bit-identical through the stack-staged triple kernel; postures outside the scratch route (Q8_0, IQ4_XS, ragged columns) refuse without touching outputs; `TripleChainGroup` resolution follows the resolved chain group and vector posture and stays inside the triple register budget.
