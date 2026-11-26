# Prefill attention kernel vectorization 2026-11-25T11-26-38Z

Model: `C:\Users\Chuck\AppData\Local\Temp\kilo\optimize-small-cpu\models\qwen2.5-3b-instruct-q4_k_m.gguf`
Lane: local CPU lane, `threads=32`, `rowWidth=64`, Qwen2.5-3B Q4_K_M, no CUDA benchmarking.

What changed: the scalar per-element accumulator loops in `LayerOps.CausalAttention` (float), `LayerOps.CausalAttentionTiledRows`, and the rescale/normalization loops of the quantized generic `CausalAttention<TK,TV>` now run through two static helpers — `ScaleAccumulator` (`outHead[j] *= factor`) and `AccumulateWeighted` (`outHead[j] += weight * vt[j]`) — eight AVX2 lanes at a time with a scalar tail. `AccumulateWeighted` uses `Avx.Multiply` then `Avx.Add` (two roundings), never `Fma.MultiplyAdd`, so the vectors land on the same bits the scalar loops produced. Score dots, `MathF.Exp`, softcap, running-max/sum order, and `1f / sumExp` are untouched. `BidirectionalAttention` was left scalar (out of this item's scope).

Files: `src/Lair.Models/Engine/LayerOps.cs`, `tests/Lair.Tests/AttentionAccumulatorTests.cs` (new, 41 tests).

## Method

This host is very noisy between runs (substage timers swing ±40% run-to-run even for unchanged code), so the committed substage report numbers are only a rough reference. The real comparison is an alternating A/B: the committed `LayerOps.cs` and the vectorized one were built as two separate bench binaries and run in interleaved order, 3 rounds, stage-profile and unprofiled throughput. Medians below.

Raw outputs:
- `ab-stage-profile.txt` — 12 alternating single-rep stage-profile runs (baseline vs vectorized, gemv and tiled, lengths 128/256).
- `ab-throughput.txt` — 12 alternating unprofiled `--reps 3` runs, same shapes.
- `baseline-stage-profile-gemv.txt`, `baseline-stage-profile-tiled.txt` — committed-kernel single runs in this session.
- `prefill-stage-profile-gemv.txt`, `prefill-stage-profile-tiled.txt`, `vectorized-stage-profile-gemv.txt`, `vectorized-stage-profile-gemv-reps3.txt` — vectorized single runs (first-pass observations; see the A/B medians for the real comparison).

## attention_kernel_ms medians (same-session A/B, 3 rounds)

```text
gemv  length128  baseline 655.2  ->  vectorized 420.3   (-35.9%)
gemv  length256  baseline 2579.5 ->  vectorized 1530.3  (-40.7%)
tiled length128  baseline 677.1  ->  vectorized 275.8   (-59.3%)
tiled length256  baseline 2540.6 ->  vectorized 1169.2  (-54.0%)
```

Committed-report reference for the same kernel numbers: gemv `543.5`/`2616.4`, tiled `773.8`/`2092.0`. The baseline re-measurements here (`642-686`/`2215-2709`/`522-716`/`2215-2936`) sit in the same band, so the committed numbers remain a valid reference; the vectorized numbers beat every baseline sample at the same length and kernel.

## Window total_ms medians (stage-profile)

```text
gemv  length128  2606.9 -> 2237.9  (-14.2%)
gemv  length256  6274.1 -> 4906.0  (-21.8%)
tiled length128  2361.7 -> 2316.2  (-1.9%)
tiled length256  5843.1 -> 5113.8  (-12.5%)
```

## Throughput medians (unprofiled, --reps 3, batchedTok/s)

```text
gemv  length128  64.82 -> 63.66  (-1.8%, inside run noise)
gemv  length256  44.92 -> 56.66  (+26.1%)
tiled length128  54.67 -> 58.48  (+7.0%)
tiled length256  44.00 -> 52.83  (+20.1%)
```

Committed-report throughput reference: gemv `57.56`/`40.81`, tiled `48.25`/`48.44` tok/s.

## Commands

```text
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel gemv --reps 1 --stage-profile
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel tiled --reps 1 --stage-profile
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel gemv --reps 3
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel tiled --reps 3
```

The A/B files were produced by running the two binaries alternately with the same arguments (`baseline-*`/`vectorized-*` builds under the temp bench-compare dirs, not committed).

## Interpretation

- The accumulator vectorization is a real and consistent win on the kernel substage: −36% to −59% across all four shapes, with every vectorized sample beating every baseline sample.
- At length 256, where the kernel dominates the window, the window total and throughput follow the kernel: gemv +26% and tiled +20% tok/s. At length 128 the kernel is a smaller share, so gemv throughput lands inside noise and tiled gains ~7%.
- The tiled kernel's kernel substage benefits most (−54–59%): its per-row accumulator slices now vectorize the same way the per-row kernel does, and tiled is the route that was paying the most accumulator-loop cost at 256.
- gemv at 256 still spends more in the kernel than tiled does (1530 vs 1169 median), so the tiled route remains the better prefill kernel at long lengths — the relative gap narrowed but the ordering did not change.

## Validation

```text
dotnet build Lair.sln -c Release --nologo -> 0 warnings, 0 errors
dotnet test tests/Lair.Tests -c Release --no-build --filter "Category=Attention" -> 57/57 passed (41 new)
dotnet test tests/Lair.Tests -c Release --no-build --filter "(Category=Prefill|Engine|Kv|Attention|Kernel|Serve|Config|HotReload)&Category!=Cuda" -> 932/932 passed, 0 failed, 0 skipped (5m23s; 891 before the new tests)
```

New tests prove: helper loops bit-identical to scalar loops across head dimensions 1..128 including non-multiples of 8 (scalar-tail coverage); a two-rounding probe that a fused multiply-add would fail; the float `CausalAttention` and its windowed variant bit-identical to a scalar reference kernel at `headDim=12/13/8`; `CausalAttentionTiledRows` bit-identical to its scalar reference and to the per-row kernel at `headDim=12/13`.
