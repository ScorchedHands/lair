# Prefill Q6_K GEMM / hybrid dense-matrix route 2026-11-25T16-11-55Z

Model: `C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf` (Qwen2.5-3B Q4_K_M, 2,104,932,768 bytes; arch=qwen2 layers=36 hidden=2048 heads=16 kvHeads=2 headDim=128 vocab=151936 avx2=True).
Lane: local CPU lane, `threads=32`, `rowWidth=64`, kernels gemv/tiled, lengths 128/256. No CUDA benchmarking; the host's only practical GPU was untouched (the one real-GPU test suite that a broad filter matched is excluded from this report — see Validation).
Baseline reference: committed report `reports/optimize-prefill-qkv-triple-fusion-2026-11-25T16-06-41Z/` (batched Q/K/V triple fusion intact; CPU blocked matmul covered only Q8_0/Q4_K/IQ4_XS, so `gemm` was unavailable for this model's Q6_K matrices).

What changed: the CPU backend now carries a real blocked matmul for **Q6_K** (`GemmQ6KAvx2` batch-major tile kernel + `GemmQ6KScalar`, same ascending group/chunk walk, FMA association, and scalar-tail cleanup as the existing per-row Q6_K kernels), so `prefill_kernel=gemm` selects cleanly for models whose window formats include Q6_K. Additionally a **hybrid dense-matrix route** was added to `GemvBatch`: at or above a posture-resolved batch width (`MatmulRouteBatchWidth`, auto = 8 with vector posture, 0 without), a single-matrix Q6_K batched pass runs its rows through the blocked matmul tile executor instead of the per-row scratch batch dot — one activation-vector load per eight-column chunk feeds all eight tile rows' accumulator chains where the scratch route re-reads the activation batch once per chain group. The dispatch (`RunMatmulDispatch`) is shared verbatim with `GemmBatch`. Measurement pin: bench flag `--matmul-route off|on|N` pins the width (default posture resolves to 8 on this host, identical to `on`).

Files: `src/Lair.Backends.Cpu/CpuBackend+Gemm.cs`, `src/Lair.Backends.Cpu/CpuBackend.cs`, `src/Lair.Backends.Cpu/CpuKernelTuning.cs`, `bench/Lair.Bench/Program.cs`, new tests `tests/Lair.Tests/Q6KGemmRouteTests.cs` (19 cases), updated `GemmKernelTests.cs` / `PrefillGemmCapabilityGateTests.cs` / `CudaPrefillGemmCapabilityGateTests.cs` (the CUDA-side refusal lanes re-pinned to Q5_K, the format this build's CPU backend genuinely has no blocked matmul for).

## Bit identity

No tolerance was needed. The hybrid route and the new GEMM kernels preserve each output element's ascending group/chunk walk, FMA association order, and scalar-tail cleanup; `Q6KGemmRouteTests` proves bit-for-bit equality between the hybrid tile route and the batched-row scratch route at five matrix shapes (including ragged column tails 137/264/300 and the Qwen-shaped down geometry) across counts 8..64, threaded vs single-threaded under forced tiny row blocks, scalar-posture parity, formats outside the hybrid format unchanged under the pin, widths below the crossover keeping their route, and tuning resolution/pin/refusal behavior. `GemmKernelTests` (Q6_K added to the supported-format theory) and both capability-gate suites are green. Full-model logit parity is carried by the existing Parity-category lanes inside the broad gate rather than re-run standalone this session.

## Method

Host noise between processes runs ±10-35% on this box (documented in prior reports), so arms were run alternating: 3 rounds of `default -> off -> on` for throughput (`--reps 3`) and 3 rounds of `default -> off` for stage-profile (`--reps 1`). Medians below; samples in `ab-throughput.txt`, `ab-stage-profile.txt`, `data.json`. The posture-resolved default is identical to `--matmul-route on` (both resolve width 8 here), so the real A/B contrast is hybrid vs `off`; measured spread between the two behaviorally-identical arms calibrates run noise (~±5% at medians, with one slow host window in round 1).

## feed_forward_down_ms medians — the target substage (3 alternating rounds)

```text
gemv  length128   off 373.1 -> hybrid 334.9  (-10.2%)  full sample separation (hybrid max 349.1 < off min 366.6)
gemv  length256   off 734.0 -> hybrid 672.7  (-8.4%)   every hybrid sample below every off sample
tiled length128   off 387.6 -> hybrid 344.2  (-11.2%)  full sample separation (hybrid max 347.7 < off min 376.3)
tiled length256   off 774.6 -> hybrid 681.1  (-12.1%)  every hybrid sample below every off sample
```

All four shapes separate cleanly across arms — this is the credible signal, matching the mechanism (one activation pass per chunk feeds eight tile rows instead of a batch re-read per chain group). `feed_forward_gate_up_ms` medians moved <±3% between arms at all four shapes, confirming untouched Q4_K routes behave identically and that the down-column win is not an arm-window artifact.

## Window totals (stage-profile medians)

```text
gemv  length128   off 1290.6 -> hybrid 1249.2  (-3.2%)  full sample separation
gemv  length256   off 2963.9 -> hybrid 2890.2  (-2.5%)  near-separated (hybrid max 2963.6 vs off min 2963.5)
tiled length128   off 1227.7 -> hybrid 1188.7  (-3.2%)  samples overlap at the edges
tiled length256   off 2684.7 -> hybrid 2619.4  (-2.4%)  samples overlap at the edges
```

## Throughput medians (batchedTok/s, --reps 3) — noise-indeterminate at window level

```text
gemv  length128   off 98.99 -> default_hybrid 99.03 / on 100.86   (+0 to +2%)
gemv  length256   off 86.59 -> default_hybrid 81.76 / on 87.81    (hybrid samples straddle; round-1 hybrid runs landed in a slow host window)
tiled length128   off 103.89 -> default_hybrid 103.66 / on 111.28 (+0 to +7%)
tiled length256   off 94.48 -> default_hybrid 98.07 / on 102.32   (+2 to +8%)
```

Window throughput is dominated by the causal attention kernel at these shapes and swings run-to-run; it neither contradicts nor amplifies the substage win (consistent with prior reports' methodology notes). Versus the committed qkv-triple-fusion report's new-arm medians (gemv128 84.9, gemv256 53.6, tiled128 65.9, tiled256 70.8), current hybrid medians are higher at every shape — but those baselines were measured in a different host session and several throughput samples here exceed both arms' medians, so cross-session numbers are directional reference only (plotted as history in `report.html`).

## Interpretation

- Q6_K prefill down-traversal is measurably cheaper through the tile executor at batch width 8: −8% to −12% on every measured shape with full sample separation. The `ffn_down` share of a gemv length-256 window fell from ~24.7% (off) to ~23.3% (hybrid); gate/up and down are now further apart again in the stage profile (`down 672 vs gate_up 775 ms` hybrid, `734 vs 783` off).
- The capability change is separate from the speed change: Q6_K models can now select `prefill_kernel=gemm` on CPU (gate suites green), and this model's dense windows benefit through the hybrid route even when run as gemv/tiled.
- Remaining weight-side target after this item: gate/up stays the larger FFN substage; the causal attention kernel remains the dominant window substage (~42% at gemv length 256).

## Validation counts

```text
dotnet build -c Release -> succeeded, 0 warnings / 0 errors (4.3 s)
Category=Kernel|Backend|DenseWeightFusion|CpuTuning|Profile & Category!=Cuda -> 449/449 passed (the exact filter without the guard also matched a real-GPU suite that fails on this host's contended GPU; excluded per CPU-only lane policy)
FullyQualifiedName~Q6KGemmRouteTests|GemmKernelTests|PrefillGemmCapabilityGateTests -> 101/101 passed
FullyQualifiedName~CudaPrefillGemmCapabilityGateTests (host-side stand-in lanes, no driver opened) -> 37/37 passed
Category!=Cuda broad gate -> 2512/2512 passed, 0 skipped (7 m 50 s)
```

## Risks / open notes

- Auto width 8 was chosen from prior tile-reuse measurements and confirmed directionally here; a dedicated width sweep (4/16/32) was not run this session — the bench pin supports it if a later pass wants to re-confirm.
- Cross-host-session throughput comparisons remain unreliable on this box; only same-session alternating medians are treated as measurement, per prior reports' convention.
- No output-bit tolerance introduced anywhere: parity is bit-identical in all new lanes (see Bit identity).
