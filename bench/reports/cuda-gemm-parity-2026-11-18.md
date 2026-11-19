# CUDA GEMM gate measurement — 2026-11-18

## Command and posture

```powershell
dotnet run --project bench/Lair.Bench -c Release --no-build -- cuda-gemm
```

Defaults used for this committed gate:

```text
shape rows=512 cols=1024 batch=32 reps=7 formats=13 cpuArms=scalar1T(threads=1,avx2=false untimed equivalence)+auto(writers below) cudaPosture=deviceCopy
host avx2=yes physicalCores=24 logicalProcessors=32 cpuAutoWorkers=24
fixture=deterministic pinned block patterns (fixed arithmetic progressions, no RNG; the same packing classes the GPU parity fixtures walk) activations=Sin(0.11*c+7*v)*1.3 fixed formula
basis: every arm runs CpuBackend/CudaBackend batched matmul over identical bytes (count activation vectors of cols, count*rows output layout); cuda warm-up (including the one-time per-tensor device copy) is untimed; cuda wall includes each call's activation upload, launch, sync, and download but not weight residency
routes: both CPU arms take that format's own dense-matrix weight pass - GemmBatch where CpuBackend.SupportsGemm is true (Q8_0/Q4_K/IQ4_XS), otherwise GemvBatch as the batched-row alternative a blocked matmul falls back to on this build; each format prints cpuRoute=gemm|gemv so every ratio states which CPU chain it compares
flops: GFLOP/s figures count one multiply and one add per dot-product element, i.e. 2*rows*cols*batch per pass - reproducible from the printed shape for every arm
```

## Median summary table

Ratios are median-based wall-time ratios; `ratio_auto_cuda > 1` means CUDA was faster than the auto-threaded CPU route in total measured pass time. Kernel-busy microseconds exclude PCIe/runtime transfer overhead and are not comparable to wall milliseconds without that caveat.

| Format | CPU route | Packed MiB | CPU scalar ms | CPU auto ms | CUDA wall ms | Kernel busy us/rep | Auto vs CUDA ratio | Scalar vs kernel-busy compute-only ratio | Max relative drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F32 | `gemv` fallback | 2.00 | 40.020 | 0.555 | 0.742 | 432.6 | 0.75 | 92.52 | 0.0063601 |
| F16 | `gemv` fallback | 1.00 | 23.253 | 0.376 | 0.498 | 444.3 | 0.75 | 52.34 | 0.0032998 |
| BF16 | `gemv` fallback | 1.00 | 10.977 | 0.463 | 0.810 | 455.4 | 0.57 | 24.10 | 0 |
| Q8_0 | `gemm` comparable | 0.53 | 6.510 | 0.426 | 0.476 | 421.3 | 0.90 | 15.45 | 0.00138574 |
| Q5_0 | `gemv` fallback | 0.34 | 17.561 | 3.334 | 0.513 | 474.1 | 6.50 | 37.04 | 0.000996202 |
| Q2_0 | `gemv` fallback | 0.14 | 15.168 | 0.602 | 0.437 | 383.1 | 1.38 | 39.59 | 0.0002476 |
| Q4_K | `gemm` comparable | 0.28 | 6.450 | 0.464 | 0.561 | 490.6 | 0.83 | 13.15 | 8.16078E-06 |
| Q6_K | `gemv` fallback | 0.41 | 16.034 | 4.597 | 0.587 | 536.6 | 7.84 | 29.88 | 0.000101947 |
| Q5_K | `gemv` fallback | 0.34 | 15.341 | 2.475 | 0.702 | 611.3 | 3.52 | 25.10 | 0.000174343 |
| Q2_K | `gemv` fallback | 0.16 | 12.663 | 0.445 | 0.517 | 464.0 | 0.86 | 27.29 | 0.000332475 |
| Q3_K | `gemv` fallback | 0.21 | 14.921 | 2.020 | 0.555 | 519.9 | 3.64 | 28.70 | 0.0058546 |
| IQ4_NL | `gemv` fallback | 0.28 | 16.875 | 0.529 | 0.543 | 492.4 | 0.97 | 34.27 | 0.000803455 |
| IQ4_XS | `gemm` comparable | 0.27 | 6.698 | 0.495 | 0.626 | 572.7 | 0.79 | 11.70 | 0.00898242 |

## Interpretation

- Comparable-route formats are the only direct CPU-blocked-matmul comparisons: `Q8_0`, `Q4_K`, and `IQ4_XS`. On this synthetic dense matmul gate, CUDA hand-written kernels land near tie or slower against auto-threaded CPU blocked matmul (`0.90`, `0.83`, and `0.79` respectively). These remain non-default/opt-in posture choices; no automatic GEMM default is promoted from this bench alone.
- Fallback-route formats compare CUDA blocked matmul against the batched-row route that would run today when CPU has no blocked-matmul kernel for that format. Several of those show large measured wins (`Q5_0`, `Q6_K`, `Q5_K`, `Q3_K`), but they are not claims of fast whole prefill because attention and device-resident KV still cross the seam per layer; full on-device transformer steps land in later CUDA-attention work.
- Unquantized F32/F16/BF16 show total-pass losses against auto-threaded CPU batched-row weight passes despite much faster kernel-busy than scalar, largely because runtime transfer/launch/sync overhead and high CPU memory bandwidth dominate this shape. Kernel busy remains useful as compute-only evidence, not a default-selection signal.
- Drift values are bench measurements only; they do not assert or widen parity bands recorded in tests. The largest relative drift here is `IQ4_XS` (`0.00898242`) with max absolute drift `0.0546875`, reflecting this synthetic shape's cancellation-exposed fixture magnitudes rather than a gate assertion.

## Toolchain escalation posture

The measured shortfall on comparable CPU-GEMM formats does not justify adopting cuBLAS/cuBLASLt in MVP5.3: the native-library dependency would break the managed-first embedded-PTX posture, and fast dense GPU prefill is gated by later device-side attention/KV composition rather than a library swap alone. The escalation trigger stays recorded for post-device-step measurement; any future reconsideration requires both measured persistent shortfall and owner acceptance to relax managed-first.
