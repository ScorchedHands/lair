# MVP5.2 CUDA GEMV parity and speed gate - 2026-11-18

## Question

MVP5.2 adds CUDA GEMV kernels for the GGUF quant formats LAIR's CPU engine already carries. This bench records two things before any default/speed claim:

1. each newly covered format has a measured CUDA-vs-CPU quality band, and
2. per-format CUDA-vs-CPU speed posture is measured rather than inferred from kernel shape or GPU presence alone.

## Speed command

```powershell
dotnet run --project bench/Lair.Bench -c Release --no-build -- cuda-gemv
```

Default measurement geometry:

- rows: `512`
- cols: `1024`
- batch activation vectors: `32`
- reps per timed arm: `7`
- formats swept: every currently CUDA-supported `QuantFormat` (18 total)
- CPU arms: scalar oracle (`threads=1`, AVX disabled) and auto-threaded engine posture
- CUDA posture: device-copy, with one-time weight copy absorbed in untimed warm-up

Host facts from this run:

- AVX2 available
- physical cores: `24`
- logical processors: `32`
- CPU auto workers: `24`
- fixtures: deterministic pinned block patterns using fixed arithmetic progressions, no RNG; same packing classes as the GPU parity fixtures
- CUDA wall time includes each call's activation upload, launch, sync, and output download; it excludes only the one-time per-tensor device weight copy

## Recorded quality bands from tests

These are the authoritative GEMV correctness gates. The bench equivalence column below is a sanity signal on different fixture magnitudes/shape and does not replace these test bands.

| Format | Band vs CPU oracle | Measured worst in parity gate |
|---|---:|---:|
| `Q8_0` | `1e-6` relative | `7.96e-7` |
| `Q5_0` | `1e-6` relative | `8.16e-7` |
| `Q2_0` | `1e-6` relative | `7.41e-7` |
| `Q5_K` | `4e-6` relative | `2.72e-6` |
| `Q2_K` | `1e-6` relative | `4.22e-7` |
| `Q3_K` | `2e-5` relative | `1.01e-5` batched |
| `IQ4_NL` | `2e-6` relative | `1.03e-6` host-mapped; exact-block ~8.97e-8 |
| `IQ4_XS` | `2e-6` relative | `1.07e-6` batched |
| `IQ2_XXS` | `5e-7` relative | measured bit-exact zero in parity gate |
| `IQ2_XS` | `5e-7` relative | measured bit-exact zero in parity gate |
| `IQ3_XXS` | `5e-7` relative | measured bit-exact zero in parity gate |
| `IQ2_S` | `5e-7` relative | `8.27e-8` partial final block |
| `IQ3_S` | `5e-7` relative | `1.47e-7` partial final block |

Pre-existing CUDA GEMV lanes for `F32`, `F16`, `BF16`, `Q4_K`, and `Q6_K` retain their prior posture; this gate does not re-band or widen them.

## Speed measurement summary

All values below are medians from the main-run command above. Ratios > 1 mean CUDA is faster for that comparison basis.

| Format | Packed MiB | CPU scalar ms | CPU auto ms | CUDA wall ms | Kernel busy µs/rep | Scalar vs CUDA wall | Auto vs CUDA wall | Compute-only ratio | Bench equivalence max rel drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `F32` | 2.00 | 38.435 | 0.525 | 2.270 | 1904.4 | 16.93 | 0.23 | 20.18 | 0.0063601 |
| `F16` | 1.00 | 23.378 | 0.498 | 2.859 | 3026.9 | 8.18 | 0.17 | 7.72 | 0.0032998 |
| `BF16` | 1.00 | 11.018 | 0.621 | 2.878 | 2476.9 | 3.83 | 0.22 | 4.45 | exact |
| `Q8_0` | 0.53 | 12.950 | 0.506 | 2.021 | 2007.0 | 6.41 | 0.25 | 6.45 | 0.00138574 |
| `Q4_K` | 0.28 | 11.717 | 0.542 | 1.714 | 1834.6 | 6.84 | 0.32 | 6.39 | 8.16e-6 |
| `Q5_0` | 0.34 | 17.813 | 2.136 | 1.957 | 1984.9 | 9.10 | 1.09 | 8.97 | 0.000996202 |
| `Q6_K` | 0.41 | 16.219 | 4.156 | 1.925 | 2089.3 | 8.42 | 2.16 | 7.76 | 0.000101947 |
| `Q5_K` | 0.34 | 15.537 | 2.582 | 2.751 | 2430.0 | 5.65 | 0.94 | 6.39 | 0.000174343 |
| `Q2_K` | 0.16 | 12.694 | 0.509 | 1.476 | 1629.9 | 8.60 | 0.34 | 7.79 | 0.000332475 |
| `Q3_K` | 0.21 | 13.662 | 2.109 | 1.967 | 2006.7 | 6.94 | 1.07 | 6.81 | 0.0058546 |
| `Q2_0` | 0.14 | 15.100 | 0.556 | 1.348 | 1297.4 | 11.20 | 0.41 | 11.64 | 2.476e-5 |
| `IQ4_NL` | 0.28 | 17.142 | 0.623 | 2.496 | 2759.4 | 6.87 | 0.25 | 6.21 | 0.000803455 |
| `IQ4_XS` | 0.27 | 13.917 | 0.548 | 2.392 | 2468.6 | 5.82 | 0.23 | 5.64 | 0.00898242 |
| `IQ2_XXS` | 0.13 | 17.893 | 1.360 | 0.572 | 413.1 | 31.28 | 2.38 | 43.31 | 8.35e-5 |
| `IQ2_XS` | 0.14 | 17.504 | 1.435 | 0.525 | 382.9 | 33.33 | 2.73 | 45.72 | 0.000858286 |
| `IQ2_S` | 0.16 | 18.096 | 1.375 | 0.529 | 351.1 | 34.23 | 2.60 | 51.54 | 1.98e-5 |
| `IQ3_XXS` | 0.19 | 18.190 | 0.729 | 0.578 | 419.3 | 31.45 | 1.26 | 43.38 | 2.446e-5 |
| `IQ3_S` | 0.21 | 25.810 | 1.120 | 0.567 | 402.9 | 45.50 | 1.98 | 64.07 | 3.047e-5 |

## Interpretation

### Speed posture

The honest comparison for serving is the auto-threaded CPU wall vs CUDA total wall, because both include a full `GemvBatch` pass and CUDA includes runtime activation/output crossing plus launch/sync overhead:

- Clear measured wins on this microbench:
  - `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`: auto-vs-CUDA wall ratios roughly `1.98`–`2.73`.
  - `Q6_K`: ratio `2.16`.
- Near-tie or slight wins:
  - `Q5_0`: `1.09`.
  - `Q3_K`: `1.07`.
- Losses against auto-threaded CPU on this shape:
  - dense and many block/i-quants with ratios below `1`, including `F32` (`0.23`), `F16` (`0.17`), `BF16` (`0.22`), `Q8_0` (`0.25`), `Q4_K` (`0.32`), `Q2_K` (`0.34`), `Q2_0` (`0.41`), `IQ4_NL` (`0.25`), and `IQ4_XS` (`0.23`).

The compute-only ratios show the CUDA kernels themselves are very fast, but they exclude PCIe/runtime transfer overhead and should not be used as a serving speed claim by themselves. For dense formats especially, the one-time weight residency plus per-call activation/output crossing makes total wall slower than threaded CPU on this modest microbench shape.

### Recommendation / default posture

No automatic CUDA GEMV format default is promoted from this bench alone:

- The runtime default remains host/CPU execution unless a model explicitly opens `placement.gpu_layers`.
- Sub-4-bit i-quants are measured total-pass wins in this gate and may be selected on GPU when their memory/quality posture is desired.
- Dense formats should not be claimed as faster CUDA GEMV paths from this bench; they remain opt-in hybrid execution, with any future default/speed claim gated by whole-model measurement.
- `Q5_0` and `Q3_K` are near-tie/slight wins on this microbench but not strong enough to promote beyond explicit selection without broader model-level evidence.

This satisfies the bench-gate posture: speed claims follow measured numbers, losses stay non-default, and quality bands remain those recorded by the parity tests rather than widened for convenience.

## Caveats

- GPU contention from other workloads on this box affects wall maxima; medians are used in the table.
- CPU auto-threaded timings can vary run-to-run, especially with dense fixtures where cache residency/boost behavior matters.
- The bench fixture shape (`512 x 1024`, batch `32`) is a repeatable microbench gate for GEMV kernels, not whole-model decode or prefill throughput. Fast GPU prefill remains MVP5.4 work because it needs device-side attention and device-resident KV composition.
- Bench equivalence drift values are not directly comparable to the parity test bands because the bench fixture magnitudes and shape differ; use the recorded quality-band table above for correctness gates.
