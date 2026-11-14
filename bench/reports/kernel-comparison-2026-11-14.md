# Prefill kernel cost comparison — 2026-11-14

This report compares the runnable prefill kernels on proven-format pinned fixtures. Kernel selection is read from the engine, not assumed: `gemm` runs only where every dense weight matrix has a blocked matmul format; mixed-format files are recorded as refusals rather than silently falling back.

## Measurement posture

Each kernel was measured in its own bench process so `peakWorkingSetMb` is per-kernel, not cumulative across kernels:

```text
dotnet run --project bench/Lair.Bench -c Release -- prefill-profile <gguf> \
  --threads 8 --reps R --lengths 512,1024,2048 --widths 32 --kernel <gemv|tiled|gemm> \
  --out bench/reports/prefill-kernel-<fixture>-<kernel>.txt
```

The proven-format fixtures used here are:

- `models/Llama-3.2-1B-Instruct-Q8_0.gguf` — all dense attention/FFN matrices run through the blocked matmul; measured with `--reps 3`.
- `models/Qwen3-0.6B-Q8_0.gguf` — all dense attention/FFN matrices run through the blocked matmul; measured with `--reps 2`.

Raw tables are committed beside this report as:

- `prefill-kernel-llama-q8-gemv.txt`, `prefill-kernel-llama-q8-tiled.txt`, `prefill-kernel-llama-q8-gemm.txt`
- `prefill-kernel-qwen3-gemv.txt`, `prefill-kernel-qwen3-tiled.txt`, `prefill-kernel-qwen3-gemm.txt`

## Llama Q8_0 kernel table

| Kernel | Weight fit (ms/traversal) | Implied weight GB/s | Attention fit (us/position pair) | Fitted crossover | 512 tok/s | 1024 tok/s | 2048 tok/s | Peak working set |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemv` | 439.52 | 2.35 | 32.491 | length >= 855 | 44.93 | 32.99 | 21.26 | 1532.0 MB |
| `tiled` | 527.01 | 1.96 | 24.202 | length >= 1368 | 45.43 | 34.34 | 24.25 | 1531.5 MB |
| `gemm` | 348.67 | 2.97 | 32.672 | length >= 669 | 51.03 | 36.34 | 22.53 | 1531.9 MB |

Reading: `tiled` lowers the fitted attention coefficient from `32.491` to `24.202 us/position pair`, and that lower quadratic term is what makes it fastest at `2048`. `gemm` lowers the fitted weight traversal cost versus `gemv` (`439.52 ms/traversal` down to `348.67 ms/traversal`) and improves implied bytes/sec, but its attention coefficient does not fall as far on this fixture, so it is strongest at shorter measured lengths rather than uniformly fastest.

## Qwen3 Q8_0 kernel table

| Kernel | Weight fit (ms/traversal) | Implied weight GB/s | Attention fit (us/position pair) | Fitted crossover | 512 tok/s | 1024 tok/s | 2048 tok/s | Peak working set |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemv` | 243.46 | 1.92 | 58.485 | length >= 274 | 39.25 | 27.39 | 14.79 | 1196.9 MB |
| `tiled` | 277.13 | 1.69 | 38.104 | length >= 467 | 53.87 | 35.54 | 20.96 | 1196.4 MB |
| `gemm` | 38.33 | 12.21 | 54.568 | length >= 53 | 57.92 | 35.23 | 17.49 | 1197.0 MB |

Reading: `tiled` is the clearest long-prompt win on Qwen3, with the fitted attention coefficient falling from `58.485` to `38.104 us/position pair`. The GEMM weight fit moves sharply down versus `gemv`, but its attention coefficient does not move consistently in this run; treat the kernel ordering as directional where residuals and coefficients disagree with wall time.

## Mixed-format refusal evidence

The pinned IQ4_XS file is useful for proving that a proven format alone is not enough: every dense matrix must be supported. `models/Llama-3.2-1B-Instruct-IQ4_XS.gguf` runs `gemv` and `tiled`, but refuses `gemm`:

```text
kernel=gemm status=refused reason=prefill kernel 'gemm' is not implemented by the forward engine: architecture 'llama' carries dense weight matrices of format(s) Q5_K ... set engine.prefill_kernel to 'gemv' or 'tiled' (or omit it) to run this file
```

Raw evidence is committed as `prefill-kernel-llama-iq4x-mixed.txt`. That file's peak line is cumulative across the multi-kernel process and is not used for per-kernel memory comparison.

## Live rollout cross-check

The serving-side proof in `kernel-gemm-rollout-2026-11-14.md` uses `models/Qwen3-0.6B-Q8_0.gguf`, changes only `engine.prefill_kernel`, and confirms an identical 32-token greedy stream between `gemv` and `gemm`. Its live peak working set was `1748.5 MB` for the server-side `gemv` posture versus `1738.8 MB` for `gemm`; its speed half measured no consistent throughput win at Q8_0 (`-7.7%` wall-throughput delta), which is why this report treats fitted cost movement and exact-stream equality as stronger than a single live speed sample.

## Caveats

These are shared-box measurements, not isolated microbenchmarks. The fits use best-of-rep passes after warm passes, but OS page-cache state differs between separate kernel processes, so cross-kernel ordering is directional unless the fitted terms and wall time agree. The committed raw tables preserve every printed fit residual; where a table says `UNRELIABLE`, it is not used as evidence for ordering.
