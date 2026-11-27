# Prefill `gemm` kernel probe 2026-11-27T09-48-07Z (headline extract)

Same lane, model, build (`app-posture-2026-11-27T06-53-30Z`, repo HEAD `016dd9c`) and harness as the posture sweeps; this probe swaps the kernel axis to `--kernels gemm` only, with 1 warmup round + 2 measured interleaved rounds at rowWidth=64. Full summary/data.json/report.html and raw logs live at `/root/lair-measurements/qwen25-small-cpu/reports/2026-11-27T09-48-07Z-optimize-prefill-gemm-kernel-xeon/` (hashes in `commands.md`); this file is a headline extract.

## batchedTok/s medians vs llama-bench pp tok/s medians

| arm | threads=8 | threads=16 | threads=32 |
| --- | ---: | ---: | ---: |
| LAIR gemm length 128 | 12.045 | 15.005 | 19.330 |
| LAIR gemm length 256 | 9.985 | 16.375 | 19.125 |
| llama pp length 128 | 70.745 | 132.640 | 238.435 |
| llama pp length 256 | 72.715 | 136.705 | 224.620 |

Matched-posture ratios: threads=8 → 13.7–17.0%; threads=16 → 11.3–12.0%; threads=32 → 8.1–8.5%. LAIR spread 0.7–13.5%, llama spread 0.3–7.7%.

## Reading — the blocked-matmul route does not transfer to this host

`gemm` is **not** faster than `gemv`/`tiled` here: at threads=8 it measures 12.045 / 9.985 tok/s against gemv's 13.655 / 10.585 (r2 sweep) and tiled's 12.030 / 11.010, so the tile route is at best equal and slightly worse at length 256. Its implied traversal bandwidth stays in the same starved band — `weightFit=3547.54 ms/traversal (UNRELIABLE: measured costs are not separable into the two terms on this run) impliedWeightGB/s=0.47` and `weightFit=2974.43 ms/traversal … impliedWeightGB/s=0.56` at threads=32 — nowhere near the 12.21 GB/s that LAIR's own gemm route reached on the developer's local AVX2 box (`docs/engine/cpu-kernels.md`). No capability refusal occurred: `gemm` ran every measured span for this Q4_K_M file.

Conclusion recorded from this run: switching `engine.prefill_kernel` to `gemm` is not a lever on this Xeon, and any claim that the blocked-matmul route recovers prefill throughput must be treated as posture-specific (local box only) until target-host evidence exists.
