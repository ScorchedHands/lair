# Prefill Q4_K wide-route A/B 2026-11-27T15-25-26Z (headline extract)

Lane: Xeon `lair-lab`, threads=8, rowWidth=64, kernels gemv/tiled, lengths 128/256, reps=1 per command with stage profiling; runner `scripts/run-xeon-prefill-stage-ab.py` arms `[off, 32, 64]` (`--q4k-matmul-route off|32|64`, production `--matmul-route on --attention-batch-dots on`), 2 warmup rounds + 5 measured interleaved rounds with odd-round rotation. Build `build/app-posture-2026-11-27T06-53-30Z/Lair.Bench.dll`, repo HEAD `016dd9c`. Full summary/data.json/report.html and raw logs live at `/root/lair-measurements/qwen25-small-cpu/reports/2026-11-27T15-25-26Z-optimize-prefill-q4k-route-wide-xeon-t8/`; this file is a headline extract.

## total_ms medians (lower is better)

| kernel | length | off | 32 | 64 |
| ---: | ---: | ---: | ---: | ---: |
| gemv | 128 | 9961.3 | **9014.0** | 9812.7 |
| tiled | 128 | 10973.7 | 10740.5 | 10654.8 |
| gemv | 256 | 22992.2 | **22126.4** | 23135.2 |
| tiled | 256 | 23212.8 | 22713.7 | **22305.3** |

Width 32 is faster than off on all four shapes (gemv@128 −9.5%, gemv@256 −3.8%, tiled@128 −2.1%, tiled@256 −2.1%), with gate/up and down substages both moving in the same direction; width 64 is mixed (worse on gemv, marginally better on tiled).

## Why this does not authorize a promotion yet
Intra-arm total_ms spreads are wide: pooled worst case off 23.9%, arm 32 26.7%, arm 64 29.9% (gemv@128 alone is ±26.7% for arm 32). Every width-32 delta sits inside its own noise band, so the run is directional evidence only — consistent sign across four shapes and substages, but not separation at this measurement power. This matches the earlier three-pass conclusion (`...T01-16-57Z`, `...T02-28-33Z`, `...T03-09-24Z`) that end-to-end bands on this host are too wide for width decisions, while the kernel-level seam separates cleanly (`reports/2026-11-27T14-14-58Z-optimize-q4k-widebatch-xeon-pinning`: tile beats auto by ~+50% GFLOP/s at count 32 and +35% at count 64 on the same host). `AutoMatmulRouteBatchWidthQ4K` therefore stays 0; a promotion attempt needs reps ≥3 per arm or rounds ≥8 focused on length 256, where bands run ~7–13%.
