# Prefill Q4_K wide-route A/B high-power 2026-11-27T19-56-50Z (headline extract)

Lane: Xeon `lair-lab`, threads=8, rowWidth=64, kernels gemv/tiled, length 256 only, reps=1 per command with stage profiling; runner `scripts/run-xeon-prefill-stage-ab.py` arms `[off, 32]` (`--q4k-matmul-route off|32`, production `--matmul-route on --attention-batch-dots on`), 3 warmup rounds + 9 measured interleaved rotated rounds (18 invocations). Build `build/app-posture-2026-11-27T06-53-30Z/Lair.Bench.dll`, repo HEAD `016dd9c`. Full summary/data.json/report.html and raw logs live at `/root/lair-measurements/qwen25-small-cpu/reports/2026-11-27T19-56-50Z-optimize-prefill-q4k-route-wide-xeon-t8-highpower/`; this file is a headline extract.

## total_ms medians (lower is better)
| kernel | off | 32 | delta | intra-arm bands off / 32 |
| ---: | ---: | ---: | ---: | ---: |
| gemv | 22068.5 | 21554.3 | −2.33% | ±9.62% / ±14.62% |
| tiled | 23426.4 | 23187.0 | −1.02% | ±17.76% / ±10.26% |

Neither window total separates beyond its own band even at rounds=9 with warmups and rotation.

## Stage-level medians (the substages the seam actually acts on)
| stage | kernel | off median [min–max] | 32 median [min–max] | delta |
| --- | ---: | ---: | ---: | ---: |
| feed_forward_down_ms | gemv | 7501.8 [6792.5–8030.8] | 6958.6 [6115.8–7272.1] | −7.24% (ranges overlap) |
| feed_forward_down_ms | tiled | 8303.8 [7812.8–8711.1] | 7866.4 [7388.9–8151.5] | −5.27% (ranges overlap) |
| attention_output_ms | gemv | 1612.1 [1554.0–1723.7] | 1226.3 [1058.7–1316.9] | −23.9% — **non-overlapping ranges** (arm-32 max below off min) |
| feed_forward_gate_up_ms | gemv | 8926.0 | 9039.3 | +1.27% (fused pair traversal is untouched by this seam, as expected) |

Mechanism holds: the Q4_K single-matrix passes (`feed_forward_down`, attention `o_proj`) improve; fused gate/up does not move because it stays on its shared-activation pair traversal. The one clean separation is gemv `attention_output_ms`; down substages are directional-only, and window totals stay inside noise even at this measurement power.

## Decision
No promotion: the recorded bar for a default change — "a committed run puts a `total_ms` delta outside its own arm's spread" — fails on both kernels after 9 rotated rounds with warmups, so `AutoMatmulRouteBatchWidthQ4K` stays 0. Route micro-tuning is closed as noise-limited at this host/model power; further width experiments are not worth more measurement time until a larger lever (host placement/dispatch) lands, which the pinning attribution already measured at +7…14% end-to-end.
