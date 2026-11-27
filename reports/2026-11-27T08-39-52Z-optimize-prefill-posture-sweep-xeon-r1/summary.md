# Prefill thread-posture sweep 2026-11-27T08-39-52Z (headline extract)

Lane: Xeon CPU-only lair-lab (root@192.168.33.53), rowWidth=64, Qwen2.5-3B Instruct Q4_K_M (`qwen2.5-3b-instruct-q4_k_m.gguf`, 2,104,932,768 bytes). Harness: `scripts/run-xeon-prefill-posture-sweep.py` against LAIR bench build `build/app-posture-2026-11-27T06-53-30Z/Lair.Bench.dll` (repo HEAD `016dd9c`) and prebuilt `llama-b12032/llama-bench`. 2 warmup rounds + 2 measured interleaved rounds, arms rotated on odd rounds, reps=1. Full summary/data.json/report.html plus all raw logs live at `/root/lair-measurements/qwen25-small-cpu/reports/2026-11-27T08-39-52Z-optimize-prefill-posture-sweep-xeon-r1/` on the measurement host (see `commands.md` for sha256); this file is a headline extract, not the full artifact.

## batchedTok/s medians (LAIR) vs llama-bench pp tok/s medians

| arm | threads=8 | threads=16 | threads=32 |
| --- | ---: | ---: | ---: |
| LAIR gemv length 128 | 14.595 | 19.005 | 27.045 |
| LAIR gemv length 256 | 11.440 | 17.490 | 19.275 |
| LAIR tiled length 128 | 12.120 | 19.425 | 21.630 |
| LAIR tiled length 256 | 11.145 | 17.455 | 19.120 |
| llama pp length 128 | 73.315 | 133.095 | 231.155 |
| llama pp length 256 | 73.145 | 137.450 | 232.800 |

Matched-posture ratios (LAIR ÷ llama): threads=8 → 15.2–19.9%; threads=16 → 12.7–14.6%; threads=32 → 8.2–11.7%. Every difference separates beyond its arm's noise band (LAIR intra-arm spread 0.1–29.7%, llama spread 2.2–5.2%).

## Reading

- Thread posture raises LAIR absolute throughput (~14.6→27 tok/s best case) but per-thread efficiency collapses monotonically: gemv@128 1.82 → 1.19 → 0.85 tok/s/thread, tiled@256 1.39 → 1.09 → 0.60.
- llama.cpp scales near-linearly on this host (73→133→231 pp tok/s), so the matched-posture ratio *worsens* as threads rise: posture is not where the deficit lives.
- The explicit `-t 8` llama arm reproduces the historical `reports/2026-11-26T08-46-39Z-optimize-prefill-vs-llama-xeon-t8` band (70.9–72.7), confirming that earlier comparison was posture-comparable rather than an unfair thread mismatch.
- Implied weight-traversal bandwidth from the same run's fit lines: 1.33 / 0.66 GB/s at threads=32 and 0.37 / 0.42 GB/s at threads=8 (`weightFit` ms/traversal in the raw logs), against a dual-socket DDR4 host capable of tens of GB/s.
