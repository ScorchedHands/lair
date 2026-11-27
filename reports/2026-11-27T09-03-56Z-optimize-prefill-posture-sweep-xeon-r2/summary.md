# Prefill thread-posture sweep 2026-11-27T09-03-56Z (headline extract)

Same lane, model, harness, build and command shape as `2026-11-27T08-39-52Z-optimize-prefill-posture-sweep-xeon-r1` — this run is the deeper repeat of it: 2 warmup rounds + **4** measured interleaved rounds with rotation. Full summary/data.json/report.html and raw logs live at `/root/lair-measurements/qwen25-small-cpu/reports/2026-11-27T09-03-56Z-optimize-prefill-posture-sweep-xeon-r2/` (hashes in `commands.md`); this file is a headline extract.

## batchedTok/s medians (LAIR) vs llama-bench pp tok/s medians

| arm | threads=8 | threads=16 | threads=32 |
| --- | ---: | ---: | ---: |
| LAIR gemv length 128 | 13.655 | 18.955 | 21.800 |
| LAIR gemv length 256 | 10.585 | 17.180 | 19.840 |
| LAIR tiled length 128 | 12.030 | 18.520 | 23.440 |
| LAIR tiled length 256 | 11.010 | 16.940 | 21.015 |
| llama pp length 128 | 74.285 | 134.100 | 230.395 |
| llama pp length 256 | 73.515 | 140.490 | 223.980 |

Matched-posture ratios (LAIR ÷ llama): threads=8 → 14.4–18.4%; threads=16 → 12.1–14.1%; threads=32 → 8.9–10.2%. All separate beyond noise (LAIR spread 6.1–43.5%, llama spread 2.6–16.1%). Per-thread efficiency again collapses: gemv@128 1.71→1.18→0.68, tiled@256 1.38→1.06→0.66 tok/s/thread; llama per-thread stays roughly flat (9.3→8.4→7.2).

## Reading

Four measured rounds reproduce the r1 sweep within its own noise: thread posture buys absolute throughput but never closes the ratio, and threads=32 is worse in relative terms than threads=8. Implied traversal bandwidth from this run's fits at threads=32: 1.17 / 0.69 GB/s (`weightFit` ms/traversal lines). Together with r1 this establishes thread posture as a dead end for closing the llama.cpp gap on this host.
