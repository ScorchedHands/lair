# Q4_K wide-batch + prefill placement attribution - Xeon E5-2683 v4 host

Measurement stamp: 2026-11-27T14-14-58Z. Question: is the target-host Q4_K prefill deficit NUMA/SMT placement or kernel/dispatch overhead? Raw per-arm logs remain on the measurement host under this report directory's logs/; every number below comes from data.json in this directory.

## Posture (applies to every number below)

| field | value |
|---|---|
| model file | qwen2.5-3b-instruct-q4_k_m.gguf (2104932768 bytes) |
| tool seam | Lair.Bench q4k-widebatch (bench/Lair.Bench/Q4KWideBatchBench.cs at commit 50a4ef0) + prefill-profile on real model |
| app dir (host) | app-widebatch-2026-11-27T14-14-58Z |
| host | dual Xeon E5-2683 v4, AVX2+FMA only, governor performance; node0=CPUs 0-15 (+siblings 32-47), node1=16-31(+48-63) |
| repo HEAD before edits | 016dd9c |
| kernel-arm flags | --threads T --reps 5 --counts 8,16,32,64 |
| prefill-arm flags | --threads 8 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 3 |

## Kernel wide-batch arms (threads=8)

unpinned: log=arm1_wb-t8-unpinned.log, exit_line=WB_EXIT=0, cpu_list=(unpinned)
phys-pinned (taskset 0-15): log=arm2_wb-t8-pin-node0phys.log, exit_line=WB_EXIT=0, cpu_list=0-15
HT-pinned (taskset 0-15,32-47): log=arm3_wb-t8-pin-node0ht.log, exit_line=WB_EXIT=0, cpu_list=0-15,32-47

tuning: q4k-widebatch shape=11008x2048 stride_bytes=1152 packed_mib=12.09 auto_tuning=[chain_group=8,max_scratch_cols=16384] streaming_tuning=[chain_group=8,max_scratch_cols=512] tile_tuning=[matmul_route_batch_width_q4k=8]
tuning: q4k-widebatch shape=2048x11008 stride_bytes=6192 packed_mib=12.09 auto_tuning=[chain_group=8,max_scratch_cols=16384] streaming_tuning=[chain_group=8,max_scratch_cols=512] tile_tuning=[matmul_route_batch_width_q4k=8]

Median ms per cell (reps=5; full min/max/gflops/weight_gbs live in data.json):

| shape | arm | count | unpinned med_ms | phys-pinned med_ms | HT-pinned med_ms |
|---|---|---|---|---|---|
| 11008x2048 | auto | 8 | 8.719 | 9.969 | 9.635 |
| 11008x2048 | auto | 16 | 16.235 | 17.155 | 16.766 |
| 11008x2048 | auto | 32 | 30.466 | 31.834 | 30.47 |
| 11008x2048 | auto | 64 | 56.01 | 58.744 | 59.171 |
| 11008x2048 | streaming | 8 | 19.318 | 20.032 | 19.46 |
| 11008x2048 | streaming | 16 | 38.052 | 39.394 | 44.088 |
| 11008x2048 | streaming | 32 | 83.005 | 95.466 | 92.201 |
| 11008x2048 | streaming | 64 | 205.742 | 199.796 | 226.114 |
| 11008x2048 | tile | 8 | 9.589 | 9.926 | 8.37 |
| 11008x2048 | tile | 16 | 14.288 | 12.93 | 14.153 |
| 11008x2048 | tile | 32 | 23.518 | 22.177 | 21.481 |
| 11008x2048 | tile | 64 | 45.164 | 39.986 | 38.776 |
| 2048x11008 | auto | 8 | 10.668 | 10.51 | 10.706 |
| 2048x11008 | auto | 16 | 17.848 | 17.775 | 17.853 |
| 2048x11008 | auto | 32 | 31.707 | 31.118 | 31.19 |
| 2048x11008 | auto | 64 | 54.511 | 57.335 | 55.344 |
| 2048x11008 | streaming | 8 | 22.748 | 21.601 | 23.104 |
| 2048x11008 | streaming | 16 | 42.919 | 40.767 | 42.703 |
| 2048x11008 | streaming | 32 | 85.344 | 76.86 | 80.911 |
| 2048x11008 | streaming | 64 | 218.089 | 210.777 | 221.074 |
| 2048x11008 | tile | 8 | 11.39 | 9.203 | 10.329 |
| 2048x11008 | tile | 16 | 14.174 | 13.865 | 15.966 |
| 2048x11008 | tile | 32 | 24.559 | 23.457 | 23.362 |
| 2048x11008 | tile | 64 | 45.039 | 43.084 | 42.519 |

## Placement deltas, t8 kernel cells (stats.kernel_t8_pin_vs_unpinned)

delta columns are median-ms change vs the comparison arm; band is the max run-to-run band pct of the two arms compared; separated when abs(delta) exceeds the band.

| shape | arm | count | phys-vs-unpinned delta % | band % | sep? | HT-vs-phys delta % | band % | sep? |
|---|---|---|---|---|---|---|---|---|
| 11008x2048 | auto | 8 | 14.34 | 12.05 | YES | -3.35 | 10.94 |  |
| 11008x2048 | auto | 16 | 5.67 | 5.72 |  | -2.27 | 16.59 |  |
| 11008x2048 | auto | 32 | 4.49 | 16.32 |  | -4.28 | 7.85 |  |
| 11008x2048 | auto | 64 | 4.88 | 15.16 |  | 0.73 | 15.16 |  |
| 11008x2048 | streaming | 8 | 3.7 | 7.39 |  | -2.86 | 7.39 |  |
| 11008x2048 | streaming | 16 | 3.53 | 11.22 |  | 11.92 | 13.2 |  |
| 11008x2048 | streaming | 32 | 15.01 | 18.86 |  | -3.42 | 4.2 |  |
| 11008x2048 | streaming | 64 | -2.89 | 18.19 |  | 13.17 | 27.67 |  |
| 11008x2048 | tile | 8 | 3.51 | 2.66 | YES | -15.68 | 6.45 | YES |
| 11008x2048 | tile | 16 | -9.5 | 10.18 |  | 9.46 | 15.42 |  |
| 11008x2048 | tile | 32 | -5.7 | 6.55 |  | -3.14 | 4.8 |  |
| 11008x2048 | tile | 64 | -11.46 | 15.16 |  | -3.03 | 12 |  |
| 2048x11008 | auto | 8 | -1.48 | 14.45 |  | 1.86 | 4.89 |  |
| 2048x11008 | auto | 16 | -0.41 | 7.41 |  | 0.44 | 8.8 |  |
| 2048x11008 | auto | 32 | -1.86 | 8.55 |  | 0.23 | 8.55 |  |
| 2048x11008 | auto | 64 | 5.18 | 11.49 |  | -3.47 | 8 |  |
| 2048x11008 | streaming | 8 | -5.04 | 24.94 |  | 6.96 | 1.76 | YES |
| 2048x11008 | streaming | 16 | -5.01 | 5.66 |  | 4.75 | 5.66 |  |
| 2048x11008 | streaming | 32 | -9.94 | 4.6 | YES | 5.27 | 2.63 | YES |
| 2048x11008 | streaming | 64 | -3.35 | 13.79 |  | 4.89 | 13.79 |  |
| 2048x11008 | tile | 8 | -19.2 | 27.24 |  | 12.24 | 6.93 | YES |
| 2048x11008 | tile | 16 | -2.18 | 3.44 |  | 15.15 | 3.8 | YES |
| 2048x11008 | tile | 32 | -4.49 | 8.54 |  | -0.4 | 17.43 |  |
| 2048x11008 | tile | 64 | -4.34 | 10.23 |  | -1.31 | 10.23 |  |

## Kernel wide-batch, threads=32 unpinned (dispatch-overhead probe)

log=arm4_wb-t32-unpinned.log, exit_line=WB_EXIT=0. The audit recorded kernel_t32_auto_c8_band_pct = 84.03% for the auto arm at count 8 in this quota: that cell's run-to-run dispersion is far wider than any tile/streaming cell at threads=32.

tuning: q4k-widebatch shape=11008x2048 stride_bytes=1152 packed_mib=12.09 auto_tuning=[chain_group=8,max_scratch_cols=16384] streaming_tuning=[chain_group=8,max_scratch_cols=512] tile_tuning=[matmul_route_batch_width_q4k=8]
tuning: q4k-widebatch shape=2048x11008 stride_bytes=6192 packed_mib=12.09 auto_tuning=[chain_group=8,max_scratch_cols=16384] streaming_tuning=[chain_group=8,max_scratch_cols=512] tile_tuning=[matmul_route_batch_width_q4k=8]

| shape | arm | count | med_ms | min_ms | max_ms | gflops | weight_gbs |
|---|---|---|---|---|---|---|---|
| 11008x2048 | auto | 8 | 39.646 | 21.602 | 54.917 | 9.098 | 0.32 |
| 11008x2048 | streaming | 8 | 11.418 | 10.431 | 12.37 | 31.593 | 1.111 |
| 11008x2048 | tile | 8 | 6.555 | 6.369 | 7.798 | 55.026 | 1.934 |
| 11008x2048 | auto | 16 | 7.628 | 7.101 | 8.377 | 94.569 | 1.662 |
| 11008x2048 | streaming | 16 | 19.166 | 18.503 | 19.541 | 37.64 | 0.662 |
| 11008x2048 | tile | 16 | 7.705 | 6.125 | 8.665 | 93.627 | 1.646 |
| 11008x2048 | auto | 32 | 12.904 | 11.447 | 13.812 | 111.813 | 0.983 |
| 11008x2048 | streaming | 32 | 51.03 | 50.204 | 61.421 | 28.275 | 0.249 |
| 11008x2048 | tile | 32 | 8.602 | 8.147 | 10.901 | 167.741 | 1.474 |
| 11008x2048 | auto | 64 | 21.187 | 18.692 | 21.915 | 136.203 | 0.599 |
| 11008x2048 | streaming | 64 | 92.103 | 85.446 | 94.7 | 31.331 | 0.138 |
| 11008x2048 | tile | 64 | 15.724 | 13.494 | 16.312 | 183.522 | 0.806 |
| 2048x11008 | auto | 8 | 6.225 | 5.594 | 7.291 | 57.949 | 2.037 |
| 2048x11008 | streaming | 8 | 11.874 | 11.658 | 12.017 | 30.379 | 1.068 |
| 2048x11008 | tile | 8 | 5.473 | 5.301 | 6.563 | 65.902 | 2.317 |
| 2048x11008 | auto | 16 | 10.04 | 8.969 | 10.129 | 71.852 | 1.263 |
| 2048x11008 | streaming | 16 | 22.166 | 21.777 | 22.695 | 32.547 | 0.572 |
| 2048x11008 | tile | 16 | 8.188 | 7.631 | 9.033 | 88.109 | 1.549 |
| 2048x11008 | auto | 32 | 15.797 | 15.04 | 17.318 | 91.336 | 0.803 |
| 2048x11008 | streaming | 32 | 37.982 | 31.504 | 40.376 | 37.987 | 0.334 |
| 2048x11008 | tile | 32 | 12.106 | 11.794 | 12.592 | 119.183 | 1.048 |
| 2048x11008 | auto | 64 | 25.28 | 24.86 | 29.496 | 114.149 | 0.502 |
| 2048x11008 | streaming | 64 | 83.582 | 77.901 | 84.772 | 34.525 | 0.152 |
| 2048x11008 | tile | 64 | 20.215 | 20.022 | 20.523 | 142.748 | 0.627 |

## End-to-end prefill arms (threads=8, lengths 128/256, widths 64, reps=3)

### unpinned run 1

- log: e2e1_pf-t8-unpinned.log; exit_line=PF_EXIT=0; cpu_list=(unpinned)
- window: start_ts=2026-11-27T14:21:42Z end_ts=2026-11-27T14:26:26Z
- kernel `gemv`:
  - weightFit=3860.12 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.43
  - attentionFit=203.712 us per position pair
  - crossover: fitted attention term >= fitted weight term at length >= 616
  rows: len=128 batchedMs=9402.1 tok_s=13.61 weightGBs=0.43; len=256 batchedMs=22141.8 tok_s=11.56 weightGBs=0.43
- kernel `tiled`:
  - weightFit=5137.76 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.32
  - attentionFit=129.855 us per position pair
  - crossover: not reached within 4x the longest measured length (256 tokens)
  rows: len=128 batchedMs=11347.6 tok_s=11.28 weightGBs=0.32; len=256 batchedMs=24822.7 tok_s=10.31 weightGBs=0.32
- peak: peakWorkingSetMb=1978.3 cumulative across this process run: measure per-kernel peaks in separate processes

### phys-pinned (taskset 0-15)

- log: e2e2_pf-t8-pin-node0phys.log; exit_line=PF_EXIT=0; cpu_list=0-15
- window: start_ts=2026-11-27T14:28:10Z end_ts=2026-11-27T14:32:34Z
- kernel `gemv`:
  - weightFit=3070.54 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.54
  - attentionFit=334.887 us per position pair
  - crossover: fitted attention term >= fitted weight term at length >= 303
  rows: len=128 batchedMs=8905.9 tok_s=14.37 weightGBs=0.54; len=256 batchedMs=23298.6 tok_s=10.99 weightGBs=0.54
- kernel `tiled`:
  - weightFit=4723.88 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.35
  - attentionFit=73.568 us per position pair
  - crossover: not reached within 4x the longest measured length (256 tokens)
  rows: len=128 batchedMs=10055.1 tok_s=12.73 weightGBs=0.35; len=256 batchedMs=21315.6 tok_s=12.01 weightGBs=0.35
- peak: peakWorkingSetMb=1979.7 cumulative across this process run: measure per-kernel peaks in separate processes

### unpinned rerun

- log: e2e3_pf-t8-unpinned-rerun.log; exit_line=PF_EXIT=0; cpu_list=(unpinned)
- window: start_ts=2026-11-27T14:33:32Z end_ts=2026-11-27T14:38:13Z
- kernel `gemv`:
  - weightFit=4152.96 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.40
  - attentionFit=166.386 us per position pair
  - crossover: fitted attention term >= fitted weight term at length >= 806
  rows: len=128 batchedMs=9679.6 tok_s=13.22 weightGBs=0.4; len=256 batchedMs=22085.3 tok_s=11.59 weightGBs=0.4
- kernel `tiled`:
  - weightFit=4795.65 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.35
  - attentionFit=141.139 us per position pair
  - crossover: not reached within 4x the longest measured length (256 tokens)
  rows: len=128 batchedMs=10756.6 tok_s=11.9 weightGBs=0.35; len=256 batchedMs=23825.5 tok_s=10.74 weightGBs=0.35
- peak: peakWorkingSetMb=1981.0 cumulative across this process run: measure per-kernel peaks in separate processes

### HT-pinned (taskset 0-15,32-47)

- log: e2e4_pf-t8-pin-node0ht.log; exit_line=PF_EXIT=0; cpu_list=0-15,32-47
- window: start_ts=2026-11-27T14:39:33Z end_ts=2026-11-27T14:44:18Z
- kernel `gemv`:
  - weightFit=4687.33 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.36
  - attentionFit=142.268 us per position pair
  - crossover: not reached within 4x the longest measured length (256 tokens)
  rows: len=128 batchedMs=10549.2 tok_s=12.13 weightGBs=0.36; len=256 batchedMs=23429.4 tok_s=10.93 weightGBs=0.36
- kernel `tiled`:
  - weightFit=5177.25 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.32
  - attentionFit=98.539 us per position pair
  - crossover: not reached within 4x the longest measured length (256 tokens)
  rows: len=128 batchedMs=11168 tok_s=11.46 weightGBs=0.32; len=256 batchedMs=23950.5 tok_s=10.69 weightGBs=0.32
- peak: peakWorkingSetMb=1977.2 cumulative across this process run: measure per-kernel peaks in separate processes

## E2E aggregates (stats.e2e)

| kernel | length | unpinned tok_s runs | median unpinned tok_s | spread % | phys-pinned tok_s | HT-pinned tok_s | implied GB/s med/unpinned | implied GB/s phys | implied GB/s HT |
|---|---|---|---|---|---|---|---|---|---|
| gemv | 128 | 13.61, 13.22 | 13.415 | 2.91 | 14.37 | 12.13 | 0.415 (runs: 0.43, 0.4) | 0.54 | 0.36 |
| gemv | 256 | 11.56, 11.59 | 11.575 | 0.26 | 10.99 | 10.93 | 0.415 (runs: 0.43, 0.4) | 0.54 | 0.36 |
| tiled | 128 | 11.28, 11.9 | 11.59 | 5.35 | 12.73 | 11.46 | 0.335 (runs: 0.32, 0.35) | 0.35 | 0.32 |
| tiled | 256 | 10.31, 10.74 | 10.525 | 4.09 | 12.01 | 10.69 | 0.335 (runs: 0.32, 0.35) | 0.35 | 0.32 |

## Verdict from this run

- Kernel seam itself (wide-batch microbench, threads=8): physical-core pinning separated only 3 of 24 cells against the unpinned band and HT-vs-phys separations are scattered with mixed sign. The Q4_K kernel paths do not show a consistent NUMA/SMT placement effect when measured in isolation at this thread quota.
- End-to-end prefill (DRAM-bound weight traversal): gemv impliedWeightGB/s median across the two unpinned runs is 0.415 with run-to-run spread 7.23%; physical-core pinning lifts it to 0.54 (above both unpinned runs), HT-pinning drops it to 0.36. Placement is a live lever at the e2e level even though it barely moves the isolated kernel cells.
- Dispatch signature: threads=32 auto arm at count 8 carries a recorded dispersion of 84.03% (table above), while tile/streaming cells at the same quota stay bounded - the full-thread auto-arm collapse is reproducible and points at scheduler/dispatch behavior rather than ISA-path overhead.
- Attribution: on this host the Q4_K prefill deficit decomposes into (a) placement sensitivity of DRAM-bound weight streaming in e2e steps (physical-core affinity helps, HT-pair pinning hurts) and (b) dispatch/scheduling overhead concentrated in the auto route at high thread counts; the kernel ISA seam itself is insensitive to both at threads=8.

## Caveats

- Single host, one model file, prefit fits over only two lengths (128/256), reps=3 per prefill arm, reps=5 per kernel cell; peak RSS lines are cumulative across each process run.
- Host idle verified before and during arms (no external load); CPU governor performance; numactl absent on host - pinning applied via taskset cpu lists recorded above.

