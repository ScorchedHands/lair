# Prefill LAIR vs llama.cpp Xeon comparison 2026-11-26T08-46-39Z-optimize-prefill-vs-llama-xeon-t8

Lane: Xeon CPU-only lair-lab (root@192.168.33.53), threads=8, rowWidth=64. Model bytes: `2104932768`.
Method: alternating same-session rounds with reps=1 per command; medians/min/max parsed from raw logs.

| harness | samples | median tok/s | min | max |
| --- | ---: | ---: | ---: | ---: |
| LAIR gemv 128 | 12.820, 11.870, 13.380 | 12.820 | 11.870 | 13.380 |
| LAIR tiled 128 | 12.440, 11.330, 12.750 | 12.440 | 11.330 | 12.750 |
| llama.cpp b12032 pp128 | 73.800, 70.220, 70.890 | 70.890 | 70.220 | 73.800 |
| LAIR gemv 256 | 10.320, 10.560, 11.560 | 10.560 | 10.320 | 11.560 |
| LAIR tiled 256 | 12.000, 10.560, 12.230 | 12.000 | 10.560 | 12.230 |
| llama.cpp b12032 pp256 | 74.010, 72.730, 72.370 | 72.730 | 72.370 | 74.010 |

| length | LAIR gemv vs llama pp | LAIR tiled vs llama pp |
| ---: | ---: | ---: |
| 128 | 18.08% | 17.55% |
| 256 | 14.52% | 16.50% |

Interpretation: LAIR's current optimized CPU prefill posture is still below the same-host llama.cpp pp band; ratios are medians only and must be read against host noise.
