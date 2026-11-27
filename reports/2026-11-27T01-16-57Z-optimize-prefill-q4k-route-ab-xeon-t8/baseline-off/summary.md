# Prefill stage profile Xeon t8 2026-11-27T01-16-57Z-optimize-prefill-stage-profile-xeon-t8

Lane: Xeon CPU-only lair-lab (root@192.168.33.53), threads=8, rowWidth=64. Method: 3 alternating same-session stage-profile reps=1 commands with kernels gemv, tiled; medians/min/max parsed from raw logs after ANSI stripping.

## Length 128 medians (ms)
| kernel | total | gate/up | down | attention_weight | projection | output | kernel | logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemv | 10426.2 | 4266.9 | 3523.4 | 1578.2 | 827.6 | 750.6 | 351.7 | 96.4 |
| tiled | 10732.1 | 4494.0 | 3703.3 | 1579.6 | 883.6 | 661.8 | 396.7 | 106.3 |

## Length 256 medians (ms)
| kernel | total | gate/up | down | attention_weight | projection | output | kernel | logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemv | 23552.5 | 9426.9 | 8000.0 | 3575.6 | 1835.6 | 1739.9 | 1279.4 | 115.8 |
| tiled | 23963.1 | 9678.6 | 8039.6 | 3309.9 | 1873.4 | 1388.0 | 1462.8 | 107.3 |

Interpretation by total_ms medians:
- length 128: fastest total_ms kernel is `gemv`; slowest is `tiled`.
- length 256: fastest total_ms kernel is `gemv`; slowest is `tiled`.
- dominant stages around `gemv` length 256: `feed_forward_gate_up_ms`, `feed_forward_down_ms`, `attention_weight_ms`
