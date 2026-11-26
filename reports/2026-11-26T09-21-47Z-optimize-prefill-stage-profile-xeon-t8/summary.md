# Prefill stage profile Xeon t8 2026-11-26T09-21-47Z-optimize-prefill-stage-profile-xeon-t8

Lane: Xeon CPU-only lair-lab (root@192.168.33.53), threads=8, rowWidth=64. Method: parsed three completed raw `--reps 1 --stage-profile` logs after ANSI stripping.

## Length 128 medians (ms)
| kernel | total | gate/up | down | attention_weight | projection | output | kernel | logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemv | 10242.9 | 4187.5 | 3535.4 | 1534.3 | 783.1 | 751.2 | 337.7 | 109.1 |
| tiled | 10328.7 | 4156.6 | 3511.5 | 1580.8 | 782.7 | 759.1 | 382.3 | 109.5 |
| gemm | 18699.1 | 10936.2 | 4104.4 | 2755.2 | 1691.5 | 1095.2 | 344.0 | 80.0 |

## Length 256 medians (ms)
| kernel | total | gate/up | down | attention_weight | projection | output | kernel | logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemv | 23876.5 | 9739.0 | 8154.3 | 3633.7 | 1860.1 | 1773.7 | 1232.6 | 108.8 |
| tiled | 23050.8 | 9504.6 | 8133.2 | 3215.4 | 1782.4 | 1670.8 | 1477.6 | 105.3 |
| gemm | 38520.4 | 22347.2 | 8234.6 | 5718.5 | 3440.4 | 2278.2 | 1290.0 | 93.9 |

Interpretation by total_ms medians:
- length 128: fastest total_ms kernel is `gemv`; slowest is `gemm`.
- length 256: fastest total_ms kernel is `tiled`; slowest is `gemm`.
- dominant stages around `gemv` length 256: `feed_forward_gate_up_ms`, `feed_forward_down_ms`, `attention_weight_ms`
