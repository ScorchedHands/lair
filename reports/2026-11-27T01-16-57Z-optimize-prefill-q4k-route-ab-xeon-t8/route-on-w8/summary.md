# Prefill stage profile Xeon t8 2026-11-27T01-25-12Z-optimize-prefill-stage-profile-xeon-t8

Lane: Xeon CPU-only lair-lab (root@192.168.33.53), threads=8, rowWidth=64. Method: 3 alternating same-session stage-profile reps=1 commands with kernels gemv, tiled; medians/min/max parsed from raw logs after ANSI stripping.

## Length 128 medians (ms)
| kernel | total | gate/up | down | attention_weight | projection | output | kernel | logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemv | 9479.8 | 4083.8 | 3159.4 | 1349.0 | 798.3 | 549.1 | 328.8 | 108.6 |
| tiled | 11375.3 | 4820.3 | 3949.1 | 1569.4 | 967.3 | 602.0 | 432.0 | 106.3 |

## Length 256 medians (ms)
| kernel | total | gate/up | down | attention_weight | projection | output | kernel | logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemv | 22445.8 | 9695.4 | 7286.4 | 3190.3 | 1894.5 | 1270.9 | 1260.6 | 109.0 |
| tiled | 22216.8 | 9175.6 | 7448.6 | 2987.0 | 1813.0 | 1174.1 | 1603.5 | 104.2 |

Interpretation by total_ms medians:
- length 128: fastest total_ms kernel is `gemv`; slowest is `tiled`.
- length 256: fastest total_ms kernel is `tiled`; slowest is `gemv`.
- dominant stages around `gemv` length 256: `feed_forward_gate_up_ms`, `feed_forward_down_ms`, `attention_weight_ms`
