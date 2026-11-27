# Prefill stage profile Xeon t8 2026-11-27T01-34-43Z-optimize-prefill-stage-profile-xeon-t8

Lane: Xeon CPU-only lair-lab (root@192.168.33.53), threads=8, rowWidth=64. Method: 3 alternating same-session stage-profile reps=1 commands with kernels gemv, tiled; medians/min/max parsed from raw logs after ANSI stripping.

## Length 128 medians (ms)
| kernel | total | gate/up | down | attention_weight | projection | output | kernel | logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemv | 9388.7 | 4164.8 | 3006.7 | 1357.7 | 793.8 | 564.0 | 314.6 | 100.8 |
| tiled | 10692.3 | 4637.5 | 3568.2 | 1483.0 | 880.0 | 603.0 | 415.0 | 99.5 |

## Length 256 medians (ms)
| kernel | total | gate/up | down | attention_weight | projection | output | kernel | logits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemv | 21747.2 | 9399.5 | 7041.7 | 3094.1 | 1812.5 | 1325.6 | 1295.3 | 105.2 |
| tiled | 23362.6 | 9918.3 | 7873.6 | 3159.3 | 1882.5 | 1273.2 | 1448.5 | 108.0 |

Interpretation by total_ms medians:
- length 128: fastest total_ms kernel is `gemv`; slowest is `tiled`.
- length 256: fastest total_ms kernel is `gemv`; slowest is `tiled`.
- dominant stages around `gemv` length 256: `feed_forward_gate_up_ms`, `feed_forward_down_ms`, `attention_weight_ms`
