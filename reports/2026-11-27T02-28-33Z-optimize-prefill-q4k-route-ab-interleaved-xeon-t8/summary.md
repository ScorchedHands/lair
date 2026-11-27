# Prefill q4k route A/B interleaved Xeon t8 2026-11-27T02-28-33Z-optimize-prefill-q4k-route-ab-interleaved-xeon-t8

Lane: Xeon CPU-only lair-lab (root@192.168.33.53), threads=8, rowWidth=64. Method: 4 rounds x arms [off, on, 16] interleaved back-to-back in one session so host drift cancels; reps=1 per command with kernels gemv, tiled; medians/min/max/spread parsed from raw logs after ANSI stripping.

## Length 128 medians (ms)
### gemv
| arm | total | gate/up | down | attention_weight |
| --- | ---: | ---: | ---: | ---: |
| off | 10360.6 | 4358.8 | 3503.4 | 1630.5 |
| on | 8734.2 | 3817.3 | 2819.2 | 1231.2 |
| 16 | 9573.0 | 4172.6 | 3133.6 | 1317.9 |

### tiled
| arm | total | gate/up | down | attention_weight |
| --- | ---: | ---: | ---: | ---: |
| off | 11337.9 | 4789.1 | 3959.7 | 1607.9 |
| on | 11232.5 | 4908.7 | 3839.3 | 1563.4 |
| 16 | 10438.2 | 4400.6 | 3760.0 | 1445.8 |

## Length 256 medians (ms)
### gemv
| arm | total | gate/up | down | attention_weight |
| --- | ---: | ---: | ---: | ---: |
| off | 23598.6 | 9681.5 | 7784.1 | 3669.4 |
| on | 22582.3 | 9748.4 | 7295.8 | 3163.3 |
| 16 | 22144.3 | 9509.7 | 7236.7 | 3041.2 |

### tiled
| arm | total | gate/up | down | attention_weight |
| --- | ---: | ---: | ---: | ---: |
| off | 23143.8 | 9540.9 | 7974.3 | 3194.4 |
| on | 23214.5 | 9846.6 | 7775.7 | 3121.8 |
| 16 | 22933.0 | 9600.5 | 7917.4 | 3093.7 |

## Noise bands (intra-arm total_ms spread = (max-min)/median)
| kernel | length | arm off % | arm on % | arm 16 % |
| ---: | ---: | ---: | ---: | ---: |
| gemv | 128 | 20.22 | 22.64 | 23.89 |
| gemv | 256 | 8.97 | 5.99 | 13.22 |
| tiled | 128 | 13.32 | 26.57 | 14.70 |
| tiled | 256 | 12.17 | 11.47 | 8.50 |
- pooled worst-case total_ms spread for arm `off`: 20.22%
- pooled worst-case total_ms spread for arm `on`: 26.57%
- pooled worst-case total_ms spread for arm `16`: 23.89%

## Interpretation (total_ms medians vs baseline)
Baseline arm: `off`.
- `gemv` length 128: `on` delta -15.70% vs noise band ±22.64% -> separates beyond own spread: no; `16` delta -7.60% vs noise band ±23.89% -> separates beyond own spread: no
- `gemv` length 256: `on` delta -4.31% vs noise band ±8.97% -> separates beyond own spread: no; `16` delta -6.16% vs noise band ±13.22% -> separates beyond own spread: no
- `tiled` length 128: `on` delta -0.93% vs noise band ±26.57% -> separates beyond own spread: no; `16` delta -7.94% vs noise band ±14.70% -> separates beyond own spread: no
- `tiled` length 256: `on` delta +0.31% vs noise band ±12.17% -> separates beyond own spread: no; `16` delta -0.91% vs noise band ±12.17% -> separates beyond own spread: no

