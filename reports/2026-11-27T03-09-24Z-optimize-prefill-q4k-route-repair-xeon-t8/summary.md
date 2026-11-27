# Prefill q4k route A/B interleaved Xeon t8 2026-11-27T03-09-24Z-optimize-prefill-q4k-route-repair-xeon-t8

Lane: Xeon CPU-only lair-lab (root@192.168.33.53), threads=8, rowWidth=64. Method: 2 warmup rounds then 5 measured rounds x arms [off, on, 16] interleaved back-to-back in one session so host drift cancels; measured arm order rotates on odd rounds to cancel positional bias while warmups run each arm once per round for cold-start/page-cache handling; reps=1 per command with kernels gemv, tiled; medians/min/max/spread parsed from raw measured logs after ANSI stripping.

## Length 128 medians (ms)
### gemv
| arm | total | gate/up | down | attention_weight |
| --- | ---: | ---: | ---: | ---: |
| off | 9650.9 | 4089.3 | 3245.2 | 1518.4 |
| on | 8910.4 | 3946.8 | 2840.0 | 1259.5 |
| 16 | 9863.4 | 4255.3 | 3202.7 | 1431.6 |

### tiled
| arm | total | gate/up | down | attention_weight |
| --- | ---: | ---: | ---: | ---: |
| off | 10050.4 | 4319.0 | 3566.9 | 1385.4 |
| on | 11416.9 | 4894.0 | 3940.1 | 1550.2 |
| 16 | 10785.4 | 4617.5 | 3715.4 | 1545.5 |

## Length 256 medians (ms)
### gemv
| arm | total | gate/up | down | attention_weight |
| --- | ---: | ---: | ---: | ---: |
| off | 23375.3 | 9444.7 | 7906.7 | 3659.7 |
| on | 22982.1 | 9945.7 | 7426.5 | 3168.7 |
| 16 | 21724.6 | 9325.7 | 7027.3 | 3015.5 |

### tiled
| arm | total | gate/up | down | attention_weight |
| --- | ---: | ---: | ---: | ---: |
| off | 23053.4 | 9595.3 | 8078.8 | 3160.2 |
| on | 22430.6 | 9439.3 | 7540.9 | 2937.2 |
| 16 | 23354.6 | 9922.5 | 7822.7 | 3169.5 |

## Noise bands (intra-arm total_ms spread = (max-min)/median)
| kernel | length | arm off % | arm on % | arm 16 % |
| ---: | ---: | ---: | ---: | ---: |
| gemv | 128 | 25.53 | 34.20 | 26.04 |
| gemv | 256 | 8.54 | 13.39 | 5.90 |
| tiled | 128 | 16.81 | 18.54 | 14.47 |
| tiled | 256 | 4.30 | 7.74 | 11.08 |
- pooled worst-case total_ms spread for arm `off`: 25.53%
- pooled worst-case total_ms spread for arm `on`: 34.20%
- pooled worst-case total_ms spread for arm `16`: 26.04%

## Interpretation (total_ms medians vs baseline)
Baseline arm: `off`.
- `gemv` length 128: `on` delta -7.67% vs noise band ±34.20% -> separates beyond own spread: no; `16` delta +2.20% vs noise band ±26.04% -> separates beyond own spread: no
- `gemv` length 256: `on` delta -1.68% vs noise band ±13.39% -> separates beyond own spread: no; `16` delta -7.06% vs noise band ±8.54% -> separates beyond own spread: no
- `tiled` length 128: `on` delta +13.60% vs noise band ±18.54% -> separates beyond own spread: no; `16` delta +7.31% vs noise band ±16.81% -> separates beyond own spread: no
- `tiled` length 256: `on` delta -2.70% vs noise band ±7.74% -> separates beyond own spread: no; `16` delta +1.31% vs noise band ±11.08% -> separates beyond own spread: no

