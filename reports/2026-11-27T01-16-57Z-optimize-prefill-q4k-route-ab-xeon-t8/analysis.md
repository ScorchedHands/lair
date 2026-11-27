# Q4_K dense matmul-route A/B medians -- Xeon t8, qwen2.5-3b-instruct-q4_k_m

All numbers below are parsed from the downloaded `data.json` files in this directory (script-computed medians of 3 same-session stage-profile rounds; sample lists and min/max spreads included). Nothing here is estimated.

- `baseline-off` -- route OFF; remote report started `2026-11-27T01:16:57Z`, rounds=3, binary `Lair.Bench.dll`, model `qwen2.5-3b-instruct-q4_k_m.gguf`
- `route-on-w8` -- route ON, width 8 (script default `on` maps to 8); remote report started `2026-11-27T01:25:12Z`, rounds=3, binary `Lair.Bench.dll`, model `qwen2.5-3b-instruct-q4_k_m.gguf`
- `route-w16` -- route ON, explicit width 16; remote report started `2026-11-27T01:34:43Z`, rounds=3, binary `Lair.Bench.dll`, model `qwen2.5-3b-instruct-q4_k_m.gguf`

## gemv @ length 128

| metric | off median (samples) | on-w8 median (delta vs off) | w16 median (delta vs off) |
| --- | ---: | ---: | ---: |
| total_ms | 10426.2 (11476.007, 10426.204, 9337.293) | 9479.8 (-9.1%) | 9388.7 (-10.0%) |
| feed_forward_gate_up_ms | 4266.9 (4871.673, 4266.929, 3868.165) | 4083.8 (-4.3%) | 4164.8 (-2.4%) |
| feed_forward_down_ms | 3523.4 (3840.346, 3523.381, 3141.657) | 3159.4 (-10.3%) | 3006.7 (-14.7%) |
| attention_weight_ms | 1578.2 (1804.027, 1578.202, 1447.228) | 1349.0 (-14.5%) | 1357.7 (-14.0%) |

## gemv @ length 256

| metric | off median (samples) | on-w8 median (delta vs off) | w16 median (delta vs off) |
| --- | ---: | ---: | ---: |
| total_ms | 23552.5 (22665.900, 23552.477, 24460.026) | 22445.8 (-4.7%) | 21747.2 (-7.7%) |
| feed_forward_gate_up_ms | 9426.9 (9414.526, 9426.881, 10053.611) | 9695.4 (+2.8%) | 9399.5 (-0.3%) |
| feed_forward_down_ms | 8000.0 (7461.541, 8072.988, 8000.003) | 7286.4 (-8.9%) | 7041.7 (-12.0%) |
| attention_weight_ms | 3575.6 (3492.679, 3575.554, 3759.586) | 3190.3 (-10.8%) | 3094.1 (-13.5%) |

## tiled @ length 128

| metric | off median (samples) | on-w8 median (delta vs off) | w16 median (delta vs off) |
| --- | ---: | ---: | ---: |
| total_ms | 10732.1 (10732.061, 10415.574, 11123.447) | 11375.3 (+6.0%) | 10692.3 (-0.4%) |
| feed_forward_gate_up_ms | 4494.0 (4494.021, 4278.896, 4731.975) | 4820.3 (+7.3%) | 4637.5 (+3.2%) |
| feed_forward_down_ms | 3703.3 (3595.550, 3703.291, 3867.101) | 3949.1 (+6.6%) | 3568.2 (-3.6%) |
| attention_weight_ms | 1579.6 (1683.216, 1444.639, 1579.590) | 1569.4 (-0.6%) | 1483.0 (-6.1%) |

## tiled @ length 256

| metric | off median (samples) | on-w8 median (delta vs off) | w16 median (delta vs off) |
| --- | ---: | ---: | ---: |
| total_ms | 23963.1 (23963.124, 24132.337, 23105.649) | 22216.8 (-7.3%) | 23362.6 (-2.5%) |
| feed_forward_gate_up_ms | 9678.6 (9678.568, 9837.067, 9551.243) | 9175.6 (-5.2%) | 9918.3 (+2.5%) |
| feed_forward_down_ms | 8039.6 (8039.567, 8558.992, 7893.896) | 7448.6 (-7.4%) | 7873.6 (-2.1%) |
| attention_weight_ms | 3309.9 (3640.012, 3309.882, 3189.841) | 2987.0 (-9.8%) | 3159.3 (-4.6%) |

## Sample noise -- (max-min)/median per variant

| combo | metric | off | on-w8 | w16 |
| --- | --- | ---: | ---: | ---: |
| gemv@128 | total_ms | 20.5% | 3.0% | 6.4% |
| gemv@128 | feed_forward_gate_up_ms | 23.5% | 4.9% | 6.9% |
| gemv@128 | feed_forward_down_ms | 19.8% | 4.7% | 5.2% |
| gemv@128 | attention_weight_ms | 22.6% | 3.6% | 10.9% |
| gemv@256 | total_ms | 7.6% | 6.1% | 7.8% |
| gemv@256 | feed_forward_gate_up_ms | 6.8% | 9.2% | 9.7% |
| gemv@256 | feed_forward_down_ms | 7.6% | 2.9% | 6.1% |
| gemv@256 | attention_weight_ms | 7.5% | 2.4% | 10.5% |
| tiled@128 | total_ms | 6.6% | 22.0% | 8.1% |
| tiled@128 | feed_forward_gate_up_ms | 10.1% | 32.0% | 8.7% |
| tiled@128 | feed_forward_down_ms | 7.3% | 9.3% | 8.5% |
| tiled@128 | attention_weight_ms | 15.1% | 22.7% | 11.6% |
| tiled@256 | total_ms | 4.3% | 10.1% | 1.3% |
| tiled@256 | feed_forward_gate_up_ms | 3.0% | 20.8% | 2.6% |
| tiled@256 | feed_forward_down_ms | 8.3% | 3.5% | 2.0% |
| tiled@256 | attention_weight_ms | 13.6% | 12.6% | 3.3% |

## total_ms medians: baseline vs candidates

| combo | off median ms | on-w8 (delta) | w16 (delta) |
| --- | ---: | ---: | ---: |
| gemv@128 | 10426.2 | 9479.8 (-9.1%) | 9388.7 (-10.0%) |
| gemv@256 | 23552.5 | 22445.8 (-4.7%) | 21747.2 (-7.7%) |
| tiled@128 | 10732.1 | 11375.3 (+6.0%) | 10692.3 (-0.4%) |
| tiled@256 | 23963.1 | 22216.8 (-7.3%) | 23362.6 (-2.5%) |

## Conclusion

- gemv: route ON improves total_ms at both lengths for both widths (width 8 -9.1% @128, -4.7% @256; width 16 -10.0% @128, -7.7% @256), and the gain tracks the expected stages: `feed_forward_down_ms` and `attention_weight_ms` fall while `feed_forward_gate_up_ms` stays within noise.
- width 16 is at least as good as width 8 on every gemv combo (-10.0% vs -9.1% @128, -7.7% vs -4.7% @256).
- tiled: mixed. Width 8 shows -7.3% at length 256 but +6.0% at length 128; width 16 shows -2.5% @256 and -0.4% @128. The tiled@128 width-8 median is not trustworthy: its own sample spread reaches 22.0% of the median and its gate/up stage spreads 32.0%, so a +/-6% reading sits inside that noise band.
- Across all four combos, mean-of-medians total_ms delta is -3.8% for width 8 and -5.1% for width 16; width 16 has no positive (slower) combo in this run while width 8 does.
- Sample spreads above 20% of median were seen in these rows, which is why single-digit deltas should not be read as decisive: 
  - `route-on-w8` tiled@128 feed_forward_gate_up_ms: spread 32.0%
  - `baseline-off` gemv@128 feed_forward_gate_up_ms: spread 23.5%
  - `route-on-w8` tiled@128 attention_weight_ms: spread 22.7%
  - `baseline-off` gemv@128 attention_weight_ms: spread 22.6%
  - `route-on-w8` tiled@128 total_ms: spread 22.0%
  - `route-on-w8` tiled@256 feed_forward_gate_up_ms: spread 20.8%
  - `baseline-off` gemv@128 total_ms: spread 20.5%
- Baseline `gemv@128` also carries a wide spread (20.5% of median, samples 11476/10426/9337), with its slowest sample in the first round; that cold-start shape appears in several variants and inflates any length-128 comparison.
- Verdict on this evidence: the Q4_K matmul route looks faster for `gemv` prefill at both lengths, slightly more so at width 16 than at the default width 8; `tiled` is improved at length 256 and unresolved-to-slightly-worse at length 128 within noise. These are 3-round medians with up to ~30% intra-variant spread, so a rounds=4 repeat of baseline vs width 8 (and ideally width 16) is needed before any promotion decision.
