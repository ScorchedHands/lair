# LAIR prefill row-width probe — dual Xeon E5-2683 v4 (Broadwell, AVX2+FMA)

As of 2026-11-27. Measurement-only run; no engine code was modified. Host: `lair-lab` = root@192.168.33.53, dual-socket E5-2683 v4 (16C/socket, 64 logical CPUs, node0 = 0–15 + HT siblings 32–47), governor `performance`, no numactl installed. LAIR checkout `/root/lair-measurements/qwen25-small-cpu/repo` at HEAD `016dd9c` (`phase/mvp7`). App: `build/app-posture-2026-11-27T06-53-30Z/Lair.Bench.dll`, run with `/usr/local/dotnet/dotnet`. Model: `qwen2.5-3b-instruct-q4_k_m.gguf` (2,104,932,768 B; engine-reported traversalBytes = 1,668,685,824). llama reference: `/root/lair-tools/llama-b12032/llama-bench`. Raw logs remain only on the host under `logs/` (not copied locally); every number below is parsed from them into `data.json` by a parse pass run on the host.

## Question and verdict

Question: does LAIR prefill throughput scale with prompt row width, or do kernels cap effective weight amortization well below the stated sub-batch width?
**Verdict: candidate B.** Kernel-internal grouping already caps effective amortization far below the stated row width (measured plateau ≈ 5–6× at t8 and ≤ ~24× at t32 against theoretical rows-per-pass of 32/64), and widening from 32→64 buys only −10%…+70%, never a near-proportional gain. Raising the engine's `MaxPrefillRowWidth` ceiling alone will not close the gap to llama.cpp; kernels must be restructured first so each weight traversal actually amortizes its stated rows efficiently (implied per-pass bandwidth *falls* when width doubles — see fit lines below).

## 1. Width axis — batched tok/s, threads=8 (`arm1`, reps=3) and threads=32 (`arm2`, reps=3); cross-check `--per-token` arms at reps=1

threads = 8:

| kernel | len | w32 (r3 / r1) | w64 (r3 / r1) | ratio w64/w32 (r3, r1) |
|---|---|---|---|---|
| gemv | 128 | 11.79 / 12.89 | 11.60 / 11.65 | 0.98, 0.90 |
| gemv | 256 | 9.61 / 10.12 | 11.00 / 11.26 | 1.14, 1.11 |
| tiled | 128 | 10.30 / 10.24 | 13.35 / 10.60 | 1.30, 1.04 |
| tiled | 256 | 9.61 / 9.47 | 12.06 / 10.39 | 1.25, 1.10 |
| gemm | 128 | 10.06 / 10.70 | 10.23 / 11.91 | 1.02, 1.11 |
| gemm | 256 | 9.57 / 9.96 | 10.00 / 10.33 | 1.04, 1.04 |

threads = 32:

| kernel | len | w32 (r3 / r1) | w64 (r3 / r1) | ratio w64/w32 (r3, r1) |
|---|---|---|---|---|
| gemv | 128 | 18.19 / 18.12 | 21.96 / 21.76 | 1.21, 1.20 |
| gemv | 256 | 20.15 / 17.80 | 19.94 / 20.51 | 0.99, 1.15 |
| tiled | 128 | 20.19 / 19.55 | 23.47 / **31.10** | 1.16, **1.59** |
| tiled | 256 | 18.73 / 17.15 | 22.47 / **29.12** | 1.20, **1.70** |
| gemm | 128 | 15.16 / 15.36 | 18.57 / 19.15 | 1.22, 1.25 |
| gemm | 256 | 13.90 / 13.93 | 17.95 / 17.35 | 1.29, 1.25 |

**Through throughput roughly double when width doubles? No.** Every ratio lies between 0.90 and 1.70; the only ratios above ~1.4 are the tiled@t32 w64 values in the reps=1 arm (see variance note §5). At threads=8, doubling stated width is essentially neutral for gemv/gemm (−10%…+14%) and at best +30% for tiled.

## 2. Amortization factor = batchedTok/s ÷ perTokenTok/s (`arm3a` t8, `arm3b` t32; reps reduced to 1 — see Deviations)

Theoretical rows-per-pass equals the stated width (32 or 64). "Fraction" below = measured factor ÷ stated width.

threads = 8:

| kernel | w | len | batched | perToken | factor | fraction of width |
|---|---|---|---|---|---|---|
| gemv | 32 | 128 | 12.89 | 1.97 | 6.54 | 20% |
| gemv | 32 | 256 | 10.12 | 1.97 | 5.14 | 16% |
| gemv | 64 | 128 | 11.65 | 2.05 | 5.68 | 9% |
| gemv | 64 | 256 | 11.26 | 1.89 | 5.96 | 9% |
| tiled | 32 | 128 | 10.24 | 2.11 | 4.85 | 15% |
| tiled | 32 | 256 | 9.47 | 1.94 | 4.88 | 15% |
| tiled | 64 | 128 | 10.60 | 2.03 | 5.22 | 8% |
| tiled | 64 | 256 | 10.39 | 1.89 | 5.50 | 9% |
| gemm | 32 | 128 | 10.70 | 2.12 | 5.05 | 16% |
| gemm | 32 | 256 | 9.96 | 2.05 | 4.86 | 15% |
| gemm | 64 | 128 | 11.91 | 1.96 | 6.08 | 9% |
| gemm | 64 | 256 | 10.33 | 1.98 | 5.22 | 8% |

threads = 32:

| kernel | w | len | batched | perToken | factor | fraction of width |
|---|---|---|---|---|---|---|
| gemv | 32 | 128 | 18.12 | 1.35 | 13.42 | 42% |
| gemv | 32 | 256 | 17.80 | 1.32 | 13.48 | 42% |
| gemv | 64 | 128 | 21.76 | 1.30 | 16.74 | 26% |
| gemv | 64 | 256 | 20.51 | 1.30 | 15.78 | 25% |
| tiled | 32 | 128 | 19.55 | 1.30 | 15.04 | 47% |
| tiled | 32 | 256 | 17.15 | 1.27 | 13.50 | 42% |
| tiled | 64 | 128 | **31.10** | 1.28 | **24.30** | 38% |
| tiled | 64 | 256 | **29.12** | 1.32 | **22.06** | 34% |
| gemm | 32 | 128 | 15.36 | 1.35 | 11.38 | 36% |
| gemm | 32 | 256 | 13.93 | 1.32 | 10.55 | 33% |
| gemm | 64 | 128 | 19.15 | 1.36 | 14.08 | 22% |
| gemm | 64 | 256 | 17.35 | 1.37 | 12.66 | 20% |

Reading: at threads=8 the factor is a flat ~5–6× regardless of kernel and **regardless of stated width** (fraction falls from ≈15–20% at w32 to ≈8–9% at w64) — effective amortization plateaus near 5–6 rows per weight traversal. At threads=32 the factor rises with thread count only because the single-token path itself degrades (~2 tok/s → ~1.3 tok/s), and even then it never exceeds 24× against a theoretical 64 (best case = 38% of stated width). Kernels amortize **partially, far below stated width**; they do not re-walk weights per token (perTokenTok/s is essentially width-independent) but neither do they carry the full sub-batch across one traversal.

### Verbatim `weightFit` / implied-bandwidth lines (one per kernel×width block; note how implied bandwidth *falls* when width doubles)

- arm1 t8 gemv w32: `weightFit=2091.68 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.80`
- arm1 t8 gemv w64: `weightFit=5214.24 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.32`
- arm1 t8 tiled w32: `weightFit=2879.88 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.58`
- arm1 t8 tiled w64: `weightFit=4277.01 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.39`
- arm1 t8 gemm w32: `weightFit=3015.65 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.55`
- arm1 t8 gemm w64: `weightFit=6114.57 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.27`
- arm2 t32 gemv w32: `weightFit=1931.02 ms/traversal (UNRELIABLE: measured costs are not separable into the two terms on this run) impliedWeightGB/s=0.86`
- arm2 t32 gemv w64: `weightFit=2615.68 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.64`
- arm2 t32 tiled w32: `weightFit=1460.12 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=1.14`
- arm2 t32 tiled w64: `weightFit=2604.69 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.64`
- arm2 t32 gemm w32: `weightFit=1917.90 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.87`
- arm2 t32 gemm w64: `weightFit=3326.87 ms/traversal (least-squares fit over 2 lengths) impliedWeightGB/s=0.50`
- arm3a t8(per-token, r1) gemv w32 `impliedWeightGB/s=0.93`; gemv w64 `weightFit=5303.55 ... impliedWeightGB/s=0.31`; tiled w32 `... impliedWeightGB/s=0.58`; tiled w64 `weightFit=5910.16 ... 0.28`; gemm w32 `... 0.60`; gemm w64 `weightFit=4543.77 ... 0.37`
- arm3b t32(per-token, r1) gemv w32 `... impliedWeightGB/s=0.96`; gemv w64 `weightFit=2761.27 ... 0.60`; tiled w32 `... impliedWeightGB/s=1.19` (best on this host); tiled w64 `weightFit=1916.80 ... 0.87`; gemm w32 `... 0.89`; gemm w64 `weightFit=2993.48 ... 0.56`
- arm4 t32 stage-profile gemv w64: `weightFit=3155.20 ms/traversal (UNRELIABLE: measured costs are not separable into the two terms on this run) impliedWeightGB/s=0.53`; tiled w64 `... 0.62`; gemm w64 `... 0.58`

Every arm's implied weight-traversal bandwidth sits in 0.27–1.19 GB/s — consistent with the previously established 0.47–1.33 GB/s envelope, and every doubling of stated width cuts per-pass efficiency roughly in half (e.g., gemv t8: 0.93→0.31; tiled t8: 0.58→0.28), i.e., wider passes re-walk weights with worse amortization rather than carrying their full row budget through one pass.

## 3. Stage shares at width 64, threads 32 (`arm4`, reps=2) — share of `total_ms` per kernel×length

| kernel | len | attention_weight | attn_projection | attn_output | attn_kernel | **gate_up** | down | total_ms |
|---|---|---|---|---|---|---|---|---|
| gemv | 128 | 14.8% | 7.7% | 7.1% | 11.9% | **31.9%** | 25.5% | 12225.1 |
| gemv | 256 | 13.9% | 7.2% | 6.8% | 19.4% | **28.9%** | 23.5% | 24925.8 |
| tiled | 128 | 14.4% | 7.6% | 6.7% | 14.4% | **30.3%** | 25.1% | 11790.2 |
| tiled | 256 | 13.0% | 6.7% | 6.3% | 24.8% | **26.9%** | 22.1% | 26652.2 |
| gemm | 128 | 18.0% | 12.3% | 5.7% | 11.8% | **33.9%** | 20.8% | 13083.1 |
| gemm | 256 | 16.0% | 11.0% | 5.1% | 20.6% | **30.4%** | 18.9% | 29309.4 |

Dominant named substages at the widest posture: `feed_forward_gate_up_ms` is the single largest substage for every kernel (≈27–34%), followed by `feed_forward_down_ms`; attention weight/projection terms add another ~13–18% + 6–12%. Weight-family FF terms dominate each step — exactly where per-traversal amortization must be fixed.

## 4. llama.cpp reference (`llama-bench` b12032, `-ngl 0 -n 0`, defaults `-b 2048 -ub 512`) and LAIR÷llama ratios

| posture | pp128 | pp256 | pp512 |
|---|---|---|---|
| t=8 | 70.69 ± 4.83 | 74.45 ± 2.56 | 69.33 ± 4.64 |
| t=32 | 228.18 ± 10.14 | 239.48 ± 9.10 | 227.23 ± 5.56 |

llama's rate is flat across pp128→pp512 because its default ubatch (512, from `--help`: `-b, --batch-size <n> (default: 2048)` / `-ub, --ubatch-size <n> (default: 512)`) already covers every prompt tested — one wide weight pass serves all rows. LAIR(width 64, r3 batched arm) ÷ llama(pp256 | pp512):

| kernel/posture | len128 vs pp256/pp512 | len256 vs pp256/pp512 |
|---|---|---|
| gemv t8 (11.60/11.00) | 0.156 / 0.167 | 0.148 / 0.159 |
| tiled t8 (13.35/12.06) | 0.179 / 0.193 | 0.162 / 0.174 |
| gemm t8 (10.23/10.00) | 0.137 / 0.148 | 0.134 / 0.144 |
| gemv t32 (21.96/19.94) | 0.092 / 0.097 | 0.083 / 0.088 |
| tiled t32 (23.47/22.47; r1 arm up to 31.10/29.12) | 0.098–0.130 / 0.103–0.137 | 0.094–0.122 / 0.099–0.128 |
| gemm t32 (18.57/17.95) | 0.078 / 0.082 | 0.075 / 0.079 |

LAIR at its widest tested posture runs ≈6–14% of llama's rate; widening to 128+ rows per pass (llama-style amortization) is what closes that gap, but §1 shows LAIR kernels currently *lose* efficiency as passes widen — so the ceiling raise only pays after kernel restructuring.

## 5. Deviations, anomalies, and variance notes

- **Run 3 reps deviation:** `--per-token` mode costs ≈ length×reps single forwards (≈0.5 s/token at t8, ≈0.75 s/token at t32 on this host); the spec's implied `--reps 3` would have run each arm for hours (first attempt aborted after 10 min wall / ~71 min CPU with no table emitted — partial log kept as evidence). Runs 3a/3b were completed at `--reps 1`; lengths/widths/kernels/postures unchanged per spec. Rates are unaffected by reps, so factor comparisons remain valid; batchedTok/s in these arms cross-check the r3 arms within a few percent except tiled@t32 (below).
- **Aborted attempt evidence:** `logs/arm3a_t8_perToken_gemv-tiled-gemm_w32-64_l128-256_r3.log` (truncated after one block header, 501 B) — see commands.md for its SHA-256.
- **Fit non-separability:** two blocks report `(UNRELIABLE: ...)` fits with negative `attentionMs` columns (arm2 gemv w32 t32 rows −688.3/−2742.7 ms; arm4 gemv w64 rows −238.8/−951.6 ms). Their tok/s and weightGB_s values are still used above; the *fit split* is not trustworthy for those two blocks only.
- **tiled@threads=32 variance:** batchedTok/s at tiled/w64/t32 measured 23.47–22.47 in the r3 arm vs 31.10/29.12 in the r1 per-token arm — run-to-run spread up to ~+30%; conclusions above hold across both values (ratios remain far from proportional).
- **No kernel refusals anywhere:** `grep -iE 'refus|unsupported|not supported'` over all logs returns none; gemm ran every span.

## 6. Left unmeasured

Widths > 64 (e.g., 128/256 — the regime where llama-style amortization lives); per-token baselines at threads=4 or with reps>1; stage profiling at width 32 and t8; NUMA/taskset-pinned variants (`taskset` exists but was not applied, matching prior sweeps' posture).