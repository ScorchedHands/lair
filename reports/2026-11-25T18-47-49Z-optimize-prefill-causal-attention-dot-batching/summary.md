# Prefill causal-attention shared-K batch dots 2026-11-25T18-47-49Z

Model: `C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf` (Qwen2.5-3B Q4_K_M, 2,104,932,768 bytes; arch=qwen2 layers=36 hidden=2048 heads=16 kvHeads=2 headDim=128 vocab=151936 avx2=True).
Lane: local CPU lane in `.agent-worktrees/mvp7`, auto thread posture (resolved threads=32), `rowWidth=64`, kernels gemv/tiled, lengths 128/256. No CUDA benchmarking; the broad gate excludes CUDA categories (`Category!=Cuda`).
Baseline reference: committed report `reports/2026-11-25T16-11-55Z-optimize-prefill-q6k-gemm/` (its stage log's own `attention_kernel_ms` medians are used below only as cross-session directional reference).

What changed: the float causal attention kernels (`CausalAttention`, `CausalAttentionTiledRows`) now walk each row in grouped-query head groups, so one K/V position load serves every head that shares it, and score dots against a shared K position go through an optional backend seam — `IDenseSharedInputDot.TryDotBatchSharedInput(inputs, count, common, outputs)` implemented by the CPU backend as `CpuBackend+SharedInputDot.cs`. The batch kernel keeps each accumulator chain in registers (arms 2..8 mirroring the tile executor's eight-chain idiom) and walks ascending eight-lane chunks with `DotCore`'s operand association, a per-chain scalar tail, and the same lane reduction before that tail. Grouping is over independent chains only; groups wider than the budget split into consecutive budget-sized chunks without changing any chain's accumulation order. Measurement pin: bench flag `--attention-batch-dots off|on|N` plus tuning knob `CpuKernelTuning.SharedInputDotChainGroup` (qualified key `engine.cpu.tuning.shared_input_dot_chain_group`, accepted 0..8; auto resolves to 8 with vector posture, 0 without — so the posture-resolved default equals pinned `on` here).

Files: new `src/Lair.Contracts/IDenseSharedInputDot.cs`; new `src/Lair.Backends.Cpu/CpuBackend+SharedInputDot.cs`; tuning plumbing in `src/Lair.Backends.Cpu/CpuKernelTuning.cs`; route wiring in `src/Lair.Models/Engine/LayerOps.cs` (both causal kernels, with exact fallback to per-head individual dots when the seam is absent, pinned off, or a group of one); bench pin + report echo in `bench/Lair.Bench/Program.cs`; new tests `tests/Lair.Tests/AttentionSharedDotTests.cs`.

## Bit identity

No tolerance was needed anywhere. `AttentionSharedDotTests` proves batch-vs-individual bit equality for common lengths 8/9/31/32/64/127/128 across counts 2..32 under both vector and scalar pinned postures; route-off leaves outputs untouched (fallback path only); scalar-posture batching equals `DotScalar` exactly; threaded vs single-threaded batches are identical; contract refusals/throws behave as specified; tuning resolution/pin/refusal behavior matches the existing CPU-knob conventions. Attention parity lanes compare route-on vs route-off for `CausalAttention` and `CausalAttentionTiledRows` across head counts/group mappings/head dimensions/softcap/window cases, including groups narrower than the batch budget, and prove tiled-row output equals per-row causal output bit-for-bit on both routes (windowed ragged fixtures). Full-model logit parity is carried by the existing Parity-category lanes inside the broad gate rather than re-run standalone this session.

## Method

Host noise between processes runs ±10-35% at window level on this box (documented in prior reports), so arms were run alternating `default -> off -> on`, 3 rounds each, for both axes: stage-profile (`--reps 1 --stage-profile`) and throughput (`--reps 3`). Medians below; all samples in the two `-v2-registerchains` raw logs and `data.json`. The posture-resolved default is identical to pinned `on` (chain group 8 on this AVX2 host), so the real A/B contrast is batched vs individual dots.

A first generation of the batch kernel kept its chain accumulators in stack scratch memory and measured *slower* than individual dots at every shape (`attention_kernel_ms`: gemv128 off 119.3 → on 134.3; gemv256 off 603.4 → default 650.2/on 712.0; tiled128 off 158.6 → on 184.8; tiled256 off 618.5 → on 671.9) — each chain paid a store+reload per chunk, which costs more than the re-reads batching removes. Replacing scratch chains with register-resident arms (switch arms for counts 2..8 plus a `DotCore` default arm) flipped every shape; raw logs of that superseded generation are kept in this directory (`ab-stage-profile.txt`, `ab-throughput.txt`) and documented in `commands.md`.

## attention_kernel_ms medians — the target substage (3 alternating rounds, rowWidth 64)

```text
gemv  length128   off 121.4 -> auto/pinned-on 109.7 / 111.2  (-9.7% / -8.4%)  full sample separation (off min 120.3 > both arms' max)
gemv  length256   off 590.0 -> auto/pinned-on 519.4 / 534.1  (-12.0% / -9.5%) full sample separation (on max 556.0 < off min 584.7)
tiled length128   off 174.5 -> auto/pinned-on 137.3 / 139.1  (-21.3% / -20.3%) full sample separation (off min 159.8 > both arms' max)
tiled length256   off 608.9 -> auto/pinned-on 518.0 / 503.0  (-14.9% / -17.4%) full sample separation (on max 564.3 < off min 591.1)
```

All four shapes separate cleanly across arms — this is the credible signal, and it matches the mechanism: one shared-K position load feeds every head in the group, the batch kernel re-enters the chunk walk once per group instead of once per head with chains resident in registers, and untouched substages (weight/projection/output) stay within arm noise.

## Window totals (stage-profile medians)

```text
gemv  length128   off 1329.1 -> auto/pinned-on 1311.3 / 1312.6  (-1.3% / -1.2%) samples overlap at the edges
gemv  length256   off 3190.0 -> auto/pinned-on 3154.7 / 3162.4  (-1.1% / -0.9%) samples overlap at the edges
tiled length128   off 1488.9 -> auto/pinned-on 1368.4 / 1385.0  (-8.1% / -7.0%) near-separated (auto max 1462.5 vs off min 1452.7; pinned-on fully separated)
tiled length256   off 3215.7 -> auto/pinned-on 2989.9 / 2968.5  (-7.0% / -7.7%) full sample separation (auto max 3133 < off min 3162)
```

The attention stage share of a gemv length-256 window fell from ~1042 ms (off, ≈33% of the 3190 ms median) to ~963 ms (auto, ≈31%). `attention_ms` medians: gemv256 off 1042.3 → auto 963.4 / on 981.4; tiled256 off 1035.4 → auto 949.4 / on 923.5.

## Throughput medians (batchedTok/s, --reps 3) — mixed at window level

```text
gemv  length128   off 96.9 -> default 99.9 (+3.1%) / on 100.0 (+3.2%)
gemv  length256   off 80.0 -> default 82.5 (+3.1%) / on 82.1 (+2.6%)
tiled length128   off 93.6 -> default 89.8 (-4.0%) / on 89.5 (-4.3%)
tiled length256   off 85.9 -> default 82.8 (-3.6%) / on 82.4 (-4.1%)
```

Window throughput at these shapes is dominated by weight traversal and swings run-to-run (±10-35% between host windows per prior reports' methodology notes); gemv arms favor batching while tiled samples straddle, so window tok/s neither contradicts nor amplifies the substage win and is treated as noise-indeterminate direction support only.

## Cross-session directional reference (different host session — not measurement)

The committed q6k-gemm lane's own stage logs recorded `attention_kernel_ms` medians of gemv128 ≈ 205, gemv256 ≈ 809-815, tiled128 ≈ 131-132, tiled256 ≈ 514-523. Current individual-dot (off) medians are 121/590/175/609 and batched medians 110-111/519-534/137-139/503-518: the gemv route is far below that session's numbers, tiled roughly comparable. Cross-session comparisons are directional reference only, per prior reports' convention; same-session alternating medians above are the measurement.

## Interpretation

- Shared-K batched score dots measurably reduce `attention_kernel_ms` at every measured shape: −8% to −21%, with full sample separation between arms everywhere. The win is largest on the tiled kernel, where row batches already amortize head-group walks and the register-resident chains remove the per-head re-entry into the chunk walk.
- Register residency was the deciding factor, not batching itself: scratch-memory chain accumulators regressed every shape (see Method), so the shipped kernel carries explicit arms for 2..8 chains plus a `DotCore` fallback arm outside that range.
- Window totals move with the substage where weight traversal is proportionally smaller (tiled −7% to −8%) and stay within noise on gemv windows (~−1%). Auto posture stays at chain group 8 because it is strictly better than individual dots at all four shapes here, bit-for-bit identical.
- Remaining attention-side target after this item: `attention_weight_ms`/projection/output remain comparable in size to the kernel substage inside a window (gemv256 auto arm ≈ 350/181/169 vs kernel 519 ms), so further wins need either weight traversal or the K/V load side, not score-dot batching alone.

## Validation counts

```text
dotnet build -c Release -> succeeded, 0 warnings / 0 errors
Category=Attention|Kernel|CpuTuning & Category!=Cuda -> 496/496 passed (32 s) [post-rewrite confirmation]
Category=Attention|Kernel|CpuTuning|Engine|Prefill & Category!=Cuda -> 681/681 passed (1 m 29 s)
FullyQualifiedName~AttentionSharedDot|AttentionAccumulatorTests|GemmKernelTests|Q6KGemmRouteTests -> 114/114 passed (1 s)
Category!=Cuda broad gate -> 2534/2534 passed, 0 skipped (5 m 48 s)
```

## Risks / open notes

- Auto chain group 8 was chosen from the grouped-query geometry (heads=16, kvHeads=2 → groups of exactly 8) and confirmed by alternating medians; a dedicated width sweep (2/4/6) was not run this session — `--attention-batch-dots N` supports it if a later pass wants to re-confirm.
- The seam is head-major-contiguous only: no strided overload exists, because grouped-query heads are adjacent in each row's layout; a model whose score-dot inputs were non-contiguous would fall back to individual dots rather than copy into a staging buffer.
- No public YAML/host-config wiring was added for `engine.cpu.tuning.shared_input_dot_chain_group` yet — pinning currently arrives through bench/test tuning overrides, matching the existing `MatmulRouteBatchWidth` precedent.
- Cross-host-session throughput comparisons remain unreliable on this box; only same-session alternating medians are treated as measurement.
- No output-bit tolerance introduced anywhere: parity is bit-identical in all new lanes (see Bit identity).
