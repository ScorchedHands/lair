# Alternating build A/B — Linux self-placement default vs prior posture build (headline extract)

Lane: Xeon `lair-lab`, threads=8 explicit, rowWidth 64, length 256 only, kernels gemv/tiled, reps=1 per command with stage profiling; harness `scripts/run-xeon-alternating-build-ab.py` alternates two published builds in interleaved rounds (3 warmup + 13 measured, odd-round order rotation). Old app: `/root/lair-measurements/qwen25-small-cpu/build/app-posture-2026-11-27T06-53-30Z/Lair.Bench.dll` (published earlier this session from the pre-affinity checkout at `016dd9c`, behaviorally identical `prefill-profile`). New app: `/root/lair-measurements/qwen25-small-cpu/build/app-affinity-2026-11-27T23Z/Lair.Bench.dll` (published from a clean detached worktree of committed HEAD `1dcb354`, which lands the Linux physical-core placement + auto sizing). Full host artifacts live at `/root/lair-measurements/qwen25-small-cpu/reports/2026-11-27T23-07-15Z-optimize-affinity-selfplacement-xeon-t8/`; this file is a headline extract.

## Medians (ms), new = self-placement build, old = prior posture build
| kernel | metric | old median [band] | new median [band] | delta | separates? |
|---:|---|---:|---:|---:|---|
| gemv | total_ms | 21350.6 ±15.16% | 22006.4 (same pooled band) | +3.07% | no |
| gemv | feed_forward_down_ms | 6964.4 | 7322.6 | +5.14% | no (±23.08%) |
| gemv | feed_forward_gate_up_ms | 8752.9 | 9185.1 | +4.94% | no (±16.78%) |
| gemv | attention_output_ms | 1564.8 | 1632.2 | +4.31% | no (±17.53%) |
| tiled | total_ms | 23414.3 ±15.76% | 22930.1 | −2.07% | no |
| tiled | feed_forward_down_ms | 8155.4 | 7904.2 | −3.08% | no (±11.94%) |
| tiled | feed_forward_gate_up_ms | 9779.0 | 9578.4 | −2.05% | no (±26.66%) |
| tiled | attention_output_ms | 1722.2 | 1809.4 | +5.06% | no (±28.70%) |

## Verdict as measured
Nothing separates beyond noise at this power, and the sign is mixed: gemv windows run slightly *slower* on the self-placement build while tiled totals run slightly faster. The engine-side placement/auto-sizing change therefore does **not** reproduce the earlier whole-process `taskset -c 0-15` gain (0.415→0.54 implied GB/s, +7…14% tok/s) at window level with worker-thread-only affinity — consistent with the hypothesis that most of the manual-pinning benefit came from *whole-process* posture (loader/first-touch weight pages and all engine threads), which this build's derivation does not cover: it places model worker threads after open, leaving load-time residency to whatever CPUs the loading path happens on. This run neither refutes the measured placement lever nor validates its current implementation as a default win; the next experiment must test whole-process posture (e.g., manual taskset against both builds, or loader/main-thread participation in the placed core set) before any claim that self-placement is a throughput improvement.
