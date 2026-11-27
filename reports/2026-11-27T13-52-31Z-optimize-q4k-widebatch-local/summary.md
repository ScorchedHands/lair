# Q4_K wide-batch kernel attribution, developer box (2026-11-27T13-52-31Z)

Seam: `bench/Lair.Bench q4k-widebatch` — synthetic pinned Q4_K tensors at the real Qwen2.5-3B feed-forward widths (`11008x2048` gate/up, `2048x11008` down), counts 8/16/32/64 (and 1 in an earlier draft run), arms measured as a set with two untimed settle passes and rotated-order timed rounds (`--reps >= 3`, true median), plus an inter-arm bit-parity guard. Posture: developer's local AVX2 box, `avx2=yes`. Raw logs are committed here verbatim; this page is the reading of them.

## Local kernel efficiency (from `q4k-wb-clean-t8-c512.log`, threads=8)

| shape | arm | count=8 med_ms | ns/MAC | GFLOP/s | weight GB/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 11008x2048 | auto (scratch route) | 1.399 | 0.0078 | 257.9 | 9.067 |
| 11008x2048 | streaming fallback | 3.155 | 0.0175 | 114.3 | 4.020 |
| 11008x2048 | tile executor (`q4k-matmul-route 8`) | 1.779 | 0.0099 | 202.7 | 7.127 |
| 11008x2048 | colblock512 (rejected candidate) | 8.079 | 0.0448 | 44.6 | 1.570 |

The auto scratch route is the best local traversal at every count ≥ 8, reaching ~9 GB/s effective weight bandwidth and ~32 GFLOP/s per thread — an order of magnitude above what the same kernels measure on the Xeon target host (`reports/2026-11-27T10-05-02Z-optimize-prefill-rowwidth-xeon`: 0.27–1.3 GB/s implied). The tile executor only wins on the `down` shape at count 64 (0.032 vs 0.041 ns/MAC in the t1 runs), consistent with the earlier decision to keep `MatmulRouteBatchWidthQ4K = 0`.

## Rejected candidate: wide-batch column-block staging (`WidebatchColBlockQ4K`)

Staging contiguous per-chain activation slices inside superblock-aligned column blocks regresses at every eligible posture on this box — e.g. threads=8 gate/up count=64 colblock512 0.0134 ns/MAC versus auto 0.0048 (≈2.8× worse), and the wider c1024 block is worse than c512; threads=1 shows the same shape of loss (gate/up count=64: 89 ms vs 48 ms). Root cause observed in code terms: staging copies activations per weight row, while the auto route's far-strided reads already feed every chain from L1-hot staged scratch without extra traffic. Bit-parity held (`arms_bit_parity=bit_identical`) — the candidate is correct but slower, so it was **reverted**: no kernel, tuning constant, or bench pin for it remains in this tree (verified by grep across `src/`, `bench/`, `tests/`).

## What these numbers change about the plan

The deficit versus llama.cpp on Xeon cannot be explained by generic LAIR kernel quality — locally the same kernels hit ~9 GB/s. The remaining question is host-specific: why identical AVX2 code delivers an order of magnitude less per thread on the dual-socket Broadwell target (NUMA first-touch placement, cross-node weight streams, SMT contention, or dispatch overhead are the open candidates), which is what a pinned-vs-unpinned `q4k-widebatch` run on `lair-lab` now has to answer.
