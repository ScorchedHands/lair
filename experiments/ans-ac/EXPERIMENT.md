# ANS / Arithmetic Coding — Independent Experiment Plan

Goal: measure, independently of the engine's hot path first, whether AC and ANS shrink real quant weight bytes enough to justify decode cost. Run offline against files already in `models/`; only integrate into kernels after Phase 2 shows a viable decoder speed. Report **actual encoded bytes** (incl tables + offsets), never theoretical entropy alone, and state plainly when there is no benefit.

## Matrix: formats × codecs

Codecs: **AC** (range/carryless) vs **ANS** (interleaved rANS). Same symbol stream, same shared frequency model, same chunk boundaries for both.

Quant families to test (all already in the `QuantFormat` registry):
- **Q8_0** — 256-symbol codes; expect weakest compression (dense), kept as low-headroom baseline.
- **Q4_K** — 16-symbol magnitude stream, scales/mins kept as metadata.
- **IQ4_NL**, **IQ4_XS** — 16-symbol codebook-index streams (`kvalue` table).
- **IQ3_XXS**, **IQ3_S** — 8-symbol primary index plus high-bit/sign side planes (decide: compress primary only vs also the planes).

## Model fixtures in `models/` (skip-if-absent via `ModelFixture.Resolve()`)

| Family | File(s) already present |
|---|---|
| Q4_K | `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, `gemma-2-2b-it.Q4_K_M.gguf`, `smollm-135m-instruct-q4_k_m.gguf` |
| IQ4_XS / IQ3_XXS / IQ3_S (mixed) | `Llama-3.2-1B-Instruct-UD-IQ3_XXS.gguf` (carries Q4_K×16, Q5_K×1, IQ3_XXS×53, IQ4_XS×5, IQ3_S×16, …) |
| IQ4_XS | `Llama-3.2-1B-Instruct-IQ4_XS.gguf` |
| Q8_0 (baseline only — no 4-bit codes; for reference scale) | `Llama-3.2-1B-Instruct-Q8_0.gguf`, `smollm-135m-instruct-q8_0.gguf` |

Q8_0 has no sub-byte code stream in the Q4 sense but its 8-bit codes are still a compressible symbol alphabet; include it to show where compression stops paying. Stay under the documented **4 GB peak working set** per loaded engine (`DEVELOP.md`).

## Phase 0 — Offline entropy probe (no engine integration, cheapest first)

For each format in each file:
1. Extract the code stream(s) from mmap spans using the same reads the dequant kernels already do; keep scales/mins/codebook as metadata.
2. Build a shared frequency model at several group sizes {1 super-block, per-row, N rows}.
3. Encode with AC and rANS; compute **total bytes** = payload + tables + offsets/lengths/counts + framing + padding + kept metadata.
4. Compare to packed baseline (Q4_K 144 B/block, IQ4_XS 136/256, Q8_0 34/32, …) and to the fallback rule (per-chunk escape when not smaller).

Output per format/model: bits/symbol, model entropy vs fixed width, **compressed % of packed**, whole-model savings %, how often fallback fires. This alone answers "is there real headroom" with zero kernel risk. Expectation to verify, not assume: Q4/IQ4 have the most room; Q8 likely little.

## Phase 1 — Round-trip correctness harness (standalone)

Encode→decode exactness on random / uniform / skewed / **real** tiles across all six formats + AC and rANS. Cover empty/min chunks, all-one-symbol, all-symbols-present, boundary values, truncated/corrupt input, invalid offsets/lengths; verify encoder/decoder reconstruct identical tables from stored counts. Mirrors the `Category=Codec` test list in IMPLEMENTATION.md §12 before any engine code exists.

## Phase 2 — Decode throughput microbench (the go/no-go gate)

Measure AC and rANS decode at **ns/symbol** on real tiles, against LAIR's measured baselines (`DEVELOP.md` "Measured GEMV cost"): Q4_K resident single-thread ≈ `.164 ns/elt`; sub-4-bit staging decoders currently land ~**1.5–2.3×** the baseline (IQ2_S .247 … IQ3_XXS .383). A codec is worth carrying into kernels only if its decode sits in or below that band — scalar AC almost certainly will not; interleaved rANS may. This gate decides whether Phase 4 happens at all.

## Phase 3 — Encode-at-load provider spike (footprint + load cost)

Streaming encode-on-first-resolve → compressed arena in DRAM:
- **Load time** vs plain mmap-Q4 open.
- **Peak working set** with the compressed arena held, vs the packed baseline for the same model (`metrics_process_working_set_mb` style reading). This is where a real resident-footprint win should show up (compressed-in-DRAM replaces packed Q4) — but confirm encode-at-load's transient mmap+arena peak doesn't erase it.
- **Cache-resident vs cold** using the bench private-copy idea already in `gemv-file … --prefetch on|off|compare` (`DEVELOP.md` Bench section), which separates page residency from kernel cost.

## Phase 4 — Staged/fused decode in matmul spike (only if Phase 2 passes)

Reuse existing GEMM scaffolding via a staged arm (decode chunk → scratch → existing tile/FMA). Measure, on one served fixture:
- **bytes fetched** per weight traversal and implied GB/s (`prefill-profile` already splits a "weight term ms/traversal + implied GB/s"),
- matmul/forward time with decode vs packed baseline,
- end-to-end tok/s via the documented serve surface (one server at a time), cache-resident and cold cases.

Compare AC vs rANS as **separate config runs diffed from logs/bench reports** — no in-engine runtime comparator (PLAN DN-M7-10 modularity invariant).

## Reporting rules

Report actual encoded size, total bytes incl metadata, encode/decode time, bytes fetched where instrumentable, matmul and end-to-end throughput. Test several chunk/group sizes aligned to the tile layout. Commit numbers under `experiments/ans-ac/reports/`. State clearly when AC or ANS provides no size or speed benefit on a given format/model — that outcome is a valid result, not a failure.

## Sequencing / stop conditions

- If Phase 0 shows little headroom for Q8 (and modest for IQ3), narrow later phases to the formats with real savings rather than forcing all six everywhere.
- If Phase 2 decode speed exceeds the bandwidth it would save, document AC/ANS as size-only (useful for disk tier / cold fetch) and do **not** pursue fused kernels — that is an acceptable conclusion.
