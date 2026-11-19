# Conclusion — ANS / rANS vs Arithmetic Coding on GGUF quant weights

Plain-language summary of what the experiment showed and what the CPU fused-decode discussion concluded. Numbers below are from `reports/ans_ac_results.json` (340 records, 12 models incl. the `Qwen3.8-Flash-Next` MoE, all round-trips verified) plus roofline estimates for the 2× Xeon E5-2683 v4 target.

## 1. Compression result (measured)

- **Overall ~10% smaller** than the raw packed quant baseline. Modest — quant codes are already fairly uniform, so little redundancy remains.
- **ANS and arithmetic coding are effectively tied on size** (10.05% vs 10.01% overall). Neither wins on bytes.
- **Grouping matters more than codec choice:** per-block ~7–8%, whole tensor/layer/model ~11%. The gain comes from sharing one frequency table over many weights, not from the coder.
- **Per-expert grouping is weak (~2.7%):** coding each expert separately re-pays table overhead; it often falls back to raw bytes. Useful for structure, not size.
- **~32% of cases (55/170) compressed nothing** — table cost exceeded savings, so the chunk stored as raw. Honest signal that entropy-coding these formats only pays off on large chunks.
- **Speed is the real differentiator:** rANS decoded ~2.5× faster than AC in the Python harness (~88 ms vs ~215 ms avg). rANS is the practical choice: same size, faster decode.

## 2. CPU fused-decode conclusion (2× E5-2683 v4)

Target: 32 cores / 64 threads, AVX2, 256KB L2/core, 40MB L3/socket, DDR4-2133 quad-channel ≈68 GB/s/socket (~136 GB/s aggregate, NUMA-split).

- **The win is gated by rANS decode throughput, not memory.** Scalar rANS is a serial state machine (~1–2 symbols/cycle → ~2–4 G symbols/s/core). Saturating ~136 GB/s at compressed ~0.45 B/symbol needs ~300 G symbols/s, so scalar rANS is **decode-bound** and fusion never gets exercised — it can regress vs a good baseline.
- **Fusion only pays off with interleaved rANS (4–8 states/core) or SIMD-ANS** (~8–16 G symbols/s/core → ~256–1000 G/s aggregate). Then decode exceeds DRAM and the in-register design + compression actually help.
- **Cache/register placement is the right shape:** probability table + dequant scale/min + codebook in **L2**; decoded symbols produced **directly into AVX2 registers** and consumed by the FMA chain with **no fp16/fp32 intermediate written to L3/DRAM**. L3 only for batch reuse.
- **Register pressure is the cap:** rANS state + table pointers + accumulator compete with the matmul's 16 GPRs / 16 AVX2 regs; keep interleave degree modest, tables compact, no indirect fetch in the hot loop.

## 3. Correction to an earlier over-optimistic note

An earlier turn suggested a large (2–5×) MoE win from "decoding only active experts." That is **wrong**: baseline inference already reads only active experts. rANS does **not** skip inactive experts. The MoE benefit from rANS is **footprint** (compressed experts reduce active-expert DRAM traffic and improve cache residency), roughly the same ~10–20% as dense — **not** a decode-throughput multiplier.

## 4. Realistic tokens/sec estimates (batch-1 GEMV, roofline)

Assumptions: NUMA-aware effective aggregate bandwidth ~110–130 GB/s; compute peak ~2 TFLOP/s fp32 with GEMV effective ~40–65%; baseline = modern on-the-fly dequant (reads quantized bytes, dequant into registers, no fp intermediate — already efficient). rANS-best = interleaved/SIMD rANS, tables in L2, in-register dequant.

| Model class | Baseline tok/s | rANS-best tok/s | rANS scalar tok/s | Boost (best) |
|---|---|---|---|---|
| Dense 1B-class Q4_K_M (~1.24B params, ~0.70 GB) | ~90–115 | ~100–130 | ~80–110 | ~1.1–1.2× (scalar ~0.9–1.0×) |
| Dense 7–8B-class Q4_K_M (~7.3B, ~4.1 GB) | ~15–20 | ~17–22 | ~14–19 | ~1.1–1.2× (scalar ~0.9–1.0×) |
| MoE 30B-A3B class (~3B active params/token, ~1.7 GB/token) | ~35–45 | ~39–50 | ~32–42 | ~1.1–1.2× (scalar ~0.9–1.0×) |
| `Qwen3.8-Flash-Next` IQ3_XXS MoE (active experts large) | low-tens | +~10–20% | ≈ baseline | ~1.1–1.2× |

Reading the table: **rANS is roughly equal to slightly faster than a good baseline (~1.0–1.2×), and only when decode is interleaved/SIMD.** With scalar rANS it is about neutral or slightly slower because the serial decode cannot keep up with the matmul. The durable value of rANS here is **memory footprint / cache residency**, not a tokens/sec multiplier.

## 5. Modern Xeon (AVX-512 / AVX-VNNI / AMX — Sapphire Rapids, Emerald Rapids, Granite Rapids)

Representative target: single socket, AVX-512 + VNNI + AMX, DDR5-4800 8-channel ≈300 GB/s/socket, L2 2MB/core, L3 64–112MB.

What changes vs the E5-2683 v4 case:
- **The decode bottleneck largely disappears.** AVX-512 SIMD-ANS processes ~16 symbols/instruction (512-bit), and 32 ZMM registers remove the Broadwell register-pressure cap. Interleaved/SIMD rANS now easily saturates the matmul, so the fused design is reliably reachable rather than "only if you vectorize or it regresses."
- **Bigger L2/L3 make the placement contract fit with headroom:** tables in 2MB L2, decoded tiles and active-expert weights more often resident in L3 → more on-chip, less DRAM.
- **Compute is no longer the limiter** (VNNI/AMX), so inference is bandwidth-bound; compression's byte reduction translates more directly into throughput, especially with batching (AMX removes the compute cap entirely).

Estimates (batch-1 GEMV, single socket):

| Model class | Baseline tok/s | rANS-best tok/s | Boost |
|---|---|---|---|
| Dense 1B Q4_K_M | ~200–350 | ~230–420 | ~1.1–1.2× |
| Dense 7–8B Q4_K_M | ~40–70 | ~45–85 | ~1.1–1.2× |
| MoE 30B-A3B (~3B active/token) | ~80–150 | ~90–180 | ~1.1–1.3× |
| Batched (AMX, batch ≥8) | — | — | ~1.15–1.4× (byte reduction maps more directly) |

Caveats: AVX-512 heavy use down-clocks, so peak compute numbers are optimistic; AMX helps batched GEMM, not single-token GEMV; the ratio stays ~1.1–1.3× because baseline already dequantizes on the fly and already reads only active experts. The larger absolute tokens/sec comes from the hardware, not from rANS. The clearest modern-Xeon advantage is **footprint/cache-residency**: compression pushing a model into L3 (or fitting more active experts on-chip) is where gains can exceed the ~1.2× bandwidth figure.

## 6. Consumer desktop — i9-13900K + 64GB DDR5

Target: 8 P-cores + 16 E-cores (24 threads), **AVX2 + AVX-VNNI** (Raptor Lake has VNNI int8/int16 dot-product but **no AVX-512**), DDR5-4800/5600 dual-channel ≈77–85 GB/s, L3 36MB, L2 1.25MB/P-core.

- **AVX-VNNI speeds the matmul side, not the rANS decode side.** VNNI (`VPMADDUSDS`/`VPMADDWD`) raises the rate at which the GEMV consumes dequantized weights, but rANS decode is table-lookup + serial state, not a dot product — VNNI gives it no help. Net effect: VNNI makes the **decode bottleneck worse** (the matmul wants weights faster than scalar rANS can produce them), so interleaved/SIMD-ANS is even more mandatory.
- **Same decode-bottleneck story as the Xeon case.** Scalar rANS on 8 P-cores delivers only ~16–32 G symbols/s, below the ~87 G symbols/s needed for a 1B model at ~70 tok/s → **scalar rANS regresses** (~0.6–0.9×). Interleaved/SIMD-ANS required for parity or better.
- **Bandwidth is lower than the dual-Xeon** (~80 vs ~136 GB/s), so absolute tokens/sec is lower and the model is firmly bandwidth-bound; VNNI does not lift the bandwidth-bound ceiling for GEMV.

Estimates (batch-1 GEMV, AVX2+VNNI, interleaved rANS-best):

| Model class | Baseline tok/s | rANS-best tok/s | rANS scalar | Boost |
|---|---|---|---|---|
| Dense 1B Q4_K_M | ~60–78 | ~66–95 | ~40–70 (slower) | ~1.1–1.2× best |
| Dense 7–8B Q4_K_M | ~10–14 | ~11–17 | ~7–12 | ~1.1–1.2× best |
| MoE 30B-A3B (~3B active/token) | ~22–35 | ~24–40 | ~15–28 | ~1.1–1.2× best |

**Capacity note:** the `Qwen3.8-Flash-Next` MoE (~75GB IQ3_XXS) does **not** fit 64GB even compressed (~67GB), so on this desktop it needs disk/system offload regardless of rANS. The ~10% reduction is not enough to make the giant MoE resident here.

## 7. CUDA / GPU version

Representative targets: RTX 4090 24GB / 1TB/s; RTX 6000 Ada 48GB / 960GB/s; A100 80GB / 2TB/s; H100 80GB / 3.35TB/s; 4×4090 = 96GB.

**Decode is not the bottleneck on GPU.** rANS decode is massively parallel: thousands of independent interleaved streams run concurrently, tables live in shared memory, and serial per-stream dependency is hidden by parallelism. GPU decode throughput is enormous, so the fused decode+dequant+matmul kernel is comfortably fed — the Broadwell/13900K "decode-bound" failure mode does not apply.

**What rANS buys on GPU is VRAM capacity + reduced VRAM traffic, not a decode speedup:**
- Storing weights compressed cuts VRAM footprint (~10% measured) and reduces bytes read per token (bandwidth-bound kernels get proportionally faster).
- The dominant GPU constraint for big models is **VRAM capacity**, so the win is "fits / leaves room for KV+context," not raw tokens/sec.

Estimates (model resident in VRAM):

| Model class | GPU | Baseline tok/s | rANS-best tok/s | Boost |
|---|---|---|---|---|
| Dense 1B Q4_K_M | 4090 | ~250–400 | ~265–430 | ~1.05–1.1× |
| Dense 7–8B Q4_K_M | 4090 | ~120–180 | ~130–200 | ~1.1× |
| MoE 30B-A3B (~13GB IQ3_XXS) | 4090 24GB | ~100–160 | ~110–180 | ~1.1× |
| Large MoE (Qwen3.8-Flash-Next) | 80GB-class | offload-bound | see §8 | footprint-bound |

## 8. GPU memory-savings analysis for the large MoE

`Qwen3.8-Flash-Next` IQ3_XXS ≈ **75GB weights** (~175B params at ~3.44 bpw). This is the case where compression matters most, but the savings are modest and the *kind* of savings matters:

- **Resident-tier savings are small.** Measured compression is ~10% overall and only ~2.7% for per-expert grouping, so 75GB → ~67–69GB, i.e. ~6–8GB freed. On an 80GB card (A100/H100) that is the difference between "weights alone fit with ~5GB headroom" and "weights fit with ~13GB headroom for KV cache + activations" — enough to avoid OOM at longer context / larger batch. It does **not** make 75GB fit on a 48GB card.
- **The real lever is the offloaded-expert tier, not entropy coding.** A 175B-param MoE only activates ~10/512 experts per token, so only a small active-expert set must be VRAM-resident; the bulk can live in system RAM / NVMe and be streamed per token. Here rANS helps where it counts most:
  - Compressed inactive experts shrink the **system-RAM / NVMe store** and cut **PCIe transfer volume** per active-expert gather (~10% fewer bytes). On PCIe4 (~32GB/s) / PCIe5 (~64GB/s) offload, that directly raises MoE tokens/sec when the expert gather is PCIe-bound.
  - Combined with active-expert selection, the practical MoE path is: keep active experts + shared experts compressed in VRAM, keep inactive experts compressed in system RAM/NVMe. rANS improves both the resident headroom and the offload bandwidth.
- **Caveat:** because per-expert chunks compress poorly (~2.7%) due to per-expert table overhead, the expert-tier savings lean toward **larger shared groupings** (whole expert tensor) rather than per-expert tables. The measured data says: group experts to share a table, don't entropy-code each expert separately.

**Net GPU conclusion:** rANS on GPU is primarily a **capacity/offload enabler** for the large MoE (frees ~6–8GB on 80GB-class cards, reduces PCIe offload traffic ~10%), with only a small tokens/sec gain (~1.1×) when the model is already VRAM-resident. Decode is never the GPU bottleneck; the value is fitting the model and feeding the offload tier.

## 9. Prefill and KV cache — where rANS fits and where it does not

Scope: rANS benefits/caveats only (plus modified approaches that reuse it). No other compression methods.

**Weights (decode and prefill) — good fit.** rANS/ANS are sequential decoders over a symbol stream, which matches weight reads: each weight is read once per forward pass, in order.
- **Decode (batch-1):** bandwidth-bound; rANS reduces streamed weight bytes (~10%). Benefit as in §4–§8.
- **Prefill:** prefill reads all weights once per batched pass. For a **small** model prefill is compute-bound, so fewer weight bytes barely helps. For a **large model that exceeds cache** (the big MoE especially), prefill is bandwidth-bound too, so the same rANS weight-streaming gives a proportional prefill bandwidth win (~10%). This is the same rANS mechanism, just applied to the prefill weight read — a modified path that genuinely benefits.

**KV cache — poor fit as-is.** KV entries are continuous activations, and attention reads them at **arbitrary/random positions** per token:
- rANS needs a discrete symbol alphabet → you must first quantize KV to codes before entropy-coding anything.
- rANS/ANS are **sequential**: you cannot seek to token *i* without decoding up to it. That collides with random-access KV reads in attention.
- Per-token / small-block KV chunks are tiny → per-chunk table overhead dominates → weak compression (the same failure mode as per-expert grouping, ~2.7%).
- Decode-on-read adds latency to the hot attention loop.

**Modified KV paths that *can* benefit from rANS:**
- **KV offload to system RAM/NVMe** (capacity/PCIe-bandwidth problem, like expert offload): rANS compresses the offloaded KV to cut PCIe transfer volume — **but only if KV is stored in large contiguous, sequentially-read pages** (paged/block KV read in big pages). Block-aligned, independently-decodable KV pages make rANS viable for the offload tier; per-token granularity makes it not.
- **Quantized-KV storage capacity:** when KV is written sequentially (append during prefill) and read back in **large blocks** (chunked attention), rANS on quantized-KV blocks can shrink KV footprint (a capacity win). The random-access caveat limits it to **block granularity**, so compression is modest and each block must be independently decodable.

**Net for prefill/KV:** rANS is worth applying to **weight streams** (decode + prefill when the model exceeds cache) and to **offloaded weight/expert tiers**. It is a poor fit for **random-access KV reads** and for **tiny per-token/per-expert chunks**; it only helps KV when KV is reorganized into large, block-random-accessible pages — and even then the gain is a capacity/offload-bandwidth win, not a tokens/sec multiplier.

## 11. Variations / alternative encodings (research-grounded, 2024–2025)

Our experiment entropy-codes **already-4-bit GGUF block codes**, which is why the win is only ~10%: per-block quantization flattens the symbol distribution, leaving little redundancy for the coder. Research shows variations with **similar or much larger** benefits, mostly by changing *what* is entropy-coded or *how* it is decoded.

**A. Entropy-code INT8-quantized weights, not 4-bit.** "On the Compressibility of Quantized LLMs" (arXiv 2403.01384): int8 quant + a lossless coder → **~8× weights / 16× activations** (quant 4×, entropy coding adds **~2× weights, 4× activations**). Entropy coding has far more headroom on int8 than on 4-bit. Caveat: int8 footprint before coding is 2× 4-bit, but the coder recovers it and can beat 4-bit-with-small-coding.

**B. Tensor-level / large-group quantization → skewed codes → strong entropy coding.** EntroLLM (arXiv 2505.02380): block quant "flattens" the integer set and *raises* entropy; **tensor-level** quant keeps a "spiky" low-entropy distribution, so Huffman compresses **8-bit → 5.58 bits (1.32× measured speedup)** and **4-bit → 1.39 bits (~146% faster generation)** on Jetson. This is the biggest lever for LAIR: switching from per-block to tensor-level (or larger-group) scales creates the redundancy rANS/Huffman needs. Caveat: coarser scales can cost accuracy; EntroLLM picks unsigned/asymmetric per layer.

**C. Sub-4-bit codebook / vector quantization + entropy-code the indices.** QuIP# (E8P lattice, arXiv 2402.04396), QTIP (trellis), AQLM, ParetoQ reach **2-bit / ternary** = 4–8× footprint — far beyond entropy coding on 4-bit. Entropy-coding the codebook indices adds ~1.5× (Huffman on frequency-sorted indices, lossless). Caveat: **large codebooks thrash cache** — AQLM's ~1MiB codebook is *slower than FP16*; QuIP#'s ~1KiB E8P stays fast. rANS pairs with **small** codebooks.

**D. Non-uniform / sensitivity quant + Huffman.** SqueezeLLM (dense-and-sparse, 3-bit, ~2.3× speedup), Huff-LLM (arXiv 2502.00922, ~32% size / 31% latency via Huffman on exponent bits). Better ratio than uniform quant before any coder.

**E. tANS/FSE vs rANS vs Huffman.** tANS/FSE (zstd) ≈ rANS ratio with a simpler table-state decode; Huffman decodes faster/parallel but is suboptimal for skewed distributions. Consensus: ANS-family for ratio, Huffman for parallel decode. For LAIR's CPU kernel, **interleaved rANS or tANS** is ratio-optimal; Huffman only if a parallel-decode path dominates.

**F. GPU fused entropy decode.** DFloat11 (arXiv 2504.11651): Huffman on BF16 exponents, **LUTs in GPU SRAM + two-phase parallel decode**, ~30% lossless, large throughput vs offload, runs Llama-3.1-405B on 8×80GB. Huff-LLM notes a naive ANS-on-GPU **33% slowdown** (Hao et al 2024) — decode cost is the risk; parallel LUT decode recovers it. GPU variation = parallel entropy decode with on-chip tables.

**G. KV cache entropy coding (corrects §9).** Ecco (arXiv 2505.06901): group-wise non-uniform quant + **shared k-means codebooks + Huffman** with a **parallel multi-stage Huffman decode pipeline** → **~4× KV capacity, ~2.9× speedup** vs AWQ/SmoothQuant. So KV *is* amenable to entropy coding when it uses group granularity, shared codebooks, and parallel decode — the random-access problem from §9 is solved by the pipeline, not by per-token coding.

**Net:** variations that beat rANS-on-4-bit-GGUF: (1) **tensor-level quant + entropy coding** (EntroLLM-style, biggest CPU win), (2) **entropy-coding int8** (more headroom), (3) **sub-4-bit codebook quant + entropy-coded indices** (biggest footprint win, needs small codebooks), (4) **parallel-decode entropy coding on GPU / KV** (DFloat11, Ecco). Recurring caveat: entropy coding only pays off if decode is parallelized (interleaved/SIMD on CPU, LUT-SRAM parallel on GPU); a serial decoder turns a size win into a latency loss.

## 12. Bottom line

- Entropy-coding GGUF quant weights gives a real but small size win (~10%), only on large groupings; per-expert splitting hurts; rANS beats AC on speed.
- On the target Xeons, the fused L2-table / register-resident-decode design is correct, but **interleaved or SIMD rANS is mandatory** to avoid a decode bottleneck.
- Expected tokens/sec: **~1.0–1.2× baseline** for both dense and MoE with the best implementation; the MoE's main advantage from rANS is reduced footprint, not a speed multiplier.
- The larger wins come from **changing the encoding target** (tensor-level quant, int8, sub-4-bit codebooks) and **parallelizing decode** (SIMD/tANS on CPU, LUT-SRAM Huffman on GPU, Ecco-style parallel Huffman for KV), not from swapping rANS for AC on 4-bit GGUF.
