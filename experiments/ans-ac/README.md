# ANS / Arithmetic Coding for Quantized Weights — Experiment Basis

## The idea

Entropy-code the fixed-width quant code streams already produced by LAIR's quant formats (Q8, Q4_K, IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S) so that resident weight bytes shrink below their packed size. Arithmetic coding (AC) and Asymmetric Numeral Systems (ANS / interleaved rANS) are compared as two codecs over the *same* symbol stream using the *same* shared frequency model.

The engine would read compressed bytes instead of packed quant blocks, decoding either **fused into the matmul tile loop** or **staged** (decode a chunk into scratch, then run the existing GEMM scaffolding). The hypothesis: real entropy per code is below its fixed width, so compression trades memory bandwidth for decode ALU — and on a CPU weight-bandwidth-bound path that trade may be net-positive.

## Basis in LAIR (why this is buildable here)

- `IWeightSource` provider seam (`src/Lair.Contracts/IWeightSource.cs`). PLAN DN-M1-11 explicitly sanctions "repacked sidecars, tiered streaming, **read-time transforms**" as providers that plug in without touching loader or kernels. Encode-at-load over an ordinary GGUF is a transient lossless read-time transform — it does not require a converted model artifact and does not violate the 2026-10-31 no-conversion-tooling decision (no requantization, no distillation, nothing persisted).
- `QuantFormat` registry (`src/Lair.Contracts/QuantFormat.cs`, geometry in `QuantDescriptor.cs`). PLAN DN-M35-2: "a format [is] one registry entry plus one kernel." A new fused quant variant is the sanctioned extension shape.
- Fused dequant already exists for every quant family (`src/Lair.Backends.Cpu/CpuBackend+Q4_K.cs`, `CpuBackend+IQ4_XS.cs`, …), and a staged "dequant into tile → batched FMA" path already exists (IQ3_XXS, PLAN DN-M38-3). A codec can reuse that scaffolding.

## Novelty

- **Novel for LAIR:** no entropy codec exists anywhere in `src` today; the byte-reduction program is provider-based and non-fused. An AC/ANS *quant format* with fused decode is a new shape here.
- **Not novel as CS:** AC, ANS/rANS are well-established. The novelty is the specific combination — entropy-coded quant formats decoded in-kernel inside this engine's tile layout, produced by encode-at-load over an unmodified GGUF, benchmarked fairly against each other with one shared model and identical chunk boundaries.

## Potential pros

- **Smaller resident weights.** Compressed codes can drop below packed-Q4/Q8 size; helps large models, cold start, and (later) a disk weight tier.
- **Cheap fair comparison.** AC-vs-rANS is a provider swap over the same model/chunks — no engine fork needed to compare sizes.
- **Lossless.** No accuracy change; decoded codes must match the original quant path exactly (within its existing tolerance).
- **Reuses seams.** Provider + format registry + staged-GEMM scaffolding already exist.

## Potential cons / risks

- **Variable length breaks a load-bearing invariant.** `WeightRef` requires "span bytes == Rows × RowStrideBytes(Format)" (`src/Lair.Contracts/WeightRef.cs`) and kernels index blocks at fixed strides (`CpuBackend+Gemm.cs`). Entropy output is variable-length by design, so the contract must widen (offset/coder-state metadata) or fall back to capped fixed slots.
- **Decode speed may not beat bandwidth saved.** LAIR's measured staging decoders cost ~1.5–2.3× the Q4_K baseline (`DEVELOP.md` "Measured GEMV cost", `.164 ns/elt`). A scalar AC decoder would likely lose; only a fast interleaved rANS has a real chance of net win.
- **Q8 headroom is small.** 8-bit codes are dense and often near-full entropy — expect the weakest compression there. Q4/IQ4 (16-symbol alphabet) and IQ3 (8-symbol + side planes) have more room, but group scales/mins and codebook structure already consume some of it.
- **Metadata overhead eats the win on high-entropy tensors.** Per-chunk framing/offsets/model tables can exceed savings; packed-Q4 fallback will fire often. Must report *actual* total bytes (tables + offsets + scales), not theoretical entropy.
- **Encode-at-load cost + peak memory.** Encoding requires reading each tensor's codes at least once and holding mmap source plus compressed arena transiently — a load-time tax that partly fights page-cache laziness for models too big to fit RAM.
- **Kernel ownership.** Fused decode means new per-codec×per-format kernels; staged reuse shrinks this but does not eliminate it.

## Honest framing

The experiment must state plainly when AC or ANS provides no size or speed benefit on a given format/model, rather than presenting compression as assumed. CPU-only for the first pass; CUDA only if proven worth it. Disk residency and pre-encoded caches are post-experiment optimizations.
