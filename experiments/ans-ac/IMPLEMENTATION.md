# ANS / Arithmetic Coding — Implementation Design

This records the exact design decisions reached in discussion, so implementation can resume without re-deriving them. It is a parked design note, not an active phase. First pass is **CPU-only**; CUDA only if measurement proves it worthwhile; disk tier deferred.

## 1. Scope and hard constraints

- Optional format behind a config flag; ordinary packed quant (Q8_0 / Q4_K / IQ4_XS / …) remains the default path and the per-chunk fallback. No change to default inference.
- Encode is **transient, lossless, at load**: read an unmodified GGUF's existing codes, entropy-code them into a compressed in-memory representation. No converted model artifact (consistent with PLAN 2026-10-31 no-conversion-tooling and DN-M1-11 "read-time transform" provider).
- AC and ANS encode the **same** symbol sequence with the **same** shared frequency model, so size and speed compare fairly.

## 2. Seam map (what this touches)

| Concern | Seam | Note |
|---|---|---|
| Encoding at load | `IWeightSource` provider wrapping `GgufMmapWeightSource` | Loader (`IModelLoader`) stays metadata-only; transform lives in the provider |
| Format identity | `QuantFormat` enum + `QuantDescriptor` registry | New members e.g. `Q4_K_AC`, `IQ4_XS_RANS`, … (DN-M35-2: one entry + one kernel) |
| Decode during inference | `IBackend.Gemv/GemmBatch/DequantRow` via a new fused/staged kernel arm | Reuse existing GEMM scaffolding (`RunGemmRows`) |
| Selection | per-model config flag → picks provider + format | No relabeling of on-disk bytes; the same GGUF serves both modes because encoding happens at load |

## 3. Coding model (generalized across quant families)

Each **quant code** is a fixed-width source symbol from its own alphabet:
- Q8_0 → 256 symbols (8-bit codes). Expect weak compression — dense, near-full entropy; test it anyway as the low-headroom baseline.
- Q4_K / IQ4_NL / IQ4_XS → 16 symbols (4-bit codes). For **IQ4**, codes are indices into the `kvalue` codebook (`CpuBackend+IQ4_XS.cs`); for **Q4_K**, codes are per-group magnitudes with scales/mins stored as separate metadata.
- IQ3_XXS / IQ3_S → 8-symbol primary index stream **plus side planes** (high-bit and sign regions, read separately in `CpuBackend+IQ2_*`/shared `CpuBackend+IQ.cs`). Decide per format whether the primary code stream is compressed alone or the planes are coded as additional streams.

Symbol extraction = exactly what each dequant path already reads (`qs`, scale/min bytes, grids). Scales/mins/codebook tables stay as ordinary metadata; only the **code stream** (and optionally side planes) is entropy-coded. A sequence of codes is the coder input. Output is a variable-length bitstream — never map a tile to a fixed smaller output.

## 4. Shared frequency model + chunking

- One probability/frequency table shared across a **configurable group of tiles**, stored once per group, not per tile.
- Chunks align with the engine's layout: smallest independently addressable unit is one quant super-block (Q4_K = 144 bytes / 256 elements; IQ4_XS = 136/256; Q8_0 = 34/32); sub-blocks of 32 elements are the natural inner chunk. Group sizes tested: {1 super-block, a row's worth, several rows} aligned to `GemmTileRows=8 × GemmTileCols=QkK`.
- Each tile / small group independently decodable with explicit offsets, lengths, symbol counts, and required coder state (rANS seed/flush words; AC carry/reservoir termination).

## 5. Zero-frequency handling

Documented smoothing applied identically to both codecs: every alphabet symbol gets a minimum nonzero frequency (e.g., +1 Laplace floor), then the table is normalized to a fixed total `M` (power of two for ANS convenience) with a documented rounding rule so encoder and decoder reconstruct byte-identical tables from stored counts. Store model once per group; both codecs consume it unchanged.

## 6. Codecs behind one interface (`ICodec`)

Modular so more codecs can be added later:
- **Arithmetic coding** — carryless/range variant with documented termination/framing (flush bits, byte-stuffing or renormalization rule). Advantage: encodes in a single streaming pass even with an adaptive model.
- **ANS = interleaved rANS** chosen for this layout — documented state init, interleave degree, flush/termination, and symbol order (encode reverse vs decode forward). Chosen because it is SIMD-friendly and fast; scalar ANS would forfeit the speed case. Advantage: cheap table-driven decode at ~1 lookup/symbol.
- Each codec exposes `Encode(symbolStream + model → bitstream)` and `Decode(bitstream + model → symbolStream)`.

## 7. Variable geometry — contract widening (the central decision)

Entropy output has no fixed stride, which collides with the invariant "span bytes == Rows × RowStrideBytes(Format)" (`WeightRef.cs`) asserted in `GemmBatch` (`CpuBackend+Gemm.cs:47-54`) and load-time geometry checks (`ModelPlans`, `GgmlTypeRegistry`).

Chosen approach — **additive widening**:
- `WeightRef` gains an optional chunk-map / coder-state member (offset table + per-chunk lengths/counts + model scope). Fixed formats pass it null and keep the constant-stride path byte-identical to today.
- Addressing becomes format-aware: fixed formats compute stride; compressed formats read offsets from the chunk map. The `GemmBatch` length guard branches on format (compressed tensors validate against the chunk map, not a computed stride).

Alternative if widening proves too invasive — **fixed worst-case slots + packed fallback**: preserves every invariant but caps each chunk at ≤ its packed size, blunting compression. Decide variable-length vs fixed-slot explicitly before coding.

## 8. Kernel strategy (staged reuse vs true fusion)

Decompose existing per-format kernels into more `static` methods so a codec kernel reuses large chunks; co-dependency is acceptable since both benefit from shared improvements and all code is owned here already. The genuinely reusable chunk is the GEMM scaffolding — accumulator chains, tile staging, batched FMA shape, `FinishGemmTile`, scale/min unpack (`GetScaleMinK4`) — already format-parameterized in `RunGemmRows` (`CpuBackend+Gemm.cs:108-138`).

Two shapes to evaluate (pick per codec):
1. **Staged** (preferred first): decode a chunk into a scratch nibble/code buffer, then feed the *existing* tile-build + batched-FMA loop — this is already precedented by IQ3_XXS staging (`CpuBackend+IQ.cs`, DN-M38-3). Near-total reuse; may need only a new decoder + tile-fill step rather than a whole matmul kernel.
2. **True inline fusion**: interleave symbol decoding with FMA in one inner loop. Higher bandwidth/latency win potential, but the shared chunk shrinks and a per-symbol "fetch next" seam risks hot-path indirection that DN-M1-3 forbids (no interface call / no indirect fetch inside the GEMV walk). Share via static helpers + duplicated decode+scale inner loop; do not introduce delegates in the hot loop.

## 9. Provider / encode-at-load mechanics

Streaming, chunk-by-chunk from the mmap span → compressed arena cached in DRAM (compressed form resident instead of packed Q4 — this is where a real footprint win can appear). Encode once per tensor at first resolve; never re-encode per token.

**Model-build vs streaming interaction:**
- Shared group model must be fixed before ANS encodes that group → either a **static model**, or a two-pass count→encode over the whole group (count all chunks in the group, build table, then encode its chunks). Adaptive AC can stream one pass with an adaptive model.
- Decide static vs data-derived shared model up front; it determines whether ANS streams single-pass or needs the counting pass.

**Post-experiment optimization:** a pre-encoded disk cache is allowed *if* keyed by and validated against the pinned source sha256 using LAIR's existing manifest machinery (DN-M315), so a stale cache can't silently load wrong bytes. Out of scope for proving the idea.

## 10. Fallback + accounting rules

Per chunk: if codec payload + metadata ≥ packed quant size, store that chunk as ordinary packed quant with an escape marker. Report **actual total bytes** including frequency tables, offsets/lengths/symbol counts, framing, padding, scales/mins, and codebook tables — never theoretical entropy or estimated bits/symbol alone.

## 11. Versioning / format spec (document before coding)

Format version byte; little-endian byte order throughout (matching existing `ReadHalfLE` conventions); frequency normalization rule (`M`, rounding tie-break); count/offset/length widths (fix to uint32 or uint64 and state it); alignment of chunk boundaries to super-block/sub-block; framing and padding rules; model scope id (which group a table covers); codec-specific fields (rANS interleave degree + seed words; AC carry/reservoir flush). Keep both codecs behind `ICodec` so future codecs slot in.

## 12. Tests (`Category=Codec`)

- Exact round trips of codes + metadata for random, uniform, skewed, and **real model** tiles across all tested formats (Q8_0, Q4_K, IQ4_NL, IQ4_XS, IQ3_XXS, IQ3_S).
- Empty / minimum-size chunks; all-one-symbol chunks; all-symbols-present chunks; boundary values; truncated/corrupt input; invalid offsets/lengths.
- Encoder and decoder agree on the model table and chunk boundaries (reconstruct identical tables from stored counts).
- Decoded output matches the existing quant path within its current tolerance (`Category=Kernel` oracle comparison style, raw-byte where applicable).
- AC-specific: termination/framing correctness across flush boundaries; ANS(rANS)-specific: state initialization, interleave/flush termination, symbol order (reverse-encode vs forward-decode) verified.

## 13. CPU fused-decode design (target: 2× Xeon E5-2683 v4, 16c/32t each @2.1GHz, AVX2, 256KB L2/core, 40MB L3/socket, DDR4-2133 quad-channel ≈68GB/s/socket)

This records the decode-side design suggested for the first (CPU-only) pass. It refines §8 with the concrete cache/register placement and the throughput gating that the measurement exposed.

**Placement contract**
- Probability/frequency table + dequant scale/min + codebook live in **L2** (256KB/core is ample for a ≤256-symbol table + decode LUT). One table per group stays hot across the group's decode; never re-read from DRAM per chunk.
- Decoded symbols are produced **directly into AVX2 registers** and consumed in-register by the GEMV/GEMM FMA chain (`VPMADDUBSW`/`VPMADDWD` + FMA). The dequantized fp16/fp32 intermediate is **not written to L3/DRAM** — symbol→value mapping is an L2 table lookup folded into the matmul inner loop.
- Decoded tiles may spill to **L3** only for reuse across a batch; the single-token hot path keeps values register-resident.

**Throughput gate (the controlling constraint)**
- Scalar rANS is a serial state machine at ~1–2 symbols/cycle → ~2–4 G symbols/s/core. To saturate ~136 GB/s at compressed ~0.45 B/symbol needs ~300 G symbols/s, so **scalar rANS is decode-bound and cannot saturate memory** → fusion never gets exercised and can regress vs a good baseline.
- Therefore fused decode is only worthwhile with **interleaved rANS (4–8 independent states/core) or SIMD-ANS**, giving ~8–16 G symbols/s/core → ~256–1000 G/s aggregate → decode exceeds DRAM and the in-register fusion/compression actually pay off.
- Interleave degree and chunk size are chosen jointly with §4 grouping: more/shorter independent streams raise decode parallelism but pay per-group table cost; fewer/longer streams share the table better but expose less ILP.

**MoE expert handling**
- Baseline inference already reads only the **active** experts per token; rANS does not change that. The MoE win from rANS is therefore **footprint** (compressed experts fit more in L3 / reduce DRAM traffic for active experts), **not** skipping inactive experts. Decode only the active experts' streams; inactive experts stay compressed and untouched.

**Register-pressure caveat**
- rANS state + L2 table pointers + accumulator compete with the matmul's registers (Broadwell: 16 GPRs / 16 AVX2 regs/core). Keep interleave degree modest and tables compact; occupancy loss here can cap the gain. No interface call / indirect fetch in the hot loop (DN-M1-3).

**Net CPU expectation (corrects an earlier over-optimistic MoE figure)**
- Dense: ~**1.0–1.2×** vs a good on-the-fly-dequant baseline (from ~10% fewer DRAM bytes), and only if decode is interleaved/SIMD; scalar rANS ≈ neutral or slightly slower.
- MoE: ~**1.0–1.3×** on active-expert traffic; the larger benefit is memory footprint / cache residency, not a decode-throughput multiplier.

## 14. Open decisions to settle at implementation time

1. Staged reuse vs true inline fusion (per codec).
2. Static vs data-derived shared model (drives whether ANS can stream single-pass).
3. Variable-length contract widening vs fixed worst-case slots + fallback.
4. Chunk/group size set aligned to `GemmTileRows/Cols`.
5. Whether IQ3 side planes are coded as separate streams or only the primary index is compressed.

## 15. Explicit non-goals for the first pass

No CUDA fused decode; no disk weight tier (gated on refcounted double-buffered prefetch staging, PLAN `:104`); no in-engine runtime A/B comparator — compare AC vs ANS as separate config runs diffed from logs/bench reports (PLAN DN-M7-10 modularity invariant).

## 16. LAIR integration candidates from the conclusions

This maps `CONCLUSION.md` into LAIR seams without changing the default path. It is a menu of where a benefit is real, not an implementation phase.

| Conclusion | LAIR seam / shape | Where benefit appears | Gate / caveat |
|---|---|---|---|
| rANS weight provider for `Q4_K`, `IQ4_XS`, `IQ3_XXS` | `IWeightSource` provider wrapping `GgufMmapWeightSource` + new `QuantFormat` / `QuantDescriptor` registry entry + staged kernel arm | Dense decode and prefill when model exceeds cache; MoE resident headroom and compressed offloaded-expert traffic | Only with interleaved/SIMD/tANS decode; per-chunk packed fallback; tables grouped over tensor/layer/expert-tensor, not per-expert |
| Grouping rule from measurements | Chunk-map scope id in `WeightRef`; shared model over whole expert tensors / layers / model tensors | Avoids the measured per-expert table overhead (~2.7% or raw fallback) | Do not entropy-code each expert separately; group experts if expert streams are coded |
| Tensor-level / large-group quant variation | New published GGUF format or sidecar `IWeightSource` provider only | EntroLLM-style large entropy-coding win: skewed code distributions, potentially 1.3–1.5× decode speed | LAIR has no conversion tooling (PLAN 2026-10-31, DN-M315-2); only useful if source artifacts already use tensor-level/large-group quant or LAIR later supports sidecar quant artifacts |
| INT8 entropy coding variation | Optional `Q8_0` compressed provider | Capacity win for int8-resident models; more entropy headroom than 4-bit coding | Not a win versus Q4 speed; low priority for LAIR's quant story |
| Sub-4-bit codebook / index entropy coding | Future `IQ2_*` / codebook format kernels plus entropy-coded index streams | Footprint win far larger than rANS-on-4-bit: 4–8× from 2-bit/ternary quant, plus index entropy | Needs small codebooks (QuIP#/E8P-like); large codebooks (AQLM-like) can thrash L1/L2 and be slower than FP16; LAIR currently does not decode `IQ2_*` expert tensors |
| KV entropy coding variation | Future KV offload / paged KV tier only | KV capacity and PCIe/NVMe transfer reduction if KV pages are large, independently decodable, and use shared codebooks | Not suitable for the hot random-access attention loop as-is; needs Ecco-style block granularity and parallel decode |
| GPU fused entropy decode | Future CUDA backend compressed weight/KV tier | VRAM capacity and offload-bandwidth win, not decode speed | Current non-goal; requires parallel LUT/SRAM decode (DFloat11 / Huff-LLM pattern) or decode cost can regress |
| Codec variants behind `ICodec` | `ICodec` remains the extension seam | tANS/FSE may give rANS-like ratio with simpler CPU decode; Huffman may be better for parallel GPU/KV pipelines | AC remains a measurement reference, not the practical LAIR codec |

**Recommended first LAIR addition, if any:** the smallest useful experiment is a config-flagged rANS provider for large groupings of existing GGUF formats (`Q4_K`, `IQ4_XS`, `IQ3_XXS`) using the staged kernel shape from §8 and the packed fallback from §10. Its benefit is capacity/offload plus ~1.0–1.2× only when decode is interleaved/SIMD. The larger wins from the research conclusions are deferred until LAIR has source formats or sidecar providers that already use tensor-level quant, small-codebook sub-4-bit quant, or block-page KV.
