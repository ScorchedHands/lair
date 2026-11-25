namespace Lair.Contracts;

/// <summary>
/// Optional backend capability: traverse three dense weight matrices that share one activation batch and column count in a single call. A backend
/// implements this only for the postures it can reproduce bit-identically — every output element matching what an independent
/// <see cref="IBackend.GemvBatch"/> call would write, with the same accumulator order per chain and the same row partitioning under threading — so
/// callers can fuse shared-input traversals (an attention block's batched Q/K/V projections) without changing output semantics. Backends that do
/// not implement the interface are unaffected; implementations return false for postures they do not cover, and the caller falls back to three
/// independent calls.
/// </summary>
public interface IDenseWeightTripleFusion
{
    /// <summary>
    /// Run one activation batch through three dense weight matrices in a single call when this backend covers those formats and the shared column
    /// layout; returns false without touching the outputs otherwise. Each produced element matches the value an independent
    /// <see cref="IBackend.GemvBatch"/> call for that matrix would write, including under threading; every output uses the same flat
    /// <c>[batch * rows + row]</c> layout as <see cref="IBackend.GemvBatch"/>. Equal column counts across the three weights and the input batch are
    /// required; unlike the pair seam, the three weights may have different row counts (an attention block's Q/K/V trio does), because the fused
    /// executor walks each matrix's own rows and only lets the matrices that still hold a row at a position share one pass over the batch.
    /// </summary>
    bool TryGemvBatchSharedInput(ReadOnlySpan<float> inputs, int count, WeightRef weightA, Span<float> outputsA, WeightRef weightB, Span<float> outputsB, WeightRef weightC, Span<float> outputsC);
}
