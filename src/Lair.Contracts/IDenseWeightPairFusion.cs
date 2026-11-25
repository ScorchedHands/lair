namespace Lair.Contracts;

/// <summary>
/// Optional backend capability: traverse two dense weight matrices that share one activation batch and column count in a single
/// call. A backend implements this only for the postures it can reproduce bit-identically — every output element matching what an
/// independent <see cref="IBackend.GemvBatch"/> call would write, with the same accumulator order per chain and the same row
/// partitioning under threading — so callers can fuse shared-input traversals (a dense feed-forward's gate/up pair) without changing
/// output semantics. Backends that do not implement the interface are unaffected; implementations return false for postures they do
/// not cover, and the caller falls back to two independent calls.
/// </summary>
public interface IDenseWeightPairFusion
{
    /// <summary>
    /// Run one activation batch through two dense weight matrices in a single call when this backend covers that format pair and
    /// column layout; returns false without touching the outputs otherwise. Each produced element matches the value an independent
    /// <see cref="IBackend.GemvBatch"/> call for that matrix would write, including under threading; both outputs use the same flat
    /// <c>[batch * rows + row]</c> layout as <see cref="IBackend.GemvBatch"/>. Equal column counts across the two weights and the input
    /// batch are required; unequal-row pairs currently fall back by returning false because this fused executor walks a shared row range.
    /// </summary>
    bool TryGemvBatchSharedInput(ReadOnlySpan<float> inputs, int count, WeightRef weightA, Span<float> outputsA, WeightRef weightB, Span<float> outputsB);
}
