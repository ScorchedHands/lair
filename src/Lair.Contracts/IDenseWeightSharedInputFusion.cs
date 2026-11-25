namespace Lair.Contracts;

/// <summary>
/// Optional backend capability: run one activation vector through two or three dense weight matrices that share it and the
/// same column count in a single call. A backend implements this only for the postures it can reproduce bit-identically —
/// every output element matching what an independent <see cref="IBackend.Gemv"/> call for that matrix would write, with the
/// same accumulator order per row and the same row partitioning under threading — so callers can fuse shared-input
/// single-token traversals (an attention block's Q/K/V projections or a dense feed-forward's gate/up pair) without changing
/// output semantics. Backends that do not implement the interface are unaffected; implementations return false for postures
/// they do not cover, and the caller falls back to independent calls.
/// </summary>
public interface IDenseWeightSharedInputFusion
{
    /// <summary>
    /// Run one activation vector through two dense weight matrices in a single call when this backend covers that format
    /// pair and column layout; returns false without touching the outputs otherwise. Each produced element matches the
    /// value an independent <see cref="IBackend.Gemv"/> call for that matrix would write, including under threading.
    /// Equal column counts across the weights and the activation vector are required; the weights may have different row
    /// counts and formats, since each row is computed independently exactly as its own <see cref="IBackend.Gemv"/> would.
    /// </summary>
    bool TryGemvSharedInput(ReadOnlySpan<float> x, WeightRef weightA, Span<float> outputA, WeightRef weightB, Span<float> outputB);

    /// <summary>Three-weight form of <see cref="TryGemvSharedInput(ReadOnlySpan{float}, WeightRef, Span{float}, WeightRef, Span{float})"/>.</summary>
    bool TryGemvSharedInput(ReadOnlySpan<float> x, WeightRef weightA, Span<float> outputA, WeightRef weightB, Span<float> outputB, WeightRef weightC, Span<float> outputC);
}
