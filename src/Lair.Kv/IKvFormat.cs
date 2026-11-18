namespace Lair.Kv;

/// <summary>
/// Storage-only static-abstract seam for one quantized KV side. Each format is a fixed-size block struct that carries its
/// group payload and its group scale inside the same struct, so a token region stays contiguous at block granularity and a
/// block-granularity read hands back a typed span with no copy. This seam is the cold storage contract only: it names the
/// footprint axes a pool sizes from and the quantizing write a token region pays, and deliberately carries no attention
/// dot/accumulate kernels — those land in a later item.
/// </summary>
/// <typeparam name="TSelf">The concrete block struct implementing this format, constrained to itself so the static members
/// resolve per format with no runtime switch.</typeparam>
public interface IKvFormat<TSelf> where TSelf : struct, IKvFormat<TSelf>
{
    /// <summary>Values one block struct stores. A side region must cover whole groups of this many values.</summary>
    static abstract int ValuesPerBlock { get; }

    /// <summary>Bytes one block struct occupies: whole payload plus that group's scale, the footprint axis a pool sizes blocks from.</summary>
    static abstract int BlockBytes { get; }

    /// <summary>Quantizes one side region (whole groups) into this format's blocks — the append-time storage write.</summary>
    static abstract void WriteRegion(ReadOnlySpan<float> values, Span<TSelf> blocks);
}
