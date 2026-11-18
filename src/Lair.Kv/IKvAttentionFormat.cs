using Lair.Backends.Cpu;

namespace Lair.Kv;

/// <summary>
/// Attention hot seam for the dot side of a quantized KV pair: the storage-only <see cref="IKvFormat{TSelf}"/> contract
/// stays untouched, and this seam adds the static-abstract query preparation and integer dot a quantized attention kernel
/// runs per scored position. A generic attention instantiation resolves every member here at JIT time for one concrete
/// format struct, so a quantized loop carries no runtime format switch and no per-token virtual dispatch. The dot side
/// prepares the query heads once per attention call — one Q8_K-quantized activation per head, the same machinery the
/// CPU backend's sub-4-bit integer dots stage against — and scores each stored region by exact integer dots over whole
/// 32-value sub-blocks, each sub-block total scaled by the stored group scale and the activation super-block scale it
/// belongs to. The integer totals are exact, so the encoding an <see cref="IntegerDotPath"/> selects changes only how
/// those totals are formed, never which totals the dot reports.
/// </summary>
/// <typeparam name="TSelf">The concrete block struct implementing this format, constrained to itself so the static
/// members resolve per format with no runtime switch.</typeparam>
public interface IKvDotSide<TSelf> where TSelf : struct, IKvFormat<TSelf>
{
    /// <summary>Prepared-query scratch bytes one attention call needs per query head: the whole Q8_K super-blocks covering one head vector.</summary>
    static abstract int DotScratchBytesPerHead(int headDim);

    /// <summary>Activation scale entries one query head needs: one per whole Q8_K super-block covering the head.</summary>
    static abstract int DotScaleCount(int headDim);

    /// <summary>
    /// Prepares the query side once per attention call: head h's region is quantized into its own scratch slot at
    /// <c>h * DotScratchBytesPerHead(headDim)</c> with its super-block scales at <c>h * DotScaleCount(headDim)</c>, the
    /// same scale rule and rounding every sub-4-bit GEMV activation uses. An all-zero head quantizes to zero quants under
    /// a zero scale and scores exactly zero.
    /// </summary>
    static abstract void PrepareQuery(ReadOnlySpan<float> qHeads, int headCount, int headDim, Span<byte> scratch, Span<float> scales);

    /// <summary>
    /// The K-dot of one scored position's kv-head region against this call's prepared query head <paramref name="headSlot"/>:
    /// each stored 32-value group is dotted against the prepared activation quants as exact integer totals under the resolved
    /// <paramref name="dot"/> encoding, and each group total is scaled by that group's stored scale times the activation scale
    /// of the super-block its values live in. The head's groups start at block <paramref name="headBlockOffset"/> in the
    /// token's contiguous region.
    /// </summary>
    static abstract float DotHead(ReadOnlySpan<TSelf> tokenRegion, int headBlockOffset, ReadOnlySpan<float> qh, ReadOnlySpan<byte> scratch, ReadOnlySpan<float> scales, int headSlot, IntegerDotPath dot);
}

/// <summary>
/// Attention hot seam for the value side of a quantized KV pair: the static-abstract weighted accumulation a quantized
/// attention kernel applies per scored position, decoding each stored group's values under that group's scale and adding
/// <c>weight * value</c> into the head accumulator in element order. A format that carries this seam alone is a value side
/// only — it stores no dot operand shape — so a pair naming it on both sides cannot instantiate the quantized attention
/// kernel at all.
/// </summary>
/// <typeparam name="TSelf">The concrete block struct implementing this format, constrained to itself so the static
/// members resolve per format with no runtime switch.</typeparam>
public interface IKvValueSide<TSelf> where TSelf : struct, IKvFormat<TSelf>
{
    /// <summary>
    /// Adds <c>weight * decodedValue</c> for every value of one scored position's kv-head region into <paramref name="outHead"/>:
    /// group g of the head covers output elements <c>g * ValuesPerBlock .. +ValuesPerBlock - 1</c>, its values decoded under the
    /// group's stored scale with exactly the pairing the committed writer stored them in. The head's groups start at block
    /// <paramref name="headBlockOffset"/> in the token's contiguous region.
    /// </summary>
    static abstract void AccumulateHead(Span<float> outHead, ReadOnlySpan<TSelf> tokenRegion, int headBlockOffset, float weight);
}
