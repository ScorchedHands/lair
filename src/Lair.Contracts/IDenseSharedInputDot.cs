namespace Lair.Contracts;

/// <summary>
/// Optional backend seam for a batch of dense FP32 dots that share one operand: several input vectors, laid out
/// consecutively, each dotted against the same common vector in one call. Every produced element is bit-identical to
/// what the matching individual <see cref="IBackend.Dot"/> call would write — same per-output accumulation order over
/// whole eight-lane chunks, same scalar tail folded into its own accumulator, same reduction order — so a caller can
/// batch dots that repeat an operand (a grouped-query attention walk scoring several query heads against one stored K
/// position) without changing output semantics. Backends that do not implement the interface are unaffected;
/// implementations return false for shapes and postures they do not cover, leaving outputs untouched, and the caller
/// falls back to independent <see cref="IBackend.Dot"/> calls.
/// </summary>
public interface IDenseSharedInputDot
{
    /// <summary>
     /// Dot <paramref name="count"/> consecutive input vectors (each exactly <c>common.Length</c> long, stored at
     /// <c>inputs[i * common.Length]</c>) against the single shared <paramref name="common"/> vector when this backend
     /// covers that shape and posture; returns false without touching <paramref name="outputs"/> otherwise. Each output
     /// element equals what an individual <see cref="IBackend.Dot"/> call for that input vector would return, bit for
     /// bit, whatever internal grouping the batched kernel uses. A backend's resolved chain-group width bounds how many
     /// accumulator chains share one pass over the common vector; a group wider than that is split into consecutive
     /// chunks of it rather than widened past the posture's register budget.
     /// </summary>
    bool TryDotBatchSharedInput(ReadOnlySpan<float> inputs, int count, ReadOnlySpan<float> common, Span<float> outputs);
}
