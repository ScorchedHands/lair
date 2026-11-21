using Lair.Contracts;

namespace Lair.Loader;

/// <summary>
/// Contiguous-expert-slab view over an existing weight source: a routed expert tensor stored as one stacked matrix
/// is exposed as per-expert slabs at their contiguous byte ranges, so a gather kernel addresses slabs by router
/// pointer while the backing store stays exactly what it was — the mmap arena today, an arena-copied source, or a
/// residency tier later. Slabs are logical views; no copy of any expert tensor is ever made here. The wrapped
/// source keeps its own lifetime and disposal, and every slab span this view returns borrows from it and stays
/// valid for the backing source's lifetime.
/// </summary>
public sealed class ExpertSlabWeightSource : IExpertSlabSource
{
    private readonly IWeightSource _backing;

    public ExpertSlabWeightSource(IWeightSource backing)
    {
        ArgumentNullException.ThrowIfNull(backing);
        _backing = backing;
    }

    /// <summary>The weight source these slab views read through.</summary>
    public IWeightSource Backing => _backing;

    public ReadOnlySpan<byte> ResolveSlab(LogicalTensor stack, int expertIndex, long expertBytes)
    {
        ArgumentNullException.ThrowIfNull(stack);
        ArgumentOutOfRangeException.ThrowIfNegative(expertIndex);
        ArgumentOutOfRangeException.ThrowIfLessThan(expertBytes, 1L);

        if (expertIndex > 0 && expertBytes > stack.SizeBytes / expertIndex)
        {
            throw new ArgumentException(
                $"slab of expert {expertIndex} at {expertBytes} bytes per expert reaches past the {stack.SizeBytes}-byte " +
                $"payload of stacked tensor '{stack.Name}'", nameof(expertIndex));
        }

        long offset = (long)expertIndex * expertBytes;
        ReadOnlySpan<byte> payload = _backing.Resolve(stack);
        if (offset > payload.Length - expertBytes)
        {
            throw new ArgumentException(
                $"slab of expert {expertIndex} at {expertBytes} bytes per expert does not fit the resolved " +
                $"{payload.Length}-byte payload of stacked tensor '{stack.Name}'", nameof(expertIndex));
        }

        return payload.Slice((int)offset, (int)expertBytes);
    }
}
