namespace Lair.Contracts;

/// <summary>
/// A provider that exposes one stacked expert tensor as contiguous per-expert slabs addressable by expert index:
/// the packed bytes of expert <c>expertIndex</c> — <c>expertBytes</c> long, starting at
/// <c>expertIndex * expertBytes</c> inside the tensor's payload — as a logical view over the provider's backing
/// store, never a copy. Storing each expert's rows contiguously keeps a slab a pointer chase into one region,
/// which is what lets a gather kernel prefetch a slab before the GEMV consumes it. A residency provider later
/// answers the same call from whichever tier holds the slab; the call shape does not change.
/// </summary>
public interface IExpertSlabSource
{
    ReadOnlySpan<byte> ResolveSlab(LogicalTensor stack, int expertIndex, long expertBytes);
}
