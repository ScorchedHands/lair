namespace Lair.Contracts;

/// <summary>
/// One stacked expert role plus the byte geometry of a single expert's slab inside it: the tensor the provider
/// stores contiguously, the two-dimensional shape a GEMV over one expert slice carries, and that slice's packed
/// byte block.
/// </summary>
public readonly record struct ExpertSlabRole(
    LogicalTensor Stack,
    TensorShape ExpertShape,
    long ExpertBytes);

/// <summary>
/// One gathered expert's gate/up/down slabs, handed to the consumer while the gather is live: the spans borrow the
/// provider's backing store and stay valid only for the duration of this call.
/// </summary>
public delegate void GatheredExpertChain(
    int expertIndex,
    ReadOnlySpan<byte> gateSlab,
    ReadOnlySpan<byte> upSlab,
    ReadOnlySpan<byte> downSlab);

/// <summary>
/// Gathers the contiguous slabs of a batch's union active-expert set for one sparse-FFN layer and runs each
/// gathered expert's chain while its slabs are warm. <paramref name="expertIndexes"/> is that union: every expert
/// the batch's router pointers select, each exactly once, in ascending expert order — so an expert several
/// sequences route to resolves and prefetches once per merged step instead of once per sequence, and the ascending
/// order is the accumulation order the dense-equivalent forward already uses. The default CPU implementation chases
/// each slab pointer through the provider's contiguous storage and issues cache-line prefetch hints over each slab
/// before the GEMV consumes it; prefetch changes no arithmetic, so gathered output stays bit-identical with the
/// dense-equivalent reference math.
/// </summary>
public interface IGatherKernel
{
    void Gather(
        IExpertSlabSource source,
        ExpertSlabRole gate,
        ExpertSlabRole up,
        ExpertSlabRole down,
        ReadOnlySpan<int> expertIndexes,
        GatheredExpertChain handle);
}
