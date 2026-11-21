using System.Runtime.Intrinsics.X86;

using Lair.Contracts;

namespace Lair.Models.Engine;

/// <summary>
/// Default CPU gather: walks the batch's union active-expert set in ascending expert order, chases each selected
/// expert's gate/up/down slab pointers through the provider's contiguous storage, issues prefetch hints across each
/// slab's cache lines before handing that expert's chain to the consumer, and resolves every selected expert exactly
/// once per step — an expert several sequences route to gathers once. Prefetch is a pure hint (prefetcht0): it moves
/// bytes toward the cache ahead of the GEMV that consumes them and touches no memory itself, so gathered arithmetic
/// stays bit-identical with the dense-equivalent forward in the ascending accumulation order both paths use.
/// </summary>
public sealed class CpuGatherKernel : IGatherKernel
{
    private const int PrefetchLineBytes = 64;

    public void Gather(
        IExpertSlabSource source,
        ExpertSlabRole gate,
        ExpertSlabRole up,
        ExpertSlabRole down,
        ReadOnlySpan<int> expertIndexes,
        GatheredExpertChain handle)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(handle);

        for (int i = 0; i < expertIndexes.Length; i++)
        {
            int expertIndex = expertIndexes[i];

            ReadOnlySpan<byte> gateSlab = source.ResolveSlab(gate.Stack, expertIndex, gate.ExpertBytes);
            ReadOnlySpan<byte> upSlab = source.ResolveSlab(up.Stack, expertIndex, up.ExpertBytes);
            ReadOnlySpan<byte> downSlab = source.ResolveSlab(down.Stack, expertIndex, down.ExpertBytes);

            PrefetchSlab(gateSlab);
            PrefetchSlab(upSlab);
            PrefetchSlab(downSlab);

            handle(expertIndex, gateSlab, upSlab, downSlab);
        }
    }

    private static unsafe void PrefetchSlab(ReadOnlySpan<byte> slab)
    {
        if (!Sse.IsSupported || slab.IsEmpty)
        {
            return;
        }

        fixed (byte* basePointer = slab)
        {
            for (int offset = 0; offset < slab.Length; offset += PrefetchLineBytes)
            {
                Sse.Prefetch0(basePointer + offset); // one hint per cache line of the slab: retain across levels, no memory access happens here.
            }
        }
    }
}
