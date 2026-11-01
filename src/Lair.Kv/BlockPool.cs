using System.Buffers;
using Lair.Core.Contracts;

namespace Lair.Kv;

/// <summary>
/// Fixed-size pool of paged KV blocks. One physical block stores the K and V cache for every
/// transformer layer of <c>blockTokens</c> token positions, so one logical position maps to one
/// physical block. Blocks are handed out from a free-list stack in O(1); each block carries a
/// reference count so later copy-on-write sharing can alias a block across sequences. The arena is
/// a single aligned allocation owned by the backend; block views are spans over that arena and are
/// valid for the pool's lifetime.
/// </summary>
/// <remarks>
/// Block layout (float units inside one block):
/// <code>
/// kvDim          = kvHeads * headDim            floats per token, per K or V
/// layerRegion    = 2 * blockTokens * kvDim      floats per layer (K half, then V half)
/// floatsPerBlock = layerCount * layerRegion
///
/// float offset of (blockId, layer, K/V, tokenSlot):
///   blockId * floatsPerBlock
/// + layer * layerRegion
/// + (isV ? blockTokens * kvDim : 0)
/// + tokenSlot * kvDim
/// </code>
/// Bytes per block = layerCount * 2 * blockTokens * kvHeads * headDim * sizeof(float).
/// The arena is zero-initialized at construction, so slots that were never written read back as 0.
/// </remarks>
public sealed class BlockPool : IDisposable
{
    private const int ArenaAlignment = 64;

    private readonly IBackend _backend;
    private readonly MemoryHandle _arena;
    private readonly unsafe float* _base;
    private readonly int[] _refCounts;
    private readonly int[] _freeStack;
    private readonly int _totalFloats;
    private int _freeTop;
    private bool _disposed;

    public unsafe BlockPool(int blockTokens, int layerCount, int kvHeads, int headDim, int capacityBlocks, IBackend backend)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacityBlocks);
        ArgumentNullException.ThrowIfNull(backend);

        BlockTokens = blockTokens;
        LayerCount = layerCount;
        KvHeads = kvHeads;
        HeadDim = headDim;
        KvDim = checked(kvHeads * headDim);
        LayerRegion = checked(2L * blockTokens * KvDim);
        FloatsPerBlock = checked((int)(LayerCount * LayerRegion));
        BytesPerBlock = checked(FloatsPerBlock * sizeof(float));
        CapacityBlocks = capacityBlocks;
        _backend = backend;

        long totalBytes = checked((long)capacityBlocks * BytesPerBlock);
        if ((long)capacityBlocks * FloatsPerBlock > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(capacityBlocks), $"A pool of {capacityBlocks} blocks exceeds the addressable span size.");
        }

        _totalFloats = capacityBlocks * FloatsPerBlock;
        _arena = backend.AllocateAligned(totalBytes, ArenaAlignment);
        if (_arena.Pointer is null)
        {
            throw new OutOfMemoryException($"Failed to allocate {totalBytes} bytes for the KV block arena.");
        }

        _base = (float*)_arena.Pointer;
        ClearArena();

        // Free list as a stack of ids: pop on Allocate, push on Release. Blocks are pushed in
        // descending order so the first allocations hand out ascending ids.
        _refCounts = new int[capacityBlocks];
        _freeStack = new int[capacityBlocks];
        for (int i = 0; i < capacityBlocks; i++)
        {
            _freeStack[i] = capacityBlocks - 1 - i;
        }

        _freeTop = capacityBlocks;
    }

    public int BlockTokens { get; }

    public int LayerCount { get; }

    public int KvHeads { get; }

    public int HeadDim { get; }

    /// <summary>Floats per token for one of K or V: kvHeads * headDim.</summary>
    public int KvDim { get; }

    internal long LayerRegion { get; }

    internal int FloatsPerBlock { get; }

    /// <summary>Total bytes stored in one physical block across all layers and both K/V halves.</summary>
    public long BytesPerBlock { get; }

    public int CapacityBlocks { get; }

    public int FreeBlocks => _freeTop;

    /// <summary>Physical blocks currently handed out, including shared (refcount &gt; 1) owners.</summary>
    public int UsedBlocks => CapacityBlocks - _freeTop;

    /// <summary>Total token slots the pool can hold at full capacity.</summary>
    public long SlotCapacity => (long)CapacityBlocks * BlockTokens;

    /// <summary>Token slots still available in free blocks.</summary>
    public long FreeSlots => (long)FreeBlocks * BlockTokens;

    /// <summary>Takes the next free block and returns its physical id with refcount 1.</summary>
    /// <exception cref="KvFullException">The pool has no free blocks left.</exception>
    public int Allocate()
    {
        ThrowIfDisposed();
        if (_freeTop == 0)
        {
            long totalBytes = (long)CapacityBlocks * BytesPerBlock;
            throw new KvFullException(
                $"KV block pool is full: all {CapacityBlocks} blocks of {BytesPerBlock} bytes " +
                $"({totalBytes / (1024 * 1024)} MiB total) are in use. Release unused sequences or raise kv_blocks.");
        }

        int id = _freeStack[--_freeTop];
        if (_refCounts[id] != 0)
        {
            throw new InvalidOperationException($"Internal pool corruption: free-listed block {id} still has refcount {_refCounts[id]}.");
        }

        _refCounts[id] = 1;
        return id;
    }

    /// <summary>Increments the reference count of a live block so another owner may share it.</summary>
    public void AddRef(int physicalBlockId)
    {
        ThrowIfDisposed();
        ValidateBlockId(physicalBlockId);
        if (_refCounts[physicalBlockId] == 0)
        {
            throw new InvalidOperationException($"Cannot add a reference to block {physicalBlockId}: it is not allocated.");
        }

        _refCounts[physicalBlockId]++;
    }

    /// <summary>Decrements the reference count; the block returns to the free list only at zero.</summary>
    public void Release(int physicalBlockId)
    {
        ThrowIfDisposed();
        ValidateBlockId(physicalBlockId);
        if (_refCounts[physicalBlockId] == 0)
        {
            throw new InvalidOperationException($"Cannot release block {physicalBlockId}: it is already free (double release).");
        }

        if (--_refCounts[physicalBlockId] == 0)
        {
            _freeStack[_freeTop++] = physicalBlockId;
        }
    }

    /// <summary>Current reference count of a block; zero means the block is free.</summary>
    public int RefCount(int physicalBlockId)
    {
        ThrowIfDisposed();
        ValidateBlockId(physicalBlockId);
        return _refCounts[physicalBlockId];
    }

    /// <summary>The whole payload of one physical block as a span over the arena.</summary>
    public unsafe ReadOnlySpan<float> BlockView(int physicalBlockId)
    {
        ThrowIfDisposed();
        ValidateBlockId(physicalBlockId);
        return new Span<float>(_base + (long)physicalBlockId * FloatsPerBlock, FloatsPerBlock);
    }

    /// <summary>Addressed view of one token slot's K or V region: kvDim floats at the documented offset.</summary>
    internal unsafe Span<float> SlotSpan(int blockId, int layer, bool isV, int offsetWithinBlock)
    {
        ThrowIfDisposed();
        ValidateBlockId(blockId);
        if ((uint)layer >= (uint)LayerCount)
        {
            throw new ArgumentOutOfRangeException(nameof(layer), $"Layer {layer} is outside the pool of {LayerCount} layers.");
        }

        if ((uint)offsetWithinBlock >= (uint)BlockTokens)
        {
            throw new ArgumentOutOfRangeException(nameof(offsetWithinBlock), $"Offset {offsetWithinBlock} is outside a block of {BlockTokens} tokens.");
        }

        long baseFloat = (long)blockId * FloatsPerBlock
                       + layer * LayerRegion
                       + (isV ? BlockTokens * (long)KvDim : 0)
                       + (long)offsetWithinBlock * KvDim;
        return new Span<float>(_base + baseFloat, KvDim);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _backend.FreeAligned(_arena);
        _disposed = true;
    }

    private unsafe void ClearArena() => new Span<float>(_base, _totalFloats).Clear();

    private void ValidateBlockId(int physicalBlockId)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(physicalBlockId);
        if (physicalBlockId >= CapacityBlocks)
        {
            throw new ArgumentOutOfRangeException(nameof(physicalBlockId), $"Block id {physicalBlockId} is outside the pool of {CapacityBlocks} blocks.");
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, typeof(BlockPool));
    }
}
