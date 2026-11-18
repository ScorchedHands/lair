using System.Buffers;
using System.Runtime.InteropServices;
using Lair.Contracts;

namespace Lair.Kv;

/// <summary>
/// Fixed-size pool of quantized paged KV blocks, the generic twin of <see cref="BlockPool"/> for a resolved quantized KV
/// format pair. One physical block stores the K and V cache for every transformer layer of <c>blockTokens</c> token
/// positions, exactly as the float pool does, except each side region is laid out as whole GGML groups of a fixed-size
/// payload+scale block struct instead of floats. The arena is one aligned allocation owned by the backend; block, side, and
/// token views are spans over that arena valid for the pool's lifetime, so a block-granularity read is copy-less. Blocks are
/// handed out from a free-list stack in O(1) and carry a reference count for copy-on-write sharing, with the same
/// free-list/refcount semantics the non-generic pool runs. This is a storage-only pool: it proves a quantized pair's
/// footprint and its block addressing, and carries no residency/window/soft-cap machinery and no attention path.
/// </summary>
/// <remarks>
/// Block layout (bytes inside one block), with <c>groupsPerTokenK = kvDim / TK.ValuesPerBlock</c> and the V side analogous:
/// <code>
/// kvDim            = kvHeads * headDim                    values per token, per K or V
/// kSideBytes       = blockTokens * groupsPerTokenK * TK.BlockBytes   K side, one layer
/// vSideBytes       = blockTokens * groupsPerTokenV * TV.BlockBytes   V side, one layer
/// layerRegionBytes = kSideBytes + vSideBytes
/// bytesPerBlock    = layerCount * layerRegionBytes
///
/// byte offset of (blockId, layer, K/V, tokenSlot):
///   blockId * bytesPerBlock
/// + layer * layerRegionBytes
/// + (isV ? kSideBytes : 0)
/// + tokenSlot * groupsPerSideToken * Side.BlockBytes
/// </code>
    /// Each side region is contiguous per token slot at whole-group granularity, so a head slice inside it reaches a block span
    /// without a copy. The arena is zero-initialized at construction, so slots never written are raw zero bytes; quantized reads
    /// expect the writer to have filled every group they cover.
/// </remarks>
public sealed unsafe class BlockPool<TK, TV> : IDisposable
    where TK : struct, IKvFormat<TK>
    where TV : struct, IKvFormat<TV>
{
    private const int ArenaAlignment = 64;

    private readonly IBackend _backend;
    private readonly MemoryHandle _arena;
    private readonly byte* _base;
    private readonly long _totalBytes;

    // The resolved quantized KV format pair this pool was sized for. Cold only: it decides the arena's byte layout at
    // construction and is read by footprint accounting, never on a hot path.
    private readonly KvFormatPair _formats;

    private readonly int[] _refCounts;
    private readonly int[] _freeStack;
    private int _freeTop;

    // Per-side group geometry resolved once from the format structs and validated against the pair descriptor. Cold only.
    private readonly int _kGroupsPerToken;
    private readonly int _vGroupsPerToken;
    private readonly int _kSideBytesPerLayer;
    private readonly int _vSideBytesPerLayer;

    private bool _disposed;

    /// <summary>
    /// Opens a host-resident quantized pool sized for an explicit KV format pair. Only the quantized sides this storage item
    /// carries (<c>q8_0</c>, <c>q4_0</c>) are accepted: float sides refuse with actionable messages because their addressing is
    /// the non-generic pool's, and a struct whose geometry disagrees with the named pair (values-per-block or group bytes) also
    /// refuses. The arena allocation, its layout, and the free-list order mirror <see cref="BlockPool"/>'s exactly for whichever
    /// quantized side bytes the pair names.
    /// </summary>
    public BlockPool(int blockTokens, int layerCount, int kvHeads, int headDim, int capacityBlocks, IBackend backend, KvFormatPair formats)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacityBlocks);
        ArgumentNullException.ThrowIfNull(backend);

        ThrowForQuantizedSide(formats.K, "k", nameof(KvFormatPair.K));
        ThrowForQuantizedSide(formats.V, "v", nameof(KvFormatPair.V));

        BlockTokens = blockTokens;
        LayerCount = layerCount;
        KvHeads = kvHeads;
        HeadDim = headDim;
        KvDim = checked(kvHeads * headDim);
        CapacityBlocks = capacityBlocks;
        _formats = formats;
        _backend = backend;

        // Struct geometry must agree with the named pair: whole 32-value groups on both sides, and each side's block struct
        // carrying exactly that side format's group bytes. A pool opened as (Q8Block, Q4Block) against q8_0/q4_0 passes; one
        // whose structs disagree refuses here rather than sizing an arena no writer would fill correctly.
        ArgumentOutOfRangeException.ThrowIfNotEqual(TK.ValuesPerBlock, KvFormatPair.GroupValues);
        ArgumentOutOfRangeException.ThrowIfNotEqual(TV.ValuesPerBlock, KvFormatPair.GroupValues);

        int kGroupBytes = KvFormatPair.GroupBytes(formats.K);
        int vGroupBytes = KvFormatPair.GroupBytes(formats.V);
        ArgumentOutOfRangeException.ThrowIfNotEqual(TK.BlockBytes, kGroupBytes);
        ArgumentOutOfRangeException.ThrowIfNotEqual(TV.BlockBytes, vGroupBytes);

        // Whole groups per token region only: quantized KV head regions in this phase are group-aligned, so a token's K/V side
        // is a whole number of blocks. Partial-group token addressing is deliberately not solved here.
        if (KvDim % TK.ValuesPerBlock != 0 || KvDim % TV.ValuesPerBlock != 0)
        {
            throw new ArgumentException(
                $"a quantized KV pool needs whole {KvFormatPair.GroupValues}-value groups per token region, " +
                $"got kvDim={KvDim} for k='{KvSideFormatNames.Of(formats.K)}' ({TK.ValuesPerBlock} values/block) and " +
                $"v='{KvSideFormatNames.Of(formats.V)}' ({TV.ValuesPerBlock} values/block).",
                nameof(kvHeads));
        }

        _kGroupsPerToken = KvDim / TK.ValuesPerBlock;
        _vGroupsPerToken = KvDim / TV.ValuesPerBlock;
        _kSideBytesPerLayer = checked(blockTokens * _kGroupsPerToken * kGroupBytes);
        _vSideBytesPerLayer = checked(blockTokens * _vGroupsPerToken * vGroupBytes);
        int layerRegionBytes = checked(_kSideBytesPerLayer + _vSideBytesPerLayer);
        BytesPerBlock = checked((long)layerCount * layerRegionBytes);

        // The struct-derived layout must equal the descriptor's footprint formula for this pair, byte for byte.
        long declaredBytes = formats.BytesPerBlock(layerCount, blockTokens, KvDim);
        if (BytesPerBlock != declaredBytes)
        {
            throw new InvalidOperationException(
                $"block structs for k='{KvSideFormatNames.Of(formats.K)}' v='{KvSideFormatNames.Of(formats.V)}' lay out " +
                $"{BytesPerBlock} bytes per block, but the pair descriptor declares {declaredBytes}.");
        }

        _totalBytes = checked((long)capacityBlocks * BytesPerBlock);
        if (_totalBytes > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(capacityBlocks), $"A pool of {capacityBlocks} blocks exceeds the addressable arena size.");
        }

        _arena = backend.AllocateAligned(_totalBytes, ArenaAlignment);
        if (_arena.Pointer is null)
        {
            throw new OutOfMemoryException($"Failed to allocate {_totalBytes} bytes for the quantized KV block arena.");
        }

        _base = (byte*)_arena.Pointer;
        new Span<byte>(_base, (int)_totalBytes).Clear(); // zero-init the arena; quantized writes fill every group a reader covers

        // Free list as a stack of ids: pop on Allocate, push on Release. Blocks are pushed in descending order so the first
        // allocations hand out ascending ids, exactly as the non-generic pool does.
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

    /// <summary>Values per token for one of K or V: kvHeads * headDim.</summary>
    public int KvDim { get; }

    /// <summary>Total bytes stored in one physical block across all layers and both K/V sides under this pair.</summary>
    public long BytesPerBlock { get; }

    /// <summary>The resolved quantized KV format pair this pool's block layout was sized for.</summary>
    public KvFormatPair Formats => _formats;

    public int CapacityBlocks { get; }

    public int FreeBlocks => _freeTop;

    /// <summary>Physical blocks currently handed out, including shared (refcount &gt; 1) owners.</summary>
    public int UsedBlocks => CapacityBlocks - _freeTop;

    /// <summary>Total token slots the pool can hold at full capacity.</summary>
    public long SlotCapacity => (long)CapacityBlocks * BlockTokens;

    /// <summary>Token slots still available in free blocks.</summary>
    public long FreeSlots => (long)FreeBlocks * BlockTokens;

    /// <summary>Takes the next free block and returns its physical id with refcount 1.</summary>
    /// <exception cref="KvFullException">The pool has no physically free blocks left.</exception>
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

    /// <summary>The whole payload of one physical block as a raw byte span over the arena.</summary>
    public ReadOnlySpan<byte> RawBlockView(int blockId)
    {
        ThrowIfDisposed();
        ValidateBlockId(blockId);
        return new ReadOnlySpan<byte>(_base + (long)blockId * BytesPerBlock, (int)BytesPerBlock);
    }

    // Writable twin of the raw block view: reserved for in-pool copies such as SequenceKv<TK,TV>'s copy-on-write, which moves
    // a shared block's whole payload into a freshly allocated private block without re-quantizing it.
    internal Span<byte> MutableRawBlockSpan(int blockId)
    {
        ThrowIfDisposed();
        ValidateBlockId(blockId);
        return new Span<byte>(_base + (long)blockId * BytesPerBlock, (int)BytesPerBlock);
    }

    /// <summary>
    /// Copies one physical block's raw bytes whole into another. No re-quantization, no per-side reinterpretation: every byte
    // of the source — written groups and untouched zero groups alike — lands in the destination bit-identically. The
    // destination must be a live block; the source may be shared.
    /// </summary>
    public void CopyBlockWhole(int sourceBlockId, int destinationBlockId)
    {
        ThrowIfDisposed();
        ValidateBlockId(sourceBlockId);
        ValidateBlockId(destinationBlockId);
        RawBlockView(sourceBlockId).CopyTo(MutableRawBlockSpan(destinationBlockId));
    }

    /// <summary>Typed view over one layer's whole K side (every token slot's groups) as q8/q4 blocks.</summary>
    public ReadOnlySpan<TK> KSideView(int blockId, int layer)
    {
        ThrowIfDisposed();
        ValidateBlockId(blockId);
        ValidateLayer(layer);
        return MemoryMarshal.Cast<byte, TK>(SideSpan(blockId, layer, isV: false, _kSideBytesPerLayer));
    }

    /// <summary>Typed view over one layer's whole V side as blocks.</summary>
    public ReadOnlySpan<TV> VSideView(int blockId, int layer)
    {
        ThrowIfDisposed();
        ValidateBlockId(blockId);
        ValidateLayer(layer);
        return MemoryMarshal.Cast<byte, TV>(SideSpan(blockId, layer, isV: true, _vSideBytesPerLayer));
    }

    /// <summary>Writable twin of <see cref="KSideView"/>.</summary>
    public Span<TK> MutableKSideSpan(int blockId, int layer)
    {
        ThrowIfDisposed();
        ValidateBlockId(blockId);
        ValidateLayer(layer);
        return MemoryMarshal.Cast<byte, TK>(MutableSideSpan(blockId, layer, isV: false, _kSideBytesPerLayer));
    }

    /// <summary>Writable twin of <see cref="VSideView"/>.</summary>
    public Span<TV> MutableVSideSpan(int blockId, int layer)
    {
        ThrowIfDisposed();
        ValidateBlockId(blockId);
        ValidateLayer(layer);
        return MemoryMarshal.Cast<byte, TV>(MutableSideSpan(blockId, layer, isV: true, _vSideBytesPerLayer));
    }

    /// <summary>Zero-copy view of one token slot's K side as its whole groups.</summary>
    public ReadOnlySpan<TK> KTokenView(int blockId, int layer, int tokenSlot)
    {
        ThrowIfDisposed();
        ValidateBlockId(blockId);
        ValidateLayer(layer);
        ValidateTokenSlot(tokenSlot);
        return MemoryMarshal.Cast<byte, TK>(TokenSpan(blockId, layer, isV: false, tokenSlot, _kGroupsPerToken * TK.BlockBytes));
    }

    /// <summary>Zero-copy view of one token slot's V side as its whole groups.</summary>
    public ReadOnlySpan<TV> VTokenView(int blockId, int layer, int tokenSlot)
    {
        ThrowIfDisposed();
        ValidateBlockId(blockId);
        ValidateLayer(layer);
        ValidateTokenSlot(tokenSlot);
        return MemoryMarshal.Cast<byte, TV>(TokenSpan(blockId, layer, isV: true, tokenSlot, _vGroupsPerToken * TV.BlockBytes));
    }

    /// <summary>Writable twin of <see cref="KTokenView"/>: the span a quantizing write fills.</summary>
    public Span<TK> MutableKTokenSpan(int blockId, int layer, int tokenSlot)
    {
        ThrowIfDisposed();
        ValidateBlockId(blockId);
        ValidateLayer(layer);
        ValidateTokenSlot(tokenSlot);
        return MemoryMarshal.Cast<byte, TK>(MutableTokenSpan(blockId, layer, isV: false, tokenSlot, _kGroupsPerToken * TK.BlockBytes));
    }

    /// <summary>Writable twin of <see cref="VTokenView"/>.</summary>
    public Span<TV> MutableVTokenSpan(int blockId, int layer, int tokenSlot)
    {
        ThrowIfDisposed();
        ValidateBlockId(blockId);
        ValidateLayer(layer);
        ValidateTokenSlot(tokenSlot);
        return MemoryMarshal.Cast<byte, TV>(MutableTokenSpan(blockId, layer, isV: true, tokenSlot, _vGroupsPerToken * TV.BlockBytes));
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

    private Span<byte> SideSpan(int blockId, int layer, bool isV, int sideBytes)
    {
        long offset = (long)blockId * BytesPerBlock + (long)layer * (_kSideBytesPerLayer + _vSideBytesPerLayer) + (isV ? _kSideBytesPerLayer : 0);
        return new Span<byte>(_base + offset, sideBytes);
    }

    private Span<byte> MutableSideSpan(int blockId, int layer, bool isV, int sideBytes) =>
        SideSpan(blockId, layer, isV, sideBytes);

    private Span<byte> TokenSpan(int blockId, int layer, bool isV, int tokenSlot, int tokenBytes)
    {
        int groupsPerToken = isV ? _vGroupsPerToken : _kGroupsPerToken;
        long offset = (long)blockId * BytesPerBlock
                    + (long)layer * (_kSideBytesPerLayer + _vSideBytesPerLayer)
                    + (isV ? _kSideBytesPerLayer : 0)
                    + (long)tokenSlot * groupsPerToken * (isV ? TV.BlockBytes : TK.BlockBytes);
        return new Span<byte>(_base + offset, tokenBytes);
    }

    private Span<byte> MutableTokenSpan(int blockId, int layer, bool isV, int tokenSlot, int tokenBytes) =>
        TokenSpan(blockId, layer, isV, tokenSlot, tokenBytes);

    private static void ThrowForQuantizedSide(KvSideFormat side, string name, string paramName)
    {
        if (side is KvSideFormat.Q8_0 or KvSideFormat.Q4_0)
        {
            return;
        }

        throw new ArgumentException(
            $"a quantized KV pool accepts only the quantized sides 'q8_0' and 'q4_0'; '{name}' declares " +
            $"'{KvSideFormatNames.Of(side)}', whose float storage the non-generic BlockPool addresses instead.",
            paramName);
    }

    private void ValidateBlockId(int physicalBlockId)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(physicalBlockId);
        if (physicalBlockId >= CapacityBlocks)
        {
            throw new ArgumentOutOfRangeException(nameof(physicalBlockId), $"Block id {physicalBlockId} is outside the pool of {CapacityBlocks} blocks.");
        }
    }

    private void ValidateLayer(int layer)
    {
        if ((uint)layer >= (uint)LayerCount)
        {
            throw new ArgumentOutOfRangeException(nameof(layer), $"Layer {layer} is outside the pool of {LayerCount} layers.");
        }
    }

    private void ValidateTokenSlot(int tokenSlot)
    {
        if ((uint)tokenSlot >= (uint)BlockTokens)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenSlot), $"Offset {tokenSlot} is outside a block of {BlockTokens} tokens.");
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, typeof(BlockPool<TK, TV>));
    }
}
