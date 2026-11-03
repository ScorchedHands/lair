using System.Buffers;
using Lair.Contracts;

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

    // Soft cap on usage within the fixed physical arena: allocations stop once this many blocks are handed out,
    // even while physically free blocks remain. Raised or lowered at runtime without touching the reservation —
    // a shrink never revokes already-allocated blocks, it only withholds new ones until usage drops back under
    // the cap on its own. Volatile so observer threads (admission checks) see a hot edit immediately.
    private int _budgetBlocks;

    // Admission-window bookkeeping: windows opened by the admitting scheduler (one per admitted sequence) carry a
    // shared ceiling — the widest reservation any still-running admitted sequence needs. While at least one window
    // is open, allocations may reach that ceiling even above a shrunk soft cap; it resets when all close.
    private readonly object _windowGate = new();
    private int _openWindows;
    private int _windowCeiling = -1;

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
        _budgetBlocks = capacityBlocks; // no cap until one is set: a fresh pool budgets its whole reservation
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

    /// <summary>Soft usage cap within the physical reservation: allocations stop once this many blocks are handed out. Equal to <see cref="CapacityBlocks"/> until a budget is set.</summary>
    public int EffectiveBudgetBlocks => Volatile.Read(ref _budgetBlocks);

    /// <summary>Free blocks still reachable under the soft cap — what admission and step accounting actually spend from.</summary>
    public int EffectiveFreeBlocks => Math.Max(Math.Min(FreeBlocks, EffectiveBudgetBlocks - UsedBlocks), 0);

    /// <summary>The ceiling allocations currently reach: the soft cap, or an open admission window's wider reservation while admitted sequences still run under it.</summary>
    public int AllocationCeiling => Math.Max(EffectiveBudgetBlocks, Volatile.Read(ref _windowCeiling));

    /// <summary>
    /// Sets the soft usage cap to <paramref name="blocks"/> blocks, clamped into [1, <see cref="CapacityBlocks"/>], and returns the value applied. The physical arena is never touched: raising the cap makes already-free blocks allocatable again immediately; lowering it withholds new allocations while in-use count exceeds the cap (already-allocated blocks are never revoked).
    /// </summary>
    public int SetEffectiveBudget(int blocks)
    {
        ThrowIfDisposed();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blocks);

        int applied = Math.Min(blocks, CapacityBlocks);
        Volatile.Write(ref _budgetBlocks, applied);
        return applied;
    }

    public int FreeBlocks => _freeTop;

    /// <summary>Physical blocks currently handed out, including shared (refcount &gt; 1) owners.</summary>
    public int UsedBlocks => CapacityBlocks - _freeTop;

    /// <summary>Total token slots the pool can hold at full capacity.</summary>
    public long SlotCapacity => (long)CapacityBlocks * BlockTokens;

    /// <summary>Token slots still available in free blocks.</summary>
    public long FreeSlots => (long)FreeBlocks * BlockTokens;

    /// <summary>Takes the next free block and returns its physical id with refcount 1.</summary>
    /// <exception cref="KvFullException">The pool has no reachable blocks left — physically empty, or usage already at whichever ceiling governs: a shrunk soft cap, or an admission window opened by the admitting scheduler (see <see cref="BeginAdmissionWindow"/>) so a sequence admitted at an earlier cap completes its reserved lifetime untouched.</exception>
    public int Allocate()
    {
        ThrowIfDisposed();

        // Physical exhaustion reports first. The budget gate then reaches only to the governing ceiling: the soft cap,
        // lifted while admission windows are open to those windows' reservations — what pre-admission accounting
        // reserved and eviction could not reclaim (blocks pinned by live sequences). Cached-block eviction and
        // finished sequences bring usage back under a shrunk cap.
        if (_freeTop == 0)
        {
            long totalBytes = (long)CapacityBlocks * BytesPerBlock;
            throw new KvFullException(
                $"KV block pool is full: all {CapacityBlocks} blocks of {BytesPerBlock} bytes " +
                $"({totalBytes / (1024 * 1024)} MiB total) are in use. Release unused sequences or raise kv_blocks.");
        }

        int ceiling = Math.Max(EffectiveBudgetBlocks, Volatile.Read(ref _windowCeiling));
        if (UsedBlocks >= ceiling)
        {
            throw new KvFullException(
                $"KV block pool is at its soft cap: the budget allows {ceiling} of {CapacityBlocks} reserved blocks " +
                $"to be in use and they are. Release unused sequences or raise kv_blocks.");
        }

        int id = _freeStack[--_freeTop];
        if (_refCounts[id] != 0)
        {
            throw new InvalidOperationException($"Internal pool corruption: free-listed block {id} still has refcount {_refCounts[id]}.");
        }

        _refCounts[id] = 1;
        return id;
    }

    /// <summary>
    /// Opens one admission window for a sequence whose whole-lifetime block need was checked against the effective
    /// budget by the admitting scheduler. Inside an open window, allocations may rise above the soft cap up to physical
    /// capacity — that headroom is exactly what pre-admission accounting reserved and eviction could not reclaim
    /// (blocks pinned by live sequences). A hot budget shrink landing mid-run therefore cannot strand a sequence that
    /// was admitted at the earlier cap. Call <see cref="EndAdmissionWindow"/> when the sequence finishes so later
    /// admissions respect the current cap again.
    /// </summary>
    public void BeginAdmissionWindow(int neededBlocks)
    {
        ThrowIfDisposed();

        lock (_windowGate)
        {
            // The shared window ceiling is the greater of "current used + what admission reserved" and any wider
            // ceiling an already-open window established (a shrink that drained nothing keeps every admitted sequence whole).
            int ceiling = Math.Max(UsedBlocks + Math.Max(neededBlocks, 0), Volatile.Read(ref _windowCeiling));
            Volatile.Write(ref _windowCeiling, Math.Min(Math.Max(ceiling, 1), CapacityBlocks));
            _openWindows++;
        }
    }

    /// <summary>Closes one admission window; the shared ceiling resets once every open window has closed.</summary>
    public void EndAdmissionWindow()
    {
        lock (_windowGate)
        {
            if (--_openWindows <= 0)
            {
                _openWindows = 0;
                Volatile.Write(ref _windowCeiling, -1);
            }
        }
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
    public unsafe ReadOnlySpan<float> BlockView(int physicalBlockId) => MutableBlockSpan(physicalBlockId);

    // Writable twin of the block view: reserved for in-pool copies such as SequenceKv's copy-on-write, which must
    // move a shared block's whole payload into a freshly allocated private block.
    internal unsafe Span<float> MutableBlockSpan(int physicalBlockId)
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
