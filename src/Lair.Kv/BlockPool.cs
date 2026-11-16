using System.Buffers;
using Lair.Contracts;

namespace Lair.Kv;

/// <summary>
/// Fixed-size pool of paged KV blocks. One physical block stores the K and V cache for every
/// transformer layer of <c>blockTokens</c> token positions, so one logical position maps to one
/// physical block. Blocks are handed out from a free-list stack in O(1); each block carries a
/// reference count so later copy-on-write sharing can alias a block across sequences. The arena is
/// a single aligned allocation owned by the backend; block views are spans over that arena and are
/// valid for the pool's lifetime. A residency mode says which block ids also carry a device-resident
/// copy through the backend's KV mirror seam: <c>dram</c> opens none, <c>vram</c> declares the whole
/// reservation device-backed, and <c>split</c> divides the ids between a device-backed leading range
/// and a host-only tail. Whichever mode runs, attention still reads this pool's own host arena — the
/// mirror keeps a second copy of declared blocks current on flush; it is not an address space any read uses.
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

    // Windowed-read accounting is resolved once at model open and never consulted on the attention hot path. It keeps
    // the pool honest about what a declared window would read under the current soft budget without changing storage:
    // whole-KV blocks still stay allocated for layers that attend outside those windows.
    private readonly int[] _layerWindowTokens;
    private int _windowedLayerCount;
    // Admission-window bookkeeping: windows opened by the admitting scheduler (one per admitted sequence) carry a
    // shared ceiling — the widest reservation any still-running admitted sequence needs. While at least one window
    // is open, allocations may reach that ceiling even above a shrunk soft cap; it resets when all close.
    private readonly object _windowGate = new();
    private int _openWindows;
    private int _windowCeiling = -1;

    // Residency seam: when the mode declares a device-backed range, block ids [0, _deviceBackedBlocks) also carry a
    // persistent device copy through the backend's KV mirror. Host attention reads never consult it — writes mark its
    // dirty set and this pool's flush points (admission-window end, release-to-zero, an explicit call, dispose) push
    // whole-block copies across. A dram-mode pool opens no mirror at all.
    private readonly BlockResidencyMode _residency;
    private readonly int _deviceBackedBlocks;
    private readonly IKvDeviceMirror? _deviceMirror;

    private bool _disposed;

    /// <summary>Opens a host-resident pool: every block id lives only in the arena, exactly as this pool has always run.</summary>
    public BlockPool(int blockTokens, int layerCount, int kvHeads, int headDim, int capacityBlocks, IBackend backend)
        : this(blockTokens, layerCount, kvHeads, headDim, capacityBlocks, backend, BlockResidencyMode.Dram)
    {
    }

    /// <summary>
    /// Opens a pool with an explicit residency mode. <see cref="BlockResidencyMode.Dram"/> opens no device mirror and is
    /// byte-for-byte the host-only posture. <see cref="BlockResidencyMode.Vram"/> declares every block id in the
    /// reservation device-backed, and <see cref="BlockResidencyMode.Split"/> divides the ids so the leading range is
    /// device-backed; both ask the backend for a KV mirror over this arena covering exactly that range, which a backend
    /// without device-resident storage refuses naming itself. The arena allocation, its layout, and the free-list order
    /// are identical in every mode — residency changes what copy of each block's bytes is kept where, never how ids or
    /// offsets address them.
    /// </summary>
    public unsafe BlockPool(int blockTokens, int layerCount, int kvHeads, int headDim, int capacityBlocks, IBackend backend, BlockResidencyMode residency)
        : this(blockTokens, layerCount, kvHeads, headDim, capacityBlocks, backend, residency, DeclaredDeviceRange(residency, capacityBlocks), mirrorOwner: null)
    {
    }

    /// <summary>
    /// Opens a pool whose device-backed range was resolved by its caller rather than derived here. The host allocation and
    /// the free list stay owned by <paramref name="backend"/>; only the persistent device copy of the declared leading block
    /// range belongs to <paramref name="mirrorOwner"/>, which is how an accelerator-resident reservation runs while compute —
    /// and therefore every attention read — stays on the host backend that owns this arena. A mirror owner with no
    /// device-resident storage refuses naming itself, exactly as it does when this pool asks its own allocator. The declared
    /// range must be one this residency could have derived from the reservation: a divided residency never widens past half
    /// of it, and an accelerator-resident one covers all of it.
    /// </summary>
    public unsafe BlockPool(int blockTokens, int layerCount, int kvHeads, int headDim, int capacityBlocks, IBackend backend, BlockResidencyMode residency, int deviceBackedBlocks, IBackend? mirrorOwner)
    {
        ValidateDeclaredDeviceRange(residency, capacityBlocks, deviceBackedBlocks);

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
        _layerWindowTokens = new int[layerCount];
        _windowedLayerCount = 0;
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

        // Residency resolves against the ids this pool will hand out, in exactly the order it hands them out: ascending.
        // The device-backed range is therefore the leading block ids, and a split reservation puts its host-only tail on
        // the ids allocation reaches last. A caller that resolved its own range from what a model declares gets the same
        // ascending coverage — only how many leading ids it covers differs.
        _residency = residency;
        _deviceBackedBlocks = deviceBackedBlocks;

        if (_deviceBackedBlocks > 0)
        {
            try
            {
                _deviceMirror = (mirrorOwner ?? backend).OpenKvMirror(new KvMirrorRequest
                {
                    HostArenaBase = (nint)_arena.Pointer,
                    HostArenaBytes = totalBytes,
                    BytesPerBlock = BytesPerBlock,
                    FirstBlockId = 0,
                    BlockCount = _deviceBackedBlocks,
                    Tier = WeightTier.Vram,
                    ResourceName = "kv block arena",
                });
            }
            catch
            {
                // A backend that refuses the residency leaves no half-open pool behind: the arena this constructor
                // just took from the backend goes back before the refusal reaches the caller.
                backend.FreeAligned(_arena);
                throw;
            }
        }
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

    /// <summary>Declares the per-layer attention windows this pool accounts for. The values are read-side accounting only: storage and allocation remain whole-KV.</summary>
    public void DeclareAttentionWindows(ReadOnlySpan<int> layerWindows)
    {
        ThrowIfDisposed();

        if (layerWindows.Length != LayerCount)
        {
            throw new ArgumentException($"attention-window table must cover {LayerCount} layers, got {layerWindows.Length}.", nameof(layerWindows));
        }

        int windowed = 0;
        for (int layer = 0; layer < layerWindows.Length; layer++)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(layerWindows[layer]);
            _layerWindowTokens[layer] = layerWindows[layer];
            if (layerWindows[layer] > 0)
            {
                windowed++;
            }
        }

        Volatile.Write(ref _windowedLayerCount, windowed);
    }

    /// <summary>The number of layers declared with a finite attention window.</summary>
    public int DeclaredWindowedLayerCount => Volatile.Read(ref _windowedLayerCount);

    /// <summary>Logical KV blocks the model's attention would select across all layers at the current soft budget.</summary>
    public long KvReadFootprintBlocksAtBudget
    {
        get
        {
            ThrowIfDisposed();
            ComputeKvReadFootprints(Volatile.Read(ref _budgetBlocks), out long read, out _);
            return read;
        }
    }

    /// <summary>Whole logical blocks excluded by declared windows at the current soft budget.</summary>
    public long SkippedKvReadFootprintBlocksAtBudget
    {
        get
        {
            ThrowIfDisposed();
            ComputeKvReadFootprints(Volatile.Read(ref _budgetBlocks), out _, out long skipped);
            return skipped;
        }
    }

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
        MarkDeviceDirty(id); // a freshly handed-out device-backed block is about to receive host writes its device copy does not hold yet
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

    /// <summary>Closes one admission window; the shared ceiling resets once every open window has closed. Closing a window is also this pool's residency flush point: everything an admitted sequence wrote into device-backed blocks crosses to its device copy at the window's end, so the next admission never sees a stale range.</summary>
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

        FlushDeviceMirror(); // window-end flush runs outside the gate: driver work must not hold the ceiling lock against admitting threads
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

    /// <summary>Decrements the reference count; the block returns to the free list only at zero. A device-backed block that drops to zero is flushed first, so the freed arena region's zeros cross to its device copy instead of leaving stale bytes resident on the device.</summary>
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
            MarkDeviceDirty(physicalBlockId); // the last owner let go: whatever its device copy holds must be brought in line with the host bytes now sitting here
            FlushDeviceMirror();
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
    public unsafe ReadOnlySpan<float> BlockView(int physicalBlockId) => ReadBlockSpan(physicalBlockId);

    // Writable twin of the block view: reserved for in-pool copies such as SequenceKv's copy-on-write, which must
    // move a shared block's whole payload into a freshly allocated private block. A write through this span marks a
    // device-backed block dirty — its host bytes just changed and the device copy is owed a crossing before it can be
    // called current again.
    internal unsafe Span<float> MutableBlockSpan(int physicalBlockId)
    {
        ThrowIfDisposed();
        ValidateBlockId(physicalBlockId);
        MarkDeviceDirty(physicalBlockId);
        return new Span<float>(_base + (long)physicalBlockId * FloatsPerBlock, FloatsPerBlock);
    }

    private unsafe Span<float> ReadBlockSpan(int physicalBlockId)
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

    /// <summary>The residency mode this pool opened with: which block ids carry a device-resident copy alongside their host bytes.</summary>
    public BlockResidencyMode Residency => _residency;

    /// <summary>How many leading block ids are declared device-backed. Zero for a host-only pool.</summary>
    public int DeviceBackedBlockCount => _deviceBackedBlocks;

    /// <summary>True when this physical id falls inside the device-backed range and its bytes carry a persistent copy on the device.</summary>
    public bool IsDeviceBacked(int physicalBlockId) => _deviceMirror is not null && (uint)physicalBlockId < (uint)_deviceBackedBlocks;

    /// <summary>Bytes of persistent device-resident KV storage this pool's mirror holds — zero for a host-only pool.</summary>
    public long DeviceMirrorBytes => _deviceMirror?.DeviceBytes ?? 0;

    /// <summary>Distinct device-backed blocks whose whole payload has crossed to the device copy at least once.</summary>
    public int DeviceMirrorCrossedBlocks => _deviceMirror?.CrossedBlocks ?? 0;

    /// <summary>Total block payloads copied host-to-device across every crossing this pool caused, counting each crossing.</summary>
    public long DeviceMirrorCrossedBytes => _deviceMirror?.CrossedBytes ?? 0;

    /// <summary>Device-backed blocks written since their last crossing — what the next flush still owes.</summary>
    public int DeviceMirrorPendingWrites => _deviceMirror?.PendingWrites ?? 0;

    /// <summary>Copies every dirty device-backed block to its device copy and clears the pending set. A host-only pool, or a mirror with nothing pending, does nothing.</summary>
    public void FlushDeviceMirror()
    {
        ThrowIfDisposed();
        _deviceMirror?.Flush();
    }

    // Records that one block's host bytes changed so its next crossing carries them. Reads never call this: only write
    // paths do — a fresh allocation about to be filled, a copy-on-write destination span, and SequenceKv's K/V writes.
    internal void MarkWritten(int physicalBlockId) => MarkDeviceDirty(physicalBlockId);

    private void MarkDeviceDirty(int physicalBlockId)
    {
        if (_deviceMirror is not null && (uint)physicalBlockId < (uint)_deviceBackedBlocks)
        {
            _deviceMirror.MarkDirty(physicalBlockId);
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        FlushDeviceMirror(); // the last crossing happens while the arena is still alive: a disposed mirror must not be left holding stale bytes
        _deviceMirror?.Dispose();
        _backend.FreeAligned(_arena);
        _disposed = true;
    }

    // The device-backed range each residency declares for itself over a reservation of this size: none at all host-only,
    // the whole reservation accelerator-resident, and half — at least one block — when divided. Leading ids either way,
    // because that is where ascending allocation lands first.
    private static int DeclaredDeviceRange(BlockResidencyMode residency, int capacityBlocks) => residency switch
    {
        BlockResidencyMode.Vram => capacityBlocks,
        BlockResidencyMode.Split => Math.Max(1, capacityBlocks / 2),
        _ => 0,
    };

    // Checks a caller-resolved range against what this residency could have declared over the reservation. Narrowing below
    // the midpoint is exactly what an explicit finite ceiling does; widening past it — or covering less than the whole
    // reservation under an accelerator-resident declaration — states a posture no residency carries, so it refuses here
    // rather than opening a pool whose reported mode holds something else.
    private static int ValidateDeclaredDeviceRange(BlockResidencyMode residency, int capacityBlocks, int deviceBackedBlocks) => residency switch
    {
        BlockResidencyMode.Dram when deviceBackedBlocks != 0 => throw new ArgumentException(
            $"a host-only residency declares no device-backed block range; {deviceBackedBlocks} blocks were declared", nameof(deviceBackedBlocks)),

        BlockResidencyMode.Vram when deviceBackedBlocks != capacityBlocks => throw new ArgumentOutOfRangeException(
            nameof(deviceBackedBlocks),
            $"an accelerator-resident residency covers the whole reservation of {capacityBlocks} blocks, not {deviceBackedBlocks}"),

        BlockResidencyMode.Split when deviceBackedBlocks < 1 || deviceBackedBlocks > Math.Max(1, capacityBlocks / 2) => throw new ArgumentOutOfRangeException(
            nameof(deviceBackedBlocks),
            $"a divided residency of {capacityBlocks} blocks device-backs at least one leading block and never more than half the reservation " +
            $"({Math.Max(1, capacityBlocks / 2)}); {deviceBackedBlocks} was declared"),

        _ => deviceBackedBlocks,
    };

    private unsafe void ClearArena() => new Span<float>(_base, _totalFloats).Clear();

    private void ValidateBlockId(int physicalBlockId)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(physicalBlockId);
        if (physicalBlockId >= CapacityBlocks)
        {
            throw new ArgumentOutOfRangeException(nameof(physicalBlockId), $"Block id {physicalBlockId} is outside the pool of {CapacityBlocks} blocks.");
        }
    }

    private void ComputeKvReadFootprints(int budgetBlocks, out long readBlocks, out long skippedBlocks)
    {
        readBlocks = 0;
        skippedBlocks = 0;

        if (budgetBlocks <= 0)
        {
            return;
        }

        long budgetSlots = (long)budgetBlocks * BlockTokens;

        for (int layer = 0; layer < LayerCount; layer++)
        {
            int window = _layerWindowTokens[layer];
            long visiblePositions = window > 0 ? Math.Min(window, budgetSlots) : budgetSlots;
            long selectedBlocks = (visiblePositions + BlockTokens - 1) / BlockTokens;

            readBlocks += selectedBlocks;

            if (window > 0)
            {
                skippedBlocks += budgetBlocks - selectedBlocks;
            }
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, typeof(BlockPool));
    }
}
