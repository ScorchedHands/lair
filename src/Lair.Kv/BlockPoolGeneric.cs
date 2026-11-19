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
/// free-list/refcount semantics the non-generic pool runs. The scheduler-facing bookkeeping the non-generic pool carries —
/// soft budget, admission windows, and windowed-read footprint accounting — runs here too, so a quantized pair admits and
/// reports through the same pool surface a float model schedules against. Quantized blocks are host-only in this build: no
/// device mirror opens for them, and the residency members report that posture.
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
public sealed unsafe class BlockPool<TK, TV> : IKvPool, IDisposable
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

    // Soft cap on usage within the fixed physical arena, the admission-window ceiling, and the windowed-read accounting —
    // the same scheduler-facing bookkeeping the non-generic pool runs, and resolved off the attention hot path exactly as
    // there. Volatile so observer threads (admission checks) see a hot budget edit immediately.
    private int _budgetBlocks;
    private readonly int[] _layerWindowTokens;
    private int _windowedLayerCount;
    private readonly object _windowGate = new();
    private int _openWindows;
    private int _windowCeiling = -1;

    // Per-side group geometry resolved once from the format structs and validated against the pair descriptor. Cold only.
    private readonly int _kGroupsPerToken;
    private readonly int _vGroupsPerToken;
    private readonly int _kSideBytesPerLayer;
    private readonly int _vSideBytesPerLayer;

    // Residency seam, resolved once at pool open. A host-only quantized pool opens no device mirror and is the posture
    // every quantized pool has always run. A device-backed quantized residency declares a leading block range whose
    // group bytes carry a persistent device copy through the backend's KV mirror, and the device-step posture hands that
    // copy to a GPU step as a store its attention walk reads. The combo the store carries is resolved from the format
    // pair here — the single place a quantized device attention kernel is selected — so a step never branches on a
    // format per position. Unsupported combos refuse at open naming the format and posture rather than opening a store
    // no device attention kernel here can walk.
    private readonly BlockResidencyMode _residency;
    private readonly KvMirrorExecution _execution;
    private readonly int _deviceBackedBlocks;
    private readonly IKvDeviceMirror? _deviceMirror;

    private bool _disposed;

    /// <summary>
    /// Opens a host-resident quantized pool sized for an explicit KV format pair. Only the quantized sides this storage item
    /// carries (<c>q8_0</c>, <c>q4_0</c>) are accepted: float sides refuse with actionable messages because their addressing is
    /// the non-generic pool's, and a struct whose geometry disagrees with the named pair (values-per-block or group bytes) also
    /// refuses. The arena allocation, its layout, and the free-list order mirror <see cref="BlockPool"/>'s exactly for whichever
    /// quantized side bytes the pair names. This is the host-only posture every quantized pool has always run.
    /// </summary>
    public BlockPool(int blockTokens, int layerCount, int kvHeads, int headDim, int capacityBlocks, IBackend backend, KvFormatPair formats)
        : this(blockTokens, layerCount, kvHeads, headDim, capacityBlocks, backend, formats, BlockResidencyMode.Dram, KvMirrorExecution.HostArenaRead)
    {
    }

    /// <summary>
    /// Opens a quantized pool with an explicit residency mode and execution posture. <see cref="BlockResidencyMode.Dram"/>
    /// opens no device mirror and is the host-only posture. <see cref="BlockResidencyMode.Vram"/> declares every block id
    /// device-backed and <see cref="BlockResidencyMode.Split"/> divides the ids so the leading range is device-backed; both
    /// ask the backend for a KV mirror over this arena covering exactly that range. <see cref="KvMirrorExecution.DeviceStep"/>
    /// hands that copy to a GPU step as a store its attention walk reads. A device-backed quantized residency resolves the
    /// combo the store carries from the format pair here — the single place a quantized device attention kernel is selected —
    /// and refuses an unsupported combo (a q4_0 K side, or a q4_0/q4_0 pair) or a host-only residency naming the format,
    /// tier, and posture rather than opening a store no device attention kernel here can walk.
    /// </summary>
    public BlockPool(int blockTokens, int layerCount, int kvHeads, int headDim, int capacityBlocks, IBackend backend, KvFormatPair formats, BlockResidencyMode residency, KvMirrorExecution execution)
        : this(blockTokens, layerCount, kvHeads, headDim, capacityBlocks, backend, formats, residency, execution, DeclaredDeviceRange(residency, capacityBlocks), mirrorOwner: null)
    {
    }

    /// <summary>
    /// Opens a quantized pool whose device-backed range was resolved by its caller rather than derived here. The host
    /// allocation and free list stay owned by <paramref name="backend"/>; only the persistent device copy of the declared
    /// leading block range belongs to <paramref name="mirrorOwner"/>, which is how a quantized device-backed reservation runs
    /// while compute stays on the backend that owns this arena. A mirror owner with no device-resident storage refuses
    /// naming itself, exactly as when this pool asks its own allocator.
    /// </summary>
    public BlockPool(int blockTokens, int layerCount, int kvHeads, int headDim, int capacityBlocks, IBackend backend, KvFormatPair formats, BlockResidencyMode residency, KvMirrorExecution execution, int deviceBackedBlocks, IBackend? mirrorOwner)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacityBlocks);
        ArgumentNullException.ThrowIfNull(backend);

        ThrowForQuantizedSide(formats.K, "k", nameof(KvFormatPair.K));
        ThrowForQuantizedSide(formats.V, "v", nameof(KvFormatPair.V));

        ValidateDeclaredDeviceRange(residency, capacityBlocks, deviceBackedBlocks);
        KvMirrorFormatCombo combo = ValidateExecutionCombo(residency, execution, deviceBackedBlocks, formats);

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

        _budgetBlocks = capacityBlocks; // no cap until one is set: a fresh pool budgets its whole reservation
        _layerWindowTokens = new int[layerCount];
        _windowedLayerCount = 0;

        // Residency resolves against the ids this pool hands out, ascending — the device-backed range is the leading ids, and
        // a split reservation puts its host-only tail on the ids allocation reaches last. The combo resolved above is the
        // layout the mirror's store carries and the device attention kernel a bound step selects.
        _residency = residency;
        _execution = execution;
        _deviceBackedBlocks = deviceBackedBlocks;

        if (_deviceBackedBlocks > 0)
        {
            try
            {
                _deviceMirror = (mirrorOwner ?? backend).OpenKvMirror(new KvMirrorRequest
                {
                    HostArenaBase = (nint)_arena.Pointer,
                    HostArenaBytes = _totalBytes,
                    BytesPerBlock = BytesPerBlock,
                    FirstBlockId = 0,
                    BlockCount = _deviceBackedBlocks,
                    Tier = WeightTier.Vram,
                    ResourceName = "quantized kv block arena",
                    Execution = execution,
                    BlockTokens = blockTokens,
                    LayerCount = layerCount,
                    KvDim = KvDim,
                    Combo = combo,
                    KGroupBytes = kGroupBytes,
                    VGroupBytes = vGroupBytes,
                    GroupsPerTokenK = _kGroupsPerToken,
                    GroupsPerTokenV = _vGroupsPerToken,
                    KSideBytesPerLayer = _kSideBytesPerLayer,
                    LayerRegionBytes = layerRegionBytes,
                });
            }
            catch
            {
                // A backend that refuses the residency leaves no half-open pool behind: the arena this constructor just took
                // from the backend goes back before the refusal reaches the caller.
                backend.FreeAligned(_arena);
                throw;
            }
        }
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
    /// Sets the soft usage cap to <paramref name="blocks"/> blocks, clamped into [1, <see cref="CapacityBlocks"/>], and returns the value applied.
    /// The physical arena is never touched: raising the cap makes already-free blocks allocatable again immediately; lowering it withholds new
    /// allocations while in-use count exceeds the cap (already-allocated blocks are never revoked).
    /// </summary>
    public int SetEffectiveBudget(int blocks)
    {
        ThrowIfDisposed();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blocks);

        int applied = Math.Min(blocks, CapacityBlocks);
        Volatile.Write(ref _budgetBlocks, applied);
        return applied;
    }

    /// <summary>
    /// Opens one admission window for a sequence whose whole-lifetime block need was checked against the effective budget by the
    /// admitting scheduler. Inside an open window, allocations may rise above the soft cap up to physical capacity — that headroom is
    /// exactly what pre-admission accounting reserved and eviction could not reclaim. Call <see cref="EndAdmissionWindow"/> when the
    /// sequence finishes so later admissions respect the current cap again.
    /// </summary>
    public void BeginAdmissionWindow(int neededBlocks)
    {
        ThrowIfDisposed();

        lock (_windowGate)
        {
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

    /// <summary>The residency mode this pool opened with: which block ids carry a device-resident copy alongside their host bytes. A host-only quantized pool reports <see cref="BlockResidencyMode.Dram"/>.</summary>
    public BlockResidencyMode Residency => _residency;

    /// <summary>The execution posture this pool's device-backed range runs in: host-read residency accounting, or the store a GPU step reads. A host-only pool reports the host-read posture.</summary>
    public KvMirrorExecution Execution => _execution;

    /// <summary>True when this pool's device-backed range is a store a GPU step reads and writes on-device rather than a copy a host seam reads back.</summary>
    public bool DeviceExecuted => _deviceMirror?.Execution == KvMirrorExecution.DeviceStep;

    /// <summary>How many leading block ids are declared device-backed. Zero for a host-only pool.</summary>
    public int DeviceBackedBlockCount => _deviceBackedBlocks;

    /// <summary>True when this physical id falls inside the device-backed range and its bytes carry a persistent copy on the device.</summary>
    public bool IsDeviceBacked(int physicalBlockId) => _deviceMirror is not null && (uint)physicalBlockId < (uint)_deviceBackedBlocks;

    /// <summary>
    /// Hands this pool's device-backed range to a device step as a store, in this pool's block layout. A host-only pool or
    /// a host-read residency answers false: nothing here exposes device bytes as a store to read.
    /// </summary>
    public bool TryGetDeviceStore(out KvDeviceStore store)
    {
        ThrowIfDisposed();
        if (_deviceMirror is null)
        {
            store = default;
            return false;
        }

        return _deviceMirror.TryGetDeviceStore(out store);
    }

    /// <summary>The mirror that owns this pool's device-backed range, when this pool runs the device-step posture and a device step is to be bound to it. Null for a host-only pool or a host-read residency.</summary>
    public IKvDeviceMirror? DeviceStepMirror => DeviceExecuted ? _deviceMirror : null;

    /// <summary>Device-backed blocks a GPU step wrote on-device through this pool's store. Zero unless this pool runs the device-step posture.</summary>
    public int DeviceWrittenBlocks => _deviceMirror?.DeviceWrittenBlocks ?? 0;

    /// <summary>Bytes of persistent device-resident KV storage this pool's mirror holds — zero for a host-only pool.</summary>
    public long DeviceMirrorBytes => _deviceMirror?.DeviceBytes ?? 0;

    /// <summary>Distinct device-backed blocks whose whole payload has crossed to the device copy at least once.</summary>
    public int DeviceMirrorCrossedBlocks => _deviceMirror?.CrossedBlocks ?? 0;

    /// <summary>Total block payloads copied host-to-device across every crossing this pool caused, counting each crossing.</summary>
    public long DeviceMirrorCrossedBytes => _deviceMirror?.CrossedBytes ?? 0;

    /// <summary>Device-backed blocks written since their last crossing — what the next flush still owes.</summary>
    public int DeviceMirrorPendingWrites => _deviceMirror?.PendingWrites ?? 0;

    /// <summary>
    /// Copies one device-backed block's whole payload from this pool's device store back into <paramref name="output"/>,
    /// the explicit host observation of bytes a device step wrote and the host arena does not hold. A host-only pool or a
    /// host-read residency refuses: their host arena already is the copy a host read uses.
    /// </summary>
    public void ReadDeviceBlockToHost(int physicalBlockId, Span<float> output)
    {
        ThrowIfDisposed();
        if (_deviceMirror is null || _deviceMirror.Execution != KvMirrorExecution.DeviceStep)
        {
            throw new InvalidOperationException(
                $"block {physicalBlockId}: this pool's residency runs the host-read posture — its host arena already is the " +
                "copy a host read uses, so it has no device store to read a block back from.");
        }

        _deviceMirror.ReadBlockToHost(physicalBlockId, output);
    }

    /// <summary>Copies every dirty device-backed block to its device copy and clears the pending set. A host-only pool, or a mirror with nothing pending, does nothing.</summary>
    public void FlushDeviceMirror()
    {
        ThrowIfDisposed();
        _deviceMirror?.Flush();
    }

    // Records that one block's host bytes changed so its next crossing carries them. Reads never call this: only write
    // paths do — a fresh allocation about to be filled, a copy-on-write destination, and a quantizing K/V write.
    internal void MarkWritten(int physicalBlockId) => MarkDeviceDirty(physicalBlockId);

    private void MarkDeviceDirty(int physicalBlockId)
    {
        if (_deviceMirror is not null && (uint)physicalBlockId < (uint)_deviceBackedBlocks)
        {
            _deviceMirror.MarkDirty(physicalBlockId);
        }
    }

    /// <summary>Opens the sequence type this pool's storage is addressed by: the typed sequence whose reads and writes address this pool's groups.</summary>
    public IKvSequence NewSequence() => new SequenceKv<TK, TV>(this);

    /// <summary>Takes the next free block and returns its physical id with refcount 1.</summary>
    /// <exception cref="KvFullException">The pool has no reachable blocks left — physically empty, or usage already at whichever ceiling governs: a shrunk soft cap, or an admission window opened by the admitting scheduler (see <see cref="BeginAdmissionWindow"/>).</exception>
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
        MarkDeviceDirty(blockId);
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

        FlushDeviceMirror(); // the last crossing happens while the arena is still alive: a disposed mirror must not be left holding stale bytes
        _deviceMirror?.Dispose();
        _backend.FreeAligned(_arena);
        _disposed = true;
    }

    private Span<byte> SideSpan(int blockId, int layer, bool isV, int sideBytes)
    {
        long offset = (long)blockId * BytesPerBlock + (long)layer * (_kSideBytesPerLayer + _vSideBytesPerLayer) + (isV ? _kSideBytesPerLayer : 0);
        return new Span<byte>(_base + offset, sideBytes);
    }

    private Span<byte> MutableSideSpan(int blockId, int layer, bool isV, int sideBytes)
    {
        MarkDeviceDirty(blockId); // a whole-side write lands host bytes the device copy must be brought in line with
        return SideSpan(blockId, layer, isV, sideBytes);
    }

    private Span<byte> TokenSpan(int blockId, int layer, bool isV, int tokenSlot, int tokenBytes)
    {
        int groupsPerToken = isV ? _vGroupsPerToken : _kGroupsPerToken;
        long offset = (long)blockId * BytesPerBlock
                    + (long)layer * (_kSideBytesPerLayer + _vSideBytesPerLayer)
                    + (isV ? _kSideBytesPerLayer : 0)
                    + (long)tokenSlot * groupsPerToken * (isV ? TV.BlockBytes : TK.BlockBytes);
        return new Span<byte>(_base + offset, tokenBytes);
    }

    private Span<byte> MutableTokenSpan(int blockId, int layer, bool isV, int tokenSlot, int tokenBytes)
    {
        MarkDeviceDirty(blockId); // a quantizing K/V write lands host bytes the device copy must be brought in line with
        return TokenSpan(blockId, layer, isV, tokenSlot, tokenBytes);
    }

    // The device-backed range each residency declares for itself over a reservation of this size: none at all host-only,
    // the whole reservation accelerator-resident, and half — at least one block — when divided. Leading ids either way.
    private static int DeclaredDeviceRange(BlockResidencyMode residency, int capacityBlocks) => residency switch
    {
        BlockResidencyMode.Vram => capacityBlocks,
        BlockResidencyMode.Split => Math.Max(1, capacityBlocks / 2),
        _ => 0,
    };

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

    // A device-step quantized execution needs both a device-backed range to execute against and a combo a CUDA quantized
    // attention kernel stands for. The quantized attention route dots a q8_0 K side against a prepared query and decodes a
    // q8_0 or q4_0 V side, so those two combos — q8_0/q8_0 and q8_0/q4_0 — are the only quantized posture a device step
    // opens here. A host-only residency has no range, and any other quantized pair (a q4_0 K side, or the dropped
    // q4_0/q4_0) has no device attention kernel in this build, so each refuses naming the model resource, tier, format,
    // and posture rather than opening a store no kernel here can walk.
    private static KvMirrorFormatCombo ValidateExecutionCombo(BlockResidencyMode residency, KvMirrorExecution execution, int deviceBackedBlocks, KvFormatPair formats)
    {
        if (execution != KvMirrorExecution.DeviceStep)
        {
            return KvMirrorFormatCombo.Float; // a host-read quantized mirror carries the pair's own bytes; no attention kernel is selected
        }

        if (residency == BlockResidencyMode.Dram || deviceBackedBlocks == 0)
        {
            throw new ArgumentException(
                $"a device-step KV execution declares blocks a GPU step reads and writes on-device, but the '{BlockResidencyModeNames.Of(residency)}' " +
                "residency opens no device-backed range to execute against — declare 'vram' or 'split' residency beside a device-step execution.",
                nameof(execution));
        }

        if (formats.K == KvSideFormat.Q8_0 && formats.V == KvSideFormat.Q8_0)
        {
            return KvMirrorFormatCombo.Q8Q8;
        }

        if (formats.K == KvSideFormat.Q8_0 && formats.V == KvSideFormat.Q4_0)
        {
            return KvMirrorFormatCombo.K8V4;
        }

        throw new ArgumentException(
            $"a device-step quantized KV execution selects a CUDA quantized attention kernel for the 'q8_0/q8_0' and 'q8_0/q4_0' " +
            $"combos, but this pool declares k='{KvSideFormatNames.Of(formats.K)}' v='{KvSideFormatNames.Of(formats.V)}' at tier " +
            $"{WeightTier.Vram} over a '{BlockResidencyModeNames.Of(residency)}' device-backed range: that pair has no device attention " +
            "kernel here, so it refuses naming the format and posture rather than walking quantized bytes with a float kernel.",
            nameof(execution));
    }

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
