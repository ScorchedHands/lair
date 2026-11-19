using System.Numerics;

using Lair.Contracts;

namespace Lair.Backends.Cuda;

/// <summary>
/// The on-device transformer step: persistent device buffers for one step's angle tables, staged window rows, and
/// per-layer KV arena, plus the kernel family that composes a dense step against them. A step stages one window's
/// query rows and new key/value rows across the PCIe seam once, then rotates them in place, writes the keys and values
/// into its own arena, and walks causal <c>softmax(QK^T)V</c> over that arena — scores, softmax state, and every K/V
/// read of the walk stay on the device, and only the staged window and the attention result cross. A window the arena
/// already holds needs no re-crossing: the arena keeps every position a step wrote until the step is released, which is
/// the resident-KV footing the device step reads from.
/// <para>
/// A step opened against a pool's device store — the device-step execution posture of a <c>vram</c> or <c>split</c>
/// KV residency — runs its KV write and attention walk against that store instead of a private arena: the store is the
/// pool's own block arena mirrored on the device, addressed through the pool's block layout (block-major, then layer,
/// then each layer's K and V half, then the token slot and element), so the step's KV bytes are the pool's resident
/// blocks the pool's host views would address, and the step's writes are recorded as residency against the mirror with
/// no byte crossing. The step's staging buffers and angle tables are its own either way.
/// </para>
/// <para>
/// The arena starts unwritten and the kernels never clear it: a step may only read positions its own KV write has
/// covered, so the forward order the engine already enforces (write a position's K and V before any query attends it)
/// is the contract this seam reads against, on every layer and on both the prefill window and the decode step.
/// </para>
/// </summary>
public sealed unsafe class CudaDeviceStep : IDisposable
{
    // The rotation and KV-write blocks stride over their work at this width; the attention block is one lane per head
    // dimension, so its block width is the request's head dimension and every lane covers exactly one accumulator.
    private const int StagedKernelThreads = 128;

    // Whole-byte size of one quantized group struct as the committed writers store it: a q8_0 group is a half group
    // scale beside 32 signed-byte quants (34 bytes); a q4_0 group is a half group scale beside 16 nibble bytes (18
    // bytes). The quantized attention kernel addresses a value through these sizes exactly as the pool's typed views do.
    private const int Q8GroupBytes = 34;
    private const int Q4GroupBytes = 18;

    private readonly CudaBackend _backend;
    private readonly CudaDeviceStepRequest _request;

    private readonly ulong _cos;
    private readonly ulong _sin;
    private readonly ulong _queries;
    private readonly ulong _keyStage;
    private readonly ulong _valueStage;
    private readonly ulong _arenaK;
    private readonly ulong _arenaV;

    // Pool-bound posture: the KV store this step's write and attention kernels address, in the pool's block layout.
    // Null for a private-arena step; the store's own allocation is the KV storage, so no arena allocation happens.
    // The geometry fields resolve the store's block paging and are filled by ValidateStoreBinding at open.
    private readonly IKvDeviceMirror? _kvStore;
    private KvDeviceStore _store;
    private int _blockShift;
    private int _floatsPerBlock;
    private int _layerRegion;
    private int _vHalf;

    // Quantized store binding: the combo this step's attention handle was resolved from, the handle itself, and the
    // byte-side geometry the quantized attention kernel addresses with. A float store leaves these unset and the step
    // keeps its element-side float fields; a quantized store fills them and the step reads quantized bytes only. The
    // handle is resolved once at open from the combo, so the attention walk never branches on format per position.
    private KvMirrorFormatCombo _combo;
    private IntPtr _attentionKernel;
    private int _bytesPerBlock;
    private int _layerRegionBytes;
    private int _kSideBytes;
    private int _groupsPerTokenK;
    private int _groupsPerTokenV;
    private int _kGroupBytes;
    private int _vGroupBytes;

    private readonly long _cosBytes;
    private readonly long _queryBytes;
    private readonly long _kvRowBytes;
    private readonly long _arenaBytes;

    private int _stagedRows;
    private bool _ropeLoaded;
    private bool _disposed;

    internal CudaDeviceStep(CudaBackend backend, CudaDeviceStepRequest request, IKvDeviceMirror? kvStore = null)
    {
        ArgumentNullException.ThrowIfNull(request);
        request.Validate();

        _backend = backend;
        _request = request;

        int ropeHalf = request.HeadDim / 2;
        _cosBytes = checked((long)request.MaxPositions * ropeHalf * sizeof(float));
        _queryBytes = checked((long)request.MaxWindowRows * request.HeadCount * request.HeadDim * sizeof(float));
        _kvRowBytes = checked((long)request.MaxWindowRows * request.KvHeadCount * request.HeadDim * sizeof(float));
        _arenaBytes = checked((long)request.LayerCount * request.MaxPositions * request.KvHeadCount * request.HeadDim * sizeof(float));

        _kvStore = kvStore;
        if (_kvStore is not null)
        {
            ValidateStoreBinding(_kvStore);
        }

        ulong cos = 0;
        ulong sin = 0;
        ulong queries = 0;
        ulong keyStage = 0;
        ulong valueStage = 0;
        ulong arenaK = 0;
        ulong arenaV = 0;

        backend.EnterDeviceWork();
        try
        {
            CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out cos, (nuint)_cosBytes));
            CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out sin, (nuint)_cosBytes));
            CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out queries, (nuint)_queryBytes));
            CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out keyStage, (nuint)_kvRowBytes));
            CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out valueStage, (nuint)_kvRowBytes));

            // A pool-bound step writes and reads the pool's resident store, which the mirror already allocated and
            // zeroed: a private arena would be a second copy of those blocks, so none is opened.
            if (_kvStore is null)
            {
                CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out arenaK, (nuint)_arenaBytes));
                CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out arenaV, (nuint)_arenaBytes));
            }
        }
        catch
        {
            foreach (ulong pointer in new[] { arenaV, arenaK, valueStage, keyStage, queries, sin, cos })
            {
                if (pointer != 0)
                {
                    CudaNative.cuMemFree(pointer);
                }
            }

            throw;
        }
        finally
        {
            backend.ExitDeviceWork();
        }

        _cos = cos;
        _sin = sin;
        _queries = queries;
        _keyStage = keyStage;
        _valueStage = valueStage;
        _arenaK = arenaK;
        _arenaV = arenaV;

        backend.RegisterDeviceStep(this);
    }

    // A store bound to this step must be exactly the storage this step's kernels walk: the pool geometry the store
    // carries has to account for the request's positions, layers, and kv region, and the block paging the kernels
    // address with a shift has to be exact. A mismatch refuses naming both sides rather than walking a layout the
    // store does not hold.
    private void ValidateStoreBinding(IKvDeviceMirror kvStore)
    {
        if (!kvStore.TryGetDeviceStore(out KvDeviceStore store))
        {
            throw new ArgumentException(
                "the KV store bound to this device step is not a device-step store: a host-read mirror's copy is residency " +
                "accounting, not storage a step's kernels read or write.",
                nameof(kvStore));
        }

        if (store.BlockTokens <= 0 || !BitOperations.IsPow2((uint)store.BlockTokens))
        {
            throw new ArgumentException(
                $"the KV store's block size of {store.BlockTokens} tokens is not a power of two: a device step resolves a " +
                "logical position to its block by a shift, so only a power-of-two block size addresses it.",
                nameof(kvStore));
        }

        long positions = (long)store.BlockCount * store.BlockTokens;
        if (positions != _request.MaxPositions)
        {
            throw new ArgumentException(
                $"the KV store covers {store.BlockCount} blocks of {store.BlockTokens} tokens = {positions} positions, but this " +
                $"step's request declares MaxPositions {_request.MaxPositions}.",
                nameof(kvStore));
        }

        if (store.LayerCount != _request.LayerCount)
        {
            throw new ArgumentException(
                $"the KV store holds {store.LayerCount} layers, but this step's request declares LayerCount {_request.LayerCount}.",
                nameof(kvStore));
        }

        if (store.KvDim != _request.KvHeadCount * _request.HeadDim)
        {
            throw new ArgumentException(
                $"the KV store's kv region is {store.KvDim} values per token, but this step's request is " +
                $"{_request.KvHeadCount} kv-heads x {_request.HeadDim} dimensions = {_request.KvHeadCount * _request.HeadDim}.",
                nameof(kvStore));
        }

        _blockShift = BitOperations.Log2((uint)store.BlockTokens);

        switch (store.Combo)
        {
            case KvMirrorFormatCombo.Float:
            {
                long floatBlockBytes = (long)store.LayerCount * 2L * store.BlockTokens * store.KvDim * sizeof(float);
                if (store.BytesPerBlock != floatBlockBytes)
                {
                    throw new ArgumentException(
                        $"the KV store's block is {store.BytesPerBlock} bytes, but its geometry is {store.LayerCount} layers x 2 halves x " +
                        $"{store.BlockTokens} tokens x {store.KvDim} values as FP32 = {floatBlockBytes} bytes.",
                        nameof(kvStore));
                }

                _store = store;
                _combo = KvMirrorFormatCombo.Float;
                _attentionKernel = _backend.AttentionPagedKernel;
                _layerRegion = checked(2 * store.BlockTokens * store.KvDim);
                _floatsPerBlock = checked(store.LayerCount * _layerRegion);
                _vHalf = checked(store.BlockTokens * store.KvDim);
                break;
            }

            case KvMirrorFormatCombo.Q8Q8:
            case KvMirrorFormatCombo.K8V4:
            {
                // A quantized store's block is whole groups of the combo's block structs: a q8_0 group is 34 bytes
                // (half scale beside 32 signed-byte quants), a q4_0 group is 18 bytes (half scale beside 16 nibble
                // bytes). Validate the geometry the mirror carried against the request's kv region and the combo's own
                // group sizes, then resolve the attention handle the combo stands for. A store whose declared byte
                // layout disagrees with its own geometry refuses naming both sides.
                int kGroupBytes = store.KGroupBytes;
                int vGroupBytes = store.VGroupBytes;
                bool vIsQ4 = store.Combo == KvMirrorFormatCombo.K8V4;

                if (kGroupBytes != Q8GroupBytes || vGroupBytes != (vIsQ4 ? Q4GroupBytes : Q8GroupBytes))
                {
                    throw new ArgumentException(
                        $"the KV store declares combo {store.Combo} with group sizes k={kGroupBytes} v={vGroupBytes} bytes, but " +
                        $"{store.Combo} stores a q8_0 K side (34 bytes per group) beside a " +
                        $"{(vIsQ4 ? "q4_0" : "q8_0")} V side ({(vIsQ4 ? 18 : 34)} bytes per group).",
                        nameof(kvStore));
                }

                if (store.KvDim % 32 != 0)
                {
                    throw new ArgumentException(
                        $"the KV store's kv region of {store.KvDim} values per token is not a whole number of 32-value groups: " +
                        "a quantized attention kernel resolves a value to its group by a shift, so only a multiple of 32 addresses it.",
                        nameof(kvStore));
                }

                int groupsPerToken = store.KvDim / 32;
                long kSideBytes = (long)groupsPerToken * store.BlockTokens * Q8GroupBytes;
                long vSideBytes = (long)groupsPerToken * store.BlockTokens * vGroupBytes;
                long layerRegionBytes = kSideBytes + vSideBytes;
                long quantBlockBytes = (long)store.LayerCount * layerRegionBytes;

                if (store.GroupsPerTokenK != groupsPerToken || store.GroupsPerTokenV != groupsPerToken)
                {
                    throw new ArgumentException(
                        $"the KV store declares {store.GroupsPerTokenK}/{store.GroupsPerTokenV} groups per token for K/V, but its " +
                        $"{store.KvDim}-value kv region is {groupsPerToken} groups per token.",
                        nameof(kvStore));
                }

                if (store.KSideBytesPerLayer != kSideBytes || store.LayerRegionBytes != layerRegionBytes || store.BytesPerBlock != quantBlockBytes)
                {
                    throw new ArgumentException(
                        $"the KV store declares layer-K-side={store.KSideBytesPerLayer} layer-region={store.LayerRegionBytes} " +
                        $"block={store.BytesPerBlock} bytes, but its geometry is {store.LayerCount} layers x " +
                        $"({groupsPerToken} groups x {store.BlockTokens} tokens x {Q8GroupBytes} K) + " +
                        $"({groupsPerToken} groups x {store.BlockTokens} tokens x {vGroupBytes} V) = {quantBlockBytes} bytes.",
                        nameof(kvStore));
                }

                _store = store;
                _combo = store.Combo;
                _attentionKernel = store.Combo == KvMirrorFormatCombo.Q8Q8
                    ? _backend.AttentionQ8Q8Kernel
                    : _backend.AttentionK8V4Kernel;
                _bytesPerBlock = checked((int)quantBlockBytes);
                _layerRegionBytes = checked((int)layerRegionBytes);
                _kSideBytes = checked((int)kSideBytes);
                _groupsPerTokenK = groupsPerToken;
                _groupsPerTokenV = groupsPerToken;
                _kGroupBytes = Q8GroupBytes;
                _vGroupBytes = vGroupBytes;
                break;
            }

            default:
                throw new ArgumentException(
                    $"the KV store declares combo {store.Combo}, which has no device attention kernel resolved here: only the " +
                    "FP32 layout and the q8_0/q8_0 and q8_0/q4_0 quantized combos stand for a device step.",
                    nameof(kvStore));
        }
    }

    /// <summary>The pairing this step's rotation kernel applies to staged query and key rows.</summary>
    public RopePairing Pairing => _request.Pairing;

    /// <summary>Positions this step's KV arena holds per layer.</summary>
    public int MaxPositions => _request.MaxPositions;

    /// <summary>Layers this step's KV arena holds a K and V region for.</summary>
    public int LayerCount => _request.LayerCount;

    /// <summary>Query heads one staged row carries.</summary>
    public int HeadCount => _request.HeadCount;

    /// <summary>Key/value heads one staged row carries.</summary>
    public int KvHeadCount => _request.KvHeadCount;

    /// <summary>Dimensions of one head region, which is also this step's attention lane count.</summary>
    public int HeadDim => _request.HeadDim;

    /// <summary>Rows one staged window may hold.</summary>
    public int MaxWindowRows => _request.MaxWindowRows;

    /// <summary>Whether the angle tables this step's rotation reads are resident on the device.</summary>
    public bool RopeTablesLoaded => _ropeLoaded;

    /// <summary>Rows the staged window currently holds; zero after <see cref="ClearWindow"/> or before the first stage.</summary>
    public int StagedRows => _stagedRows;

    /// <summary>Whether this step's KV write and attention walk run against a pool's device store rather than a private arena.</summary>
    public bool KvStoreBacked => _kvStore is not null;

    /// <summary>Device bytes this step's persistent buffers hold: angle tables and staged window rows, plus the KV storage it walks — its own arenas for a private-arena step, the pool store it is bound to for a store-bound one.</summary>
    public long DeviceBytes => checked(2 * _cosBytes + 2 * _queryBytes + 2 * _kvRowBytes)
        + (_kvStore is null ? 2 * _arenaBytes : _store.BlockCount * _store.BytesPerBlock);

    /// <summary>
    /// Loads the position-major rotary angle tables this step rotates with: one <c>headDim / 2</c> pair entry per
    /// position the arena holds, in the layout the rotation kernel indexes. The tables cross once and stay resident
    /// for the step's lifetime, so a step's rotations read them on the device without another crossing.
    /// </summary>
    public void LoadRopeTables(ReadOnlySpan<float> cos, ReadOnlySpan<float> sin)
    {
        ThrowIfDisposed();

        long entries = (long)_request.MaxPositions * (_request.HeadDim / 2);
        if (cos.Length < entries || sin.Length < entries)
        {
            throw new ArgumentException(
                $"rotary tables for this step need {entries} entries each ({_request.MaxPositions} positions x " +
                $"{_request.HeadDim / 2} pairs), got {cos.Length} and {sin.Length}.",
                nameof(cos));
        }

        _backend.EnterDeviceWork();
        try
        {
            Upload(_cos, cos);
            Upload(_sin, sin);
            _ropeLoaded = true;
        }
        finally
        {
            _backend.ExitDeviceWork();
        }
    }

    /// <summary>
    /// Stages one window across the seam: <paramref name="qRows"/> holds <c>rowCount</c> rows of
    /// <c>HeadCount * HeadDim</c> query elements, <paramref name="kRows"/> and <paramref name="vRows"/> hold the same
    /// rows' <c>KvHeadCount * HeadDim</c> key and value elements, each row-major and head-major — the layout the host
    /// forward produces. The rows land in this step's staging buffers, where the rotation and the KV write consume
    /// them; nothing else crosses.
    /// </summary>
    public void StageWindow(ReadOnlySpan<float> qRows, ReadOnlySpan<float> kRows, ReadOnlySpan<float> vRows)
    {
        ThrowIfDisposed();

        int queryRegion = _request.HeadCount * _request.HeadDim;
        int kvRegion = _request.KvHeadCount * _request.HeadDim;

        if (qRows.Length == 0 || qRows.Length % queryRegion != 0)
        {
            throw new ArgumentException(
                $"staged query rows must cover whole rows of HeadCount x HeadDim ({queryRegion} elements), got {qRows.Length}.",
                nameof(qRows));
        }

        int rows = qRows.Length / queryRegion;
        if (rows > _request.MaxWindowRows)
        {
            throw new ArgumentException(
                $"staged window of {rows} rows exceeds this step's MaxWindowRows {_request.MaxWindowRows}.",
                nameof(qRows));
        }

        if (kRows.Length != (long)rows * kvRegion || vRows.Length != (long)rows * kvRegion)
        {
            throw new ArgumentException(
                $"staged key and value rows must each cover {rows} rows of KvHeadCount x HeadDim ({kvRegion} elements), " +
                $"got {kRows.Length} and {vRows.Length}.",
                nameof(kRows));
        }

        _backend.EnterDeviceWork();
        try
        {
            Upload(_queries, qRows);
            _backend.AccountStepTransfer((long)qRows.Length << 2);
            Upload(_keyStage, kRows);
            _backend.AccountStepTransfer((long)kRows.Length << 2);
            Upload(_valueStage, vRows);
            _backend.AccountStepTransfer((long)vRows.Length << 2);
            _stagedRows = rows;
        }
        finally
        {
            _backend.ExitDeviceWork();
        }
    }

    /// <summary>
    /// Reads the staged window's query rows back to the host: the observation path for the bytes the staging buffers
    /// hold, so a caller can confirm what a rotation wrote before the next window replaces it. Reading staged bytes is
    /// not a compute call's traffic, so it stays out of the runtime transfer tallies the way a mirror's crossing does.
    /// </summary>
    public void ReadStagedQueries(Span<float> output) => ReadStaged(_queries, output, _request.HeadCount * _request.HeadDim, nameof(output));

    /// <summary>Reads the staged window's key rows back to the host, the same observation path over the key staging buffer.</summary>
    public void ReadStagedKeys(Span<float> output) => ReadStaged(_keyStage, output, _request.KvHeadCount * _request.HeadDim, nameof(output));

    /// <summary>Reads the staged window's value rows back to the host, the same observation path over the value staging buffer.</summary>
    public void ReadStagedValues(Span<float> output) => ReadStaged(_valueStage, output, _request.KvHeadCount * _request.HeadDim, nameof(output));

    private void ReadStaged(ulong stagingBase, Span<float> output, int region, string paramName)
    {
        ThrowIfDisposed();

        int rows = RequireStagedRows();
        if (output.Length < (long)rows * region)
        {
            throw new ArgumentException(
                $"staged rows must be read back into {rows} rows of {region} elements, got {output.Length}.",
                paramName);
        }

        long bytes = (long)rows * region * sizeof(float);
        _backend.EnterDeviceWork();
        try
        {
            fixed (float* p = output)
            {
                CudaException.ThrowIf("cuMemcpyDtoH_v2", CudaNative.cuMemcpyDtoH((byte*)p, stagingBase, (nuint)bytes));
            }
        }
        finally
        {
            _backend.ExitDeviceWork();
        }
    }

    /// <summary>
    /// Clears the staged window: the staging buffers keep their bytes, but no row count stands, so a rotation, a KV
    /// write, or an attention walk refuses until the next <see cref="StageWindow"/> rather than re-consuming rows the
    /// caller has already retired.
    /// </summary>
    public void ClearWindow()
    {
        ThrowIfDisposed();
        _stagedRows = 0;
    }

    /// <summary>
    /// Rotates the staged window's query rows and key rows in place at positions
    /// <c>[positionBase, positionBase + StagedRows)</c>, with the pairing this step was opened for. Both rotations
    /// read the resident angle tables and write back into the staging buffers; the values themselves cross no seam.
    /// </summary>
    public void ApplyRope(int positionBase)
    {
        ThrowIfDisposed();

        int rows = RequireStagedRows();
        ValidatePosition(positionBase, rows);

        if (!_ropeLoaded)
        {
            throw new InvalidOperationException(
                "this device step has no resident rotary tables: call LoadRopeTables before ApplyRope.");
        }

        IntPtr function = _request.Pairing == RopePairing.AdjacentPairs ? _backend.RopePairsKernel : _backend.RopeHalfKernel;
        int half = _request.HeadDim / 2;

        _backend.EnterDeviceWork();
        try
        {
            bool timed = _backend.BeginStepKernelTiming();

            LaunchRope(function, _queries, rows, _request.HeadCount, positionBase, half);
            LaunchRope(function, _keyStage, rows, _request.KvHeadCount, positionBase, half);

            if (timed)
            {
                _backend.StampStepKernelEnd();
            }

            CudaException.ThrowIf("cuCtxSynchronize", CudaNative.cuCtxSynchronize());
            if (timed)
            {
                _backend.FoldStepKernelElapsed();
            }
        }
        finally
        {
            _backend.ExitDeviceWork();
        }
    }

    /// <summary>
    /// Writes the staged window's key and value rows into this step's KV storage at <c>(layer, position)</c> for
    /// positions <c>[positionBase, positionBase + StagedRows)</c>. The move runs on the device from the staged rows
    /// into the slots the attention walk reads back, so a position written here is resident for every later
    /// window and every decode step this step serves. For a private-arena step the slots are the arena's own; for a
    /// store-bound step they are the pool's resident blocks at the pool's block-layout offsets, and the write is
    /// recorded against the mirror as residency the device now holds — no byte crosses the seam for it.
    /// </summary>
    public void WriteKv(int layer, int positionBase)
    {
        ThrowIfDisposed();

        int rows = RequireStagedRows();
        ValidatePosition(positionBase, rows);
        ValidateLayer(layer);

        int rowElements = _request.KvHeadCount * _request.HeadDim;
        ulong srcK = _keyStage;
        ulong srcV = _valueStage;
        uint rowElementsU = (uint)rowElements;
        uint layerU = (uint)layer;
        uint positionBaseU = (uint)positionBase;

        _backend.EnterDeviceWork();
        try
        {
            if (_kvStore is null)
            {
                ulong dstK = _arenaK;
                ulong dstV = _arenaV;
                uint maxPositionsU = (uint)_request.MaxPositions;

                void** kernelParams = stackalloc void*[8];
                kernelParams[0] = &dstK;
                kernelParams[1] = &dstV;
                kernelParams[2] = &srcK;
                kernelParams[3] = &srcV;
                kernelParams[4] = &rowElementsU;
                kernelParams[5] = &maxPositionsU;
                kernelParams[6] = &layerU;
                kernelParams[7] = &positionBaseU;

                LaunchStepKernel(_backend.KvStoreKernel, kernelParams, (uint)rows, 1, StagedKernelThreads);
            }
            else if (_combo != KvMirrorFormatCombo.Float)
            {
                // A quantized store's bytes are the quantized groups the pool's host writers produced and the mirror
                // flushed — a half group scale beside signed-byte quants or nibbles. This step has no device-side
                // quantizer, so it never writes a quantized group from staged FP32 rows: writing KV for a quantized
                // combo is the host writer's job, and the step only reads the resident quantized bytes the attention
                // kernel decodes. Refuse naming the posture rather than storing FP32 rows where quantized groups belong.
                throw new InvalidOperationException(
                    $"this step is bound to a {_combo} quantized KV store, which it reads but does not write: quantized KV is " +
                    "written host-side and flushed to the store, so WriteKv refuses for a quantized combo rather than " +
                    "staging FP32 key/value rows as quantized groups.");
            }
            else
            {
                ulong store = _store.DeviceBase;
                uint blockShiftU = (uint)_blockShift;
                uint floatsPerBlockU = (uint)_floatsPerBlock;
                uint layerRegionU = (uint)_layerRegion;
                uint vHalfU = (uint)_vHalf;
                uint firstBlockU = (uint)_store.FirstBlockId;

                void** kernelParams = stackalloc void*[11];
                kernelParams[0] = &store;
                kernelParams[1] = &srcK;
                kernelParams[2] = &srcV;
                kernelParams[3] = &rowElementsU;
                kernelParams[4] = &blockShiftU;
                kernelParams[5] = &floatsPerBlockU;
                kernelParams[6] = &layerRegionU;
                kernelParams[7] = &vHalfU;
                kernelParams[8] = &firstBlockU;
                kernelParams[9] = &layerU;
                kernelParams[10] = &positionBaseU;

                LaunchStepKernel(_backend.KvStorePagedKernel, kernelParams, (uint)rows, 1, StagedKernelThreads);

                // The kernel wrote these positions' blocks on-device; record each covered block as residency the
                // device holds. This is accounting for what the store now contains, not a crossing: no byte moved.
                for (int pos = positionBase; pos < positionBase + rows; pos++)
                {
                    _kvStore.AccountDeviceWrite(_store.FirstBlockId + (pos >> _blockShift));
                }
            }
        }
        finally
        {
            _backend.ExitDeviceWork();
        }
    }

    /// <summary>
    /// Causal <c>softmax(QK^T)V</c> over this step's resident KV storage for the staged window: row <c>r</c> attends
    /// positions <c>[max(0, positionBase + r - window + 1), positionBase + r]</c> under a positive window, and every
    /// position up to its own under a non-positive one — the same causal selection the host attention resolves,
    /// narrowed the same way before any score is seen. Each query head reads the kv-head its grouped-query mapping
    /// assigns it, scores against that head's resident K, and accumulates its weighted V in the host kernel's
    /// ascending element order, online-softmax rescaling on each new max and normalizing by the reciprocal weight sum
    /// at the end. Scores, softmax state, and every K/V read stay on the device; only <paramref name="output"/>
    /// crosses back. The window's rows must already have their K and V written in this storage at these positions. A
    /// store-bound step resolves each position through the pool's block layout — block, slot, layer region, half —
    /// exactly as the pool's own host views address the same bytes.
    /// </summary>
    public void Attend(int layer, int positionBase, float attnScale, float softcap, int window, Span<float> output)
    {
        ThrowIfDisposed();

        int rows = RequireStagedRows();
        ValidatePosition(positionBase, rows);
        ValidateLayer(layer);

        int region = _request.HeadCount * _request.HeadDim;
        if (output.Length < (long)rows * region)
        {
            throw new ArgumentException(
                $"attention output must cover {rows} rows of HeadCount x HeadDim ({region} elements each), got {output.Length}.",
                nameof(output));
        }

        long outBytes = checked((long)rows * region * sizeof(float));
        ulong outDevice = 0;

        _backend.EnterDeviceWork();
        try
        {
            CudaException.ThrowIf("cuMemAlloc_v2", CudaNative.cuMemAlloc(out outDevice, (nuint)outBytes));

            ulong q = _queries;
            uint headsU = (uint)_request.HeadCount;
            uint kvHeadsU = (uint)_request.KvHeadCount;
            uint headDimU = (uint)_request.HeadDim;
            uint layerU = (uint)layer;
            uint positionBaseU = (uint)positionBase;
            uint windowU = (uint)(window > 0 ? window : 0);
            uint groupShiftU = (uint)BitOperations.Log2((uint)_request.HeadsPerKvHead);
            uint scaleBits = (uint)BitConverter.SingleToInt32Bits(attnScale);
            uint capBits = (uint)BitConverter.SingleToInt32Bits(softcap);

            uint gridY = headsU;

            if (_kvStore is null)
            {
                ulong arenaK = _arenaK;
                ulong arenaV = _arenaV;
                uint maxPositionsU = (uint)_request.MaxPositions;

                void** kernelParams = stackalloc void*[14];
                kernelParams[0] = &q;
                kernelParams[1] = &arenaK;
                kernelParams[2] = &arenaV;
                kernelParams[3] = &outDevice;
                kernelParams[4] = &headsU;
                kernelParams[5] = &kvHeadsU;
                kernelParams[6] = &headDimU;
                kernelParams[7] = &maxPositionsU;
                kernelParams[8] = &layerU;
                kernelParams[9] = &positionBaseU;
                kernelParams[10] = &windowU;
                kernelParams[11] = &groupShiftU;
                kernelParams[12] = &scaleBits;
                kernelParams[13] = &capBits;

                LaunchStepKernel(_backend.AttentionKernel, kernelParams, (uint)rows, gridY, headDimU);
            }
            else if (_combo == KvMirrorFormatCombo.Float)
            {
                ulong store = _store.DeviceBase;
                uint blockShiftU = (uint)_blockShift;
                uint floatsPerBlockU = (uint)_floatsPerBlock;
                uint layerRegionU = (uint)_layerRegion;
                uint vHalfU = (uint)_vHalf;
                uint firstBlockU = (uint)_store.FirstBlockId;

                void** kernelParams = stackalloc void*[17];
                kernelParams[0] = &q;
                kernelParams[1] = &store;
                kernelParams[2] = &outDevice;
                kernelParams[3] = &headsU;
                kernelParams[4] = &kvHeadsU;
                kernelParams[5] = &headDimU;
                kernelParams[6] = &blockShiftU;
                kernelParams[7] = &floatsPerBlockU;
                kernelParams[8] = &layerRegionU;
                kernelParams[9] = &vHalfU;
                kernelParams[10] = &firstBlockU;
                kernelParams[11] = &layerU;
                kernelParams[12] = &positionBaseU;
                kernelParams[13] = &windowU;
                kernelParams[14] = &groupShiftU;
                kernelParams[15] = &scaleBits;
                kernelParams[16] = &capBits;

                LaunchStepKernel(_backend.AttentionPagedKernel, kernelParams, (uint)rows, gridY, headDimU);
            }
            else
            {
                // Quantized store: the attention handle was resolved from the combo at open, so this walk never picks a
                // kernel from the format here — it launches the handle the combo selected, addressing the store's
                // quantized groups through the byte geometry the binding validated. The kernel decodes each K value as
                // quant * groupScale and each V value under its own group scale, in the same ascending order the CPU
                // quantized attention is measured against.
                ulong store = _store.DeviceBase;
                uint blockShiftU = (uint)_blockShift;
                uint bytesPerBlockU = (uint)_bytesPerBlock;
                uint layerRegionBytesU = (uint)_layerRegionBytes;
                uint kSideBytesU = (uint)_kSideBytes;
                uint gpkU = (uint)_groupsPerTokenK;
                uint gpvU = (uint)_groupsPerTokenV;
                uint kgbU = (uint)_kGroupBytes;
                uint vgbU = (uint)_vGroupBytes;
                uint firstBlockU = (uint)_store.FirstBlockId;

                void** kernelParams = stackalloc void*[21];
                kernelParams[0] = &q;
                kernelParams[1] = &store;
                kernelParams[2] = &outDevice;
                kernelParams[3] = &headsU;
                kernelParams[4] = &kvHeadsU;
                kernelParams[5] = &headDimU;
                kernelParams[6] = &blockShiftU;
                kernelParams[7] = &bytesPerBlockU;
                kernelParams[8] = &layerRegionBytesU;
                kernelParams[9] = &kSideBytesU;
                kernelParams[10] = &gpkU;
                kernelParams[11] = &gpvU;
                kernelParams[12] = &kgbU;
                kernelParams[13] = &vgbU;
                kernelParams[14] = &firstBlockU;
                kernelParams[15] = &layerU;
                kernelParams[16] = &positionBaseU;
                kernelParams[17] = &windowU;
                kernelParams[18] = &groupShiftU;
                kernelParams[19] = &scaleBits;
                kernelParams[20] = &capBits;

                LaunchStepKernel(_attentionKernel, kernelParams, (uint)rows, gridY, headDimU);
            }

            fixed (float* pOut = output)
            {
                CudaException.ThrowIf("cuMemcpyDtoH_v2", CudaNative.cuMemcpyDtoH((byte*)pOut, outDevice, (nuint)outBytes));
                _backend.AccountStepTransfer(outBytes);
            }
        }
        finally
        {
            if (outDevice != 0)
            {
                CudaNative.cuMemFree(outDevice);
            }

            _backend.ExitDeviceWork();
        }
    }

    /// <summary>
    /// The composed dense step for one window at one layer: stage the window's query rows and new key/value rows,
    /// rotate the queries and keys in place, write the keys and values into this step's arena at these positions, and
    /// walk causal attention over the arena into <paramref name="output"/>. Between the stage and the result nothing
    /// crosses the seam — no score row, no softmax state, no K/V read — which is the composition the device step
    /// exists to get. A window the arena already holds stages no K/V at all: stage the queries alone with
    /// <see cref="StageQueries"/> and the walk reads what earlier steps wrote.
    /// </summary>
    public void RunStep(
        int layer,
        int positionBase,
        ReadOnlySpan<float> qRows,
        ReadOnlySpan<float> kRows,
        ReadOnlySpan<float> vRows,
        float attnScale,
        float softcap,
        int window,
        Span<float> output)
    {
        ThrowIfDisposed();

        _backend.EnterDeviceWork();
        try
        {
            StageWindow(qRows, kRows, vRows);
            ApplyRope(positionBase);
            WriteKv(layer, positionBase);
            Attend(layer, positionBase, attnScale, softcap, window, output);
        }
        finally
        {
            _backend.ExitDeviceWork();
        }
    }

    /// <summary>
    /// Stages only the window's query rows, leaving the staged row count in place for a rotation and an attention walk
    /// over KV the arena already holds: the decode-step footing, where a position's key and value were written by an
    /// earlier step and only the new queries need to reach the device. Key and value staging is untouched by this
    /// call, so a caller may write KV for these rows through <see cref="StageWindow"/> and <see cref="WriteKv"/>
    /// first, or leave the arena's earlier contents in place.
    /// </summary>
    public void StageQueries(ReadOnlySpan<float> qRows)
    {
        ThrowIfDisposed();

        int queryRegion = _request.HeadCount * _request.HeadDim;
        if (qRows.Length == 0 || qRows.Length % queryRegion != 0)
        {
            throw new ArgumentException(
                $"staged query rows must cover whole rows of HeadCount x HeadDim ({queryRegion} elements), got {qRows.Length}.",
                nameof(qRows));
        }

        int rows = qRows.Length / queryRegion;
        if (rows > _request.MaxWindowRows)
        {
            throw new ArgumentException(
                $"staged window of {rows} rows exceeds this step's MaxWindowRows {_request.MaxWindowRows}.",
                nameof(qRows));
        }

        _backend.EnterDeviceWork();
        try
        {
            Upload(_queries, qRows);
            _backend.AccountStepTransfer((long)qRows.Length << 2);
            _stagedRows = rows;
        }
        finally
        {
            _backend.ExitDeviceWork();
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        _backend.EnterDeviceWork();
        try
        {
            // A store-bound step owns no KV arenas — the mirror owns the store it wrote — so only its staging and
            // table allocations come back here. The store itself is released with the pool's mirror.
            foreach (ulong pointer in _kvStore is null
                ? new[] { _arenaV, _arenaK, _valueStage, _keyStage, _queries, _sin, _cos }
                : new[] { _valueStage, _keyStage, _queries, _sin, _cos })
            {
                CudaNative.cuMemFree(pointer);
            }
        }
        finally
        {
            _backend.ExitDeviceWork();
        }

        _backend.UnregisterDeviceStep(this);
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, typeof(CudaDeviceStep));
        ObjectDisposedException.ThrowIf(_backend.IsDisposed, typeof(CudaDeviceStep));
    }

    private int RequireStagedRows()
    {
        if (_stagedRows == 0)
        {
            throw new InvalidOperationException(
                "this device step has no staged window rows: call StageWindow or StageQueries before this kernel.");
        }

        return _stagedRows;
    }

    private void ValidatePosition(int positionBase, int rows)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(positionBase);
        if (positionBase + rows > _request.MaxPositions)
        {
            throw new ArgumentOutOfRangeException(
                nameof(positionBase),
                $"positions [{positionBase}..{positionBase + rows - 1}] lie outside this step's MaxPositions {_request.MaxPositions}.");
        }
    }

    private void ValidateLayer(int layer)
    {
        if ((uint)layer >= (uint)_request.LayerCount)
        {
            throw new ArgumentOutOfRangeException(
                nameof(layer),
                $"layer {layer} lies outside this step's {LayerCount} arena layers.");
        }
    }

    // Launches one step kernel on the backend's gate with the timing pair bracketing exactly this launch, then
    // synchronizes the context and folds the elapsed answer into the kernel-busy tally — the same device-measured
    // region every other call site on this seam brackets, whichever kernel and grid the branch built.
    private void LaunchStepKernel(IntPtr kernel, void** kernelParams, uint gridX, uint gridY, uint blockX)
    {
        bool timed = _backend.BeginStepKernelTiming();
        CudaException.ThrowIf(
            "cuLaunchKernel",
            CudaNative.cuLaunchKernel(kernel, gridX, gridY, 1, blockX, 1, 1, 0, IntPtr.Zero, kernelParams, null));
        if (timed)
        {
            _backend.StampStepKernelEnd();
        }

        CudaException.ThrowIf("cuCtxSynchronize", CudaNative.cuCtxSynchronize());
        if (timed)
        {
            _backend.FoldStepKernelElapsed();
        }
    }

    private void LaunchRope(IntPtr function, ulong rowsBase, int rowCount, int regions, int positionBase, int half)
    {
        ulong cos = _cos;
        ulong sin = _sin;
        uint regionsU = (uint)regions;
        uint headDimU = (uint)_request.HeadDim;
        uint halfU = (uint)half;
        uint positionBaseU = (uint)positionBase;

        void** kernelParams = stackalloc void*[7];
        kernelParams[0] = &rowsBase;
        kernelParams[1] = &cos;
        kernelParams[2] = &sin;
        kernelParams[3] = &regionsU;
        kernelParams[4] = &headDimU;
        kernelParams[5] = &halfU;
        kernelParams[6] = &positionBaseU;

        CudaException.ThrowIf(
            "cuLaunchKernel",
            CudaNative.cuLaunchKernel(function, (uint)rowCount, (uint)regions, 1, StagedKernelThreads, 1, 1, 0, IntPtr.Zero, kernelParams, null));
    }

    private static void Upload(ulong devicePointer, ReadOnlySpan<float> values)
    {
        fixed (float* p = values)
        {
            CudaException.ThrowIf("cuMemcpyHtoD_v2", CudaNative.cuMemcpyHtoD(devicePointer, (byte*)p, (nuint)(values.Length * sizeof(float))));
        }
    }
}
