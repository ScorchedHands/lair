namespace Lair.Contracts;

/// <summary>
/// The KV storage layout a device-resident store holds, as the combo a device attention kernel is selected against.
/// <see cref="Float"/> is the FP32 block layout every device-backed residency has always walked; <see cref="Q8Q8"/> and
/// <see cref="K8V4"/> are the two quantized combos a device attention kernel stands for today — a q8_0 K side dotted
/// against a prepared query, with a q8_0 or q4_0 V side decoded for weighted accumulation. The combo is resolved once at
/// pool open and carried on the store so a step selects its attention kernel from it rather than branching on a format
/// per position. A combo with no device attention kernel here (a q4_0 K side, or a q4_0/q4_0 pair) is refused at pool
/// open naming the format, never silently walked by a float kernel.
/// </summary>
public enum KvMirrorFormatCombo
{
    /// <summary>FP32 block storage: the layout every device-backed residency has always walked.</summary>
    Float,

    /// <summary>q8_0 K side and q8_0 V side: the CUDA quantized attention combo with both sides faithful.</summary>
    Q8Q8,

    /// <summary>q8_0 K side and q4_0 V side (k8v4): the CUDA quantized attention combo whose value side decodes nibbles.</summary>
    K8V4,
}

/// <summary>
/// How a device-backed KV range is executed once its bytes are resident. <see cref="HostArenaRead"/> is the posture
/// every device-backed residency in this build has always run: the device copy is residency accounting kept current on
/// flush, and attention keeps reading the caller's host arena. <see cref="DeviceStep"/> is the on-device posture: the
/// device copy is the store a GPU step reads and writes through its own kernels, and host arena reads are not the
/// served path for those blocks — a host read of a block a step wrote device-side goes through the mirror's explicit
/// read-back seam rather than through stale arena bytes.
/// </summary>
public enum KvMirrorExecution
{
    /// <summary>The device copy is residency only: the host arena stays the copy attention reads, and flushes keep the copy current.</summary>
    HostArenaRead,

    /// <summary>The device copy is the store a GPU step reads and writes on-device; the host arena is bookkeeping for those blocks, not the read path.</summary>
    DeviceStep,
}

/// <summary>
/// The geometry of a device-executed KV store, as a mirror hands it to the step that consumes it: the persistent device
/// allocation's base, the block range it covers, and the pool block layout its bytes follow — block-major, then layer,
/// then the K half and V half of each layer, then the token slot and element within each half. A step addresses a
/// logical position through this layout the same way the pool's host views address it.
/// </summary>
public readonly record struct KvDeviceStore
{
    // Explicitly declared: this struct carries field initializers on the combo and group geometry, and a struct with
    // field initializers must declare its constructor. Every required member still lands through the initializer list.
    public KvDeviceStore()
    {
    }

    /// <summary>First byte of the persistent device allocation: one whole block per device-backed block id, in pool block order.</summary>
    public required ulong DeviceBase { get; init; }

    /// <summary>First physical block id the store covers, inclusive — the id logical position 0 addresses.</summary>
    public required int FirstBlockId { get; init; }

    /// <summary>Whole blocks the store covers: positions <c>[0, BlockCount * BlockTokens)</c> address it.</summary>
    public required int BlockCount { get; init; }

    /// <summary>Tokens one block covers — the paging granularity positions resolve at.</summary>
    public required int BlockTokens { get; init; }

    /// <summary>Layers the store holds a K and V region for, in block-major order.</summary>
    public required int LayerCount { get; init; }

    /// <summary>Values per token for one of K or V: kvHeads * headDim.</summary>
    public required int KvDim { get; init; }

    /// <summary>Whole byte size of one block across every layer and both halves — the store's block stride.</summary>
    public required long BytesPerBlock { get; init; }

    /// <summary>The storage layout this store holds, as the combo a device attention kernel selects against. FP32 unless a quantized combo was resolved at pool open.</summary>
    public KvMirrorFormatCombo Combo { get; init; } = KvMirrorFormatCombo.Float;

    // Quantized-side group geometry the attention kernel addresses through. Zero for the FP32 layout; filled for a
    // quantized combo so a step resolves a value's group and byte-in-group the same way the pool's typed views do.
    /// <summary>Bytes one K-side group covers (q8_0: 34).</summary>
    public int KGroupBytes { get; init; }

    /// <summary>Bytes one V-side group covers (q8_0: 34, q4_0: 18).</summary>
    public int VGroupBytes { get; init; }

    /// <summary>Whole groups one token's K region covers.</summary>
    public int GroupsPerTokenK { get; init; }

    /// <summary>Whole groups one token's V region covers.</summary>
    public int GroupsPerTokenV { get; init; }

    /// <summary>Bytes of one layer's K side across a block's tokens — the V half's offset within a layer region.</summary>
    public long KSideBytesPerLayer { get; init; }

    /// <summary>Bytes of one layer region (K side beside V side) across a block's tokens.</summary>
    public long LayerRegionBytes { get; init; }
}

/// <summary>
/// What a device-resident KV mirror is asked to back: the caller's host arena (which stays the copy attention reads in
/// the host-read posture), the whole-byte size of one block, the physical block range whose bytes are declared
/// device-resident, and the execution posture the range runs in. The tier and resource name carry no storage meaning on
/// their own — they name what a backend without device-resident KV storage must report when it refuses the request. In
/// the device-step posture the pool geometry — block tokens, layer count, and kv-dim — is part of the request, because
/// the store a step consumes is addressed through the pool's block layout.
/// </summary>
public readonly record struct KvMirrorRequest
{
    // Explicitly declared: this struct carries an initializer on Execution, and a struct with field initializers must
    // declare its constructor. Every required member still lands through the initializer list at every call site.
    public KvMirrorRequest()
    {
    }
    /// <summary>First byte of the host arena the mirror stages from: an aligned allocation the caller owns and keeps alive for the mirror's whole lifetime.</summary>
    public required nint HostArenaBase { get; init; }

    /// <summary>Total bytes of that host arena. The device-backed block range must lie inside it.</summary>
    public required long HostArenaBytes { get; init; }

    /// <summary>Whole byte size of one KV block — the unit a flush copies per crossed block, K and V halves across every layer included.</summary>
    public required long BytesPerBlock { get; init; }

    /// <summary>First physical block id the device-backed range covers, inclusive.</summary>
    public required int FirstBlockId { get; init; }

    /// <summary>Number of consecutive physical blocks the device-backed range covers. The persistent device copy is sized to exactly this many whole blocks.</summary>
    public required int BlockCount { get; init; }

    /// <summary>The residency tier the mirrored KV is declared at. A backend that holds no storage at this tier refuses naming it.</summary>
    public required WeightTier Tier { get; init; }

    /// <summary>The resource being mirrored, named verbatim in refusal reports so a failed open says what could not be backed where.</summary>
    public required string ResourceName { get; init; }

    /// <summary>The execution posture this range runs in. Absent, the host-read posture every device-backed residency has always run.</summary>
    public KvMirrorExecution Execution { get; init; } = KvMirrorExecution.HostArenaRead;

    /// <summary>Tokens one block covers. Required by the device-step posture, which hands the store to a step that addresses it by block paging.</summary>
    public int BlockTokens { get; init; }

    /// <summary>Layers the range holds a K and V region for. Required by the device-step posture.</summary>
    public int LayerCount { get; init; }

    /// <summary>Values per token for one of K or V. Required by the device-step posture.</summary>
    public int KvDim { get; init; }

    /// <summary>The storage layout this device store holds, as the combo a device attention kernel is selected against. Absent, the FP32 layout every device-backed residency has always walked.</summary>
    public KvMirrorFormatCombo Combo { get; init; } = KvMirrorFormatCombo.Float;

    // Quantized-side group geometry, required by a quantized device-step combo and unused for the FP32 combo. Each side's
    // region is whole groups of a fixed payload+scale block: GroupsPerToken*BlockBytes per token slot, and a layer region
    // is the K side bytes beside the V side bytes. The attention kernel addresses a value through these, exactly as the
    // pool's typed token views do.
    /// <summary>Bytes one K-side group covers (q8_0: 34). Required by a quantized combo.</summary>
    public int KGroupBytes { get; init; }

    /// <summary>Bytes one V-side group covers (q8_0: 34, q4_0: 18). Required by a quantized combo.</summary>
    public int VGroupBytes { get; init; }

    /// <summary>Whole groups one token's K region covers. Required by a quantized combo.</summary>
    public int GroupsPerTokenK { get; init; }

    /// <summary>Whole groups one token's V region covers. Required by a quantized combo.</summary>
    public int GroupsPerTokenV { get; init; }

    /// <summary>Bytes of one layer's K side across a block's tokens — the V half's offset within a layer region. Required by a quantized combo.</summary>
    public long KSideBytesPerLayer { get; init; }

    /// <summary>Bytes of one layer region (K side beside V side) across a block's tokens. Required by a quantized combo.</summary>
    public long LayerRegionBytes { get; init; }

    /// <summary>Total bytes of the device-backed block range: exactly the persistent device copy's size.</summary>
    public long DeviceRangeBytes => checked(BytesPerBlock * (long)BlockCount);
}

/// <summary>
/// A backend-owned mirror of a host KV arena at the accelerator tier: a persistent device allocation sized to one whole
/// block per device-backed block id, plus the pinned staging path those bytes cross on. In the host-read posture host
/// attention never goes through this seam — it keeps reading the caller's own arena, and the mirror is what keeps the
/// device-resident copy of the declared range current as host writes land; crossing a block's bytes to device memory does
/// not mean anything on the device reads them. In the device-step posture the same allocation is instead the store a GPU
/// step consumes: <see cref="TryGetDeviceStore"/> hands it out for on-device read/write, <see cref="AccountDeviceWrite"/>
/// records what those kernels wrote without any byte crossing, and <see cref="ReadBlockToHost"/> serves an explicit host
/// observation of a block the device holds. Residency accounting — crossed blocks, crossed bytes, pending writes — reads
/// identically in both postures.
/// </summary>
public interface IKvDeviceMirror : IDisposable
{
    /// <summary>The residency tier this mirror backs.</summary>
    WeightTier Tier { get; }

    /// <summary>The execution posture this mirror runs in: host-read residency accounting, or the store a device step reads and writes.</summary>
    KvMirrorExecution Execution { get; }

    /// <summary>Bytes of persistent device-resident storage this mirror holds: one whole block per device-backed block id, allocated once and kept for the mirror's lifetime.</summary>
    long DeviceBytes { get; }

    /// <summary>Device-backed block ids whose host bytes have crossed to the device copy at least once.</summary>
    int CrossedBlocks { get; }

    /// <summary>Total bytes copied host-to-device across every flush, counting each crossing.</summary>
    long CrossedBytes { get; }

    /// <summary>Device-backed blocks written since their last crossing — dirty and awaiting the next flush.</summary>
    int PendingWrites { get; }

    /// <summary>Device-backed block ids a GPU step wrote on-device through this mirror's store. Zero in the host-read posture.</summary>
    int DeviceWrittenBlocks { get; }

    /// <summary>Flags one device-backed block dirty: its host bytes changed since its last crossing and its next flush must carry them.</summary>
    void MarkDirty(int blockId);

    /// <summary>Copies every dirty device-backed block's whole host payload to the device copy and clears the pending set. A mirror with nothing pending copies nothing.</summary>
    void Flush();

    /// <summary>
    /// Hands the persistent device allocation to a device step as a store, in the pool's block layout. A host-read
    /// mirror answers false: its copy is residency accounting, not a store any step is entitled to read.
    /// </summary>
    bool TryGetDeviceStore(out KvDeviceStore store);

    /// <summary>
    /// Records that a device step wrote this block on-device through the store. No byte crosses the seam and the host
    /// arena is not touched: the record is residency accounting for what the device now holds. A host-read mirror refuses it.
    /// </summary>
    void AccountDeviceWrite(int blockId);

    /// <summary>
    /// Copies one block's whole payload from the device store back to host memory: the explicit observation seam for a
    /// block whose current bytes the device holds. A host-read mirror refuses it — its host arena already is the copy
    /// a host read uses, and reading the device copy instead would serve bytes no host write produced.
    /// </summary>
    void ReadBlockToHost(int blockId, Span<float> output);
}
