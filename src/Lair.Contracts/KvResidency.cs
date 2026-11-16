namespace Lair.Contracts;

/// <summary>
/// What a device-resident KV mirror is asked to back: the caller's host arena (which stays the copy attention reads),
/// the whole-byte size of one block, and the physical block range whose bytes are declared device-resident. The tier and
/// resource name carry no storage meaning on their own — they name what a backend without device-resident KV storage must
/// report when it refuses the request.
/// </summary>
public readonly record struct KvMirrorRequest
{
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

    /// <summary>Total bytes of the device-backed block range: exactly the persistent device copy's size.</summary>
    public long DeviceRangeBytes => checked(BytesPerBlock * (long)BlockCount);
}

/// <summary>
/// A backend-owned mirror of a host KV arena at the accelerator tier: a persistent device allocation sized to one whole
/// block per device-backed block id, plus the pinned staging path those bytes cross on. Host attention reads never go
/// through this seam — they keep reading the caller's own arena; the mirror is what keeps the device-resident copy of the
/// declared range current as host writes land, and it reports its residency accounting so a test can prove which blocks
/// crossed and how many bytes moved. No compute kernel of any kind is implied or claimed by this seam: crossing a block's
/// bytes to device memory does not mean anything on the device reads them.
/// </summary>
public interface IKvDeviceMirror : IDisposable
{
    /// <summary>The residency tier this mirror backs.</summary>
    WeightTier Tier { get; }

    /// <summary>Bytes of persistent device-resident storage this mirror holds: one whole block per device-backed block id, allocated once and kept for the mirror's lifetime.</summary>
    long DeviceBytes { get; }

    /// <summary>Device-backed block ids whose host bytes have crossed to the device copy at least once.</summary>
    int CrossedBlocks { get; }

    /// <summary>Total bytes copied host-to-device across every flush, counting each crossing.</summary>
    long CrossedBytes { get; }

    /// <summary>Device-backed blocks written since their last crossing — dirty and awaiting the next flush.</summary>
    int PendingWrites { get; }

    /// <summary>Flags one device-backed block dirty: its host bytes changed since its last crossing and its next flush must carry them.</summary>
    void MarkDirty(int blockId);

    /// <summary>Copies every dirty device-backed block's whole host payload to the device copy and clears the pending set. A mirror with nothing pending copies nothing.</summary>
    void Flush();
}
