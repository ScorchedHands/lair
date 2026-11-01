namespace Lair.Kv;

/// <summary>
/// Identifies one token slot inside a <see cref="BlockPool"/>: which physical block holds it, and
/// which logical position (0..blockTokens-1) within that block. Slot addressing is resolved
/// through the pool, never through raw arena offsets.
/// </summary>
public readonly struct SlotRef : IEquatable<SlotRef>
{
    public SlotRef(int blockId, int offsetWithinBlock)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(blockId);
        ArgumentOutOfRangeException.ThrowIfNegative(offsetWithinBlock);
        BlockId = blockId;
        OffsetWithinBlock = offsetWithinBlock;
    }

    /// <summary>Physical block id that backs this slot.</summary>
    public int BlockId { get; }

    /// <summary>Logical token position within the block (0-based).</summary>
    public int OffsetWithinBlock { get; }

    public bool Equals(SlotRef other) => BlockId == other.BlockId && OffsetWithinBlock == other.OffsetWithinBlock;

    public override bool Equals(object? obj) => obj is SlotRef other && Equals(other);

    public override int GetHashCode() => HashCode.Combine(BlockId, OffsetWithinBlock);

    public static bool operator ==(SlotRef left, SlotRef right) => left.Equals(right);

    public static bool operator !=(SlotRef left, SlotRef right) => !left.Equals(right);

    public override string ToString() => $"Slot({BlockId}+{OffsetWithinBlock})";
}
