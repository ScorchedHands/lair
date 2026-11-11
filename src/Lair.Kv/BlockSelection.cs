namespace Lair.Kv;

/// <summary>
/// What one attention read selects out of a sequence's paged block table: the logical positions it covers, the
/// consecutive logical blocks those positions fall in, and the whole-table size that makes boundedness visible.
/// A selection is produced by <see cref="SequenceKv"/> from an already-resolved per-layer window — never from a
/// guess inside the read loop — so a layer whose window excludes earlier positions selects fewer blocks than the
/// sequence holds, while an unbounded layer's selection spans every block in the table. Positions outside the
/// selected range are not addressed by that read at all: their bytes stay in the pool for other layers to use, and
/// nothing reads them under this window.
/// </summary>
public readonly struct BlockSelection
{
    public BlockSelection(int firstPosition, int lastPosition, int firstBlockIndex, int selectedBlocks, int totalBlocks)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(firstPosition);
        if (lastPosition < firstPosition)
        {
            throw new ArgumentException(
                $"a selection must cover at least one position: [{firstPosition}..{lastPosition}] is empty.", nameof(lastPosition));
        }

        ArgumentOutOfRangeException.ThrowIfNegative(selectedBlocks);
        ArgumentOutOfRangeException.ThrowIfNegative(totalBlocks);
        if (selectedBlocks > totalBlocks)
        {
            throw new ArgumentException(
                $"a selection covering {selectedBlocks} blocks cannot exceed the table of {totalBlocks}.", nameof(selectedBlocks));
        }

        FirstPosition = firstPosition;
        LastPosition = lastPosition;
        FirstBlockIndex = firstBlockIndex;
        SelectedBlocks = selectedBlocks;
        TotalBlocks = totalBlocks;
    }

    /// <summary>First logical position the read covers: the window's left edge clamped to the sequence.</summary>
    public int FirstPosition { get; }

    /// <summary>Last logical position the read covers: the query's own position (or the range end a bidirectional read was given).</summary>
    public int LastPosition { get; }

    /// <summary>Logical block index where the selected run starts. Zero means the read reaches back to the first block.</summary>
    public int FirstBlockIndex { get; }

    /// <summary>Consecutive logical blocks the selection covers, in ascending order — never more than <see cref="TotalBlocks"/>.</summary>
    public int SelectedBlocks { get; }

    /// <summary>The whole table size at this query position, so a bounded selection compares directly against the unbounded one.</summary>
    public int TotalBlocks { get; }

    /// <summary>Last logical block index of the selected run (inclusive): <see cref="FirstBlockIndex"/> plus <see cref="SelectedBlocks"/> minus one.</summary>
    public int LastBlockIndex => FirstBlockIndex + SelectedBlocks - 1;

    /// <summary>Logical blocks the window keeps unread: how much of the table this read does not select.</summary>
    public int SkippedBlocks => TotalBlocks - SelectedBlocks;

    /// <summary>True when the selection covers every block from the first to the last — an unbounded read over this sequence.</summary>
    public bool CoversWholeTable => FirstBlockIndex == 0 && SelectedBlocks == TotalBlocks;
}
