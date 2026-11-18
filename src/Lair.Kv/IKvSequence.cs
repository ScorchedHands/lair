namespace Lair.Kv;

/// <summary>
/// The per-sequence block-table surface a forward route addresses, whichever KV format pair its pool opened with: the same
/// append-only logical→physical table, the same adopt/copy-on-write posture, and the same block-selection reads whether the
/// pool stores FP32 slots or a quantized pair's groups. <see cref="SequenceKv"/> and <see cref="SequenceKv{TK, TV}"/> implement
/// it, so a forward body can hold a sequence generically while the format-specific reads and quantizing writes stay on the
/// concrete type the route closed over at open. Appends, adoption, and selection addressing are format-independent; the K/V
/// writes take float values because every storage format's own writer quantizes them at the storage seam.
/// </summary>
public interface IKvSequence : IDisposable
{
    /// <summary>Tokens one logical block covers: the granularity this sequence's table is addressed at.</summary>
    int BlockTokens { get; }

    /// <summary>Number of token positions appended so far.</summary>
    int Length { get; }

    /// <summary>Token slots reserved by the blocks this sequence currently owns.</summary>
    int TokenCapacity { get; }

    /// <summary>Logical→physical block table, in append order. An entry only changes when copy-on-write replaces a shared block.</summary>
    IReadOnlyList<int> Blocks { get; }

    /// <summary>Appends one token, allocating a new private block when the length crosses the block size.</summary>
    SlotRef Append();

    /// <summary>Places a shared physical block at the next logical block index and marks it adopted (read-only).</summary>
    void MarkAdopted(int physicalBlockId, int startLogicalPos);

    /// <summary>Sets the number of appended positions after adoption placed shared blocks in the table.</summary>
    void SetLength(int length);

    /// <summary>True while the logical block covering this position is adopted (shared), false when privately owned.</summary>
    bool IsAdopted(int logicalPos);

    /// <summary>The physical block backing a logical position. Valid for positions below <see cref="Length"/>.</summary>
    int BlockFor(int logicalPos);

    /// <summary>Writes the K side (kvDim values) for one token position and layer — quantized storage quantizes through its own writer.</summary>
    void WriteK(int logicalPos, int layer, ReadOnlySpan<float> values);

    /// <summary>Writes the V side (kvDim values) for one token position and layer.</summary>
    void WriteV(int logicalPos, int layer, ReadOnlySpan<float> values);

    /// <summary>Selects what one causal attention read at <paramref name="queryPosition"/> covers under that layer's resolved window.</summary>
    BlockSelection SelectCausalRead(int queryPosition, int layerWindow);

    /// <summary>Selects what a read over an explicit position range covers.</summary>
    BlockSelection SelectRangeRead(int fromPosition, int toPosition);
}
