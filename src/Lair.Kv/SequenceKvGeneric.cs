namespace Lair.Kv;

/// <summary>
/// Per-sequence block table for a quantized paged KV cache: the generic twin of <see cref="SequenceKv"/> over a
/// <see cref="BlockPool{TK, TV}"/>. An append-only mapping from logical token positions to physical blocks, with the same
/// adopt/copy-on-write semantics the float sequence runs: adopted blocks are read-only, and a write into one copies the whole
/// block — raw bytes, no re-quantization — into fresh private storage first, so sharing never lets one sequence disturb
/// another's stored groups. A K/V write quantizes the token's values into that token's typed block span through the format's
/// own writer; a read hands back the typed block span with no copy. Disposing the sequence releases every reference it holds.
/// </summary>
public sealed class SequenceKv<TK, TV> : IDisposable
    where TK : struct, IKvFormat<TK>
    where TV : struct, IKvFormat<TV>
{
    private readonly BlockPool<TK, TV> _pool;
    private readonly List<int> _blocks = [];

    // Logical block indexes whose physical block is shared, not owned: the copy-on-write guard consults this set before every
    // write. Indexes (not ids) because a CoW swap replaces one index's mapping.
    private readonly HashSet<int> _adopted = [];

    private int _length;
    private bool _disposed;

    public SequenceKv(BlockPool<TK, TV> pool)
    {
        ArgumentNullException.ThrowIfNull(pool);
        _pool = pool;
    }

    /// <summary>Logical→physical block table, in append order. An entry only changes when copy-on-write replaces a shared block.</summary>
    public IReadOnlyList<int> Blocks => _blocks;

    /// <summary>Number of token positions appended so far.</summary>
    public int Length => _length;

    /// <summary>True while the logical block covering this position is adopted (shared), false when privately owned.</summary>
    public bool IsAdopted(int logicalPos) => _adopted.Contains(logicalPos / _pool.BlockTokens);

    /// <summary>Token slots reserved by the blocks this sequence currently owns.</summary>
    public int TokenCapacity => _blocks.Count * _pool.BlockTokens;

    /// <summary>Tokens one logical block covers: the granularity this sequence's table is addressed at.</summary>
    public int BlockTokens => _pool.BlockTokens;

    /// <summary>
    /// Appends one token, allocating a new private block when the length crosses the block size. An adopted shared block is
    /// never extended by an append, so a crossing append always takes fresh private storage.
    /// </summary>
    /// <returns>The slot the appended token lives in.</returns>
    /// <exception cref="KvFullException">The pool has no free blocks left for the crossing append.</exception>
    public SlotRef Append()
    {
        ThrowIfDisposed();

        int pos = _length;
        if (pos % _pool.BlockTokens == 0)
        {
            _blocks.Add(_pool.Allocate());
        }

        _length = pos + 1;
        return new SlotRef(_blocks[pos / _pool.BlockTokens], pos % _pool.BlockTokens);
    }

    /// <summary>
    /// Places a shared physical block at the next logical block index and marks it adopted (read-only) so copy-on-write guards
    /// later writes. The caller has already added this sequence's reference to the block; disposing this sequence releases it.
    /// Adoption must proceed in order from position 0. The logical length is untouched here; use <see cref="SetLength"/> once
    /// every adopted block is in place.
    /// </summary>
    public void MarkAdopted(int physicalBlockId, int startLogicalPos)
    {
        ThrowIfDisposed();

        if (startLogicalPos % _pool.BlockTokens != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(startLogicalPos),
                $"adopted blocks must start on a block boundary ({_pool.BlockTokens} tokens), got position {startLogicalPos}.");
        }

        int index = startLogicalPos / _pool.BlockTokens;
        if (index != _blocks.Count)
        {
            throw new InvalidOperationException(
                $"adoption out of order: logical block {index} would overwrite an existing mapping or leave a gap " +
                $"in the table of {_blocks.Count} blocks.");
        }

        _blocks.Add(physicalBlockId);
        _adopted.Add(index);
    }

    /// <summary>
    /// Sets the number of appended positions after adoption placed shared blocks in the table. Only a value already covered by
    /// whole adopted or private block entries is accepted.
    /// </summary>
    public void SetLength(int length)
    {
        ThrowIfDisposed();

        int requiredBlocks = (length + _pool.BlockTokens - 1) / _pool.BlockTokens;
        if (length < _length || length > TokenCapacity || requiredBlocks > _blocks.Count)
        {
            throw new ArgumentOutOfRangeException(
                nameof(length),
                $"cannot set the sequence to {length} positions: it currently holds {_length} appended positions across {_blocks.Count} blocks.");
        }

        _length = length;
    }

    /// <summary>Quantizes and writes the K side (kvDim values) for one token position and layer.</summary>
    public void WriteK(int logicalPos, int layer, ReadOnlySpan<float> values) => WriteSlot(logicalPos, layer, isV: false, values);

    /// <summary>Quantizes and writes the V side (kvDim values) for one token position and layer.</summary>
    public void WriteV(int logicalPos, int layer, ReadOnlySpan<float> values) => WriteSlot(logicalPos, layer, isV: true, values);

    /// <summary>Reads back the K side for one token position and layer as its typed block span. The span lives as long as the pool does.</summary>
    public ReadOnlySpan<TK> ReadK(int logicalPos, int layer) => _pool.KTokenView(Resolve(logicalPos), layer, logicalPos % _pool.BlockTokens);

    /// <summary>Reads back the V side for one token position and layer as its typed block span.</summary>
    public ReadOnlySpan<TV> ReadV(int logicalPos, int layer) => _pool.VTokenView(Resolve(logicalPos), layer, logicalPos % _pool.BlockTokens);

    /// <summary>The physical block backing a logical position. Valid for positions below <see cref="Length"/>.</summary>
    public int BlockFor(int logicalPos) => Resolve(logicalPos);

    private void WriteSlot(int logicalPos, int layer, bool isV, ReadOnlySpan<float> values)
    {
        if (values.Length != _pool.KvDim)
        {
            throw new ArgumentException($"Expected {_pool.KvDim} values ({_pool.KvHeads} heads x {_pool.HeadDim}), got {values.Length}.", nameof(values));
        }

        int blockIndex = logicalPos / _pool.BlockTokens;
        if (_adopted.Contains(blockIndex))
        {
            CopyOnWrite(blockIndex); // a genuine mid-block rewrite: the shared copy becomes private before any group lands in it
        }

        int physicalBlockId = _blocks[blockIndex];
        int tokenSlot = logicalPos % _pool.BlockTokens;

        if (isV)
        {
            TV.WriteRegion(values, _pool.MutableVTokenSpan(physicalBlockId, layer, tokenSlot));
        }
        else
        {
            TK.WriteRegion(values, _pool.MutableKTokenSpan(physicalBlockId, layer, tokenSlot));
        }
    }

    /// <summary>
    /// Replaces one adopted logical block with a private copy: the shared block's raw bytes move whole into a fresh block via
    /// <see cref="BlockPool{TK, TV}.CopyBlockWhole"/>, so every stored group — written or untouched — survives bit-identically
    /// with no decode/requantize step, then the mapping swaps and this sequence's reference on the shared block is released.
    /// </summary>
    private void CopyOnWrite(int blockIndex)
    {
        int shared = _blocks[blockIndex];

        if (_pool.FreeBlocks == 0)
        {
            throw new KvFullException(
                $"KV block pool is full: copy-on-write needs a private copy of shared block {shared} " +
                $"and all {_pool.CapacityBlocks} blocks are in use.");
        }

        int fresh = _pool.Allocate(); // refcount 1, owned privately by this sequence alone
        _pool.CopyBlockWhole(shared, fresh); // whole raw block: no re-quantization of the copied groups

        _blocks[blockIndex] = fresh;
        _adopted.Remove(blockIndex);
        _pool.Release(shared);
    }

    private int Resolve(int logicalPos)
    {
        ThrowIfDisposed();
        if (logicalPos < 0 || logicalPos >= _length)
        {
            throw new ArgumentOutOfRangeException(nameof(logicalPos), $"Position {logicalPos} is outside the sequence of length {_length}.");
        }

        int blockIndex = logicalPos / _pool.BlockTokens;
        if (blockIndex >= _blocks.Count)
        {
            throw new InvalidOperationException($"Internal table gap: position {logicalPos} has no physical block.");
        }

        return _blocks[blockIndex];
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        foreach (int block in _blocks)
        {
            _pool.Release(block); // private and adopted references are symmetric: every entry is one reference this sequence holds
        }

        _blocks.Clear();
        _adopted.Clear();
        _length = 0;
        _disposed = true;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, typeof(SequenceKv<TK, TV>));
}
