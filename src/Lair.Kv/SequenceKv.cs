namespace Lair.Kv;

/// <summary>
/// Per-sequence block table for a paged KV cache: an append-only mapping from logical token positions to physical blocks in a
/// <see cref="BlockPool"/>. A new block is allocated when the sequence crosses a multiple of the pool's block size, so a length-40
/// sequence holds three blocks with the last one partially filled. Blocks may be owned privately (allocated by this sequence) or
/// adopted from shared cache entries through <see cref="MarkAdopted"/>; adopted blocks are read-only — a write into one copies the
/// whole block to fresh private storage first, so sharing never lets one sequence disturb another's KV bytes. K/V writes and reads
/// are addressed by logical position plus layer. A read that honors a per-layer window asks <see cref="SelectCausalRead"/> or
/// <see cref="SelectRangeRead"/> for its block selection first: the selection names the positions and blocks the read covers, the
/// windowed read overloads refuse any position outside it, and every earlier block stays allocated for the layers that still attend
/// unbounded. Disposing the sequence releases every reference it holds back to the pool.
/// </summary>
public sealed class SequenceKv : IDisposable
{
    private readonly BlockPool _pool;
    private readonly List<int> _blocks = [];

    // Logical block indexes whose physical block is shared, not owned: the copy-on-write guard consults this set before every write.
    // Indexes (not ids) because a CoW swap replaces one index's mapping, and an id would make freed-then-reallocated blocks
    // indistinguishable from still-shared ones.
    private readonly HashSet<int> _adopted = [];

    private int _length;
    private bool _disposed;

    public SequenceKv(BlockPool pool)
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

    /// <summary>Appends one token, allocating a new block when the length crosses the block size.</summary>
    /// <returns>The slot the appended token lives in.</returns>
    /// <exception cref="KvFullException">The pool has no free blocks left for the crossing append.</exception>
    public SlotRef Append()
    {
        ThrowIfDisposed();

        int pos = _length;
        if (pos % _pool.BlockTokens == 0)
        {
            // Appends always cross into a fresh private block: an adopted shared block is never extended by an append.
            _blocks.Add(_pool.Allocate());
        }

        _length = pos + 1;
        return new SlotRef(_blocks[pos / _pool.BlockTokens], pos % _pool.BlockTokens);
    }

    /// <summary>
    /// Places a shared physical block at the next logical block index and marks it adopted (read-only) so copy-on-write guards later
    /// writes. The caller has already added this sequence's reference to the block; disposing this sequence releases exactly that
    /// reference. Adoption must proceed in order from position 0 — an adoption can never rewrite a live mapping, and consecutive
    /// adoptions keep the table dense because <see cref="Append"/> relies on it. The logical length is untouched here; use
    /// <see cref="SetLength"/> once every adopted block is in place so prefill resumes after the shared prefix.
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
    /// Sets the number of appended positions after adoption placed shared blocks in the table. Only a value already covered by whole
    /// adopted or private block entries is accepted: live appends are never rewound, and every position below the new length must
    /// resolve to an existing block entry.
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

    /// <summary>Writes the K vector (kvHeads * headDim floats) for one token position and layer.</summary>
    public void WriteK(int logicalPos, int layer, ReadOnlySpan<float> values) => WriteSlot(logicalPos, layer, isV: false, values);

    /// <summary>Writes the V vector (kvHeads * headDim floats) for one token position and layer.</summary>
    public void WriteV(int logicalPos, int layer, ReadOnlySpan<float> values) => WriteSlot(logicalPos, layer, isV: true, values);

    /// <summary>Reads back the K vector for one token position and layer. The span lives as long as the pool does.</summary>
    public ReadOnlySpan<float> ReadK(int logicalPos, int layer) => _pool.SlotSpan(Resolve(logicalPos), layer, isV: false, logicalPos % _pool.BlockTokens);

    /// <summary>Reads back the V vector for one token position and layer. The span lives as long as the pool does.</summary>
    public ReadOnlySpan<float> ReadV(int logicalPos, int layer) => _pool.SlotSpan(Resolve(logicalPos), layer, isV: true, logicalPos % _pool.BlockTokens);

    /// <summary>
    /// Reads one position's K vector inside an existing block selection. A position the selection does not cover is a
    /// failure rather than a silent read: once a layer has resolved its window into a selection, that layer's reads
    /// cannot reach past the window by accident, and reaching it again takes a different (selection-free) call.
    /// </summary>
    public ReadOnlySpan<float> ReadK(BlockSelection selection, int logicalPos, int layer) => SelectedSlot(selection, logicalPos, layer, isV: false);

    /// <summary>V twin of <see cref="ReadK(BlockSelection, int, int)"/>.</summary>
    public ReadOnlySpan<float> ReadV(BlockSelection selection, int logicalPos, int layer) => SelectedSlot(selection, logicalPos, layer, isV: true);

    /// <summary>The physical block backing a logical position. Valid for positions below <see cref="Length"/>.</summary>
    public int BlockFor(int logicalPos) => Resolve(logicalPos);

    /// <summary>
    /// Selects what one causal attention read at <paramref name="queryPosition"/> covers under that layer's resolved
    /// window: the positions <c>[max(0, queryPosition - layerWindow + 1), queryPosition]</c> for a positive window and
    /// the whole prefix <c>[0..queryPosition]</c> when the window is non-positive (unbounded), narrowed to exactly the
    /// logical blocks those positions live in. A window narrower than the prefix selects fewer blocks than the sequence
    /// holds — the excluded ones are reported as <see cref="BlockSelection.SkippedBlocks"/> — while an unbounded read
    /// reports <see cref="BlockSelection.CoversWholeTable"/>, so boundedness is a property of the selection lair.kv
    /// produced rather than a loop bound the reader happens to obey. Selection reads the table only: adopted (shared)
    /// and private blocks select identically, which keeps prompt-cache reuse addressing the same physical blocks.
    /// </summary>
    public BlockSelection SelectCausalRead(int queryPosition, int layerWindow)
    {
        ThrowIfDisposed();

        if (queryPosition < 0 || queryPosition >= _length)
        {
            throw new ArgumentOutOfRangeException(nameof(queryPosition), $"Position {queryPosition} is outside the sequence of length {_length}.");
        }

        int firstVisible = layerWindow > 0 ? Math.Max(0, queryPosition - layerWindow + 1) : 0;
        return SelectRangeRead(firstVisible, queryPosition);
    }

    /// <summary>
    /// Selects what a read over an explicit position range covers — the variant every query of a bidirectional pass
    /// resolves for its own windowed range inside the sequence's finished KV. Both edges must lie inside the positions
    /// this sequence holds; partial blocks at either edge select because their in-range slots are what such a read
    /// addresses, and no block outside the run does.
    /// </summary>
    public BlockSelection SelectRangeRead(int fromPosition, int toPosition)
    {
        ThrowIfDisposed();

        ArgumentOutOfRangeException.ThrowIfNegative(fromPosition);
        if (toPosition >= _length)
        {
            throw new ArgumentOutOfRangeException(nameof(toPosition), $"Position {toPosition} is outside the sequence of length {_length}.");
        }

        int firstBlock = fromPosition / _pool.BlockTokens;
        int lastBlock = toPosition / _pool.BlockTokens;
        if (lastBlock >= _blocks.Count)
        {
            throw new InvalidOperationException($"Internal table gap: position {toPosition} has no physical block.");
        }

        return new BlockSelection(fromPosition, toPosition, firstBlock, lastBlock - firstBlock + 1, _blocks.Count);
    }

    private void WriteSlot(int logicalPos, int layer, bool isV, ReadOnlySpan<float> values)
    {
        if (values.Length != _pool.KvDim)
        {
            throw new ArgumentException($"Expected {_pool.KvDim} floats ({_pool.KvHeads} heads x {_pool.HeadDim}), got {values.Length}.", nameof(values));
        }

        int blockIndex = logicalPos / _pool.BlockTokens;
        if (_adopted.Contains(blockIndex))
        {
            CopyOnWrite(blockIndex); // a genuine mid-block rewrite: the shared copy becomes private before any byte lands in it
        }

        Span<float> slot = _pool.SlotSpan(_blocks[blockIndex], layer, isV, logicalPos % _pool.BlockTokens);
        values.CopyTo(slot);
    }

    // Addresses one slot inside a selection the sequence itself produced: the coverage check is what makes a windowed
    // read bounded instead merely narrow, and it costs nothing beyond the bounds checks every span read already pays.
    private Span<float> SelectedSlot(BlockSelection selection, int logicalPos, int layer, bool isV)
    {
        if (logicalPos < selection.FirstPosition || logicalPos > selection.LastPosition)
        {
            throw new ArgumentOutOfRangeException(
                nameof(logicalPos),
                $"position {logicalPos} lies outside the selected read [{selection.FirstPosition}..{selection.LastPosition}] " +
                $"covering {selection.SelectedBlocks} of {selection.TotalBlocks} blocks.");
        }

        return _pool.SlotSpan(Resolve(logicalPos), layer, isV, logicalPos % _pool.BlockTokens);
    }

    /// <summary>
    /// Replaces one adopted logical block with a private copy: the shared payload moves whole into a fresh block so every byte this
    /// sequence would keep — written or never-written zeros alike — survives bit-identically, then the mapping swaps and this
    /// sequence's reference on the shared block is released. The original holders (a cache entry, another sequence) see their copy
    /// untouched; a freed shared block returns to the free list only when its last reference goes. Appends never route here because
    /// they always allocate fresh blocks — this fires only for writes into positions an adoption already covered.
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
        _pool.BlockView(shared).CopyTo(_pool.MutableBlockSpan(fresh));

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

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, typeof(SequenceKv));
}
