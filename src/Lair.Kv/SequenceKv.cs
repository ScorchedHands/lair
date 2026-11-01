namespace Lair.Kv;

/// <summary>
/// Per-sequence block table for a paged KV cache: an append-only mapping from logical token
/// positions to physical blocks in a <see cref="BlockPool"/>. A new block is allocated when the
/// sequence crosses a multiple of the pool's block size, so a length-40 sequence holds three
/// blocks with the last one partially filled. K/V writes and reads are addressed by logical
/// position plus layer; disposing the sequence releases every block it owns back to the pool.
/// </summary>
public sealed class SequenceKv : IDisposable
{
    private readonly BlockPool _pool;
    private readonly List<int> _blocks = [];
    private int _length;
    private bool _disposed;

    public SequenceKv(BlockPool pool)
    {
        ArgumentNullException.ThrowIfNull(pool);
        _pool = pool;
    }

    /// <summary>Logical→physical block table, in append order. Blocks never change once appended.</summary>
    public IReadOnlyList<int> Blocks => _blocks;

    /// <summary>Number of token positions appended so far.</summary>
    public int Length => _length;

    /// <summary>Token slots reserved by the blocks this sequence currently owns.</summary>
    public int TokenCapacity => _blocks.Count * _pool.BlockTokens;

    /// <summary>Appends one token, allocating a new block when the length crosses the block size.</summary>
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

    /// <summary>Writes the K vector (kvHeads * headDim floats) for one token position and layer.</summary>
    public void WriteK(int logicalPos, int layer, ReadOnlySpan<float> values) => WriteSlot(logicalPos, layer, isV: false, values);

    /// <summary>Writes the V vector (kvHeads * headDim floats) for one token position and layer.</summary>
    public void WriteV(int logicalPos, int layer, ReadOnlySpan<float> values) => WriteSlot(logicalPos, layer, isV: true, values);

    /// <summary>Reads back the K vector for one token position and layer. The span lives as long as the pool does.</summary>
    public ReadOnlySpan<float> ReadK(int logicalPos, int layer) => _pool.SlotSpan(Resolve(logicalPos), layer, isV: false, logicalPos % _pool.BlockTokens);

    /// <summary>Reads back the V vector for one token position and layer. The span lives as long as the pool does.</summary>
    public ReadOnlySpan<float> ReadV(int logicalPos, int layer) => _pool.SlotSpan(Resolve(logicalPos), layer, isV: true, logicalPos % _pool.BlockTokens);

    /// <summary>The physical block backing a logical position. Valid for positions below <see cref="Length"/>.</summary>
    public int BlockFor(int logicalPos) => Resolve(logicalPos);

    private void WriteSlot(int logicalPos, int layer, bool isV, ReadOnlySpan<float> values)
    {
        if (values.Length != _pool.KvDim)
        {
            throw new ArgumentException($"Expected {_pool.KvDim} floats ({_pool.KvHeads} heads x {_pool.HeadDim}), got {values.Length}.", nameof(values));
        }

        Span<float> slot = _pool.SlotSpan(Resolve(logicalPos), layer, isV, logicalPos % _pool.BlockTokens);
        values.CopyTo(slot);
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
            _pool.Release(block);
        }

        _blocks.Clear();
        _length = 0;
        _disposed = true;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, typeof(SequenceKv));
}
