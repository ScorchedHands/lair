using System.IO.MemoryMappedFiles;

using Lair.Contracts;

namespace Lair.Loader;

/// <summary>
/// Maps one PLE n-gram table's rows out of the physical shard that holds them, resolving a row index to a zero-copy span of
/// exactly one packed table row. This is the gather seam for tables far too large to resolve whole: an assembled split model keeps
/// every <see cref="LogicalTensor.Offset"/> relative to its own shard's data region, so this source locates the shard that declares
/// the tensor, maps only that file, and addresses rows at <c>shard.DataStart + tensor.Offset + rowIndex * RowStrideBytes</c>. No whole-table copy is taken — a geometry or layout failure throws
/// <see cref="PleTableException"/> naming the model file, shard, and tensor rather than falling back to reading everything.
///
/// Spans returned by <see cref="ResolveRow(long)"/> remain valid for the lifetime of this source: dispose it only after every span
/// obtained from it is no longer used. Mapping reserves address space; only the pages a resolved row actually touches are read.
/// </summary>
public sealed unsafe class PleRowSource : IDisposable
{
    private MemoryMappedFile? _file;

    // The view and its acquired pointer exist only after mapping succeeds, so the release paths must tolerate a failure made before either.
    private MemoryMappedViewAccessor? _view;
    private byte* _base;

    // Row addressing state: the owning shard's data region plus this tensor's payload start within it.
    private readonly long _dataStart;
    private readonly long _tensorOffset;
    private readonly string _shardPath;
    private bool _disposed;

    public PleRowSource(GgufFile model, PleNgramTable table)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(table);

        if (!string.Equals(System.IO.Path.GetFullPath(model.Path), table.ModelPath, StringComparison.Ordinal))
        {
            throw new PleTableException(
                $"'{model.Path}' is not the model file '{table.ModelPath}' this PLE descriptor was read from — row gather must address the bytes the declared geometry describes");
        }

        if (table.RowStrideBytes <= 0 || table.RowStrideBytes > int.MaxValue)
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{table.Tensor.Name}': a packed PLE row of {table.RowStrideBytes} bytes ({table.Tensor.Format}) is not a size this gather can address");
        }

        RowStrideBytes = (int)table.RowStrideBytes;
        Rows = table.TableRows;
        Table = table;

        // Find the physical file that actually holds this tensor's payload: for an assembled split, each shard declares its own tensors with offsets relative to its own data region.
        GgufFile owner;
        if (model.Shards is null)
        {
            owner = model;
        }
        else
        {
            GgufFile? found = null;

            foreach (GgufFile shard in model.Shards)
            {
                LogicalTensor? declared = shard.Tensors.Find(table.Tensor.Name);
                if (declared is null)
                {
                    continue;
                }

                if (found is not null)
                {
                    throw new PleTableException(
                        $"model '{model.Path}': PLE tensor '{table.Tensor.Name}' is declared by both '{found.Path}' and sibling '{shard.Path}' — row gather needs exactly one owner");
                }

                found = shard;
            }

            if (found is null)
            {
                throw new PleTableException(
                    $"model '{model.Path}': PLE tensor '{table.Tensor.Name}' is declared by no shard of this split model, so its rows cannot be gathered from the files on disk");
            }

            owner = found;

            // The assembled view and the physical shard must agree about where these bytes are: a disagreement means the descriptor describes payload this file does not hold.
            LogicalTensor ownerDeclared = owner.Tensors.Find(table.Tensor.Name)!;
            if (ownerDeclared.Offset != table.Tensor.Offset || ownerDeclared.SizeBytes != table.Tensor.SizeBytes)
            {
                throw new PleTableException(
                    $"model '{model.Path}': shard '{owner.Path}' declares tensor '{table.Tensor.Name}' at offset {ownerDeclared.Offset} with {ownerDeclared.SizeBytes} payload bytes, " +
                    $"but the assembled model describes it at offset {table.Tensor.Offset} with {table.Tensor.SizeBytes} — row gather refuses rather than reading a different file's bytes");
            }
        }

        _shardPath = owner.Path;
        _dataStart = owner.DataStart;
        _tensorOffset = table.Tensor.Offset;

        long fileSize = owner.FileSize; // each parsed shard carries its own physical length, so row bounds are checked against the file being mapped.
        if (table.Tensor.SizeBytes > fileSize - _dataStart || _tensorOffset > fileSize - _dataStart)
        {
            throw new PleTableException(
                $"model '{model.Path}': tensor '{table.Tensor.Name}' payload [{_dataStart + _tensorOffset}..{_dataStart + _tensorOffset + table.Tensor.SizeBytes}) " +
                $"does not fit in the {fileSize}-byte shard '{owner.Path}'");
        }

        // One mapping of exactly the file that owns these rows: capacity 0 maps the whole physical shard, and only touched row pages are ever read.
        MemoryMappedFile mapped = MemoryMappedFile.CreateFromFile(owner.Path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
        _file = mapped;

        try
        {
            MemoryMappedViewAccessor view = mapped.CreateViewAccessor(0, fileSize, MemoryMappedFileAccess.Read);
            _view = view;
            view.SafeMemoryMappedViewHandle.AcquirePointer(ref _base);

            long rowsEnd = checked(_tensorOffset + Rows * RowStrideBytes); // the last row must land inside this tensor's declared payload, which was already file-bounds-checked.
            if (rowsEnd > table.Tensor.SizeBytes)
            {
                throw new PleTableException(
                    $"model '{model.Path}': tensor '{table.Tensor.Name}' rows at a {RowStrideBytes}-byte stride reach byte {rowsEnd}, past the " +
                    $"{table.Tensor.SizeBytes}-byte payload this gather addresses");
            }
        }
        catch
        {
            ReleaseMapping();
            mapped.Dispose();
            throw;
        }
    }

    /// <summary>The descriptor whose geometry these rows are gathered under.</summary>
    public PleNgramTable Table { get; }

    /// <summary>Packed row size in bytes for this table's quant format.</summary>
    public int RowStrideBytes { get; }

    /// <summary>The table's row space: valid indices run 0 through <c>Rows - 1</c>, and every declared head range was validated to land inside it.</summary>
    public long Rows { get; }

    /// <summary>The physical file whose mapping backs these rows — the shard that declares this tensor for a split model, or the single-file model itself.</summary>
    public string ShardPath => _shardPath;

    /// <summary>
    /// The span of one packed table row: <c>RowStrideBytes</c> bytes at <c>DataStart + Offset + rowIndex * RowStrideBytes</c> in the owning shard. A row index outside this table's declared
    /// row space is refused naming the tensor and file, because a gather that invents rows would read bytes no head range ever declared — there is no whole-table fallback here to reach instead.
    /// </summary>
    public ReadOnlySpan<byte> ResolveRow(long rowIndex)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (rowIndex < 0 || rowIndex >= Rows)
        {
            throw new ArgumentOutOfRangeException(
                nameof(rowIndex),
                $"row {rowIndex} is outside the {Rows}-row PLE table '{Table.Tensor.Name}' in '{_shardPath}'");
        }

        long byteOffset = checked(_tensorOffset + rowIndex * RowStrideBytes); // row addressing multiplies a validated index by this format's fixed stride, exactly as every other kernel indexes blocks.
        return new ReadOnlySpan<byte>(_base + _dataStart + byteOffset, RowStrideBytes);
    }

    /// <summary>Whether one resolved row lies inside the mapped view — the layout check gather callers take before trusting a computed index.</summary>
    public bool ContainsRow(long rowIndex) => rowIndex >= 0 && rowIndex < Rows;

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        ReleaseMapping();
        _file?.Dispose();
    }

    private void ReleaseMapping()
    {
        if (_base is not null && _view is not null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _base = null;
        }

        _view?.Dispose();
    }
}
