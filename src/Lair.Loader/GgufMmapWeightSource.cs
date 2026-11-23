using System.IO.MemoryMappedFiles;

using Lair.Contracts;

namespace Lair.Loader;

/// <summary>
/// Resolves GGUF tensor payloads from memory-mapped views of the model file.
/// For an ordinary single-file model, one <see cref="MemoryMappedFile"/> and one
/// whole-file <see cref="MemoryMappedViewAccessor"/> back every resolution;
/// <see cref="Resolve(LogicalTensor)"/> returns a span over the mapped bytes at
/// <c>dataStart + tensor.Offset</c> with length <c>tensor.SizeBytes</c>.
///
/// For an assembled split model (<see cref="GgufFile.Shards"/> non-null), each
/// physical shard is memory-mapped separately and every resolution maps the
/// tensor to its own shard: a logical tensor's offset addresses that shard's
/// data region, so <c>shard.DataStart + tensor.Offset</c> yields the span from
/// the file that actually holds those bytes. No whole-model copy exists — only
/// the shard files are mapped, and spans address them zero-copy.
///
/// Spans returned by <see cref="Resolve(LogicalTensor)"/> remain valid for the
/// lifetime of this source: dispose the source only after every span obtained
/// from it is no longer used.
/// </summary>
public sealed unsafe class GgufMmapWeightSource : IWeightSource
{
    private readonly MemoryMappedFile? _file; // single-file model mapping
    private readonly MemoryMappedViewAccessor? _view;
    private readonly long _dataStart; // data region start of the mapped file (single-file models)
    private bool _disposed;
    private byte* _base;

    private readonly GgufFile[]? _shards; // split model: index-aligned with GgufFile.Shards
    private MemoryMappedFile?[]? _shardFiles;
    private MemoryMappedViewAccessor?[]? _shardViews;
    private byte*[]? _shardBases;

    // Tensor name to shard position, built from the per-shard tables of an assembled split (the assembly has 
    // already refused any tensor declared by two shards, so this mapping is 1:1). Null for a single-file model.
    private Dictionary<string, int>? _tensorToShardIndex;

    public GgufMmapWeightSource(GgufFile gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);

        if (gguf.Shards is null)
        {
            // Single-file model: the historical whole-file mapping, unchanged.
            _dataStart = gguf.DataStart;
            _file = MemoryMappedFile.CreateFromFile(gguf.Path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
            try
            {
                _view = _file.CreateViewAccessor(0, gguf.FileSize, MemoryMappedFileAccess.Read);
                _view.SafeMemoryMappedViewHandle.AcquirePointer(ref _base);
            }
            catch
            {
                _file.Dispose();
                throw;
            }

            return;
        }

        // Split model: one mapping per physical shard. A failed shard mapping releases the mappings already made.
        GgufFile[] shards = [.. gguf.Shards];
        MemoryMappedFile?[] files = new MemoryMappedFile?[shards.Length];
        MemoryMappedViewAccessor?[] views = new MemoryMappedViewAccessor?[shards.Length];
        byte*[] bases = new byte*[shards.Length];
        try
        {
            for (int i = 0; i < shards.Length; i++)
            {
                MemoryMappedFile mapped = MemoryMappedFile.CreateFromFile(shards[i].Path, FileMode.Open, mapName: null, capacity: 0, MemoryMappedFileAccess.Read);
                files[i] = mapped;
                views[i] = mapped.CreateViewAccessor(0, shards[i].FileSize, MemoryMappedFileAccess.Read);
            }

            for (int i = 0; i < views.Length; i++)
            {
                views[i]!.SafeMemoryMappedViewHandle.AcquirePointer(ref bases[i]);
            }
        }
        catch
        {
            ReleaseShardMappings(views, bases);
            DisposeFiles(files);
            throw;
        }

        Dictionary<string, int> tensorToShard = new(StringComparer.Ordinal);
        for (int i = 0; i < shards.Length; i++)
        {
            foreach (LogicalTensor tensor in shards[i].Tensors.Tensors)
            {
                tensorToShard[tensor.Name] = i; // disjointness was validated by the assembly.
            }
        }

        _shards = shards;
        _shardFiles = files;
        _shardViews = views;
        _shardBases = bases;
        _tensorToShardIndex = tensorToShard;
    }

    public ReadOnlySpan<byte> Resolve(LogicalTensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (tensor.SizeBytes > int.MaxValue)
        {
            throw new GgufException(
                $"tensor '{tensor.Name}' payload size {tensor.SizeBytes} exceeds the largest span this source can return ({int.MaxValue} bytes)");
        }

        if (_shards is null)
        {
            long capacity = _view!.Capacity;
            if (tensor.Offset < 0 || tensor.SizeBytes < 0 || tensor.Offset > capacity || tensor.SizeBytes > capacity - tensor.Offset)
            {
                throw new GgufException(
                    $"tensor '{tensor.Name}' payload [{_dataStart + tensor.Offset}..{_dataStart + tensor.Offset + tensor.SizeBytes}) " +
                    $"does not fit in the {capacity}-byte mapped view");
            }

            return new ReadOnlySpan<byte>(_base + _dataStart + tensor.Offset, (int)tensor.SizeBytes);
        }

        if (!_tensorToShardIndex!.TryGetValue(tensor.Name, out int shardIndex))
        {
            throw new GgufException(
                $"tensor '{tensor.Name}' is declared by no shard of the split model mapped by this source");
        }

        GgufFile shard = _shards[shardIndex];
        long shardCapacity = _shardViews![shardIndex]!.Capacity; // the whole physical shard file.
        if (tensor.Offset < 0 || tensor.SizeBytes < 0 || tensor.Offset > shardCapacity - shard.DataStart || tensor.SizeBytes > shardCapacity - shard.DataStart - tensor.Offset)
        {
            throw new GgufException(
                $"tensor '{tensor.Name}' payload [{shard.DataStart + tensor.Offset}..{shard.DataStart + tensor.Offset + tensor.SizeBytes}) " +
                $"does not fit in the {shardCapacity}-byte mapped view of shard '{shard.Path}'");
        }

        return new ReadOnlySpan<byte>(_shardBases![shardIndex] + shard.DataStart + tensor.Offset, (int)tensor.SizeBytes);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_base is not null && _view is not null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _base = null;
        }

        _view?.Dispose();
        _file?.Dispose();

        if (_shardViews is not null)
        {
            ReleaseShardMappings(_shardViews, _shardBases!);
            DisposeFiles(_shardFiles!);
            _shardViews = null;
            _shardFiles = null;
            _shardBases = null;
        }
    }

    private static void ReleaseShardMappings(MemoryMappedViewAccessor?[] views, byte*[] bases)
    {
        for (int i = 0; i < views.Length; i++)
        {
            MemoryMappedViewAccessor? view = views[i];
            if (bases[i] is not null && view is not null)
            {
                view.SafeMemoryMappedViewHandle.ReleasePointer();
                bases[i] = null;
            }

            view?.Dispose();
            views[i] = null;
        }
    }

    private static void DisposeFiles(MemoryMappedFile?[] files)
    {
        for (int i = 0; i < files.Length; i++)
        {
            files[i]?.Dispose();
            files[i] = null;
        }
    }
}
