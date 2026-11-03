using System.IO.MemoryMappedFiles;

using Lair.Contracts;

namespace Lair.Loader;

/// <summary>
/// Resolves GGUF tensor payloads from a memory-mapped view of the model file.
/// One <see cref="MemoryMappedFile"/> and one whole-file
/// <see cref="MemoryMappedViewAccessor"/> back every resolution;
/// <see cref="Resolve(LogicalTensor)"/> returns a span over the mapped bytes at
/// <c>dataStart + tensor.Offset</c> with length <c>tensor.SizeBytes</c>.
///
/// Spans returned by <see cref="Resolve(LogicalTensor)"/> remain valid for the
/// lifetime of this source: dispose the source only after every span obtained
/// from it is no longer used.
/// </summary>
public sealed unsafe class GgufMmapWeightSource : IWeightSource
{
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly long _dataStart;
    private bool _disposed;
    private byte* _base;

    public GgufMmapWeightSource(GgufFile gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);

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
    }

    public ReadOnlySpan<byte> Resolve(LogicalTensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ObjectDisposedException.ThrowIf(_disposed, this);

        long capacity = _view.Capacity;
        if (tensor.Offset < 0 || tensor.SizeBytes < 0 || tensor.Offset > capacity || tensor.SizeBytes > capacity - tensor.Offset)
        {
            throw new GgufException(
                $"tensor '{tensor.Name}' payload [{_dataStart + tensor.Offset}..{_dataStart + tensor.Offset + tensor.SizeBytes}) " +
                $"does not fit in the {capacity}-byte mapped view");
        }

        return new ReadOnlySpan<byte>(_base + _dataStart + tensor.Offset, (int)tensor.SizeBytes);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_base is not null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _base = null;
        }

        _view.Dispose();
        _file.Dispose();
    }
}
