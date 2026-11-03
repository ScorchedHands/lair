using System.Buffers;
using System.Runtime.InteropServices;

namespace Lair.Contracts;

/// <summary>
/// In-memory <see cref="IWeightSource"/>: a logical tensor table whose payloads are
/// backed by a caller-supplied byte array or native memory region. Doubles as the
/// alternate provider used to prove weight retrieval is pluggable behind the seam.
/// </summary>
public sealed class CpuWeightArena : IWeightSource
{
    private readonly LogicalTensorTable _table;
    private readonly byte[]? _image;
    private IntPtr _region;
    private readonly long _length;
    private readonly bool _ownsRegion;

    public CpuWeightArena(LogicalTensorTable table, byte[] image)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(image);

        _table = table;
        _image = image;
        _length = image.Length;
        Validate(table, _length);
    }

    /// <summary>
    /// Wraps a caller-supplied native region. The arena takes ownership: the region
    /// must have been allocated with NativeMemory aligned-alloc semantics and is
    /// freed with NativeMemory aligned-free on <see cref="Dispose"/>.
    /// </summary>
    public unsafe CpuWeightArena(LogicalTensorTable table, MemoryHandle region, long length)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentOutOfRangeException.ThrowIfNegative(length);

        _table = table;
        _region = (IntPtr)region.Pointer;
        _length = length;
        _ownsRegion = true;
        Validate(table, length);
    }

    public LogicalTensorTable Table => _table;

    public unsafe ReadOnlySpan<byte> Resolve(LogicalTensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        if (tensor.Offset < 0 || tensor.SizeBytes < 0 || tensor.Offset + tensor.SizeBytes > _length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(tensor),
                $"{tensor.Name} [{tensor.Offset}..{tensor.Offset + tensor.SizeBytes}) is outside the arena [0..{_length}).");
        }

        if (_image is not null)
        {
            return _image.AsSpan((int)tensor.Offset, (int)tensor.SizeBytes);
        }

        return new Span<byte>((byte*)_region + tensor.Offset, (int)tensor.SizeBytes);
    }

    public unsafe void Dispose()
    {
        if (_ownsRegion && _region != IntPtr.Zero)
        {
            NativeMemory.AlignedFree(_region.ToPointer());
            _region = IntPtr.Zero;
        }
    }

    private static void Validate(LogicalTensorTable table, long length)
    {
        foreach (LogicalTensor tensor in table.Tensors)
        {
            if (tensor.Offset < 0 || tensor.SizeBytes < 0 || tensor.Offset + tensor.SizeBytes > length)
            {
                throw new ArgumentException(
                    $"{tensor.Name} [{tensor.Offset}..{tensor.Offset + tensor.SizeBytes}) is outside the arena [0..{length}).",
                    nameof(table));
            }
        }
    }
}
