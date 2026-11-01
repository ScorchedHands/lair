using System.Text;

using Lair.Core.Contracts;

namespace Lair.Loader;

/// <summary>
/// Parses a GGUF v2/v3 container: magic, KV metadata, and tensor infos with
/// offsets and payload sizes. Tensor payload bytes are not read here; an
/// <see cref="IWeightSource"/> such as <see cref="GgufMmapWeightSource"/> maps
/// them on demand.
/// </summary>
public sealed class GgufFile : IDisposable
{
    private const uint Magic = 0x46554747; // "GGUF" little-endian
    private const int DefaultAlignment = 32;

    private FileStream? _stream;

    private GgufFile(
        string path,
        int version,
        int alignment,
        GgufMetadata metadata,
        LogicalTensorTable tensors,
        long dataStart,
        long fileSize,
        FileStream stream)
    {
        Path = path;
        Version = version;
        Alignment = alignment;
        Metadata = metadata;
        Tensors = tensors;
        DataStart = dataStart;
        FileSize = fileSize;
        _stream = stream;
    }

    public string Path { get; }

    public int Version { get; }

    public int Alignment { get; }

    public GgufMetadata Metadata { get; }

    public LogicalTensorTable Tensors { get; }

    public long DataStart { get; }

    public long FileSize { get; }

    public static GgufFile Open(string path)
    {
        ArgumentNullException.ThrowIfNull(path);

        FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        try
        {
            using BinaryReader reader = new(stream, Encoding.UTF8, leaveOpen: true);
            return Parse(path, reader, stream);
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    public void Dispose()
    {
        FileStream? stream = _stream;
        _stream = null;
        stream?.Dispose();
    }

    private static GgufFile Parse(string path, BinaryReader reader, FileStream stream)
    {
        try
        {
            return ParseCore(path, reader, stream);
        }
        catch (IOException ex)
        {
            throw new GgufException($"'{path}' is truncated or unreadable as GGUF: {ex.Message}", ex);
        }
        catch (OverflowException ex)
        {
            throw new GgufException($"'{path}' declares tensor sizes that overflow a 64-bit length", ex);
        }
    }

    private static GgufFile ParseCore(string path, BinaryReader reader, FileStream stream)
    {
        uint magic = reader.ReadUInt32();
        if (magic != Magic)
        {
            throw new GgufException($"'{path}' is not a GGUF file (magic 0x{magic:x8})");
        }

        uint version = reader.ReadUInt32();
        if (version is not (2 or 3))
        {
            throw new GgufException($"'{path}' has unsupported GGUF version {version} (expected 2 or 3)");
        }

        ulong nTensors = version == 3 ? reader.ReadUInt64() : reader.ReadUInt32();
        ulong nKv = version == 3 ? reader.ReadUInt64() : reader.ReadUInt32();

        long fileSize = stream.Length;
        if (nTensors > (ulong)fileSize || nKv > (ulong)fileSize || nTensors > int.MaxValue || nKv > int.MaxValue)
        {
            throw new GgufException(
                $"'{path}' declares {nTensors} tensors / {nKv} KV pairs, which cannot fit in {fileSize} bytes");
        }

        Dictionary<string, GgufValue> kv = new(StringComparer.Ordinal);
        for (ulong i = 0; i < nKv; i++)
        {
            string key = ReadString(reader, fileSize);
            uint vtype = reader.ReadUInt32();
            if (vtype > (uint)GgufValueType.Float64)
            {
                throw new GgufException($"GGUF key '{key}' has unsupported value type {vtype}");
            }

            kv[key] = ReadValue(reader, (GgufValueType)vtype, fileSize);
        }

        int alignment = ReadAlignment(kv);

        List<LogicalTensor> tensors = new((int)nTensors);
        for (ulong i = 0; i < nTensors; i++)
        {
            tensors.Add(ReadTensorInfo(reader, fileSize));
        }

        long dataStart = AlignUp(stream.Position, alignment);

        long dataAvailable = fileSize - dataStart;

        foreach (LogicalTensor tensor in tensors)
        {
            if (tensor.Offset < 0 || tensor.Offset > dataAvailable || tensor.SizeBytes > dataAvailable - tensor.Offset)
            {
                throw new GgufException(
                    $"tensor '{tensor.Name}' payload [{dataStart + tensor.Offset}..{dataStart + tensor.Offset + tensor.SizeBytes}) " +
                    $"does not fit in {fileSize} bytes");
            }
        }

        GgufMetadata metadata = new(kv);
        LogicalTensorTable table = new(tensors);
        return new GgufFile(path, (int)version, alignment, metadata, table, dataStart, fileSize, stream);
    }

    private static int ReadAlignment(Dictionary<string, GgufValue> kv)
    {
        if (!kv.TryGetValue("general.alignment", out GgufValue? value))
        {
            return DefaultAlignment;
        }

        long alignment = value.AsLong();
        if (alignment <= 0 || alignment > int.MaxValue)
        {
            throw new GgufException($"general.alignment must be a positive 32-bit value, got {alignment}");
        }

        return (int)alignment;
    }

    private static LogicalTensor ReadTensorInfo(BinaryReader reader, long fileSize)
    {
        string name = ReadString(reader, fileSize);

        uint nDims = reader.ReadUInt32();
        if (nDims is < 1 or > 4)
        {
            throw new GgufException($"tensor '{name}' declares {nDims} dimensions (expected 1..4)");
        }

        // GGUF stores dims in ggml order (fastest-varying first), reversed relative
        // to logical row-major order: stored [in, out] becomes logical [out, in].
        uint[] stored = new uint[nDims];
        for (int d = 0; d < nDims; d++)
        {
            ulong dim = reader.ReadUInt64();
            if (dim == 0 || dim > uint.MaxValue)
            {
                throw new GgufException($"tensor '{name}' has invalid dimension {dim} at index {d}");
            }

            stored[d] = (uint)dim;
        }

        uint ggmlType = reader.ReadUInt32();
        ulong offset = reader.ReadUInt64();

        if (!GgmlType.TryGetSupported(ggmlType, out QuantFormat format, out int blockElements, out int blockBytes))
        {
            throw new GgufException(
                $"tensor '{name}' has unsupported ggml type {ggmlType} ({GgmlType.Name(ggmlType)})");
        }

        if (stored[0] % (uint)blockElements != 0)
        {
            throw new GgufException(
                $"tensor '{name}' fastest dimension {stored[0]} is not a multiple of the " +
                $"{GgmlType.Name(ggmlType)} block size {blockElements}");
        }

        long rowBytes = (long)blockBytes * (stored[0] / blockElements);
        long sizeBytes = rowBytes;
        for (int d = 1; d < stored.Length; d++)
        {
            sizeBytes = checked(sizeBytes * stored[d]);
        }

        int[] logical = new int[nDims];
        for (int d = 0; d < nDims; d++)
        {
            logical[d] = (int)stored[nDims - 1 - d];
        }

        return new LogicalTensor(
            name,
            new TensorShape(logical),
            format,
            (long)offset,
            sizeBytes);
    }

    private static GgufValue ReadValue(BinaryReader reader, GgufValueType type, long fileSize)
    {
        switch (type)
        {
            case GgufValueType.UInt8:
                return GgufValue.FromInteger(type, reader.ReadByte());
            case GgufValueType.Int8:
                return GgufValue.FromInteger(type, reader.ReadSByte());
            case GgufValueType.UInt16:
                return GgufValue.FromInteger(type, reader.ReadUInt16());
            case GgufValueType.Int16:
                return GgufValue.FromInteger(type, reader.ReadInt16());
            case GgufValueType.UInt32:
                return GgufValue.FromInteger(type, reader.ReadUInt32());
            case GgufValueType.Int32:
                return GgufValue.FromInteger(type, reader.ReadInt32());
            case GgufValueType.Float32:
                return GgufValue.FromReal(type, reader.ReadSingle());
            case GgufValueType.Bool:
                return GgufValue.FromInteger(type, reader.ReadByte() != 0 ? 1 : 0);
            case GgufValueType.String:
                return GgufValue.FromString(ReadString(reader, fileSize));
            case GgufValueType.Array:
                return ReadArray(reader, fileSize);
            case GgufValueType.UInt64:
                return GgufValue.FromUnsigned(type, reader.ReadUInt64());
            case GgufValueType.Int64:
                return GgufValue.FromInteger(type, reader.ReadInt64());
            case GgufValueType.Float64:
                return GgufValue.FromReal(type, reader.ReadDouble());
            default:
                throw new GgufException($"unsupported GGUF value type {(uint)type}");
        }
    }

    private static GgufValue ReadArray(BinaryReader reader, long fileSize)
    {
        uint elementType = reader.ReadUInt32();
        if (elementType > (uint)GgufValueType.Float64)
        {
            throw new GgufException($"GGUF array has unsupported element type {elementType}");
        }

        ulong count = reader.ReadUInt64();
        long remaining = fileSize - reader.BaseStream.Position;
        if (count > (ulong)remaining)
        {
            throw new GgufException($"GGUF array declares {count} elements but only {remaining} bytes remain");
        }

        GgufValue[] elements = new GgufValue[(int)count];
        for (int i = 0; i < elements.Length; i++)
        {
            elements[i] = ReadValue(reader, (GgufValueType)elementType, fileSize);
        }

        return GgufValue.FromArray(elements);
    }

    private static string ReadString(BinaryReader reader, long fileSize)
    {
        ulong length = reader.ReadUInt64();
        long remaining = fileSize - reader.BaseStream.Position;
        if (length > (ulong)remaining || length > int.MaxValue)
        {
            throw new GgufException($"GGUF string declares {length} bytes but only {remaining} remain");
        }

        return Encoding.UTF8.GetString(reader.ReadBytes((int)length));
    }

    private static long AlignUp(long value, long alignment) =>
        (value + alignment - 1) / alignment * alignment;
}
