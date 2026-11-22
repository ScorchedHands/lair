using System.Text;

using Lair.Contracts;

namespace Lair.Loader;

/// <summary>
/// Parses a GGUF v2/v3 container: magic, KV metadata, and tensor infos with
/// offsets and payload sizes. Tensor payload bytes are not read here; an
/// <see cref="IWeightSource"/> such as <see cref="GgufMmapWeightSource"/> maps
/// them on demand.
///
/// A model whose file name carries the llama.cpp split shape
/// `<prefix>-NNNNN-of-NNNNN.<ext>` is opened as a whole: its sibling shards in
/// the same directory are discovered, validated as one internally consistent
/// set, and assembled into one logical tensor table with merged metadata. In an
/// assembled model every <see cref="LogicalTensor.Offset"/> stays relative to
/// the data region of **its own shard's** physical file (as the GGUF spec
/// defines it), so only a weight source that consults <see cref="Shards"/> —
/// such as <see cref="GgufMmapWeightSource"/> — can map tensor spans. An
/// ordinary single-file model keeps today's meaning: offsets relative to this
/// file's own data region, and <see cref="Shards"/> is null.
/// </summary>
public sealed class GgufFile : IDisposable
{
    private const uint Magic = 0x46554747; // "GGUF" little-endian
    private const int DefaultAlignment = 32;

    private FileStream? _stream;
    private readonly IReadOnlyList<GgufFile>? _shards;

    private GgufFile(
        string path,
        int version,
        int alignment,
        GgufMetadata metadata,
        LogicalTensorTable tensors,
        long dataStart,
        long fileSize,
        FileStream? stream,
        IReadOnlyList<GgufFile>? shards = null)
    {
        Path = path;
        Version = version;
        Alignment = alignment;
        Metadata = metadata;
        Tensors = tensors;
        DataStart = dataStart;
        FileSize = fileSize;
        _stream = stream;
        _shards = shards;
    }

    public string Path { get; }

    /// <summary>GGUF version of the primary (lowest-numbered) physical shard.</summary>
    public int Version { get; }

    /// <summary><c>general.alignment</c> in effect for the primary (lowest-numbered) physical shard.</summary>
    public int Alignment { get; }

    public GgufMetadata Metadata { get; }

    public LogicalTensorTable Tensors { get; }

    /// <summary>Data-region start of the primary (lowest-numbered) physical shard. Each shard in 
    /// <see cref="Shards"/> carries its own data region.</summary>
    public long DataStart { get; }

    /// <summary>Total length of the primary (lowest-numbered) physical shard.</summary>
    public long FileSize { get; }

    /// <summary>The physical files backing this model in ascending shard order when a split was assembled, 
    /// each parsed from its own file with its own data region and tensor table; null for an ordinary 
    /// single-file model. The entry point that opened the split is one of these shards — any member of a 
    /// complete set opens the same logical model.</summary>
    public IReadOnlyList<GgufFile>? Shards => _shards;

    /// <summary>Opens a GGUF model. A file named `<prefix>-NNNNN-of-NNNNN.<ext>` is treated as one member of 
    /// a split: its sibling shards are discovered in the same directory, validated as one set, and assembled 
    /// into this single view (see the type summary). Any other name opens exactly one container.</summary>
    public static GgufFile Open(string modelPath) => OpenCore(modelPath, assemblyAllowed: true);

    /// <summary>Opens exactly one physical GGUF container with no split discovery — every sibling member of a 
    /// split set is inspected per-file through this entry point rather than assembled.</summary>
    public static GgufFile OpenSingle(string path) => OpenCore(path, assemblyAllowed: false);

    private static GgufFile OpenCore(string modelPath, bool assemblyAllowed)
    {
        ArgumentNullException.ThrowIfNull(modelPath);

        if (assemblyAllowed &&
            GgufShardNaming.TryParse(System.IO.Path.GetFileName(modelPath), out string prefix, out string extension, out int index, out int total))
        {
            return OpenSplitModel(modelPath, prefix, extension, index, total);
        }

        return OpenSingleFile(modelPath);
    }

    private static GgufFile OpenSingleFile(string path)
    {
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

    private static GgufFile OpenSplitModel(string modelPath, string prefix, string extension, int index, int total)
    {
        if (index < 1 || total < 1 || index > total)
        {
            throw new GgufException(
                $"'{modelPath}' names shard {index:D5} of a split whose name declares {total:D5} shards — the numbering is inconsistent");
        }

        string directory = System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(modelPath))!;
        List<string> shardPaths = [];
        foreach (string fileName in GgufShardNaming.ShardFileNames(prefix, extension, total))
        {
            string candidate = System.IO.Path.Combine(directory, fileName);
            if (!File.Exists(candidate))
            {
                throw new GgufException(
                    $"'{modelPath}' is shard {index:D5} of a split named with {total:D5} shards but the sibling " +
                    $"'{candidate}' was not found in '{directory}'");
            }

            shardPaths.Add(candidate);
        }

        List<GgufFile> shards = [];
        try
        {
            foreach (string shardPath in shardPaths)
            {
                shards.Add(OpenSingleFile(shardPath)); // a broken sibling refuses naming that file.
            }
        }
        catch
        {
            foreach (GgufFile opened in shards)
            {
                opened.Dispose();
            }

            throw;
        }

        return AssembleSplit(modelPath, [.. shards]);
    }

    private static GgufFile AssembleSplit(string modelPath, IReadOnlyList<GgufFile> shards)
    {
        int total = shards.Count;
        GgufFile primary = shards[0];

        foreach (GgufFile shard in shards)
        {
            if (shard.Version != primary.Version)
            {
                throw new GgufException(
                    $"'{modelPath}' is not internally consistent: '{primary.Path}' has GGUF version {primary.Version} " +
                    $"but '{shard.Path}' has GGUF version {shard.Version}");
            }
        }

        // The split identity keys every llama.cpp shard declares. Each key must appear in all shards or none, and 
        // each declared value must agree with the position its own file name claims — a shard whose metadata 
        // disagrees with its place in the set cannot silently map to the wrong bytes.
        string[] splitKeys = [GgufSplitKeys.No, GgufSplitKeys.Count, GgufSplitKeys.TensorsCount];
        foreach (string key in splitKeys)
        {
            List<GgufFile> declaring = shards.Where(shard => shard.Metadata.Has(key)).ToList();
            if (declaring.Count == 0)
            {
                continue; // no split identity declared: the file-name shape alone assembles the set.
            }

            foreach (GgufFile shard in shards)
            {
                if (!shard.Metadata.Has(key))
                {
                    throw new GgufException(
                        $"'{modelPath}' is not internally consistent: '{declaring[0].Path}' declares '{key}' " +
                        $"but sibling '{shard.Path}' carries no such key");
                }
            }

            if (key == GgufSplitKeys.No)
            {
                for (int i = 0; i < total; i++)
                {
                    long declared = shards[i].Metadata.Get(key).AsLong();
                    if (declared != i) // split.no is zero-based in every llama.cpp writer.
                    {
                        throw new GgufException(
                            $"'{modelPath}' is not internally consistent: '{shards[i].Path}' declares " +
                            $"{GgufSplitKeys.No} = {declared} but its name places it at shard position {i + 1:D5} of the split");
                    }
                }

                continue;
            }

            long firstValue = declaring[0].Metadata.Get(key).AsLong();
            foreach (GgufFile shard in declaring)
            {
                long value = shard.Metadata.Get(key).AsLong();
                if (value != firstValue)
                {
                    throw new GgufException(
                        $"'{modelPath}' is not internally consistent: '{declaring[0].Path}' declares " +
                        $"'{key}' = {firstValue} but sibling '{shard.Path}' declares {value}");
                }

                if (key == GgufSplitKeys.Count && value != total)
                {
                    throw new GgufException(
                        $"'{modelPath}' is not internally consistent: '{shard.Path}' declares " +
                        $"{GgufSplitKeys.Count} = {value} but the file-name shape names a split of {total:D5} shards");
                }
            }

            if (key == GgufSplitKeys.TensorsCount)
            {
                long declaredTensors = firstValue;
                long assembledTensors = 0;
                foreach (GgufFile shard in shards)
                {
                    assembledTensors += shard.Tensors.Tensors.Count;
                }

                if (declaredTensors < 0 || declaredTensors != assembledTensors)
                {
                    throw new GgufException(
                        $"'{modelPath}' is not internally consistent: the split headers declare " +
                        $"{GgufSplitKeys.TensorsCount} = {firstValue} tensors but the shard tensor tables assemble {assembledTensors}");
                }
            }
        }

        // Tensor names must be disjoint across shards, and every name maps to exactly one physical file. A 
        // repeated declaration is a broken set: resolving it could read either payload.
        Dictionary<string, GgufFile> tensorShardIndex = new(StringComparer.Ordinal);
        List<LogicalTensor> tensors = [];
        foreach (GgufFile shard in shards)
        {
            foreach (LogicalTensor tensor in shard.Tensors.Tensors)
            {
                if (!tensorShardIndex.TryAdd(tensor.Name, shard))
                {
                    throw new GgufException(
                        $"'{modelPath}' is not internally consistent: tensor '{tensor.Name}' is declared by both " +
                        $"'{tensorShardIndex[tensor.Name].Path}' and sibling '{shard.Path}'; split shards must carry disjoint tensor tables");
                }

                tensors.Add(tensor); // offsets stay relative to this shard's own data region.
            }
        }

        // Metadata merges with the primary (lowest-numbered) shard winning every collision — it carries the 
        // architecture and tokenizer keys while continuation shards add nothing new. Only `split.no` is allowed 
        // to differ across files, because that key records each file's own position; any other shared key with a 
        // different value is two headers disagreeing about one model.
        Dictionary<string, GgufValue> merged = new(StringComparer.Ordinal);
        List<GgufFile> sources = [];
        foreach (GgufFile shard in shards)
        {
            foreach (string key in shard.Metadata.Keys)
            {
                if (!merged.TryGetValue(key, out GgufValue? existing))
                {
                    merged[key] = shard.Metadata.Get(key);
                    sources.Add(shard);
                    continue;
                }

                if (key != GgufSplitKeys.No && !ValuesEqual(existing, shard.Metadata.Get(key)))
                {
                    int ownerIndex = 0;
                    while (!sources[ownerIndex].Metadata.Has(key))
                    {
                        ownerIndex++;
                    }

                    throw new GgufException(
                        $"'{modelPath}' is not internally consistent: metadata key '{key}' carries different values " +
                        $"in '{sources[ownerIndex].Path}' and sibling '{shard.Path}'");
                }
            }
        }

        GgufMetadata mergedMetadata = new(merged);
        LogicalTensorTable table = new(tensors);
        return new GgufFile(
            modelPath,
            primary.Version,
            primary.Alignment,
            mergedMetadata,
            table,
            primary.DataStart,
            primary.FileSize,
            stream: null,
            shards);
    }

    private static bool ValuesEqual(GgufValue left, GgufValue right)
    {
        if (left.Type != right.Type)
        {
            return false;
        }

        switch (left.Type)
        {
            case GgufValueType.String:
                return string.Equals(left.AsString(), right.AsString(), StringComparison.Ordinal);
            case GgufValueType.Array:
                IReadOnlyList<GgufValue> leftElements = left.AsArray();
                IReadOnlyList<GgufValue> rightElements = right.AsArray();
                if (leftElements.Count != rightElements.Count)
                {
                    return false;
                }

                for (int i = 0; i < leftElements.Count; i++)
                {
                    if (!ValuesEqual(leftElements[i], rightElements[i]))
                    {
                        return false;
                    }
                }

                return true;
            case GgufValueType.Float32 or GgufValueType.Float64:
                return left.AsDouble() == right.AsDouble();
            case GgufValueType.UInt64:
                return left.AsULong() == right.AsULong(); // unsigned values carry 64-bit patterns past the Int64 range, so comparing one through AsLong would refuse a shard that declares it exactly as written.

            default: // integer and bool kinds compare through their stored value.
                return left.AsLong() == right.AsLong();
        }
    }

    public void Dispose()
    {
        FileStream? stream = _stream;
        _stream = null;
        stream?.Dispose();

        if (_shards is not null)
        {
            foreach (GgufFile shard in _shards)
            {
                shard.Dispose(); // idempotent: each physical shard releases its own container handle.
            }
        }
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
            tensors.Add(ReadTensorInfo(reader, fileSize, out QuantDescriptor _));
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

    private static LogicalTensor ReadTensorInfo(BinaryReader reader, long fileSize, out QuantDescriptor descriptor)
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

        if (!GgmlTypeRegistry.TryResolve(ggmlType, out descriptor))
        {
            throw new GgufException(
                $"tensor '{name}' has unsupported ggml type {ggmlType} ({GgmlType.Name(ggmlType)})");
        }

        if (stored[0] % (uint)descriptor.BlockSize != 0)
        {
            throw new GgufException(
                $"tensor '{name}' with ggml type {ggmlType} ({descriptor.Name}) fastest dimension {stored[0]} " +
                $"is not a multiple of the {descriptor.Name} block size {descriptor.BlockSize}");
        }

        // The payload is one descriptor block per BlockSize elements: the element
        // count must divide into whole blocks and the byte length must match the
        // descriptor arithmetic exactly. A tensor whose declared payload disagrees
        // with its type geometry is a malformed file.
        bool ConsistentWithDescriptor(QuantDescriptor geometry, long sizeBytes, out long elementCount)
        {
            elementCount = 1;
            foreach (uint dim in stored)
            {
                elementCount = checked(elementCount * dim);
            }

            if (elementCount % geometry.BlockSize != 0)
            {
                return false;
            }

            try
            {
                return sizeBytes == checked(geometry.TypeSize * (elementCount / geometry.BlockSize));
            }
            catch (OverflowException)
            {
                return false;
            }
        }

        long rowBytes = (long)descriptor.TypeSize * (stored[0] / descriptor.BlockSize);
        long sizeBytes = rowBytes;
        for (int d = 1; d < stored.Length; d++)
        {
            sizeBytes = checked(sizeBytes * stored[d]);
        }

        if (!ConsistentWithDescriptor(descriptor, sizeBytes, out long elementCount))
        {
            throw new GgufException(
                $"tensor '{name}' with ggml type {ggmlType} ({descriptor.Name}) declares {elementCount} elements " +
                $"but its {sizeBytes} payload bytes do not match {descriptor.Name} geometry " +
                $"({descriptor.TypeSize} bytes per {descriptor.BlockSize}-element block)");
        }

        long payloadEnd = checked((long)offset + sizeBytes);
        if (payloadEnd > fileSize)
        {
            throw new GgufException(
                $"tensor '{name}' with ggml type {ggmlType} ({descriptor.Name}) needs {sizeBytes} payload bytes at " +
                $"offset {offset}, which runs past the end of the {fileSize}-byte file " +
                $"(expected {descriptor.TypeSize} bytes per {descriptor.BlockSize}-element block)");
        }

        int[] logical = new int[nDims];
        for (int d = 0; d < nDims; d++)
        {
            logical[d] = (int)stored[nDims - 1 - d];
        }

        return new LogicalTensor(
            name,
            new TensorShape(logical),
            descriptor.Format,
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
