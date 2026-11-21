using System.Text;

using Lair.Contracts;
using Lair.Loader;

namespace Lair.Tests;

/// <summary>
/// Synthetic split-GGUF fixtures written byte-by-byte to a temp directory, parsed by
/// <see cref="GgufFile"/> and resolved through <see cref="GgufMmapWeightSource"/>. Each shard is a real GGUF v3
/// container with its own tensor table, KV pairs, alignment padding, and data region — the same shape llama.cpp's
/// gguf-split produces: per-shard offsets are relative to each shard's own data start, continuation shards carry only
/// split identity keys, and <c>split.tensors.count</c> records the global count. Payload patterns differ per shard so a
/// tensor mapped onto the wrong physical file is caught byte-for-byte even when two shards declare tensors at the same
/// relative offsets. Ordinary (non-split-shaped) files must stay single-file models.
/// </summary>
[Trait("Category", "Loader")]
public sealed class GgufSplitLoaderTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"
    private const uint GgmlF16 = 1;

    private const string ModelPrefix = "lair-split-model";

    private readonly string _directory = Path.Combine(Path.GetTempPath(), $"lair-split-{Guid.NewGuid():N}");

    public GgufSplitLoaderTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        try
        {
            foreach (string file in Directory.GetFiles(_directory))
            {
                File.Delete(file);
            }

            Directory.Delete(_directory, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    // ---------------------------------------------------------------- name-shape recognition

    [Theory]
    // The exact llama.cpp split shape is recognized; anything else stays an ordinary file. Prefixes may contain dashes.
    [InlineData("lair-split-model-00001-of-00002.gguf", true, "lair-split-model", ".gguf", 1, 2)]
    [InlineData("model-q4_k-split-00003-of-00006.bin", true, "model-q4_k-split", ".bin", 3, 6)]
    [InlineData("model-00002-of-00002", true, "model", "", 2, 2)] // extensionless name still carries the split shape
    [InlineData("lair-split-model.gguf", false, "", "", 0, 0)]
    [InlineData("model-1-of-2.gguf", false, "", "", 0, 0)] // shard numbers are exactly five digits
    [InlineData("model-00001-of-2.gguf", false, "", "", 0, 0)]
    [InlineData("model-0000a-of-00002.gguf", false, "", "", 0, 0)] // shard numbers are digits only
    [InlineData("x-of-y.gguf", false, "", "", 0, 0)]
    public void GgufShardNaming_TryParse_RecognizesOnlyTheSplitShape(
        string fileName, bool valid, string prefix, string extension, int index, int total)
    {
        if (!valid)
        {
            Assert.False(GgufShardNaming.TryParse(fileName, out _, out _, out _, out _));
            return;
        }

        Assert.True(GgufShardNaming.TryParse(fileName, out string actualPrefix, out string actualExtension, out int actualIndex, out int actualTotal));
        Assert.Equal(prefix, actualPrefix);
        Assert.Equal(extension, actualExtension);
        Assert.Equal(index, actualIndex);
        Assert.Equal(total, actualTotal);
    }

    [Fact]
    public void GgufShardNaming_ShardFileNames_EnumeratesTheSetInAscendingOrder()
    {
        string[] names = [.. GgufShardNaming.ShardFileNames("model", ".gguf", 3)];
        Assert.Equal(["model-00001-of-00003.gguf", "model-00002-of-00003.gguf", "model-00003-of-00003.gguf"], names);
    }

    // ---------------------------------------------------------------- assembled split from either entry point

    [Fact]
    public void Open_FirstShard_AssemblesSplitWithMergedMetadataAndUnionTable()
    {
        SplitFixture fixture = WritePair();

        using GgufFile model = GgufFile.Open(fixture.First);

        Assert.NotNull(model.Shards);
        IReadOnlyList<GgufFile> shards = model.Shards!;
        Assert.Equal(2, shards.Count);
        Assert.Equal(Path.GetFullPath(fixture.First), Path.GetFullPath(shards[0].Path)); // discovered in ascending shard order.
        Assert.Equal(Path.GetFullPath(fixture.Second), Path.GetFullPath(shards[1].Path));

        // Each physical view keeps its own container facts: version/alignment come from the primary, data regions and lengths are per file.
        Assert.Equal(3, model.Version);
        Assert.Equal(32, model.Alignment);
        Assert.Equal(new FileInfo(fixture.First).Length, model.FileSize); // assembled FileSize is the primary shard's length.
        Assert.True(model.DataStart < new FileInfo(fixture.First).Length);

        Assert.Equal(3, shards[0].Version);
        Assert.Equal(new FileInfo(fixture.First).Length, shards[0].FileSize); // each shard reports its own real length.
        Assert.True(shards[0].DataStart < shards[0].FileSize && shards[0].DataStart % 32 == 0);
        Assert.Equal(3, shards[1].Version);
        Assert.Equal(new FileInfo(fixture.Second).Length, shards[1].FileSize);
        Assert.True(shards[1].DataStart < shards[1].FileSize && shards[1].DataStart % 32 == 0);

        // Merged metadata: the primary shard's model identity plus one agreed split identity; only `split.no` differs across files (each records its own position).
        Assert.Equal("split-test", model.Metadata.GetString("general.architecture", ""));
        Assert.Equal(2, model.Metadata.GetInt(GgufSplitKeys.Count, -1));
        Assert.Equal(4, model.Metadata.GetInt(GgufSplitKeys.TensorsCount, -1));
        Assert.Equal(0, model.Metadata.GetInt(GgufSplitKeys.No, -1)); // primary wins the merge.

        // Shard 2 is a continuation header: split identity only, no model metadata — exactly the shape gguf-split writes.
        Assert.False(shards[1].Metadata.Has("general.architecture"));
        Assert.Equal(1, shards[1].Metadata.GetInt(GgufSplitKeys.No, -1));

        // The union table is the disjoint pair of shard tables, and offsets stay relative to each tensor's own shard data region.
        Assert.Equal(4, model.Tensors.Tensors.Count);
        LogicalTensor embed = model.Tensors.Find("tok.embeddings")!;
        LogicalTensor headB = model.Tensors.Find("tok.head.weight")!; // declared only by shard 2 — present in the assembled table.
        Assert.Equal(QuantFormat.F16, embed.Format);
        Assert.Equal([3, 4], embed.Shape.Dims); // stored [4, 3] reverses to logical [rows, cols].
        Assert.Equal(0L, embed.Offset);
        Assert.Equal(24L, embed.SizeBytes);
        Assert.Equal(32L, headB.Offset); // shard-relative: offset into shard 2's data region.

        Assert.Single(shards[1].Tensors.Tensors, t => t.Name == "tok.head.weight"); // shard 2's own table keeps its declaration.
        Assert.Equal(2, shards[0].Tensors.Tensors.Count); // each physical header keeps its own table too.
    }

    [Fact]
    public void Open_LastShard_AssemblesTheSameModel()
    {
        SplitFixture fixture = WritePair();

        using GgufFile fromLast = GgufFile.Open(fixture.Second);

        Assert.NotNull(fromLast.Shards);
        IReadOnlyList<GgufFile> shards = fromLast.Shards!;
        Assert.Equal(2, shards.Count);
        Assert.Equal(Path.GetFullPath(fixture.First), Path.GetFullPath(shards[0].Path)); // the set is ordered regardless of entry point.
        Assert.Equal(Path.GetFullPath(fixture.Second), Path.GetFullPath(shards[1].Path));

        Assert.Equal("split-test", fromLast.Metadata.GetString("general.architecture", "")); // primary metadata wins even when opened through shard 2.
        Assert.Equal(4, fromLast.Tensors.Tensors.Count);
        Assert.NotNull(fromLast.Tensors.Find("tok.embeddings"));
        Assert.NotNull(fromLast.Tensors.Find("blk.b.weight"));

        using GgufFile fromFirst = GgufFile.Open(fixture.First);
        Assert.Equal(fromFirst.Tensors.Tensors.Select(t => t.Name), fromLast.Tensors.Tensors.Select(t => t.Name)); // same logical model either way.
    }

    [Fact]
    public void ModelLoader_Load_SplitPath_ReturnsUnionTensorTable()
    {
        SplitFixture fixture = WritePair();

        LogicalTensorTable table = new GgufModelLoader().Load(fixture.First);

        Assert.Equal(4, table.Tensors.Count);
        Assert.NotNull(table.Find("tok.embeddings")); // shard 1 tensor
        Assert.NotNull(table.Find("blk.b.weight"));   // shard 2 tensor
    }

    [Fact]
    public void MmapWeightSource_SplitModel_ResolvesEachTensorFromItsOwnShardDataRegion()
    {
        SplitFixture fixture = WritePair();

        using GgufFile model = GgufFile.Open(fixture.First);
        Assert.NotNull(model.Shards);
        GgufFile shardA = model.Shards![0];
        GgufFile shardB = model.Shards![1];

        using GgufMmapWeightSource source = new(model);

        LogicalTensor embedTensor = model.Tensors.Find("tok.embeddings")!;
        LogicalTensor blkBTensor = model.Tensors.Find("blk.b.weight")!;

        // "tok.embeddings" (shard 1, relative offset 0) and "blk.b.weight" (shard 2, same relative offset 0): the mapping must follow tensor names to their declaring shard's file — not table order or first-file-wins.
        ReadOnlySpan<byte> embed = source.Resolve(embedTensor);
        ReadOnlySpan<byte> blkB = source.Resolve(blkBTensor);

        Assert.Equal(24, embed.Length);
        Assert.Equal(32, blkB.Length);

        // Byte-for-byte against each shard's own written image at that shard's data start. Different payload patterns per fixture make a wrong-shard mapping fail here.
        Assert.True(embed.SequenceEqual(new ReadOnlySpan<byte>(fixture.FirstImage, (int)shardA.DataStart, 24)));
        Assert.True(blkB.SequenceEqual(new ReadOnlySpan<byte>(fixture.SecondImage, (int)shardB.DataStart, 32)));

        // Same relative offset in both files: the resolved bytes must differ — shard 1's pattern at its offset 0 is not shard 2's.
        Assert.False(embed.SequenceEqual(blkB[..embed.Length]));

        ReadOnlySpan<byte> blkA = source.Resolve(model.Tensors.Find("blk.a.weight")!); // shard 1, relative offset 32 (past the first tensor).
        Assert.Equal(32, blkA.Length);
        Assert.True(blkA.SequenceEqual(new ReadOnlySpan<byte>(fixture.FirstImage, (int)shardA.DataStart + 32, 32)));

        ReadOnlySpan<byte> headB = source.Resolve(model.Tensors.Find("tok.head.weight")!); // shard 2, relative offset 32.
        Assert.Equal(8, headB.Length);
        Assert.True(headB.SequenceEqual(new ReadOnlySpan<byte>(fixture.SecondImage, (int)shardB.DataStart + 32, 8)));

        source.Dispose();
        Assert.Throws<ObjectDisposedException>(() => source.Resolve(embedTensor)); // spans die with the source's shard mappings.
    }

    // ---------------------------------------------------------------- refusals for broken sets

    [Fact]
    public void Open_MissingSibling_ThrowsNamingTheMissingShardFile()
    {
        SplitFixture fixture = WritePair();
        File.Delete(fixture.Second);

        GgufException fromFirst = Assert.Throws<GgufException>(() => GgufFile.Open(fixture.First));
        Assert.Contains("lair-split-model-00002-of-00002.gguf", fromFirst.Message); // the missing sibling is named.
        Assert.Contains(_directory, fromFirst.Message);

        SplitFixture partial = WritePair(); // fresh directory-safe pair for the other entry point
        File.Delete(partial.First);
        GgufException fromLast = Assert.Throws<GgufException>(() => GgufFile.Open(partial.Second));
        Assert.Contains("lair-split-model-00001-of-00002.gguf", fromLast.Message); // opening through shard 2 names the missing first sibling.
    }

    [Fact]
    public void Open_SplitCountDisagreesWithShardNumber_ThrowsNamingKeyAndFiles()
    {
        byte[] first = BuildShard(no: 0, count: 2, tensorsDeclared: 4, architecture: "split-test", seed: 1);
        // Shard 2's name says -of-00002 but its metadata claims a three-shard split.
        byte[] second = BuildShard(
            no: 1, count: 3, tensorsDeclared: 4, architecture: null, seed: 2,
            ("blk.b.weight", [8, 2], 0), ("tok.head.weight", [4, 1], 32));

        string firstPath = WriteNamed("lair-split-model-00001-of-00002.gguf", first);
        string secondPath = WriteNamed("lair-split-model-00002-of-00002.gguf", second);

        GgufException ex = Assert.Throws<GgufException>(() => GgufFile.Open(firstPath));
        Assert.Contains(GgufSplitKeys.Count, ex.Message); // the key that disagrees is named.
        Assert.Contains(Path.GetFullPath(secondPath), ex.Message); // and so is the sibling carrying it.
    }

    [Fact]
    public void Open_DeclaredTensorCountDisagreesWithTables_ThrowsNamingKeyAndCounts()
    {
        byte[] first = BuildShard(no: 0, count: 2, tensorsDeclared: 3, architecture: "split-test", seed: 1); // headers claim 3...
        byte[] second = BuildShard(
            no: 1, count: 2, tensorsDeclared: 3, architecture: null, seed: 2,
            ("blk.b.weight", [8, 2], 0), ("tok.head.weight", [4, 1], 32)); // ...but the two tables assemble 4.

        string firstPath = WriteNamed("lair-split-model-00001-of-00002.gguf", first);
        string secondPath = WriteNamed("lair-split-model-00002-of-00002.gguf", second);

        GgufException ex = Assert.Throws<GgufException>(() => GgufFile.Open(firstPath));
        Assert.Contains(GgufSplitKeys.TensorsCount, ex.Message);
        Assert.Contains("assemble 4", ex.Message); // the assembled count is reported against the declared one.
    }

    [Fact]
    public void Open_DuplicateTensorAcrossShards_ThrowsRequiringDisjointTables()
    {
        byte[] first = BuildShard(no: 0, count: 2, tensorsDeclared: 4, architecture: "split-test", seed: 1); // declares blk.a.weight + tok.embeddings
        byte[] second = BuildShard(
            no: 1, count: 2, tensorsDeclared: 4, architecture: null, seed: 2,
            ("blk.b.weight", [8, 2], 0), ("blk.a.weight", [8, 2], 32)); // re-declares a shard-1 tensor.

        string firstPath = WriteNamed("lair-split-model-00001-of-00002.gguf", first);
        string secondPath = WriteNamed("lair-split-model-00002-of-00002.gguf", second);

        GgufException ex = Assert.Throws<GgufException>(() => GgufFile.Open(firstPath));
        Assert.Contains("blk.a.weight", ex.Message); // the duplicated tensor is named...
        Assert.Contains("disjoint", ex.Message);     // ...with the disjoint-tables rule stated.
    }

    [Fact]
    public void Open_SharedModelMetadataKeyDisagrees_ThrowsNamingKeyAndBothShards()
    {
        byte[] first = BuildShard(no: 0, count: 2, tensorsDeclared: 4, architecture: "split-test", seed: 1);
        byte[] second = BuildShard( // continuation shard that also claims a different architecture.
            no: 1, count: 2, tensorsDeclared: 4, architecture: "other-arch", seed: 2,
            ("blk.b.weight", [8, 2], 0), ("tok.head.weight", [4, 1], 32));

        string firstPath = WriteNamed("lair-split-model-00001-of-00002.gguf", first);
        string secondPath = WriteNamed("lair-split-model-00002-of-00002.gguf", second);

        GgufException ex = Assert.Throws<GgufException>(() => GgufFile.Open(firstPath));
        Assert.Contains("'general.architecture'", ex.Message); // the shared key that disagrees is named...
        Assert.Contains(Path.GetFullPath(secondPath), ex.Message); // ...with both disagreeing files.

        // `split.no` differs across shards by design (each records its own position) — the happy-path tests prove it never triggers this refusal.
    }

    [Fact]
    public void Open_OrdinaryNamesStaySingleFileModels()
    {
        byte[] single = BuildShard(no: 0, count: 2, tensorsDeclared: 4, architecture: "split-test", seed: 1); // valid GGUF whose NAME is not split-shaped.

        using GgufFile ordinary = GgufFile.Open(WriteNamed("lair-split-model-00001.gguf", single));
        Assert.Null(ordinary.Shards); // a digit-bearing but non-split name stays exactly the historical single-file behavior.

        using GgufFile nearMiss = GgufFile.Open(WriteNamed("lair-of-x.gguf", single));
        Assert.Null(nearMiss.Shards); // "-of-" without five-digit shard numbers is not a split shape either.

        Assert.Equal(2, ordinary.Tensors.Tensors.Count); // the file's own table only — no assembly happened.
    }

    [Fact]
    public void OpenSingle_SplitMember_InspectsThatFileAloneWithoutAssembly()
    {
        SplitFixture fixture = WritePair();

        using GgufFile shardOnly = GgufFile.OpenSingle(fixture.Second); // per-file header inspection, as ModelFetcher does while shards are still landing.
        Assert.Null(shardOnly.Shards);
        Assert.Equal(2, shardOnly.Tensors.Tensors.Count); // only this file's declarations...
        Assert.False(shardOnly.Metadata.Has("general.architecture")); // ...and none of the primary shard's model metadata.
        Assert.Equal(4, shardOnly.Metadata.GetInt(GgufSplitKeys.TensorsCount, -1));

        using GgufFile assembled = GgufFile.Open(fixture.Second);
        Assert.NotNull(assembled.Shards);
    }

    // ---------------------------------------------------------------- fixtures

    private readonly record struct SplitFixture(string First, string Second, byte[] FirstImage, byte[] SecondImage);

    /// <summary>Writes the canonical healthy pair: shard 1 carries model metadata + split identity and two tensors;
    /// shard 2 is a continuation header (split identity only) declaring its own two tensors at overlapping relative
    /// offsets. Payload patterns differ per shard.</summary>
    private SplitFixture WritePair()
    {
        byte[] first = BuildShard(no: 0, count: 2, tensorsDeclared: 4, architecture: "split-test", seed: 1); // ("tok.embeddings" @0, "blk.a.weight" @32)
        byte[] second = BuildShard(
            no: 1, count: 2, tensorsDeclared: 4, architecture: null, seed: 2,
            ("blk.b.weight", [8, 2], 0), ("tok.head.weight", [4, 1], 32));

        return new SplitFixture(
            WriteNamed("lair-split-model-00001-of-00002.gguf", first),
            WriteNamed("lair-split-model-00002-of-00002.gguf", second),
            first,
            second);
    }

    private string WriteNamed(string fileName, byte[] image)
    {
        string path = Path.Combine(_directory, fileName);
        File.WriteAllBytes(path, image);
        return path;
    }

    // One physical shard: a full GGUF v3 container with its own KV pairs (alignment + split identity [+ architecture]),
    // tensor infos at the given relative offsets, 32-byte alignment padding, and a payload region filled to the last
    // tensor's end using this shard's pattern seed. `tensors` defaults to ("tok.embeddings" @0 F16 [4,3], "blk.a.weight" @32 F16 [8,2]).
    private static byte[] BuildShard(
        int no,
        int count,
        long tensorsDeclared,
        string? architecture,
        byte seed,
        params (string Name, int[] StoredDims, long Offset)[] tensors)
    {
        if (tensors.Length == 0)
        {
            tensors = [("tok.embeddings", [4, 3], 0L), ("blk.a.weight", [8, 2], 32L)];
        }

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        int kvCount = architecture is null ? 4 : 5; // alignment + split.no/split.count/tensors.count (+ general.architecture)

        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write((ulong)tensors.Length); // n_tensors — this shard's own table only.
        writer.Write((ulong)kvCount); // n_kv

        WriteKvUint32(writer, "general.alignment", 32);
        if (architecture is not null)
        {
            WriteKvString(writer, "general.architecture", architecture);
        }

        WriteKvUint32(writer, GgufSplitKeys.No, (uint)no); // zero-based position, as every llama.cpp writer emits it.
        WriteKvUint32(writer, GgufSplitKeys.Count, (uint)count);
        WriteKvUint64(writer, GgufSplitKeys.TensorsCount, (ulong)tensorsDeclared);

        foreach ((string name, int[] storedDims, long offset) in tensors)
        {
            WriteTensorInfo(writer, name, storedDims, GgmlF16, offset);
        }

        long headerLength = stream.Position;
        long dataStart = (headerLength + 31) / 32 * 32;
        for (long i = headerLength; i < dataStart; i++)
        {
            writer.Write((byte)0);
        }

        // Payload region: filled through the end of the last tensor so every declared offset lands on pattern bytes.
        long payloadEnd = tensors.Max(t => t.Offset + StoredElementCount(t.StoredDims) * 2L);
        for (long i = 0; i < payloadEnd; i++)
        {
            writer.Write(ShardPattern(seed, i));
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static long StoredElementCount(int[] storedDims)
    {
        long elements = 1;
        foreach (int dim in storedDims)
        {
            elements *= dim;
        }

        return elements;
    }

    // Distinct per shard: same relative offset yields different bytes, so a tensor resolved from the wrong file fails byte comparison.
    private static byte ShardPattern(byte seed, long i) => (byte)(i * 7 + seed * 91 + seed);

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private static void WriteKvString(BinaryWriter writer, string key, string value)
    {
        WriteGgufString(writer, key);
        writer.Write(8u); // STRING
        WriteGgufString(writer, value);
    }

    private static void WriteKvUint32(BinaryWriter writer, string key, uint value)
    {
        WriteGgufString(writer, key);
        writer.Write(4u); // UINT32
        writer.Write(value);
    }

    private static void WriteKvUint64(BinaryWriter writer, string key, ulong value)
    {
        WriteGgufString(writer, key);
        writer.Write(10u); // UINT64 (the GGUF vtype id — 1 is INT8)
        writer.Write(value);
    }

    private static void WriteTensorInfo(BinaryWriter writer, string name, int[] storedDims, uint ggmlType, long offset)
    {
        WriteGgufString(writer, name);
        writer.Write((uint)storedDims.Length);
        foreach (int dim in storedDims)
        {
            writer.Write((ulong)dim);
        }

        writer.Write(ggmlType);
        writer.Write((ulong)offset);
    }
}
