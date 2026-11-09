using System.Text;

using Lair.Contracts;
using Lair.Loader;

namespace Lair.Tests;

/// <summary>
/// Synthetic GGUF v3 fixtures written byte-by-byte to temp files, parsed by
/// <see cref="GgufFile"/> and resolved through <see cref="GgufMmapWeightSource"/>.
/// </summary>
[Trait("Category", "Loader")]
public sealed class GgufLoaderTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"
    private const uint GgmlF16 = 1;
    private const uint GgmlQ4_0 = 2;
    private const uint GgmlQ5_0 = 6;
    private const uint GgmlQ8_0 = 8;
    private const uint GgmlQ2_K = 10;
    private const uint GgmlQ3_K = 11;
    private const uint GgmlQ4_K = 12;
    private const uint GgmlQ5_K = 13;
    private const uint GgmlQ6_K = 14;
    private const uint GgmlQ8_K = 15;
    private const uint GgmlIQ2_XXS = 16;

    private const int EmbedCols = 4;
    private const int EmbedRows = 3;
    private const int Q8Cols = 64;
    private const int Q8Rows = 2;

    private static readonly int[] EmbedDimsStored = [EmbedCols, EmbedRows];
    private static readonly int[] Q8DimsStored = [Q8Cols, Q8Rows];

    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
            }
        }

        _tempFiles.Clear();
    }

    [Fact]
    public void Open_ParsesVersionMetadataTensorsAndAlignment()
    {
        byte[] image = BuildFixture();
        string path = WriteTemp(image);

        using GgufFile gguf = GgufFile.Open(path);

        Assert.Equal(path, gguf.Path);
        Assert.Equal(3, gguf.Version);
        Assert.Equal(32, gguf.Alignment);
        Assert.Equal(image.Length, gguf.FileSize);

        Assert.Equal("llama", gguf.Metadata.GetString("general.architecture", ""));
        Assert.Equal("lair-fixture", gguf.Metadata.GetString("general.name", ""));
        Assert.Equal(32, gguf.Metadata.GetInt("general.alignment", 0));
        Assert.Equal("gpt2", gguf.Metadata.GetString("tokenizer.ggml.model", ""));
        Assert.Equal(151643, gguf.Metadata.GetInt("tokenizer.ggml.bos_token_id", -1));
        Assert.Equal(151645, gguf.Metadata.GetInt("tokenizer.ggml.eos_token_id", -1));

        Assert.Equal(["<unk>", "hello", "world", "!"], gguf.Metadata.GetTokens());
        Assert.Equal([1, 1, 1, 3], gguf.Metadata.GetTokenTypes());
        Assert.Equal(["h e", "e l", "l o"], gguf.Metadata.GetMerges());
        Assert.Equal("chatml", gguf.Metadata.ChatTemplate);

        Assert.Equal([10, 20, 30], gguf.Metadata.Get("test.array_int32").AsInt32Array());
        Assert.True(gguf.Metadata.GetBool("test.flag", false));
        Assert.Equal(2.5d, gguf.Metadata.Get("test.real_float32").AsDouble(), 10);

        Assert.True(gguf.Metadata.Has("general.architecture"));
        Assert.False(gguf.Metadata.Has("general.missing"));
        Assert.Equal("fallback", gguf.Metadata.GetString("general.missing", "fallback"));
        Assert.Equal(-7, gguf.Metadata.GetInt("general.missing", -7));
        Assert.False(gguf.Metadata.GetBool("general.missing", false));
        // BOOL values coerce through the numeric accessors: true reads as 1.
        Assert.True(gguf.Metadata.GetBool("test.flag", false));
        Assert.Equal(1, gguf.Metadata.GetInt("test.flag", 0));
        Assert.Throws<GgufException>(() => gguf.Metadata.Get("general.missing"));
        Assert.Throws<GgufException>(() => gguf.Metadata.GetString("test.flag", "x"));
        Assert.Throws<GgufException>(() => gguf.Metadata.GetInt("general.name", 0));
        Assert.Throws<GgufException>(() => gguf.Metadata.GetBool("general.name", false));

        Assert.Equal(2, gguf.Tensors.Tensors.Count);

        LogicalTensor embed = Assert.IsType<LogicalTensor>(gguf.Tensors.Find("tok_embeddings"));
        Assert.Equal(QuantFormat.F16, embed.Format);
        Assert.Equal([EmbedRows, EmbedCols], embed.Shape.Dims);
        Assert.Equal(12L, embed.Shape.ElementCount);
        Assert.Equal(0L, embed.Offset);
        Assert.Equal(24L, embed.SizeBytes);

        LogicalTensor q8 = Assert.IsType<LogicalTensor>(gguf.Tensors.Find("blk.0.attn_q.weight"));
        Assert.Equal(QuantFormat.Q8_0, q8.Format);
        Assert.Equal([Q8Rows, Q8Cols], q8.Shape.Dims);
        Assert.Equal(128L, q8.Shape.ElementCount);
        Assert.Equal(24L, q8.Offset);
        Assert.Equal(136L, q8.SizeBytes);

        // The fixture header is larger than GGUF's default alignment.
        Assert.True(gguf.DataStart > 32);
        Assert.Equal(0L, gguf.DataStart % 32);
        Assert.True(gguf.DataStart < gguf.FileSize);
    }

    [Fact]
    public void MmapWeightSource_Resolve_ReturnsExactWrittenBytes()
    {
        byte[] image = BuildFixture();
        string path = WriteTemp(image);

        using GgufFile gguf = GgufFile.Open(path);
        using GgufMmapWeightSource source = new(gguf);

        LogicalTensor embed = gguf.Tensors.Find("tok_embeddings")!;
        LogicalTensor q8 = gguf.Tensors.Find("blk.0.attn_q.weight")!;

        ReadOnlySpan<byte> embedSpan = source.Resolve(embed);
        ReadOnlySpan<byte> q8Span = source.Resolve(q8);

        Assert.Equal((int)embed.SizeBytes, embedSpan.Length);
        Assert.Equal((int)q8.SizeBytes, q8Span.Length);

        Assert.True(embedSpan.SequenceEqual(new ReadOnlySpan<byte>(image, (int)gguf.DataStart, (int)embed.SizeBytes)));
        Assert.True(q8Span.SequenceEqual(new ReadOnlySpan<byte>(image, (int)gguf.DataStart + (int)q8.Offset, (int)q8.SizeBytes)));

        Assert.Equal(EmbedPattern(0), embedSpan[0]);
        Assert.Equal(EmbedPattern((int)embed.SizeBytes - 1), embedSpan[^1]);
        Assert.Equal(Q8Pattern(0), q8Span[0]);
        Assert.Equal(Q8Pattern((int)q8.SizeBytes - 1), q8Span[^1]);

        source.Dispose();
        Assert.Throws<ObjectDisposedException>(() => source.Resolve(embed));
    }

    [Fact]
    public void ModelLoader_Load_ReturnsLogicalTensorTable()
    {
        byte[] image = BuildFixture();
        string path = WriteTemp(image);

        GgufModelLoader loader = new();
        LogicalTensorTable table = loader.Load(path);

        Assert.Equal(2, table.Tensors.Count);
        Assert.NotNull(table.Find("tok_embeddings"));
        Assert.NotNull(table.Find("blk.0.attn_q.weight"));
        Assert.Null(table.Find("nope"));
    }

    [Fact]
    public void Open_BadMagic_ThrowsGgufException()
    {
        byte[] image = BuildFixture();
        image[0] = (byte)'X';
        string path = WriteTemp(image);

        GgufException ex = Assert.Throws<GgufException>(() => GgufFile.Open(path));
        Assert.Contains("not a GGUF file", ex.Message);
    }

    [Fact]
    public void Open_AcceptsQ5_0Q5_KQ6_KTensorsWithCorrectRowStrides()
    {
        byte[] image = BuildQuantFixture();
        string path = WriteTemp(image);

        using GgufFile gguf = GgufFile.Open(path);
        Assert.Equal(3, gguf.Tensors.Tensors.Count);

        LogicalTensor q50 = Assert.IsType<LogicalTensor>(gguf.Tensors.Find("blk.0.attn_q.weight"));
        Assert.Equal(QuantFormat.Q5_0, q50.Format);
        Assert.Equal([2, 64], q50.Shape.Dims); // ggml stores [cols, rows]: stored [64, 2] reverses to logical [rows, cols]
        Assert.Equal(44L, q50.Shape.RowStrideBytes(QuantFormat.Q5_0)); // ceil(64/32) * 22
        Assert.Equal(0L, q50.Offset);
        Assert.Equal(88L, q50.SizeBytes);

        LogicalTensor q5k = Assert.IsType<LogicalTensor>(gguf.Tensors.Find("blk.0.attn_k.weight"));
        Assert.Equal(QuantFormat.Q5_K, q5k.Format);
        Assert.Equal([3, 256], q5k.Shape.Dims);
        Assert.Equal(176L, q5k.Shape.RowStrideBytes(QuantFormat.Q5_K)); // ceil(256/256) * 176
        Assert.Equal(96L, q5k.Offset);
        Assert.Equal(528L, q5k.SizeBytes);

        LogicalTensor q6k = Assert.IsType<LogicalTensor>(gguf.Tensors.Find("blk.0.attn_v.weight"));
        Assert.Equal(QuantFormat.Q6_K, q6k.Format);
        Assert.Equal([4, 512], q6k.Shape.Dims);
        Assert.Equal(420L, q6k.Shape.RowStrideBytes(QuantFormat.Q6_K)); // ceil(512/256) * 210 (block_q6_K is 210 bytes, no alignment padding inside a row)
        Assert.Equal(640L, q6k.Offset);
        Assert.Equal(1680L, q6k.SizeBytes);

        using GgufMmapWeightSource source = new(gguf);
        ReadOnlySpan<byte> q50Span = source.Resolve(q50);
        ReadOnlySpan<byte> q5kSpan = source.Resolve(q5k);
        ReadOnlySpan<byte> q6kSpan = source.Resolve(q6k);

        Assert.Equal(88, q50Span.Length);
        Assert.Equal(528, q5kSpan.Length);
        Assert.Equal(1680, q6kSpan.Length);

        Assert.True(q50Span.SequenceEqual(new ReadOnlySpan<byte>(image, (int)gguf.DataStart, 88)));
        Assert.True(q5kSpan.SequenceEqual(new ReadOnlySpan<byte>(image, (int)gguf.DataStart + 96, 528)));
        Assert.True(q6kSpan.SequenceEqual(new ReadOnlySpan<byte>(image, (int)gguf.DataStart + 640, 1680)));

        Assert.Equal(QuantPattern(0), q50Span[0]);
        Assert.Equal(QuantPattern(87), q50Span[^1]); // payload pattern is indexed from the data-region start: last Q5_0 byte sits at data offset 87
        Assert.Equal(QuantPattern(96), q5kSpan[0]);
        Assert.Equal(QuantPattern(640 + 1680 - 1), q6kSpan[^1]);
    }

    [Fact]
    public void Open_AcceptsQ2_KQ3_KTensorsWithCorrectRowStrides()
    {
        byte[] image = BuildQ2Q3Fixture();
        string path = WriteTemp(image);

        using GgufFile gguf = GgufFile.Open(path);
        Assert.Equal(2, gguf.Tensors.Tensors.Count);

        LogicalTensor q2k = Assert.IsType<LogicalTensor>(gguf.Tensors.Find("blk.0.attn_q.weight"));
        Assert.Equal(QuantFormat.Q2_K, q2k.Format);
        Assert.Equal([2, 256], q2k.Shape.Dims); // ggml stores [cols, rows]: stored [256, 2] reverses to logical [rows, cols]
        Assert.Equal(84L, q2k.Shape.RowStrideBytes(QuantFormat.Q2_K)); // ceil(256/256) * 84
        Assert.Equal(0L, q2k.Offset);
        Assert.Equal(168L, q2k.SizeBytes);

        LogicalTensor q3k = Assert.IsType<LogicalTensor>(gguf.Tensors.Find("blk.0.attn_k.weight"));
        Assert.Equal(QuantFormat.Q3_K, q3k.Format);
        Assert.Equal([3, 256], q3k.Shape.Dims);
        Assert.Equal(110L, q3k.Shape.RowStrideBytes(QuantFormat.Q3_K)); // ceil(256/256) * 110
        Assert.Equal(224L, q3k.Offset); // next offset after 168 bytes, aligned to general.alignment (32)
        Assert.Equal(330L, q3k.SizeBytes);

        using GgufMmapWeightSource source = new(gguf);
        ReadOnlySpan<byte> q2kSpan = source.Resolve(q2k);
        ReadOnlySpan<byte> q3kSpan = source.Resolve(q3k);

        Assert.Equal(168, q2kSpan.Length);
        Assert.Equal(330, q3kSpan.Length);

        Assert.True(q2kSpan.SequenceEqual(new ReadOnlySpan<byte>(image, (int)gguf.DataStart, 168)));
        Assert.True(q3kSpan.SequenceEqual(new ReadOnlySpan<byte>(image, (int)gguf.DataStart + 224, 330)));

        Assert.Equal(QuantPattern(0), q2kSpan[0]);
        Assert.Equal(QuantPattern(168 - 1), q2kSpan[^1]);
        Assert.Equal(QuantPattern(224), q3kSpan[0]);
        Assert.Equal(QuantPattern(224 + 330 - 1), q3kSpan[^1]);
    }

    [Fact]
    public void Open_UnsupportedTensorType_ThrowsWithTypeIdAndName()
    {
        byte[] image = BuildFixture(q8GgmlType: GgmlQ8_K);
        string path = WriteTemp(image);

        GgufException ex = Assert.Throws<GgufException>(() => GgufFile.Open(path));
        Assert.Contains("unsupported ggml type 15 (Q8_K)", ex.Message);
    }

    [Theory]
    [InlineData(0u, "F32", QuantFormat.F32, 1, 4)]
    [InlineData(1u, "F16", QuantFormat.F16, 1, 2)]
    [InlineData(6u, "Q5_0", QuantFormat.Q5_0, 32, 22)]
    [InlineData(8u, "Q8_0", QuantFormat.Q8_0, 32, 34)]
    [InlineData(10u, "Q2_K", QuantFormat.Q2_K, 256, 84)]
    [InlineData(11u, "Q3_K", QuantFormat.Q3_K, 256, 110)]
    [InlineData(12u, "Q4_K", QuantFormat.Q4_K, 256, 144)]
    [InlineData(13u, "Q5_K", QuantFormat.Q5_K, 256, 176)]
    [InlineData(14u, "Q6_K", QuantFormat.Q6_K, 256, 210)]
    public void GgmlTypeRegistry_ResolvesEverySupportedIdWithExactGeometry(
        uint id, string name, QuantFormat format, int blockSize, int typeSize)
    {
        Assert.True(GgmlTypeRegistry.TryResolve(id, out QuantDescriptor descriptor));
        Assert.Equal(id, descriptor.TypeId);
        Assert.Equal(name, descriptor.Name);
        Assert.Equal(format, descriptor.Format);
        Assert.Equal(blockSize, descriptor.BlockSize);
        Assert.Equal(typeSize, descriptor.TypeSize);
    }

    [Theory]
    [InlineData(2u, "Q4_0")]
    [InlineData(15u, "Q8_K")]
    [InlineData(16u, "IQ2_XXS")]
    public void GgmlTypeRegistry_RejectsUnsupportedIdsButNamesThem(uint id, string ggmlName)
    {
        Assert.False(GgmlTypeRegistry.TryResolve(id, out QuantDescriptor descriptor));
        Assert.Equal(default, descriptor);
        Assert.Equal(ggmlName, GgmlType.Name(id));
    }

    [Theory]
    [InlineData(GgmlQ4_0, "Q4_0")]
    [InlineData(GgmlIQ2_XXS, "IQ2_XXS")]
    public void Open_UnsupportedTensorType_RejectsWithGgmlIdAndName(uint ggmlType, string ggmlName)
    {
        byte[] image = BuildFixture(q8GgmlType: ggmlType);
        string path = WriteTemp(image);

        GgufException ex = Assert.Throws<GgufException>(() => GgufFile.Open(path));
        Assert.Contains($"unsupported ggml type {ggmlType} ({ggmlName})", ex.Message);
    }

    [Fact]
    public void Open_FastestDimensionNotBlockMultiple_RejectsNamingTensorIdAndGeometry()
    {
        // 96 is not a multiple of the Q4_K block size 256.
        byte[] image = BuildTensorFixture("blk.0.attn_q.weight", [96, 2], GgmlQ4_K, 0, 0);
        string path = WriteTemp(image);

        GgufException ex = Assert.Throws<GgufException>(() => GgufFile.Open(path));
        Assert.Contains("tensor 'blk.0.attn_q.weight'", ex.Message);
        Assert.Contains("ggml type 12 (Q4_K)", ex.Message);
        Assert.Contains("fastest dimension 96", ex.Message);
        Assert.Contains("not a multiple of the Q4_K block size 256", ex.Message);
    }

    [Fact]
    public void Open_PayloadPastEndOfFile_RejectsNamingTensorIdAndGeometry()
    {
        // A Q8_0 tensor stored [64, 2] needs 2 * ceil(64 / 32) * 34 = 136 payload
        // bytes; the declared offset puts the payload beyond the end of the file.
        byte[] image = BuildTensorFixture("blk.0.attn_q.weight", [64, 2], GgmlQ8_0, 65_536, 0);
        string path = WriteTemp(image);

        GgufException ex = Assert.Throws<GgufException>(() => GgufFile.Open(path));
        Assert.Contains("tensor 'blk.0.attn_q.weight'", ex.Message);
        Assert.Contains("ggml type 8 (Q8_0)", ex.Message);
        Assert.Contains("136 payload bytes", ex.Message);
        Assert.Contains("runs past the end", ex.Message);
        Assert.Contains("34 bytes per 32-element block", ex.Message);
    }

    [Fact]
    public void Open_TruncatedFile_ThrowsGgufException()
    {
        byte[] full = BuildFixture(); // 13 KV pairs; real_float32 is written last
        long dataStart;
        using (GgufFile parsed = GgufFile.Open(WriteTemp(full)))
        {
            dataStart = parsed.DataStart;
        }

        // Cut inside the tensor-info region: a fixed-size read hits EOF.
        byte[] headerCut = full.AsSpan(0, (int)dataStart - 32).ToArray();
        GgufException headerEx = Assert.Throws<GgufException>(() => GgufFile.Open(WriteTemp(headerCut)));
        Assert.Contains("truncated", headerEx.Message);

        // Full header intact; the Q8_0 payload is clipped mid-block.
        byte[] clipped = full.AsSpan(0, (int)dataStart + 160 - 32).ToArray();
        GgufException payloadEx = Assert.Throws<GgufException>(() => GgufFile.Open(WriteTemp(clipped)));
        Assert.Contains("does not fit", payloadEx.Message);
    }

    [Fact]
    public void Open_UnsupportedArrayElementType_ThrowsGgufException()
    {
        byte[] image = BuildFixture(arrayElementType: 13); // no such GGUF vtype
        string path = WriteTemp(image);

        GgufException ex = Assert.Throws<GgufException>(() => GgufFile.Open(path));
        Assert.Contains("unsupported element type 13", ex.Message);
    }

    [Fact]
    public void GgufValueCoercion_ReturnsTypedArrays()
    {
        byte[] image = BuildFixture();
        string path = WriteTemp(image);

        using GgufFile gguf = GgufFile.Open(path);

        Assert.Equal(13, gguf.Metadata.Count);
        Assert.Contains("general.name", gguf.Metadata.Keys);
        Assert.Equal(["<unk>", "hello", "world", "!"], gguf.Metadata.GetTokens());
        Assert.Equal([1, 1, 1, 3], gguf.Metadata.GetTokenTypes());

        GgufException arrayEx = Assert.Throws<GgufException>(() => gguf.Metadata.Get("test.flag").AsArray());
        Assert.Contains("not an array", arrayEx.Message);
    }

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-gguf-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    private static byte EmbedPattern(int i) => (byte)(i * 7 + 1);

    private static byte Q8Pattern(int i) => (byte)(255 - i * 3);

    private static byte QuantPattern(int i) => (byte)(i * 13 + 5);

    // Three quantized tensors exercising the Q5_0/Q5_K/Q6_K acceptance path: ggml-stored dims, offsets aligned to
    // general.alignment (32), and payload bytes filled to each aligned offset so Resolve spans can be compared
    // against the fixture image.
    private static byte[] BuildQuantFixture()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write(3ul); // n_tensors
        writer.Write(12ul); // n_kv — must match the number of KV pairs written below

        WriteKvString(writer, "general.architecture", "llama");
        WriteKvString(writer, "general.name", "lair-quant-fixture");
        WriteKvUint32(writer, "general.alignment", 32);
        WriteKvString(writer, "tokenizer.ggml.model", "gpt2");
        WriteKvStringArray(writer, "tokenizer.ggml.tokens", ["<unk>", "hello", "world", "!"]);
        WriteKvInt32Array(writer, "tokenizer.ggml.token_type", [1, 1, 1, 3]);
        WriteKvStringArray(writer, "tokenizer.ggml.merges", ["h e", "e l", "l o"]);
        WriteKvUint32(writer, "tokenizer.ggml.bos_token_id", 151643);
        WriteKvUint32(writer, "tokenizer.ggml.eos_token_id", 151645);
        WriteKvString(writer, "tokenizer.chat_template", "chatml");
        WriteKvInt32Array(writer, "test.array_int32", [10, 20, 30]);
        WriteKvBool(writer, "test.flag", true);

        WriteTensorInfo(writer, "blk.0.attn_q.weight", [64, 2], GgmlQ5_0, 0);
        WriteTensorInfo(writer, "blk.0.attn_k.weight", [256, 3], GgmlQ5_K, 96);
        WriteTensorInfo(writer, "blk.0.attn_v.weight", [512, 4], GgmlQ6_K, 640);

        long headerLength = stream.Position;
        long dataStart = (headerLength + 31) / 32 * 32;
        for (long i = headerLength; i < dataStart; i++)
        {
            writer.Write((byte)0);
        }

        // Payload region: 0..95 = Q5_0 (88 bytes + 8 alignment pad), 96..639 = Q5_K (528 bytes + 112 pad to
        // offset 640), 640..2319 = Q6_K (1680 bytes).
        for (int i = 0; i < 640 + 1680; i++)
        {
            writer.Write(QuantPattern(i));
        }

        writer.Flush();
        return stream.ToArray();
    }

    // Two quantized tensors exercising the Q2_K/Q3_K acceptance path: ggml-stored dims, offsets aligned to
    // general.alignment (32), and payload bytes filled to each aligned offset so Resolve spans can be compared
    // against the fixture image.
    private static byte[] BuildQ2Q3Fixture()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write(2ul); // n_tensors
        writer.Write(12ul); // n_kv — must match the number of KV pairs written below

        WriteKvString(writer, "general.architecture", "llama");
        WriteKvString(writer, "general.name", "lair-23k-fixture");
        WriteKvUint32(writer, "general.alignment", 32);
        WriteKvString(writer, "tokenizer.ggml.model", "gpt2");
        WriteKvStringArray(writer, "tokenizer.ggml.tokens", ["<unk>", "hello", "world", "!"]);
        WriteKvInt32Array(writer, "tokenizer.ggml.token_type", [1, 1, 1, 3]);
        WriteKvStringArray(writer, "tokenizer.ggml.merges", ["h e", "e l", "l o"]);
        WriteKvUint32(writer, "tokenizer.ggml.bos_token_id", 151643);
        WriteKvUint32(writer, "tokenizer.ggml.eos_token_id", 151645);
        WriteKvString(writer, "tokenizer.chat_template", "chatml");
        WriteKvInt32Array(writer, "test.array_int32", [10, 20, 30]);
        WriteKvBool(writer, "test.flag", true);

        WriteTensorInfo(writer, "blk.0.attn_q.weight", [256, 2], GgmlQ2_K, 0);
        WriteTensorInfo(writer, "blk.0.attn_k.weight", [256, 3], GgmlQ3_K, 224);

        long headerLength = stream.Position;
        long dataStart = (headerLength + 31) / 32 * 32;
        for (long i = headerLength; i < dataStart; i++)
        {
            writer.Write((byte)0);
        }

        // Payload region: 0..167 = Q2_K (168 bytes + 56 alignment pad to offset 224), 224..553 = Q3_K (330 bytes).
        for (int i = 0; i < 224 + 330; i++)
        {
            writer.Write(QuantPattern(i));
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static byte[] BuildFixture(uint q8GgmlType = GgmlQ8_0, bool includeFloat32 = true, uint arrayElementType = 5u)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write(2ul); // n_tensors
        writer.Write(13ul); // n_kv — must match the number of KV pairs written below

        WriteKvString(writer, "general.architecture", "llama");
        WriteKvString(writer, "general.name", "lair-fixture");
        WriteKvUint32(writer, "general.alignment", 32);
        WriteKvString(writer, "tokenizer.ggml.model", "gpt2");
        WriteKvStringArray(writer, "tokenizer.ggml.tokens", ["<unk>", "hello", "world", "!"]);
        WriteKvInt32Array(writer, "tokenizer.ggml.token_type", [1, 1, 1, 3]);
        WriteKvStringArray(writer, "tokenizer.ggml.merges", ["h e", "e l", "l o"]);
        WriteKvUint32(writer, "tokenizer.ggml.bos_token_id", 151643);
        WriteKvUint32(writer, "tokenizer.ggml.eos_token_id", 151645);
        WriteKvString(writer, "tokenizer.chat_template", "chatml");
        WriteKvInt32Array(writer, "test.array_int32", [10, 20, 30], arrayElementType);
        WriteKvBool(writer, "test.flag", true);
        if (includeFloat32)
        {
            // Appended last so the truncation fixture can cut exactly this pair off.
            WriteKvFloat32(writer, "test.real_float32", 2.5f);
        }

        WriteTensorInfo(writer, "tok_embeddings", EmbedDimsStored, GgmlF16, 0);
        WriteTensorInfo(writer, "blk.0.attn_q.weight", Q8DimsStored, q8GgmlType, 24);

        long headerLength = stream.Position;
        long dataStart = (headerLength + 31) / 32 * 32;
        for (long i = headerLength; i < dataStart; i++)
        {
            writer.Write((byte)0);
        }

        for (int i = 0; i < EmbedRows * EmbedCols * 2; i++)
        {
            writer.Write(EmbedPattern(i));
        }

        for (int i = 0; i < Q8Rows * Q8Cols / 32 * 34; i++)
        {
            writer.Write(Q8Pattern(i));
        }

        writer.Flush();
        return stream.ToArray();
    }

    // Single-tensor fixture with a minimal header: one tensor info, alignment 16,
    // and a payload region of the given size so a byte-length rejection can be
    // aimed at one tensor without writing a full model image.
    private static byte[] BuildTensorFixture(string tensorName, int[] storedDims, uint ggmlType, long offset, long payloadBytes)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write(1ul); // n_tensors
        writer.Write(3ul); // n_kv — must match the number of KV pairs written below

        WriteKvString(writer, "general.architecture", "llama");
        WriteKvString(writer, "general.name", "lair-tensor-fixture");
        WriteKvUint32(writer, "general.alignment", 16);

        WriteTensorInfo(writer, tensorName, storedDims, ggmlType, offset);

        long headerLength = stream.Position;
        long dataStart = (headerLength + 15) / 16 * 16;
        for (long i = headerLength; i < dataStart; i++)
        {
            writer.Write((byte)0);
        }

        for (long i = 0; i < payloadBytes; i++)
        {
            writer.Write(QuantPattern((int)i));
        }

        writer.Flush();
        return stream.ToArray();
    }

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

    private static void WriteKvBool(BinaryWriter writer, string key, bool value)
    {
        WriteGgufString(writer, key);
        writer.Write(7u); // BOOL
        writer.Write((byte)(value ? 1 : 0));
    }

    private static void WriteKvFloat32(BinaryWriter writer, string key, float value)
    {
        WriteGgufString(writer, key);
        writer.Write(6u); // FLOAT32
        writer.Write(value);
    }

    private static void WriteKvStringArray(BinaryWriter writer, string key, string[] values)
    {
        WriteGgufString(writer, key);
        writer.Write(9u); // ARRAY
        writer.Write(8u); // element type STRING
        writer.Write((ulong)values.Length);
        foreach (string value in values)
        {
            WriteGgufString(writer, value);
        }
    }

    private static void WriteKvInt32Array(BinaryWriter writer, string key, int[] values, uint elementType = 5u)
    {
        WriteGgufString(writer, key);
        writer.Write(9u); // ARRAY
        writer.Write(elementType);
        writer.Write((ulong)values.Length);
        foreach (int value in values)
        {
            writer.Write(value);
        }
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
