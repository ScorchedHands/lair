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
    private const uint GgmlQ8_0 = 8;
    private const uint GgmlQ5_0 = 6;

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
    public void Open_UnsupportedTensorType_ThrowsWithTypeIdAndName()
    {
        byte[] image = BuildFixture(q8GgmlType: GgmlQ5_0);
        string path = WriteTemp(image);

        GgufException ex = Assert.Throws<GgufException>(() => GgufFile.Open(path));
        Assert.Contains("unsupported ggml type 6 (Q5_0)", ex.Message);
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
