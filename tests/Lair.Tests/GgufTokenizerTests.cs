using System.Text;

using Lair.Loader;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Synthetic GGUF fixtures (byte-by-byte, zero tensors) parsed through
/// <see cref="GgufFile"/> and driven through <see cref="GgufTokenizer"/>, plus
/// real-model tests against models/SmolLM-135M-Instruct-F16.gguf when present.
/// </summary>
[Trait("Category", "Tokenizer")]
public sealed class GgufTokenizerTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

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
    public void Encode_AppliesMergesInRankOrder()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // "a b" -> "ab" (rank 0), then "ab c" -> "abc" (rank 1).
        Assert.Equal([7], tokenizer.Encode("abc"));
        Assert.Equal([6], tokenizer.Encode("ab"));
    }

    [Fact]
    public void Encode_ByteEncodesPreTokensAndMatchesVocabPieces()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // " ab" byte-encodes to "Ġab" (merge "Ġ ab").
        Assert.Equal([8], tokenizer.Encode(" ab"));

        // "a b" byte-encodes to "aĠb": no merge covers it, so the space atom stays split.
        Assert.Equal([3, 10, 4], tokenizer.Encode("a b"));

        Assert.Equal([11], tokenizer.Encode("hello"));
        Assert.Equal([12], tokenizer.Encode("world"));
        Assert.Equal([13], tokenizer.Encode("!"));
    }

    [Fact]
    public void Encode_ControlTokensBecomeSingleSpecialIdsInline()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        Assert.Equal([1], tokenizer.Encode("<|im_start|>"));
        Assert.Equal([1, 11, 2], tokenizer.Encode("<|im_start|>hello<|im_end|>"));
    }

    [Fact]
    public void Encode_UnknownCharsFallBackToByteTokens()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // "x" and "y" are not vocabulary pieces; byte control tokens cover them.
        IReadOnlyList<int> ids = tokenizer.Encode("x");
        Assert.Equal([14 + 0x78], ids);

        // "é" is two UTF-8 bytes: 0xC3 0xA9.
        Assert.Equal([14 + 0xC3, 14 + 0xA9], tokenizer.Encode("é"));

        // Fallback keeps inline special handling intact.
        Assert.Equal([14 + 0x78, 2, 14 + 0x79], tokenizer.Encode("x<|im_end|>y"));
    }

    [Fact]
    public void EncodeDecode_RoundTripsThroughByteFallback()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        string[] samples =
        [
            "abc",
            "a b",
            " ab",
            "hello world!",
            "café",
        ];

        foreach (string sample in samples)
        {
            IReadOnlyList<int> ids = tokenizer.Encode(sample);
            Assert.Equal(sample, tokenizer.Decode(ids));
        }

        // Specials are stripped by default and rendered only with includeSpecial.
        IReadOnlyList<int> withSpecial = tokenizer.Encode("a<|im_end|>b");
        Assert.Equal("ab", tokenizer.Decode(withSpecial));
        Assert.Equal("a<|im_end|>b", tokenizer.Decode(withSpecial, includeSpecial: true));
    }

    [Fact]
    public void Decode_IncludeSpecialControlsSpecialRendering()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        IReadOnlyList<int> hi = tokenizer.Encode("hi");

        Assert.Equal("<|im_start|>hi<|im_end|>", tokenizer.Decode([1, .. hi, 2], includeSpecial: true));
        Assert.Equal("hi", tokenizer.Decode([1, .. hi, 2]));
        Assert.Equal(string.Empty, tokenizer.Decode([1, 2]));
    }

    [Fact]
    public void Decode_CleansSpacesBeforePunctuation()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // "hello" + " !" byte-encode to "hello" and "Ġ!" -> rendered "hello !".
        IReadOnlyList<int> ids = tokenizer.Encode("hello !");
        Assert.Equal([11, 10, 13], ids);
        Assert.Equal("hello!", tokenizer.Decode(ids));
    }

    [Fact]
    public void EncodeWithBos_HonorsAddBosTokenFlag()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        Assert.True(tokenizer.AddBosToken);
        Assert.Equal([1, 6], tokenizer.EncodeWithBos("ab"));
        Assert.Equal([6], tokenizer.Encode("ab"));
    }

    [Fact]
    public void LlamaBpePre_GroupsDigitsInThreesAndMatchesCaseInsensitiveContractions()
    {
        using GgufFile gguf = GgufFile.Open(WriteSyntheticLlamaBpe());
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        // \p{N}{1,3}: "1234" splits as "123" + "4", so the "3 4" merge never spans
        // the group boundary and the "34" piece is unreachable. With the gpt2 splitter
        // the same input would merge to "12" + "34".
        Assert.Equal([4, 2, 3], tokenizer.Encode("1234"));

        // (?i:'s|'t|'re|'ve|'m|'ll|'d) is case-insensitive, and ignore_merges keeps
        // the whole pre-token as one id.
        Assert.Equal([7], tokenizer.Encode("'S"));
        Assert.Equal([6], tokenizer.Encode("'s"));
    }

    [Fact]
    public void IdOf_AndTokenAttrs_ExposeVocabulary()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        Assert.Equal(8, tokenizer.IdOf("Ġab"));
        Assert.Null(tokenizer.IdOf("nope"));

        Assert.Equal(270, tokenizer.VocabSize);
        Assert.Equal("<unk>", tokenizer.Tokens[0].Piece);
        Assert.Equal(2, tokenizer.Tokens[0].Type);
        Assert.Equal(3, tokenizer.Tokens[1].Type);
        Assert.Equal(6, tokenizer.Tokens[14].Type);
        Assert.Equal("<0xFF>", tokenizer.Tokens[269].Piece);
    }

    [Fact]
    public void FromGguf_UnsupportedPreTokenizer_ThrowsWithItsName()
    {
        byte[] image = BuildSyntheticImage(pre: "qwen2");
        GgufException ex = Assert.Throws<GgufException>(() => GgufTokenizer.FromGguf(GgufFile.Open(WriteTemp(image))));
        Assert.Contains("unsupported pre-tokenizer 'qwen2'", ex.Message);
    }

    [Fact]
    public void FromGguf_UnsupportedModel_ThrowsWithItsName()
    {
        byte[] image = BuildSyntheticImage(model: "llama");
        GgufException ex = Assert.Throws<GgufException>(() => GgufTokenizer.FromGguf(GgufFile.Open(WriteTemp(image))));
        Assert.Contains("tokenizer.ggml.model is 'llama'", ex.Message);
    }

    [Fact]
    public void Encode_CachesRepeatedStrings()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        IReadOnlyList<int> first = tokenizer.Encode("hello world!");
        Assert.Same(first, tokenizer.Encode("hello world!"));
        Assert.NotSame(first, tokenizer.EncodeWithBos("hello world!"));
    }

    [Fact]
    public void RealModel_GgufFileOpen_Succeeds()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile gguf = GgufFile.Open(modelPath);

        Assert.Equal(3, gguf.Version);
        Assert.Equal(272, gguf.Tensors.Tensors.Count);
        Assert.Equal(49152, gguf.Metadata.GetTokens().Length);
    }

    [Fact]
    public void RealModel_MetadataMatchesTokenizerIds()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        Assert.Equal(49152, tokenizer.VocabSize);
        Assert.Equal(gguf.Metadata.GetInt("tokenizer.ggml.bos_token_id", -1), tokenizer.BosId);
        Assert.Equal(gguf.Metadata.GetInt("tokenizer.ggml.eos_token_id", -1), tokenizer.EosId);
        Assert.Equal(gguf.Metadata.GetInt("tokenizer.ggml.padding_token_id", -1), tokenizer.PadId);
        Assert.Equal(1, tokenizer.BosId);
        Assert.Equal(2, tokenizer.EosId);
        Assert.False(tokenizer.AddBosToken);
    }

    [Fact]
    public void RealModel_EncodeDecode_RoundTripsSamples()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        string[] samples =
        [
            "Hello, world!",
            "First paragraph.\n\nSecond paragraph, with tabs\tand a trailing space.  ",
            "Unicode: émoji 😀 test — café naïve ½",
            "a   b     c",
            "Numbers 12345 and 67.",
        ];

        foreach (string sample in samples)
        {
            IReadOnlyList<int> ids = tokenizer.Encode(sample);
            Assert.NotEmpty(ids);
            Assert.Equal(sample, tokenizer.Decode(ids));
        }
    }

    [Fact]
    public void RealModel_ControlTokenEncodesToSingleId()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        int? imStart = tokenizer.IdOf("<|im_start|>");
        Assert.NotNull(imStart);
        Assert.Equal([imStart.Value], tokenizer.Encode("<|im_start|>"));
        Assert.Equal(3, tokenizer.Tokens[imStart.Value].Type);

        Assert.Equal(string.Empty, tokenizer.Decode(tokenizer.Encode("<|im_start|>")));
        Assert.Equal("<|im_start|>", tokenizer.Decode(tokenizer.Encode("<|im_start|>"), includeSpecial: true));
    }

    [Fact]
    public void RealModel_DetectsUnsupportedPreTokenizer()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile gguf = GgufFile.Open(modelPath);

        // The fixture declares tokenizer.ggml.pre = "smollm"; anything else must fail loudly.
        Assert.Equal("smollm", gguf.Metadata.GetString("tokenizer.ggml.pre", ""));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        Assert.Equal(49152, tokenizer.VocabSize);
    }

    private GgufTokenizer OpenSynthetic()
    {
        byte[] image = BuildSyntheticImage();
        return GgufTokenizer.FromGguf(GgufFile.Open(WriteTemp(image)));
    }

    private string WriteSyntheticLlamaBpe()
    {
        List<string> pieces = ["1", "2", "3", "4", "12", "34", "'s", "'S"];
        List<int> types = [1, 1, 1, 1, 1, 1, 1, 1];
        for (int b = 0; b < 256; b++)
        {
            pieces.Add($"<0x{b:X2}>");
            types.Add(6);
        }

        string[] merges = ["1 2", "3 4"];
        return WriteTemp(BuildImage(
            "gpt2",
            "llama-bpe",
            [.. pieces],
            [.. types],
            merges,
            extraKv: [("tokenizer.ggml.add_bos_token", 7u, writer => writer.Write((byte)0))]));
    }

    private byte[] BuildSyntheticImage(string model = "gpt2", string pre = "gpt2")
    {
        List<string> pieces =
        [
            "<unk>", "<|im_start|>", "<|im_end|>",
            "a", "b", "c", "ab", "abc", "Ġab", "aĠb", "Ġ", "hello", "world", "!",
        ];
        List<int> types = [2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        for (int b = 0; b < 256; b++)
        {
            pieces.Add($"<0x{b:X2}>");
            types.Add(6);
        }

        string[] merges =
        [
            "a b", "ab c", "Ġ ab",
            "h e", "he l", "hel l", "hell o",
            "w o", "wo r", "wor l", "worl d",
        ];

        return BuildImage(
            model,
            pre,
            [.. pieces],
            [.. types],
            merges,
            extraKv:
            [
                ("tokenizer.ggml.bos_token_id", 4u, writer => writer.Write(1u)),
                ("tokenizer.ggml.eos_token_id", 4u, writer => writer.Write(2u)),
                ("tokenizer.ggml.padding_token_id", 4u, writer => writer.Write(2u)),
                ("tokenizer.ggml.unknown_token_id", 4u, writer => writer.Write(0u)),
                ("tokenizer.ggml.add_bos_token", 7u, writer => writer.Write((byte)1)),
            ]);
    }

    private static byte[] BuildImage(
        string model,
        string pre,
        string[] pieces,
        int[] types,
        string[] merges,
        List<(string Key, uint ValueType, Action<BinaryWriter> Write)>? extraKv = null)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        int kvCount = 6 + (extraKv?.Count ?? 0);

        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write(0ul); // n_tensors
        writer.Write((ulong)kvCount);

        WriteKvString(writer, "general.architecture", "llama");
        WriteKvString(writer, "tokenizer.ggml.model", model);
        WriteKvString(writer, "tokenizer.ggml.pre", pre);
        WriteKvStringArray(writer, "tokenizer.ggml.tokens", pieces);
        WriteKvInt32Array(writer, "tokenizer.ggml.token_type", types);
        WriteKvStringArray(writer, "tokenizer.ggml.merges", merges);

        if (extraKv is not null)
        {
            foreach ((string key, uint valueType, Action<BinaryWriter> writeValue) in extraKv)
            {
                WriteGgufString(writer, key);
                writer.Write(valueType);
                writeValue(writer);
            }
        }

        writer.Flush();
        return stream.ToArray();
    }

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-tokenizer-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
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

    private static void WriteKvInt32Array(BinaryWriter writer, string key, int[] values)
    {
        WriteGgufString(writer, key);
        writer.Write(9u); // ARRAY
        writer.Write(5u); // element type INT32
        writer.Write((ulong)values.Length);
        foreach (int value in values)
        {
            writer.Write(value);
        }
    }
}
