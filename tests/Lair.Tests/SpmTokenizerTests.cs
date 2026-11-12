using System.Text;

using Lair.Loader;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Synthetic SentencePiece fixtures (byte-by-byte, zero tensors) parsed through
/// <see cref="GgufFile"/> and driven through <see cref="GgufTokenizer"/>, plus a real-model sanity
/// check against models/gemma-2-2b-it.Q4_K_M.gguf when present. The synthetic vocabulary carries a
/// <c>tokenizer.ggml.scores</c> float array, <c>▁</c>-prefixed word-start pieces, <c>&lt;0xNN&gt;</c>
/// byte pieces, and <c>tokenizer.ggml.token_type</c> entries marking the specials.
/// </summary>
[Trait("Category", "Tokenizer")]
public sealed class SpmTokenizerTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    // Piece ids the assertions read directly: 0 <unk>, 1 <|im_start|>, 2 "hello", 3 "hel", 4 "lo",
    // 5 "▁world", 6 "▁worl", 7/8 the duplicate "ab" pair (scores -3.0 / -1.0), 9/10 the equal-score
    // "cd" pair. The byte block starts at id 11: <0xNN> for every byte value, hex-uppercase spelling.
    private const int UnknownId = 0;
    private const int ControlId = 1;
    private const int HelloId = 2;
    private const int MarkerWorldId = 5;
    private const int AbHighScoreId = 8;
    private const int CdFirstId = 9;
    private const int ByteBlockId = 11;
    private const int VocabSize = ByteBlockId + 256;

    private static readonly string[] Pieces =
    [
        "<unk>", "<|im_start|>",
        "hello", "hel", "lo", "\u2581world", "\u2581worl", "ab", "ab", "cd", "cd",
    ];
    private static readonly int[] Types = [2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    private static readonly float[] Scores = [-1000f, -1000f, -3.0f, -0.4f, -0.5f, -1.2f, -1.1f, -3.0f, -1.0f, -2.0f, -2.0f];

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
    public void Encode_PrefersLongestMatchOverPieceScores()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // "hello" scores -3.0 while "hel" (-0.4) and "lo" (-0.5) score higher: match length decides
        // first, so the score-preference shortcut must not split the word.
        Assert.Equal([HelloId], tokenizer.Encode("hello"));

        // With no single covering piece the greedy walk assembles from the shorter ones, and the
        // uncovered tail falls to the byte pieces.
        Assert.Equal([3, ByteBlockId + 0x6F], tokenizer.Encode("helo"));
    }

    [Fact]
    public void Encode_TiesOnMatchLengthResolveByHigherScore()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // The vocabulary carries the same piece twice: the higher-score entry wins the tie even
        // though it sits at the higher vocabulary index.
        Assert.Equal([AbHighScoreId], tokenizer.Encode("ab"));
        Assert.Equal([AbHighScoreId, CdFirstId], tokenizer.Encode("abcd"));
    }

    [Fact]
    public void Encode_EqualScoresResolveToLowerVocabularyIndex()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // Equal scores on equal-length matches keep the lower vocabulary index deterministically.
        Assert.Equal([CdFirstId], tokenizer.Encode("cd"));
    }

    [Fact]
    public void Encode_SpaceMapsToWordStartPiece()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // A space in the input maps to the ▁-prefixed piece, mid-sequence and at the start. Both
        // ▁ candidates match " world"; the longer one wins.
        Assert.Equal([HelloId, MarkerWorldId], tokenizer.Encode("hello world"));
        Assert.Equal([MarkerWorldId], tokenizer.Encode(" world"));

        // The ▁ piece only matches at a word boundary: the same letters without a space fall to the
        // byte pieces instead.
        Assert.Equal([ByteBlockId + 0x77, ByteBlockId + 0x6F, ByteBlockId + 0x72, ByteBlockId + 0x6C, ByteBlockId + 0x64], tokenizer.Encode("world"));
    }

    [Fact]
    public void Encode_UnrepresentableTextFallsBackToBytePieces()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // "x" has no piece: the vocabulary's <0xNN> byte pieces carry its UTF-8 bytes.
        Assert.Equal([ByteBlockId + 0x78], tokenizer.Encode("x"));

        // "é" is two UTF-8 bytes: 0xC3 0xA9.
        Assert.Equal([ByteBlockId + 0xC3, ByteBlockId + 0xA9], tokenizer.Encode("é"));

        // An uncovered space falls back to its own byte piece — not the marker's UTF-8 bytes.
        Assert.Equal([HelloId, ByteBlockId + 0x20, ByteBlockId + 0x21], tokenizer.Encode("hello !"));

        // The byte pieces decode back to the original text.
        Assert.Equal("xé", tokenizer.Decode(tokenizer.Encode("xé")));
    }

    [Fact]
    public void EncodeDecode_RoundTripsSamples()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        string[] samples =
        [
            "hello",
            "hello world",
            "world hello",
            "café",
            "xé",
            "ab cd",
            " leading",
            "trailing  ",
            "hello world!",
        ];

        foreach (string sample in samples)
        {
            IReadOnlyList<int> ids = tokenizer.Encode(sample);
            Assert.NotEmpty(ids);
            Assert.Equal(sample, tokenizer.Decode(ids));
        }
    }

    [Fact]
    public void Encode_MatchesSpecialsWholeAndInline()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // A control token is matched whole wherever it appears and never reaches piece selection.
        Assert.Equal([ControlId], tokenizer.Encode("<|im_start|>"));
        Assert.Equal([ControlId, HelloId, ControlId], tokenizer.Encode("<|im_start|>hello<|im_start|>"));

        // Specials stay intact around byte-fallback text.
        Assert.Equal([ByteBlockId + 0x61, ControlId, ByteBlockId + 0x62], tokenizer.Encode("a<|im_start|>b"));
    }

    [Fact]
    public void Decode_SpecialsHonoredByIncludeSpecial()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        IReadOnlyList<int> ids = [ControlId, HelloId, MarkerWorldId];
        Assert.Equal("hello world", tokenizer.Decode(ids));
        Assert.Equal("<|im_start|>hello world", tokenizer.Decode(ids, includeSpecial: true));
        Assert.Equal(string.Empty, tokenizer.Decode([ControlId]));
        Assert.Equal("<|im_start|>", tokenizer.Decode([ControlId], includeSpecial: true));
    }

    [Fact]
    public void Decode_LeadingMarkerDroppedAtSequenceStartKeptMidSequence()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // ▁ renders as a space except at the very start of the sequence.
        Assert.Equal("world", tokenizer.Decode([MarkerWorldId]));
        Assert.Equal("hello world", tokenizer.Decode([HelloId, MarkerWorldId]));
        Assert.Equal("worldhello", tokenizer.Decode([MarkerWorldId, HelloId]));

        // Encode maps a leading space to the marker piece, and decode drops it there: the documented
        // sequence-start behavior, matching the reference's word-boundary spelling.
        Assert.Equal("world", tokenizer.Decode(tokenizer.Encode(" world")));
    }

    [Fact]
    public void EncodeWithBos_PrependsDeclaredBosId()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        Assert.True(tokenizer.AddBosToken);
        Assert.Equal([ControlId, AbHighScoreId], tokenizer.EncodeWithBos("ab"));
        Assert.Equal([AbHighScoreId], tokenizer.Encode("ab"));
    }

    [Fact]
    public void FromGguf_UnknownModel_FailsNamingValueAndSupportedList()
    {
        byte[] image = BuildSpmImage(model: "wander22");
        GgufException ex = Assert.Throws<GgufException>(() => GgufTokenizer.FromGguf(GgufFile.Open(WriteTemp(image))));
        Assert.Contains("tokenizer.ggml.model is 'wander22'", ex.Message);
        Assert.Contains("(supported: 'llama', 'spm', 'gpt2', 'llama-bpe')", ex.Message);
    }

    [Fact]
    public void FromGguf_ModelAbsent_KeepsBpePath()
    {
        // No tokenizer.ggml.model key at all: the BPE pipeline (pre-split, byte-encode, merges) runs
        // exactly as before, and the merges key is consumed by it.
        byte[] image = BuildBpeImage(includeModelKey: false);
        using GgufFile gguf = GgufFile.Open(WriteTemp(image));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        Assert.Equal(6, tokenizer.VocabSize);

        // "a b" -> "ab" (rank 0), then "ab c" -> "abc" (rank 1).
        Assert.Equal([5], tokenizer.Encode("abc"));
        Assert.Equal([4], tokenizer.Encode("ab"));
    }

    [Fact]
    public void FromGguf_SpmFileIgnoresPreAxis()
    {
        // tokenizer.ggml.pre is BPE pre-split machinery: an SPM vocabulary loads even with a pre
        // value the BPE profile registry would reject.
        byte[] image = BuildSpmImage(model: "llama", pre: "nova2");
        using GgufFile gguf = GgufFile.Open(WriteTemp(image));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        Assert.Equal(VocabSize, tokenizer.VocabSize);
        Assert.Equal([HelloId], tokenizer.Encode("hello"));
    }

    [Fact]
    public void FromGguf_ScoresLengthMismatch_FailsNamingKeyAndCounts()
    {
        byte[] image = BuildSpmImage(scoresOverride: [-1f, -2f, -3f]);
        GgufException ex = Assert.Throws<GgufException>(() => GgufTokenizer.FromGguf(GgufFile.Open(WriteTemp(image))));
        Assert.Contains("tokenizer.ggml.scores", ex.Message);
        Assert.Contains("has 3 entries", ex.Message);
        Assert.Contains($"(expected {VocabSize})", ex.Message);
    }

    [Fact]
    public void RealModel_Gemma2_ResolvesSpmAndRoundTripsSentence()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative("models/gemma-2-2b-it.Q4_K_M.gguf");
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile gguf = GgufFile.Open(modelPath);

        // The pinned file declares the SentencePiece marker and a pre value the BPE registry does
        // not define: the successful load is the vocabulary-type seam resolving it to SPM.
        Assert.Equal("llama", gguf.Metadata.GetString("tokenizer.ggml.model", ""));
        Assert.Equal("default", gguf.Metadata.GetString("tokenizer.ggml.pre", ""));

        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        Assert.Equal(gguf.Metadata.GetTokens().Length, tokenizer.VocabSize);

        string sentence = "The quick brown fox jumps over the lazy dog.";
        IReadOnlyList<int> ids = tokenizer.Encode(sentence);
        Assert.NotEmpty(ids);
        Assert.Equal(sentence, tokenizer.Decode(ids));
    }

    private GgufTokenizer OpenSynthetic()
    {
        return GgufTokenizer.FromGguf(GgufFile.Open(WriteTemp(BuildSpmImage())));
    }

    private static byte[] BuildSpmImage(
        string model = "spm",
        string? pre = null,
        float[]? scoresOverride = null)
    {
        List<string> pieces = [.. Pieces];
        List<int> types = [.. Types];
        List<float> scores = [.. Scores];
        for (int b = 0; b < 256; b++)
        {
            pieces.Add($"<0x{b:X2}>");
            types.Add(6);
            scores.Add(-1000f);
        }

        if (scoresOverride is not null)
        {
            scores = [.. scoresOverride];
        }

        List<(string Key, uint ValueType, Action<BinaryWriter> Write)> kv =
        [
            ("general.architecture", 8u, writer => WriteStringValue(writer, "gemma2")),
            ("tokenizer.ggml.model", 8u, writer => WriteStringValue(writer, model)),
            ("tokenizer.ggml.tokens", 9u, writer => WriteStringArrayValue(writer, [.. pieces])),
            ("tokenizer.ggml.token_type", 9u, writer => WriteInt32ArrayValue(writer, [.. types])),
            ("tokenizer.ggml.scores", 9u, writer => WriteFloatArrayValue(writer, [.. scores])),
            ("tokenizer.ggml.bos_token_id", 4u, writer => writer.Write((uint)ControlId)),
            ("tokenizer.ggml.eos_token_id", 4u, writer => writer.Write((uint)UnknownId)),
            ("tokenizer.ggml.unknown_token_id", 4u, writer => writer.Write((uint)UnknownId)),
            ("tokenizer.ggml.add_bos_token", 7u, writer => writer.Write((byte)1)),
        ];
        if (pre is not null)
        {
            kv.Insert(1, ("tokenizer.ggml.pre", 8u, writer => WriteStringValue(writer, pre)));
        }

        return BuildImage(kv);
    }

    /// <summary>
    /// A minimal BPE-shaped fixture for the absent-<c>tokenizer.ggml.model</c> control: the key is
    /// omitted entirely and the merges array is present for the BPE pipeline to consume.
    /// </summary>
    private static byte[] BuildBpeImage(bool includeModelKey)
    {
        List<(string Key, uint ValueType, Action<BinaryWriter> Write)> kv =
        [
            ("general.architecture", 8u, writer => WriteStringValue(writer, "llama")),
        ];
        if (includeModelKey)
        {
            kv.Add(("tokenizer.ggml.model", 8u, writer => WriteStringValue(writer, "gpt2")));
        }

        kv.AddRange(
        [
            ("tokenizer.ggml.pre", 8u, writer => WriteStringValue(writer, "gpt2")),
            ("tokenizer.ggml.tokens", 9u, writer => WriteStringArrayValue(writer, ["<unk>", "a", "b", "c", "ab", "abc"])),
            ("tokenizer.ggml.token_type", 9u, writer => WriteInt32ArrayValue(writer, [2, 1, 1, 1, 1, 1])),
            ("tokenizer.ggml.merges", 9u, writer => WriteStringArrayValue(writer, ["a b", "ab c"])),
            ("tokenizer.ggml.bos_token_id", 4u, writer => writer.Write(1u)),
            ("tokenizer.ggml.eos_token_id", 4u, writer => writer.Write(1u)),
            ("tokenizer.ggml.unknown_token_id", 4u, writer => writer.Write(0u)),
            ("tokenizer.ggml.add_bos_token", 7u, writer => writer.Write((byte)0)),
        ]);

        return BuildImage(kv);
    }

    private static byte[] BuildImage(List<(string Key, uint ValueType, Action<BinaryWriter> Write)> kv)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write(0ul); // n_tensors
        writer.Write((ulong)kv.Count);

        foreach ((string key, uint valueType, Action<BinaryWriter> writeValue) in kv)
        {
            WriteGgufString(writer, key);
            writer.Write(valueType);
            writeValue(writer);
        }

        writer.Flush();
        return stream.ToArray();
    }

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-spm-tokenizer-{Guid.NewGuid():N}.gguf");
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

    private static void WriteStringValue(BinaryWriter writer, string value) => WriteGgufString(writer, value);

    private static void WriteStringArrayValue(BinaryWriter writer, string[] values)
    {
        writer.Write(8u); // element type STRING
        writer.Write((ulong)values.Length);
        foreach (string value in values)
        {
            WriteGgufString(writer, value);
        }
    }

    private static void WriteInt32ArrayValue(BinaryWriter writer, int[] values)
    {
        writer.Write(5u); // element type INT32
        writer.Write((ulong)values.Length);
        foreach (int value in values)
        {
            writer.Write(value);
        }
    }

    private static void WriteFloatArrayValue(BinaryWriter writer, float[] values)
    {
        writer.Write(6u); // element type FLOAT32
        writer.Write((ulong)values.Length);
        foreach (float value in values)
        {
            writer.Write(value);
        }
    }
}
