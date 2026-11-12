using System.Text;

using Lair.Loader;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Synthetic SentencePiece fixtures (byte-by-byte, zero tensors) parsed through
/// <see cref="GgufFile"/> and driven through <see cref="GgufTokenizer"/>, plus a real-model sanity
/// check against models/gemma-2-2b-it.Q4_K_M.gguf when present. The synthetic vocabulary carries a
/// <c>tokenizer.ggml.scores</c> float array, boundary-spelled pieces with the <c>▁</c> word-start marker —
/// including every chain node needed to assemble them and the single-marker piece itself — duplicate bare-form
/// entries, and <c>&lt;0xNN&gt;</c> byte pieces marked through <c>tokenizer.ggml.token_type</c>. Expectations mirror
/// what the pinned b12032 oracle does on real vocabularies: a sequence starts at a word boundary so its first
/// words tokenize in marker-prefixed forms, bare entries are reached only mid-run, and adjacent pieces compete
/// through score-ordered merging. Each changed expectation names the fixture case that justifies it.
/// </summary>
[Trait("Category", "Tokenizer")]
public sealed class SpmTokenizerTests : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF"

    // Piece ids the assertions read directly: 0 <unk>, 1 <|im_start|>. The bare-form pair is duplicated at 2/3
    // (scores -3.0 / -1.0) and the equal-score pair at 4/5 — the reference's text lookup resolves each merged text to its last vocabulary entry, so mid-run merges name ids 3 and 5 while entries 2 and 4 stay unreachable. Boundary chains carry every prefix node: world through "▁w"/"▁wo"/"▁wor"/"▁worl" up to "▁world", hello through "▁h"/"▁he"/"▁hel"/"▁hell" (including the doubled l) up to "▁hello", ab and cd each through their single-letter chain head, plus the single-marker piece itself. The byte block starts at id 21: <0xNN> for every byte value, hex-uppercase spelling.
    private const int UnknownId = 0;
    private const int ControlId = 1;

    // Byte-piece ids the assertions read directly, spelled as offsets into the vocabulary's byte block.
    private static int ByteOf(int byteValue) => ByteBlockId + byteValue;

    private static readonly string[] Pieces =
    [
        "<unk>", "<|im_start|>",
        "ab", "ab", "cd", "cd",
        "\u2581world", "\u2581worl", "\u2581w", "\u2581wo", "\u2581wor",
        "\u2581h", "\u2581he", "\u2581hel", "\u2581hell", "\u2581hello",
        "\u2581a", "\u2581ab", "\u2581c", "\u2581cd", "\u2581",
    ];
    private static readonly int[] Types = [2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    private static readonly float[] Scores = [-1000f, -1000f, -3.0f, -1.0f, -2.0f, -2.0f, -1.2f, -1.1f, -1.4f, -1.3f, -1.15f, -0.7f, -0.65f, -1.5f, -1.45f, -2.8f, -0.9f, -0.85f, -0.95f, -0.9f, -0.35f];

    private const int ByteBlockId = 21;
    private const int VocabSize = ByteBlockId + 256;

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
    public void Encode_PrependsWordBoundaryAtSequenceStart()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // A sequence starts at a word boundary, so the first words tokenize in their marker-prefixed forms: "hello" walks its chain to the full boundary piece. The pinned phi-3 oracle shows this directly — case 'hello_world' resolves Hello to 15043 ("▁Hello"), where LAIR emitted the bare entry before.
        Assert.Equal([15], tokenizer.Encode("hello"));

        // A leading space becomes a second boundary spelling: the single-marker piece plus the word's own marker-prefixed piece, as phi-3 case 'leading_space' pins [29871 ("▁"), 3186 ("▁world")].
        Assert.Equal([20, 6], tokenizer.Encode(" world"));

        // The boundary re-opens after a special token: the next run is prefixed again.
        Assert.Equal([ControlId, 15], tokenizer.Encode("<|im_start|>hello"));
    }

    [Fact]
    public void Encode_MergesAdjacentPiecesInScoreOrder()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // Mid-run bare forms are reached by merging adjacent symbols, and the merged text names its piece's last vocabulary entry (id 3 here). phi-3 case 'hello_world' shows why this matters for real vocabularies — a bare-form entry exists for Hello in that vocabulary too; only mid-run contexts reach it.
        Assert.Equal([20, ByteOf(0x78), 3], tokenizer.Encode("xab"));

        // Equal scores keep the same last-entry resolution deterministically: "cd" names id 5 here.
        Assert.Equal([20, ByteOf(0x7A), 5], tokenizer.Encode("zcd"));

        // A higher-score merge candidate is applied before a lower-scoring one that overlaps it; after a special token the run starts at a boundary again and "ab" completes through its marker-prefixed chain instead. phi-3 case 'repeated_words' pins [278 ×10] — every word after each separator resolves to the same boundary piece, where LAIR's first piece used to differ.
        Assert.Equal([ControlId, 17], tokenizer.Encode("<|im_start|>ab"));
    }

    [Fact]
    public void Encode_ResolvesDuplicatePiecesToLastVocabularyEntry()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // The vocabulary carries "ab" twice: the text lookup names its last entry regardless of which score is higher, and id 2 stays unreachable. At a boundary position the marker-prefixed form completes through its own chain head and wins for the whole run — phi-3 'leading_space' pins that split shape ([29871 ("▁"), 3186 ("▁world")]).
        Assert.Equal([17], tokenizer.Encode("ab"));

        // Equal scores keep the same last-entry resolution: "cd" resolves to id 5 at a boundary position too. phi-3 'hello_world' shows both forms of one word coexisting in a real vocabulary (bare Hello and marker-prefixed ▁Hello), with only the spelled form reachable where the boundary stands.
        Assert.Equal([19], tokenizer.Encode("cd"));
    }

    [Fact]
    public void Encode_SpaceMapsToWordStartPiece()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // Mid-sequence spaces become marker spellings and re-open boundaries: the world chain assembles to its longest piece. The pinned phi-3 'hello_world' case pins exactly this shape for Hello, world! — [15043 ("▁Hello"), 29892 (","), 3186 ("▁world")].
        Assert.Equal([15, ByteOf(0x2C), 6], tokenizer.Encode("hello, world"));

        // The marker only matches at a word boundary: the same letters mid-run fall to their byte pieces. phi-3 'multibyte' pins uncovered codepoints falling back per-byte after spelled pieces (ids 229/155/152).
        Assert.Equal([20, ByteOf(0x7A), ByteOf(0x7B)], tokenizer.Encode("z{"));

        // The ▁-prefixed chain nodes never match the bare letters: at a boundary position only the spelled forms complete. phi-3 'digits' shows the same split — marker-prefixed pieces name every word there and nothing merges across positions without them.
        Assert.Equal([20, ByteOf(0x7A), 5], tokenizer.Encode("zcd"));
    }

    [Fact]
    public void Encode_UnrepresentableTextFallsBackToBytePieces()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // "x" has no piece: the vocabulary's <0xNN> byte pieces carry its UTF-8 bytes. The prepended boundary marker is a spelled piece, not a fallback byte — phi-3 'single_space' pins [259] for " ", where that vocabulary carries the combined two-marker run instead of this one's single-marker piece.
        Assert.Equal([20, ByteOf(0x78)], tokenizer.Encode("x"));

        // "é" is two UTF-8 bytes: 0xC3 0xA9 — matching the pinned phi-3 'multibyte' case where café opens with spelled pieces and byte fallback carries its accented tail.
        Assert.Equal([20, ByteOf(0xC3), ByteOf(0xA9)], tokenizer.Encode("é"));

        // A space at a boundary position is spelled as the marker piece, never left as its raw byte: phi-3 'leading_space_punct' pins [29871 ("▁"), 3186 ("▁world")] for " world!".
        Assert.Equal([20, ByteOf(0x7A), ByteOf(0xC3), ByteOf(0xA9)], tokenizer.Encode("zé"));

        // The byte pieces decode back to the original text.
        Assert.Equal("xé", tokenizer.Decode(tokenizer.Encode("xé")));
    }

    [Fact]
    public void Encode_WhitespaceOnlyTextUsesBoundarySpellings()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // An input that is only whitespace still starts at a boundary and its space becomes a marker spelling; the single-marker piece carries each. phi-3 'single_space' pins [259] for " " — that vocabulary happens to carry the combined two-marker run, while this one spells it as two singles.
        Assert.Equal([20, 20], tokenizer.Encode(" "));

        // Tabs and newlines are not marker spellings: they stay verbatim and fall back to byte pieces here because no piece names them, mirroring how phi-3 'newline_tab' keeps its control characters in separate tokens (ids 12/13).
        Assert.Equal([20, ByteOf(0x0A), ByteOf(0x7B)], tokenizer.Encode("\n{"));

        // Trailing whitespace after a word spells another boundary: the marker piece follows the completed chain. phi-3 'long_prompt' ends on spelled pieces for its trailing "Answer:" ([22550, 29901]).
        Assert.Equal([15, 20], tokenizer.Encode("hello "));
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

        // A control token is matched whole wherever it appears and never reaches piece selection. The text on both sides re-opens a word boundary, so each side tokens in its marker-prefixed form.
        Assert.Equal([ControlId], tokenizer.Encode("<|im_start|>"));
        Assert.Equal([ControlId, 15, ControlId], tokenizer.Encode("<|im_start|>hello<|im_start|>"));

        // Specials stay intact around byte-fallback text: the run before a special keeps its sequence-start spelling ("a" completes through its boundary chain head) and the run after it re-opens one (the trailing space spells as marker pieces). phi-3 'long_prompt' pins specials never reaching piece selection — every ChatML control in that prompt stays outside the id stream when add_special=false, while plain-text runs still start at boundaries.
        Assert.Equal([16, ControlId, 20, 20], tokenizer.Encode("a<|im_start|> "));

        // A run after a special tokens through its own boundary spelling: the prepended marker stands as its piece and an uncovered letter falls byte-wise.
        Assert.Equal([ControlId, 20, ByteOf(0x62)], tokenizer.Encode("<|im_start|>b"));
    }

    [Fact]
    public void Decode_SpecialsHonoredByIncludeSpecial()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // The reference strips a leading space only from the first rendered token slot: a suppressed control consumes that slot, so the marker-prefixed piece after it keeps its space.
        IReadOnlyList<int> ids = [ControlId, 15, 6];
        Assert.Equal(" hello world", tokenizer.Decode(ids));
        Assert.Equal("<|im_start|> hello world", tokenizer.Decode(ids, includeSpecial: true));
        Assert.Equal(string.Empty, tokenizer.Decode([ControlId]));
        Assert.Equal("<|im_start|>", tokenizer.Decode([ControlId], includeSpecial: true));

        // Without the control ahead of them the same pieces drop their opening space at slot zero.
        Assert.Equal("hello world", tokenizer.Decode([15, 6]));
    }

    [Fact]
    public void Decode_LeadingMarkerDroppedAtSequenceStartKeptMidSequence()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        // ▁ renders as a space except at the very start of the sequence. phi-3 'leading_space' pins both sides: Encode(" world") is [29871 ("▁"), 3186 ("▁world")] and decoding those ids back through the reference's leading-space strip reproduces " world" exactly, so encode and decode are exact inverses on oracle contents.
        Assert.Equal("world", tokenizer.Decode([6]));
        Assert.Equal("hello world", tokenizer.Decode([15, 6]));

        // A mid-sequence piece with no marker concatenates without a space: the opening space belongs to markers only.
        Assert.Equal("worldab", tokenizer.Decode([6, 3]));

        // A standalone marker renders as its space mid-sequence — whitespace-only pieces are carried on decode exactly as encode spells them. phi-3 'single_space' pins [259] for " ", the combined-run spelling of that vocabulary's two-marker run.
        Assert.Equal("world ", tokenizer.Decode([6, 20]));

        // Encode marks a leading boundary and decode drops its space there: the documented sequence-start behavior that makes encode→decode exact inverses on the pinned oracles' contents.
        Assert.Equal(" world", tokenizer.Decode(tokenizer.Encode(" world")));
    }

    [Fact]
    public void EncodeWithBos_PrependsDeclaredBosId()
    {
        GgufTokenizer tokenizer = OpenSynthetic();

        Assert.True(tokenizer.AddBosToken);
        Assert.Equal([ControlId, 17], tokenizer.EncodeWithBos("ab"));
        Assert.Equal([17], tokenizer.Encode("ab"));
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
        Assert.Equal([15], tokenizer.Encode("hello"));
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

