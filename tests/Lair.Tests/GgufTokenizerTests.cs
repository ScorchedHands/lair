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
        byte[] image = BuildSyntheticImage(pre: "nova2");
        GgufException ex = Assert.Throws<GgufException>(() => GgufTokenizer.FromGguf(GgufFile.Open(WriteTemp(image))));
        Assert.Contains("unsupported pre-tokenizer 'nova2'", ex.Message);
    }

    [Fact]
    public void FromGguf_UnsupportedPreTokenizer_ListsSupportedProfiles()
    {
        byte[] image = BuildSyntheticImage(pre: "nova2");
        GgufException ex = Assert.Throws<GgufException>(() => GgufTokenizer.FromGguf(GgufFile.Open(WriteTemp(image))));

        // The supported list is sourced from the profile registry itself.
        Assert.Contains("(supported: 'llama-bpe', 'gpt2', 'qwen2', 'smollm')", ex.Message);
    }

    [Fact]
    public void ProfileSelection_Gpt2Pre_SplitsCaseSensitiveContractionsAndRunsMerges()
    {
        using GgufFile gguf = GgufFile.Open(WriteSyntheticVocab("gpt2"));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        // "'s" is a contraction pre-token, but the merge shortcut is off for gpt2: the piece
        // is still assembled from the "'" + "s" merge chain.
        Assert.Equal([16], tokenizer.Encode("'s"));

        // (?-i:'s|'t|'re|'ve|'m|'ll|'d) is case-sensitive: "'S" is not a contraction, so the
        // apostrophe pre-token ends at the quote and the uppercase letter splits off to byte fallback.
        Assert.Equal([14, 24 + 0x53], tokenizer.Encode("'S"));

        // No merge shortcut: "abc" goes through the merge chain, not a whole-piece vocab hit.
        Assert.Equal([7], tokenizer.Encode("abc"));

        // Whole-number digit runs: the "3 4" merge spans digits the llama-bpe splitter would cut.
        Assert.Equal([22, 23], tokenizer.Encode("1234"));
    }

    [Fact]
    public void ProfileSelection_SmollmPre_SplitsDigitRunsAndSkipsDecodeSpaceCleaning()
    {
        using GgufFile gguf = GgufFile.Open(WriteSyntheticVocab("smollm"));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        // GPT-2 split first: "hello" + " !" + " 1234"; then the digit-run pass cuts the space off
        // the " 1234" pre-token, so "1234" merges to "12" + "34" exactly like gpt2.
        Assert.Equal([11, 10, 13, 10, 22, 23], tokenizer.Encode("hello ! 1234"));

        // clean_spaces is off for smollm: the "hello" + "Ġ!" decode stays "hello !" with the space.
        Assert.Equal("hello !", tokenizer.Decode([11, 10, 13]));

        // add_bos defaults to false for smollm.
        Assert.False(tokenizer.AddBosToken);
    }

    [Fact]
    public void ProfileSelection_MissingPreKey_DefaultsToLlamaBpe()
    {
        using GgufFile gguf = GgufFile.Open(WriteSyntheticVocab(pre: null));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        // llama-bpe digit grouping: "1234" splits "123" + "4", so the "3 4" merge is unreachable.
        Assert.Equal([22, 20, 21], tokenizer.Encode("1234"));

        // Case-insensitive contractions plus the ignore-merges shortcut: "'S" hits the vocab whole.
        Assert.Equal([17], tokenizer.Encode("'S"));

        // add_bos defaults to true for llama-bpe.
        Assert.True(tokenizer.AddBosToken);
    }

    [Fact]
    public void ProfileSelection_Qwen2Pre_UsesTheGpt2FamilySplit_AndBosDefaultsOff()
    {
        using GgufFile gguf = GgufFile.Open(WriteSyntheticVocab("qwen2"));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        // pre=qwen2 resolves to the byte-level GPT-2 family pipeline: same case-sensitive contractions,
        // no merge shortcut, whole digit runs — identical to the gpt2 profile above.
        Assert.Equal([16], tokenizer.Encode("'s"));
        Assert.Equal([14, 24 + 0x53], tokenizer.Encode("'S"));
        Assert.Equal([7], tokenizer.Encode("abc"));
        Assert.Equal([22, 23], tokenizer.Encode("1234"));

        Assert.False(tokenizer.AddBosToken);
    }

    [Fact]
    public void ProfileSelection_AllSupportedPreKeys_LoadSuccessfully()
    {
        foreach (string pre in new[] { "llama-bpe", "gpt2", "qwen2", "smollm" })
        {
            using GgufFile gguf = GgufFile.Open(WriteSyntheticVocab(pre));
            GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
            Assert.Equal(280, tokenizer.VocabSize);
        }
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

    // ---------------------------------------------------------------- tokenizer.ggml.byte_to_token

    [Fact]
    public void ByteToToken_DeclaredMapOverridesVocabScan()
    {
        // The fixture vocabulary places a byte-type piece "<0X78>" at id 14 — deliberately invisible to
        // the "<0xNN>" vocab scan — ahead of the scan block at 15..270. The declared map points 0x78 at
        // id 14, so the declared id must win over the scan's 15 + 0x78 = 135.
        int[] byteToToken = new int[256];
        for (int b = 0; b < 256; b++)
        {
            byteToToken[b] = 15 + b;
        }

        byteToToken[0x78] = 14;

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildByteTokenImage(byteToToken)));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        Assert.Equal([14], tokenizer.Encode("x"));
        Assert.Equal([15 + 0x79], tokenizer.Encode("y"));

        // Control: the same vocabulary without the declared map falls back to the scan.
        using GgufFile control = GgufFile.Open(WriteTemp(BuildByteTokenImage(null)));
        Assert.Equal([15 + 0x78], GgufTokenizer.FromGguf(control).Encode("x"));
    }

    [Fact]
    public void ByteToToken_OutOfVocabEntry_FailsAtLoadNamingKeyAndByteIndex()
    {
        int[] byteToToken = new int[256];
        for (int b = 0; b < 256; b++)
        {
            byteToToken[b] = 15 + b;
        }

        byteToToken[7] = 9999;

        byte[] image = BuildByteTokenImage(byteToToken);
        GgufException ex = Assert.Throws<GgufException>(() => GgufTokenizer.FromGguf(GgufFile.Open(WriteTemp(image))));
        Assert.Contains("tokenizer.ggml.byte_to_token", ex.Message);
        Assert.Contains("entry 7", ex.Message);
        Assert.Contains("9999", ex.Message);
    }

    [Fact]
    public void ByteToToken_NonByteTypeEntry_FailsAtLoadNamingKeyAndByteIndex()
    {
        int[] byteToToken = new int[256];
        for (int b = 0; b < 256; b++)
        {
            byteToToken[b] = 15 + b;
        }

        byteToToken[7] = 3; // vocabulary id 3 is the normal-type piece "a", not a byte token

        byte[] image = BuildByteTokenImage(byteToToken);
        GgufException ex = Assert.Throws<GgufException>(() => GgufTokenizer.FromGguf(GgufFile.Open(WriteTemp(image))));
        Assert.Contains("tokenizer.ggml.byte_to_token", ex.Message);
        Assert.Contains("entry 7", ex.Message);
        Assert.Contains("not a byte token", ex.Message);
    }

    // ---------------------------------------------------------------- tokenizer.ggml.byte_fallback

    [Fact]
    public void ByteFallbackDisabled_RoutesUnencodableTextToUnknownId()
    {
        byte[] image = BuildSyntheticImage(extraKv: [("tokenizer.ggml.byte_fallback", 7u, writer => writer.Write((byte)0))]);
        using GgufFile gguf = GgufFile.Open(WriteTemp(image));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        // "x" has no vocabulary piece: with byte_fallback=false it routes to the declared unk id
        // instead of the byte tokens. In "café" the unencodable symbols (the two é bytes' visual
        // atoms) each route to unk while the encodable letters keep their ids.
        Assert.Equal([0], tokenizer.Encode("x"));
        Assert.Equal([5, 3, 0, 0, 0], tokenizer.Encode("café"));

        // Encodable text is unaffected.
        Assert.Equal([7], tokenizer.Encode("abc"));
    }

    [Fact]
    public void ByteFallbackDisabled_NoUnknownToken_DropsUnencodableText()
    {
        byte[] image = BuildSyntheticImage(
            extraKv: [("tokenizer.ggml.byte_fallback", 7u, writer => writer.Write((byte)0))],
            declareUnknownId: false);
        using GgufFile gguf = GgufFile.Open(WriteTemp(image));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        // No declared unk: the unencodable symbols are dropped entirely, matching the drop behavior
        // for bytes with no byte token; encodable letters stay.
        Assert.Empty(tokenizer.Encode("x"));
        Assert.Equal([5, 3], tokenizer.Encode("café"));
        Assert.Equal([7], tokenizer.Encode("abc"));
    }

    // ---------------------------------------------------------------- tokenizer.ggml.add_space_prefix

    [Fact]
    public void AddSpacePrefixFalse_IsReadAndExposed_PreSplitUnchanged()
    {
        byte[] image = BuildSyntheticImage(extraKv: [("tokenizer.ggml.add_space_prefix", 7u, writer => writer.Write((byte)0))]);
        using GgufFile gguf = GgufFile.Open(WriteTemp(image));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        // The key is read and reported, but the split is unchanged: b12032 /tokenize on a model that
        // declares it false (smollm) still folds the leading space into the pre-token, so stripping it
        // would diverge from the reference and drop text on round-trip.
        Assert.False(tokenizer.AddSpacePrefix);
        Assert.Equal([11, 10, 12], tokenizer.Encode("hello world"));
        Assert.Equal([8], tokenizer.Encode(" ab"));
        Assert.Equal("hello world", tokenizer.Decode(tokenizer.Encode("hello world")));
    }

    [Fact]
    public void AddSpacePrefixDefaultTrue_KeepsFoldedLeadingSpace()
    {
        // Control for the fact above: the same synthetic without the key reports the default.
        GgufTokenizer tokenizer = OpenSynthetic();

        Assert.True(tokenizer.AddSpacePrefix);
        Assert.Equal([11, 10, 12], tokenizer.Encode("hello world"));
        Assert.Equal([8], tokenizer.Encode(" ab"));
    }

    // ---------------------------------------------------------------- profile-scoped BOS/EOS defaults

    [Fact]
    public void ProfileDefaults_LlamaBpeAppliesBosEosFallback11WithLogLines()
    {
        List<string> logLines = [];
        using GgufFile gguf = GgufFile.Open(WriteSyntheticVocab("llama-bpe"));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf, logLines.Add);

        Assert.Equal(11, tokenizer.BosId);
        Assert.Equal(11, tokenizer.EosId);
        Assert.Equal(2, logLines.Count);
        Assert.Contains(logLines, line => line.StartsWith("tokenizer_default ", StringComparison.Ordinal)
            && line.Contains("key=tokenizer.ggml.bos_token_id default=11", StringComparison.Ordinal));
        Assert.Contains(logLines, line => line.StartsWith("tokenizer_default ", StringComparison.Ordinal)
            && line.Contains("key=tokenizer.ggml.eos_token_id default=11", StringComparison.Ordinal));
    }

    [Fact]
    public void ProfileDefaults_Gpt2AppliesBosEosFallback1And2WithLogLines()
    {
        List<string> logLines = [];
        using GgufFile gguf = GgufFile.Open(WriteSyntheticVocab("gpt2"));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf, logLines.Add);

        Assert.Equal(1, tokenizer.BosId);
        Assert.Equal(2, tokenizer.EosId);
        Assert.Equal(2, logLines.Count);
        Assert.Contains(logLines, line => line.Contains("key=tokenizer.ggml.bos_token_id default=1", StringComparison.Ordinal));
        Assert.Contains(logLines, line => line.Contains("key=tokenizer.ggml.eos_token_id default=2", StringComparison.Ordinal));
    }

    [Fact]
    public void ProfileDefaults_SmollmAppliesBosEosFallback1And2WithLogLines()
    {
        List<string> logLines = [];
        using GgufFile gguf = GgufFile.Open(WriteSyntheticVocab("smollm"));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf, logLines.Add);

        Assert.Equal(1, tokenizer.BosId);
        Assert.Equal(2, tokenizer.EosId);
        Assert.Equal(2, logLines.Count);
        Assert.Contains(logLines, line => line.Contains("key=tokenizer.ggml.bos_token_id default=1", StringComparison.Ordinal));
        Assert.Contains(logLines, line => line.Contains("key=tokenizer.ggml.eos_token_id default=2", StringComparison.Ordinal));
    }

    [Fact]
    public void ProfileDefaults_DeclaredSpecialIds_EmitNoDefaultLogLines()
    {
        // BuildSyntheticImage declares bos/eos/padding/unknown ids and omits eot (whose NullToken
        // fallback is not a default application): nothing to log.
        List<string> logLines = [];
        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildSyntheticImage()));
        GgufTokenizer.FromGguf(gguf, logLines.Add);

        Assert.Empty(logLines);
    }

    // ---------------------------------------------------------------- tokenizer.ggml.byte_to_utf8

    [Fact]
    public void ByteToUtf8_DeclaredMapOverridesTheGpt2VisualMap()
    {
        int[] byteToUtf8 = new int[256];
        for (int b = 0; b < 256; b++)
        {
            byteToUtf8[b] = b;
        }

        byteToUtf8[0x78] = 0x00E9; // byte 0x78 renders as the visual atom "é" in this fixture

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildVisualMapImage(byteToUtf8)));
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        // "x" has no vocabulary piece; the declared map byte-encodes it to the "é" atom, which the vocabulary holds.
        Assert.Equal([14], tokenizer.Encode("x"));

        // Control: without the declared key the GPT-2 visual map keeps "x" as "x", which the vocabulary does not hold.
        using GgufFile control = GgufFile.Open(WriteTemp(BuildVisualMapImage(null)));
        Assert.Empty(GgufTokenizer.FromGguf(control).Encode("x"));
    }

    [Fact]
    public void ByteToUtf8_SurrogateEntry_FailsAtLoadNamingKeyAndByteIndex()
    {
        int[] byteToUtf8 = new int[256];
        for (int b = 0; b < 256; b++)
        {
            byteToUtf8[b] = b;
        }

        byteToUtf8[5] = 0xD800;

        byte[] image = BuildVisualMapImage(byteToUtf8);
        GgufException ex = Assert.Throws<GgufException>(() => GgufTokenizer.FromGguf(GgufFile.Open(WriteTemp(image))));
        Assert.Contains("tokenizer.ggml.byte_to_utf8", ex.Message);
        Assert.Contains("entry 5", ex.Message);
        Assert.Contains("55296", ex.Message);
    }

    // ---------------------------------------------------------------- tokenizer.ggml.utf8_flush_between_controls

    [Fact]
    public void Utf8FlushBetweenControlsDisabled_SplitsMultiByteTextAtControlBoundaries()
    {
        GgufTokenizer stitching = OpenSynthetic();

        // Byte tokens for the two bytes of "é" separated by a control token: the default stitches the
        // partial sequence across the skipped control boundary.
        IReadOnlyList<int> split = [14 + 0xC3, 2, 14 + 0xA9];
        Assert.Equal("é", stitching.Decode(split));

        byte[] image = BuildSyntheticImage(extraKv: [("tokenizer.ggml.utf8_flush_between_controls", 7u, writer => writer.Write((byte)0))]);
        using GgufFile gguf = GgufFile.Open(WriteTemp(image));
        GgufTokenizer flushing = GgufTokenizer.FromGguf(gguf);

        // With the flush on, each half becomes its own invalid byte buffer: two replacement characters.
        Assert.Equal("\uFFFD\uFFFD", flushing.Decode(split));
    }

    // ---------------------------------------------------------------- tokenizer.ggml.eot_token_id

    [Fact]
    public void EotTokenId_ExposedWhenDeclared_AndAbsentStaysSentinel()
    {
        byte[] image = BuildSyntheticImage(extraKv: [("tokenizer.ggml.eot_token_id", 4u, writer => writer.Write(13u))]);
        using GgufFile gguf = GgufFile.Open(WriteTemp(image));
        Assert.Equal(13, GgufTokenizer.FromGguf(gguf).EotId);

        using GgufFile control = GgufFile.Open(WriteTemp(BuildSyntheticImage()));
        Assert.Equal(-1, GgufTokenizer.FromGguf(control).EotId);
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

    [Fact]
    public void RealModel_Llama32_DeclaresBosEosAndAddBos()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative("models/Llama-3.2-1B-Instruct-Q4_K_M.gguf");
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        Assert.True(tokenizer.AddBosToken);
        Assert.Equal(gguf.Metadata.GetInt("tokenizer.ggml.bos_token_id", -1), tokenizer.BosId);
        Assert.Equal(gguf.Metadata.GetInt("tokenizer.ggml.eos_token_id", -1), tokenizer.EosId);
    }

    [Fact]
    public void RealModel_SmolLm_AddSpacePrefixFalse_IsExposedAndRoundTripKeepsSpaces()
    {
        string? modelPath = ModelFixture.Resolve();
        if (modelPath is null)
        {
            return;
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        // The fixture declares the key false; the value is exposed, and the pre-split still folds the
        // leading space — matching b12032 /tokenize, which encodes " world!" with the space intact.
        Assert.False(gguf.Metadata.GetBool("tokenizer.ggml.add_space_prefix", true));
        Assert.False(tokenizer.AddSpacePrefix);
        Assert.Equal("Hello, world!", tokenizer.Decode(tokenizer.Encode("Hello, world!")));
    }

    private byte[] BuildSyntheticImage(
        string model = "gpt2",
        string? pre = "gpt2",
        List<(string Key, uint ValueType, Action<BinaryWriter> Write)>? extraKv = null,
        bool declareUnknownId = true)
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

        List<(string Key, uint ValueType, Action<BinaryWriter> Write)> specialKv =
        [
            ("tokenizer.ggml.bos_token_id", 4u, writer => writer.Write(1u)),
            ("tokenizer.ggml.eos_token_id", 4u, writer => writer.Write(2u)),
            ("tokenizer.ggml.padding_token_id", 4u, writer => writer.Write(2u)),
            ("tokenizer.ggml.add_bos_token", 7u, writer => writer.Write((byte)1)),
        ];
        if (declareUnknownId)
        {
            specialKv.Add(("tokenizer.ggml.unknown_token_id", 4u, writer => writer.Write(0u)));
        }

        if (extraKv is not null)
        {
            specialKv.AddRange(extraKv);
        }

        return BuildImage(
            model,
            pre,
            [.. pieces],
            [.. types],
            merges,
            extraKv: specialKv);
    }

    /// <summary>
    /// Byte-fallback fixture with a shifted byte block: a byte-type piece <c>&lt;0X78&gt;</c> (uppercase X,
    /// so the <c>&lt;0xNN&gt;</c> scan never finds it) sits at id 14 before the scan block at 15..270, and a
    /// declared <c>tokenizer.ggml.byte_to_token</c> (when given) decides which byte token each byte value uses.
    /// </summary>
    private byte[] BuildByteTokenImage(int[]? byteToToken)
    {
        List<string> pieces =
        [
            "<unk>", "<|im_start|>", "<|im_end|>",
            "a", "b", "c", "ab", "abc", "Ġab", "aĠb", "Ġ", "hello", "world", "!", "<0X78>",
        ];
        List<int> types = [2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6];

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

        List<(string Key, uint ValueType, Action<BinaryWriter> Write)> specialKv =
        [
            ("tokenizer.ggml.bos_token_id", 4u, writer => writer.Write(1u)),
            ("tokenizer.ggml.eos_token_id", 4u, writer => writer.Write(2u)),
            ("tokenizer.ggml.padding_token_id", 4u, writer => writer.Write(2u)),
            ("tokenizer.ggml.unknown_token_id", 4u, writer => writer.Write(0u)),
            ("tokenizer.ggml.add_bos_token", 7u, writer => writer.Write((byte)1)),
        ];
        if (byteToToken is not null)
        {
            specialKv.Add(("tokenizer.ggml.byte_to_token", 9u, writer => WriteInt32ArrayValue(writer, byteToToken)));
        }

        return BuildImage("gpt2", "gpt2", [.. pieces], [.. types], merges, extraKv: specialKv);
    }

    /// <summary>
    /// Byte-atom fixture with no <c>&lt;0xNN&gt;</c> byte tokens: the vocabulary holds "é" (id 14), so a declared
    /// <c>tokenizer.ggml.byte_to_utf8</c> (when given) decides what the byte-encoding of "x" collapses to.
    /// </summary>
    private byte[] BuildVisualMapImage(int[]? byteToUtf8)
    {
        List<string> pieces =
        [
            "<unk>", "<|im_start|>", "<|im_end|>",
            "a", "b", "c", "ab", "abc", "Ġab", "aĠb", "Ġ", "hello", "world", "!", "é",
        ];
        List<int> types = [2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        string[] merges = ["a b", "ab c", "Ġ ab"];

        List<(string Key, uint ValueType, Action<BinaryWriter> Write)> specialKv =
        [
            ("tokenizer.ggml.bos_token_id", 4u, writer => writer.Write(1u)),
            ("tokenizer.ggml.eos_token_id", 4u, writer => writer.Write(2u)),
            ("tokenizer.ggml.padding_token_id", 4u, writer => writer.Write(2u)),
            ("tokenizer.ggml.unknown_token_id", 4u, writer => writer.Write(0u)),
            ("tokenizer.ggml.add_bos_token", 7u, writer => writer.Write((byte)1)),
        ];
        if (byteToUtf8 is not null)
        {
            specialKv.Add(("tokenizer.ggml.byte_to_utf8", 9u, writer => WriteInt32ArrayValue(writer, byteToUtf8)));
        }

        return BuildImage("gpt2", "gpt2", [.. pieces], [.. types], merges, extraKv: specialKv);
    }

    /// <summary>
    /// Shared profile-selection vocab: the synthetic base vocab plus digit pieces, the "1 2"/"3 4"
    /// merges, and apostrophe/contraction pieces, so one fixture distinguishes the three profiles.
    /// A null <c>pre</c> omits the <c>tokenizer.ggml.pre</c> key entirely.
    /// </summary>
    private string WriteSyntheticVocab(string? pre)
    {
        List<string> pieces =
        [
            "<unk>", "<|im_start|>", "<|im_end|>",
            "a", "b", "c", "ab", "abc", "Ġab", "aĠb", "Ġ", "hello", "world", "!",
            "'", "Ġ'", "'s", "'S",
            "1", "2", "3", "4", "12", "34",
        ];
        List<int> types = [2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

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
            "1 2", "3 4", "' s",
        ];

        return WriteTemp(BuildImage(
            "gpt2",
            pre,
            [.. pieces],
            [.. types],
            merges));
    }

    private static byte[] BuildImage(
        string model,
        string? pre,
        string[] pieces,
        int[] types,
        string[] merges,
        List<(string Key, uint ValueType, Action<BinaryWriter> Write)>? extraKv = null)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        int kvCount = (pre is null ? 5 : 6) + (extraKv?.Count ?? 0);

        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write(0ul); // n_tensors
        writer.Write((ulong)kvCount);

        WriteKvString(writer, "general.architecture", "llama");
        WriteKvString(writer, "tokenizer.ggml.model", model);
        if (pre is not null)
        {
            WriteKvString(writer, "tokenizer.ggml.pre", pre);
        }
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
        WriteInt32ArrayValue(writer, values);
    }

    // The ARRAY payload (element type + count + values) without the key/type prefix, for extraKv writers.
    private static void WriteInt32ArrayValue(BinaryWriter writer, int[] values)
    {
        writer.Write(5u); // element type INT32
        writer.Write((ulong)values.Length);
        foreach (int value in values)
        {
            writer.Write(value);
        }
    }
}
