using System.Security.Cryptography;
using System.Text.Json;
using Lair.Loader;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Id-for-id tokenizer parity against the pinned llama.cpp b12032 build: token sequences captured from
/// <c>llama-server /tokenize</c> (add_special=false, parse_special=false — the same surface
/// <see cref="GgufTokenizer.Encode"/> exposes) for both real fixtures are compared against LAIR's
/// <c>Encode</c> (see scripts/generate-tokenizer-parity.ps1). Guards: the model file's sha256 must match
/// the fixture and the model's <c>tokenizer.ggml.pre</c> must match the profile the fixture was captured
/// for. The whole category skips cleanly when a fixture or its pinned model file is absent; the fixtures
/// are committed, so the comparisons never need llama.cpp.
/// </summary>
[Trait("Category", "Tokenizer")]
public sealed class TokenizerParityTests
{
    private const string SmolLmFixture = "tests/fixtures/parity-tokenizer-smollm.json";
    private const string Llama32Fixture = "tests/fixtures/parity-tokenizer-llama32.json";
    private const string OlmoFixture = "tests/fixtures/parity-tokenizer-olmo.json";

    [Fact]
    public void SmolLm_Encode_MatchesPinnedTokenizeOracle() => AssertFixture(SmolLmFixture);

    [Fact]
    public void Llama32_Encode_MatchesPinnedTokenizeOracle() => AssertFixture(Llama32Fixture);

    [Fact]
    public void Olmo_Encode_MatchesPinnedTokenizeOracle() => AssertFixture(OlmoFixture);

    private static void AssertFixture(string fixtureRelativePath)
    {
        string? fixtureFile = ModelFixture.ResolveRepoRelative(fixtureRelativePath);
        if (fixtureFile is null)
        {
            return; // the parity fixture is absent: nothing to compare against
        }

        ParityFixture fixture = LoadFixture(fixtureFile);

        string? modelPath = ModelFixture.ResolveRepoRelative(fixture.ModelFile);
        if (modelPath is null)
        {
            return; // real-model cases skip when the pinned GGUF is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        string modelSha = Sha256File(modelPath);
        if (modelSha != fixture.ModelSha256)
        {
            throw new InvalidOperationException(
                $"tokenizer parity guard: {fixture.ModelFile} hashes to {modelSha} but the fixture was captured against {fixture.ModelSha256}; " +
                "regenerate the fixture with scripts/generate-tokenizer-parity.ps1 against the pinned model.");
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        string pre = gguf.Metadata.GetString("tokenizer.ggml.pre", "llama-bpe");
        if (pre != fixture.Pre)
        {
            throw new InvalidOperationException(
                $"tokenizer parity guard: {fixture.ModelFile} declares tokenizer.ggml.pre='{pre}' but the fixture was captured for '{fixture.Pre}'; " +
                "regenerate the fixture with scripts/generate-tokenizer-parity.ps1.");
        }

        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);

        List<string> mismatches = [];
        foreach (FixtureCase entry in fixture.Cases)
        {
            IReadOnlyList<int> actual = tokenizer.Encode(entry.Content);
            if (!actual.SequenceEqual(entry.Tokens))
            {
                mismatches.Add(
                    $"case '{entry.Name}' content='{entry.Content}': reference [{string.Join(",", entry.Tokens)}] " +
                    $"({string.Join("|", entry.TokensStr)}) vs LAIR [{string.Join(",", actual)}]");
            }
        }

        Assert.Empty(mismatches);
    }

    private static ParityFixture LoadFixture(string path)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;

        List<FixtureCase> cases = [];
        foreach (JsonElement entry in root.GetProperty("cases").EnumerateArray())
        {
            List<int> tokens = [];
            foreach (JsonElement token in entry.GetProperty("tokens").EnumerateArray())
            {
                tokens.Add(token.GetInt32());
            }

            List<string> tokensStr = [];
            foreach (JsonElement piece in entry.GetProperty("tokens_str").EnumerateArray())
            {
                tokensStr.Add(piece.GetString()!);
            }

            cases.Add(new FixtureCase(
                entry.GetProperty("name").GetString()!,
                entry.GetProperty("content").GetString()!,
                tokens,
                tokensStr));
        }

        return new ParityFixture(
            root.GetProperty("model_file").GetString()!,
            root.GetProperty("model_sha256").GetString()!,
            root.GetProperty("pre").GetString()!,
            cases);
    }

    private static string Sha256File(string path)
    {
        using SHA256 hasher = SHA256.Create();
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(hasher.ComputeHash(stream)).ToLowerInvariant();
    }

    private sealed record ParityFixture(
        string ModelFile,
        string ModelSha256,
        string Pre,
        IReadOnlyList<FixtureCase> Cases);

    private sealed record FixtureCase(
        string Name,
        string Content,
        IReadOnlyList<int> Tokens,
        IReadOnlyList<string> TokensStr);
}
