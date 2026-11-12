using System.Security.Cryptography;
using Lair.Contracts;
using Lair.Http;
using Lair.Loader;

namespace Lair.Tests;

/// <summary>
/// The quantization a served model advertises is the format its stored bytes bulk lives in, not the format with the most
/// tensors: hybrid files (quantized weights beside full-precision norm vectors) must report a weight group even when the
/// norms outnumber every other group, and a huge-vocabulary file whose embedding outweighs its quantized weights reports
/// that instead. Cases cover the rule on synthetic tables and against pinned real fixtures whose mix actually exposes it;
/// a case skips when its fixture file is absent.
/// </summary>
[Trait("Category", "Api")]
public sealed class ModelQuantReportingTests
{
    private const string Gemma3FixtureRelativePath = "models/gemma-3-1b-it-Q3_K_S.gguf";

    private const string Gemma3FixtureSha256 = "0ffb15257ccee33bee17e732fb3db55a332801d187e87164488888022fe1634c";

    private const string Phi3FixtureRelativePath = "models/Phi-3-mini-4k-instruct-Q2_K.gguf";

    private const string Phi3FixtureSha256 = "72237d1e7c56abc777a2f4d1482f3eff4bcfc156566f226749bc47baa9608285";

    [Fact]
    public void DominantQuant_ReportsTheFormatTheStoredBytesBulkIn_NotTheTensorCount()
    {
        LogicalTensorTable table = new(
        [
            .. Enumerable.Range(0, 12).Select(i => Tensor($"blk.{i}.attn_norm.weight", QuantFormat.F32, 4 * 1024)),
            .. Enumerable.Range(0, 6).Select(i => Tensor($"blk.{i}.ffn_up.weight", QuantFormat.Q4_K, 1_048_576)),
        ]);

        Assert.Equal("Q4_K", GenerationService.DominantQuant(table));
    }

    [Fact]
    public void DominantQuant_BreaksByteTiesOnTensorCountThenOrdinalName()
    {
        LogicalTensorTable byCount = new(
        [
            Tensor("a.norm0", QuantFormat.F32, 4_096),
            Tensor("b.weight0", QuantFormat.Q8_0, 4_096),
            Tensor("c.weight1", QuantFormat.Q8_0, 4_096),
        ]);

        Assert.Equal("Q8_0", GenerationService.DominantQuant(byCount));

        LogicalTensorTable byName = new(
        [
            Tensor("a.norm0", QuantFormat.F32, 4_096),
            Tensor("b.weight0", QuantFormat.Q8_0, 4_096),
        ]);

        Assert.Equal("F32", GenerationService.DominantQuant(byName));
    }

    [Fact]
    public void DominantQuant_Gemma3FixtureReportsItsLargestByteGroupNotItsNormVectors()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(Gemma3FixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        AssertFixtureHash(modelPath, Gemma3FixtureSha256);

        using GgufFile gguf = GgufFile.Open(modelPath);

        int norms = gguf.Tensors.Tensors.Count(tensor => tensor.Format == QuantFormat.F32);
        Assert.True(norms > 100, $"expected the fixture's full-precision norm vectors to outnumber every other group, got {norms}");

        // Measured on this file: its Q8_0 token embedding carries 320,864,256 stored bytes against 258,785,280 for
        // the IQ4_NL weight group and 102,136,320 for Q3_K. The bulk of this file really is its embedding, so that is
        // what it advertises; a count-based rule would have reported F32 from 157 norm vectors totaling 537,088 bytes.
        Assert.Equal("Q8_0", GenerationService.DominantQuant(gguf.Tensors));
    }

    [Fact]
    public void DominantQuant_Phi3FixtureReportsItsDominantWeightFormat()
    {
        string? modelPath = ModelFixture.ResolveRepoRelative(Phi3FixtureRelativePath);
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        AssertFixtureHash(modelPath, Phi3FixtureSha256);

        using GgufFile gguf = GgufFile.Open(modelPath);

        // Measured on this file: 65 Q2_K tensors totaling 858,074,112 bytes against 475,791,360 for the Q3_K group and
        // 80,801,280 for the single untied Q6_K output head.
        Assert.Equal("Q2_K", GenerationService.DominantQuant(gguf.Tensors));
    }

    private static LogicalTensor Tensor(string name, QuantFormat format, long sizeBytes) =>
        new(name, new TensorShape([1, 1]), format, Offset: 0, sizeBytes);

    private static void AssertFixtureHash(string modelPath, string expectedSha256)
    {
        using SHA256 hasher = SHA256.Create();
        using FileStream stream = File.OpenRead(modelPath);
        string actual = Convert.ToHexString(hasher.ComputeHash(stream)).ToLowerInvariant();

        Assert.True(
            string.Equals(actual, expectedSha256, StringComparison.Ordinal),
            $"fixture {modelPath} hash {actual} does not match the pinned {expectedSha256}");
    }
}
