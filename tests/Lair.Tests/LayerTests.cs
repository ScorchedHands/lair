using System.Diagnostics;
using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Tests;

/// <summary>
/// Layer math unit checks (RMSNorm, RoPE pairs, SiLU, causal attention) against hand-computed values, ModelDesc
/// parsing from synthetic and real GGUF metadata, Open-time tensor validation, a tiny tied-weight forward for
/// determinism, and an integration smoke over the real SmolLM F16 fixture: finite logits, run-to-run determinism,
/// stable top-5 ids, and printed activation magnitudes.
/// </summary>
[Trait("Category", "Layers")]
public sealed class LayerTests : IDisposable
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

    // ---------------------------------------------------------------- layer math units

    [Fact]
    public void RmsNormWeighted_MatchesHandComputedSmallVector()
    {
        float[] src = [0.5f, -1f, 2f];
        float[] weight = [2f, 1f, 0.5f];
        float eps = 0f;

        double meanSq = (0.5 * 0.5 + 1.0 + 4.0) / 3.0; // hand: sum of squares over n
        double invRms = 1.0 / Math.Sqrt(meanSq);

        float[] dst = new float[3];
        LayerOps.RmsNormWeighted(src, weight, eps, dst);

        for (int i = 0; i < src.Length; i++)
        {
            double expected = src[i] * invRms * weight[i];
            Assert.True(
                Math.Abs(dst[i] - expected) <= 1e-6 * Math.Max(1.0, Math.Abs(expected)),
                $"rmsnorm[{i}]: expected {expected:G9}, got {dst[i]:G9}");
        }

        // eps enters under the square root and keeps a zero vector finite instead of dividing by zero.
        LayerOps.RmsNormWeighted(new float[3], weight, 1e-5f, dst);
        foreach (float v in dst)
        {
            Assert.Equal(0f, v);
        }

        Assert.Throws<ArgumentException>(() => LayerOps.RmsNormWeighted(src, new float[2], eps, dst));
    }

    [Fact]
    public void RopeTables_AndRotation_MatchManualPairMath_AndPreserveHeadNorm()
    {
        const int HeadDim = 8; // adjacent pairs (2i, 2i+1); theta_i = base^(-2i/8)
        const double Base = 10_000.0;

        float[] cos = new float[2 * (HeadDim / 2)];
        float[] sin = new float[2 * (HeadDim / 2)];
        LayerOps.BuildRopeTables(Base, HeadDim, maxPositions: 2, cos, sin);

        // Position 0 is the identity rotation: every angle is zero.
        for (int i = 0; i < HeadDim / 2; i++)
        {
            Assert.Equal(1f, cos[i]);
            Assert.Equal(0f, sin[i]);
        }

        // Position 1 must reproduce the manual double math bit-for-bit after the float cast.
        for (int i = 0; i < HeadDim / 2; i++)
        {
            double invFreq = Math.Pow(Base, -2.0 * i / HeadDim);
            Assert.Equal((float)Math.Cos(1.0 * invFreq), cos[HeadDim / 2 + i]);
            Assert.Equal((float)Math.Sin(1.0 * invFreq), sin[HeadDim / 2 + i]);
        }

        float[] head = [3f, -1f, 4f, 1f, 5f, 9f, 2f, 6f]; // known vector rotated at position 1
        double normBefore = 0;
        foreach (float v in head)
        {
            normBefore += (double)v * v;
        }

        float[] rotated = [.. head];
        LayerOps.ApplyRopeToHead(rotated, HeadDim / 2, cos.AsSpan(4), sin.AsSpan(4));

        for (int i = 0; i < HeadDim / 2; i++)
        {
            double invFreq = Math.Pow(Base, -2.0 * i / HeadDim);
            float c = (float)Math.Cos(invFreq); // pos == 1: the angle equals the inverse frequency itself
            float s = (float)Math.Sin(invFreq);

            // Pair i occupies adjacent head dimensions 2i and 2i + 1, the layout rotary rows carry in GGUF.
            int lowIndex = 2 * i;
            int highIndex = 2 * i + 1;

            double loExpected = head[lowIndex] * (double)c - head[highIndex] * s;
            double hiExpected = head[highIndex] * (double)c + head[lowIndex] * s;

            Assert.True(
                Math.Abs(rotated[lowIndex] - loExpected) <= 1e-6 * Math.Max(1.0, Math.Abs(loExpected)),
                $"rope pair [{i}] low: expected {loExpected:G9}, got {rotated[lowIndex]:G9}");
            Assert.True(
                Math.Abs(rotated[highIndex] - hiExpected) <= 1e-6 * Math.Max(1.0, Math.Abs(hiExpected)),
                $"rope pair [{i}] high: expected {hiExpected:G9}, got {rotated[highIndex]:G9}");
        }

        // Each pair is a rotation matrix: the head's L2 norm must survive within float rounding.
        double normAfter = 0;
        foreach (float v in rotated)
        {
            normAfter += (double)v * v;
        }

        Assert.True(Math.Abs(normAfter - normBefore) <= 1e-4, $"RoPE must preserve head norm: before {normBefore:G9}, after {normAfter:G9}");

        // Position 0 tables rotate nothing.
        float[] identity = [.. head];
        LayerOps.ApplyRopeToHead(identity, HeadDim / 2, cos.AsSpan(0), sin.AsSpan(0));
        Assert.True(identity.AsSpan().SequenceEqual(head.AsSpan()), "position-0 RoPE must be the exact identity");

        Assert.Throws<ArgumentException>(() => LayerOps.BuildRopeTables(Base, 7, maxPositions: 1, new float[4], new float[4])); // odd head dim
    }

    [Fact]
    public void SplitHalfRoPE_MatchesManualPairMath_AndKeepsHeadNorm()
    {
        const int HeadDim = 8; // rotate-half pairs (i, i + headDim/2); theta_i = base^(-2i/headDim)
        const double Base = 10_000.0;

        float[] cos = new float[2 * (HeadDim / 2)];
        float[] sin = new float[2 * (HeadDim / 2)];
        LayerOps.BuildRopeTables(Base, HeadDim, maxPositions: 2, cos, sin); // the same angle table the adjacent pairing uses

        const int half = HeadDim / 2;

        // Position 0 is the identity rotation for this pairing too.
        float[] head = [3f, -1f, 4f, 1f, 5f, 9f, 2f, 6f];
        float[] identity = [.. head];
        LayerOps.ApplyRopeToSplitHalfHead(identity, half, cos.AsSpan(0), sin.AsSpan(0));
        Assert.True(identity.AsSpan().SequenceEqual(head.AsSpan()), "position-0 rotate-half RoPE must be the exact identity");

        double normBefore = 0;
        foreach (float v in head)
        {
            normBefore += (double)v * v;
        }

        float[] rotated = [.. head];
        LayerOps.ApplyRopeToSplitHalfHead(rotated, half, cos.AsSpan(half), sin.AsSpan(half)); // position 1

        for (int i = 0; i < half; i++)
        {
            double invFreq = Math.Pow(Base, -2.0 * i / HeadDim);
            float c = (float)Math.Cos(invFreq); // pos == 1: the angle equals the inverse frequency itself
            float s = (float)Math.Sin(invFreq);

            // Pair i occupies head dimensions i and i + half — the halves of the head, not adjacent dimensions.
            int lowIndex = i;
            int highIndex = i + half;

            double loExpected = head[lowIndex] * (double)c - head[highIndex] * s;
            double hiExpected = head[highIndex] * (double)c + head[lowIndex] * s;

            Assert.True(
                Math.Abs(rotated[lowIndex] - loExpected) <= 1e-6 * Math.Max(1.0, Math.Abs(loExpected)),
                $"rotate-half rope pair [{i}] low: expected {loExpected:G9}, got {rotated[lowIndex]:G9}");
            Assert.True(
                Math.Abs(rotated[highIndex] - hiExpected) <= 1e-6 * Math.Max(1.0, Math.Abs(hiExpected)),
                $"rotate-half rope pair [{i}] high: expected {hiExpected:G9}, got {rotated[highIndex]:G9}");
        }

        // The pairing differs from adjacent dimensions on this vector at position 1: the two rotations are not equal.
        float[] adjacent = [.. head];
        LayerOps.ApplyRopeToHead(adjacent, half, cos.AsSpan(half), sin.AsSpan(half));
        Assert.False(rotated.AsSpan().SequenceEqual(adjacent.AsSpan()), "rotate-half and adjacent-dimension pairing must differ");

        double normAfter = 0;
        foreach (float v in rotated)
        {
            normAfter += (double)v * v;
        }

        Assert.True(Math.Abs(normAfter - normBefore) <= 1e-4, $"rotate-half RoPE must preserve head norm: before {normBefore:G9}, after {normAfter:G9}");

        Assert.Throws<ArgumentException>(() => LayerOps.ApplyRopeToSplitHalfHead(new float[3], half, cos.AsSpan(half), sin.AsSpan(half)));
    }

    [Theory]
    [InlineData(-3f)]
    [InlineData(-1f)]
    [InlineData(0f)]
    [InlineData(0.5f)]
    [InlineData(2f)]
    public void Silu_MatchesReferenceActivationTable(float x)
    {
        double expected = (double)x / (1.0 + Math.Exp(-(double)x));

        if (x == 0f)
        {
            Assert.Equal(0f, LayerOps.Silu(x)); // exact zero in both paths
            return;
        }

        float actual = LayerOps.Silu(x);
        Assert.True(Math.Abs(actual - expected) <= 1e-6 * Math.Max(1.0, Math.Abs(expected)), $"silu({x}): expected {expected:G9}, got {actual:G9}");
    }

    [Fact]
    public void CausalAttention_TwoHeadsFourPositions_MatchesHandComputedSoftmaxAverage()
    {
        const int HeadDim = 3; // deliberately odd for attention alone: this op carries no rope coupling
        const int Heads = 2;
        const int KvHeads = 2;

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using BlockPool pool = new(blockTokens: 2, layerCount: 1, kvHeads: KvHeads, headDim: HeadDim, capacityBlocks: 4, backend);
        using SequenceKv kv = new(pool);

        // Two distinct K/V vectors per kv-head; positions alternate which one lands where so scores differ across t.
        float[][] kVariants = [[0.5f, -0.25f, 0f], [1f, 0f, -1f]];
        float[][] vVariants = [[0.25f, 0.75f, -0.5f], [-1f, 0.125f, 0.5f]];

        int positions = 4; // spans two physical blocks: reads across the block boundary must be exact
        float[][] kRows = new float[positions][];
        float[][] vRows = new float[positions][];

        for (int t = 0; t < positions; t++)
        {
            kv.Append();

            kRows[t] = new float[KvHeads * HeadDim];
            vRows[t] = new float[KvHeads * HeadDim];

            for (int g = 0; g < KvHeads; g++)
            {
                int pick = (t + g) % 2; // stagger so each kv-head sees both variants across positions
                Array.Copy(kVariants[pick], 0, kRows[t], g * HeadDim, HeadDim);
                Array.Copy(vVariants[pick], 0, vRows[t], g * HeadDim, HeadDim);
            }

            kv.WriteK(t, 0, kRows[t]);
            kv.WriteV(t, 0, vRows[t]);
        }

        float[] q = [1f, 2f, 0f, 0f, 1f, -1f]; // head-major: head0 then head1; heads == kvHeads is the identity mapping
        float scale = 1f / MathF.Sqrt(HeadDim);

        float[] outConcat = new float[Heads * HeadDim];
        LayerOps.CausalAttention(backend, q, Heads, KvHeads, HeadDim, kv, layer: 0, position: positions - 1, scale, softcap: 0f, window: 0, outConcat);

        for (int h = 0; h < Heads; h++)
        {
            // Hand oracle in double precision over the exact stored float rows.
            double[] scores = new double[positions];
            double maxScore = double.NegativeInfinity;
            for (int t = 0; t < positions; t++)
            {
                double dot = 0;
                for (int j = 0; j < HeadDim; j++)
                {
                    dot += q[h * HeadDim + j] * kRows[t][h * HeadDim + j]; // identity kv mapping: heads == kvHeads
                }

                scores[t] = dot * scale;
                maxScore = Math.Max(maxScore, scores[t]);
            }

            double sumExp = 0;
            double[] weights = new double[positions];
            for (int t = 0; t < positions; t++)
            {
                weights[t] = Math.Exp(scores[t] - maxScore);
                sumExp += weights[t];
            }

            for (int j = 0; j < HeadDim; j++)
            {
                double expected = 0;
                for (int t = 0; t < positions; t++)
                {
                    expected += weights[t] / sumExp * vRows[t][h * HeadDim + j];
                }

                float actual = outConcat[h * HeadDim + j];
                Assert.True(
                    Math.Abs(actual - expected) <= 1e-5,
                    $"attention head {h} dim {j}: expected {expected:G9}, got {actual:G9}");
            }
        }

        // The softmax average of bounded V rows stays inside the convex-hull magnitude (plus float slack).
        foreach (float v in outConcat)
        {
            Assert.True(Math.Abs(v) <= 1.000_002f, $"attention output {v:G6} exceeds the convex hull of the stored V values");
        }
    }

    [Fact]
    public void CausalAttention_GroupedQueryMapping_SharesKvHeadsByFloorRatio()
    {
        const int HeadDim = 2;
        const int Heads = 4; // heads map to kv-heads h * 2 / 4: {0,1} -> 0 and {2,3} -> 1
        const int KvHeads = 2;

        using CpuBackend backend = new(threads: 1, allowAvx2: true);
        using BlockPool pool = new(blockTokens: 4, layerCount: 1, kvHeads: KvHeads, headDim: HeadDim, capacityBlocks: 2, backend);
        using SequenceKv kv = new(pool);

        kv.Append();
        float[] kRow = [7f, -3f, 0.5f, 9f]; // kv-head0 then kv-head1
        float[] vRow = [3f, -4f, 5f, 6f];
        kv.WriteK(0, 0, kRow);
        kv.WriteV(0, 0, vRow);

        // Zero queries make every score exactly zero: each head's output must equal its kv-head's V verbatim.
        float[] q = new float[Heads * HeadDim];
        float scale = 1f / MathF.Sqrt(HeadDim);

        float[] outConcat = new float[Heads * HeadDim];
        LayerOps.CausalAttention(backend, q, Heads, KvHeads, HeadDim, kv, layer: 0, position: 0, scale, softcap: 0f, window: 0, outConcat);

        for (int h = 0; h < Heads; h++)
        {
            int expectedKvHead = h * KvHeads / Heads; // 0,0,1,1
            Assert.True(
                outConcat.AsSpan(h * HeadDim, HeadDim).SequenceEqual(vRow.AsSpan(expectedKvHead * HeadDim, HeadDim)),
                $"head {h} must read kv-head {expectedKvHead}: got [{outConcat[h * HeadDim]:G6}, {outConcat[h * HeadDim + 1]:G6}]");
        }
    }

    // ---------------------------------------------------------------- ModelDesc parsing

    [Fact]
    public void ModelDesc_RealSmolLMFixture_MatchesFileMetadata()
    {
        string? modelPath = ResolveModelPath();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        ModelDesc desc = ModelDesc.FromGguf(gguf);

        // Values read from the actual container, not assumed: SmolLM-135M-Instruct is a 30-layer hidden-576 llama.
        Assert.Equal("llama", desc.Architecture);
        Assert.Equal(30, desc.LayerCount);
        Assert.Equal(576, desc.HiddenSize);
        Assert.Equal(9, desc.HeadCount);
        Assert.Equal(3, desc.KvHeadCount);
        Assert.Equal(64, desc.HeadDim); // derived hidden/head_count: the file declares no attention.key_length
        Assert.Equal(1536, desc.IntermediateSize);
        Assert.Equal(10_000f, desc.RopeTheta);
        Assert.True(Math.Abs(desc.EpsRmsNorm - 1e-5f) < 1e-7f, $"rms epsilon {desc.EpsRmsNorm:G9}");
        Assert.Equal(49_152, desc.VocabSize);

        // The fixture ships tied weights: it has no output.weight tensor.
        Assert.False(desc.HasOutputWeights);
    }

    [Fact]
    public void ModelDesc_UnsupportedArchitecture_NamesSupportedArchitectures()
    {
        // The tensor table carries a name the llama-shaped map cannot name, so the generic fallback is
        // genuinely rejected and the registry wording — with its supported-profile list — surfaces.
        byte[] image = BuildGguf([("general.architecture", "bert")], [new TensorDef("layers.0.attention.query.weight", [4, 4])]);

        using GgufFile gguf = GgufFile.Open(WriteTemp(image));
        ModelException ex = Assert.Throws<ModelException>(() => ModelDesc.FromGguf(gguf));
        Assert.Contains("'bert' is not supported", ex.Message);
        Assert.Contains("llama, qwen2", ex.Message);
        Assert.Contains("llama-shaped fallback", ex.Message);
        Assert.Contains("layers.0.attention.query.weight", ex.Message);

        // Missing architecture key entirely must also name the supported set.
        using GgufFile ggufNoArch = GgufFile.Open(WriteTemp(BuildGguf([("general.name", "mystery")], [])));
        ModelException archEx = Assert.Throws<ModelException>(() => ModelDesc.FromGguf(ggufNoArch));
        Assert.Contains("general.architecture", archEx.Message);
    }

    [Fact]
    public void ModelDesc_MissingRequiredKey_ThrowsWithKeyName()
    {
        List<(string Key, object Value)> pairs = [.. RequiredLlamaMetadata()];
        pairs.RemoveAll(p => p.Key == "llama.block_count"); // drop one required integer

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf([.. pairs], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelDesc.FromGguf(gguf));
        Assert.Contains("llama.block_count", ex.Message);
    }

    [Fact]
    public void ModelDesc_RopeScalingVariant_IsRejectedWithKeyNames()
    {
        List<(string Key, object Value)> pairs = [.. RequiredLlamaMetadata(), ("llama.rope.scaling.type", "linear")];

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf([.. pairs], [])));
        ModelException ex = Assert.Throws<ModelException>(() => ModelDesc.FromGguf(gguf));
        Assert.Contains("rope scaling", ex.Message);
        Assert.Contains("llama.rope.scaling.type", ex.Message);

        // A partial rotary slice is accepted: the leading dimensions rotate and the rest pass through.
        List<(string Key, object Value)> partial = [.. RequiredLlamaMetadata(), ("llama.rope.dimension_count", 32)];
        using GgufFile ggufPartial = GgufFile.Open(WriteTemp(BuildGguf([.. partial], [])));
        Assert.Equal(64, ModelDesc.FromGguf(ggufPartial).HeadDim);

        // The fixture-consistent full-dimension slice (64 == 576/9) must pass.
        List<(string Key, object Value)> exact = [.. RequiredLlamaMetadata(), ("llama.rope.dimension_count", 64)];
        using GgufFile ggufOk = GgufFile.Open(WriteTemp(BuildGguf([.. exact], [])));
        Assert.Equal(64, ModelDesc.FromGguf(ggufOk).HeadDim);
    }

    [Fact]
    public void ModelDesc_Qwen2Keys_AreParsedUnderTheQwenPrefix()
    {
        List<(string Key, object Value)> pairs = [.. RequiredLlamaMetadata("qwen2")]; // same canonical keys under a qwen2 prefix

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf([.. pairs], [])));
        ModelDesc desc = ModelDesc.FromGguf(gguf);

        Assert.Equal("qwen2", desc.Architecture);
        Assert.Equal(576, desc.HiddenSize); // parsed from qwen2.embedding_length in the same synthetic values
    }

    [Fact]
    public void ModelDesc_ExplicitHeadDim_IsHonored_AndTheDerivedDimMustSplitHiddenExactly()
    {
        // A declared explicit head dimension is honored even when head_count x head_dim differs from
        // embedding_length (the gemma-shaped case): the plan validates the projection shapes at Open instead.
        List<(string Key, object Value)> independent = [.. RequiredLlamaMetadata(), ("llama.attention.key_length", 70), ("llama.attention.value_length", 70)];

        using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf([.. independent], [])));
        Assert.Equal(70, ModelDesc.FromGguf(gguf).HeadDim); // 9 x 70 != 576: the declared dimension wins

        List<(string Key, object Value)> disagree = [.. RequiredLlamaMetadata(), ("llama.attention.key_length", 64), ("llama.attention.value_length", 32)];
        using GgufFile ggufBad = GgufFile.Open(WriteTemp(BuildGguf([.. disagree], [])));
        Assert.Contains("disagree", Assert.Throws<ModelException>(() => ModelDesc.FromGguf(ggufBad)).Message);

        List<(string Key, object Value)> consistent = [.. RequiredLlamaMetadata(), ("llama.attention.key_length", 64), ("llama.attention.value_length", 64)];
        using GgufFile ggufOk = GgufFile.Open(WriteTemp(BuildGguf([.. consistent], [])));
        Assert.Equal(64, ModelDesc.FromGguf(ggufOk).HeadDim);

        // Without any explicit key the derived dimension must split hidden exactly.
        List<(string Key, object Value)> indivisible = [.. RequiredLlamaMetadata(), ("llama.attention.head_count", 7)]; // 576 % 7 != 0
        using GgufFile ggufIndiv = GgufFile.Open(WriteTemp(BuildGguf([.. indivisible], [])));
        Assert.Contains("does not divide", Assert.Throws<ModelException>(() => ModelDesc.FromGguf(ggufIndiv)).Message);
    }

    // ---------------------------------------------------------------- Open-time tensor validation + tiny forward

    [Fact]
    public void Open_MissingLayerTensor_ThrowsNamingTheFirstMissingTensor()
    {
        TinyModelSpec spec = new();
        using CpuBackend backend = new();

        List<TensorDef> tensors = [.. spec.RequiredTensors];
        tensors.RemoveAll(t => t.Name == "blk.0.attn_q.weight"); // drop a mid-list tensor

        {
            using GgufFile gguf = GgufFile.Open(WriteTemp(BuildGguf(spec.MetadataPairs(), [.. tensors])));
            ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(gguf, backend, maxContextTokens: 8));
            Assert.Contains("blk.0.attn_q.weight", ex.Message);
        }

        // Missing global tensors are named the same way.
        List<TensorDef> noNorm = [.. spec.RequiredTensors];
        noNorm.RemoveAll(t => t.Name == "output_norm.weight");
        {
            using GgufFile ggufNoNorm = GgufFile.Open(WriteTemp(BuildGguf(spec.MetadataPairs(), [.. noNorm])));
            ModelException normEx = Assert.Throws<ModelException>(() => TransformerModel.Open(ggufNoNorm, backend, maxContextTokens: 8));
            Assert.Contains("output_norm.weight", normEx.Message);

            List<TensorDef> noEmbed = [.. spec.RequiredTensors];
            noEmbed.RemoveAll(t => t.Name == "token_embd.weight");
            using GgufFile ggufNoEmbed = GgufFile.Open(WriteTemp(BuildGguf(spec.MetadataPairs(), [.. noEmbed])));
            Assert.Contains("token_embd.weight", Assert.Throws<ModelException>(() => TransformerModel.Open(ggufNoEmbed, backend, maxContextTokens: 8)).Message);
        }

        // A layer tensor with the wrong shape is rejected by name and dimensions.
        List<TensorDef> badShape = [.. spec.RequiredTensors];
        int idx = badShape.FindIndex(t => t.Name == "blk.0.ffn_gate.weight");
        badShape[idx] = new TensorDef("blk.0.ffn_gate.weight", [4, 3]); // gate must be logical [8, 4]

        {
            using GgufFile ggufBad = GgufFile.Open(WriteTemp(BuildGguf(spec.MetadataPairs(), [.. badShape])));
            ModelException shapeEx = Assert.Throws<ModelException>(() => TransformerModel.Open(ggufBad, backend, maxContextTokens: 8));
            Assert.Contains("blk.0.ffn_gate.weight", shapeEx.Message);
            Assert.Contains("[8, 4]", shapeEx.Message);
        }
    }

    [Fact]
    public void TinyTiedModel_ForwardIsFiniteAndDeterministicAcrossRuns()
    {
        TinyModelSpec spec = new();
        string path = WriteTemp(BuildGguf(spec.MetadataPairs(), [.. spec.RequiredTensors])); // no output.weight: tied logits

        float[] run1;
        float maxMagnitude;

        {
            using GgufFile ggufA = GgufFile.Open(path);
            using CpuBackend backendA = new(threads: 2, allowAvx2: true);
            using TransformerModel model = TransformerModel.Open(ggufA, backendA, maxContextTokens: 8);

            Assert.Equal(NormStrategy.RmsNorm, model.Plan.Norm);
            Assert.Equal(ActivationStrategy.SwiGLU, model.Plan.Activation);
            Assert.False(model.Desc.HasOutputWeights); // tied path selected from the tensor table

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks: 8, backendA);
            using SequenceKv seqA = new(pool);
            using SequenceKv seqB = new(pool);

            run1 = model.DecodeNext(3, seqA).ToArray(); // first token of sequence A at position 0
            Assert.Equal(model.Desc.VocabSize, run1.Length);

            foreach (float v in model.DecodeNext(4, seqB)) // fresh sequence: buffer reuse must not leak state
            {
                Assert.True(float.IsFinite(v), $"logit {v} is not finite");
            }

            maxMagnitude = MaxAbsMag(run1);
        }

        float[] run2;
        {
            using GgufFile ggufB = GgufFile.Open(path);
            using CpuBackend backendB = new(threads: 4, allowAvx2: true);
            using TransformerModel model = TransformerModel.Open(ggufB, backendB, maxContextTokens: 8);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks: 8, backendB);
            using SequenceKv seqA = new(pool);

            run2 = model.DecodeNext(3, seqA).ToArray(); // same token on a second instance with another thread count
        }

        Assert.True(run1.AsSpan().SequenceEqual(run2.AsSpan()), "decode must be bit-identical across runs and thread counts");
        foreach (float v in run2)
        {
            Assert.True(float.IsFinite(v), $"logit {v} is not finite");
        }

        Assert.True(maxMagnitude > 1e-6f, $"tiny model forward produced a degenerate logit vector (max |v| = {maxMagnitude:G6})");

        // Prefill continuation: two appended positions, then a third token forwards at position 2.
        {
            using GgufFile ggufC = GgufFile.Open(path);
            using CpuBackend backendC = new();
            using TransformerModel model = TransformerModel.Open(ggufC, backendC, maxContextTokens: 8);

            using BlockPool pool = new(blockTokens: 4, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks: 4, backendC);
            using SequenceKv kv = new(pool);

            model.Prefill([4, 5], kv); // token ids inside the synthetic vocabulary of six tokens
            Assert.Equal(2, kv.Length);

            foreach (float v in model.DecodeNext(1, kv))
            {
                Assert.True(float.IsFinite(v), $"logit {v} after prefill is not finite");
            }

            Assert.Throws<ArgumentOutOfRangeException>(() => model.DecodeNext(model.Desc.VocabSize, kv)); // out-of-vocabulary id
        }
    }

    // ---------------------------------------------------------------- integration smoke over the real fixture

    [Fact]
    public void SmolLM_F16_DecodeSmoke_IsFiniteDeterministicWithStableTop5()
    {
        string? modelPath = ResolveModelPath();
        if (modelPath is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        const int decodeSteps = 5;

        using GgufFile gguf = GgufFile.Open(modelPath);
        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        IReadOnlyList<int> promptIds = tokenizer.Encode("Hello");
        Assert.NotEmpty(promptIds);

        // Prefill discards logits by contract, so the first decode step is seeded with a deterministic plausible
        // continuation token rather than a sampled one; later steps follow greedy top-1 for an identical path.
        int seedTokenId = tokenizer.Encode(" world")[0];

        (int[][] TopFive, float[] FirstStepLogits, float MaxActivation, double PrefillMs, double AvgDecodeMs) RunOnce(bool threaded)
        {
            using CpuBackend backend = threaded ? new(threads: 4) : new(); // auto vs fixed thread counts must not change results
            using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 64);

            Assert.Equal("llama", model.Desc.Architecture);
            Assert.False(model.Desc.HasOutputWeights); // tied lm_head path over token_embd rows

            using BlockPool pool = new(blockTokens: 16, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks: 4, backend);
            using SequenceKv kv = new(pool);

            Stopwatch sw = Stopwatch.StartNew();
            model.Prefill(promptIds, kv);
            double prefillMs = sw.Elapsed.TotalMilliseconds; // snapshot before Restart clears it

            int[][] topFive = new int[decodeSteps][];
            float[] firstStepLogits = [];
            float maxActivation = 0f;
            double decodeTotalMs = 0.0;
            int nextId = seedTokenId;

            for (int step = 0; step < decodeSteps; step++)
            {
                sw.Restart();
                Span<float> logits = model.DecodeNext(nextId, kv);
                decodeTotalMs += sw.Elapsed.TotalMilliseconds; // read while running: a safe snapshot of this step

                foreach (float v in logits)
                {
                    Assert.True(float.IsFinite(v), $"step {step}: logit {v} is not finite");
                    maxActivation = Math.Max(maxActivation, Math.Abs(v));
                }

                topFive[step] = TopIds(logits, 5); // stable descending ids (ties break toward the lower id)
                if (step == 0)
                {
                    firstStepLogits = logits.ToArray();
                }

                nextId = topFive[step][0]; // greedy continuation keeps both runs on the identical path
            }

            Assert.Equal(promptIds.Count + decodeSteps, kv.Length);

            return (topFive, firstStepLogits, maxActivation, prefillMs, decodeTotalMs / decodeSteps);
        }

        var runA = RunOnce(threaded: false);
        var runB = RunOnce(threaded: true);

        Assert.True(
            runA.FirstStepLogits.AsSpan().SequenceEqual(runB.FirstStepLogits.AsSpan()),
            "first-step logits must be bit-identical across thread counts");

        for (int step = 0; step < decodeSteps; step++)
        {
            Assert.Equal(runA.TopFive[step], runB.TopFive[step]); // top-5 ids stable between independent runs
        }

        Assert.True(runA.MaxActivation > 1e-3f && float.IsFinite(runA.MaxActivation), $"activation magnitude {runA.MaxActivation:G6} looks degenerate");

        Console.WriteLine(
            $"SmolLM decode smoke: prefill {promptIds.Count} tokens in {runA.PrefillMs:F1} ms (threaded {runB.PrefillMs:F1} ms); " +
            $"avg decode {runA.AvgDecodeMs:F2} ms/token (threaded {runB.AvgDecodeMs:F2}); max activation magnitude {runA.MaxActivation:G6}; " +
            $"top-5 ids step 0 [{string.Join(", ", runA.TopFive[0])}], final step [{string.Join(", ", runA.TopFive[^1])}]");
    }

    // ---------------------------------------------------------------- shared helpers

    private static float MaxAbsMag(ReadOnlySpan<float> values)
    {
        float max = 0f;
        foreach (float v in values)
        {
            max = Math.Max(max, Math.Abs(v));
        }

        return max;
    }

    /// <summary>Picks the top ids by descending logit value with a strict "greater than" scan: equal logits keep the lower id.</summary>
    private static int[] TopIds(ReadOnlySpan<float> logits, int count)
    {
        bool[] taken = new bool[logits.Length];
        int[] ids = new int[count];

        for (int c = 0; c < count; c++)
        {
            float best = float.NegativeInfinity;
            int bestIdx = -1;

            for (int i = 0; i < logits.Length; i++)
            {
                if (!taken[i] && logits[i] > best)
                {
                    best = logits[i];
                    bestIdx = i;
                }
            }

            Assert.True(bestIdx >= 0, "logit vector had no selectable entry");
            ids[c] = bestIdx;
            taken[bestIdx] = true;
        }

        return ids;
    }

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-layers-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    // The test host's working directory varies per runner: walk up from the assembly location to find the repo fixture.
    private static string? ResolveModelPath() => ModelFixture.Resolve();

    // Canonical llama/qwen2 metadata pairs with the same values as the real fixture (parse-only fixtures need no tensors).
    private static List<(string Key, object Value)> RequiredLlamaMetadata(string prefix = "llama") =>
    [
        ("general.architecture", prefix),
        ($"{prefix}.embedding_length", 576),
        ($"{prefix}.block_count", 30),
        ($"{prefix}.attention.head_count", 9),
        ($"{prefix}.attention.head_count_kv", 3),
        ($"{prefix}.feed_forward_length", 1536),
        ($"{prefix}.attention.layer_norm_rms_epsilon", 1e-5f),
        ($"{prefix}.vocab_size", 49_152),
    ];

    private sealed record TensorDef(string Name, int[] Dims); // dims in logical (row-major) order; vectors are one-dimensional

    /// <summary>A complete single-layer llama with hidden 4, heads 2, kv-heads 2, ffn 8, vocab 6 — all F32 tensors.</summary>
    private sealed class TinyModelSpec
    {
        public List<(string Key, object Value)> MetadataPairs() =>
        [
            ("general.architecture", "llama"),
            ("llama.embedding_length", 4),
            ("llama.block_count", 1),
            ("llama.attention.head_count", 2),
            ("llama.attention.head_count_kv", 2),
            ("llama.feed_forward_length", 8),
            ("llama.attention.layer_norm_rms_epsilon", 1e-5f),
            ("llama.vocab_size", 6),
        ];

        public List<TensorDef> RequiredTensors =>
        [
            new("token_embd.weight", [6, 4]),
            new("blk.0.attn_q.weight", [4, 4]), // head_count * head_dim == hidden == 4
            new("blk.0.attn_k.weight", [4, 4]), // kv_heads * head_dim == 4
            new("blk.0.attn_v.weight", [4, 4]),
            new("blk.0.attn_output.weight", [4, 4]),
            new("blk.0.ffn_gate.weight", [8, 4]),
            new("blk.0.ffn_up.weight", [8, 4]),
            new("blk.0.ffn_down.weight", [4, 8]),
            new("blk.0.attn_norm.weight", [4]),
            new("blk.0.ffn_norm.weight", [4]),
            new("output_norm.weight", [4]), // no output.weight: the tied path is exercised instead
        ];
    }

    /// <summary>Builds a GGUF v3 image from KV pairs and F32 tensors with synthetic-but-nondegenerate payloads.</summary>
    private static byte[] BuildGguf(IReadOnlyList<(string Key, object Value)> kvPairs, IReadOnlyList<TensorDef> tensors)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(GgufMagic);
        writer.Write(3u); // version
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)kvPairs.Count);

        foreach ((string key, object value) in kvPairs)
        {
            WriteGgufString(writer, key);
            switch (value)
            {
                case string s:
                    writer.Write(8u); // STRING
                    WriteGgufString(writer, s);
                    break;
                case int i:
                    writer.Write(5u); // INT32
                    writer.Write(i);
                    break;
                case float f:
                    writer.Write(6u); // FLOAT32
                    writer.Write(f);
                    break;
                default:
                    throw new ArgumentException($"Unsupported fixture value type {value.GetType().Name}.", nameof(kvPairs));
            }
        }

        long offset = 0;
        foreach (TensorDef tensor in tensors)
        {
            // GGUF stores dims fastest-varying first: the logical row-major order is written reversed.
            int[] stored = new int[tensor.Dims.Length];
            for (int d = 0; d < tensor.Dims.Length; d++)
            {
                stored[d] = tensor.Dims[tensor.Dims.Length - 1 - d];
            }

            WriteGgufString(writer, tensor.Name);
            writer.Write((uint)stored.Length);
            foreach (int dim in stored)
            {
                writer.Write((ulong)dim);
            }

            writer.Write(0u); // ggml type F32
            writer.Write((ulong)offset);

            long elements = 1;
            foreach (int dim in tensor.Dims)
            {
                elements *= dim;
            }

            offset += elements * 4;
        }

        long headerLength = stream.Position;
        long dataStart = (headerLength + 31) / 32 * 32; // GGUF default alignment is 32
        for (long i = headerLength; i < dataStart; i++)
        {
            writer.Write((byte)0);
        }

        int ordinal = 0;
        foreach (TensorDef tensor in tensors)
        {
            long elements = 1;
            foreach (int dim in tensor.Dims)
            {
                elements *= dim;
            }

            bool isNorm = tensor.Name.Contains("norm");
            for (long e = 0; e < elements; e++)
            {
                float value = MathF.Sin((ordinal * 17 + e) * 0.13f); // deterministic, finite values everywhere
                writer.Write(isNorm ? 1.25f + value : value);       // norm weights stay clearly nonzero
            }

            ordinal++;
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
}
