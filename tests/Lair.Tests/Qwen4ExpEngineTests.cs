using System.Text;

using Lair.Backends.Cpu;
using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;

namespace Lair.Tests;

/// <summary>
/// Engine-seam coverage of the bounded synthetic qwen4exp forward path against small synthetic GGUF fixtures only — never
/// the 75 GB target pair. A valid fixture now opens on the internal qwen4exp engine instead of refusing: every block runs
/// in the reference order over the file's own tensors, the gated-delta-net recurrent state and convolution history carry
/// across chunk splits, the sparse block's whole-sequence caches and indexer selection carry unchanged, a ratio-2 layer
/// runs the pooling selection deterministically, and a declared shared expert gates under the qwen4exp reference's plain
/// sigmoid — finite at a zero score where qwen2moe's silu(score)/score divides 0 by 0. The serving routes (Prefill,
/// DecodeNext, DecodeBatch, LastLogits) refuse against the seam because persistent production caches for this family do
/// not exist yet, and a fixture whose material is not F32 is refused naming the format.
/// </summary>
[Trait("Category", "Engine")]
public sealed class Qwen4ExpEngineTests : IDisposable
{
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
                // temp cleanup is best-effort
            }
        }
    }

    private const int Layers = 4;          // ratios [0,0,0,1]: three linear blocks, one sparse full block per group of four
    private const int Hidden = 8;
    private const int HeadCount = 2;
    private const int HeadDim = 4;         // hidden / head_count — even, so the engine rotates full heads
    private const int KvHeadCount = 1;
    private const int StateDim = 2;        // ssm.state_size — the shared head dimension every value head carries
    private const int InnerSize = 4;       // ssm.inner_size over state dim: two derived value heads per linear block
    private const int TimeStepRank = 2;    // must equal the derived value-head count exactly
    private const int GroupCount = 1;      // key-head groups → conv channels 2 * (1 x 2) + 4 = 8 == hidden
    private const int ConvKernel = 3;
    private const int IndexerHeads = 2;
    private const int IndexerKeyLength = 4; // even component width so the engine rotates indexer heads in pairs
    private const int TopKCells = 6;
    private const int StreamCount = 2;     // flattened residual width: 2 x hidden = 16
    private const int LowRank = 3;
    private const int ExpertCount = 2;
    private const int ExpertsPerToken = 1;
    private const int IntermediateSize = 6; // declared only as qwen4exp.expert_feed_forward_length: no dense key exists
    private const int VocabSize = 5;

    private static readonly AttentionKind[] ExpectedKinds =
    [
        AttentionKind.LinearGatedDeltaNet,
        AttentionKind.LinearGatedDeltaNet,
        AttentionKind.LinearGatedDeltaNet,
        AttentionKind.FullQuerySparseAttention,
    ];

    // ---------------------------------------------------------------- the seam opens where the refusal used to stand

    [Fact]
    public void TransformerModel_Qwen4ExpFixture_OpensTheBoundedForwardSeamAndProducesFiniteDeterministicLogits()
    {
        using GgufFile file = OpenFixture(BuildGguf(MetadataPairs(), FixtureTensors()));
        using CpuBackend backend = new();

        using TransformerModel model = TransformerModel.Open(file, backend, maxContextTokens: 8);

        Assert.True(model.HasQwen4ExpForward); // the resolved geometry composes now: the engine seam reads this fixture's own tensors

        Qwen4ExpSequence sequence = model.NewQwen4ExpSequence();
        Span<float> logits = model.ForwardQwen4ExpChunk(sequence, [0, 1, 2]);

        Assert.Equal(3 * VocabSize, logits.Length); // one vocab-sized row per forwarded token
        foreach (float value in logits)
        {
            Assert.True(float.IsFinite(value), "every synthetic qwen4exp logit must land finite");
        }

        // The seam's last-logits read is exactly the chunk's final row.
        Assert.True(model.LastQwen4ExpLogits(sequence).SequenceEqual(logits.Slice(2 * VocabSize, VocabSize)));

        // A second identical run over a fresh sequence reproduces the first bit-for-bit: no per-run randomness anywhere in the composition.
        Qwen4ExpSequence repeat = model.NewQwen4ExpSequence();
        Span<float> again = model.ForwardQwen4ExpChunk(repeat, [0, 1, 2]);
        Assert.True(again.SequenceEqual(logits));
    }

    // ---------------------------------------------------------------- carried state across chunk splits

    [Fact]
    public void TransformerModel_Qwen4ExpRecurrentStateAndSparseCaches_CarryAcrossChunkSplits()
    {
        using GgufFile file = OpenFixture(BuildGguf(MetadataPairs(), FixtureTensors()));
        using CpuBackend backend = new();

        using TransformerModel model = TransformerModel.Open(file, backend, maxContextTokens: 8);

        // One unsplit pass of three tokens: the three linear blocks advance their delta-rule slots and convolution history
        // through the GDN chunk, and the sparse block appends its KV rows and raw indexer keys as it goes.
        Qwen4ExpSequence whole = model.NewQwen4ExpSequence();
        Span<float> wholeLogits = model.ForwardQwen4ExpChunk(whole, [0, 1, 2]);

        // The same tokens split at a chunk boundary: the carried state must make the split run land bit-for-bit on the unsplit one.
        Qwen4ExpSequence split = model.NewQwen4ExpSequence();
        model.ForwardQwen4ExpChunk(split, [0, 1]);
        Span<float> splitTail = model.ForwardQwen4ExpChunk(split, [2]);

        Assert.True(splitTail.SequenceEqual(wholeLogits.Slice(2 * VocabSize, VocabSize)), "a chunk boundary must not move a position, a carried state value, or a logit");
        Assert.True(model.LastQwen4ExpLogits(split).SequenceEqual(model.LastQwen4ExpLogits(whole)));
    }

    [Fact]
    public void TransformerModel_Qwen4ExpIndexerSelection_IsDeterministicAcrossRunsAndChunkSplits()
    {
        // A ratio-2 sparse layer pools complete blocks of two cells, ranks them through the indexer budget (top_k 6 at ratio
        // 2 keeps three pools), and restricts attention to the selected cells — the same selection the layer-level QSA
        // fixtures pin, now reached through the engine seam.
        List<(string Key, object Value)> pairs = MetadataPairs();
        int ratioIndex = pairs.FindIndex(p => p.Key == "qwen4exp.attention.compress_ratios");
        pairs[ratioIndex] = ("qwen4exp.attention.compress_ratios", new int[] { 0, 0, 0, 2 });

        using GgufFile file = OpenFixture(BuildGguf(pairs, FixtureTensors()));
        using CpuBackend backend = new();

        using TransformerModel model = TransformerModel.Open(file, backend, maxContextTokens: 8);

        Qwen4ExpSequence whole = model.NewQwen4ExpSequence();
        Span<float> wholeLogits = model.ForwardQwen4ExpChunk(whole, [0, 1, 2, 3, 4, 0]);

        Qwen4ExpSequence split = model.NewQwen4ExpSequence();
        Span<float> splitHead = model.ForwardQwen4ExpChunk(split, [0, 1, 2]);
        Assert.True(splitHead.SequenceEqual(wholeLogits.Slice(0, 3 * VocabSize)), "selection over a pooled cache must not change across a chunk boundary: the carried head rows");

        Span<float> splitTail = model.ForwardQwen4ExpChunk(split, [3, 4, 0]);

        // Each row of the second chunk, compared against the same position in the unsplit run: a chunk boundary must not
        // move a pooled selection, a carried cache cell, or a logit.
        Assert.True(splitTail.Slice(0, VocabSize).SequenceEqual(wholeLogits.Slice(3 * VocabSize, VocabSize)), "selection over a pooled cache must not change across a chunk boundary: row 3");
        Assert.True(splitTail.Slice(VocabSize, VocabSize).SequenceEqual(wholeLogits.Slice(4 * VocabSize, VocabSize)), "selection over a pooled cache must not change across a chunk boundary: row 4");
        Assert.True(splitTail.Slice(2 * VocabSize, VocabSize).SequenceEqual(wholeLogits.Slice(5 * VocabSize, VocabSize)), "selection over a pooled cache must not change across a chunk boundary: row 5");

        Qwen4ExpSequence repeat = model.NewQwen4ExpSequence();
        Span<float> repeated = model.ForwardQwen4ExpChunk(repeat, [0, 1, 2, 3, 4, 0]);
        Assert.True(repeated.SequenceEqual(wholeLogits), "the same selection must reproduce bit-for-bit on a fresh sequence");
    }

    // ---------------------------------------------------------------- the qwen4exp shared-expert gate posture

    [Fact]
    public void TransformerModel_Qwen4ExpSharedExpert_GatesUnderPlainSigmoidFiniteAtZeroScore()
    {
        // The target declares a shared expert beside the routed ones; this fixture ships one whose selector vector is all
        // zeros, so the shared block's score is exactly zero for every token. The qwen4exp reference gates it with a plain
        // sigmoid — sigmoid(0) = 0.5, finite — where qwen2moe's silu(score) / score divides 0 by 0 and poisons the whole
        // forward with NaN. A run that lands all-finite proves the seam applies the qwen4exp posture.
        List<(string Key, object Value)> pairs = MetadataPairs();
        pairs.Add(("qwen4exp.expert_shared_count", 1));

        List<TensorDef> tensors = [.. FixtureTensors()];
        for (int layer = 0; layer < Layers; layer++)
        {
            tensors.Add(new TensorDef($"blk.{layer}.ffn_gate_shexp.weight", [IntermediateSize, Hidden]));
            tensors.Add(new TensorDef($"blk.{layer}.ffn_up_shexp.weight", [IntermediateSize, Hidden]));
            tensors.Add(new TensorDef($"blk.{layer}.ffn_down_shexp.weight", [Hidden, IntermediateSize]));
            tensors.Add(new TensorDef($"blk.{layer}.ffn_gate_inp_shexp.weight", [Hidden])); // zero-filled by the builder below
        }

        HashSet<string> zeroFilled = ["blk.0.ffn_gate_inp_shexp.weight", "blk.1.ffn_gate_inp_shexp.weight", "blk.2.ffn_gate_inp_shexp.weight", "blk.3.ffn_gate_inp_shexp.weight"];
        using GgufFile file = OpenFixture(BuildGguf(pairs, tensors, zeroFilledNames: zeroFilled));
        using CpuBackend backend = new();

        using TransformerModel model = TransformerModel.Open(file, backend, maxContextTokens: 8);

        Assert.True(model.Plan.SparseFfn!.HasSharedExpert);

        Qwen4ExpSequence sequence = model.NewQwen4ExpSequence();
        Span<float> logits = model.ForwardQwen4ExpChunk(sequence, [0, 1, 2]);

        foreach (float value in logits)
        {
            Assert.True(float.IsFinite(value), "a zero-score shared expert must gate through sigmoid(0), not a 0/0 division");
        }
    }

    // ---------------------------------------------------------------- the seam refuses what it cannot read or compose

    [Fact]
    public void TransformerModel_Qwen4ExpSeam_RefusesNonF32FixtureMaterialNamingTheFormat()
    {
        // The bounded engine reads F32 fixture rows through the backend's row dequantization; a quantized posture needs the
        // production weight views and kernels this synthetic path does not carry, so it is refused naming the tensor and format.
        List<TensorDef> tensors = [.. FixtureTensors()];
        int normIndex = tensors.FindIndex(t => t.Name == "blk.0.ssm_norm.weight");
        tensors[normIndex] = new TensorDef("blk.0.ssm_norm.weight", [StateDim], GgmlTypeF16);

        using GgufFile file = OpenFixture(BuildGguf(MetadataPairs(), tensors));
        using CpuBackend backend = new();

        ModelException ex = Assert.Throws<ModelException>(() => TransformerModel.Open(file, backend, maxContextTokens: 8));
        Assert.Contains("blk.0.ssm_norm.weight", ex.Message);
        Assert.Contains("f16", ex.Message);
    }

    [Fact]
    public void TransformerModel_Qwen4ExpSeam_ServingRoutesRefuseBecauseProductionCachesDoNotExistYet()
    {
        using GgufFile file = OpenFixture(BuildGguf(MetadataPairs(), FixtureTensors()));
        using CpuBackend backend = new();

        using TransformerModel model = TransformerModel.Open(file, backend, maxContextTokens: 8);
        using BlockPool pool = new(16, model.Desc.LayerCount, model.Desc.KvHeadCount, model.Desc.HeadDim, 64, backend);
        using SequenceKv kv = new(pool);

        InvalidOperationException prefill = Assert.Throws<InvalidOperationException>(() => model.Prefill([0, 1], kv));
        Assert.Contains("bounded synthetic qwen4exp forward seam", prefill.Message);
        Assert.Contains("persistent production caches", prefill.Message);

        Assert.Contains("qwen4exp", Assert.Throws<InvalidOperationException>(() => model.DecodeNext(0, kv)).Message);
        Assert.Contains("qwen4exp", Assert.Throws<InvalidOperationException>(() => model.DecodeBatch([0], [kv])).Message);
        Assert.Contains("qwen4exp", Assert.Throws<InvalidOperationException>(() => model.LastLogits()).Message);

        // The seam itself stays reachable only through the assembly-internal chunk forwards, and the fixture never touched a
        // KV block pool: the refusal is what separates the synthetic path from the serving routes.
        Assert.Equal(0, kv.Length);
    }

    // ---------------------------------------------------------------- fixture builders (synthetic GGUF v3 images; array metadata supported)

    private static List<(string Key, object Value)> MetadataPairs() =>
        [
            ("general.architecture", "qwen4exp"),
            ("qwen4exp.embedding_length", Hidden),
            ("qwen4exp.block_count", Layers),
            ("qwen4exp.attention.head_count", HeadCount),
            ("qwen4exp.attention.head_count_kv", KvHeadCount),
            ("qwen4exp.expert_feed_forward_length", IntermediateSize),
            ("qwen4exp.attention.layer_norm_rms_epsilon", 1e-5f),
            ("qwen4exp.ssm.group_count", GroupCount),
            ("qwen4exp.ssm.state_size", StateDim),
            ("qwen4exp.ssm.inner_size", InnerSize),
            ("qwen4exp.ssm.time_step_rank", TimeStepRank),
            ("qwen4exp.ssm.conv_kernel", ConvKernel),
            ("qwen4exp.attention.compress_ratios", new int[] { 0, 0, 0, 1 }),
            ("qwen4exp.full_attention_interval", 4),
            ("qwen4exp.attention.indexer.head_count", IndexerHeads),
            ("qwen4exp.attention.indexer.key_length", IndexerKeyLength),
            ("qwen4exp.attention.indexer.top_k", TopKCells),
            ("qwen4exp.hyper_connection.count", StreamCount),
            ("qwen4exp.hyper_connection.low_rank", LowRank),
            ("qwen4exp.expert_count", ExpertCount),
            ("qwen4exp.expert_used_count", ExpertsPerToken),
            ("tokenizer.ggml.tokens", new string[] { "a", "b", "c", "d", "e" }),
        ];

    private static List<TensorDef> FixtureTensors()
    {
        List<TensorDef> tensors = GlobalDefs();

        for (int layer = 0; layer < Layers; layer++)
        {
            tensors.AddRange(LayerCommonDefs(layer));
            tensors.AddRange(ExpectedKinds[layer] == AttentionKind.LinearGatedDeltaNet ? LinearDefs(layer) : FullDefs(layer));
        }

        int flattened = StreamCount * Hidden;
        tensors.Add(new TensorDef("output_hc_down.weight", [LowRank, flattened]));
        tensors.Add(new TensorDef("output_hc_norm.weight", [flattened]));
        tensors.Add(new TensorDef("output_hc_up.weight", [flattened, LowRank]));

        return tensors;
    }

    private static List<TensorDef> GlobalDefs() =>
        [new("token_embd.weight", [VocabSize, Hidden]), new("output.weight", [VocabSize, Hidden])];

    private static List<TensorDef> LayerCommonDefs(int layer)
    {
        int flattened = StreamCount * Hidden;
        List<TensorDef> defs = [];

        foreach (string side in new[] { "hc_attn", "hc_ffn" })
        {
            defs.Add(new TensorDef($"blk.{layer}.{side}_inject.weight", [StreamCount, flattened]));
            defs.Add(new TensorDef($"blk.{layer}.{side}_down.weight", [LowRank, flattened]));
            defs.Add(new TensorDef($"blk.{layer}.{side}_up.weight", [flattened, LowRank]));
            defs.Add(new TensorDef($"blk.{layer}.{side}_norm.weight", [flattened]));
        }

        defs.Add(new TensorDef($"blk.{layer}.ffn_gate_inp.weight", [ExpertCount, Hidden]));
        defs.Add(new TensorDef($"blk.{layer}.ffn_gate_exps.weight", [ExpertCount, IntermediateSize, Hidden]));
        defs.Add(new TensorDef($"blk.{layer}.ffn_up_exps.weight", [ExpertCount, IntermediateSize, Hidden]));
        defs.Add(new TensorDef($"blk.{layer}.ffn_down_exps.weight", [ExpertCount, Hidden, IntermediateSize]));

        return defs;
    }

    private static List<TensorDef> LinearDefs(int layer) =>
        [
            new($"blk.{layer}.attn_qkv.weight", [2 * GroupCount * StateDim + InnerSize, Hidden]),
            new($"blk.{layer}.attn_gate.weight", [InnerSize, Hidden]),
            new($"blk.{layer}.ssm_a", [TimeStepRank]),
            new($"blk.{layer}.ssm_alpha.weight", [TimeStepRank, Hidden]),
            new($"blk.{layer}.ssm_beta.weight", [TimeStepRank, Hidden]),
            new($"blk.{layer}.ssm_conv1d.weight", [2 * GroupCount * StateDim + InnerSize, ConvKernel]),
            new($"blk.{layer}.ssm_dt.bias", [TimeStepRank]),
            new($"blk.{layer}.ssm_norm.weight", [StateDim]),
            new($"blk.{layer}.ssm_out.weight", [Hidden, InnerSize]),
        ];

    private static List<TensorDef> FullDefs(int layer) =>
        [
            // The engine reads the reference's interleaved query-and-gate rows: a query slice then its gate slice per head,
            // so the q projection carries 2 * head_count * head_dim rows, not the plain head_count * head_dim width.
            new($"blk.{layer}.attn_q.weight", [2 * HeadCount * HeadDim, Hidden]),
            new($"blk.{layer}.attn_k.weight", [KvHeadCount * HeadDim, Hidden]),
            new($"blk.{layer}.attn_v.weight", [KvHeadCount * HeadDim, Hidden]),
            new($"blk.{layer}.attn_output.weight", [Hidden, HeadCount * HeadDim]),
            new($"blk.{layer}.indexer.q_proj.weight", [IndexerHeads * IndexerKeyLength, Hidden]),
            new($"blk.{layer}.indexer.k_proj.weight", [IndexerKeyLength, Hidden]),
            new($"blk.{layer}.indexer.q_norm.weight", [IndexerKeyLength]),
            new($"blk.{layer}.indexer.k_norm.weight", [IndexerKeyLength]),
        ];

    private string OpenFixtureWriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-qwen4exp-engine-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, image);
        _tempFiles.Add(path);
        return path;
    }

    private GgufFile OpenFixture(byte[] image) => GgufFile.Open(OpenFixtureWriteTemp(image));

    private const uint GgmlTypeF32 = 0;
    private const uint GgmlTypeF16 = 1;

    // GGUF v3 builder with the metadata value types these fixtures need: strings, int32s, floats — and int32/string arrays.
    // Tensor payloads are F32 by default; a TensorDef can name the F16 ggml type to prove the engine refuses a format it
    // cannot read. Names listed in zeroFilledNames land as exact zeros instead of the deterministic fill.
    private static byte[] BuildGguf(
        IReadOnlyList<(string Key, object Value)> kvPairs,
        IReadOnlyList<TensorDef> tensors,
        IReadOnlySet<string>? zeroFilledNames = null)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        const uint GgufMagic = 0x46554747; // "GGUF" — the same constant the loader validates against

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
                case int[] ints:
                    writer.Write(9u); // ARRAY of INT32
                    writer.Write(5u);
                    writer.Write((ulong)ints.Length);
                    foreach (int i in ints)
                    {
                        writer.Write(i);
                    }

                    break;
                case string[] strings:
                    writer.Write(9u); // ARRAY of STRING
                    writer.Write(8u);
                    writer.Write((ulong)strings.Length);
                    foreach (string s in strings)
                    {
                        WriteGgufString(writer, s);
                    }

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

            writer.Write(tensor.GgmlType);
            writer.Write((ulong)offset);

            long elements = 1;
            foreach (int dim in tensor.Dims)
            {
                elements *= dim;
            }

            offset += elements * BytesPerElement(tensor.GgmlType);
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
            bool isZero = zeroFilledNames is not null && zeroFilledNames.Contains(tensor.Name);

            for (long e = 0; e < elements; e++)
            {
                float value = MathF.Sin((ordinal * 17 + e) * 0.13f); // deterministic, finite values everywhere
                float written = isZero ? 0f : isNorm ? 1.25f + value : value;

                if (tensor.GgmlType == GgmlTypeF16)
                {
                    writer.Write((Half)written);
                }
                else
                {
                    writer.Write(written);
                }
            }

            ordinal++;
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static long BytesPerElement(uint ggmlType) => ggmlType switch
    {
        GgmlTypeF32 => 4,
        GgmlTypeF16 => 2,
        _ => throw new ArgumentException($"Unsupported fixture ggml type {ggmlType}.", nameof(ggmlType)),
    };

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private sealed record TensorDef(string Name, int[] Dims, uint GgmlType = GgmlTypeF32); // dims in logical (row-major) order; vectors are one-dimensional
}
