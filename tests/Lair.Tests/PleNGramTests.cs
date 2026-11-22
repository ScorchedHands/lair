using System.Text;

using Lair.Contracts;
using Lair.Loader;
using Lair.Models;

namespace Lair.Tests;

/// <summary>
/// Storage and governance coverage for the PLE n-gram table seam: the geometry a model's GGUF metadata must declare before rows can be
/// gathered, the reference n-gram hash and history normalization that turn a token position into one row per head, the row gather that resolves
/// those rows out of the physical shard holding them without ever taking the whole table, and the residency ceiling that bounds how many gathered
/// rows a model may hold. Fixtures are synthetic GGUF files written here; the real pinned target is exercised by the ple-latency bench, not here.
/// </summary>
[Trait("Category", "Ple")]
public sealed class PleNGramTests : IDisposable
{
    private const uint GgufMagic = 0x46554747;
    private const uint GgmlF32 = 0;
    private const uint GgmlIQ4Nl = 20; // ggml type id for IQ4_NL: 18 bytes per 32-element block.

    private const string PleTensorName = "per_layer_token_embd.weight";
    private const int RowElements = 160; // 5 IQ4_NL blocks: 90 packed bytes per row, matching the pinned target's row width.
    private const int RowBytes = 90;

    private readonly List<string> _tempPaths = [];

    public void Dispose()
    {
        foreach (string path in _tempPaths)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
                // Best-effort cleanup of temp fixtures.
            }
        }
    }

    [Fact]
    public void FromGguf_ReadsDeclaredGeometryAgainstTheBackingTensor()
    {
        using GgufFile model = GgufFile.Open(WriteTemp(BuildPleGguf(PleMetadata(ngramSize: 3, headsPerNgram: 2), tableRows: 64)));
        PleNgramTable table = PleNgramTable.FromGguf(model, Tensor(model));

        Assert.Equal(3, table.NgramSize);
        Assert.Equal(2, table.HeadsPerNgram);
        Assert.Equal(4, table.HeadCount); // head count is (ngram_size - 1) * heads_per_ngram, exactly as the reference gather indexes heads.
        Assert.Equal(64L, table.TableRows);
        Assert.Equal(RowElements, table.RowElements);
        Assert.Equal((long)RowBytes, table.RowStrideBytes); // IQ4_NL packs 160 elements as five 18-byte blocks.
        Assert.Equal(QuantFormat.IQ4_NL, table.Tensor.Format);
        Assert.Equal(3, table.EosTokenId);

        Assert.Equal((0L, 16L), table.HeadRange(0));
        Assert.Equal((16L, 16L), table.HeadRange(1));
        Assert.Equal((32L, 16L), table.HeadRange(2));
        Assert.Equal((48L, 16L), table.HeadRange(3));
    }

    [Fact]
    public void ComputeRowIndices_IndexesEachHeadInsideItsOwnDeclaredRange()
    {
        using GgufFile model = GgufFile.Open(WriteTemp(BuildPleGguf(PleMetadata(ngramSize: 3, headsPerNgram: 2), tableRows: 64)));
        PleNgramTable table = PleNgramTable.FromGguf(model, Tensor(model));

        long[] rows = table.RowIndicesFor([1000, 2000, 3000]); // newest-first history: the token itself, then the two predecessors.

        Assert.Equal(4, rows.Length);

        // The heads of n-gram order n start at (n - 2) * heads_per_ngram: heads 0-1 take the bigram mix, heads 2-3 the trigram mix, each modulo its own vocabulary inside its own offset band.
        Assert.InRange(rows[0], 0L, 15L);
        Assert.InRange(rows[1], 16L, 31L);
        Assert.InRange(rows[2], 32L, 47L);
        Assert.InRange(rows[3], 48L, 63L);
    }

    [Fact]
    public void ComputeRowIndices_ReusesOneMixAcrossTheHeadsOfAnOrder()
    {
        // Vocabulary sizes chosen as 16 and 4 for the bigram heads (and 8 and 2 for the trigram heads): if both heads of an order index the same mix, the narrower head's row must agree with the wider head's row modulo the narrower size.
        using GgufFile model = GgufFile.Open(WriteTemp(BuildPleGguf(
            PleMetadata(ngramSize: 3, headsPerNgram: 2) with
            {
                HeadOffsets = [0, 32, 40, 48],
                HeadVocabSizes = [16, 4, 8, 2],
            },
            tableRows: 64)));
        PleNgramTable table = PleNgramTable.FromGguf(model, Tensor(model));

        long[] rows = table.RowIndicesFor([1000, 2000, 3000]);

        Assert.InRange(rows[0], 0L, 15L);
        Assert.InRange(rows[1], 32L, 35L);
        Assert.Equal(rows[0] % 4, rows[1] - 32); // the same order-2 mix indexes both bigram heads.

        Assert.InRange(rows[2], 40L, 47L);
        Assert.InRange(rows[3], 48L, 49L);
        Assert.Equal(rows[2] % 2, rows[3] - 48); // and the same order-3 mix indexes both trigram heads.
    }

    [Fact]
    public void ComputeRowIndices_HandCheckedMixAndOffsetArithmetic()
    {
        // Multipliers of 1 make this fixture's mix directly readable: row = (token XOR predecessor) % 8 + 16, with the head declared at offset 16 over a 24-row table.
        using GgufFile model = GgufFile.Open(WriteTemp(BuildPleGguf(SimpleMetadata(), tableRows: 24)));
        PleNgramTable table = PleNgramTable.FromGguf(model, Tensor(model));

        Assert.Equal([20L], table.RowIndicesFor([7, 3])); // 7 ^ 3 == 4; 4 % 8 + 16 == 20.
        Assert.Equal([18L], table.RowIndicesFor([3, 9])); // an EOS token keeps its own context: 3 ^ 9 == 10; 10 % 8 + 16 == 18.
    }

    [Fact]
    public void ComputeRowIndices_NormalizesMissingAndCutHistoryToEos()
    {
        using GgufFile model = GgufFile.Open(WriteTemp(BuildPleGguf(SimpleMetadata(), tableRows: 24)));
        PleNgramTable table = PleNgramTable.FromGguf(model, Tensor(model));

        // A predecessor with no cache cell reads as EOS (id 3 here), so a lone token and a window whose only predecessor is EOS gather the same row.
        Assert.Equal([22L], table.RowIndicesFor([5])); // 5 ^ 3 == 6; 6 % 8 + 16 == 22.
        Assert.Equal([22L], table.RowIndicesFor([5, 3]));

        // An EOS predecessor cuts everything at and behind it: the window [5, 9, 7] gathers as if its history began with the cut.
        Assert.Equal(table.RowIndicesFor([5]), table.RowIndicesFor([5, 3, 7]));
    }

    [Fact]
    public void MixNgram_WrapsUnsignedExactlyAsTheReferenceHash()
    {
        // Two's-complement extension of token ids and 64-bit wrap are the reference posture: -1 * 0x7FFFFFFF is 2^64 - 0x7FFFFFFF, so a mix built from two such products cancels to zero.
        Assert.Equal(0UL, PleNgramTable.MixNgram([int.MaxValue, int.MaxValue], [ulong.MaxValue, ulong.MaxValue], order: 2));

        // (2^64 - 0x7FFFFFFF) ^ 1 == 0xFFFFFFFF80000000.
        Assert.Equal(0xFFFFFFFF80000000UL, PleNgramTable.MixNgram([int.MaxValue, 1], [ulong.MaxValue, 1UL], order: 2));

        Assert.Throws<ArgumentException>(() => PleNgramTable.MixNgram([7], [1UL, 1UL], order: 2));
        Assert.Throws<ArgumentException>(() => PleNgramTable.MixNgram([7, 3], [1UL], order: 2));
    }

    [Fact]
    public void RowSource_ResolvesExactPackedRowsWithoutReadingTheWholeTable()
    {
        string path = WriteTemp(BuildPleGguf(PleMetadata(ngramSize: 2, headsPerNgram: 1) with { HeadOffsets = [0], HeadVocabSizes = [8] }, tableRows: 64));

        using GgufFile model = GgufFile.Open(path);
        using PleRowSource rows = new(model, PleNgramTable.FromGguf(model, Tensor(model)));

        Assert.Equal(64L, rows.Rows);
        Assert.Equal(RowBytes, rows.RowStrideBytes);
        Assert.Equal(Path.GetFullPath(path), Path.GetFullPath(rows.ShardPath)); // a single-file model gathers straight out of its own container.

        for (int rowIndex = 0; rowIndex < 3; rowIndex++)
        {
            ReadOnlySpan<byte> row = rows.ResolveRow(rowIndex);
            Assert.Equal(RowBytes, row.Length);

            for (int byteIndex = 0; byteIndex < RowBytes; byteIndex++)
            {
                Assert.Equal(ExpectedRowByte(rowIndex, byteIndex), row[byteIndex]); // addressing lands on this row's own packed bytes, not a neighbour's.
            }
        }

        ReadOnlySpan<byte> last = rows.ResolveRow(rows.Rows - 1); // the final declared row must land inside the tensor payload the gather addresses.
        Assert.Equal(ExpectedRowByte(63, 0), last[0]);
        Assert.Equal(ExpectedRowByte(63, RowBytes - 1), last[RowBytes - 1]);

        Assert.True(rows.ContainsRow(63));
        Assert.False(rows.ContainsRow(64));

        ArgumentOutOfRangeException outside = Assert.Throws<ArgumentOutOfRangeException>(() => rows.ResolveRow(64));
        Assert.Contains(PleTensorName, outside.Message);
        Assert.Contains(path, outside.Message); // a row no head range declared is refused naming the tensor and the file it would have been read from.
    }

    [Fact]
    public void RowSource_MapsOnlyTheShardThatOwnsTheTable()
    {
        string directory = Path.Combine(Path.GetTempPath(), "lair-ple-" + Guid.NewGuid().ToString("N"));
        string firstPath = Path.Combine(directory, "ple-fixture-00001-of-00002.gguf");
        string secondPath = Path.Combine(directory, "ple-fixture-00002-of-00002.gguf");

        try
        {
            Directory.CreateDirectory(directory);

            byte[] firstShard = BuildPleGguf(PleMetadata(ngramSize: 3, headsPerNgram: 2), tableRows: 0, tensors: []); // the primary shard carries architecture and PLE metadata but no table bytes.
            byte[] secondShard = BuildPleGguf(EmptySplitHeader(), tableRows: 64);

            File.WriteAllBytes(firstPath, firstShard);
            File.WriteAllBytes(secondPath, secondShard);

            using (GgufFile model = GgufFile.Open(firstPath)) // opening any member assembles the split; the PLE tensor lives in the sibling.
            using (PleRowSource rows = new(model, PleNgramTable.FromGguf(model, Tensor(model))))
            {
                Assert.NotNull(model.Shards);

                Assert.Equal(Path.GetFullPath(secondPath), Path.GetFullPath(rows.ShardPath)); // only the owning shard is mapped: a 28 GB table is never copied and its sibling's bytes are not even addressed.

                ReadOnlySpan<byte> row = rows.ResolveRow(5);
                for (int byteIndex = 0; byteIndex < RowBytes; byteIndex++)
                {
                    Assert.Equal(ExpectedRowByte(5, byteIndex), row[byteIndex]);
                }
            }
        }
        finally
        {
            // the assembled split's two physical shards live in this fixture directory.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            File.Delete(secondPath);
            File.Delete(firstPath);
            Directory.Delete(directory);
        }
    }

    [Fact]
    public void RowSource_RefusesADescriptorReadFromAnotherModelFile()
    {
        string pathA = WriteTemp(BuildPleGguf(SimpleMetadata(), tableRows: 24));
        string pathB = WriteTemp(BuildPleGguf(SimpleMetadata(), tableRows: 24));

        using GgufFile modelA = GgufFile.Open(pathA);
        using GgufFile modelB = GgufFile.Open(pathB);
        PleNgramTable tableA = PleNgramTable.FromGguf(modelA, Tensor(modelA));

        PleTableException error = Assert.Throws<PleTableException>(() => new PleRowSource(modelB, tableA));
        Assert.Contains(pathB, error.Message);
        Assert.Contains(pathA, error.Message);
    }

    [Fact]
    public void FromGguf_RefusesUndeclaredOrInconsistentGeometryNamingModelTensorAndKey()
    {
        string missingNgram = WriteTemp(BuildPleGguf(
            PleMetadata(ngramSize: 3, headsPerNgram: 2) with { NgramSizeDeclared = null },
            tableRows: 64));

        using (GgufFile model = GgufFile.Open(missingNgram))
        {
            PleTableException error = Assert.Throws<PleTableException>(() => PleNgramTable.FromGguf(model, Tensor(model)));
            Assert.Contains(missingNgram, error.Message);
            Assert.Contains(PleTensorName, error.Message);
            Assert.Contains("qwen4exp.ple.ngram_size", error.Message);
        }

        string wrongArch = WriteTemp(BuildPleGguf(
            PleMetadata(ngramSize: 3, headsPerNgram: 2) with { Architecture = "llama" },
            tableRows: 64));

        using (GgufFile model = GgufFile.Open(wrongArch))
        {
            PleTableException error = Assert.Throws<PleTableException>(() => PleNgramTable.FromGguf(model, Tensor(model)));
            Assert.Contains("llama.ple.ngram_size", error.Message); // keys are read under the resolved architecture prefix, and this file declares none there.
        }

        string headPastTable = WriteTemp(BuildPleGguf(
            PleMetadata(ngramSize: 3, headsPerNgram: 2) with { HeadVocabSizes = [16, 16, 16, 32] },
            tableRows: 64));

        using (GgufFile model = GgufFile.Open(headPastTable))
        {
            PleTableException error = Assert.Throws<PleTableException>(() => PleNgramTable.FromGguf(model, Tensor(model)));
            Assert.Contains("head 3", error.Message);
            Assert.Contains("only 64 rows", error.Message);
        }

        string zeroMultiplier = WriteTemp(BuildPleGguf(
            PleMetadata(ngramSize: 3, headsPerNgram: 2) with { Multipliers = [0x9E3779B97F4A7C15UL, 0UL, 0xFF51AFD7ED558CCDUL] },
            tableRows: 64));

        using (GgufFile model = GgufFile.Open(zeroMultiplier))
        {
            PleTableException error = Assert.Throws<PleTableException>(() => PleNgramTable.FromGguf(model, Tensor(model)));
            Assert.Contains("'qwen4exp.ple.layer_multipliers'[1] is 0", error.Message);
        }

        string widthMismatch = WriteTemp(BuildPleGguf(
            PleMetadata(ngramSize: 3, headsPerNgram: 2) with { EmbeddingLength = 128 },
            tableRows: 64));

        using (GgufFile model = GgufFile.Open(widthMismatch))
        {
            PleTableException error = Assert.Throws<PleTableException>(() => PleNgramTable.FromGguf(model, Tensor(model)));
            Assert.Contains("qwen4exp.embedding_length_per_layer_input", error.Message);
            Assert.Contains("160 elements wide", error.Message); // the gather indexes whole packed rows of this tensor, so the declared width must match its row width.
        }
    }

    [Fact]
    public void ResidencyGate_NarrowsGatheredRowsToTheBindingCeilingOfThePleChain()
    {
        ResolvedBudgetChain chain = ModelBudgets.Resolve(new BudgetRequest
        {
            ModelName = "qwen-test",
            Declared =
            [
                new BudgetChain
                {
                    Resource = BudgetResource.Ple,
                    Order = [BudgetTier.Dram, BudgetTier.Vram, BudgetTier.Disk],
                    Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Vram] = 1, [BudgetTier.Dram] = 4 },
                },
            ],
            OpenedPoolTiers = [BudgetTier.Vram, BudgetTier.Dram],
        }).For(BudgetResource.Ple)!;

        Assert.Equal(BudgetResource.Ple, chain.Resource);

        // A 1 MiB ceiling at 90 packed bytes per row admits 11650 rows; the 4 MiB position admits more, so the tighter position binds.
        PleRowResidencyClamp? clamp = PleResidencyGate.Compute(chain, RowBytes, requestedRows: 100_000);
        Assert.NotNull(clamp);
        Assert.Equal(BudgetTier.Vram, clamp.Tier);
        Assert.Equal(1L, clamp.CapMegabytes);
        Assert.Equal((long)RowBytes, clamp.RowBytes);
        Assert.Equal(11650L, clamp.AllowedRows); // 1 MiB / 90 bytes.
        Assert.True(clamp.AllowsPositiveRows);
        Assert.Contains("ple residency at 1 megabytes", clamp.Describe());

        // A request the ceilings already fit produces no clamp at all: today's shape unchanged.
        Assert.Null(PleResidencyGate.Compute(chain, RowBytes, requestedRows: 1000));

        // A ceiling below one packed row admits nothing, and that posture is reported rather than vanished.
        ResolvedBudgetChain tiny = ModelBudgets.Resolve(new BudgetRequest
        {
            ModelName = "qwen-test",
            Declared =
            [
                new BudgetChain
                {
                    Resource = BudgetResource.Ple,
                    Order = [BudgetTier.Dram],
                    Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 1 },
                },
            ],
            OpenedPoolTiers = [BudgetTier.Dram],
        }).For(BudgetResource.Ple)!;

        Assert.True(PleResidencyGate.FitsRows(0, RowBytes) == 0);
        Assert.Throws<ArgumentException>(() => PleResidencyGate.FitsRows(1, rowBytes: 0)); // whole packed rows are what this seam bounds; a zero-cost row describes nothing.

        ArgumentException wrongChain = Assert.Throws<ArgumentException>(
            () => PleResidencyGate.Compute(ModelBudgets.Resolve(new BudgetRequest
            {
                ModelName = "qwen-test",
                Declared = [new BudgetChain { Resource = BudgetResource.Weights, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = 8 } }],
                OpenedPoolTiers = [BudgetTier.Dram],
            }).For(BudgetResource.Weights)!, RowBytes, 10));
        Assert.Contains("'weights' is not 'ple'", wrongChain.Message); // the PLE ceiling bounds gathered rows, nothing else's residency.
    }

    private static LogicalTensor Tensor(GgufFile model) =>
        model.Tensors.Find(PleTensorName) ?? throw new InvalidOperationException($"fixture declares no '{PleTensorName}' tensor");

    private static byte ExpectedRowByte(int rowIndex, int byteIndex) => (byte)(rowIndex * 7 + byteIndex * 11);

    private sealed record PleMetadataShape
    {
        public required string Architecture { get; init; }

        /// <summary>The prefix the fixture writes its PLE keys under — the same as the architecture unless a fixture wants keys a declared architecture will never find.</summary>
        public required string KeyPrefix { get; init; }

        public required int? NgramSizeDeclared { get; init; }
        public required int HeadsPerNgram { get; init; }
        public required int EmbeddingLength { get; init; }
        public required int EosTokenId { get; init; }
        public required ulong[] Multipliers { get; init; }
        public required ulong[] HeadOffsets { get; init; }
        public required ulong[] HeadVocabSizes { get; init; }
    }

    private static PleMetadataShape PleMetadata(int ngramSize, int headsPerNgram) => new()
    {
        Architecture = "qwen4exp",
        KeyPrefix = "qwen4exp",
        NgramSizeDeclared = ngramSize,
        HeadsPerNgram = headsPerNgram,
        EmbeddingLength = RowElements,
        EosTokenId = 3,
        Multipliers = [0x9E3779B97F4A7C15UL, 0xBF58476D1CE4E5B9UL, 0xFF51AFD7ED558CCDUL],
        HeadOffsets = [0, 16, 32, 48],
        HeadVocabSizes = [16, 16, 16, 16],
    };

    private static PleMetadataShape SimpleMetadata() =>
        PleMetadata(ngramSize: 2, headsPerNgram: 1) with { HeadOffsets = [16], HeadVocabSizes = [8], Multipliers = [1UL, 1UL] };

    private static PleMetadataShape EmptySplitHeader() =>
        PleMetadata(ngramSize: 3, headsPerNgram: 2);

    private string WriteTemp(byte[] image)
    {
        string path = Path.Combine(Path.GetTempPath(), "lair-ple-" + Guid.NewGuid().ToString("N") + ".gguf");
        File.WriteAllBytes(path, image);
        _tempPaths.Add(path);
        return path;
    }

    private static List<(string Key, object Value)> MetadataPairs(PleMetadataShape shape)
    {
        List<(string Key, object Value)> pairs =
        [
            ("general.architecture", shape.Architecture),
            (shape.KeyPrefix + ".embedding_length_per_layer_input", shape.EmbeddingLength),
            (shape.KeyPrefix + ".ple.heads_per_ngram", shape.HeadsPerNgram),
            (shape.KeyPrefix + ".ple.eos_token_id", shape.EosTokenId),
            (shape.KeyPrefix + ".ple.layer_multipliers", shape.Multipliers),
            (shape.KeyPrefix + ".ple.head_offsets", shape.HeadOffsets),
            (shape.KeyPrefix + ".ple.head_vocab_sizes", shape.HeadVocabSizes),
        ];

        if (shape.NgramSizeDeclared is int ngramSize)
        {
            pairs.Add((shape.KeyPrefix + ".ple.ngram_size", ngramSize));
        }

        return pairs;
    }

    private static byte[] BuildPleGguf(PleMetadataShape shape, int tableRows, IReadOnlyList<TensorDef>? tensors = null)
    {
        List<TensorDef> defs = tensors is not null
            ? [.. tensors]
            : [new TensorDef(PleTensorName, [tableRows, RowElements], GgmlIQ4Nl)];

        return BuildGguf(MetadataPairs(shape), defs);
    }

    private sealed record TensorDef(string Name, int[] Dims, uint GgmlType);

    /// <summary>Builds a GGUF v3 image from KV pairs (strings, ints, ulong arrays, int arrays) and tensors, writing each tensor's payload as a deterministic per-row byte pattern.</summary>
    private static byte[] BuildGguf(IReadOnlyList<(string Key, object Value)> kvPairs, IReadOnlyList<TensorDef> tensors)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

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

                case ulong[] ulongs:
                    writer.Write(9u); // ARRAY
                    writer.Write(10u); // UINT64 elements, the width llama.cpp declares these arrays at
                    writer.Write((ulong)ulongs.Length);
                    foreach (ulong entry in ulongs)
                    {
                        writer.Write(entry);
                    }

                    break;

                case int[] ints:
                    writer.Write(9u); // ARRAY
                    writer.Write(5u); // INT32 elements
                    writer.Write((ulong)ints.Length);
                    foreach (int entry in ints)
                    {
                        writer.Write(entry);
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
            offset += PayloadBytes(tensor);
        }

        long headerLength = stream.Position;
        long dataStart = (headerLength + 31) / 32 * 32; // GGUF default alignment is 32
        for (long i = headerLength; i < dataStart; i++)
        {
            writer.Write((byte)0);
        }

        foreach (TensorDef tensor in tensors)
        {
            long payload = PayloadBytes(tensor);
            long rows = tensor.Dims.Length > 0 ? tensor.Dims[0] : 1;
            long rowBytes = rows > 0 ? payload / Math.Max(1, rows) : payload;

            for (long b = 0; b < payload; b++)
            {
                writer.Write(ExpectedRowByte((int)(b / rowBytes), (int)(b % rowBytes))); // every row carries its own recognizable pattern, so a resolved span can be checked against the row it claims to be.
            }
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static long PayloadBytes(TensorDef tensor)
    {
        long elements = 1;
        foreach (int dim in tensor.Dims)
        {
            elements *= dim;
        }

        return tensor.GgmlType switch
        {
            GgmlIQ4Nl => elements / 32 * 18, // IQ4_NL: 18 bytes per 32-element block.
            GgmlF32 => elements * 4,
            _ => throw new ArgumentException($"Unsupported fixture ggml type {tensor.GgmlType}."),
        };
    }

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }
}
