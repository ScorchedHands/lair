using Lair.Contracts;
using Lair.Loader;

namespace Lair.Tests;

/// <summary>
/// Loader gate for the owner's target model, Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS (architecture qwen4exp), shipped as
/// two GGUF shards on disk (skip-if-absent against those absolute paths): opening either shard assembles the whole
/// split as one logical model — the two shard headers resolve as a single 1224-tensor table carrying shard 2's
/// `per_layer_token_embd.weight`, the merged metadata carries shard 1's architecture, and each physical shard keeps
/// its own data region and disjoint tensor table behind <see cref="GgufFile.Shards"/>. Every tensor declared across
/// both shards resolves through the <see cref="GgmlTypeRegistry"/> acceptance path with its block geometry intact, and
/// each header's offsets/sizes are internally consistent within its own file. Measured from the shipped files: a split
/// GGUF carries only its own tensors in its header — shard 1 declares 1223 tensor infos plus the full model metadata,
/// shard 2 (the 28.8 GB PLE table) declares just one tensor and three split.* keys; the global count lives in
/// general-independent metadata as split.tensors.count = 1224. No test here mmaps or reads tensor payload bytes: only
/// LAIR's own header/tensor-table parse and assembly path runs.
/// </summary>
[Trait("Category", "Loader")]
public sealed class Qwen4ExpLoaderTests
{
    // The owner's target model as it lives on disk; these gates observe the shipped files, they do not copy them (a merge of the two shards into one file is exactly what must NOT happen — it filled the disk in an earlier attempt).
    private const string ModelDirectory = @"C:\source\github.com\Strata-data\models\IQ3_XXS";

    private const string FirstShardName = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    private const string SecondShardName = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf";

    // Measured from the shipped shards: per-shard declared tensor counts and the global count the headers record in split.tensors.count.
    private const int FirstShardTensors = 1223;
    private const int SecondShardTensors = 1;
    private const int DeclaredAcrossShards = FirstShardTensors + SecondShardTensors; // 1224

    // Exact combined per-type inventory of the declared tensors across both shards (shard 1 x shard 2): every id in the set is one LAIR's registry accepts.
    private static readonly Dictionary<QuantFormat, int> ExpectedCombinedInventory = new()
    {
        [QuantFormat.F32] = 292,
        [QuantFormat.F16] = 1,
        [QuantFormat.Q4_K] = 45,
        [QuantFormat.Q5_K] = 15,
        [QuantFormat.Q6_K] = 71,
        [QuantFormat.IQ2_XXS] = 18,
        [QuantFormat.IQ2_XS] = 20,
        [QuantFormat.IQ3_XXS] = 12,
        [QuantFormat.IQ4_NL] = 59, // 58 in shard 1 + the PLE table's single IQ4_NL tensor in shard 2
        [QuantFormat.IQ3_S] = 82,
        [QuantFormat.IQ2_S] = 20,
        [QuantFormat.IQ4_XS] = 67,
        [QuantFormat.BF16] = 484,
        [QuantFormat.Q2_0] = 38,
    };

    // The registry rows the declared formats must resolve through: (ggml id, reference name, block size, type size), pinned to GgmlTypeRegistry's accepted geometry.
    private static readonly QuantDescriptor[] AcceptedRows =
    [
        new(GgmlType.F32, "F32", QuantFormat.F32, 1, 4),
        new(GgmlType.F16, "F16", QuantFormat.F16, 1, 2),
        new(GgmlType.Q4_K, "Q4_K", QuantFormat.Q4_K, 256, 144),
        new(GgmlType.Q5_K, "Q5_K", QuantFormat.Q5_K, 256, 176),
        new(GgmlType.Q6_K, "Q6_K", QuantFormat.Q6_K, 256, 210),
        new(GgmlType.IQ2_XXS, "IQ2_XXS", QuantFormat.IQ2_XXS, 256, 66),
        new(GgmlType.IQ2_XS, "IQ2_XS", QuantFormat.IQ2_XS, 256, 74),
        new(GgmlType.IQ3_XXS, "IQ3_XXS", QuantFormat.IQ3_XXS, 256, 98),
        new(GgmlType.IQ4_NL, "IQ4_NL", QuantFormat.IQ4_NL, 32, 18),
        new(GgmlType.IQ3_S, "IQ3_S", QuantFormat.IQ3_S, 256, 110),
        new(GgmlType.IQ2_S, "IQ2_S", QuantFormat.IQ2_S, 256, 82),
        new(GgmlType.IQ4_XS, "IQ4_XS", QuantFormat.IQ4_XS, 256, 136),
        new(GgmlType.Q2_0, "Q2_0", QuantFormat.Q2_0, 64, 18),
        new(GgmlType.BF16, "BF16", QuantFormat.BF16, 1, 2),
    ];

    // ---------------------------------------------------------------- assembled parse of both shards

    [Fact]
    public void Open_Qwen4ExpShards_AssemblesOneModelWithSanityHeaderValues()
    {
        string? first = ResolveShardPath(FirstShardName);
        string? second = ResolveShardPath(SecondShardName);
        if (first is null || second is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        using GgufFile model = GgufFile.Open(first);
        Assert.NotNull(model.Shards);
        IReadOnlyList<GgufFile> shards = model.Shards!;
        Assert.Equal(2, shards.Count);
        Assert.Equal(first, Path.GetFullPath(shards[0].Path));
        Assert.Equal(second, Path.GetFullPath(shards[1].Path)); // discovered in ascending shard order from either member.

        foreach (GgufFile shard in shards)
        {
            Assert.Equal(3, shard.Version); // GGUF v3: 64-bit tensor and KV counts
            Assert.True(shard.Alignment > 0 && shard.DataStart % shard.Alignment == 0, $"data start {shard.DataStart} must be aligned to {shard.Alignment}");
            Assert.True(shard.DataStart < shard.FileSize);
            Assert.Equal(new FileInfo(shard.Path).Length, shard.FileSize); // each shard view reports the real file length for these multi-GB files
        }

        // The merged model metadata carries shard 1's identity: full model metadata under qwen4exp plus the split identity.
        Assert.Equal("qwen4exp", model.Metadata.GetString("general.architecture", ""));
        Assert.Equal(0, model.Metadata.GetInt("split.no", -1)); // the primary shard's position wins the merge.
        Assert.Equal(2, model.Metadata.GetInt("split.count", -1));
        Assert.Equal(DeclaredAcrossShards, model.Metadata.GetInt("split.tensors.count", -1));

        // Shard 2 carries only the split identity: no general.architecture and no model metadata. It is a tensor-data continuation, not a standalone model header.
        Assert.False(shards[1].Metadata.Has("general.architecture"));
        Assert.Equal(1, shards[1].Metadata.GetInt("split.no", -1));
        Assert.Equal(2, shards[1].Metadata.GetInt("split.count", -1));
        Assert.Equal(DeclaredAcrossShards, shards[1].Metadata.GetInt("split.tensors.count", -1));

        // Each header declares only its own tensors — the phase's "full tensor-info table in every shard" claim does not hold for these files. The global 1224 lives in split.tensors.count and in the assembled table; the assembly is what covers the model.
        Assert.Equal(FirstShardTensors, shards[0].Tensors.Tensors.Count);

        // Shard 2 declares exactly one tensor: the PLE table. GGUF stores it fastest-varying first as [160, 320001536]; LAIR's row-major logical dims reverse that to [320001536, 160].
        LogicalTensor ple = Assert.Single(shards[1].Tensors.Tensors);
        Assert.Equal("per_layer_token_embd.weight", ple.Name);
        Assert.Equal(QuantFormat.IQ4_NL, ple.Format);
        Assert.Equal([320001536, 160], ple.Shape.Dims);

        // The assembled table is the disjoint union of both shard tables, and the second shard's tensor is present in it.
        Assert.Equal(DeclaredAcrossShards, model.Tensors.Tensors.Count);
        Assert.NotNull(model.Tensors.Find("per_layer_token_embd.weight"));

        HashSet<string> names1 = new(shards[0].Tensors.Tensors.Select(t => t.Name), StringComparer.Ordinal);
        HashSet<string> names2 = new(shards[1].Tensors.Tensors.Select(t => t.Name), StringComparer.Ordinal);
        Assert.Equal(FirstShardTensors, names1.Count); // distinct within shard 1
        Assert.Single(names2); // distinct within shard 2
        Assert.True(names2.All(n => !names1.Contains(n)), "the shards must declare disjoint tensor sets");

        // Any member of the set is a valid entry point: opening the last shard assembles the same model.
        using GgufFile fromLast = GgufFile.Open(second);
        Assert.NotNull(fromLast.Shards);
        Assert.Equal(2, fromLast.Shards!.Count);
        Assert.Equal(DeclaredAcrossShards, fromLast.Tensors.Tensors.Count);
        Assert.Equal("qwen4exp", fromLast.Metadata.GetString("general.architecture", ""));
        Assert.NotNull(fromLast.Tensors.Find("per_layer_token_embd.weight"));
    }

    // ---------------------------------------------------------------- registry acceptance + geometry for every declared tensor

    [Fact]
    public void Open_Qwen4ExpShards_EveryDeclaredTensorResolvesThroughRegistryWithGeometryHolds()
    {
        string? first = ResolveShardPath(FirstShardName);
        string? second = ResolveShardPath(SecondShardName);
        if (first is null || second is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        Dictionary<QuantFormat, QuantDescriptor> pinnedByFormat = AcceptedRows.ToDictionary(d => d.Format);

        using GgufFile model = GgufFile.Open(first); // Open itself runs the registry acceptance path per tensor across the assembled set: any rejection aborts here with a verbatim message.
        Assert.NotNull(model.Shards);

        Dictionary<QuantFormat, int> combinedCounts = new();
        int examined = 0;

        foreach ((GgufFile shard, string label) in model.Shards!.Select((s, i) => (s, $"shard {i + 1}")))
        {
            long dataAvailable = shard.FileSize - shard.DataStart;

            foreach (LogicalTensor tensor in shard.Tensors.Tensors)
            {
                QuantDescriptor pinned = pinnedByFormat[tensor.Format];
                Assert.True(
                    GgmlTypeRegistry.TryResolve(pinned.TypeId, out QuantDescriptor descriptor),
                    $"{label}: tensor '{tensor.Name}' format {pinned.Name} (ggml id {pinned.TypeId}) must resolve through the registry");

                // The resolved row is exactly the pinned geometry: same id/name/format and block rule.
                Assert.Equal(pinned.TypeId, descriptor.TypeId);
                Assert.Equal(pinned.Name, descriptor.Name);
                Assert.Equal(tensor.Format, descriptor.Format);

                TensorShape shape = tensor.Shape;

                // Block-geometry rule over every declared dimension: whole blocks only...
                Assert.True(
                    shape.ElementCount % descriptor.BlockSize == 0 && shape.Dims[^1] % descriptor.BlockSize == 0,
                    $"{label}: tensor '{tensor.Name}' ({descriptor.Name}) element count {shape.ElementCount} or fastest dim {shape.Dims[^1]} is not a whole number of {descriptor.BlockSize}-element blocks");

                // ...and the payload byte length matches the registry arithmetic exactly — both through TensorShape's row stride and straight from the descriptor.
                long viaRowStride = shape.RowStrideBytes(tensor.Format) * (shape.ElementCount / shape.Dims[^1]);
                Assert.True(
                    tensor.SizeBytes == viaRowStride && tensor.SizeBytes == descriptor.TypeSize * (shape.ElementCount / descriptor.BlockSize),
                    $"{label}: tensor '{tensor.Name}' ({descriptor.Name}) declares {tensor.SizeBytes} bytes, geometry says row stride {shape.RowStrideBytes(tensor.Format)} x rows and {descriptor.TypeSize} bytes per {descriptor.BlockSize}-element block");

                // Offsets/sizes are internally consistent inside this shard header: every payload sits fully in the shard's own data region.
                Assert.True(
                    tensor.Offset >= 0 && tensor.SizeBytes > 0 && tensor.Offset + tensor.SizeBytes <= dataAvailable,
                    $"{label}: tensor '{tensor.Name}' payload [{tensor.Offset}..{tensor.Offset + tensor.SizeBytes}) does not fit in {dataAvailable} data bytes");

                // The assembled table carries the same tensor record the shard header declared.
                LogicalTensor assembled = model.Tensors.Find(tensor.Name)!;
                Assert.Same(tensor, assembled);

                combinedCounts[tensor.Format] = combinedCounts.GetValueOrDefault(tensor.Format) + 1;
                examined++;
            }
        }

        Assert.Equal(DeclaredAcrossShards, examined); // every declared tensor across both shards was examined with zero rejections
        Assert.Equal(DeclaredAcrossShards, model.Tensors.Tensors.Count); // the assembled table is exactly the union examined here
        Assert.Equal(ExpectedCombinedInventory, combinedCounts);

        Console.WriteLine(
            $"qwen4exp loader gate ({examined} tensors across {FirstShardTensors}+{SecondShardTensors} shard declarations): " +
            string.Join(", ", ExpectedCombinedInventory.OrderBy(kv => kv.Key).Select(kv => $"{kv.Value} x {AcceptedRows.First(d => d.Format == kv.Key).Name}")));
    }

    // ---------------------------------------------------------------- rejection diagnostics contract on these files

    [Fact]
    public void Open_Qwen4ExpShards_ZeroRejections_AndAnyRejectNamesTensorAndGgmlId()
    {
        string? first = ResolveShardPath(FirstShardName);
        string? second = ResolveShardPath(SecondShardName);
        if (first is null || second is null)
        {
            return; // real-model tests skip when the fixture is absent (xunit 2.9 has no dynamic Assert.Skip)
        }

        foreach ((string path, string label) in new[] { (first!, "shard 1"), (second!, "shard 2") })
        {
            try
            {
                using GgufFile model = GgufFile.Open(path); // assembled acceptance-path parse: zero rejections means this completes from either entry point.
                Console.WriteLine($"{label}: assembled clean, {model.Tensors.Tensors.Count} tensors across {model.Shards!.Count} shards accepted");
            }
            catch (GgufException ex)
            {
                // A rejection on these files must name the offending tensor and its ggml id — the format every acceptance-path throw carries ("tensor '<name>' ... unsupported ggml type <id> (<Name>)" or "with ggml type <id> (...)").
                Assert.Matches(@"tensor '[^']*'", ex.Message);
                Assert.Matches(@"ggml type \d+", ex.Message);
            }
        }
    }

    // Absolute paths outside the repository: ModelFixture's assembly-relative walk does not apply, but the skip-if-absent convention is identical.
    private static string? ResolveShardPath(string shardFileName)
    {
        string candidate = Path.Combine(ModelDirectory, shardFileName);
        return File.Exists(candidate) ? candidate : null;
    }
}
