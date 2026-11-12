using Lair.Models;

namespace Lair.Http;

/// <summary>
/// What a served model advertises: the id requests must name, the effective context window, the KV pool size,
/// the dominant tensor quantization, the architecture, the vocabulary size, the tool output parser and reasoning
/// visibility policy resolved for this model at load, and — only for a model whose plan carries an expert-routed
/// block — the sparse-FFN strategy that plan resolved to. Both fields default to the configuration's own defaults so records
/// built without a resolved plan stay well-formed; every engine-opened record carries its resolved values instead.
/// </summary>
public sealed record ModelInfo(
    string Id,
    int ContextLength,
    int KvBlocks,
    string Quant,
    string Architecture,
    int VocabSize,
    string ToolParser = "auto",
    string Reasoning = "keep",
    string? SparseFfnStrategy = null);

/// <summary>
/// The wire name of each resolved sparse-FFN strategy, shared by the load log and the advertised model facts. A dense
/// model carries no value at all: these names exist only for a block that routes experts.
/// </summary>
public static class SparseFfnStrategyNames
{
    public const string DenseResident = "dense_resident";
    public const string LowRankSharedBasis = "low_rank_shared_basis";

    public static string Of(SparseFfnStrategy strategy) => strategy switch
    {
        SparseFfnStrategy.LowRankSharedBasis => LowRankSharedBasis,
        _ => DenseResident,
    };
}
