using Lair.Loader;

namespace Lair.Models;

/// <summary>
/// Flat scalar view of a resolved <see cref="ModelPlan"/>: the geometry the plan derives from the container's
/// KV metadata plus one fact from its tensor table (whether an untied output matrix is present). The plan is
/// the load product; this record is kept as the terse read surface for consumers that only need the scalars.
/// </summary>
public sealed record ModelDesc(
    int LayerCount,
    int HiddenSize,
    int HeadCount,
    int KvHeadCount,
    int HeadDim,
    int IntermediateSize,
    float RopeTheta,
    float EpsRmsNorm,
    int VocabSize,
    bool HasOutputWeights,
    string Architecture,
    float AttentionSoftcap,
    float FinalLogitSoftcap)
{
    /// <summary>Supported architecture names, straight from the profile registry.</summary>
    public static IReadOnlyList<string> SupportedArchitectures => [.. ModelProfiles.SupportedArchitectures];

    /// <summary>Resolves the file's plan and returns its flat scalar view.</summary>
    public static ModelDesc FromGguf(GgufFile gguf) => ModelPlans.Resolve(gguf).ToDesc();
}
