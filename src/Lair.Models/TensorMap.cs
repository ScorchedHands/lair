using System.Globalization;

using Lair.Contracts;

namespace Lair.Models;

/// <summary>
/// Identifies one logical tensor role the forward engine consumes, independent of the GGUF names a profile
/// maps it to.
/// </summary>
public enum TensorRole
{
    TokenEmbedding,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    FeedForwardGate,
    FeedForwardUp,
    FeedForwardDown,
    AttentionNorm,
    FeedForwardNorm,
    AttentionQueryBias,
    AttentionKeyBias,
    AttentionValueBias,
    AttentionOutputBias,
    OutputNorm,
    OutputHead,
    OutputBias,
    AttentionQueryNorm,
    AttentionKeyNorm,
    AttentionNormBias,
    FeedForwardNormBias,
    OutputNormBias,
    PostAttentionNorm,
    PostFeedForwardNorm,
    MoeRouter,
    MoeExpertGate,
    MoeExpertUp,
    MoeExpertDown,
    MoeSharedExpertGate,
    MoeSharedExpertUp,
    MoeSharedExpertDown,
}

/// <summary>Whether a tensor name is a single global name or a per-block name carrying a <c>{layer}</c> placeholder.</summary>
public enum TensorScope
{
    Global,
    PerLayer,
}

/// <summary>
/// One slot of a profile's <see cref="TensorMap"/>: the role it fills, the GGUF name it maps to (a
/// <c>{layer}</c> template for <see cref="TensorScope.PerLayer"/> roles), whether the tensor is required,
/// and alias names tried after the primary name. Aliases are declared against the same scope rules as the
/// primary name; the llama and qwen2 profiles declare none, so their map resolves to exactly the canonical
/// GGUF names.
/// </summary>
public sealed record TensorDescriptor(
    TensorRole Role,
    TensorScope Scope,
    string Name,
    bool Required,
    IReadOnlyList<string> Aliases)
{
    /// <summary>
    /// The concrete GGUF tensor name for <paramref name="layer"/>: the primary name with the placeholder
    /// substituted for per-layer roles, and the name itself for global roles (the layer is ignored there).
    /// </summary>
    public string NameFor(int layer) =>
        Scope == TensorScope.PerLayer ? Name.Replace("{layer}", layer.ToString(CultureInfo.InvariantCulture)) : Name;

    /// <summary>
    /// Finds this descriptor's tensor in <paramref name="tensors"/>: the primary name first, then each alias
    /// in order. Returns <c>null</c> when no candidate is present.
    /// </summary>
    public LogicalTensor? Resolve(LogicalTensorTable tensors, int layer = -1)
    {
        LogicalTensor? hit = tensors.Find(NameFor(layer));
        if (hit is not null)
        {
            return hit;
        }

        foreach (string alias in Aliases)
        {
            hit = tensors.Find(Scope == TensorScope.PerLayer ? alias.Replace("{layer}", layer.ToString(CultureInfo.InvariantCulture)) : alias);
            if (hit is not null)
            {
                return hit;
            }
        }

        return null;
    }
}

/// <summary>
/// A profile's complete tensor map: exactly one <see cref="TensorDescriptor"/> per <see cref="TensorRole"/>
/// except the sparse-FFN roles, which a dense profile leaves out of its map entirely, resolved once at load
/// and carried immutably by the <see cref="ModelPlan"/>. The forward engine looks tensors up through this map
/// instead of through hardcoded name suffixes.
/// </summary>
public sealed class TensorMap
{
    private readonly TensorDescriptor[] _byRole;

    public TensorMap(IReadOnlyList<TensorDescriptor> descriptors)
    {
        ArgumentNullException.ThrowIfNull(descriptors);

        _byRole = new TensorDescriptor[Enum.GetValues<TensorRole>().Length];
        foreach (TensorDescriptor descriptor in descriptors)
        {
            if (_byRole[(int)descriptor.Role] is not null)
            {
                throw new ModelException($"tensor map declares role '{descriptor.Role}' more than once");
            }

            _byRole[(int)descriptor.Role] = descriptor;
        }

        foreach (TensorRole role in Enum.GetValues<TensorRole>())
        {
            if (_byRole[(int)role] is null && !IsSparseRole(role))
            {
                throw new ModelException($"tensor map is missing a descriptor for role '{role}'");
            }
        }

        // Sparse roles stay absent from a dense profile's map: the map names only the tensors its forward can
        // consume, so a dense map carries no router or expert descriptors and a tensor under a sparse name is
        // material its map cannot match.
        List<TensorDescriptor> declared = [];
        foreach (TensorDescriptor descriptor in _byRole)
        {
            if (descriptor is not null)
            {
                declared.Add(descriptor);
            }
        }

        Descriptors = declared;
    }

    /// <summary>Every descriptor, in role order.</summary>
    public IReadOnlyList<TensorDescriptor> Descriptors { get; }

    /// <summary>The descriptor for one role.</summary>
    public TensorDescriptor this[TensorRole role] => _byRole[(int)role];

    /// <summary>
    /// Whether a role belongs to the sparse-FFN block: the router and the expert matrices, dense or shared.
    /// A dense profile names these roles for the load-time report only — its forward consumes none of them,
    /// and the generic llama-shaped fallback treats a tensor under a sparse name as material its map cannot
    /// match rather than as a matchable optional role.
    /// </summary>
    public static bool IsSparseRole(TensorRole role) =>
        role is TensorRole.MoeRouter
            or TensorRole.MoeExpertGate
            or TensorRole.MoeExpertUp
            or TensorRole.MoeExpertDown
            or TensorRole.MoeSharedExpertGate
            or TensorRole.MoeSharedExpertUp
            or TensorRole.MoeSharedExpertDown;
}
