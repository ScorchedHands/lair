namespace Lair.Models;

/// <summary>
/// Load-time model profile for one <c>general.architecture</c> value: the metadata prefix its keys use, the
/// strategy values the forward engine switches on, the complete tensor map naming every logical tensor role,
/// and the sparse-FFN slot (unset while no profile declares experts). Resolved once at load into a
/// <see cref="ModelPlan"/> so the hot forward path stays concrete and never dispatches per token.
/// </summary>
public sealed record ModelProfile
{
    public required string Architecture { get; init; }

    /// <summary>Prefix every architecture-scoped metadata key of this profile carries, e.g. <c>"llama."</c>.</summary>
    public required string Prefix { get; init; }

    public required NormStrategy Norm { get; init; }

    public required ActivationStrategy Activation { get; init; }

    public required BiasPolicy Bias { get; init; }

    public required RopeStrategy Rope { get; init; }

    public required TensorMap Tensors { get; init; }

    /// <summary>Declared sparse-FFN shape, or <c>null</c> for a dense gate/up/down block.</summary>
    public SparseFfnDescriptor? SparseFfn { get; init; }

    /// <summary>
    /// Feature default for per-head QK-norm, applied when the file declares no
    /// <c>&lt;prefix&gt;attention.use_qk_norm</c>: the declaration wins in either direction, and a default of
    /// <c>true</c> makes the per-head norm vectors required at load for a file that declares nothing.
    /// </summary>
    public bool DefaultQkNorm { get; init; }
}
