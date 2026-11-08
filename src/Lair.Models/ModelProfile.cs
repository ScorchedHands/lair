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
    /// <c>&lt;prefix&gt;attention.use_qk_norm</c>: a declared false overrides it to <see cref="QkNormMode.Off"/>
    /// whatever the profile defaults to, and a profile defaulting to a weighted mode makes the per-head norm vectors
    /// required at load for a file that declares nothing. A profile defaults this to
    /// <see cref="Lair.Models.QkNormMode.Off"/>.
    /// </summary>
    public QkNormMode DefaultQkNorm { get; init; } = QkNormMode.Off;

    /// <summary>
    /// Where this architecture's forward places its norm vectors relative to the residual add: a profile defaults to
    /// <see cref="Lair.Models.NormPlacement.PreOnly"/>, and
    /// <see cref="Lair.Models.NormPlacement.PostSublayerOutputs"/> makes every layer's post-sublayer norm tensors
    /// required at load. No metadata key selects it: the profile decides, so a file carrying those tensors under a
    /// pre-only placement is a layout failure rather than ignored material.
    /// </summary>
    public NormPlacement NormPlacement { get; init; } = NormPlacement.PreOnly;
}
