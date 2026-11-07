using Lair.Loader;

namespace Lair.Models;

/// <summary>
/// Immutable resolved model plan, produced once per load from a <see cref="ModelProfile"/> plus the GGUF
/// container's KV metadata and one fact from its tensor table (whether an untied output matrix is present).
/// It carries the resolved profile name and metadata prefix, the concrete geometry, the profile's tensor map,
/// the strategy values the forward engine switches on, and the sparse-FFN slot. Metadata keys follow the
/// canonical GGUF naming (<c>&lt;prefix&gt;embedding_length</c>, <c>&lt;prefix&gt;block_count</c>,
/// <c>&lt;prefix&gt;attention.head_count</c>, <c>&lt;prefix&gt;attention.head_count_kv</c>,
/// <c>&lt;prefix&gt;feed_forward_length</c>, <c>&lt;prefix&gt;rope.freq_base</c>,
/// <c>&lt;prefix&gt;attention.layer_norm_rms_epsilon</c>, <c>&lt;prefix&gt;vocab_size</c>), with
/// <c>&lt;prefix&gt;attention.key_length</c> as the only accepted source for an explicit head dimension.
/// </summary>
public sealed record ModelPlan
{
    public required string Architecture { get; init; }

    /// <summary>Prefix the resolved profile's metadata keys use, e.g. <c>"llama."</c>.</summary>
    public required string Prefix { get; init; }

    public required int LayerCount { get; init; }

    public required int HiddenSize { get; init; }

    public required int HeadCount { get; init; }

    public required int KvHeadCount { get; init; }

    public required int HeadDim { get; init; }

    public required int IntermediateSize { get; init; }

    public required float RopeTheta { get; init; }

    public required float EpsRmsNorm { get; init; }

    public required int VocabSize { get; init; }

    /// <summary>Whether the tensor table carries an untied output matrix, i.e. the output head is not tied to the token embedding.</summary>
    public required bool HasOutputWeights { get; init; }

    /// <summary>The resolved profile's tensor map: the GGUF names behind every logical tensor role.</summary>
    public required TensorMap Tensors { get; init; }

    public required NormStrategy Norm { get; init; }

    public required ActivationStrategy Activation { get; init; }

    public required BiasPolicy Bias { get; init; }

    public required RopeStrategy Rope { get; init; }

    /// <summary>Declared sparse-FFN shape, or <c>null</c> for the dense gate/up/down block every supported profile declares today.</summary>
    public SparseFfnDescriptor? SparseFfn { get; init; }

    /// <summary>The flat scalar view of this plan's resolved geometry.</summary>
    public ModelDesc ToDesc() =>
        new(
            LayerCount,
            HiddenSize,
            HeadCount,
            KvHeadCount,
            HeadDim,
            IntermediateSize,
            RopeTheta,
            EpsRmsNorm,
            VocabSize,
            HasOutputWeights,
            Architecture);
}
