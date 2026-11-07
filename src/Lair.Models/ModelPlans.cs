using Lair.Loader;

namespace Lair.Models;

/// <summary>
/// Resolves a GGUF file into a <see cref="ModelPlan"/>: the architecture is looked up in
/// <see cref="ModelProfiles"/>, the geometry is read from the profile's metadata prefix, and the tensor map
/// comes from the profile. Only standard RoPE over the full head dimension is accepted: any rope-scaling
/// variant or partial-rotary declaration fails at load.
/// </summary>
public static class ModelPlans
{
    /// <summary>
    /// Parses the architecture metadata of a GGUF file into its resolved plan. The architecture must name a
    /// registered profile; the declared dimensions must split the hidden size exactly across the heads.
    /// </summary>
    public static ModelPlan Resolve(GgufFile gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);

        GgufMetadata metadata = gguf.Metadata;

        if (!metadata.Has("general.architecture"))
        {
            throw new ModelException(
                "GGUF metadata key 'general.architecture' is not present (supported architectures: " +
                $"{string.Join(", ", ModelProfiles.SupportedArchitectures)})");
        }

        string architecture = metadata.Get("general.architecture").AsString();
        ModelProfile profile = ModelProfiles.Resolve(architecture);

        string prefix = profile.Prefix;

        foreach (string key in metadata.Keys)
        {
            if (key.StartsWith(prefix + "rope.scaling.", StringComparison.Ordinal))
            {
                throw new ModelException(
                    $"model architecture '{architecture}' declares rope scaling via GGUF key '{key}'; " +
                    "only standard RoPE without scaling variants is supported");
            }
        }

        int hidden = RequireInt(metadata, prefix + "embedding_length");
        int layers = RequireInt(metadata, prefix + "block_count");
        int heads = RequireInt(metadata, prefix + "attention.head_count");
        int kvHeads = RequireInt(metadata, prefix + "attention.head_count_kv");
        int ffn = RequireInt(metadata, prefix + "feed_forward_length");

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hidden);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(heads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ffn);

        // GGUF stores the per-head dimension explicitly only when it deviates from hidden/heads (key_length /
        // value_length). When present they must agree with each other and split the hidden size exactly.
        int headDim;
        bool hasKeyLength = metadata.Has(prefix + "attention.key_length");
        if (hasKeyLength || metadata.Has(prefix + "attention.value_length"))
        {
            int keyLength = RequireInt(metadata, prefix + "attention.key_length");
            int valueLength = RequireInt(metadata, prefix + "attention.value_length");
            if (keyLength != valueLength)
            {
                throw new ModelException(
                    $"{prefix}attention.key_length ({keyLength}) and {prefix}attention.value_length ({valueLength}) disagree; " +
                    "head dimensions must be equal for this architecture family");
            }

            headDim = keyLength;
        }
        else if (hidden % heads != 0)
        {
            throw new ModelException(
                $"{prefix}attention.head_count ({heads}) does not divide {prefix}embedding_length ({hidden}); " +
                $"the architecture needs an explicit {prefix}attention.key_length to define the head dimension");
        }
        else
        {
            headDim = hidden / heads;
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        if (heads * headDim != hidden)
        {
            throw new ModelException(
                $"{prefix}attention.head_count ({heads}) x head dimension ({headDim}) must equal " +
                $"{prefix}embedding_length ({hidden})");
        }

        // Standard full-head RoPE only: a declared rotary slice smaller than the head is not supported.
        if (metadata.Has(prefix + "rope.dimension_count"))
        {
            int rot = RequireInt(metadata, prefix + "rope.dimension_count");
            if (rot != headDim)
            {
                throw new ModelException(
                    $"{prefix}rope.dimension_count ({rot}) differs from the head dimension ({headDim}); " +
                    "partial rotary embeddings are not supported");
            }
        }

        float ropeTheta = metadata.Has(prefix + "rope.freq_base")
            ? (float)metadata.Get(prefix + "rope.freq_base").AsDouble()
            : 10_000f;

        if (!metadata.Has(prefix + "attention.layer_norm_rms_epsilon"))
        {
            throw new ModelException($"GGUF metadata key '{prefix}attention.layer_norm_rms_epsilon' is not present");
        }

        float eps = (float)metadata.Get(prefix + "attention.layer_norm_rms_epsilon").AsDouble();

        int vocab = RequireInt(metadata, prefix + "vocab_size");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocab);

        bool hasOutputWeights = profile.Tensors[TensorRole.OutputHead].Resolve(gguf.Tensors) is not null;

        return new ModelPlan
        {
            Architecture = architecture,
            Prefix = prefix,
            LayerCount = layers,
            HiddenSize = hidden,
            HeadCount = heads,
            KvHeadCount = kvHeads,
            HeadDim = headDim,
            IntermediateSize = ffn,
            RopeTheta = ropeTheta,
            EpsRmsNorm = eps,
            VocabSize = vocab,
            HasOutputWeights = hasOutputWeights,
            Tensors = profile.Tensors,
            Norm = profile.Norm,
            Activation = profile.Activation,
            Bias = profile.Bias,
            Rope = profile.Rope,
            SparseFfn = profile.SparseFfn,
        };
    }

    private static int RequireInt(GgufMetadata metadata, string key)
    {
        if (!metadata.Has(key))
        {
            throw new ModelException($"GGUF metadata key '{key}' is not present");
        }

        try
        {
            return checked((int)metadata.Get(key).AsLong());
        }
        catch (GgufException ex)
        {
            throw new ModelException($"GGUF metadata key '{key}' must be an integer", ex);
        }
    }
}
