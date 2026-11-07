using System.Globalization;

using Lair.Contracts;
using Lair.Loader;

namespace Lair.Models;

/// <summary>
/// Resolves a GGUF file into a <see cref="ModelPlan"/>: the architecture is looked up in
/// <see cref="ModelProfiles"/>, the geometry is read from the profile's metadata prefix, and the tensor map
/// comes from the profile. Only standard RoPE over the full head dimension is accepted: any rope-scaling
/// variant or partial-rotary declaration fails at load. An unregistered architecture is accepted only
/// through the generic llama-shaped fallback profile, which requires the file's tensor table to match the
/// llama map exactly and reports the substitution as one structured log line when applied.
/// </summary>
public static class ModelPlans
{
    /// <summary>
    /// Parses the architecture metadata of a GGUF file into its resolved plan. The architecture must name a
    /// registered profile or pass the generic llama-shaped fallback check; an empty or whitespace-only name is
    /// rejected outright and is never a fallback candidate. The declared dimensions must split the hidden size
    /// exactly across the heads.
    /// </summary>
    /// <param name="gguf">The GGUF to resolve.</param>
    /// <param name="logLine">Sink for the structured fallback-applied line; hosts may capture instead.</param>
    public static ModelPlan Resolve(GgufFile gguf, Action<string>? logLine = null)
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

        // An empty or whitespace-only architecture name is a load failure in its own right. It must not reach
        // the generic fallback, where it would re-prefix the llama-shaped profile to a degenerate '.' and read
        // the file's geometry under that prefix.
        if (string.IsNullOrWhiteSpace(architecture))
        {
            throw new ModelException("GGUF metadata key 'general.architecture' is present but empty");
        }

        ModelProfile profile = ModelProfiles.ResolveOrFallback(architecture);
        bool fallback = profile.Architecture != architecture;

        // The fallback is only ever a disguise for a llama-shaped file, so its tensor-table match is verified
        // before any metadata read: a table the map cannot match is rejected with the registry wording, which
        // also carries the supported-profile list.
        if (fallback)
        {
            ValidateLlamaShapedTable(profile.Tensors, architecture, gguf.Tensors);
        }

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

        LogicalTensorTable tensors = gguf.Tensors;
        bool hasOutputWeights = profile.Tensors[TensorRole.OutputHead].Resolve(tensors) is not null;

        if (fallback)
        {
            // The fallback was accepted: one structured line naming the model, the declared architecture,
            // and the profile actually applied, in the same style as the tokenizer profile defaults.
            if (logLine is not null)
            {
                string modelName = metadata.Has("general.name") ? metadata.Get("general.name").AsString() : "";
                logLine(
                    $"model_profile model='{(modelName.Length > 0 ? modelName : gguf.Path)}' " +
                    $"architecture='{architecture}' profile={profile.Architecture}");
            }
        }

        return new ModelPlan
        {
            Architecture = architecture,
            Profile = profile.Architecture,
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

    // The generic fallback accepts a tensor table only when it matches the llama-shaped map exactly: every
    // required descriptor resolves, and every tensor in the table is named by some descriptor's primary or
    // alias name (the optional ones included). A tensor no descriptor can name is unexpected; the rejection
    // carries the registry wording with its supported-profile list, names both sides of the mismatch, and
    // states that the llama-shaped fallback was attempted. The block extent is derived from the table itself,
    // so this gate needs no metadata: it runs before any metadata read.
    private static void ValidateLlamaShapedTable(TensorMap map, string architecture, LogicalTensorTable tensors)
    {
        int layers = 1;
        while (map[TensorRole.AttentionQuery].Resolve(tensors, layers) is not null)
        {
            layers++;
        }

        HashSet<string> nameable = new(StringComparer.Ordinal);
        List<string> missing = [];

        foreach (TensorDescriptor descriptor in map.Descriptors)
        {
            bool perLayer = descriptor.Scope == TensorScope.PerLayer;
            int slots = perLayer ? layers : 1;

            for (int slot = 0; slot < slots; slot++)
            {
                int layer = perLayer ? slot : -1;
                nameable.Add(descriptor.NameFor(layer));
                foreach (string alias in descriptor.Aliases)
                {
                    nameable.Add(perLayer ? AliasName(alias, layer) : alias);
                }

                if (descriptor.Required && descriptor.Resolve(tensors, layer) is null)
                {
                    missing.Add(descriptor.NameFor(layer));
                }
            }
        }

        List<string> unexpected = [];
        foreach (LogicalTensor tensor in tensors.Tensors)
        {
            if (!nameable.Contains(tensor.Name))
            {
                unexpected.Add(tensor.Name);
            }
        }

        if (missing.Count > 0 || unexpected.Count > 0)
        {
            throw new ModelException(
                $"model architecture '{architecture}' is not supported (supported architectures: " +
                $"{string.Join(", ", ModelProfiles.SupportedArchitectures)}): the llama-shaped fallback profile " +
                $"was attempted but the tensor table does not match it (missing tensors: " +
                $"{string.Join(", ", missing)}; unexpected tensors: {string.Join(", ", unexpected)})");
        }
    }

    private static string AliasName(string alias, int layer) =>
        alias.Replace("{layer}", layer.ToString(CultureInfo.InvariantCulture));

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
