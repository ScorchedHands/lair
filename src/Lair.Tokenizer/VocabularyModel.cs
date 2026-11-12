using Lair.Loader;

namespace Lair.Tokenizer;

/// <summary>
/// The vocabulary implementations <see cref="GgufTokenizer"/> can build, selected from
/// <c>tokenizer.ggml.model</c> at load: the BPE pipeline (pre-split, byte-encode, merge-rank)
/// and the SentencePiece pipeline (word-boundary spelling with the U+2581 marker, score-driven
/// piece merging, byte fallback). The selection axis is orthogonal to the <c>tokenizer.ggml.pre</c>
/// profiles, which are BPE pre-split machinery.
/// </summary>
internal enum VocabularyModel
{
    Bpe,
    Spm,
}

/// <summary>
/// Registry mapping <c>tokenizer.ggml.model</c> values to vocabulary implementations. An absent
/// key keeps the BPE implementation; an unknown value fails at load naming the value and the
/// supported list, in the same posture as an unknown <c>tokenizer.ggml.pre</c>. The SPM keys are
/// ordered first so a multi-key lookup probe lands on the SPM path.
/// </summary>
internal static class VocabularyModels
{
    private static readonly (string Key, VocabularyModel Model)[] Models =
    [
        // llama.cpp's SentencePiece marker: the gemma, phi, and older llama-family vocab files.
        ("llama", VocabularyModel.Spm),
        ("spm", VocabularyModel.Spm),
        ("gpt2", VocabularyModel.Bpe),
        // llama.cpp writes this pre-tokenizer name into tokenizer.ggml.model on some BPE files.
        ("llama-bpe", VocabularyModel.Bpe),
    ];

    private static readonly Dictionary<string, VocabularyModel> ByModelKey =
        Models.ToDictionary(entry => entry.Key, entry => entry.Model);

    /// <summary>Supported vocabulary-model keys, in registry order.</summary>
    public static IEnumerable<string> SupportedModelKeys => Models.Select(entry => entry.Key);

    /// <summary>
    /// Resolves the vocabulary implementation for one <c>tokenizer.ggml.model</c> value. An empty
    /// key (the GGUF omits it) keeps the BPE implementation; an unknown value fails at load with
    /// a <see cref="GgufException"/> naming the value and the registry's supported values.
    /// </summary>
    public static VocabularyModel Resolve(string modelKey)
    {
        if (modelKey.Length == 0)
        {
            return VocabularyModel.Bpe;
        }

        if (ByModelKey.TryGetValue(modelKey, out VocabularyModel model))
        {
            return model;
        }

        throw new GgufException(
            $"unsupported vocabulary model: tokenizer.ggml.model is '{modelKey}' (supported: " +
            $"{string.Join(", ", SupportedModelKeys.Select(key => $"'{key}'"))})");
    }
}
