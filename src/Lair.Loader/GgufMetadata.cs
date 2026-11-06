namespace Lair.Loader;

/// <summary>
/// KV metadata parsed from a GGUF file, keyed by GGUF metadata key.
/// </summary>
public sealed class GgufMetadata
{
    private readonly Dictionary<string, GgufValue> _values;

    internal GgufMetadata(Dictionary<string, GgufValue> values)
    {
        _values = values;
    }

    public int Count => _values.Count;

    public IEnumerable<string> Keys => _values.Keys;

    public bool Has(string key) => _values.ContainsKey(key);

    public GgufValue Get(string key) =>
        _values.TryGetValue(key, out GgufValue? value)
            ? value
            : throw new GgufException($"GGUF metadata key '{key}' is not present");

    public int GetInt(string key, int fallback) =>
        Has(key) ? checked((int)Get(key).AsLong()) : fallback;

    public bool GetBool(string key, bool fallback) =>
        Has(key) ? Get(key).AsBool() : fallback;

    public string GetString(string key, string fallback) =>
        Has(key) ? Get(key).AsString() : fallback;

    public string? ChatTemplate =>
        Has("tokenizer.chat_template") ? Get("tokenizer.chat_template").AsString() : null;

    public string[] GetTokens() => TokenStringArray("tokenizer.ggml.tokens");

    public string[] GetMerges() => TokenStringArray("tokenizer.ggml.merges");

    public int[] GetTokenTypes() => GetInt32Array("tokenizer.ggml.token_type");

    /// <summary>Required int32-array value (e.g. <c>tokenizer.ggml.byte_to_token</c>), failing like the other required accessors when the key is absent.</summary>
    public int[] GetInt32Array(string key)
    {
        if (!Has(key))
        {
            throw new GgufException($"GGUF metadata key '{key}' is not present");
        }

        return Get(key).AsInt32Array();
    }

    private string[] TokenStringArray(string key)
    {
        if (!Has(key))
        {
            throw new GgufException($"GGUF metadata key '{key}' is not present");
        }

        IReadOnlyList<GgufValue> elements = Get(key).AsArray();
        string[] values = new string[elements.Count];
        for (int i = 0; i < elements.Count; i++)
        {
            values[i] = elements[i].AsString();
        }

        return values;
    }
}
