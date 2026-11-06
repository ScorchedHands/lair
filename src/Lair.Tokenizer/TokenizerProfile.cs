using System.Globalization;

using Lair.Loader;

namespace Lair.Tokenizer;

/// <summary>
/// Pre-split strategy: cuts the whole text into pre-token lengths (offsets) before the shared
/// byte-encode/merge pipeline runs. Resolved once per tokenizer by <see cref="TokenizerProfiles"/>.
/// </summary>
internal delegate List<int> PreSplitStrategy(string text, List<int> offsets);

/// <summary>
/// Load-time tokenizer profile for one <c>tokenizer.ggml.pre</c> key: a factory of concrete
/// tokenizer state — the pre-split strategy, the merge-lookup shortcut (<c>ignore_merges</c>),
/// decode space cleaning, the <c>add_bos</c> default, and the BOS/EOS fallback ids — resolved
/// once at construction so the hot BPE path stays concrete and never dispatches per pre-token.
/// </summary>
internal sealed record TokenizerProfile
{
    public required string PreKey { get; init; }

    public required PreSplitStrategy Split { get; init; }

    /// <summary>Whole pre-tokens hit the vocabulary before merges (llama.cpp ignore_merges).</summary>
    public required bool IgnoreMerges { get; init; }

    /// <summary>Decode applies llama.cpp's clean-spaces passes.</summary>
    public required bool CleanSpaces { get; init; }

    /// <summary>Default for <c>tokenizer.ggml.add_bos_token</c> when the GGUF does not declare it.</summary>
    public required bool AddBosDefault { get; init; }

    /// <summary>Default for <c>tokenizer.ggml.bos_token_id</c> when the GGUF does not declare it.</summary>
    public required int BosFallbackId { get; init; }

    /// <summary>Default for <c>tokenizer.ggml.eos_token_id</c> when the GGUF does not declare it.</summary>
    public required int EosFallbackId { get; init; }
}

/// <summary>
/// Registry of the supported tokenizer profiles, keyed by <c>tokenizer.ggml.pre</c>. It is the
/// single source for profile selection and for the supported-profile list in the rejection
/// message for an unknown key. A new profile is a new entry here plus its split strategy.
/// </summary>
internal static class TokenizerProfiles
{
    /// <summary>Profile used when a GGUF omits <c>tokenizer.ggml.pre</c>.</summary>
    public const string DefaultPreKey = "llama-bpe";

    private const int SpecialIdFallback = 11;

    private static readonly TokenizerProfile[] Profiles =
    [
        new TokenizerProfile
        {
            PreKey = "llama-bpe",
            Split = SplitLlama3,
            IgnoreMerges = true,
            CleanSpaces = true,
            AddBosDefault = true,
            BosFallbackId = SpecialIdFallback,
            EosFallbackId = SpecialIdFallback,
        },
        new TokenizerProfile
        {
            PreKey = "gpt2",
            Split = SplitGpt2,
            IgnoreMerges = false,
            CleanSpaces = true,
            AddBosDefault = false,
            BosFallbackId = SpecialIdFallback,
            EosFallbackId = SpecialIdFallback,
        },
        new TokenizerProfile
        {
            PreKey = "smollm",
            Split = SplitSmollm,
            IgnoreMerges = false,
            CleanSpaces = false,
            AddBosDefault = false,
            BosFallbackId = SpecialIdFallback,
            EosFallbackId = SpecialIdFallback,
        },
    ];

    private static readonly Dictionary<string, TokenizerProfile> ByPreKey =
        Profiles.ToDictionary(profile => profile.PreKey, StringComparer.Ordinal);

    /// <summary>Supported pre-tokenizer keys, in registry order.</summary>
    public static IEnumerable<string> SupportedPreKeys => Profiles.Select(profile => profile.PreKey);

    /// <summary>
    /// Resolves the profile for one <c>tokenizer.ggml.pre</c> value. An unknown key fails at load
    /// with a <see cref="GgufException"/> naming the value and the registry's supported profiles.
    /// </summary>
    public static TokenizerProfile Resolve(string preKey)
    {
        if (ByPreKey.TryGetValue(preKey, out TokenizerProfile? profile))
        {
            return profile;
        }

        throw new GgufException(
            $"unsupported pre-tokenizer '{preKey}' in tokenizer.ggml.pre (supported: " +
            $"{string.Join(", ", SupportedPreKeys.Select(key => $"'{key}'"))})");
    }

    private static List<int> SplitSmollm(string text, List<int> offsets) => SplitDigits(text, SplitGpt2(text, offsets));

    private static List<int> SplitDigits(string text, List<int> offsets)
    {
        List<int> next = [];
        int start = 0;
        foreach (int length in offsets)
        {
            int end = start + length;
            int runStart = start;
            for (int pos = start; pos < end;)
            {
                bool isDigit = char.IsNumber(text, pos);
                if (pos > runStart && isDigit != char.IsNumber(text, runStart))
                {
                    next.Add(pos - runStart);
                    runStart = pos;
                }

                // Step whole codepoints so a surrogate pair never lands on a digit-run boundary.
                pos += char.IsHighSurrogate(text[pos]) && pos + 1 < end ? 2 : 1;
            }

            next.Add(end - runStart);
            start = end;
        }

        return next;
    }

    private static List<int> SplitGpt2(string text, List<int> offsets)
    {
        List<int> words = [];
        int start = 0;
        foreach (int length in offsets)
        {
            int end = start + length;
            int previousEnd = start;
            int pos = start;

            while (pos < end)
            {
                void AddToken(int tokenEnd)
                {
                    if (tokenEnd > previousEnd)
                    {
                        words.Add(tokenEnd - previousEnd);
                    }

                    previousEnd = tokenEnd;
                }

                    // A surrogate pair is one codepoint and must stay in a single pre-token: splitting it would make
                    // UTF-8 encoding emit replacement bytes.
                    if (char.IsHighSurrogate(text[pos]) && pos + 1 < end && char.IsLowSurrogate(text[pos + 1]))
                    {
                        pos += 2;
                        AddToken(pos);
                        continue;
                    }

                    if (text[pos] == '\'' && pos + 1 < end)
                    {
                        int contractionLength = 0;
                    bool isContraction = IsAsciiLetter(text, pos + 1)
                        && IsContractionSuffix(text, pos, end, out contractionLength);

                    if (!isContraction)
                    {
                        contractionLength = 0;
                    }

                    if (contractionLength > 0)
                    {
                        pos += contractionLength;
                        AddToken(pos);
                        continue;
                    }
                }

                int letterStart = pos;
                if (text[pos] == ' ' && pos + 1 < end && char.IsLetter(text, pos + 1))
                {
                    letterStart = pos + 1;
                }

                if (char.IsLetter(text, letterStart))
                {
                    pos = letterStart;
                    while (pos < end && char.IsLetter(text, pos))
                    {
                        pos++;
                    }

                    AddToken(pos);
                    continue;
                }

                int numberStart = pos;
                if (text[pos] == ' ' && pos + 1 < end && char.IsNumber(text, pos + 1))
                {
                    numberStart = pos + 1;
                }

                if (char.IsNumber(text, numberStart))
                {
                    pos = numberStart;
                    while (pos < end && char.IsNumber(text, pos))
                    {
                        pos++;
                    }

                    AddToken(pos);
                    continue;
                }

                int symbolStart = pos;
                if (text[pos] == ' ' && pos + 1 < end && IsGpt2Symbol(text, pos + 1))
                {
                    symbolStart = pos + 1;
                }

                if (IsGpt2Symbol(text, symbolStart))
                {
                    pos = symbolStart;
                    while (pos < end && IsGpt2Symbol(text, pos))
                    {
                        pos++;
                    }

                    AddToken(pos);
                    continue;
                }

                int whitespaceRun = 0;
                while (pos + whitespaceRun < end && char.IsWhiteSpace(text, pos + whitespaceRun))
                {
                    whitespaceRun++;
                }

                if (whitespaceRun > 1 && pos + whitespaceRun < end)
                {
                    pos += whitespaceRun - 1;
                    AddToken(pos);
                    continue;
                }

                if (whitespaceRun > 0)
                {
                    pos += whitespaceRun;
                    AddToken(pos);
                    continue;
                }

                pos++;
                AddToken(pos);
            }

            start = end;
        }

        return words;
    }

    private static List<int> SplitLlama3(string text, List<int> offsets)
    {
        List<int> words = [];
        int start = 0;
        foreach (int length in offsets)
        {
            int end = start + length;
            int previousEnd = start;
            int pos = start;

            while (pos < end)
            {
                void AddToken(int tokenEnd)
                {
                    if (tokenEnd > previousEnd)
                    {
                        words.Add(tokenEnd - previousEnd);
                    }

                    previousEnd = tokenEnd;
                }

                    // A surrogate pair is one codepoint and must stay in a single pre-token: splitting it would make
                    // UTF-8 encoding emit replacement bytes.
                    if (char.IsHighSurrogate(text[pos]) && pos + 1 < end && char.IsLowSurrogate(text[pos + 1]))
                    {
                        pos += 2;
                        AddToken(pos);
                        continue;
                    }

                    if (text[pos] == '\'' && pos + 1 < end)
                    {
                        int suffixLength = 0;
                    bool isContraction = IsAsciiLetter(text, pos + 1)
                        && IsContractionSuffix(text, pos, end, out suffixLength, caseInsensitive: true);

                    if (!isContraction)
                    {
                        suffixLength = 0;
                    }

                    if (suffixLength > 0)
                    {
                        pos += suffixLength;
                        AddToken(pos);
                        continue;
                    }
                }

                if (text[pos] != '\r' && text[pos] != '\n' && !char.IsNumber(text, pos))
                {
                    int letterStart = char.IsLetter(text, pos) ? pos : pos + 1;
                    if (letterStart < end && char.IsLetter(text, letterStart))
                    {
                        pos = letterStart;
                        while (pos < end && char.IsLetter(text, pos))
                        {
                            pos++;
                        }

                        AddToken(pos);
                        continue;
                    }
                }

                if (char.IsNumber(text, pos))
                {
                    int groupStart = pos;
                    while (pos < end && char.IsNumber(text, pos))
                    {
                        pos++;
                        if (pos - groupStart >= 3)
                        {
                            AddToken(pos);
                            groupStart = pos;
                        }
                    }

                    AddToken(pos);
                    continue;
                }

                int symbolStart = pos;
                if (text[pos] == ' ' && pos + 1 < end && IsLlama3Symbol(text, pos + 1))
                {
                    symbolStart = pos + 1;
                }

                if (IsLlama3Symbol(text, pos) && IsLlama3Symbol(text, symbolStart))
                {
                    pos = symbolStart;
                    while (pos < end && IsLlama3Symbol(text, pos))
                    {
                        pos++;
                    }

                    while (pos < end && (text[pos] == '\r' || text[pos] == '\n'))
                    {
                        pos++;
                    }

                    AddToken(pos);
                    continue;
                }

                int whitespaceRun = 0;
                int lastNewlineEnd = 0;
                while (pos + whitespaceRun < end && char.IsWhiteSpace(text, pos + whitespaceRun))
                {
                    if (text[pos + whitespaceRun] == '\r' || text[pos + whitespaceRun] == '\n')
                    {
                        lastNewlineEnd = pos + whitespaceRun + 1;
                    }

                    whitespaceRun++;
                }

                if (lastNewlineEnd > 0)
                {
                    pos = lastNewlineEnd;
                    AddToken(pos);
                    continue;
                }

                if (whitespaceRun > 1 && pos + whitespaceRun < end)
                {
                    pos += whitespaceRun - 1;
                    AddToken(pos);
                    continue;
                }

                if (whitespaceRun > 0)
                {
                    pos += whitespaceRun;
                    AddToken(pos);
                    continue;
                }

                pos++;
                AddToken(pos);
            }

            start = end;
        }

        return words;
    }

    private static bool IsAsciiLetter(string text, int pos)
    {
        char ch = text[pos];
        return ch is >= 'a' and <= 'z' or >= 'A' and <= 'Z';
    }

    private static bool IsContractionSuffix(string text, int pos, int limit, out int suffixLength, bool caseInsensitive = false)
    {
        char first = text[pos + 1];
        char? second = pos + 2 < limit ? text[pos + 2] : null;

        if (caseInsensitive)
        {
            first = char.ToLowerInvariant(first);
            second = second is null ? null : char.ToLowerInvariant(second.Value);
        }

        if (first is 's' or 't' or 'm' or 'd')
        {
            suffixLength = 2;
            return true;
        }

        if (second is not null && (first, second.Value) is ('r', 'e') or ('v', 'e') or ('l', 'l'))
        {
            suffixLength = 3;
            return true;
        }

        suffixLength = 0;
        return false;
    }

    private static bool IsGpt2Symbol(string text, int pos)
    {
        return pos < text.Length
            && !char.IsWhiteSpace(text, pos)
            && !char.IsLetter(text, pos)
            && !char.IsNumber(text, pos)
            && !IsUndefined(text[pos]);
    }

    private static bool IsLlama3Symbol(string text, int pos)
    {
        return pos < text.Length
            && !char.IsWhiteSpace(text, pos)
            && !char.IsLetter(text, pos)
            && !char.IsNumber(text, pos)
            && !IsUndefined(text[pos]);
    }

    private static bool IsUndefined(char ch)
    {
        UnicodeCategory category = CharUnicodeInfo.GetUnicodeCategory(ch);
        return category is UnicodeCategory.Format
            or UnicodeCategory.EnclosingMark
            or UnicodeCategory.Surrogate
            or UnicodeCategory.PrivateUse
            or UnicodeCategory.MathSymbol
            or UnicodeCategory.OtherNotAssigned;
    }
}
