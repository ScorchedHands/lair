using System.Globalization;

namespace Lair.Loader;

/// <summary>
/// Recognizes the llama.cpp split-GGUF file-name convention
/// `<prefix>-NNNNN-of-NNNNN.<ext>` — five-digit, 1-based shard numbers — and
/// rebuilds the sibling file names of a set in ascending order. A name that
/// does not carry exactly this shape is never treated as a split: an ordinary
/// `model.gguf` (or any other name) stays a single-file model. The prefix is
/// taken as everything before the last dash preceding `-of-`, so prefixes that
/// themselves contain dashes (`model-q4_k-split-00001-of-00006.gguf`) resolve
/// correctly.
/// </summary>
public static class GgufShardNaming
{
    /// <summary>Number of digits in each shard number of the split name shape.</summary>
    public const int ShardDigits = 5;

    /// <summary>
    /// Parses `<prefix>-NNNNN-of-NNNNN.<ext>` into the prefix, the extension
    /// (with its leading dot, empty when the name carries none), the 1-based
    /// shard index, and the declared shard total. Returns false — leaving every
    /// out value empty/zero — when the name does not carry the split shape.
    /// </summary>
    public static bool TryParse(string fileName, out string prefix, out string extension, out int index, out int total)
    {
        ArgumentNullException.ThrowIfNull(fileName);

        prefix = string.Empty;
        extension = string.Empty;
        index = 0;
        total = 0;

        int dot = fileName.LastIndexOf('.');
        string stem = dot >= 0 ? fileName[..dot] : fileName;
        extension = dot >= 0 ? fileName[dot..] : string.Empty;

        int ofPosition = stem.LastIndexOf("-of-", StringComparison.Ordinal);
        if (ofPosition < 0)
        {
            return false;
        }

        if (!TryShardNumber(stem[(ofPosition + 4)..], out total))
        {
            return false;
        }

        string head = stem[..ofPosition];
        int dash = head.LastIndexOf('-');
        if (dash < 0)
        {
            return false;
        }

        if (!TryShardNumber(head[(dash + 1)..], out index))
        {
            return false;
        }

        prefix = head[..dash];
        return true;
    }

    /// <summary>
    /// The file names of shards 1..total for a parsed split name, in ascending
    /// shard order. Names are rebuilt from the parsed shape rather than matched
    /// loosely, so a shard whose own name disagrees with its position cannot
    /// silently take part in the set under a different identity.
    /// </summary>
    public static IEnumerable<string> ShardFileNames(string prefix, string extension, int total)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(total, 1);

        for (int i = 1; i <= total; i++)
        {
            yield return $"{prefix}-{i:D5}-of-{total:D5}{extension}";
        }
    }

    private static bool TryShardNumber(string text, out int value)
    {
        value = 0;
        if (text.Length != ShardDigits)
        {
            return false;
        }

        foreach (char ch in text)
        {
            if (!char.IsAsciiDigit(ch))
            {
                return false;
            }
        }

        return int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out value);
    }
}
