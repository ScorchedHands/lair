using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;

using Lair.Loader;

namespace Lair.Tokenizer;

/// <summary>
/// BPE tokenizer for GGUF-embedded vocabularies, following the semantics of
/// llama.cpp's llama-vocab BPE path: codepoint pre-splitting, GPT-2 byte
/// encoding, merge-rank BPE, special-token partitioning, and byte fallback.
/// </summary>
public sealed class GgufTokenizer
{
    /// <summary>GGUF token types, matching the tokenizer.ggml.token_type encoding.</summary>
    public const int TokenTypeUnused = 0;
    public const int TokenTypeNormal = 1;
    public const int TokenTypeUnknown = 2;
    public const int TokenTypeControl = 3;
    public const int TokenTypeBegin = 4;
    public const int TokenTypeEnd = 5;
    public const int TokenTypeByte = 6;
    public const int TokenTypeUndefined = 7;
    public const int TokenTypeUserDefined = 8;

    private const int NullToken = -1;
    private const int EncodeCacheLimit = 50_000;

    private static readonly int[] ByteToEncoded = BuildByteToEncoded();
    private static readonly int[] EncodedToByte = BuildEncodedToByte();

    private readonly string[] _pieces;
    private readonly int[] _types;
    private readonly Dictionary<string, int> _pieceToId;
    private readonly int[] _specialIds;
    private readonly Dictionary<(string First, string Second), int> _mergeRanks;
    private readonly int[] _byteTokenByValue;
    private readonly PreRule _pre;
    private readonly bool _ignoreMerges;
    private readonly bool _cleanSpaces;
    private readonly bool _addBos;
    private readonly Dictionary<string, int[]> _encodeCache = new(StringComparer.Ordinal);

    public GgufTokenizer(
        string[] pieces,
        int[] types,
        string[] merges,
        PreRule pre,
        int bosId = NullToken,
        int eosId = NullToken,
        int padId = NullToken,
        int unkId = NullToken,
        bool? addBos = null)
    {
        ArgumentNullException.ThrowIfNull(pieces);
        ArgumentNullException.ThrowIfNull(types);
        ArgumentNullException.ThrowIfNull(merges);
        if (types.Length != pieces.Length)
        {
            throw new ArgumentException("token types and pieces must have the same length", nameof(types));
        }

        _pieces = pieces;
        _types = types;

        _pieceToId = new Dictionary<string, int>(pieces.Length, StringComparer.Ordinal);
        for (int id = 0; id < pieces.Length; id++)
        {
            _pieceToId[pieces[id]] = id;
        }

        List<int> specials = [];
        for (int id = 0; id < types.Length; id++)
        {
            if (types[id] is TokenTypeUnknown or TokenTypeControl or TokenTypeUserDefined)
            {
                specials.Add(id);
            }
        }

        _specialIds = specials
            .OrderByDescending(id => Encoding.UTF8.GetByteCount(_pieces[id]))
            .ToArray();

        _mergeRanks = new Dictionary<(string, string), int>(merges.Length);
        for (int rank = 0; rank < merges.Length; rank++)
        {
            string merge = merges[rank];
            int sep = merge.Length > 0 ? merge.IndexOf(' ', 1) : -1;
            if (sep >= 0)
            {
                _mergeRanks[(merge[..sep], merge[(sep + 1)..])] = rank;
            }
            else
            {
                _mergeRanks[(string.Empty, string.Empty)] = rank;
            }
        }

        _byteTokenByValue = new int[256];
        Array.Fill(_byteTokenByValue, NullToken);
        for (int id = 0; id < pieces.Length; id++)
        {
            if (types[id] != TokenTypeByte || pieces[id].Length != 6)
            {
                continue;
            }

            string piece = pieces[id];
            if (piece.StartsWith("<0x", StringComparison.Ordinal) && piece[5] == '>'
                && TryHexDigit(piece[3], out int high) && TryHexDigit(piece[4], out int low))
            {
                int value = (high << 4) | low;
                if (_byteTokenByValue[value] == NullToken)
                {
                    _byteTokenByValue[value] = id;
                }
            }
        }

        _pre = pre;
        _ignoreMerges = pre == PreRule.LlamaBpe;
        _cleanSpaces = pre is not PreRule.Smollm;
        _addBos = addBos ?? pre == PreRule.LlamaBpe;
        BosId = bosId;
        EosId = eosId;
        PadId = padId;
        UnkId = unkId;
    }

    public int BosId { get; }

    public int EosId { get; }

    public int PadId { get; }

    public int UnkId { get; }

    public int VocabSize => _pieces.Length;

    public bool AddBosToken => _addBos;

    public IReadOnlyList<GgufTokenAttrs> Tokens
    {
        get
        {
            GgufTokenAttrs[] attrs = new GgufTokenAttrs[_pieces.Length];
            for (int id = 0; id < attrs.Length; id++)
            {
                attrs[id] = new GgufTokenAttrs(_pieces[id], _types[id]);
            }

            return attrs;
        }
    }

    public static GgufTokenizer FromGguf(GgufFile file)
    {
        ArgumentNullException.ThrowIfNull(file);

        GgufMetadata metadata = file.Metadata;

        string model = metadata.GetString("tokenizer.ggml.model", "");
        if (model != "gpt2")
        {
            throw new GgufException(
                $"tokenizer.ggml.model is '{model}' (expected 'gpt2'): only GGUF BPE vocabularies are supported");
        }

        string pre = metadata.GetString("tokenizer.ggml.pre", "llama-bpe");
        PreRule rule = pre switch
        {
            "llama-bpe" => PreRule.LlamaBpe,
            "gpt2" => PreRule.Gpt2,
            "smollm" => PreRule.Smollm,
            _ => throw new GgufException(
                $"unsupported pre-tokenizer '{pre}' in tokenizer.ggml.pre (supported: 'llama-bpe', 'gpt2', 'smollm')"),
        };

        string[] pieces = metadata.GetTokens();
        int[] types = metadata.GetTokenTypes();
        string[] merges = metadata.GetMerges();

        int bosId = ReadSpecialId(metadata, "tokenizer.ggml.bos_token_id", fallback: 11, pieces.Length);
        int eosId = ReadSpecialId(metadata, "tokenizer.ggml.eos_token_id", fallback: 11, pieces.Length);
        int padId = ReadSpecialId(metadata, "tokenizer.ggml.padding_token_id", fallback: NullToken, pieces.Length);
        int unkId = ReadSpecialId(metadata, "tokenizer.ggml.unknown_token_id", fallback: NullToken, pieces.Length);

        bool addBos = metadata.GetBool("tokenizer.ggml.add_bos_token", rule == PreRule.LlamaBpe);

        return new GgufTokenizer(pieces, types, merges, rule, bosId, eosId, padId, unkId, addBos);
    }

    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return EncodeCached(text, withBos: false);
    }

    public IReadOnlyList<int> EncodeWithBos(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return EncodeCached(text, withBos: true);
    }

    public string Decode(IReadOnlyList<int> ids, bool includeSpecial = false)
    {
        ArgumentNullException.ThrowIfNull(ids);

        List<byte> bytes = new(ids.Count * 4);
        foreach (int id in ids)
        {
            if (id < 0 || id >= _pieces.Length)
            {
                continue;
            }

            AppendPieceBytes(id, includeSpecial, bytes);
        }

        // Byte fallback tokens only form valid text when their bytes are reassembled
        // across token boundaries, so the whole id stream decodes as one byte buffer.
        string text = Encoding.UTF8.GetString(CollectionsMarshal.AsSpan(bytes));
        return _cleanSpaces ? CleanSpaces(text) : text;
    }

    public int? IdOf(string token)
    {
        ArgumentNullException.ThrowIfNull(token);
        return _pieceToId.TryGetValue(token, out int id) ? id : null;
    }

    private IReadOnlyList<int> EncodeCached(string text, bool withBos)
    {
        string key = withBos ? "\u0000bos\u0000" + text : text;
        if (_encodeCache.TryGetValue(key, out int[]? cached))
        {
            return cached;
        }

        List<int> ids = [];
        if (withBos && _addBos && BosId != NullToken)
        {
            ids.Add(BosId);
        }

        foreach (Fragment fragment in PartitionSpecial(text))
        {
            if (fragment.TokenId != NullToken)
            {
                ids.Add(fragment.TokenId);
            }
            else
            {
                TokenizeText(fragment.Text, ids);
            }
        }

        int[] result = ids.ToArray();
        if (_encodeCache.Count >= EncodeCacheLimit)
        {
            _encodeCache.Clear();
        }

        _encodeCache[key] = result;
        return result;
    }

    private List<Fragment> PartitionSpecial(string text)
    {
        List<Fragment> fragments = [new Fragment(text, NullToken)];

        foreach (int specialId in _specialIds)
        {
            string piece = _pieces[specialId];
            List<Fragment> next = [];

            foreach (Fragment fragment in fragments)
            {
                if (fragment.TokenId != NullToken)
                {
                    next.Add(fragment);
                    continue;
                }

                string scan = fragment.Text;
                int position = 0;
                while (true)
                {
                    int match = scan.IndexOf(piece, position, StringComparison.Ordinal);
                    if (match < 0)
                    {
                        break;
                    }

                    if (match > position)
                    {
                        next.Add(new Fragment(scan[position..match], NullToken));
                    }

                    next.Add(new Fragment(string.Empty, specialId));
                    position = match + piece.Length;
                }

                if (position < scan.Length)
                {
                    next.Add(new Fragment(scan[position..], NullToken));
                }
            }

            fragments = next;
        }

        return fragments;
    }

    private void TokenizeText(string text, List<int> output)
    {
        foreach (string word in PreSplit(text))
        {
            if (_ignoreMerges && _pieceToId.TryGetValue(word, out int wholeId))
            {
                output.Add(wholeId);
                continue;
            }

            string encoded = ByteEncode(word);
            ApplyMerges(encoded, output);
        }
    }

    private List<string> PreSplit(string text)
    {
        List<int> offsets = [text.Length];

        offsets = _pre switch
        {
            PreRule.LlamaBpe => SplitLlama3(text, offsets),
            PreRule.Gpt2 => SplitGpt2(text, offsets),
            PreRule.Smollm => SplitDigits(text, SplitGpt2(text, offsets)),
            _ => throw new InvalidOperationException("unknown pre-tokenizer rule"),
        };

        List<string> words = new(offsets.Count);
        int start = 0;
        foreach (int offset in offsets)
        {
            words.Add(text.Substring(start, offset));
            start += offset;
        }

        return words;
    }

    private static List<int> SplitDigits(string text, List<int> offsets)
    {
        List<int> next = [];
        int start = 0;
        foreach (int length in offsets)
        {
            int end = start + length;
            int runStart = start;
            for (int pos = start; pos < end; pos++)
            {
                bool isDigit = char.IsNumber(text, pos);
                if (pos > runStart && (!isDigit || !char.IsNumber(text, runStart)))
                {
                    next.Add(pos - runStart);
                    runStart = pos;
                }
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

    private static string ByteEncode(string word)
    {
        byte[] utf8 = Encoding.UTF8.GetBytes(word);
        char[] encoded = new char[utf8.Length];
        for (int i = 0; i < utf8.Length; i++)
        {
            encoded[i] = (char)ByteToEncoded[utf8[i]];
        }

        return new string(encoded);
    }

    private void ApplyMerges(string word, List<int> output)
    {
        int n = word.Length;
        int[] symbolEnd = new int[n];
        int[] next = new int[n];
        int[] previous = new int[n];
        bool[] alive = new bool[n];
        for (int i = 0; i < n; i++)
        {
            symbolEnd[i] = i + 1;
            next[i] = i + 1;
            previous[i] = i - 1;
            alive[i] = true;
        }

        int SymbolStart(int index) => previous[index] < 0 ? 0 : symbolEnd[previous[index]];

        if (n > 1)
        {
            next[n - 1] = -1;

            PriorityQueue<BigramEntry, (int Rank, int Left)> queue =
                new(Comparer<(int Rank, int Left)>.Create((a, b) =>
                    a.Rank != b.Rank ? b.Rank.CompareTo(a.Rank) : b.Left.CompareTo(a.Left)));

            void TryAddBigram(int left, int right)
            {
                if (left < 0 || right < 0 || right >= n || !alive[left] || !alive[right])
                {
                    return;
                }

                int leftStart = SymbolStart(left);
                int rightStart = SymbolStart(right);
                string leftText = word.Substring(leftStart, symbolEnd[left] - leftStart);
                string rightText = word.Substring(rightStart, symbolEnd[right] - rightStart);
                if (!_mergeRanks.TryGetValue((leftText, rightText), out int rank))
                {
                    return;
                }

                queue.Enqueue(new BigramEntry(left, right, leftText + rightText), (rank, left));
            }

            for (int i = 0; i + 1 < n; i++)
            {
                TryAddBigram(i, next[i]);
            }

            while (true)
            {
                if (!queue.TryDequeue(out BigramEntry bigram, out _))
                {
                    break;
                }

                int left = bigram.Left;
                int right = bigram.Right;

                if (!alive[left] || !alive[right])
                {
                    continue;
                }

                int leftStart = SymbolStart(left);
                int rightStart = SymbolStart(right);
                string leftText = word.Substring(leftStart, symbolEnd[left] - leftStart);
                string rightText = word.Substring(rightStart, symbolEnd[right] - rightStart);
                if (leftText + rightText != bigram.Text)
                {
                    continue;
                }

                symbolEnd[left] = symbolEnd[right];
                alive[right] = false;

                int rightNext = next[right];
                next[left] = rightNext;
                if (rightNext >= 0 && rightNext < n)
                {
                    previous[rightNext] = left;
                }

                TryAddBigram(previous[left], left);
                TryAddBigram(left, next[left]);
            }
        }

        for (int i = 0; i < n; i++)
        {
            if (!alive[i])
            {
                continue;
            }

            int start = SymbolStart(i);
            EmitSymbol(word.Substring(start, symbolEnd[i] - start), output);
        }
    }

    private void EmitSymbol(string symbol, List<int> output)
    {
        if (_pieceToId.TryGetValue(symbol, out int id))
        {
            output.Add(id);
            return;
        }

        Span<char> single = stackalloc char[2];
        Span<byte> utf8 = stackalloc byte[4];

        foreach (Rune rune in symbol.EnumerateRunes())
        {
            int cp = rune.Value;
            if (cp < EncodedToByte.Length && EncodedToByte[cp] >= 0)
            {
                EmitByteToken(EncodedToByte[cp], output);
                continue;
            }

            int charCount = rune.EncodeToUtf16(single);
            int byteCount = Encoding.UTF8.GetBytes(single[..charCount], utf8);
            for (int i = 0; i < byteCount; i++)
            {
                EmitByteToken(utf8[i], output);
            }
        }
    }

    private void EmitByteToken(int byteValue, List<int> output)
    {
        int id = _byteTokenByValue[byteValue];
        if (id == NullToken)
        {
            string atom = ((char)ByteToEncoded[byteValue]).ToString();
            id = _pieceToId.TryGetValue(atom, out int atomId) ? atomId : NullToken;
        }

        if (id != NullToken)
        {
            output.Add(id);
        }
    }

    private void AppendPieceBytes(int id, bool includeSpecial, List<byte> bytes)
    {
        int type = _types[id];
        if (!includeSpecial && type is TokenTypeUnknown or TokenTypeControl or TokenTypeUserDefined)
        {
            return;
        }

        string piece = _pieces[id];

        if (type == TokenTypeByte)
        {
            if (piece.Length == 6 && piece.StartsWith("<0x", StringComparison.Ordinal) && piece[5] == '>'
                && TryHexDigit(piece[3], out int high) && TryHexDigit(piece[4], out int low))
            {
                bytes.Add((byte)((high << 4) | low));
                return;
            }

            foreach (byte b in Encoding.UTF8.GetBytes(piece))
            {
                bytes.Add(b);
            }

            return;
        }

        if (type == TokenTypeNormal)
        {
            Span<char> single = stackalloc char[2];
            Span<byte> utf8 = stackalloc byte[4];

            foreach (Rune rune in piece.EnumerateRunes())
            {
                int cp = rune.Value;
                if (cp < 0x100)
                {
                    bytes.Add((byte)cp);
                }
                else if (cp < EncodedToByte.Length && EncodedToByte[cp] >= 0)
                {
                    bytes.Add((byte)EncodedToByte[cp]);
                }
                else
                {
                    int charCount = rune.EncodeToUtf16(single);
                    int byteCount = Encoding.UTF8.GetBytes(single[..charCount], utf8);
                    for (int i = 0; i < byteCount; i++)
                    {
                        bytes.Add(utf8[i]);
                    }
                }
            }

            return;
        }

        foreach (byte b in Encoding.UTF8.GetBytes(piece))
        {
            bytes.Add(b);
        }
    }

    private static string CleanSpaces(string text)
    {
        if (text.Length == 0)
        {
            return text;
        }

        char[] buffer = text.ToCharArray();

        int total = 1;
        for (int i = 1; i < text.Length; i++)
        {
            char x = buffer[i];
            if (buffer[total - 1] == ' ' && x is '?' or '!' or '.' or ',')
            {
                total--;
            }

            buffer[total++] = x;
        }

        int pass2 = total;
        total = 1;
        for (int i = 1; i < pass2 - 1; i++)
        {
            char x = buffer[i];

            // " ' ": drop the space before and after a lone apostrophe.
            if (x == '\'' && buffer[total - 1] == ' ' && buffer[i + 1] == ' ')
            {
                total--;
                i++;
            }

            buffer[total++] = x;
        }

        if (pass2 > 1)
        {
            buffer[total++] = buffer[pass2 - 1];
        }

        int pass3 = total;
        total = 1;
        for (int i = 1; i < pass3; i++)
        {
            char x = buffer[i];
            if (buffer[total - 1] == ' ' && x == '\'' && i + 1 < pass3)
            {
                char x1 = buffer[i + 1];
                if (x1 is 's' or 'm')
                {
                    total--;
                }
                else if (x1 is not ('t' or 'd') && i + 2 < pass3)
                {
                    char x2 = buffer[i + 2];
                    if ((x1 == 'r' && x2 == 'e') || (x1 == 'v' && x2 == 'e'))
                    {
                        total--;
                    }
                }
            }

            buffer[total++] = x;
        }

        return new string(buffer, 0, total);
    }

    private static int ReadSpecialId(GgufMetadata metadata, string key, int fallback, int vocabSize)
    {
        if (!metadata.Has(key))
        {
            return fallback >= 0 && fallback < vocabSize ? fallback : NullToken;
        }

        long id = metadata.Get(key).AsLong();
        return id >= 0 && id < vocabSize ? (int)id : NullToken;
    }

    private static bool TryHexDigit(char ch, out int value)
    {
        if (ch is >= '0' and <= '9')
        {
            value = ch - '0';
            return true;
        }

        if (ch is >= 'A' and <= 'F')
        {
            value = ch - 'A' + 10;
            return true;
        }

        if (ch is >= 'a' and <= 'f')
        {
            value = ch - 'a' + 10;
            return true;
        }

        value = 0;
        return false;
    }

    private static int[] BuildByteToEncoded()
    {
        int[] map = new int[256];
        int n = 0;
        for (int ch = 0; ch < 256; ch++)
        {
            if (ch is >= 0x21 and <= 0x7E or >= 0xA1 and <= 0xAC or >= 0xAE and <= 0xFF)
            {
                map[ch] = ch;
            }
            else
            {
                map[ch] = 256 + n;
                n++;
            }
        }

        return map;
    }

    private static int[] BuildEncodedToByte()
    {
        int[] map = new int[0x200];
        Array.Fill(map, -1);
        for (int ch = 0; ch < 256; ch++)
        {
            map[ByteToEncoded[ch]] = ch;
        }

        return map;
    }

    public enum PreRule
    {
        LlamaBpe,
        Gpt2,
        Smollm,
    }

    private readonly record struct Fragment(string Text, int TokenId);

    private readonly record struct BigramEntry(int Left, int Right, string Text);
}
