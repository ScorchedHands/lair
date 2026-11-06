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
    private const int MaxUnicodeCodepoint = 0x10FFFF;

    // The process-static GPT-2 byte-encoding map: byte -> visual codepoint, and its inverse.
    // These are the fallback when a GGUF does not declare tokenizer.ggml.byte_to_utf8.
    private static readonly int[] Gpt2ByteToUtf8 = BuildGpt2ByteToUtf8();
    private static readonly int[] Gpt2VisualToByte = BuildVisualToByte(Gpt2ByteToUtf8);

    private readonly string[] _pieces;
    private readonly int[] _types;
    private readonly Dictionary<string, int> _pieceToId;
    private readonly int[] _specialIds;
    private readonly Dictionary<(string First, string Second), int> _mergeRanks;
    private readonly int[] _byteTokenByValue;
    private readonly int[] _byteToUtf8;
    private readonly int[] _visualToByte;
    private readonly PreSplitStrategy _split;
    private readonly bool _ignoreMerges;
    private readonly bool _cleanSpaces;
    private readonly bool _byteFallback;
    private readonly bool _addSpacePrefix;
    private readonly bool _utf8FlushBetweenControls;
    private readonly bool _addBos;
    private readonly Dictionary<string, int[]> _encodeCache = new(StringComparer.Ordinal);

    private GgufTokenizer(
        string[] pieces,
        int[] types,
        string[] merges,
        TokenizerProfile profile,
        int bosId,
        int eosId,
        int padId,
        int unkId,
        int eotId,
        bool addBos,
        bool byteFallback,
        bool addSpacePrefix,
        bool utf8FlushBetweenControls,
        int[] byteToUtf8,
        int[] byteTokenByValue)
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

        // All byte-map and byte-token state is resolved once at construction (FromGguf honors
        // tokenizer.ggml.byte_to_token / byte_to_utf8 over the defaults here) so the hot path stays concrete.
        _byteTokenByValue = byteTokenByValue;
        _byteToUtf8 = byteToUtf8;
        _visualToByte = BuildVisualToByte(byteToUtf8);

        _split = profile.Split;
        _ignoreMerges = profile.IgnoreMerges;
        _cleanSpaces = profile.CleanSpaces;
        _byteFallback = byteFallback;
        _addSpacePrefix = addSpacePrefix;
        _utf8FlushBetweenControls = utf8FlushBetweenControls;
        _addBos = addBos;
        BosId = bosId;
        EosId = eosId;
        PadId = padId;
        UnkId = unkId;
        EotId = eotId;
    }

    public int BosId { get; }

    public int EosId { get; }

    public int PadId { get; }

    public int UnkId { get; }

    /// <summary><c>tokenizer.ggml.eot_token_id</c> when the model declares one in range; <see cref="NullToken"/> otherwise. When declared, it is the stop signal in place of <see cref="EosId"/>.</summary>
    public int EotId { get; }

    /// <summary>Declared value of <c>tokenizer.ggml.add_space_prefix</c> (default <see langword="true"/>).
    /// Reported for parity diagnostics; the supported profiles' pre-split matches the pinned reference
    /// whether the key is true or false.</summary>
    public bool AddSpacePrefix => _addSpacePrefix;

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

    /// <summary>
    /// Builds a tokenizer from one GGUF, resolving every metadata-selected behavior once here:
    /// the profile from <c>tokenizer.ggml.pre</c>, the special ids (with profile-scoped fallbacks
    /// logged through <paramref name="logLine"/> when applied), <c>byte_fallback</c>,
    /// <c>byte_to_token</c>, <c>byte_to_utf8</c>, <c>add_space_prefix</c>, <c>eot_token_id</c>, and
    /// <c>utf8_flush_between_controls</c>.
    /// </summary>
    /// <param name="file">The GGUF to read tokenizer metadata from.</param>
    /// <param name="logLine">Sink for each structured profile-default log line; hosts may capture instead.</param>
    public static GgufTokenizer FromGguf(GgufFile file, Action<string>? logLine = null)
    {
        ArgumentNullException.ThrowIfNull(file);

        GgufMetadata metadata = file.Metadata;

        string model = metadata.GetString("tokenizer.ggml.model", "");
        if (model != "gpt2")
        {
            throw new GgufException(
                $"tokenizer.ggml.model is '{model}' (expected 'gpt2'): only GGUF BPE vocabularies are supported");
        }

        string pre = metadata.GetString("tokenizer.ggml.pre", TokenizerProfiles.DefaultPreKey);
        TokenizerProfile profile = TokenizerProfiles.Resolve(pre);

        string[] pieces = metadata.GetTokens();
        int[] types = metadata.GetTokenTypes();
        string[] merges = metadata.GetMerges();

        int bosId = ReadSpecialId(metadata, "tokenizer.ggml.bos_token_id", profile.BosFallbackId, pieces.Length, file, logLine);
        int eosId = ReadSpecialId(metadata, "tokenizer.ggml.eos_token_id", profile.EosFallbackId, pieces.Length, file, logLine);
        int padId = ReadSpecialId(metadata, "tokenizer.ggml.padding_token_id", NullToken, pieces.Length, file, logLine);
        int unkId = ReadSpecialId(metadata, "tokenizer.ggml.unknown_token_id", NullToken, pieces.Length, file, logLine);
        int eotId = ReadSpecialId(metadata, "tokenizer.ggml.eot_token_id", NullToken, pieces.Length, file, logLine);

        bool addBos = metadata.GetBool("tokenizer.ggml.add_bos_token", profile.AddBosDefault);
        bool byteFallback = metadata.GetBool("tokenizer.ggml.byte_fallback", true);
        bool addSpacePrefix = metadata.GetBool("tokenizer.ggml.add_space_prefix", true);
        bool utf8Flush = metadata.GetBool("tokenizer.ggml.utf8_flush_between_controls", true);

        int[] byteToUtf8 = metadata.Has("tokenizer.ggml.byte_to_utf8")
            ? ReadByteToUtf8(metadata, "tokenizer.ggml.byte_to_utf8")
            : Gpt2ByteToUtf8;

        int[] byteTokenByValue = ResolveByteTokenByValue(metadata, pieces, types);

        return new GgufTokenizer(
            pieces,
            types,
            merges,
            profile,
            bosId,
            eosId,
            padId,
            unkId,
            eotId,
            addBos,
            byteFallback,
            addSpacePrefix,
            utf8Flush,
            byteToUtf8,
            byteTokenByValue);
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
        StringBuilder text = new();

        foreach (int id in ids)
        {
            if (id < 0 || id >= _pieces.Length)
            {
                continue;
            }

            bool control = _types[id] is TokenTypeUnknown or TokenTypeControl or TokenTypeUserDefined;
            if (control && !_utf8FlushBetweenControls)
            {
                // utf8_flush_between_controls=false: a partial UTF-8 sequence accumulated before the control
                // boundary is flushed there — the incomplete tail decodes to replacement text instead of
                // stitching across the boundary.
                FlushBytes(bytes, text);
                if (includeSpecial)
                {
                    text.Append(_pieces[id]);
                }

                continue;
            }

            AppendPieceBytes(id, includeSpecial, bytes);
        }

        // Byte fallback tokens only form valid text when their bytes are reassembled
        // across token boundaries, so each flush window decodes as one byte buffer.
        FlushBytes(bytes, text);

        string decoded = text.ToString();
        return _cleanSpaces ? CleanSpaces(decoded) : decoded;
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
        // The profile resolved the strategy once at construction; this is a direct call, not a dispatch.
        // add_space_prefix is read and exposed but never alters the split: b12032 /tokenize on a model
        // that declares it false (smollm) still folds the leading space into the pre-token.
        List<int> offsets = _split(text, [text.Length]);

        List<string> words = new(offsets.Count);
        int start = 0;
        foreach (int offset in offsets)
        {
            words.Add(text.Substring(start, offset));
            start += offset;
        }

        return words;
    }

    private string ByteEncode(string word)
    {
        byte[] utf8 = Encoding.UTF8.GetBytes(word);
        char[] encoded = new char[utf8.Length];
        for (int i = 0; i < utf8.Length; i++)
        {
            encoded[i] = (char)_byteToUtf8[utf8[i]];
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
                    a.Rank != b.Rank ? a.Rank.CompareTo(b.Rank) : a.Left.CompareTo(b.Left)));

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

        if (!_byteFallback)
        {
            // byte_fallback=false: unencodable text never falls through to byte tokens — it routes to the
            // unknown token, or is dropped when the vocabulary declares none (matching the drop behavior
            // for bytes with no byte token today).
            if (UnkId != NullToken)
            {
                output.Add(UnkId);
            }

            return;
        }

        Span<char> single = stackalloc char[2];
        Span<byte> utf8 = stackalloc byte[4];

        foreach (Rune rune in symbol.EnumerateRunes())
        {
            int cp = rune.Value;
            if (cp < _visualToByte.Length && _visualToByte[cp] >= 0)
            {
                EmitByteToken(_visualToByte[cp], output);
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
            string atom = ((char)_byteToUtf8[byteValue]).ToString();
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
                else if (cp < _visualToByte.Length && _visualToByte[cp] >= 0)
                {
                    bytes.Add((byte)_visualToByte[cp]);
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

    private static int ReadSpecialId(
        GgufMetadata metadata,
        string key,
        int fallback,
        int vocabSize,
        GgufFile file,
        Action<string>? logLine)
    {
        if (!metadata.Has(key))
        {
            int resolved = fallback >= 0 && fallback < vocabSize ? fallback : NullToken;
            if (resolved != NullToken && logLine is not null)
            {
                // A profile-scoped default was applied because the GGUF does not declare the key: one
                // structured line naming the model, the key, and the default applied.
                string modelName = metadata.GetString("general.name", "");
                logLine($"tokenizer_default model='{(modelName.Length > 0 ? modelName : file.Path)}' key={key} default={resolved}");
            }

            return resolved;
        }

        long id = metadata.Get(key).AsLong();
        return id >= 0 && id < vocabSize ? (int)id : NullToken;
    }

    /// <summary>
    /// Resolves the byte-value -> byte-token-id table. A declared <c>tokenizer.ggml.byte_to_token</c>
    /// replaces the <c>&lt;0xNN&gt;</c> vocab scan entirely; each of its 256 entries must be a vocab id
    /// whose token type is byte, and an invalid entry fails at load naming the key and the byte index.
    /// </summary>
    private static int[] ResolveByteTokenByValue(GgufMetadata metadata, string[] pieces, int[] types)
    {
        int[] byValue = new int[256];

        if (metadata.Has("tokenizer.ggml.byte_to_token"))
        {
            int[] declared = ReadByteMap(metadata, "tokenizer.ggml.byte_to_token");
            for (int byteValue = 0; byteValue < 256; byteValue++)
            {
                int id = declared[byteValue];
                if (id < 0 || id >= pieces.Length)
                {
                    throw new GgufException(
                        $"GGUF metadata key 'tokenizer.ggml.byte_to_token' entry {byteValue} is {id} " +
                        $"(expected a token id in [0, {pieces.Length}) with byte type)");
                }

                if (types[id] != TokenTypeByte)
                {
                    throw new GgufException(
                        $"GGUF metadata key 'tokenizer.ggml.byte_to_token' entry {byteValue} maps to token {id} " +
                        $"'{pieces[id]}' which is not a byte token");
                }

                byValue[byteValue] = id;
            }

            return byValue;
        }

        Array.Fill(byValue, NullToken);
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
                if (byValue[value] == NullToken)
                {
                    byValue[value] = id;
                }
            }
        }

        return byValue;
    }

    /// <summary>
    /// Resolves the byte -> codepoint mapping (<c>tokenizer.ggml.byte_to_utf8</c>): exactly 256 entries,
    /// each a valid non-surrogate codepoint, or the load fails naming the key and the byte index.
    /// </summary>
    private static int[] ReadByteToUtf8(GgufMetadata metadata, string key)
    {
        int[] map = ReadByteMap(metadata, key);
        for (int byteValue = 0; byteValue < 256; byteValue++)
        {
            int codepoint = map[byteValue];
            if (codepoint < 0 || codepoint > MaxUnicodeCodepoint || (codepoint >= 0xD800 && codepoint <= 0xDFFF))
            {
                throw new GgufException(
                    $"GGUF metadata key '{key}' entry {byteValue} is {codepoint} (expected a Unicode codepoint in [0, {MaxUnicodeCodepoint}] that is not a surrogate)");
            }
        }

        return map;
    }

    private static int[] ReadByteMap(GgufMetadata metadata, string key)
    {
        int[] map = metadata.GetInt32Array(key);
        if (map.Length != 256)
        {
            throw new GgufException($"GGUF metadata key '{key}' has {map.Length} entries (expected 256)");
        }

        return map;
    }

    private static void FlushBytes(List<byte> bytes, StringBuilder text)
    {
        if (bytes.Count == 0)
        {
            return;
        }

        text.Append(Encoding.UTF8.GetString(CollectionsMarshal.AsSpan(bytes)));
        bytes.Clear();
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

    /// <summary>The GPT-2 byte-encoding map (byte -> visual codepoint): the fallback when a GGUF omits <c>tokenizer.ggml.byte_to_utf8</c>.</summary>
    private static int[] BuildGpt2ByteToUtf8()
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

    /// <summary>The inverse byte -> codepoint map (visual codepoint -> byte, -1 when unmapped), sized to cover the map's codepoints.</summary>
    private static int[] BuildVisualToByte(int[] byteToUtf8)
    {
        int maxCodepoint = 0xFF;
        foreach (int codepoint in byteToUtf8)
        {
            maxCodepoint = Math.Max(maxCodepoint, codepoint);
        }

        int[] map = new int[maxCodepoint + 1];
        Array.Fill(map, -1);
        for (int byteValue = 0; byteValue < byteToUtf8.Length; byteValue++)
        {
            map[byteToUtf8[byteValue]] = byteValue;
        }

        return map;
    }

    private readonly record struct Fragment(string Text, int TokenId);

    private readonly record struct BigramEntry(int Left, int Right, string Text);
}
