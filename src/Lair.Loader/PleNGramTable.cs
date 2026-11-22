using System.Buffers;

using Lair.Contracts;

namespace Lair.Loader;
/// <summary>
/// Refusal raised when a model's PLE n-gram metadata or table geometry cannot drive a row-indexed gather. Every message
/// names the model file, the tensor, and the metadata field it rejected, so a broken sidecar or an architecture this seam
/// cannot read reports what is wrong instead of falling back to reading the whole table.
/// </summary>
public sealed class PleTableException : Exception
{
    public PleTableException(string message)
        : base(message)
    {
    }
}

/// <summary>
/// The n-gram geometry and hash constants of one model's PLE (per-layer token embedding) table, read from GGUF metadata
/// under the resolved architecture prefix: <c>&lt;arch&gt;.ple.ngram_size</c>, <c>&lt;arch&gt;.ple.heads_per_ngram</c>,
/// <c>&lt;arch&gt;.ple.layer_multipliers</c> (an array at least as long as the n-gram size), <c>&lt;arch&gt;.ple.head_offsets</c>
/// and <c>&lt;arch&gt;.ple.head_vocab_sizes</c> (arrays covering every head), <c>&lt;arch&gt;.ple.eos_token_id</c>, and
/// <c>&lt;arch&gt;.embedding_length_per_layer_input</c>. Heads are ordered by n-gram order: the heads of order <c>n</c> start at
/// <c>(n - 2) * heads_per_ngram</c>, so head count is <c>(ngram_size - 1) * heads_per_ngram</c>, exactly as llama.cpp's qwen4exp
/// gather indexes them. Every declared head range must land inside the backing tensor's row space, and the table must pack its
/// payload as fixed-stride rows — a shape this seam cannot confirm refuses the gather rather than copying the whole table.
/// </summary>
public sealed class PleNgramTable
{
    private readonly ulong[] _multipliers;
    private readonly long[] _headOffsets;
    private readonly long[] _headVocabSizes;

    // llama.cpp reports a predecessor with no cache cell as its null token id; a window position past the known history reads as exactly that.
    private const int MissingTokenSentinel = -1;

    // The normalized n-gram context is small in every real table; larger declarations take pooled scratch instead of growing the stack frame.
    private const int StackContextLimit = 64;

    private PleNgramTable(
        string modelPath,
        LogicalTensor tensor,
        int ngramSize,
        int headsPerNgram,
        int eosTokenId,
        ulong[] multipliers,
        long[] headOffsets,
        long[] headVocabSizes)
    {
        ModelPath = modelPath;
        Tensor = tensor;
        NgramSize = ngramSize;
        HeadsPerNgram = headsPerNgram;
        HeadCount = (ngramSize - 1) * headsPerNgram;
        EosTokenId = eosTokenId;
        _multipliers = multipliers;
        _headOffsets = headOffsets;
        _headVocabSizes = headVocabSizes;
    }

    /// <summary>The model file whose metadata and tensor table this descriptor was read from, named in every refusal.</summary>
    public string ModelPath { get; }

    /// <summary>The backing PLE tensor: rows x embedding_length_per_layer_input, packed as fixed-stride quantized rows.</summary>
    public LogicalTensor Tensor { get; }

    public int NgramSize { get; }

    public int HeadsPerNgram { get; }

    /// <summary><c>(ngram_size - 1) * heads_per_ngram</c>: the number of gather rows one token position produces.</summary>
    public int HeadCount { get; }

    /// <summary>The backing tensor's row count — every head range must fit inside this row space.</summary>
    public long TableRows => Tensor.Shape.Rows;

    /// <summary><c>&lt;arch&gt;.embedding_length_per_layer_input</c>, the element width of one table row.</summary>
    public int RowElements => Tensor.Shape.Dims[^1];

    /// <summary>The token id a discontiguous or missing predecessor position reads as, and the id that cuts everything at or before it in a history window.</summary>
    public int EosTokenId { get; }

    /// <summary>Packed row size in bytes for this table's quant format: the gather resolves one row by multiplying a row index by this stride.</summary>
    public long RowStrideBytes => Tensor.Shape.RowStrideBytes(Tensor.Format);

    /// <summary>The head range one head gathers within: its base row offset and its own vocabulary modulus, exactly as metadata declares them.</summary>
    public (long Offset, long VocabSize) HeadRange(int headIndex)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(headIndex, 0);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(headIndex, HeadCount - 1);

        return (_headOffsets[headIndex], _headVocabSizes[headIndex]);
    }

    /// <summary>
    /// Reads the PLE n-gram geometry of <paramref name="model"/> for <paramref name="tensor"/>, validating every declaration against
    /// each other and against the tensor's row space: known scalar fields present and in range, head count agreeing with both array
    /// declarations, a modulus-bearing vocabulary size per head, head ranges landing inside the table, multipliers covering every
    /// n-gram order without collapsing one to zero, the declared embedding width matching the tensor's row width, and the payload
    /// packing as fixed-stride rows. A declaration this seam cannot read throws <see cref="PleTableException"/> naming the model file,
    /// the tensor, and the field at fault — there is no fallback posture here, because reading a table whole is exactly what this seam exists to avoid.
    /// </summary>
    public static PleNgramTable FromGguf(GgufFile model, LogicalTensor tensor)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tensor);

        string architecture = model.Metadata.GetString("general.architecture", "");
        if (architecture.Length == 0)
        {
            throw new PleTableException(
                $"model '{model.Path}': tensor '{tensor.Name}' is a PLE table but the file declares no 'general.architecture', " +
                $"so its n-gram metadata keys have no prefix to be read under");
        }

        string prefix = architecture + ".";
        string ngramKey = prefix + "ple.ngram_size";
        string headsPerNgramKey = prefix + "ple.heads_per_ngram";
        string multipliersKey = prefix + "ple.layer_multipliers";
        string offsetsKey = prefix + "ple.head_offsets";
        string vocabSizesKey = prefix + "ple.head_vocab_sizes";
        string eosKey = prefix + "ple.eos_token_id";
        string embeddingKey = prefix + "embedding_length_per_layer_input";

        int ngramSize = RequirePositiveInt(model, tensor, ngramKey);
        if (ngramSize < 2)
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': '{ngramKey}' is {ngramSize}; a PLE n-gram table hashes at least a bigram, so the size must be 2 or greater");
        }

        int headsPerNgram = RequirePositiveInt(model, tensor, headsPerNgramKey);

        long headCountLong = checked((long)(ngramSize - 1)) * headsPerNgram;
        if (headCountLong > int.MaxValue)
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': '{ngramKey}' = {ngramSize} with '{headsPerNgramKey}' = {headsPerNgram} " +
                $"declares {headCountLong} heads, which exceeds the head count a gather can index");
        }

        int headCount = (int)headCountLong;

        ulong[] multipliers = ReadUnsignedArray(model, tensor, multipliersKey);
        if (multipliers.Length < ngramSize)
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': '{multipliersKey}' carries {multipliers.Length} entries but an " +
                $"'{ngramKey}' = {ngramSize} table multiplies every position of its longest n-gram — at least {ngramSize} are required");
        }

        for (int i = 0; i < ngramSize; i++)
        {
            if (multipliers[i] == 0)
            {
                throw new PleTableException(
                    $"model '{model.Path}', tensor '{tensor.Name}': '{multipliersKey}'[{i}] is 0, which collapses every token's contribution " +
                    $"to position {i} of an n-gram mix into the same value");
            }
        }

        long[] headOffsets = ReadRowSpaceArray(model, tensor, offsetsKey);
        long[] headVocabSizes = ReadRowSpaceArray(model, tensor, vocabSizesKey);

        if (headOffsets.Length < headCount)
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': '{offsetsKey}' carries {headOffsets.Length} entries but " +
                $"'{ngramKey}' = {ngramSize} with '{headsPerNgramKey}' = {headsPerNgram} declares {headCount} heads");
        }

        if (headVocabSizes.Length < headCount)
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': '{vocabSizesKey}' carries {headVocabSizes.Length} entries but " +
                $"'{ngramKey}' = {ngramSize} with '{headsPerNgramKey}' = {headsPerNgram} declares {headCount} heads");
        }

        int[] dims = tensor.Shape.Dims;
        if (dims.Length != 2)
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': a row-indexed PLE gather reads a two-dimensional [rows, embedding width] table " +
                $"but this tensor declares dims [{string.Join(",", dims)}]");
        }

        int declaredEmbedding = RequirePositiveInt(model, tensor, embeddingKey);
        if (declaredEmbedding != dims[^1])
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': '{embeddingKey}' is {declaredEmbedding} but the tensor's rows are " +
                $"{dims[^1]} elements wide — the gather indexes whole packed rows of this tensor");
        }

        long rowStride = tensor.Shape.RowStrideBytes(tensor.Format);
        if (rowStride <= 0 || tensor.SizeBytes != checked((long)dims[0] * rowStride))
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': its {dims[0]} rows at a {tensor.Format} stride of {rowStride} bytes do not pack " +
                $"the declared {tensor.SizeBytes}-byte payload — PLE row gather needs fixed-stride rows and takes no whole-table copy as a fallback");
        }

        long tableRows = dims[0];
        for (int h = 0; h < headCount; h++)
        {
            if (headVocabSizes[h] == 0)
            {
                throw new PleTableException(
                    $"model '{model.Path}', tensor '{tensor.Name}': '{vocabSizesKey}'[{h}] is 0, and a head with no vocabulary has nothing to index modulo");
            }

            if (headOffsets[h] < 0 || headVocabSizes[h] < 0)
            {
                throw new PleTableException(
                    $"model '{model.Path}', tensor '{tensor.Name}': head {h} declares offset {headOffsets[h]} with vocabulary size {headVocabSizes[h]}, which is not a row range");
            }

            if (headOffsets[h] > long.MaxValue - headVocabSizes[h])
            {
                throw new PleTableException(
                    $"model '{model.Path}', tensor '{tensor.Name}': head {h} range [{headOffsets[h]}..{headOffsets[h]} + {headVocabSizes[h]}) overflows the row space");
            }

            long end = headOffsets[h] + headVocabSizes[h];
            if (end > tableRows)
            {
                throw new PleTableException(
                    $"model '{model.Path}', tensor '{tensor.Name}': head {h} range from '{offsetsKey}'[{h}] and '{vocabSizesKey}'[{h}] reaches row " +
                    $"{end - 1} but the table declares only {tableRows} rows");
            }
        }

        int eosTokenId = RequireNonNegativeInt(model, tensor, eosKey);

        return new PleNgramTable(
            model.Path,
            tensor,
            ngramSize,
            headsPerNgram,
            eosTokenId,
            multipliers,
            [.. headOffsets],
            [.. headVocabSizes]);
    }

    /// <summary>
    /// The order-<c>n</c> token mix of a PLE n-gram hash, exactly as llama.cpp's qwen4exp gather computes it: position 0 multiplied by its
    /// multiplier, every later position XOR-combined through its own multiplier, all in unsigned 64-bit arithmetic that wraps rather than
    /// traps. Token ids extend to <see cref="ulong"/> by two's complement (a negative id is the same bits an unchecked C++ cast of a
    /// signed token produces), so the mix never depends on sign interpretation. This is the raw hash: history normalization — missing or
    /// discontiguous predecessors reading as EOS, and an EOS cutting everything at or before it — belongs to <see cref="ComputeRowIndices"/>,
    /// which mirrors llama.cpp's cache posture.
    /// </summary>
    public static ulong MixNgram(ReadOnlySpan<int> tokens, IReadOnlyList<ulong> multipliers, int order)
    {
        ArgumentNullException.ThrowIfNull(multipliers);

        if (order < 2 || order > tokens.Length)
        {
            throw new ArgumentException($"an n-gram mix of order {order} needs at least {order} tokens", nameof(order));
        }

        if (multipliers.Count < order)
        {
            throw new ArgumentException(
                $"an n-gram mix of order {order} needs a multiplier per position but only {multipliers.Count} are declared", nameof(multipliers));
        }

        ulong mixed = unchecked((ulong)(long)tokens[0]) * multipliers[0]; // unsigned 64-bit wrap: the product is taken modulo 2^64 exactly as in C++
        for (int j = 1; j < order; j++)
        {
            mixed ^= unchecked((ulong)(long)tokens[j]) * multipliers[j];
        }

        return mixed;
    }

    /// <summary>The raw order-<c>n</c> mix of the leading positions of this table's own multiplier set.</summary>
    public ulong MixForOrder(ReadOnlySpan<int> ctxPrefix, int order) => MixNgram(ctxPrefix, _multipliers, order);

    /// <summary>
    /// The gather rows one token position reads: one row per head, written into <paramref name="headRows"/> in head order (the heads of
    /// n-gram order <c>n</c> starting at index <c>(n - 2) * HeadsPerNgram</c>). <paramref name="historyWindow"/> is the position's contiguous
    /// history newest-first: window[0] is the token itself, and window[s] is the predecessor s positions back. A predecessor beyond the
    /// window — before a sequence start, or where no cache cell holds it — reads as EOS exactly as llama.cpp's cache reports nothing; any
    /// negative id is that same missing-cell sentinel. An EOS at or before a position cuts everything at and behind it to EOS, while the
    /// token itself being EOS does not cut its own context, matching the reference gather. The head of order <c>n</c> indexes row
    /// <c>mixed % head_vocab_sizes[head] + head_offsets[head]</c> with <c>mixed</c> this table's unsigned n-gram mix.
    /// </summary>
    public void ComputeRowIndices(ReadOnlySpan<int> historyWindow, Span<long> headRows)
    {
        if (historyWindow.Length == 0)
        {
            throw new ArgumentException("a token position needs at least its own token id in the history window", nameof(historyWindow));
        }

        if (headRows.Length != HeadCount)
        {
            throw new ArgumentException(
                $"this table gathers {HeadCount} rows per token position but 'headRows' holds {headRows.Length}", nameof(headRows));
        }

        // The window is normalized newest-first exactly as llama.cpp builds it: the token itself stays whatever it is (an EOS token does not cut its own context), and once a missing or EOS predecessor cuts the history, every position at and behind that cut reads as EOS.
        if (NgramSize <= StackContextLimit)
        {
            FillRowIndices(stackalloc int[NgramSize], historyWindow, headRows); // the stack scratch is handed straight to the gather call and never leaves this frame
        }
        else
        {
            int[] rented = ArrayPool<int>.Shared.Rent(NgramSize);

            try
            {
                FillRowIndices(rented.AsSpan(0, NgramSize), historyWindow, headRows); // an n-gram wider than the stack limit gathers through pooled scratch rather than growing every frame
            }
            finally
            {
                ArrayPool<int>.Shared.Return(rented);
            }
        }
    }

    private void FillRowIndices(Span<int> ctx, ReadOnlySpan<int> historyWindow, Span<long> headRows)
    {
        ctx[0] = historyWindow[0];
        bool cut = false;

        for (int s = 1; s < NgramSize; s++)
        {
            // A predecessor past the window has no cache cell and reads as llama.cpp's null token (-1); once a cut is in force, everything behind it reports nothing rather than stale history.
            int token = !cut && s < historyWindow.Length ? historyWindow[s] : MissingTokenSentinel;

            if (!cut)
            {
                cut = token == MissingTokenSentinel || token == EosTokenId; // an EOS predecessor, or a position with no cached cell at all, cuts this n-gram window
            }

            ctx[s] = cut ? EosTokenId : token; // an un-cut predecessor is a real token id, and the current token's own value survives even when it is EOS
        }

        for (int n = 2; n <= NgramSize; n++)
        {
            ulong mixed = MixNgram(ctx.Slice(0, n), _multipliers, n);
            int baseHead = (n - 2) * HeadsPerNgram; // head order is by n-gram length: the heads of order n start at (n - 2) * heads_per_ngram

            for (int g = 0; g < HeadsPerNgram; g++)
            {
                int headIndex = baseHead + g;
                headRows[headIndex] = unchecked((long)(mixed % (ulong)_headVocabSizes[headIndex])) + _headOffsets[headIndex]; // the same mix indexes every head of this order, each modulo its own vocabulary size
            }
        }
    }

    /// <summary>The gather rows for one token position, as a fresh array in head order — the convenience form tests and benches read through.</summary>
    public long[] RowIndicesFor(ReadOnlySpan<int> historyWindow)
    {
        long[] rows = new long[HeadCount];
        ComputeRowIndices(historyWindow, rows);
        return rows;
    }

    private static int RequirePositiveInt(GgufFile model, LogicalTensor tensor, string key)
    {
        int value = RequireNonNegativeInt(model, tensor, key);
        if (value == 0)
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': '{key}' is 0; it must be greater than zero to describe a PLE n-gram table");
        }

        return value;
    }

    private static int RequireNonNegativeInt(GgufFile model, LogicalTensor tensor, string key)
    {
        if (!model.Metadata.Has(key))
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': metadata declares no '{key}' — a PLE n-gram table must declare its geometry before rows can be gathered from the file");
        }

        long value;
        try
        {
            value = model.Metadata.Get(key).AsLong();
        }
        catch (GgufException ex)
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': '{key}' is not an integer this seam can read: {ex.Message}");
        }

        if (value < 0 || value > int.MaxValue)
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': '{key}' is {value}, which does not fit the positive range this seam reads it in");
        }

        return (int)value;
    }

    private static ulong[] ReadUnsignedArray(GgufFile model, LogicalTensor tensor, string key)
    {
        if (!model.Metadata.Has(key))
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': metadata declares no '{key}' — a PLE n-gram table must declare its per-head ranges and multipliers before rows can be gathered from the file");
        }

        GgufValue value = model.Metadata.Get(key);
        if (value.Type != GgufValueType.Array)
        {
            throw new PleTableException(
                $"model '{model.Path}', tensor '{tensor.Name}': '{key}' is a {value.Type}, not the array of unsigned values this seam reads");
        }

        IReadOnlyList<GgufValue> elements = value.AsArray();
        ulong[] entries = new ulong[elements.Count]; // llama.cpp stores these arrays at 64-bit width: read them there rather than narrowing first.
        for (int i = 0; i < elements.Count; i++)
        {
            try
            {
                entries[i] = elements[i].AsULong();
            }
            catch (GgufException ex)
            {
                throw new PleTableException(
                    $"model '{model.Path}', tensor '{tensor.Name}': '{key}'[{i}] is not an unsigned value this seam can read: {ex.Message}");
            }
        }

        return entries;
    }

    /// <summary>A per-head array read at GGUF's 64-bit width and narrowed to signed row arithmetic: an entry past the row space an Int64 index can address is a broken declaration, not something this seam clamps.</summary>
    private static long[] ReadRowSpaceArray(GgufFile model, LogicalTensor tensor, string key)
    {
        ulong[] entries = ReadUnsignedArray(model, tensor, key); // head offsets and vocabulary sizes are declared unsigned; row arithmetic takes them as Int64 row positions.
        long[] narrowed = new long[entries.Length];

        for (int i = 0; i < entries.Length; i++)
        {
            if (entries[i] > long.MaxValue)
            {
                throw new PleTableException(
                    $"model '{model.Path}', tensor '{tensor.Name}': '{key}'[{i}] is {entries[i]}, past the row space an Int64 row index can address");
            }

            narrowed[i] = unchecked((long)entries[i]);
        }

        return narrowed;
    }
}

