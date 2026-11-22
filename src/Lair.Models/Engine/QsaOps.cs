using Lair.Contracts;

namespace Lair.Models.Engine;

/// <summary>
/// CPU semantics of one query-sparse-attention block at layer level for a single query token against an in-memory synthetic cache.
/// The reference graph scores each pool with the sum over indexer heads of rectified dot products weighted by 1 / sqrt(indexer
/// dimension) — the lightning-indexer scoring with every head weight preset to that scale; one pool is a block of compress-ratio
/// cache cells in sequence order whose pooled key is the mean of its members' raw indexer keys, normalized and then rotated to its
/// first member's position, so pooling precedes norm and rotation exactly as stored. Selection keeps the highest-scoring pools up to
/// the declared cell budget divided by the ratio — ties resolve to the lower block index here while the reference leaves equal scores
/// unordered — plus the always-visible window covering the block that holds this query's own cache cell, whether that block is a short
/// trailing partial pool or a complete one whose score ranked it out; attention then runs dense grouped-query causal softmax restricted
/// to those cells in ascending order through exactly the pinned kernel's online-max update sequence and dot seam, so a selection
/// covering the whole causal prefix lands bit-for-bit on it. Causality itself lives in pool visibility upstream of a served model; these
/// layer-level fixtures hold only cells at or behind the query position, where both readings agree. Indexer dots accumulate in double;
/// attention dots run through the backend seam unchanged from the dense path. A block whose ratio groups fewer than two cells runs dense and never reaches selection.
/// </summary>
internal static class QsaOps
{
    /// <summary>Dense operands one sparse-attention block reads for a layer-level fixture: projection rows list their input-column
    /// weights contiguously, following the reference view layouts — query-and-gate rows alternate per head (query slice then gate
    /// slice), key and value rows are grouped-query head regions, indexer queries carry one slice per indexer head.</summary>
    internal struct QsaBlockTensors
    {
        public QsaBlockTensors()
        {
        }

        public int HiddenSize;

        public int HeadCount; // query heads of the block's own dense attention
        public int KvHeadCount; // grouped-query key/value head count
        public int HeadDim; // shared q/k/v head dimension, rotated full-head in these fixtures

        /// <summary>Indexer scoring heads and their shared component width.</summary>
        public int IndexerHeads;
        public int IndexerDim;

        /// <summary>Declared selection budget in cells: the reference caps pools at top-k / ratio blocks per token.</summary>
        public int TopKCells;

        /// <summary>Cache cells one pool covers. A value below 2 means this layer runs dense — no indexer selection.</summary>
        public int CompressRatio;

        public float[] QueryGate = []; // 2 * HeadCount * HeadDim rows x HiddenSize: every head's query slice then its gate slice, alternating per head
        public float[] Key = []; // KvHeadCount * HeadDim rows x HiddenSize
        public float[] Value = []; // same shape as Key

        /// <summary>HiddenSize rows x (HeadCount * HeadDim): the block output projection over concatenated attended heads.</summary>
        public float[] Output = [];

        public float[] IndexerQueryProj = []; // IndexerHeads * IndexerDim rows x HiddenSize
        public float[] IndexerKeyProj = []; // IndexerDim rows x HiddenSize

        /// <summary>Norm weight vectors applied to indexer queries, pooled keys, and the block's own q/k heads.</summary>
        public float[] IndexerQueryNorm = []; // IndexerDim
        public float[] IndexerKeyNorm = []; // IndexerDim
        public float[] QueryHeadNorm = []; // HeadDim shared by every query head
        public float[] KeyHeadNorm = []; // HeadDim shared by every key head

        /// <summary>Complete pool blocks this selection budget can keep per token.</summary>
        public readonly int PoolBudget => TopKCells / Math.Max(1, CompressRatio);
    }

    /// <summary>
    /// Lightning-indexer scores for one query against pooled keys: score[p] = sum over indexer heads of max(q_head · k_p, 0) * (1 /
    /// sqrt(IndexerDim)), each head's dot accumulating in double. The reference pre-fills every head weight with exactly this scale
    /// before the rectified sum; pools a query cannot see are excluded by whoever feeds these scores to selection.
    /// </summary>
    internal static void IndexerScores(ReadOnlySpan<float> q, ReadOnlySpan<float> pooledKeys, int indexerHeads, int indexerDim, int poolCount, Span<float> outScores)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(indexerHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(indexerDim);

        if (q.Length < indexerHeads * indexerDim || pooledKeys.Length < poolCount * indexerDim || outScores.Length < poolCount)
        {
            throw new ArgumentException("Indexer scoring needs every head region of the query and one key region per pool.", nameof(q));
        }

        float scale = 1f / MathF.Sqrt((float)indexerDim);

        for (int p = 0; p < poolCount; p++)
        {
            ReadOnlySpan<float> kp = pooledKeys.Slice(p * indexerDim, indexerDim);
            double score = 0.0;

            for (int h = 0; h < indexerHeads; h++)
            {
                ReadOnlySpan<float> qh = q.Slice(h * indexerDim, indexerDim);

                double dot = 0.0; // rectified head contribution: the reference weights every head by exactly this scale after the relu
                for (int i = 0; i < indexerDim; i++)
                {
                    dot += (double)qh[i] * kp[i];
                }

                score += Math.Max(dot, 0.0);
            }

            outScores[p] = (float)(score * scale);
        }
    }

    /// <summary>
    /// Pool each complete block of ratio consecutive cache cells: pooledKey[p][i] = mean over the block's raw indexer-key rows at position i, then RMS-normalize with this weight vector and epsilon exactly as <see cref="LayerOps.RmsNormWeighted"/> does — rotation to the block's first member position happens in a separate step because pooling precedes norm and rotation. A short trailing block gets no pooled entry: its cells stay raw members under selection instead.
    /// </summary>
    internal static void PoolKeys(ReadOnlySpan<float> rawIndexerKeys, int cacheCells, QsaBlockTensors w, float normEps, Span<float> outPooled)
    {
        if (w.CompressRatio < 2)
        {
            throw new ArgumentException("Pooling groups at least two cells per pool.", nameof(w));
        }

        int fullPools = cacheCells / w.CompressRatio;
        if (rawIndexerKeys.Length < cacheCells * w.IndexerDim || outPooled.Length < fullPools * w.IndexerDim)
        {
            throw new ArgumentException("Pooling needs one raw indexer-key row per cached cell and room for every complete pool.", nameof(rawIndexerKeys));
        }

        float inv = 1f / w.CompressRatio;
        Span<float> mean = stackalloc float[w.IndexerDim]; // pooling scratch stays on the stack: IndexerDim is a small fixed head width per layer

        for (int p = 0; p < fullPools; p++)
        {
            mean.Clear();
            int baseRow = p * w.CompressRatio;

            for (int m = 0; m < w.CompressRatio; m++)
            {
                ReadOnlySpan<float> row = rawIndexerKeys.Slice((baseRow + m) * w.IndexerDim, w.IndexerDim);
                for (int i = 0; i < w.IndexerDim; i++)
                {
                    mean[i] += row[i];
                }
            }

            for (int i = 0; i < w.IndexerDim; i++)
            {
                mean[i] *= inv; // the mean of member keys is what the reference normalizes and rotates next
            }

            LayerOps.RmsNormWeighted(mean, w.IndexerKeyNorm, normEps, outPooled.Slice(p * w.IndexerDim, w.IndexerDim));
        }
    }

    /// <summary>
    /// Select the cells one query attends to: complete pools are floor(cacheCells / ratio) blocks of consecutive cache positions in sequence order scored by scores[p]. The block holding this query's own cell is always visible — a short trailing partial pool when one exists, otherwise the final complete pool itself whatever its score. Remaining budget goes to the highest-scoring other pools (PoolBudget minus any forced complete pool), equal scores resolving to the lower block index here because a fixture must pin what it asserts against while the reference leaves ties unordered. The result is ascending cell indices; an empty cache yields nothing and a cache shorter than one complete pool yields exactly its cells in order.
    /// </summary>
    internal static List<int> SelectCells(ReadOnlySpan<float> scores, int cacheCells, QsaBlockTensors w)
    {
        if (w.CompressRatio < 2)
        {
            throw new ArgumentException("Selection runs only on a layer whose compress ratio groups at least two cells per pool.", nameof(w));
        }

        List<int> cells = []; // no cached cell means nothing to attend yet; every fixture query carries its own position already written into the cache it scores against
        if (cacheCells <= 0)
        {
            return cells;
        }

        int fullPools = cacheCells / w.CompressRatio;
        if (scores.Length < fullPools)
        {
            throw new ArgumentException("Selection needs one score per complete pool.", nameof(scores));
        }

        HashSet<int> selected = [];

        // The block containing the newest cached cell is visible whatever its score: start at that block and keep every remaining cell, covering a partial tail or a final complete pool alike.
        int forcedStart = ((cacheCells - 1) / w.CompressRatio) * w.CompressRatio;
        bool forcedIsCompletePool = fullPools > 0 && (cacheCells - 1) / w.CompressRatio == fullPools - 1;

        for (int cell = forcedStart; cell < cacheCells; cell++)
        {
            selected.Add(cell); // this window is what guarantees the query can attend its own position even when ranking drops its block
        }

        List<int> eligible = [];
        for (int p = 0; p < fullPools; p++)
        {
            if (!forcedIsCompletePool || p != fullPools - 1) // a forced complete pool already counts against the budget and is not ranked again here
            {
                eligible.Add(p);
            }
        }

        // Selection-sort eligible pools by descending score and ascending index: a ref-like scores span cannot cross into a sort lambda, so the ranking stays in this method body.
        for (int start = 0; start < eligible.Count; start++)
        {
            int best = start;

            for (int probe = start + 1; probe < eligible.Count; probe++)
            {
                int byScore = scores[eligible[probe]].CompareTo(scores[eligible[best]]); // descending score first
                if (byScore > 0 || (byScore == 0 && eligible[probe] < eligible[best])) // ties resolve to the lower block index deterministically
                {
                    best = probe;
                }
            }

            int swap = eligible[start];
            eligible[start] = eligible[best];
            eligible[best] = swap;
        }

        int budgetLeft = Math.Max(0, w.PoolBudget - (forcedIsCompletePool ? 1 : 0));
        for (int kept = 0; kept < Math.Min(eligible.Count, budgetLeft); kept++)
        {
            int pool = eligible[kept];
            for (int m = 0; m < w.CompressRatio; m++)
            {
                selected.Add(pool * w.CompressRatio + m); // a chosen pool contributes every member cell
            }
        }

        cells.AddRange(selected);
        cells.Sort(); // attention walks ascending positions exactly as the causal kernel does

        return cells;
    }

    /// <summary>
    /// Dense grouped-query softmax restricted to an explicit ascending cell list: head h reads kv-head (h * kvHeadCount / headCount) and each selected position scores through the backend dot seam times attnScale, then runs the pinned kernel's online-softmax update sequence in exactly its order — running max, rescale-on-new-max over both accumulator and weight sum, exp(score - max), weighted value accumulation in element order, final 1f / sumExp normalization. With a selection covering every position up to the query's own, this lands bit-for-bit on <see cref="LayerOps.CausalAttention"/> for the same operands; dropping cells is what makes it sparse: an excluded cell never touches the max or the accumulator at all. Key and value caches are token-major rows of kvHeadCount * headDim values, exactly as a per-layer KV slot stores them.
    /// </summary>
    internal static void RestrictedAttention(
        IBackend backend,
        ReadOnlySpan<float> qHeads, // HeadCount * HeadDim rotated query heads of one token
        ReadOnlySpan<float> keyCache, // cacheCells rows x (KvHeadCount * HeadDim)
        ReadOnlySpan<float> valueCache, // same shape as keyCache
        IReadOnlyList<int> selectedCells, // ascending cell indices this token attends to; must be non-empty and inside the cache
        int cacheCells,
        QsaBlockTensors w,
        float attnScale,
        Span<float> outConcat) // HeadCount * HeadDim head-major result for the output projection
    {
        ArgumentNullException.ThrowIfNull(backend);

        if (selectedCells.Count == 0)
        {
            throw new ArgumentException("Restricted attention needs at least one visible cell.", nameof(selectedCells));
        }

        int region = w.HeadCount * w.HeadDim;
        if (qHeads.Length < region || outConcat.Length < region || keyCache.Length < cacheCells * w.KvHeadCount * w.HeadDim)
        {
            throw new ArgumentException("Attention operands must cover every head's full region and the whole cache.", nameof(qHeads));
        }

        int kvRegion = w.KvHeadCount * w.HeadDim;

        for (int h = 0; h < w.HeadCount; h++)
        {
            ReadOnlySpan<float> qh = qHeads.Slice(h * w.HeadDim, w.HeadDim);
            int kvHead = h * w.KvHeadCount / w.HeadCount; // grouped-query mapping: consecutive heads share a kv-head, exactly as the dense kernel maps them
            Span<float> outHead = outConcat.Slice(h * w.HeadDim, w.HeadDim);

            float runningMax = float.NegativeInfinity;
            float sumExp = 0f;
            outHead.Clear();

            foreach (int cell in selectedCells)
            {
                if (cell < 0 || cell >= cacheCells)
                {
                    throw new ArgumentException($"Selected cell {cell} lies outside the cache of {cacheCells} positions.", nameof(selectedCells));
                }

                ReadOnlySpan<float> kt = keyCache.Slice(cell * kvRegion + kvHead * w.HeadDim, w.HeadDim);
                float score = backend.Dot(qh, kt) * attnScale; // same scaled-score path as the pinned kernel before its online softmax sees it

                if (score > runningMax)
                {
                    // Rescale everything accumulated under the old max so all weights share the new exponent base.
                    if (!float.IsNegativeInfinity(runningMax))
                    {
                        float rescale = MathF.Exp(runningMax - score);
                        sumExp *= rescale;

                        for (int j = 0; j < w.HeadDim; j++)
                        {
                            outHead[j] *= rescale;
                        }
                    }

                    runningMax = score;
                }

                float weight = MathF.Exp(score - runningMax); // the current max term contributes exactly 1
                sumExp += weight;

                ReadOnlySpan<float> vt = valueCache.Slice(cell * kvRegion + kvHead * w.HeadDim, w.HeadDim);
                for (int j = 0; j < w.HeadDim; j++)
                {
                    outHead[j] += weight * vt[j]; // weighted values in element order, the same accumulation order the dense kernel uses
                }
            }

            float invSum = 1f / sumExp; // >= 1 because the max term contributes exp(0)
            for (int j = 0; j < w.HeadDim; j++)
            {
                outHead[j] *= invSum;
            }
        }
    }

    /// <summary>Apply RoPE to one head region at a position from tables built by <see cref="LayerOps.BuildRopeTables"/>: adjacent-pair rotation over the whole head, matching what these fixtures carry for indexer queries and pooled keys.</summary>
    internal static void RotateAtPosition(Span<float> headRegion, int ropeHalf, ReadOnlySpan<float> cosTable, ReadOnlySpan<float> sinTable, int position) => LayerOps.ApplyRopeToHead(headRegion, ropeHalf, cosTable.Slice(position * ropeHalf, ropeHalf), sinTable.Slice(position * ropeHalf, ropeHalf));

    /// <summary>RMS-normalize one indexer query head region or pooled key with its own weight vector and epsilon (the checkpoint's norm epsilon at the call site).</summary>
    internal static void NormHead(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, float eps, Span<float> dst) => LayerOps.RmsNormWeighted(src, weight, eps, dst);

    /// <summary>Dense dot over one projection row against this token's input: the fixture-side stand-in for a packed GEMV.</summary>
    internal static float ProjectRow(float[] matrix, int row, ReadOnlySpan<float> vector)
    {
        double sum = 0.0;
        for (int i = 0; i < vector.Length; i++)
        {
            sum += (double)matrix[row * vector.Length + i] * vector[i];
        }

        return (float)sum;
    }
}
