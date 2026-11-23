using Lair.Contracts;

namespace Lair.Models.Engine;

/// <summary>
/// CPU reference semantics of one gated-delta-net (linear attention) block, isolated so each stage can be verified against
/// hand-derived math independently of any model file. The stages mirror the reference graph exactly: a fused qkv/gate
/// projection whose channels are every key head's slice, then their duplicate region for k, then value heads; beta and alpha
/// read one merged projection whose first rows are beta and remaining rows are alpha; beta passes through sigmoid; the decay
/// gate is each head's stored negative log-decay weight multiplied by softplus(alpha + dt-bias); a causal depthwise convolution
/// over the concatenated conv history and this chunk's raw qkv rows — tap j of channel c reads input row (t - kernel + 1) + j, so
/// its last tap multiplies the current token itself — followed by SiLU; each convolved q and k head is normalized as value *
/// rsqrt(sum(value^2) + eps), which is exactly rms-with-eps-over-n scaled by 1/sqrt(n); the per-token delta rule runs over a
/// recurrent state held in the reference's transposed slot layout, where entry j * S + i of head h holds S[i][j] so dotting one
/// row against x yields (S^T @ x)[j]: scale every stored value by exp(gate), form delta[j] = (v[j] - stateRow_j · k) * beta, fold
/// the outer product k ⊗ delta back into the state, and read each token's output as (S_new^T @ q) scaled by 1/sqrt(stateDim); the
/// head outputs are normalized per value head with RMS over its own slice times that head's norm vector, gated element-wise by
/// sigmoid of this token's gate rows, concatenated in ascending head order and projected through the output matrix. Dots
/// accumulate in double so a run is deterministic on any single machine; every buffer is caller-owned and nothing here allocates
/// per step. The recurrent footprint one sequence carries through a block — valueHeads * stateDim * stateDim slot floats of
/// delta-rule state plus (convKernel - 1) * convChannels raw qkv history floats — is exactly what a layer-level fixture budgets
/// against. Single-sequence only: the batched engine wiring reuses these kernels behind its own sequence-major buffers, and
/// nothing here reads a model file.
/// </summary>
internal static class GdnOps
{
    /// <summary>Sigmoid used by the beta selection and the output gate.</summary>
    internal static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>Natural-log softplus with an overflow guard: values above 30 pass through unchanged.</summary>
    internal static float Softplus(float x) => x > 30f ? x : (float)Math.Log(1.0 + Math.Exp((double)x)); // natural-log softplus with an overflow guard: values above 30 pass through unchanged

    /// <summary>
    /// GDN q/k normalization applied to each head region independently: dst[i] = src[i] / sqrt(sum(src_head^2) + eps), with the
    /// sum accumulating in double. This is rms-with-eps-over-n scaled by 1/sqrt(n) — dividing by n inside and multiplying by
    /// 1/sqrt(n) outside leave one division of the whole-head sum under the root — not a mean-based RMSNorm.
    /// </summary>
    internal static void L2NormHeads(ReadOnlySpan<float> src, int headDim, float eps, Span<float> dst)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

        if (src.Length % headDim != 0 || dst.Length < src.Length)
        {
            throw new ArgumentException("L2-norm heads needs operands covering whole head regions.", nameof(src));
        }

        for (int region = 0; region < src.Length; region += headDim)
        {
            double sumSq = 0.0;
            for (int i = 0; i < headDim; i++)
            {
                sumSq += (double)src[region + i] * src[region + i];
            }

            float invNorm = (float)(1.0 / Math.Sqrt(sumSq + eps));
            for (int i = 0; i < headDim; i++)
            {
                dst[region + i] = src[region + i] * invNorm;
            }
        }
    }

    /// <summary>
    /// Causal depthwise convolution over one channel run: the virtual input is the stored conv history (oldest first, exactly
    /// convKernel - 1 rows) followed by this chunk's raw token rows, and out[t][c] = sum_j kernel[c * convKernel + j] * in[(t -
    /// convKernel + 1 + j) * channels + c], so tap convKernel - 1 multiplies the current token. The result carries SiLU applied on
    /// top of the raw dot, which accumulates in double. newHistory — a buffer distinct from history and tokens at every call site
    /// here — receives the last convKernel - 1 rows of that virtual concatenation: raw values, not convolution outputs, which is
    /// what the reference stores back into this convolution's recurrent row for the next chunk; a chunk shorter than the kernel
    /// leaves part of the older history shifted forward.
    /// </summary>
    internal static void DepthwiseCausalConvSilu(
        ReadOnlySpan<float> history, // (convKernel - 1) * channels, oldest first; empty when convKernel == 1
        ReadOnlySpan<float> tokens, // tokenCount * channels
        int channels,
        int tokenCount,
        int convKernel,
        ReadOnlySpan<float> kernel, // channels * convKernel rows, taps contiguous per channel
        Span<float> outTokens, // tokenCount * channels
        Span<float> newHistory) // (convKernel - 1) * channels; must not overlap history or tokens
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);

        if (convKernel < 1)
        {
            throw new ArgumentException("A causal depthwise convolution needs at least one tap.", nameof(convKernel));
        }

        int historyRows = convKernel - 1;
        if (history.Length != historyRows * channels || tokens.Length < tokenCount * channels || outTokens.Length < tokenCount * channels)
        {
            throw new ArgumentException("Convolution operands must cover the full history, every chunk row, and every output.", nameof(history));
        }

        if (newHistory.Length < historyRows * channels)
        {
            throw new ArgumentException("The conv state update needs room for one kernel's worth of raw rows.", nameof(newHistory));
        }

        // in[t - historyRows + j] walks the virtual concatenation [history; tokens] without materializing it.
        for (int t = 0; t < tokenCount; t++)
        {
            for (int c = 0; c < channels; c++)
            {
                double sum = 0.0;
                for (int j = 0; j < convKernel; j++)
                {
                    int row = t - historyRows + j; // negative indexes the stored history oldest-first, non-negative this chunk's rows
                    float value = row < 0 ? history[(row + historyRows) * channels + c] : tokens[row * channels + c];
                    sum += (double)kernel[c * convKernel + j] * value;
                }

                outTokens[t * channels + c] = LayerOps.Silu((float)sum); // SiLU after the raw convolution, as the reference applies it
            }
        }

        for (int r = 0; r < historyRows; r++)
        {
            int virtualRow = tokenCount + r; // tail index into [history; tokens]: the last kernel - 1 rows of that concatenation
            if (virtualRow < historyRows)
            {
                history.Slice(virtualRow * channels, channels).CopyTo(newHistory.Slice(r * channels, channels));
            }
            else
            {
                tokens.Slice((virtualRow - historyRows) * channels, channels).CopyTo(newHistory.Slice(r * channels, channels));
            }
        }
    }

    /// <summary>
    /// One chunk of the per-token delta rule for every value head of one sequence. State holds each head's slot array in the
    /// reference transposed layout — entry h * stateDim * stateDim + j * stateDim + i carries S[i][j], so dotting row j against x
    /// yields (S^T @ x)[j] — and is updated in place. Per token: scale every stored value by exp(gate) with the gate read as a log
    /// decay, form delta[j] = (v[j] - stateRow_j · k) * beta, fold the outer product k ⊗ delta into the state, then write out[token
    /// * valueDim + h * stateDim + j] = updatedStateRow_j · q * (1 / sqrt(stateDim)). Value head h reads its key/q heads at index h
    /// % keyHeads. Each dot accumulates in double; outputs land in float once per read.
    /// </summary>
    internal static void GatedDeltaNetChunk(
        ReadOnlySpan<float> q, // tokenCount * (keyHeads * stateDim), normalized head-major rows
        ReadOnlySpan<float> k, // same shape as q
        ReadOnlySpan<float> v, // tokenCount * valueDim
        ReadOnlySpan<float> gate, // tokenCount * valueHeads: log decay per value head
        ReadOnlySpan<float> beta, // tokenCount * valueHeads: already sigmoided
        int keyHeads,
        int valueHeads,
        int stateDim,
        int tokenCount,
        Span<float> state, // valueHeads * stateDim * stateDim, updated in place
        Span<float> outValues) // tokenCount * valueDim
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(keyHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(valueHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stateDim);

        if (valueHeads % keyHeads != 0)
        {
            throw new ArgumentException("Every value head must map onto a whole number of key heads.", nameof(keyHeads));
        }

        int qWidth = keyHeads * stateDim;
        int vWidth = valueHeads * stateDim;
        if (q.Length < tokenCount * qWidth || k.Length < q.Length || v.Length < tokenCount * vWidth)
        {
            throw new ArgumentException("Delta-rule operands must cover every head region of every token.", nameof(q));
        }

        if (gate.Length < tokenCount * valueHeads || beta.Length < tokenCount * valueHeads)
        {
            throw new ArgumentException("Gate and beta vectors need one entry per value head per token.", nameof(gate));
        }

        if (state.Length < valueHeads * stateDim * stateDim || outValues.Length < tokenCount * vWidth)
        {
            throw new ArgumentException("The delta rule needs a full slot array per head and an output row per token.", nameof(state));
        }

        float scale = 1f / MathF.Sqrt((float)stateDim);

        for (int h = 0; h < valueHeads; h++)
        {
            int mappedKeyHead = h % keyHeads; // grouped-value mapping: the reference reads q/k head index modulo the key-head count
            Span<float> headState = state.Slice(h * stateDim * stateDim, stateDim * stateDim);

            for (int t = 0; t < tokenCount; t++)
            {
                ReadOnlySpan<float> qHead = q.Slice(t * qWidth + mappedKeyHead * stateDim, stateDim);
                ReadOnlySpan<float> kHead = k.Slice(t * qWidth + mappedKeyHead * stateDim, stateDim);
                ReadOnlySpan<float> vRow = v.Slice(t * vWidth + h * stateDim, stateDim);

                float decay = MathF.Exp(gate[t * valueHeads + h]); // the stored gate is a negative log decay: exp of it is the per-token multiplier
                float betaValue = beta[t * valueHeads + h];

                for (int j = 0; j < stateDim; j++)
                {
                    Span<float> row = headState.Slice(j * stateDim, stateDim);

                    double scaledSum = 0.0; // the decayed state read against k: sum_i exp(gate) * S[i][j] * k[i], with the scale folded into each stored entry first
                    for (int i = 0; i < stateDim; i++)
                    {
                        row[i] *= decay;
                        scaledSum += (double)row[i] * kHead[i];
                    }

                    float delta = ((float)((double)vRow[j] - scaledSum)) * betaValue; // the correction this token writes into column j of the state

                    for (int i = 0; i < stateDim; i++)
                    {
                        row[i] += kHead[i] * delta; // fold k ⊗ delta into the slot array
                    }
                }

                Span<float> outRow = outValues.Slice(t * vWidth + h * stateDim, stateDim);
                for (int j = 0; j < stateDim; j++)
                {
                    ReadOnlySpan<float> row = headState.Slice(j * stateDim, stateDim); // the updated slot array reads this token's output

                    double sum = 0.0;
                    for (int i = 0; i < stateDim; i++)
                    {
                        sum += (double)row[i] * qHead[i];
                    }

                    outRow[j] = (float)sum * scale;
                }
            }
        }
    }

    /// <summary>
    /// RMS-normalize each head's output slice with its own weight vector and gate the result element-wise by this token's sigmoid
    /// of the matching raw gate rows: dst[j] = src_head[j] / sqrt(mean(src_head^2) + eps) * normWeight[j] * sigmoid(gateHead[j]).
    /// The mean accumulates in double exactly as <see cref="LayerOps.RmsNormWeighted"/> does.
    /// </summary>
    internal static void NormGatedHeads(
        ReadOnlySpan<float> src, // headCount * headDim head-major rows of one token
        ReadOnlySpan<float> normWeight, // headDim shared by every head
        ReadOnlySpan<float> gate, // same length as src: this token's raw gate rows, sigmoid applied here
        float eps,
        Span<float> dst)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(normWeight.Length);

        if (src.Length % normWeight.Length != 0 || gate.Length < src.Length || dst.Length < src.Length)
        {
            throw new ArgumentException("Gated normalization needs a weight vector covering one head and operands covering whole heads.", nameof(src));
        }

        int headDim = normWeight.Length;
        for (int region = 0; region < src.Length; region += headDim)
        {
            double sumSq = 0.0;
            for (int i = 0; i < headDim; i++)
            {
                sumSq += (double)src[region + i] * src[region + i];
            }

            float invRms = (float)(1.0 / Math.Sqrt(sumSq / headDim + eps));
            for (int i = 0; i < headDim; i++)
            {
                dst[region + i] = src[region + i] * invRms * normWeight[i] * Sigmoid(gate[region + i]);
            }
        }
    }

    /// <summary>Row-major dense operands one gated-delta-net block reads, in the orientation a GGUF tensor carries: every output
    /// row lists its input-column weights contiguously. Row order follows what the reference graph views out of each projection —
    /// qkv rows are every key head's slice for q, their duplicate region for k, then value-head slices for v; gate rows carry one
    /// StateDim-wide slice per value head; the merged beta/alpha matrix puts beta's ValueHeads rows first and alpha's after them.
    /// A synthetic fixture fills these fields directly.</summary>
    internal struct GdnBlockTensors
    {
        public GdnBlockTensors()
        {
        }

        public int HiddenSize;

        /// <summary>Key heads of the linear attention (its group count); each maps onto one convolved q/k head region.</summary>
        public int KeyHeads;

        /// <summary>Value heads — the recurrent time-step rank: one delta-rule state matrix per value head.</summary>
        public int ValueHeads;

        /// <summary>Shared q/k/v head dimension of the recurrent state, identical on both sides as the reference asserts.</summary>
        public int StateDim;

        public int ConvKernel;

        public float[] Qkv = []; // (2 * KeyHeads + ValueHeads) * StateDim rows x HiddenSize — key heads for q, their duplicate region for k, then value-head slices for v
        public float[] Gate = []; // ValueHeads * StateDim rows x HiddenSize — the z gate this block reads from its input

        /// <summary>2 * ValueHeads rows x HiddenSize: beta's rows first, alpha's after them.</summary>
        public float[] BetaAlphaCat = [];

        public float[] SsmA = []; // ValueHeads: stored negative log-decay weights (already negated by conversion)
        public float[] DtBias = []; // ValueHeads added to each head's raw alpha before softplus

        /// <summary>One depthwise row per qkv channel, this channel's taps contiguous — the convolution kernel.</summary>
        public float[] Conv1d = [];

        public float[] NormWeight = []; // StateDim applied per value-head output slice
        public float[] Output = []; // HiddenSize rows x (ValueHeads * StateDim)

        /// <summary>Conv channels the fused projection and its convolution run over.</summary>
        public readonly int ConvChannels => 2 * KeyHeads * StateDim + ValueHeads * StateDim;

        /// <summary>Concatenated value-head width every delta-rule token output carries before this block's own output projection.</summary>
        public readonly int ValueDim => ValueHeads * StateDim;

        /// <summary>Floating-point slots one sequence holds as per-block recurrent state: each value head's full slot matrix.</summary>
        public readonly long RecurrentStateValues => (long)ValueHeads * StateDim * StateDim;

        /// <summary>Floating-point slots one sequence holds as this block's raw qkv convolution history.</summary>
        public readonly long ConvHistoryValues => (long)(ConvKernel - 1) * ConvChannels;
    }

    /// <summary>Weight views one gated-delta-net block projects through a compute backend's GEMV kernels instead of
    /// dequantized copies: every projection role is the GGUF tensor itself, resolved to packed bytes at each pass and
    /// dotted through <see cref="IBackend.Gemv"/> against a <see cref="WeightRef"/> over the source's own span. The
    /// beta and alpha sides stay separate tensors — the merged beta/alpha matrix is a fixture-side convenience of
    /// <see cref="GdnBlockTensors"/>, not something a file ships. What has no dense-GEMV kernel stays row-materialized
    /// exactly as the engine's norms, biases, and small vectors do: the depthwise convolution taps (no matmul shape),
    /// each head's stored decay weight, its dt bias, and its output-norm vector.</summary>
    internal struct GdnKernelTensors
    {
        public GdnKernelTensors()
        {
        }

        public int HiddenSize;

        /// <summary>Key heads of the linear attention (its group count); each maps onto one convolved q/k head region.</summary>
        public int KeyHeads;

        /// <summary>Value heads — the recurrent time-step rank: one delta-rule state matrix per value head.</summary>
        public int ValueHeads;

        /// <summary>Shared q/k/v head dimension of the recurrent state, identical on both sides as the reference asserts.</summary>
        public int StateDim;

        public int ConvKernel;

        public LogicalTensor Qkv = null!; // (2 * KeyHeads + ValueHeads) * StateDim rows x HiddenSize — key heads for q, their duplicate region for k, then value-head slices for v
        public LogicalTensor Gate = null!; // ValueHeads * StateDim rows x HiddenSize — the z gate this block reads from its input

        public LogicalTensor Beta = null!; // ValueHeads rows x HiddenSize: one row per value head
        public LogicalTensor Alpha = null!; // same shape as Beta, read after it through softplus and each head's decay weight

        public LogicalTensor Output = null!; // HiddenSize rows x (ValueHeads * StateDim)

        /// <summary>One depthwise row per qkv channel, this channel's taps contiguous — the convolution kernel. A causal
        /// depthwise pass has no dense-GEMV shape to run through a backend kernel, so this stays materialized.</summary>
        public float[] Conv1d = [];

        public float[] SsmA = []; // ValueHeads: stored negative log-decay weights (already negated by conversion)
        public float[] DtBias = []; // ValueHeads added to each head's raw alpha before softplus

        public float[] NormWeight = []; // StateDim applied per value-head output slice

        /// <summary>Conv channels the fused projection and its convolution run over.</summary>
        public readonly int ConvChannels => 2 * KeyHeads * StateDim + ValueHeads * StateDim;

        /// <summary>Concatenated value-head width every delta-rule token output carries before this block's own output projection.</summary>
        public readonly int ValueDim => ValueHeads * StateDim;
    }

    /// <summary>Scratch a gated-delta-net forward needs for one chunk of one sequence, sized once by the runner. Every array is
    /// token-major where it carries per-token rows and head-major inside those rows.</summary>
    internal struct GdnChunkScratch
    {
        public GdnChunkScratch()
        {
        }

        public float[] QkvRaw = []; // tokenCount * ConvChannels — projections before convolution
        public float[] ConvOut = []; // tokenCount * ConvChannels — convolved rows, split into q / k / v regions
        public float[] NormQ = []; // tokenCount * KeyHeads * StateDim normalized q head rows
        public float[] NormK = []; // same shape, normalized k head rows
        public float[] Vraw = []; // tokenCount * ValueDim convolved value-head slices untouched by the L2 norm
        public float[] GateRows = []; // tokenCount * ValueDim raw gate rows read from this block's input
        public float[] LogGate = []; // tokenCount * ValueHeads decay gates per value head
        public float[] Beta = []; // tokenCount * ValueHeads sigmoided betas
        public float[] ValueOut = []; // tokenCount * ValueDim raw delta-rule outputs
        public float[] GatedValue = []; // ValueDim gated-normalized head rows of one token, reused across tokens

        /// <summary>Size every array for a block and chunk width; each covers exactly what the forward reads per step.</summary>
        public static GdnChunkScratch Create(in GdnBlockTensors w, int maxTokenCount) =>
            Create(w.KeyHeads, w.ValueHeads, w.StateDim, w.ConvKernel, maxTokenCount);

        /// <summary>The same sizing from a block's geometry alone — what a kernel-fed block carries before any weight tensor is read.</summary>
        public static GdnChunkScratch Create(int keyHeads, int valueHeads, int stateDim, int convKernel, int maxTokenCount)
        {
            int valueDim = valueHeads * stateDim;
            int channels = 2 * keyHeads * stateDim + valueDim;

            return new GdnChunkScratch
            {
                QkvRaw = new float[maxTokenCount * channels],
                ConvOut = new float[maxTokenCount * channels],
                NormQ = new float[maxTokenCount * keyHeads * stateDim],
                NormK = new float[maxTokenCount * keyHeads * stateDim],
                Vraw = new float[maxTokenCount * valueDim],
                GateRows = new float[maxTokenCount * valueDim],
                LogGate = new float[maxTokenCount * valueHeads],
                Beta = new float[maxTokenCount * valueHeads],
                ValueOut = new float[maxTokenCount * valueDim],
                GatedValue = new float[valueDim],
            };
        }
    }

    /// <summary>
    /// Forward one chunk of one sequence through a gated-delta-net block exactly as the reference graph composes its stages: fused
    /// qkv/gate projections and the merged beta/alpha read this token's hidden row; sigmoid picks beta, softplus over alpha plus each
    /// head's dt bias times that head's stored negative decay gives the log gate; the raw qkv rows convolve causally against the
    /// carried history with SiLU; convolved q and k heads take the whole-head L2 normalization while convolved value slices pass
    /// through untouched; the delta rule advances every value head's state slot array in place; each token's value outputs are
    /// gated-normalized per head, concatenated in ascending head order, and projected through the block output matrix into
    /// <paramref name="hiddenOut"/> for the residual stream upstream. State arrives zeroed from a fresh sequence and advances across
    /// chunks; newConvHistory receives this chunk's raw tail for the next call while convHistory still holds the previous one.
    /// </summary>
    internal static void ForwardChunk(
        ReadOnlySpan<float> hiddenIn, // tokenCount * HiddenSize rows of the normalized attention input
        in GdnBlockTensors w,
        float eps,
        int tokenCount,
        ReadOnlySpan<float> convHistory, // (ConvKernel - 1) * ConvChannels raw qkv history, oldest first
        Span<float> newConvHistory, // same length: this chunk's tail written out for the next step
        Span<float> state, // ValueHeads * StateDim * StateDim slot arrays, advanced in place; zeroed at sequence start
        GdnChunkScratch scratch,
        Span<float> hiddenOut) // tokenCount * HiddenSize projected block output
    {
        if (w.KeyHeads <= 0 || w.ValueHeads <= 0 || w.StateDim <= 0 || w.ConvKernel < 1)
        {
            throw new ArgumentException("A gated-delta-net block needs whole heads and at least one convolution tap.", nameof(w));
        }

        int channels = w.ConvChannels;
        int historyRows = w.ConvKernel - 1;
        int qWidth = w.KeyHeads * w.StateDim; // one token's convolved q region (and, duplicated, its k region) inside the channel row

        if (hiddenIn.Length < tokenCount * w.HiddenSize || hiddenOut.Length < tokenCount * w.HiddenSize)
        {
            throw new ArgumentException("The block forward needs one input and one output row per token.", nameof(hiddenIn));
        }

        if (scratch.QkvRaw.Length < tokenCount * channels || scratch.ConvOut.Length < tokenCount * channels
            || scratch.NormQ.Length < tokenCount * qWidth || scratch.NormK.Length < scratch.NormQ.Length
            || scratch.Vraw.Length < tokenCount * w.ValueDim || scratch.GateRows.Length < tokenCount * w.ValueDim
            || scratch.LogGate.Length < tokenCount * w.ValueHeads || scratch.Beta.Length < tokenCount * w.ValueHeads)
        {
            throw new ArgumentException("Scratch buffers must cover the whole chunk.", nameof(scratch));
        }

        if (convHistory.Length != historyRows * channels || newConvHistory.Length < historyRows * channels)
        {
            throw new ArgumentException("The conv state operands must cover one kernel's worth of raw rows each.", nameof(convHistory));
        }

        // Stage 1: fused projections read the same hidden rows — qkv channels, then gate z, then merged beta/alpha.
        for (int t = 0; t < tokenCount; t++)
        {
            ReadOnlySpan<float> cur = hiddenIn.Slice(t * w.HiddenSize, w.HiddenSize);

            for (int c = 0; c < channels; c++)
            {
                scratch.QkvRaw[t * channels + c] = DotRows(w.Qkv, c, cur, w.HiddenSize);
            }

            for (int r = 0; r < w.ValueDim; r++)
            {
                scratch.GateRows[t * w.ValueDim + r] = DotRows(w.Gate, r, cur, w.HiddenSize);
            }

            for (int h = 0; h < w.ValueHeads; h++)
            {
                scratch.Beta[t * w.ValueHeads + h] = Sigmoid(DotRows(w.BetaAlphaCat, h, cur, w.HiddenSize));

                float alphaBiased = DotRows(w.BetaAlphaCat, w.ValueHeads + h, cur, w.HiddenSize) + w.DtBias[h];
                scratch.LogGate[t * w.ValueHeads + h] = w.SsmA[h] * Softplus(alphaBiased); // the stored weight is already negative: gate = -exp(A_log) * softplus(...)
            }
        }

        RunRecurrentCore(
            tokenCount,
            channels,
            qWidth,
            w.ConvKernel,
            w.KeyHeads,
            w.ValueHeads,
            w.StateDim,
            w.ValueDim,
            eps,
            convHistory,
            newConvHistory,
            state,
            w.Conv1d,
            w.NormWeight,
            scratch);

        // Stage 5's projection: each token's gated-normalized head rows, concatenated in ascending head order, project through
        // its output matrix into the residual stream upstream. The normalization rewrites the one-token scratch before each projection.
        for (int t = 0; t < tokenCount; t++)
        {
            NormGatedHeads(
                scratch.ValueOut.AsSpan(t * w.ValueDim, w.ValueDim),
                w.NormWeight,
                scratch.GateRows.AsSpan(t * w.ValueDim, w.ValueDim),
                eps,
                scratch.GatedValue);

            for (int o = 0; o < w.HiddenSize; o++)
            {
                hiddenOut[t * w.HiddenSize + o] = DotRows(w.Output, o, scratch.GatedValue, w.ValueDim);
            }
        }
    }

    /// <summary>
    /// The same block forward with every projection role read through a compute backend's GEMV kernels over the file's own
    /// packed weight views: the fused qkv rows, the z gate, and the beta and alpha sides each land as one <see
    /// cref="IBackend.Gemv"/> pass over a <see cref="WeightRef"/> resolved from the weight source for this pass, and the
    /// block output projection runs per token through that same seam. Everything downstream of stage 1 — the causal depthwise
    /// convolution against the carried history, the head norms, the delta rule, and the gated normalization — runs through
    /// <see cref="RunRecurrentCore"/> unchanged, so a kernel-fed block and a synthetic block advance identical state for
    /// identical operands. The conv taps, decay weights, dt biases, and head norm stay materialized because no dense-GEMV
    /// kernel exists for a depthwise pass or a scalar head vector.
    /// </summary>
    internal static void ForwardChunkKernel(
        IBackend engine,
        IWeightSource weights,
        in GdnKernelTensors w,
        float eps,
        int tokenCount,
        ReadOnlySpan<float> hiddenIn, // tokenCount * HiddenSize rows of the normalized attention input
        ReadOnlySpan<float> convHistory, // (ConvKernel - 1) * ConvChannels raw qkv history, oldest first
        Span<float> newConvHistory, // same length: this chunk's tail written out for the next step
        Span<float> state, // ValueHeads * StateDim * StateDim slot arrays, advanced in place; zeroed at sequence start
        GdnChunkScratch scratch,
        Span<float> hiddenOut) // tokenCount * HiddenSize projected block output
    {
        ArgumentNullException.ThrowIfNull(engine);
        ArgumentNullException.ThrowIfNull(weights);

        if (w.KeyHeads <= 0 || w.ValueHeads <= 0 || w.StateDim <= 0 || w.ConvKernel < 1)
        {
            throw new ArgumentException("A gated-delta-net block needs whole heads and at least one convolution tap.", nameof(w));
        }

        int channels = w.ConvChannels;
        int historyRows = w.ConvKernel - 1;
        int qWidth = w.KeyHeads * w.StateDim;
        int valueDim = w.ValueDim;

        if (hiddenIn.Length < tokenCount * w.HiddenSize || hiddenOut.Length < tokenCount * w.HiddenSize)
        {
            throw new ArgumentException("The block forward needs one input and one output row per token.", nameof(hiddenIn));
        }

        if (scratch.QkvRaw.Length < tokenCount * channels || scratch.ConvOut.Length < tokenCount * channels
            || scratch.NormQ.Length < tokenCount * qWidth || scratch.NormK.Length < scratch.NormQ.Length
            || scratch.Vraw.Length < tokenCount * valueDim || scratch.GateRows.Length < tokenCount * valueDim
            || scratch.LogGate.Length < tokenCount * w.ValueHeads || scratch.Beta.Length < tokenCount * w.ValueHeads)
        {
            throw new ArgumentException("Scratch buffers must cover the whole chunk.", nameof(scratch));
        }

        if (convHistory.Length != historyRows * channels || newConvHistory.Length < historyRows * channels)
        {
            throw new ArgumentException("The conv state operands must cover one kernel's worth of raw rows each.", nameof(convHistory));
        }

        // Stage 1 through the backend: every projection writes its whole row region in one GEMV pass over the tensor's own bytes.
        for (int t = 0; t < tokenCount; t++)
        {
            ReadOnlySpan<float> cur = hiddenIn.Slice(t * w.HiddenSize, w.HiddenSize);

            engine.Gemv(cur, new WeightRef(weights.Resolve(w.Qkv), w.Qkv.Shape, w.Qkv.Format), scratch.QkvRaw.AsSpan(t * channels, channels));

            engine.Gemv(cur, new WeightRef(weights.Resolve(w.Gate), w.Gate.Shape, w.Gate.Format), scratch.GateRows.AsSpan(t * valueDim, valueDim));

            engine.Gemv(cur, new WeightRef(weights.Resolve(w.Beta), w.Beta.Shape, w.Beta.Format), scratch.Beta.AsSpan(t * w.ValueHeads, w.ValueHeads));
            for (int h = 0; h < w.ValueHeads; h++)
            {
                scratch.Beta[t * w.ValueHeads + h] = Sigmoid(scratch.Beta[t * w.ValueHeads + h]);
            }

            engine.Gemv(cur, new WeightRef(weights.Resolve(w.Alpha), w.Alpha.Shape, w.Alpha.Format), scratch.LogGate.AsSpan(t * w.ValueHeads, w.ValueHeads));
            for (int h = 0; h < w.ValueHeads; h++)
            {
                float alphaBiased = scratch.LogGate[t * w.ValueHeads + h] + w.DtBias[h];
                scratch.LogGate[t * w.ValueHeads + h] = w.SsmA[h] * Softplus(alphaBiased); // the stored weight is already negative: gate = -exp(A_log) * softplus(...)
            }
        }

        RunRecurrentCore(
            tokenCount,
            channels,
            qWidth,
            w.ConvKernel,
            w.KeyHeads,
            w.ValueHeads,
            w.StateDim,
            valueDim,
            eps,
            convHistory,
            newConvHistory,
            state,
            w.Conv1d,
            w.NormWeight,
            scratch);

        // Stage 5's projection through the same kernel seam. Each token's gated-normalized head rows rewrite the one-token scratch
        // before the backend dots them through the output matrix.
        WeightRef outputRef = new(weights.Resolve(w.Output), w.Output.Shape, w.Output.Format);
        for (int t = 0; t < tokenCount; t++)
        {
            NormGatedHeads(
                scratch.ValueOut.AsSpan(t * valueDim, valueDim),
                w.NormWeight,
                scratch.GateRows.AsSpan(t * valueDim, valueDim),
                eps,
                scratch.GatedValue);

            engine.Gemv(scratch.GatedValue, outputRef, hiddenOut.Slice(t * w.HiddenSize, w.HiddenSize));
        }
    }

    // Stages 2 through 5's normalization: the causal depthwise convolution over the carried raw history, the head-region split
    // with its whole-head L2 norms, the delta rule advancing every value head's slot array in place, and each token's gated
    // per-head normalization into the shared GatedValue buffer. Both block front ends run this unchanged.
    private static void RunRecurrentCore(
        int tokenCount,
        int channels,
        int qWidth,
        int convKernel,
        int keyHeads,
        int valueHeads,
        int stateDim,
        int valueDim,
        float eps,
        ReadOnlySpan<float> convHistory,
        Span<float> newConvHistory,
        Span<float> state,
        ReadOnlySpan<float> conv1d,
        ReadOnlySpan<float> normWeight,
        GdnChunkScratch scratch)
    {
        // Stage 2: causal depthwise convolution over every qkv channel with SiLU; this chunk's raw tail becomes next step's history.
        DepthwiseCausalConvSilu(
            convHistory,
            scratch.QkvRaw.AsSpan(0, tokenCount * channels),
            channels,
            tokenCount,
            convKernel,
            conv1d,
            scratch.ConvOut.AsSpan(0, tokenCount * channels),
            newConvHistory);

        // Stage 3: split the convolved rows into head-major regions in channel order — q first, k second, v last. The L2 norm
        // applies to q and k heads only; value slices ride through untouched until the delta rule consumes them.
        for (int t = 0; t < tokenCount; t++)
        {
            ReadOnlySpan<float> convRow = scratch.ConvOut.AsSpan(t * channels, channels);

            L2NormHeads(convRow.Slice(0, qWidth), stateDim, eps, scratch.NormQ.AsSpan(t * qWidth, qWidth));
            L2NormHeads(convRow.Slice(qWidth, qWidth), stateDim, eps, scratch.NormK.AsSpan(t * qWidth, qWidth));

            convRow.Slice(2 * qWidth, valueDim).CopyTo(scratch.Vraw.AsSpan(t * valueDim, valueDim));
        }

        // Stage 4: the delta rule overwrites ValueOut with each token's raw value-head outputs and advances state in place.
        GatedDeltaNetChunk(
            scratch.NormQ.AsSpan(0, tokenCount * qWidth),
            scratch.NormK.AsSpan(0, tokenCount * qWidth),
            scratch.Vraw.AsSpan(0, tokenCount * valueDim),
            scratch.LogGate.AsSpan(0, tokenCount * valueHeads),
            scratch.Beta.AsSpan(0, tokenCount * valueHeads),
            keyHeads,
            valueHeads,
            stateDim,
            tokenCount,
            state,
            scratch.ValueOut.AsSpan(0, tokenCount * valueDim));

        // Stage 5's gated normalization and projection run per token in each caller: GatedValue is one-token scratch and cannot survive a whole-chunk loop.
    }

    private static float DotRows(float[] matrix, int row, ReadOnlySpan<float> vector, int width)
    {
        double sum = 0.0;
        for (int i = 0; i < width; i++)
        {
            sum += (double)matrix[row * width + i] * vector[i];
        }

        return (float)sum;
    }
}
