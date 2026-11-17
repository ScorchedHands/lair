using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models;
using Lair.Models.Engine;
using Lair.Tokenizer;

namespace Lair.Bench;

// Format-abstraction de-risk measurement: a generic LayerOps<TK,TV> quantized-dot attention seam (the static-abstract
// block-struct shape the KV-format design commits to) put against today's hand-written ReadOnlySpan<float> CPU attention
// path, on pinned fixtures through the engine's own single-token decode route.
//
// Nothing here is a production path: the only engine-side change it relies on is TransformerModel's internal opt-in
// attention route, which stays null for every host configuration and never touches the prefill route. The candidate K-dot
// runs over the Q8_K activation-quant machinery Lair.Backends.Cpu already ships - one quantizer call per stored region and
// its own integer sub-block dot with sign fold on the resulting quants, reached through that assembly's measurement
// visibility seams - so no parallel quantizer arithmetic exists in this file.
internal static class KvFormatSpike
{
    private const string DefaultModel = @"C:\source\gitlab.scorchedhands.com\personal\LAIR\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf";

    internal static int RunDecode(string[] args) => Run(args, micro: false);

    internal static int RunMicro(string[] args) => Run(args, micro: true);

    private static int Run(string[] args, bool micro)
    {
        List<string> rest = [.. args[1..]];
        string modelPath = DefaultModel;

        if (rest.Count > 0 && !rest[0].StartsWith("--", StringComparison.Ordinal))
        {
            modelPath = rest[0];
            rest.RemoveAt(0);
        }

        int threads = ParseInt(rest, "--threads", 8);
        int context = ParseInt(rest, "--context", micro ? 256 : 1024);
        int decodeTokens = ParseInt(rest, "--decode", 64);
        int windows = ParseInt(rest, "--windows", 3);
        List<string> modes = [.. ParseList(rest, "--modes", ["baseline", "f32block", "q8k"])];

        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"kv-spike: no fixture at {modelPath}");
            return 1;
        }

        using GgufFile gguf = GgufFile.Open(modelPath);
        ModelPlan geometry = ModelPlans.Resolve(gguf);
        int capacity = context + (windows + 1) * decodeTokens; // every position the run addresses: prefilled, warmed up, and decoded

        using CpuBackend backend = new(threads: threads, allowAvx2: true);
        KvKernelOptions options = new(backend.ResolvedIntegerDotPath, backend.KvSpikeUsesAvx2FloatKernels);

        Console.WriteLine($"kv-spike {(micro ? "micro" : "decode")} file={gguf.Path} bytes={gguf.FileSize} threads={backend.EffectiveWorkerCount} isaResolved={ResolvedIntegerDotNames.Of(backend.ResolvedIntegerDotPath)} avx2Float={options.Avx2}");
        Console.WriteLine(
            $"  arch={geometry.Architecture} layers={geometry.LayerCount} hidden={geometry.HiddenSize} heads={geometry.HeadCount} " +
            $"kvHeads={geometry.KvHeadCount} headDim={geometry.HeadDim} vocab={geometry.VocabSize} context={context}");

        foreach (string mode in modes)
        {
            if (mode is not ("baseline" or "f32block" or "q8k"))
            {
                Console.WriteLine($"kv-spike: unknown mode '{mode}' - expected baseline, f32block, q8k");
                return 1;
            }

            try
            {
                if (micro)
                {
                    RunMicroCase(mode, geometry, backend, options, context);
                }
                else
                {
                    RunDecodeCase(mode, gguf, geometry, backend, options, capacity, context, decodeTokens, windows);
                }
            }
            catch (Exception failure)
            {
                Console.WriteLine($"kv-spike mode={mode} status=failed: {failure.Message}");
                return 1;
            }

            Console.WriteLine();
        }

        double peakWorkingSetMb = Process.GetCurrentProcess().PeakWorkingSet64 / (1024.0 * 1024.0);
        Console.WriteLine($"  peakWorkingSetMb={peakWorkingSetMb:F1} across this process run");
        return 0;
    }

    // Real pinned-fixture decode: one engine per mode, the prompt windowed through the production prefill route to fill the
    // context, then each timed window advances that sequence by --decode single-token steps - exactly the attention call
    // shape a served decode runs at this position. Only the attention step (and its storage) differs between modes.
    private static void RunDecodeCase(
        string mode, GgufFile gguf, ModelPlan geometry, CpuBackend backend, KvKernelOptions options, int capacity,
        int context, int decodeTokens, int windows)
    {
        int blockTokens = 16; // the granularity every engine KV side uses
        int capacityBlocks = (capacity + blockTokens - 1) / blockTokens + 4;

        using TransformerModel model = TransformerModel.Open(
            gguf, backend, maxContextTokens: capacity + blockTokens, logLine: null, PrefillKernel.GemV);

        using BlockPool pool = new(blockTokens, geometry.LayerCount, geometry.KvHeadCount, geometry.HeadDim, capacityBlocks, backend);
        using SequenceKv kv = new(pool);

        GgufTokenizer tokenizer = GgufTokenizer.FromGguf(gguf);
        int[] prompt = BuildPrompt(tokenizer, context);

        Stopwatch prefillWatch = Stopwatch.StartNew();
        model.Prefill(prompt, kv); // fills every position the decode steps attend over: setup cost, reported apart from timing
        double prefillMs = ElapsedMs(prefillWatch);

        if (mode == "baseline")
        {
            Console.WriteLine($"kv-spike mode={mode} storage=production f32/f32 attention=pinned span<float> kernel");
            _baselineProbe = RunWindows(model, kv, mode, prompt[0], decodeTokens, windows); // a fixed in-vocabulary id: every window does identical work
            ReportProbe(mode, _baselineProbe, geometry.VocabSize);

            return;
        }

        using SpikeMirror mirror = new(mode, geometry.LayerCount, capacity, geometry.KvHeadCount * geometry.HeadDim);

        Stopwatch fillWatch = Stopwatch.StartNew();
        QuantizeRange(mirror, kv, geometry, 0, context - 1); // the prefilled positions this decode's reads address from candidate storage
        double fillMs = ElapsedMs(fillWatch);

        model.EnableKvFormatSpike(RouteFor(mode, geometry, options, mirror));
        Console.WriteLine($"kv-spike mode={mode} prefillSetupMs={prefillMs:F1} mirrorQuantizeMs={fillMs:F1} quantizedPositions={context}");

        ReportProbe(mode, RunWindows(model, kv, mode, prompt[0], decodeTokens, windows), geometry.VocabSize);
    }

    private static float[] _baselineProbe = [];

    // The same sequence's final decoded token per mode: identical prompt and identical forced continuation, so any logit drift is the candidate attention's own error.
    private static void ReportProbe(string mode, float[] probe, int vocabSize)
    {
        if (probe.Length == 0)
        {
            return;
        }

        int top = 0;
        for (int i = 1; i < probe.Length; i++)
        {
            if (probe[i] > probe[top])
            {
                top = i;
            }
        }

        Console.WriteLine($"kv-spike mode={mode} probeTopId={top}/{vocabSize} probeTopLogit={probe[top]:F4}");

        if (_baselineProbe.Length != probe.Length || mode == "baseline")
        {
            return;
        }

        double maxAbs = 0d;
        for (int i = 0; i < probe.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(probe[i] - _baselineProbe[i]));
        }

        Console.WriteLine($"kv-spike mode={mode} probeEquivalence argmaxMatch={(top == ArgMax(_baselineProbe))} logitsMaxAbsDiffVsBaseline={maxAbs:F4}");
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }

    private static float[] RunWindows(TransformerModel model, SequenceKv kv, string mode, int nextToken, int decodeTokens, int windows)
    {
        Stopwatch warmWatch = Stopwatch.StartNew();
        for (int i = 0; i < decodeTokens; i++)
        {
            model.DecodeNext(nextToken, kv); // JIT warm-up window: never timed
        }

        double warmMs = ElapsedMs(warmWatch);
        Console.WriteLine($"kv-spike mode={mode} warmWindowTok/s={(decodeTokens / (warmMs / 1000.0)):F3}");

        List<double> perWindowTokS = [];
        float[] probe = [];

        for (int w = 0; w < windows; w++)
        {
            int firstPosition = kv.Length;

            Stopwatch windowWatch = Stopwatch.StartNew();
            for (int i = 0; i < decodeTokens; i++)
            {
                Span<float> logits = model.DecodeNext(nextToken, kv);

                if (w == windows - 1 && i == decodeTokens - 1)
                {
                    probe = [.. logits]; // the final forced token's full distribution: copied out because the engine reuses its logit buffer every step
                }
            }

            double ms = ElapsedMs(windowWatch);
            perWindowTokS.Add(decodeTokens / (ms / 1000.0));

            Console.WriteLine(
                $"kv-spike mode={mode} window={w + 1} positions={firstPosition}..{kv.Length - 1} decodeMs={ms:F1} " +
                $"tok/s={(decodeTokens / (ms / 1000.0)):F3}");
        }

        Console.WriteLine(FormatStats(mode, perWindowTokS));

        return probe;
    }

    // Kernel-only shape comparison over synthetic KV at the fixture's geometry: production LayerOps.CausalAttention (the
    // hand-written span<float> path with its pool addressing and IBackend.Dot call) against each candidate seam kernel run
    // over a block-struct mirror of identical bytes, timed per attention call.
    private static void RunMicroCase(string mode, ModelPlan geometry, CpuBackend backend, KvKernelOptions options, int positions)
    {
        int blockTokens = 16;
        int capacityBlocks = (positions + blockTokens - 1) / blockTokens + 4;

        using BlockPool pool = new(blockTokens, geometry.LayerCount, geometry.KvHeadCount, geometry.HeadDim, capacityBlocks, backend);
        using SequenceKv kv = new(pool);

        Random rng = new(20261117);
        int kvDim = geometry.KvHeadCount * geometry.HeadDim;
        int headRegion = geometry.HeadCount * geometry.HeadDim;

        float[] region = new float[kvDim];

        // One sequence position per Append, exactly as the engine's own prefill lays a sequence out: every layer of that token writes into the same block index.
        for (int position = 0; position < positions; position++)
        {
            kv.Append();

            for (int layer = 0; layer < geometry.LayerCount; layer++)
            {
                foreach (ref float value in region.AsSpan())
                {
                    value = (float)(rng.NextDouble() * 2.0 - 1.0);
                }

                kv.WriteK(position, layer, region);

                foreach (ref float value in region.AsSpan())
                {
                    value = (float)(rng.NextDouble() * 2.0 - 1.0);
                }

                kv.WriteV(position, layer, region);
            }
        }

        float[] qHeads = new float[headRegion];
        foreach (ref float value in qHeads.AsSpan())
        {
            value = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        float attnScale = 1f / MathF.Sqrt(geometry.HeadDim);
        int layerCount = geometry.LayerCount;
        const int walksPerRep = 8; // eight full walks of every layer per repetition

        List<double> handMs = [];
        for (int rep = 0; rep < 5; rep++)
        {
            float[] outConcat = new float[headRegion]; // one output buffer per repetition: the kernel normalizes what it wrote, not a reused residue

            Stopwatch watch = Stopwatch.StartNew();
            for (int i = 0; i < walksPerRep; i++)
            {
                for (int layer = 0; layer < layerCount; layer++)
                {
                    LayerOps.CausalAttention(
                        backend, qHeads, geometry.HeadCount, geometry.KvHeadCount, geometry.HeadDim, kv, layer, positions - 1,
                        attnScale, softcap: 0f, window: 0, outConcat);
                }
            }

            handMs.Add(ElapsedMs(watch) / (walksPerRep * layerCount));
        }

        Console.WriteLine($"kv-spike micro mode=hand positions={positions} perCallUs=" + string.Join(", ", handMs.Select(v => $"{v * 1000.0:F2}")));

        if (mode == "baseline")
        {
            return;
        }

        using SpikeMirror mirror = new(mode, geometry.LayerCount, positions, kvDim);
        QuantizeRange(mirror, kv, geometry, 0, positions - 1);

        TransformerModel.KvAttentionSpikeRoute route = RouteFor(mode, geometry, options, mirror);

        // Equivalence check at the attention-output level before any timing claim: same query heads and same KV bytes through
        // the pinned kernel and through the candidate seam, layer by layer. A seam that scores faster while scoring wrong is not a win.
        float[] handOut = new float[headRegion];
        float[] spikeOut0 = new float[headRegion];
        double maxAbsDiff = 0d;
        double meanRelDiff = 0d;

        for (int layer = 0; layer < layerCount; layer++)
        {
            LayerOps.CausalAttention(
                backend, qHeads, geometry.HeadCount, geometry.KvHeadCount, geometry.HeadDim, kv, layer, positions - 1,
                attnScale, softcap: 0f, window: 0, handOut);

            route(qHeads, geometry.HeadCount, geometry.KvHeadCount, geometry.HeadDim, kv, layer, positions - 1, attnScale, softcap: 0f, window: 0, spikeOut0);

            for (int j = 0; j < headRegion; j++)
            {
                double diff = Math.Abs(handOut[j] - spikeOut0[j]);
                maxAbsDiff = Math.Max(maxAbsDiff, diff);
                meanRelDiff += handOut[j] == 0f ? diff : diff / Math.Abs(handOut[j]);
            }
        }

        Console.WriteLine(
            $"kv-spike micro mode={mode} attentionEquivalence positions={positions} maxAbsDiff={maxAbsDiff:E3} " +
            $"meanRelDiff={(meanRelDiff / (layerCount * headRegion)):E3}");

        List<double> candidateMs = [];
        for (int rep = 0; rep < 5; rep++)
        {
            float[] spikeOut = new float[headRegion];

            Stopwatch watch = Stopwatch.StartNew();
            for (int i = 0; i < walksPerRep; i++)
            {
                for (int layer = 0; layer < layerCount; layer++)
                {
                    route(qHeads, geometry.HeadCount, geometry.KvHeadCount, geometry.HeadDim, kv, layer, positions - 1, attnScale, softcap: 0f, window: 0, spikeOut);
                }
            }

            candidateMs.Add(ElapsedMs(watch) / (walksPerRep * layerCount));
        }

        double handMedian = Median(handMs);
        double candidateMedian = Median(candidateMs);

        Console.WriteLine($"kv-spike micro mode={mode} positions={positions} perCallUs=" + string.Join(", ", candidateMs.Select(v => $"{v * 1000.0:F2}")));
        Console.WriteLine(
            $"kv-spike micro comparison handMedianUsPerCall={handMedian * 1000.0:F3} candidateMedianUsPerCall={candidateMedian * 1000.0:F3} " +
            $"deltaPct={(candidateMedian - handMedian) / handMedian * 100.0:+0.0;-0.0}");
    }

    private static void QuantizeRange(SpikeMirror mirror, SequenceKv kv, ModelPlan geometry, int fromPosition, int toPosition)
    {
        for (int layer = 0; layer < geometry.LayerCount; layer++)
        {
            for (int position = fromPosition; position <= toPosition; position++)
            {
                mirror.QuantizeOne(kv, layer, position); // the same storage write every decoded token pays
            }
        }
    }

    // The candidate attention route for one mode, closing over its own block-struct storage and query scratch. Each combo is
    // instantiated once as a closed generic over (TK,TV), so no format branch exists inside the position loop it runs; every
    // decoded token's K/V enters that storage in the same step that attends them, which is what quantized append costs.
    private static TransformerModel.KvAttentionSpikeRoute RouteFor(
        string mode, ModelPlan geometry, KvKernelOptions options, SpikeMirror mirror) =>
        mode == "f32block"
            ? RouteGeneric<F32Block, F32Block>(mirror, options, geometry)
            : RouteGeneric<Q8DotBlock, Q8ValueBlock>(mirror, options, geometry);

    private static TransformerModel.KvAttentionSpikeRoute RouteGeneric<TK, TV>(
        SpikeMirror mirror, KvKernelOptions options, ModelPlan geometry)
        where TK : struct, IKvFormat<TK>
        where TV : struct, IKvFormat<TV>
    {
        byte[] queryScratch = new byte[Math.Max(1, TK.QueryScratchBytesPerHead) * geometry.HeadCount]; // prepared-query side reused across every call
        float[] queryScales = new float[geometry.HeadCount];

        return (ReadOnlySpan<float> qHeads, int heads, int kvHeadCount, int headDim, SequenceKv kv, int layer, int position,
                float attnScale, float softcap, int window, Span<float> outConcat) =>
        {
            // This token's K/V enter candidate storage before the step that attends them: production attends a freshly appended
            // token against its own K/V too, so scoring it without its region would understate both the write and the read.
            TK.WriteRegion(kv.ReadK(position, layer), MemoryMarshal.Cast<byte, TK>(mirror.KRegion(layer, position)));
            TV.WriteRegion(kv.ReadV(position, layer), MemoryMarshal.Cast<byte, TV>(mirror.VRegion(layer, position)));

            LayerOpsSpike.CausalAttention<TK, TV>(
                qHeads.Slice(0, heads * headDim), heads, kvHeadCount, headDim, attnScale, softcap, position, window,
                mirror.KStorage.AsSpan(), mirror.VStorage.AsSpan(), layer, mirror.CapacityPositions,
                queryScratch.AsSpan(), queryScales.AsSpan(), options, outConcat);
        };
    }

    // The candidate hot path: one generic causal-attention body over block-struct storage, instantiated per (TK,TV) combo.
    // Its online softmax mirrors the pinned kernel's update order exactly; only K/V reads and their dots/accumulations differ.
    internal static class LayerOpsSpike
    {
        internal unsafe static void CausalAttention<TK, TV>(
            ReadOnlySpan<float> qHeads, int headCount, int kvHeadCount, int headDim, float attnScale, float softcap,
            int position, int window, Span<byte> kStorage, Span<byte> vStorage, int layer, int capacityPositions,
            Span<byte> queryScratch, Span<float> queryScales, in KvKernelOptions options, Span<float> outConcat)
            where TK : struct, IKvFormat<TK>
            where TV : struct, IKvFormat<TV>
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

            int region = headCount * headDim;
            if (qHeads.Length < region || outConcat.Length < region)
            {
                throw new ArgumentException("Attention operands must cover every head's full region.", nameof(qHeads));
            }

            TK.PrepareQuery(qHeads, headCount, headDim, queryScratch, queryScales, options); // once per attention call: the quantized-activation step this combo stages

            int firstPosition = window > 0 ? Math.Max(0, position - window + 1) : 0; // the same causal narrowing the pinned kernel resolves
            int regionBlocks = kvHeadCount * headDim / 32; // every spike format groups 32 values per block
            int kRegionBytes = regionBlocks * TK.BlockBytes;
            int vRegionBytes = regionBlocks * TV.BlockBytes;

            for (int h = 0; h < headCount; h++)
            {
                ReadOnlySpan<float> qh = qHeads.Slice(h * headDim, headDim);
                int kvHead = h * kvHeadCount / headCount; // grouped-query mapping identical to the pinned kernel's
                int kBlockOffset = kvHead * (headDim / TK.ValuesPerBlock);
                int vBlockOffset = kvHead * (headDim / TV.ValuesPerBlock);

                Span<float> outHead = outConcat.Slice(h * headDim, headDim);

                float runningMax = float.NegativeInfinity;
                float sumExp = 0f;
                outHead.Clear();

                for (int t = firstPosition; t <= position; t++)
                {
                    long regionBase = ((long)layer * capacityPositions + t) * regionBlocks;

                    ReadOnlySpan<TK> kRegion = MemoryMarshal.Cast<byte, TK>(kStorage.Slice((int)checked(regionBase * TK.BlockBytes), kRegionBytes));
                    float score = TK.DotHead(kRegion, kBlockOffset, qh, queryScratch, queryScales, h, options) * attnScale;

                    if (softcap > 0f)
                    {
                        score = LayerOps.Softcap(score, softcap); // after scaling, before the online softmax sees it
                    }

                    if (score > runningMax)
                    {
                        // Rescale everything accumulated under the old max so all weights share the new exponent base.
                        if (!float.IsNegativeInfinity(runningMax))
                        {
                            float rescale = MathF.Exp(runningMax - score);
                            sumExp *= rescale;

                            for (int j = 0; j < headDim; j++)
                            {
                                outHead[j] *= rescale;
                            }
                        }

                        runningMax = score;
                    }

                    float weight = MathF.Exp(score - runningMax); // the current max term contributes exactly 1
                    sumExp += weight;

                    ReadOnlySpan<TV> vRegion = MemoryMarshal.Cast<byte, TV>(vStorage.Slice((int)checked(regionBase * TV.BlockBytes), vRegionBytes));
                    TV.AccumulateHead(outHead, vRegion, vBlockOffset, weight, options);
                }

                float invSum = 1f / sumExp; // >= 1 because the max term contributes exp(0)
                for (int j = 0; j < headDim; j++)
                {
                    outHead[j] *= invSum;
                }
            }
        }
    }

    /// <summary>
    /// The cold descriptor axes a storage item would read (values per block, bytes per block) beside the hot-path static
    /// abstract dot/accumulate members. Each format keeps payload and scale inside one fixed-size struct, so a token region
    /// stays contiguous at block granularity - which is what lets a head slice inside it reach a dot without a copy.
    /// </summary>
    internal interface IKvFormat<TSelf> where TSelf : struct, IKvFormat<TSelf>
    {
        static abstract int ValuesPerBlock { get; }

        static abstract int BlockBytes { get; }

        /// <summary>Query-side scratch bytes one attention call needs per query head (zero for a side that dots against FP32 queries).</summary>
        static abstract int QueryScratchBytesPerHead { get; }

        /// <summary>Quantizes or copies one token's whole K or V region into this format's blocks - the append-time storage write.</summary>
        static abstract void WriteRegion(ReadOnlySpan<float> values, Span<TSelf> blocks);

        /// <summary>Prepares the query side once per attention call: a Q8_K quantized activation for integer dots, nothing for FP32 dots.</summary>
        static abstract void PrepareQuery(ReadOnlySpan<float> qHeads, int headCount, int headDim, Span<byte> scratch, Span<float> scales, in KvKernelOptions options);

        /// <summary>The K-dot of one scored position's kv-head region against this call's prepared query head.</summary>
        static abstract float DotHead(ReadOnlySpan<TSelf> tokenRegion, int headBlockOffset, ReadOnlySpan<float> qh, Span<byte> scratch, Span<float> scales, int headSlot, in KvKernelOptions options);

        /// <summary>The V-side weighted accumulation for one scored position's kv-head region.</summary>
        static abstract void AccumulateHead(Span<float> outHead, ReadOnlySpan<TSelf> tokenRegion, int headBlockOffset, float weight, in KvKernelOptions options);
    }

    /// <summary>The two kernel facts a candidate seam resolves once at open and never re-decides: the backend's resolved integer-dot encoding and its AVX2 float posture.</summary>
    internal readonly record struct KvKernelOptions(IntegerDotPath Dot, bool Avx2);

    // K side of the quantized combo: one 32-value group holding a scale, packed sign bits, and unsigned magnitudes - exactly
    // the operand shape this backend's shipped Q8_K sub-block dot consumes (magnitudes below 128, signs folded onto the
    // activation). The quants come straight from QuantizeActivationToQ8K over the whole region; splitting them into
    // magnitude and sign is storage layout only, and the fold that pairs with it runs inside the same shipped helper.
    internal unsafe struct Q8DotBlock : IKvFormat<Q8DotBlock>
    {
        public const int Values = 32;

        public float Scale;
        public uint Signs;
        public fixed byte Magnitudes[Values];

        public static int ValuesPerBlock => Values;

        public static int BlockBytes => Unsafe.SizeOf<Q8DotBlock>(); // the footprint axis a pool would size blocks from

        public static int QueryScratchBytesPerHead => CpuBackend.KvSpikeQ8KActivationBytes(128); // one Q8_K super-block covers one head vector

        public static void WriteRegion(ReadOnlySpan<float> values, Span<Q8DotBlock> blocks)
        {
            int cols = values.Length;

            byte* qa = stackalloc byte[CpuBackend.KvSpikeQ8KActivationBytes(cols)]; // whole super-blocks: a group never runs off the end
            float* da = stackalloc float[CpuBackend.KvSpikeQ8KBlockCount(cols)];

            fixed (float* pValues = values)
            {
                CpuBackend.KvSpikeQuantizeActivationToQ8K(pValues, cols, qa, da); // the shipped quantizer: same scale rule and rounding every sub-4-bit GEMV stages against
            }

            for (int blockIndex = 0; blockIndex < blocks.Length; blockIndex++)
            {
                int start = blockIndex * Values;
                Q8DotBlock* block = (Q8DotBlock*)Unsafe.AsPointer(ref blocks[0]) + blockIndex;

                uint signs = 0u;
                for (int i = 0; i < Values; i++)
                {
                    sbyte q = (sbyte)qa[start + i]; // already in [-127,127] by the quantizer's own scale rule: no -128 to fold

                    if (q < 0)
                    {
                        signs |= 1u << i;
                        block->Magnitudes[i] = (byte)(-q); // magnitude stays below 128: the sign belongs to the activation fold
                    }
                    else
                    {
                        block->Magnitudes[i] = (byte)q;
                    }
                }

                block->Signs = signs;
                block->Scale = da[start / QkSuper]; // this group's super-block scale, carried in the block beside its payload
            }
        }

        public static void PrepareQuery(ReadOnlySpan<float> qHeads, int headCount, int headDim, Span<byte> scratch, Span<float> scales, in KvKernelOptions options)
        {
            fixed (float* pQ = qHeads)
            fixed (byte* pScratch = scratch)
            fixed (float* pScales = scales)
            {
                for (int h = 0; h < headCount; h++)
                {
                    // One quantized activation per query head, once per attention call: the same helper every sub-4-bit GEMV stages.
                    CpuBackend.KvSpikeQuantizeActivationToQ8K(pQ + (long)h * headDim, headDim, pScratch + (long)h * QueryScratchBytesPerHead, pScales + h);
                }
            }

            _ = options; // the dot encoding this side runs arrives with each call's resolved options
        }

        public static float DotHead(ReadOnlySpan<Q8DotBlock> tokenRegion, int headBlockOffset, ReadOnlySpan<float> qh, Span<byte> scratch, Span<float> scales, int headSlot, in KvKernelOptions options)
        {
            fixed (byte* pScratch = scratch)
            {
                sbyte* pAct = (sbyte*)pScratch + (long)headSlot * QueryScratchBytesPerHead; // this call's quantized query head

                // The region is contiguous block storage, so the head's blocks are reached by pointer offset - the same shape the pinned kernels use.
                Q8DotBlock* blocks = (Q8DotBlock*)Unsafe.AsPointer(ref Unsafe.AsRef(in MemoryMarshal.GetReference(tokenRegion))) + headBlockOffset;

                float acc = 0f;
                int headBlocks = qh.Length / Values;

                for (int blockIndex = 0; blockIndex < headBlocks; blockIndex++)
                {
                    Q8DotBlock* block = blocks + blockIndex;

                    CpuBackend.KvSpikeSubBlockDotQ8K(
                        block->Magnitudes, block->Signs, pAct + blockIndex * Values, out int loSum, out int hiSum, options.Dot); // the shipped integer dot over 32 values: magnitudes times sign-folded Q8_K quants

                    acc += (float)(loSum + hiSum) * block->Scale; // exact int group total scaled by this side's stored scale
                }

                return acc * scales[headSlot]; // then the activation super-block scale, once per head region as every Q8_K kernel applies it
            }
        }

        public static void AccumulateHead(Span<float> outHead, ReadOnlySpan<Q8DotBlock> tokenRegion, int headBlockOffset, float weight, in KvKernelOptions options) =>
            throw new NotSupportedException("The integer-dot side stores magnitudes and signs for the K-dot; a value side accumulates signed quants.");

        private const int QkSuper = 256; // one scale per Q8_K super-block, exactly as the shipped quantizer lays them out
    }

    // V side of the quantized combo: one 32-value group of signed Q8_K quants plus its scale - dequantize-then-FMA
    // accumulation in this backend's own Q8_0 unpack shape (vpmovsxbd widening, broadcast scale, FMA).
    internal unsafe struct Q8ValueBlock : IKvFormat<Q8ValueBlock>
    {
        public const int Values = 32;

        public float Scale;
        public fixed sbyte Quants[Values];

        public static int ValuesPerBlock => Values;

        public static int BlockBytes => Unsafe.SizeOf<Q8ValueBlock>();

        public static int QueryScratchBytesPerHead => 0; // the value side never touches the prepared query scratch

        public static void WriteRegion(ReadOnlySpan<float> values, Span<Q8ValueBlock> blocks)
        {
            int cols = values.Length;

            byte* qa = stackalloc byte[CpuBackend.KvSpikeQ8KActivationBytes(cols)];
            float* da = stackalloc float[CpuBackend.KvSpikeQ8KBlockCount(cols)];

            fixed (float* pValues = values)
            {
                CpuBackend.KvSpikeQuantizeActivationToQ8K(pValues, cols, qa, da); // the shipped quantizer again: stored quants are its own bytes, unmodified
            }

            for (int blockIndex = 0; blockIndex < blocks.Length; blockIndex++)
            {
                ref Q8ValueBlock block = ref blocks[blockIndex];
                int start = blockIndex * Values;

                for (int i = 0; i < Values; i++)
                {
                    block.Quants[i] = (sbyte)qa[start + i]; // signed quants straight out of the quantizer, padding included
                }

                block.Scale = da[start / QkSuper]; // this group's super-block scale, carried in the block beside its payload
            }
        }

        public static void PrepareQuery(ReadOnlySpan<float> qHeads, int headCount, int headDim, Span<byte> scratch, Span<float> scales, in KvKernelOptions options)
        {
            _ = qHeads;
            _ = headCount;
            _ = headDim;
            _ = scratch;
            _ = scales;
            _ = options; // the value side has no query preparation: it only accumulates weighted values
        }

        public static float DotHead(ReadOnlySpan<Q8ValueBlock> tokenRegion, int headBlockOffset, ReadOnlySpan<float> qh, Span<byte> scratch, Span<float> scales, int headSlot, in KvKernelOptions options) =>
            throw new NotSupportedException("The value side is not a K-dot format; the combo pairs it with the integer-dot side.");

        public static void AccumulateHead(Span<float> outHead, ReadOnlySpan<Q8ValueBlock> tokenRegion, int headBlockOffset, float weight, in KvKernelOptions options)
        {
            int blocks = outHead.Length / Values;

            // Contiguous block storage again: the value side's groups are reached by offset from the region base.
            Q8ValueBlock* stored = (Q8ValueBlock*)Unsafe.AsPointer(ref Unsafe.AsRef(in MemoryMarshal.GetReference(tokenRegion))) + headBlockOffset;

            fixed (float* pOut = outHead)
            {
                if (!options.Avx2 || !Avx2.IsSupported || !Fma.IsSupported)
                {
                    for (int blockIndex = 0; blockIndex < blocks; blockIndex++)
                    {
                        Q8ValueBlock* block = stored + blockIndex;

                        for (int i = 0; i < Values; i++)
                        {
                            pOut[blockIndex * Values + i] += weight * ((float)block->Quants[i] * block->Scale); // scalar posture: same arithmetic, no widening
                        }
                    }

                    return;
                }

                float w = weight;
                Vector256<float> wv = Avx.BroadcastScalarToVector256(&w);

                for (int blockIndex = 0; blockIndex < blocks; blockIndex++)
                {
                    Q8ValueBlock* block = stored + blockIndex;

                    float scale = block->Scale;
                    Vector256<float> sv = Avx.BroadcastScalarToVector256(&scale);

                    for (int lane = 0; lane < Values; lane += 8)
                    {
                        Vector256<int> qi = Avx2.ConvertToVector256Int32(block->Quants + lane); // vpmovsxbd: eight signed quants, in-block only

                        float* pSlot = pOut + blockIndex * Values + lane;
                        Vector256<float> dequant = Avx.Multiply(Avx.ConvertToVector256Single(qi), sv); // q * d unpacked once per chunk
                        Avx.Store(pSlot, Fma.MultiplyAdd(wv, dequant, Avx.LoadVector256(pSlot))); // out[j] += weight * (q * d)
                    }
                }
            }
        }

        private const int QkSuper = 256; // one scale per Q8_K super-block, exactly as the shipped quantizer lays them out
    }

    // The mechanism control's element: one 32-value group of plain FP32 values. Same block-granularity storage shape and the
    // same generic call structure as the quantized combo, so what separates it from the pinned path is the seam itself - not
    // any quantization arithmetic. Its dot calls this backend's own AVX2 float kernel over a contiguous head region.
    internal unsafe struct F32Block : IKvFormat<F32Block>
    {
        public const int Values = 32;

        public fixed float Elements[Values]; // the whole payload of one group: this format stores no scale

        public static int ValuesPerBlock => Values;

        public static int BlockBytes => Unsafe.SizeOf<F32Block>();

        public static int QueryScratchBytesPerHead => 0; // FP32 dots consume no prepared query side

        public static void WriteRegion(ReadOnlySpan<float> values, Span<F32Block> blocks)
        {
            MemoryMarshal.AsBytes(values).CopyTo(MemoryMarshal.Cast<F32Block, byte>(blocks)); // a copy is this format's whole storage write
        }

        public static void PrepareQuery(ReadOnlySpan<float> qHeads, int headCount, int headDim, Span<byte> scratch, Span<float> scales, in KvKernelOptions options)
        {
            _ = qHeads;
            _ = headCount;
            _ = headDim;
            _ = scratch;
            _ = scales;
            _ = options; // nothing to prepare: the dot runs against this call's FP32 query heads
        }

        public static float DotHead(ReadOnlySpan<F32Block> tokenRegion, int headBlockOffset, ReadOnlySpan<float> qh, Span<byte> scratch, Span<float> scales, int headSlot, in KvKernelOptions options)
        {
            _ = scratch;
            _ = scales;
            _ = headSlot;

            // The head's blocks sit consecutively inside the region, so one call over the whole head vector is what a copy-less
            // block read hands a float dot: same operand count and the same kernel the pinned attention path calls.
            fixed (float* pQ = qh)
            {
                float* pV = ((F32Block*)Unsafe.AsPointer(ref Unsafe.AsRef(in MemoryMarshal.GetReference(tokenRegion))))[headBlockOffset].Elements;

                return options.Avx2
                    ? CpuBackend.KvSpikeFloatDot(pQ, pV, qh.Length)
                    : CpuBackend.KvSpikeFloatDotScalar(pQ, pV, qh.Length);
            }
        }

        public static void AccumulateHead(Span<float> outHead, ReadOnlySpan<F32Block> tokenRegion, int headBlockOffset, float weight, in KvKernelOptions options)
        {
            _ = options;

            float* pV = ((F32Block*)Unsafe.AsPointer(ref Unsafe.AsRef(in MemoryMarshal.GetReference(tokenRegion))))[headBlockOffset].Elements;

            fixed (float* pOut = outHead)
            {
                for (int j = 0; j < outHead.Length; j++)
                {
                    pOut[j] += weight * pV[j]; // the pinned kernel's accumulation, element order included
                }
            }
        }
    }

    /// <summary>
    /// Block-granularity storage for one sequence: a byte arena laid out as [layer][position][region blocks], contiguous per
    /// token region so a head slice inside it stays copy-less. Sized from the format's own block size - the cold descriptor axis.
    /// </summary>
    private sealed class SpikeMirror : IDisposable
    {
        public byte[] KStorage { get; }

        public byte[] VStorage { get; }

        public int CapacityPositions { get; }

        private readonly bool _f32Storage;
        private readonly int _regionBlocks;
        private readonly int _kBlockBytes;
        private readonly int _vBlockBytes;

        public SpikeMirror(string mode, int layerCount, int capacityPositions, int kvDimValues)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacityPositions);

            if (kvDimValues % 32 != 0)
            {
                throw new ArgumentException($"a spike KV region must cover whole 32-value groups, got {kvDimValues} values.", nameof(kvDimValues));
            }

            _f32Storage = mode == "f32block";
            CapacityPositions = capacityPositions;
            _regionBlocks = kvDimValues / 32; // whole groups per token region, whatever block size each side uses
            _kBlockBytes = _f32Storage ? Unsafe.SizeOf<F32Block>() : Unsafe.SizeOf<Q8DotBlock>();
            _vBlockBytes = _f32Storage ? Unsafe.SizeOf<F32Block>() : Unsafe.SizeOf<Q8ValueBlock>();

            int blocksPerSide = layerCount * capacityPositions * _regionBlocks; // one region per (layer, position) the sequence can address
            KStorage = new byte[checked(blocksPerSide * _kBlockBytes)];
            VStorage = new byte[checked(blocksPerSide * _vBlockBytes)];
        }

        public Span<byte> KRegion(int layer, int position) => Region(KStorage, _kBlockBytes, layer, position);

        public Span<byte> VRegion(int layer, int position) => Region(VStorage, _vBlockBytes, layer, position);

        /// <summary>Quantizes or copies one token's K and V regions into this mirror - the storage write every append pays.</summary>
        public void QuantizeOne(SequenceKv kv, int layer, int position)
        {
            if (_f32Storage)
            {
                F32Block.WriteRegion(kv.ReadK(position, layer), MemoryMarshal.Cast<byte, F32Block>(KRegion(layer, position)));
                F32Block.WriteRegion(kv.ReadV(position, layer), MemoryMarshal.Cast<byte, F32Block>(VRegion(layer, position)));

                return;
            }

            Q8DotBlock.WriteRegion(kv.ReadK(position, layer), MemoryMarshal.Cast<byte, Q8DotBlock>(KRegion(layer, position)));
            Q8ValueBlock.WriteRegion(kv.ReadV(position, layer), MemoryMarshal.Cast<byte, Q8ValueBlock>(VRegion(layer, position)));
        }

        private Span<byte> Region(byte[] storage, int blockBytes, int layer, int position)
        {
            long baseBlock = ((long)layer * CapacityPositions + position) * _regionBlocks;

            return new Span<byte>(storage, checked((int)(baseBlock * blockBytes)), _regionBlocks * blockBytes);
        }

        public void Dispose()
        {
            // The arenas are managed arrays: releasing this mirror drops its references and the collector takes both sides.
        }
    }

    private static string FormatStats(string mode, List<double> values)
    {
        double median = Median(values);
        double min = values.Min();
        double max = values.Max();

        return $"kv-spike mode={mode} tok/s=[{string.Join(", ", values.Select(v => v.ToString("F3", CultureInfo.InvariantCulture)))}] " +
               $"median={median:F3} min={min:F3} max={max:F3} spread={(max - min) / median * 100.0:F2}%";
    }

    private static double Median(List<double> values)
    {
        List<double> sorted = [.. values];
        sorted.Sort();

        return sorted.Count % 2 == 1
            ? sorted[sorted.Count / 2]
            : (sorted[sorted.Count / 2 - 1] + sorted[sorted.Count / 2]) / 2.0;
    }

    private static double ElapsedMs(Stopwatch watch) => watch.Elapsed.TotalMilliseconds;

    // A prompt of exactly the requested token count from the repeated sentence every prefill harness uses: token values only
    // drive embedding lookups, so any in-vocabulary ids exercise the same weight and attention work.
    private static int[] BuildPrompt(GgufTokenizer tokenizer, int tokens)
    {
        const string line = "The quick brown fox jumps over the lazy dog while the patient cat watches the rain. ";

        List<int> ids = [];
        for (int i = 0; ids.Count < tokens && i < 4096; i++)
        {
            foreach (int id in tokenizer.Encode(line))
            {
                ids.Add(id);

                if (ids.Count == tokens)
                {
                    break;
                }
            }
        }

        return [.. ids.Take(tokens)];
    }

    private static int ParseInt(List<string> args, string key, int fallback)
    {
        int index = args.IndexOf(key);
        if (index < 0 || index + 1 >= args.Count || !int.TryParse(args[index + 1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int value))
        {
            return fallback;
        }

        return value;
    }

    private static IEnumerable<string> ParseList(List<string> args, string key, IEnumerable<string> fallback)
    {
        int index = args.IndexOf(key);
        if (index < 0 || index + 1 >= args.Count)
        {
            return fallback;
        }

        return [.. args[index + 1].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)];
    }
}
