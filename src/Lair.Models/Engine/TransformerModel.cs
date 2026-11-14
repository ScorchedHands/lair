using System.Buffers;
using System.Globalization;

using Lair.Contracts;
using Lair.Kv;
using Lair.Loader;

namespace Lair.Models.Engine;

/// <summary>
/// Forward engine for supported GGUF architectures (llama, qwen2): embedding lookup, per-layer norm / GQA
/// attention with RoPE and paged KV via <see cref="SequenceKv"/>, gated feed-forward, final norm, and an untied
/// output matrix or tied token-embedding rows for logits. The graph mirrors the reference structure: each layer
    /// normalizes its input stream (norm x weight under the plan's resolved norm strategy) before the QKV GEMVs; RoPE
    /// rotates q/k pairs at the absolute position with theta_i = base^(-2i/headDim), pairing adjacent dimensions
    /// (2i, 2i + 1) under the llama-family strategy or the head halves (i, i + headDim/2) under the qwen3 strategy;
    /// attention scores scale by 1/sqrt(headDim), pass through the plan's attention-score softcap when one is resolved, and
/// softmax runs online over positions 0..pos, narrowed per layer to <c>[max(0, pos - window + 1), pos]</c> on the
/// layers the plan's resolved window interleave masks and left unbounded on the layers it leaves alone. The narrowing
/// is a block selection lair.kv resolves for that read: masked positions are not addressed at all, while every
/// position's K/V is still written and stays readable for the layers whose own selection covers it. The residual added before the
/// FFN norm is the pre-attention stream value, or that stream plus a normalized attention output under a placement that
/// norms sublayer outputs — either way matching the reference graph's ffn_inp add; rope scaling variants
/// are not supported. FP32 buffers are allocated once at Open and reused across calls; packed matrix rows are
/// consumed through <see cref="WeightRef"/> without being materialized, except for embedding lookups and small
/// norm/bias vectors preloaded at Open. Attention-bias tensors follow the plan's resolved policy: under
/// <see cref="BiasPolicy.Declared"/> the resolved <c>use_bias</c> flag — a declaration or the profile's feature
/// default — makes the q/k/v bias vectors required or leaves every attention bias unloaded and unapplied, with one
/// structured log line naming the ignored tensors, while an output-projection bias is loaded whenever the file
/// carries one; the plan's resolved <see cref="QkNormMode"/> normalizes each q and k head between its projection
/// (plus any bias) and its rotation — per head scaled by the layer's head-dimension vector, over each side's whole
/// projected stream scaled by that side's full-width vector, normalized with the epsilon alone, or not at all. Under
/// <see cref="NormPlacement.PostSublayerOutputs"/> every sublayer output is normalized once more with its own vector before
/// it joins the residual stream, which makes each layer's post-sublayer norm tensors required; a pre-only placement reads
/// none and rejects a file that carries one. A resolved
/// final-logit softcap saturates every logit vector — untied output head or tied token-embedding rows — after
/// the full vector is computed and before any softmax, sampling, or copy-out. A resolved input-embedding gain
/// multiplies every freshly dequantized embedding row by its value before the first layer on every forward path.
/// A resolved sparse-FFN descriptor replaces this block's dense gate/up/down triplet with an expert-routed block
/// whose every expert is resident: the router matrix scores all experts, top-k selection keeps the
/// <c>&lt;prefix&gt;expert_used_count</c> best-scoring experts — equal scores resolve to the lower expert index — and
/// renormalizes the selected routing weights by their own sum when the descriptor declares normalization; each
/// selected expert's gate/up/down GEMVs then run over the same normalized stream in ascending expert-index order and
/// accumulate into the residual scaled by its weight. A declared shared expert adds one dense gate/up/down block to
/// that accumulation once per token, an undeclared shared expert reads no sparse role tensor at all, and a file
    /// declaring none keeps the dense triplet path unchanged. Sparse routing runs on the decode token path and on both
/// batched forwards — <see cref="DecodeBatch"/> and the batched rows of <see cref="Prefill"/> — where each row routes
/// over its own normalized stream; <see cref="Encode"/> forwards dense blocks only and fails naming the descriptor when
/// a model carries one.
/// </summary>

public sealed class TransformerModel : IDisposable
{
    /// <summary>Row ceiling of the engine's batched decode forwards: the largest number of sequences a single <see cref="DecodeBatch"/> call may advance. Prompt windows carry their own resolved row count — see <see cref="ModelPlan.PrefillRowWidth"/>.</summary>
    public const int MaxBatchSequences = 16;

    /// <summary>Ceiling on the prefill row width an engine may resolve: the widest prompt sub-batch its batched streams are ever sized for. A resolved count outside <c>1..MaxPrefillRowWidth</c> is refused at open.</summary>
    public const int MaxPrefillRowWidth = 64;

    /// <summary>The row width a plan resolves when nothing says otherwise: the measured balance point across both reference shapes. It recovers most of the narrower default's deficit on the wider-hidden shape while staying within measurement noise of the narrower default on the other, and it avoids doubling to the engine ceiling where one fixture gains only slightly at twice the activation bytes.</summary>
    public const int DefaultPrefillRowWidth = 32;

    private readonly IBackend _backend;
    private readonly IWeightSource _weights;

    // The resolved prefill kernel selection: pinned at open, re-pointed only through ApplyPrefillKernel, and read
    // fresh at every window boundary so a hot edit lands from the next window — the same re-read discipline the
    // scheduler's chunk width keeps. Volatile because the apply path writes it from a caller thread while worker
    // threads consume it per forward.
    private volatile PrefillKernel _prefillKernel;

    // The quant formats this engine's dense weight matrices carry, resolved once from the plan's tensor map in role
    // order with repeats collapsed. It is what a blocked-matmul selection needs kernels for: every format it lists has
    // to have one before that selection can run this file, and they are the names an unsupported file reports back.
    private readonly IReadOnlyList<QuantFormat> _matmulFormats;

    // Only a source this instance created is its own to release: a caller-supplied source stays under the caller's
    // lifetime rules and must outlive the model.
    private readonly bool _ownsWeights;

    // Kept for tensor name lookups only: the mmap source owns payload access.
    private readonly LogicalTensorTable _fileTensors;

    // Fused qkv tensors the split role map resolved past, and the companion scales of nameable tensors, are
    // recorded at Open: the fused ones are reported as one structured line, the companion ones validated.
    private readonly List<string> _ignoredFusedTensors = [];
    private HashSet<string>? _nameableTensors;

    // Per-layer packed matrix tensors, resolved from the mmap view on each use so no dequantized copy is kept. The q/k/v
    // trio stays empty when the plan resolves a fused buffer instead: those roles then read slices of one tensor per layer.
    private LogicalTensor[] _wq = [];
    private LogicalTensor[] _wk = [];
    private LogicalTensor[] _wv = [];
    private readonly LogicalTensor[] _wo;
    private LogicalTensor[] _ffnGate = []; // empty when a sparse descriptor replaces the dense block, or when the gate half rides inside the up tensor
    private LogicalTensor[] _ffnUp = []; // empty when a sparse descriptor replaces the dense block

    // Fused-layout slice geometry, pinned once at Open: one packed buffer per layer carrying every head of q, then k,
    // then v; each role's byte range inside that payload and the two-dimensional shape a GEMV over its slice carries.
    // Byte lengths are per layer because a converted file is free to hold different quant tiers in this tensor at
    // different layers. The gate half of a fused gate/up buffer needs only its own block size: both halves share one
    // format, so the up side starts exactly one block after the gate side. Empty arrays mean neither layout runs.
    private LogicalTensor[] _wqkv = [];
    private TensorShape _qkvQueryShape = new([0, 0]);
    private TensorShape _qkvKeyValueShape = new([0, 0]);
    private long[] _qkvQueryBytes = [];
    private long[] _qkvKeyBytes = [];
    private TensorShape _gateUpHalfShape = new([0, 0]);
    private long[] _gateUpHalfBytes = [];
    private LogicalTensor[] _ffnDown = []; // empty when a sparse descriptor replaces the dense block

    // Sparse-FFN block state, resolved once at Open from the plan's descriptor; empty arrays and zero geometry mean
    // the dense gate/up/down triplet runs. The router tensor is [experts x hidden]; each expert matrix role is one
    // stacked tensor whose logical shape carries the expert count as its outermost dimension, so expert e's block is
    // the contiguous row range [e * rows .. (e + 1) * rows) and a per-expert GEMV reads it through a slice of that
    // role's payload with the fixed two-dimensional shape below. Every expert stays resident: routing selects among
    // these tensors instead of fetching them.
    private readonly SparseFfnDescriptor? _sparseFfn;
    private LogicalTensor[] _moeRouter = [];
    private LogicalTensor[] _moeExpertGate = [];
    private LogicalTensor[] _moeExpertUp = [];
    private LogicalTensor[] _moeExpertDown = [];
    private LogicalTensor[] _sharedGate = [];
    private LogicalTensor[] _sharedUp = [];
    private LogicalTensor[] _sharedDown = [];

    // The shared expert's per-token selector rows, dequantized once at load exactly like the norm and bias vectors.
    // One row of the hidden size per layer when the file ships them for every layer; an empty array when it ships
    // none, which is what leaves the shared contribution ungated.
    private float[][] _sharedInputGate = [];

    // Per-expert slice geometry: one expert's rows and cols as the shape a GEMV over its slice carries, plus that
    // slice's packed byte block per role — indexed by layer, because a converted file is free to hold different quant
    // tiers in the same role at different layers. Built once at Open so routing and expert execution allocate nothing
    // per token; empty means no sparse block runs.
    private TensorShape _moeExpertGateShape = new([0, 0]);
    private TensorShape _moeExpertUpShape = new([0, 0]);
    private TensorShape _moeExpertDownShape = new([0, 0]);
    private long[] _moeExpertGateBytes = [];
    private long[] _moeExpertUpBytes = [];
    private long[] _moeExpertDownBytes = [];

    // Routing scratch, sized at Open for descriptor-bearing models only. The token path consumes one
    // routing at a time; the batch path stores every sequence's selection flat, sequence by sequence, so
    // each expert group can pack its member rows into one batched gate/up/down pass instead of paying a
    // batch call per member. Selection order is ascending expert index, which is the order the batch path
    // groups by.

    private float[] _moeLogits = [];      // experts: router scores for one sequence, consumed during selection
    private int[] _moeIds = [];           // routedExpertsPerToken: selected expert indexes, ascending order
    private float[] _moeWeights = [];     // routedExpertsPerToken: the weight of each selected expert
    private int[] _batchMoeIds = [];      // _rowCapacity * routedExpertsPerToken: per-sequence selections
    private float[] _batchMoeWeights = [];// same length, aligned with _batchMoeIds
    private int[] _expertMembers = []; // _rowCapacity: member rows of the expert under execution
    private float[] _moeSubsetIn = []; // _rowCapacity * hidden: one expert group's activations and down results, packed in place

    // Small FP32 vectors preloaded at Open: per-layer norm weights and any biases present in the file. Null
    // entries mean the tensor is absent (llama files carry no attention bias, qwen2 files may).
    private readonly float[][] _attnNormWeight;
    private readonly float[][] _ffnNormWeight;
    private readonly float[]?[] _biasQ;
    private readonly float[]?[] _biasK;
    private readonly float[]?[] _biasV;
    private readonly float[]?[] _biasO;

    // Per-head QK-norm weight vectors (one headDim vector per layer, shared by every head of its side),
    // preloaded only when the plan resolves a weighted mode: every head normalizes its own headDim slice.
    private readonly float[][] _attnQNorm;
    private readonly float[][] _attnKNorm;

    // Post-sublayer norm weight vectors, preloaded only when the plan places norms on sublayer outputs: each holds one
    // hidden-sized vector per layer and normalizes that sublayer's own output before its residual add. A pre-only
    // placement loads neither and rejects a file carrying either tensor.
    private readonly float[][] _postAttnNorm;
    private readonly float[][] _postFfnNorm;

    // Norm-bias vectors preloaded only when the plan declares them: the LayerNorm path adds them after the
    // weighted normalization. Null entries mean the layer's tensor is absent.
    private readonly float[]?[] _attnNormBias;
    private readonly float[]?[] _ffnNormBias;
    private readonly float[]? _outputNormBias;

    private readonly LogicalTensor _tokenEmbd;
    private readonly LogicalTensor? _output;
    private readonly float[] _outputNormWeight;
    private readonly float[]? _outputBias;

    // FP32 buffers reused across every forward call.
    private readonly float[] _cur;     // hidden: layer input stream (residual carrier) and FFN output accumulator
    private readonly float[] _normed;  // hidden: weighted norm before the QKV / FFN GEMVs
    private readonly float[] _q;       // headCount * headDim
    private readonly float[] _k;       // kvHeads * headDim (also the KV slot write source)
    private readonly float[] _v;       // kvHeads * headDim
    private readonly float[] _attnOut; // headCount * headDim: heads concatenated in order for wo GEMV
    private readonly float[] _proj;    // hidden: attn_output / ffn_down GEMV target before the residual add
    private readonly float[] _gateUp;  // intermediateSize: gate result, then activation(gate) * up written back into it
    private readonly float[] _up;      // intermediateSize: up GEMV result
    private readonly float[] _logits;  // vocabSize

    // Batched decode buffers: sequence-major blocks of _rowCapacity copies of the single-token streams, allocated once
    // at Open so a batched step adds no per-step heap pressure. Each sequence's block mirrors the same-sized buffer in
    // its own single-token forward; a wider prefill window widens these allocations and nothing else.
    private readonly int _rowCapacity;

    private readonly float[] _batchCur;     // _rowCapacity * hidden
    private readonly float[] _batchNormed;  // _rowCapacity * hidden
    private readonly float[] _batchQ;       // _rowCapacity * headCount * headDim
    private readonly float[] _batchK;       // _rowCapacity * kvHeads * headDim
    private readonly float[] _batchV;       // _rowCapacity * kvHeads * headDim
    private readonly float[] _batchAttnOut; // _rowCapacity * headCount * headDim
    private readonly float[] _batchProj;    // _rowCapacity * hidden
    private readonly float[] _batchGateUp;  // _rowCapacity * intermediateSize
    private readonly float[] _batchUp;      // _rowCapacity * intermediateSize
    private readonly float[] _batchLogits;  // _rowCapacity * vocabSize

    // Row-tiled prefill attention state: one online-softmax entry per head for every row of a sub-batch — running max,
    // then weight sum — held across the tiled walk so each row advances its own state in exactly the order its per-row
    // kernel would. Sized once at Open for the widest window; the accumulators ride on _batchAttnOut's per-row head
    // regions, so a tiled window allocates nothing per call.
    private readonly float[] _tiledRowMax;    // _rowCapacity * headCount: every row's per-head online max across the walk
    private readonly float[] _tiledRowSumExp; // same shape: every row's per-head weight sum across the walk

    private readonly float[] _cos;          // maxContext * rope pair count
    private readonly float[] _sin;
    private readonly float[]? _cosWindowed; // the same shape again at a second base, or null when every layer shares one
    private readonly float[]? _sinWindowed;
    private readonly int _ropeHalf;         // pairs the rotation covers: the declared slice, or the full head
    private readonly float _attnScale;      // head-dimension score gain times the variant's attention factor

    /// <summary>The resolved model plan this engine forwards under: geometry, profile tensor map, and strategy values.</summary>
    public ModelPlan Plan { get; }

    /// <summary>Flat scalar view of <see cref="Plan"/>, kept as the terse read surface for geometry consumers.</summary>
    public ModelDesc Desc { get; }

    /// <summary>Largest sequence length this model can forward: the extent of its precomputed RoPE tables.</summary>
    public int MaxContextTokens { get; }

    /// <summary>The prefill kernel selection this engine forwards prompt windows through: resolved at open and re-read at every window boundary.</summary>
    public PrefillKernel ResolvedPrefillKernel => _prefillKernel;

    /// <summary>The row count this engine's prompt windows forward in, held on the plan and read fresh at every prefill window boundary.</summary>
    public int ResolvedPrefillRowWidth => Plan.PrefillRowWidth;

    /// <summary>
    /// Re-points the engine's prefill path selection. The new value is read fresh at the next prefill window
    /// boundary, so sequences already mid-window finish that window on the route they started with — the same
    /// boundary discipline a chunk-width edit keeps. A selection this build has no kernel for is refused naming it
    /// and what is missing; nothing falls back to the route that works. The dense-matrix selection is also refused
    /// when one of this engine's own weight formats has no blocked matmul, naming that format.
    /// </summary>
    public void ApplyPrefillKernel(PrefillKernel kernel)
    {
        if (!KernelExecutes(kernel))
        {
            throw KernelRefusal(kernel);
        }

        _prefillKernel = kernel;
    }

    /// <summary>The quant formats this engine's dense weight matrices carry, deduplicated in role order: the formats a blocked matmul needs kernels for before the dense-matrix selection can run this file.</summary>
    public IReadOnlyList<QuantFormat> MatmulWeightFormats => _matmulFormats;

    // Whether a selection has kernels for this engine's own weights. The batched-row window and its row-tiled variant
    // always do; the dense-matrix window does only when every dense weight matrix it multiplies activation rows
    // against is a format the backend's blocked matmul executes. Decided from the same list an unsupported file names.
    private bool KernelExecutes(PrefillKernel kernel) => kernel switch
    {
        PrefillKernel.GemV or PrefillKernel.Tiled => true,
        PrefillKernel.Gemm => FormatsWithoutKernels().Count is 0 && _matmulFormats.Count > 0,
        _ => false,
    };

    // The dense weight formats this engine carries that its backend has no blocked matmul for. Empty when every one of
    // them runs through the dense-matrix window; otherwise exactly the names a refusal reports.
    private List<QuantFormat> FormatsWithoutKernels()
    {
        List<QuantFormat> missing = [];
        foreach (QuantFormat format in _matmulFormats)
        {
            if (!_backend.SupportsGemm(format))
            {
                missing.Add(format);
            }
        }

        return missing;
    }

    // The one refusal wording an unrunnable prefill selection gets, at open and on apply alike: it names the value, what
    // the forward engine lacks for it, and the selections that execute today. For the dense-matrix selection it also
    // names this file's own weight formats when they are the reason the selection cannot run.
    private ModelException KernelRefusal(PrefillKernel kernel)
    {
        string name = PrefillKernelNames.Of(kernel);
        List<QuantFormat> missing = kernel == PrefillKernel.Gemm ? FormatsWithoutKernels() : [];

        if (missing.Count > 0)
        {
            return new ModelException(
                $"prefill kernel '{name}' is not implemented by the forward engine: architecture '{Plan.Architecture}' " +
                $"carries dense weight matrices of format(s) {string.Join(", ", missing)} and no {name} attention/weight path exists in this build for them — " +
                $"the window shape that multiplies a whole sub-batch against one matrix runs only where every dense weight matrix is {QuantFormat.Q8_0}, {QuantFormat.Q4_K}, or {QuantFormat.IQ4_XS} today, " +
                $"while every other family forwards its prompt windows through batched row passes — set engine.prefill_kernel to '{PrefillKernelNames.GemV}' or " +
                $"'{PrefillKernelNames.Tiled}' (or omit it) to run this file");
        }

        return new ModelException(
            $"prefill kernel '{name}' is not implemented by the forward engine: architecture " +
            $"'{Plan.Architecture}' selects its prompt windows through this selection and no {name} attention/weight path exists in this build — only " +
            $"{PrefillKernelNames.GemV} and {PrefillKernelNames.Tiled} execute today — set engine.prefill_kernel to '{PrefillKernelNames.GemV}' or " +
            $"'{PrefillKernelNames.Tiled}' (or omit it) to run this file");
    }

    // The quant format of every dense weight matrix a prompt window multiplies activation rows against: the attention
    // trio — or one fused qkv buffer, whichever layout the plan resolves — the attention output, and the feed-forward
    // block's gate/up/down as its layout names them. Deduplicated in role order because a family that keeps one format
    // across every layer should report it once. A sparse-FFN block contributes nothing here: its expert matrices stay on
    // batched row passes whatever selection is resolved, so they do not constrain which window shape can run.
    private static IReadOnlyList<QuantFormat> ResolveMatmulFormats(ModelPlan plan, LogicalTensorTable tensors)
    {
        List<QuantFormat> formats = [];

        void AddRole(TensorRole role)
        {
            if (plan.Tensors[role] is not TensorDescriptor descriptor)
            {
                return;
            }

            for (int l = 0; l < plan.LayerCount; l++)
            {
                if (descriptor.Resolve(tensors, l) is LogicalTensor tensor && !formats.Contains(tensor.Format))
                {
                    formats.Add(tensor.Format);
                }
            }
        }

        if (plan.AttentionWeights == QkvLayout.FusedBufferSlices)
        {
            AddRole(TensorRole.AttentionQkv);
        }
        else
        {
            AddRole(TensorRole.AttentionQuery);
            AddRole(TensorRole.AttentionKey);
            AddRole(TensorRole.AttentionValue);
        }

        AddRole(TensorRole.AttentionOutput);

        if (plan.SparseFfn is null)
        {
            if (plan.GateSource == FeedForwardGateSource.UpTensorHalves)
            {
                AddRole(TensorRole.FeedForwardUp); // the gate half rides inside this buffer, so one format covers both sides
            }
            else
            {
                AddRole(TensorRole.FeedForwardGate);
                AddRole(TensorRole.FeedForwardUp);
            }

            AddRole(TensorRole.FeedForwardDown);
        }

        return formats;
    }

    /// <summary>Opens the engine over a memory-mapped view of the file's own tensor payloads.</summary>
    public static TransformerModel Open(GgufFile file, IBackend backend, int maxContextTokens) =>
        Open(file, backend, maxContextTokens, logLine: null);

    /// <summary>
    /// Opens the engine over a memory-mapped view of the file's own tensor payloads and reports profile-
    /// resolution events (the generic llama-shaped fallback applied to an unknown architecture) through
    /// <paramref name="logLine"/>.
    /// </summary>
    public static TransformerModel Open(GgufFile file, IBackend backend, int maxContextTokens, Action<string>? logLine) =>
        new(file, backend, maxContextTokens, null, logLine);

    /// <summary>
    /// Opens the engine over a memory-mapped view of the file's own tensor payloads with an explicit prefill kernel
    /// selection: every prompt window forwards through that route. A selection this build has no kernel for is
    /// refused at open naming it and what is missing — never fallen back to the kernel that happens to work.
    /// </summary>
    public static TransformerModel Open(GgufFile file, IBackend backend, int maxContextTokens, Action<string>? logLine, PrefillKernel prefillKernel) =>
        new(file, backend, maxContextTokens, weights: null, logLine, profileOverride: null, prefillKernel);

    /// <summary>
    /// Opens the engine over a memory-mapped view of the file's own tensor payloads with an explicit prefill kernel
    /// selection and the route its placement resolves to: the resolved segments land on the plan before any allocation, so
    /// what step composition reads is exactly what load decided. A route this build cannot execute — any segment not on a
    /// single DRAM-resident span of the device <paramref name="backend"/> reports — is refused at open naming that segment and
    /// what is missing, never run somewhere else behind the selection that happens to work.
    /// </summary>
    public static TransformerModel Open(GgufFile file, IBackend backend, int maxContextTokens, Action<string>? logLine, PrefillKernel prefillKernel, ModelRoute route) =>
        new(file, backend, maxContextTokens, weights: null, logLine, profileOverride: null, prefillKernel, prefillRowWidth: 0, route);

    /// <summary>
    /// Opens the engine with weight payloads supplied by an alternate <see cref="IWeightSource"/>: every tensor read
    /// goes through that source instead of a mmap view. Tensor metadata (names, shapes, formats, offsets) still comes
    /// from <paramref name="file"/>, and the caller keeps ownership of the source — it must outlive the model and is
    /// disposed by the caller, not here.
    /// </summary>
    public static TransformerModel Open(GgufFile file, IWeightSource weights, IBackend backend, int maxContextTokens) =>
        Open(file, weights, backend, maxContextTokens, logLine: null);

    /// <summary>
    /// Opens the engine with an alternate <see cref="IWeightSource"/> and reports profile-resolution events (the
    /// generic llama-shaped fallback applied to an unknown architecture) through <paramref name="logLine"/>.
    /// </summary>
    public static TransformerModel Open(GgufFile file, IWeightSource weights, IBackend backend, int maxContextTokens, Action<string>? logLine) =>
        new(file, backend, maxContextTokens, weights, logLine);

    /// <summary>
    /// Opens the engine with an explicitly supplied <see cref="ModelProfile"/> resolving the plan instead of the
    /// registry selection: the profile's prefix supplies every metadata key and its strategy values drive the
    /// forward, with profile- and tensor-resolution events reported through <paramref name="logLine"/>. Assembly-internal
    /// so no host configuration path can reach it.
    /// </summary>
    internal static TransformerModel Open(GgufFile file, IBackend backend, int maxContextTokens, ModelProfile profile, Action<string>? logLine = null) =>
        new(file, backend, maxContextTokens, weights: null, logLine, profile);

    /// <summary>
    /// Opens the engine over a memory-mapped view of the file's own tensor payloads with an explicit prefill row
    /// width and kernel selection resolved onto the plan before any allocation: prompt windows forward in sub-batches
    /// of exactly that many rows while decode keeps its own batch ceiling, and batched streams size for whichever
    /// family needs them most. A row count outside <c>1..MaxPrefillRowWidth</c> is refused at open naming it and the
    /// range; zero keeps the plan's resolved default in force. Assembly-internal so no host configuration path can
    /// reach it — tests and bench resolve candidate widths through this seam.
    /// </summary>
    internal static TransformerModel Open(
        GgufFile file, IBackend backend, int maxContextTokens, Action<string>? logLine, PrefillKernel prefillKernel, int prefillRowWidth) =>
        new(file, backend, maxContextTokens, weights: null, logLine, profileOverride: null, prefillKernel, prefillRowWidth);

    /// <summary>
    /// The alternate-<see cref="IWeightSource"/> form of the explicit row-width seam: same resolution as the memory-mapped
    /// one, with every tensor read going through the supplied source and its ownership staying under the caller's rules.
    /// </summary>
    internal static TransformerModel Open(
        GgufFile file, IWeightSource weights, IBackend backend, int maxContextTokens, PrefillKernel prefillKernel, int prefillRowWidth) =>
        new(file, backend, maxContextTokens, weights, logLine: null, profileOverride: null, prefillKernel, prefillRowWidth);

    // The route this engine can execute today: one DRAM-resident segment on the device its opened backend reports. A wider
    // route is real material — split points and accelerator segments are decided where more backends exist to run them, and a
    // resolved plan may carry one — but opening an engine over it refuses naming the first segment this build has no backend
    // for rather than advancing part of the model on some device that happens to work. Nothing here executes a route: the check
    // lands before any weight view is spent on a shape that cannot run, exactly as an unrunnable kernel selection does.
    private void ValidateRoute()
    {
        ModelRoute route = Plan.Route;

        for (int index = 0; index < route.Segments.Count; index++)
        {
            DeviceSegment segment = route.Segments[index];

            if (!string.Equals(segment.DeviceId, _backend.DeviceId, StringComparison.Ordinal))
            {
                throw new ModelException(
                    $"{segment.Describe(index)} names a device this engine has no backend for: the opened compute backend reports " +
                    $"'{_backend.DeviceId}' and architecture '{Plan.Architecture}' resolves a route of {route.Segments.Count} segment(s) — " +
                    $"a crossing to '{segment.DeviceId}' is refused rather than run on CPU");
            }

            if (segment.WeightTier != WeightTier.Dram)
            {
                throw new ModelException(
                    $"{segment.Describe(index)} holds its weights at tier '{segment.WeightTier}': the forward engine reads " +
                    $"DRAM-resident weights only, so architecture '{Plan.Architecture}' is refused rather than run against a residency it cannot address");
            }
        }
    }

    private TransformerModel(GgufFile file, IBackend backend, int maxContextTokens, IWeightSource? weights = null, Action<string>? logLine = null, ModelProfile? profileOverride = null, PrefillKernel prefillKernel = PrefillKernel.GemV, int prefillRowWidth = 0, ModelRoute? route = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxContextTokens);

        _backend = backend;
        Plan = profileOverride is null ? ModelPlans.Resolve(file, logLine) : ModelPlans.Resolve(file, profileOverride, logLine);
        Desc = Plan.ToDesc();
        MaxContextTokens = maxContextTokens;

        // The formats this file's dense weight matrices carry, read from the plan's tensor map before any payload is
        // touched: it decides whether a selection that multiplies a whole sub-batch against one matrix can run here at
        // all, and it names the formats a refusal reports.
        _matmulFormats = ResolveMatmulFormats(Plan, file.Tensors);

        // A prefill kernel selection this engine has no kernel for is refused at open, the posture every other
        // unimplemented capability takes: the selection stays visible in the resolved value's vocabulary, the
        // refusal names it and what is missing, and nothing falls back to the route that happens to work. The batched-
        // row window and its row-tiled variant run for every family; the dense-matrix shape runs only where this file's
        // own weight formats all have a blocked matmul. The check lands before any tensor maps are built, so an
        // unrunnable selection never spends a weight view on it.
        if (!KernelExecutes(prefillKernel))
        {
            throw KernelRefusal(prefillKernel);
        }

        _prefillKernel = prefillKernel;

        // A resolved row count outside the engine's allocation ceiling is refused at open naming it and the range, the
        // posture every other unrunnable forward shape takes: nothing silently sizes some other width or forwards a
        // different sub-batch than the plan names. Zero means no explicit resolution — the plan's own value stays in
        // force, which for every file resolved today is the default row count. The check lands before any tensor maps,
        // so an unusable width never spends a weight view on it.
        if (prefillRowWidth != 0 && prefillRowWidth is not (> 0 and <= MaxPrefillRowWidth))
        {
            throw new ModelException(
                $"prefill row width '{prefillRowWidth}' is outside the range 1..{MaxPrefillRowWidth} this engine sizes its batched streams for: " +
                $"prompt windows forward in sub-batches of at most that many rows — omit it to keep the plan's resolved {Plan.PrefillRowWidth}.");
        }

        Plan = prefillRowWidth == 0 ? Plan : Plan with { PrefillRowWidth = prefillRowWidth };

        // The route a host resolved from its own placement lands on the plan before any allocation, so what step composition
        // reads later is exactly what load decided. A file opened without one keeps the single CPU segment its own resolution
        // put there — which is where every model served today runs.
        Plan = route is null ? Plan : Plan with { Route = route };

        ValidateRoute();

        // Row capacity of every batched stream: whichever forward family needs the wider sub-batch — never below what a
        // decode batch may advance, and above that only when the plan resolves wider prompt windows. Allocation is the
        // one thing width moves; no arithmetic in any forward reads it as an operand.
        _rowCapacity = Math.Max(MaxBatchSequences, Plan.PrefillRowWidth);

        // Only now does the weight view get created: a refused kernel selection or row count never spends one, which is
        // what keeps a failed open from holding the file's mmap alive behind its own refusal.
        _ownsWeights = weights is null;
        _weights = weights ?? new GgufMmapWeightSource(file);
        _fileTensors = file.Tensors;

        if (Plan.HeadDim % 2 != 0)
        {
            throw new ModelException($"head dimension {Plan.HeadDim} is odd; RoPE pairs require an even head dimension");
        }

        // A resolved sparse-FFN descriptor replaces this block's dense gate/up/down triplet with expert-routed
        // execution: the dense feed-forward roles are neither required nor loaded, and any dense feed-forward tensor
        // such a file still ships is reported as ignored. Every declared expert loads here — all experts of every
        // layer stay resident in the mmap view for the model's lifetime, so forward-time routing selects among them
        // instead of fetching them, and no residency machinery runs on this path. The dense gate/up/down block stays
        // exactly as it was for a file that declares no expert count.
        _sparseFfn = Plan.SparseFfn;

        // A resolved strategy other than the dense-resident variant is a shape this engine has no kernel for: the plan
        // can represent it, so the load records it and reports here rather than running some other variant behind the
        // selection the file declared. The message names the strategy, what the descriptor carries for it, and the one
        // value that runs today — nothing falls back silently.
        if (Plan.SparseFfnStrategy is { } resolvedStrategy && resolvedStrategy != SparseFfnStrategy.DenseResident)
        {
            throw new ModelException(
                $"sparse-FFN strategy '{resolvedStrategy}' is not implemented by the forward engine: architecture " +
                $"'{Plan.Architecture}' declares a sparse-FFN block with {_sparseFfn!.ExpertCount} experts routing " +
                $"{_sparseFfn.RoutedExpertsPerToken} per token over a shared basis at rank {_sparseFfn.Rank?.ToString(CultureInfo.InvariantCulture) ?? "an undeclared rank"}: " +
                $"only {SparseFfnStrategy.DenseResident} executes today — declare {Plan.Prefix}sparse_ffn_strategy 'dense_resident' to run this file");
        }

        int layers = Plan.LayerCount;

        // The plan's resolved q/k/v layout decides which tensors this block reads before any of them load: three matrices
        // the map names one role each, or one fused buffer per layer whose head regions each projection slices for itself.
        // Under the split layout a fused tensor the file still carries is recorded for the structured line naming ignored
        // material; under the fused layout it is the other side's tensors that go unread.
        if (Plan.AttentionWeights == QkvLayout.FusedBufferSlices)
        {
            LoadFusedQkvBuffers(layers);
            ReportIgnoredTensors(logLine, "split_qkv_ignored", file, [TensorRole.AttentionQuery, TensorRole.AttentionKey, TensorRole.AttentionValue]);
        }
        else
        {
            _wq = LoadMatrices(layers, TensorRole.AttentionQuery, FusedQkvName(TensorRole.AttentionQuery));
            _wk = LoadMatrices(layers, TensorRole.AttentionKey, FusedQkvName(TensorRole.AttentionKey));
            _wv = LoadMatrices(layers, TensorRole.AttentionValue, FusedQkvName(TensorRole.AttentionValue));
        }

        _wo = LoadMatrices(layers, TensorRole.AttentionOutput);
        if (_sparseFfn is null)
        {
            // The same rule on the feed-forward side: a block whose gate half rides inside its up tensor loads one buffer
            // per layer and reads each half as a slice, while a block with a gate matrix of its own loads the triplet. A
            // gate tensor a halves-form file still ships is reported rather than loaded.
            if (Plan.GateSource == FeedForwardGateSource.UpTensorHalves)
            {
                LoadFusedGateUpBuffers(layers);
                ReportIgnoredTensors(logLine, "ffn_gate_ignored", file, [TensorRole.FeedForwardGate]);
            }
            else
            {
                _ffnGate = LoadMatrices(layers, TensorRole.FeedForwardGate);
                _ffnUp = LoadMatrices(layers, TensorRole.FeedForwardUp);
            }

            _ffnDown = LoadMatrices(layers, TensorRole.FeedForwardDown);
        }
        else
        {
            int experts = _sparseFfn.ExpertCount;

            // The router emits one score per declared expert from the hidden stream. Each stacked role is one tensor
            // whose logical shape carries that expert count as its outermost dimension followed by this matrix's rows
            // and columns, so a GEMV over one expert reads a contiguous row range of the stack: gate/up experts emit
            // intermediateSize values from hidden inputs, down experts emit hidden values from intermediate inputs.
            _moeRouter = LoadMatrices(layers, _sparseFfn.RouterRole);
            RequireShape(_moeRouter, experts, Plan.HiddenSize);

            LogicalTensor[] expertGate = LoadExpertStacks(layers, _sparseFfn.ExpertGateRole, experts, Plan.IntermediateSize, Plan.HiddenSize);
            LogicalTensor[] expertUp = LoadExpertStacks(layers, _sparseFfn.ExpertUpRole, experts, Plan.IntermediateSize, Plan.HiddenSize);
            LogicalTensor[] expertDown = LoadExpertStacks(layers, _sparseFfn.ExpertDownRole, experts, Plan.HiddenSize, Plan.IntermediateSize);

            // Per-expert slice geometry for these roles: one expert's packed block and the two-dimensional shape a
            // GEMV over that slice carries, per layer — each role sizes its slice from that layer's own tensor format
            // and row stride, because a converted file is free to hold different quant tiers in one role at different
            // layers (the reference converter does exactly that), while LoadExpertStacks has already pinned every
            // layer of a role to the same rows and columns.
            _moeExpertGate = expertGate;
            _moeExpertUp = expertUp;
            _moeExpertDown = expertDown;
            long[] gateBytes = new long[layers];
            long[] upBytes = new long[layers];
            long[] downBytes = new long[layers];
            for (int l = 0; l < layers; l++)
            {
                gateBytes[l] = Plan.IntermediateSize * expertGate[l].Shape.RowStrideBytes(expertGate[l].Format);
                upBytes[l] = Plan.IntermediateSize * expertUp[l].Shape.RowStrideBytes(expertUp[l].Format);
                downBytes[l] = Plan.HiddenSize * expertDown[l].Shape.RowStrideBytes(expertDown[l].Format);
            }

            _moeExpertGateShape = new TensorShape([Plan.IntermediateSize, Plan.HiddenSize]);
            _moeExpertUpShape = new TensorShape([Plan.IntermediateSize, Plan.HiddenSize]);
            _moeExpertDownShape = new TensorShape([Plan.HiddenSize, Plan.IntermediateSize]);
            _moeExpertGateBytes = gateBytes;
            _moeExpertUpBytes = upBytes;
            _moeExpertDownBytes = downBytes;

            // Routing scratch and packed-expert storage sized from the descriptor's concrete counts: one selection at
            // a time for routing plus per-sequence copies of every batched selection, member-row indexes for one
            // expert group, and one hidden-sized buffer carrying each group's activations and down results in place.
            _moeLogits = new float[_sparseFfn.ExpertCount];
            _moeIds = new int[_sparseFfn.RoutedExpertsPerToken];
            _moeWeights = new float[_sparseFfn.RoutedExpertsPerToken];
            _batchMoeIds = new int[_rowCapacity * _sparseFfn.RoutedExpertsPerToken];
            _batchMoeWeights = new float[_rowCapacity * _sparseFfn.RoutedExpertsPerToken];
            _expertMembers = new int[_rowCapacity];
            _moeSubsetIn = new float[_rowCapacity * Plan.HiddenSize];

            // A declared shared expert is one dense-shaped matrix triplet the routing adds to every token's output,
            // plus — for a family whose reference gates it — one selector vector per layer. An undeclared shared
            // block loads and reads nothing here at all.
            if (_sparseFfn.HasSharedExpert)
            {
                TensorRole gateRole = RequiredSparseRole(_sparseFfn.SharedGateRole, "gate");
                TensorRole upRole = RequiredSparseRole(_sparseFfn.SharedUpRole, "up");
                TensorRole downRole = RequiredSparseRole(_sparseFfn.SharedDownRole, "down");

                _sharedGate = LoadMatrices(layers, gateRole);
                _sharedUp = LoadMatrices(layers, upRole);
                _sharedDown = LoadMatrices(layers, downRole);
                RequireShape(_sharedGate, Plan.IntermediateSize, Plan.HiddenSize);
                RequireShape(_sharedUp, Plan.IntermediateSize, Plan.HiddenSize);
                RequireShape(_sharedDown, Plan.HiddenSize, Plan.IntermediateSize);

                // The reference also ships one selector vector of the hidden size per layer for the shared expert and
                // takes this token's score from it. A file that carries none of them stays ungated; a table carrying
                // them for only part of the block fails here rather than gating some layers and not others.
                if (_sparseFfn.SharedInputGateRole is { } inputGateRole)
                {
                    _sharedInputGate = LoadSharedExpertSelectors(layers, Plan.Tensors[inputGateRole]);
                }
            }

            if (logLine is not null)
            {
                ReportIgnoredTensors(logLine, "dense_ffn_ignored", file, [TensorRole.FeedForwardGate, TensorRole.FeedForwardUp, TensorRole.FeedForwardDown]);
            }
        }

        // Matrix shapes must agree with the declared dimensions: rows are the output side, columns the input side. A fused
        // buffer validates its own region geometry against those same declarations when it loads, so only the tensors this
        // layout reads one role each get checked here.
        if (Plan.AttentionWeights == QkvLayout.SeparateTensors)
        {
            RequireShape(_wq, (long)Plan.HeadCount * Plan.HeadDim, Plan.HiddenSize); // q emits headCount * head_dim
            RequireShape(_wk, (long)Plan.KvHeadCount * Plan.HeadDim, Plan.HiddenSize);
            RequireShape(_wv, (long)Plan.KvHeadCount * Plan.HeadDim, Plan.HiddenSize);
        }

        RequireShape(_wo, Plan.HiddenSize, (long)Plan.HeadCount * Plan.HeadDim);
        if (_sparseFfn is null && Plan.GateSource == FeedForwardGateSource.SeparateGateMatrix)
        {
            RequireShape(_ffnGate, Plan.IntermediateSize, Plan.HiddenSize);
            RequireShape(_ffnUp, Plan.IntermediateSize, Plan.HiddenSize);
        }

        if (_sparseFfn is null)
        {
            RequireShape(_ffnDown, Plan.HiddenSize, Plan.IntermediateSize);
        }

        // Quantization companions of nameable tensors are validated before anything loads, and fused qkv
        // tensors the split map resolved past are reported once, in the style of the ignored-tensor lines.
        ValidateCompanionTensors();

        if (logLine is not null && _ignoredFusedTensors.Count > 0)
        {
            GgufMetadata metadata = file.Metadata;
            string modelName = metadata.Has("general.name") ? metadata.Get("general.name").AsString() : "";
            logLine(
                $"tensors_ignored model='{(modelName.Length > 0 ? modelName : file.Path)}' " +
                $"architecture='{Plan.Architecture}' tensors={string.Join(",", _ignoredFusedTensors)}");
        }

        _attnNormWeight = LoadLayerVectors(layers, TensorRole.AttentionNorm, Plan.HiddenSize);
        _ffnNormWeight = LoadLayerVectors(layers, TensorRole.FeedForwardNorm, Plan.HiddenSize);

        if (Plan.Bias == BiasPolicy.Declared && !Plan.UseAttentionBias)
        {
            // Declared false: present attention-bias tensors are neither loaded nor applied, and the file is
            // reported once naming them.
            _biasQ = new float[layers][];
            _biasK = new float[layers][];
            _biasV = new float[layers][];
            _biasO = new float[layers][];
            ReportIgnoredTensors(
                logLine,
                "bias_ignored",
                file,
                [TensorRole.AttentionQueryBias, TensorRole.AttentionKeyBias, TensorRole.AttentionValueBias, TensorRole.AttentionOutputBias]);
        }
        else
        {
            // Resolved on: the q/k/v bias vectors are required — a declared use_bias or a profile default that
            // turns the feature on makes them part of the file's graph, and the failure names the first one the
            // file lacks. The output-projection bias is used when present but demanded by no resolved value:
            // only some conversions ship one. ByTensorPresence keeps the presence rule for the q/k/v side too.
            bool required = Plan.Bias == BiasPolicy.Declared;
            _biasQ = LoadAttentionBias(layers, TensorRole.AttentionQueryBias, (long)Plan.HeadCount * Plan.HeadDim, required);
            _biasK = LoadAttentionBias(layers, TensorRole.AttentionKeyBias, (long)Plan.KvHeadCount * Plan.HeadDim, required);
            _biasV = LoadAttentionBias(layers, TensorRole.AttentionValueBias, (long)Plan.KvHeadCount * Plan.HeadDim, required);
            _biasO = LoadAttentionBias(layers, TensorRole.AttentionOutputBias, Plan.HiddenSize, required: false);
        }

        if (Plan.UseNormBias)
        {
            _attnNormBias = LoadLayerVectors(layers, TensorRole.AttentionNormBias, Plan.HiddenSize);
            _ffnNormBias = LoadLayerVectors(layers, TensorRole.FeedForwardNormBias, Plan.HiddenSize);
        }
        else
        {
            _attnNormBias = new float[layers][];
            _ffnNormBias = new float[layers][];
            ReportIgnoredTensors(logLine, "norm_bias_ignored", file, [TensorRole.AttentionNormBias, TensorRole.FeedForwardNormBias]);
        }

        switch (Plan.QkNorm)
        {
            case QkNormMode.PerHeadWeighted:
                _attnQNorm = LoadLayerVectors(layers, TensorRole.AttentionQueryNorm, Plan.HeadDim);
                _attnKNorm = LoadLayerVectors(layers, TensorRole.AttentionKeyNorm, Plan.HeadDim);
                break;

            case QkNormMode.StreamWeighted:
                // The whole-stream form normalizes each side's projected width at once, before any head split, so the
                // file's vector must cover that side's full projection — head_count × head_dim for q and kv_head_count ×
                // head_dim for k — rather than one head. A per-head-sized vector fails the load naming its shape.
                _attnQNorm = LoadLayerVectors(layers, TensorRole.AttentionQueryNorm, Plan.HeadCount * Plan.HeadDim);
                _attnKNorm = LoadLayerVectors(layers, TensorRole.AttentionKeyNorm, Plan.KvHeadCount * Plan.HeadDim);
                break;

            case QkNormMode.Off:
                // Resolved off: present per-head vectors are neither loaded nor applied, and the file is reported once naming them.
                _attnQNorm = new float[layers][];
                _attnKNorm = new float[layers][];
                ReportIgnoredTensors(logLine, "qk_norm_ignored", file, [TensorRole.AttentionQueryNorm, TensorRole.AttentionKeyNorm]);
                break;

            case QkNormMode.PerHeadUnweighted:
                // The unweighted form normalizes with the plan epsilon alone and demands no tensor at all. Present
                // per-head vectors are reported as unused rather than mistaken for a requirement this mode cannot express.
                _attnQNorm = new float[layers][];
                _attnKNorm = new float[layers][];
                ReportIgnoredTensors(logLine, "qk_norm_vectors_unused", file, [TensorRole.AttentionQueryNorm, TensorRole.AttentionKeyNorm]);
                break;

            default:
                throw new ModelException($"QK-norm mode '{Plan.QkNorm}' is not implemented by the forward engine");
        }

        if (Plan.NormPlacement == NormPlacement.PostSublayerOutputs)
        {
            _postAttnNorm = LoadPostOutputNorms(layers, TensorRole.PostAttentionNorm);
            _postFfnNorm = LoadPostOutputNorms(layers, TensorRole.PostFeedForwardNorm);
        }
        else
        {
            // A pre-only placement reads no post-sublayer norm vector: a file carrying one describes a graph this
            // profile does not implement, so it fails naming every such tensor instead of leaving them unread.
            RejectUnconsumedPostNorms(layers);
            _postAttnNorm = new float[layers][];
            _postFfnNorm = new float[layers][];
        }

        LogicalTensor tokenEmbd = RequireTensor(TensorRole.TokenEmbedding);
        RequireSingle(tokenEmbd, Plan.VocabSize, Plan.HiddenSize);
        _tokenEmbd = tokenEmbd;

        LogicalTensor outputNorm = RequireTensor(TensorRole.OutputNorm);
        _outputNormWeight = LoadVector(outputNorm, Plan.HiddenSize);
        _outputNormBias = Plan.UseNormBias
            ? LoadOptionalVector(Plan.Tensors[TensorRole.OutputNormBias].Resolve(_fileTensors), Plan.HiddenSize)
            : null;

        LogicalTensor? output = Plan.Tensors[TensorRole.OutputHead].Resolve(_fileTensors);
        if (output is not null)
        {
            RequireSingle(output, Plan.VocabSize, Plan.HiddenSize);
        }

        _output = output;
        _outputBias = LoadOptionalVector(Plan.Tensors[TensorRole.OutputBias].Resolve(_fileTensors), Plan.VocabSize);

        int hidden = Plan.HiddenSize;
        _cur = new float[hidden];
        _normed = new float[hidden];
        _q = new float[Plan.HeadCount * Plan.HeadDim];
        _k = new float[Plan.KvHeadCount * Plan.HeadDim];
        _v = new float[Plan.KvHeadCount * Plan.HeadDim];
        _attnOut = new float[Plan.HeadCount * Plan.HeadDim];
        _proj = new float[hidden];
        _gateUp = new float[Plan.IntermediateSize];
        _up = new float[Plan.IntermediateSize];
        _logits = new float[Plan.VocabSize];

        _batchCur = new float[_rowCapacity * hidden];
        _batchNormed = new float[_rowCapacity * hidden];
        _batchQ = new float[_rowCapacity * Plan.HeadCount * Plan.HeadDim];
        _batchK = new float[_rowCapacity * Plan.KvHeadCount * Plan.HeadDim];
        _batchV = new float[_rowCapacity * Plan.KvHeadCount * Plan.HeadDim];
        _batchAttnOut = new float[_rowCapacity * Plan.HeadCount * Plan.HeadDim];
        _batchProj = new float[_rowCapacity * hidden];
        _batchGateUp = new float[_rowCapacity * Plan.IntermediateSize];
        _batchUp = new float[_rowCapacity * Plan.IntermediateSize];
        _batchLogits = new float[_rowCapacity * Plan.VocabSize];

        _tiledRowMax = new float[_rowCapacity * Plan.HeadCount];
        _tiledRowSumExp = new float[_rowCapacity * Plan.HeadCount];

        // The rotation covers the declared rotary slice, or the full head dimension when the file declares none.
        _ropeHalf = (Plan.RopeRotaryDimensions > 0 ? Plan.RopeRotaryDimensions : Plan.HeadDim) / 2;

        // The score gain is the head-dimension scale times the variant's attention factor: the scaled angle
        // tables carry their own magnitude, so the score path carries the factor exactly once. An unscaled plan
        // multiplies by one and the gain is the plain head-dimension scale.
        _attnScale = (Plan.RopeScaling?.AttentionFactor ?? 1f) / MathF.Sqrt(Plan.HeadDim);

        // Precompute cos/sin per position and pair up to the maximum context so decode is table-driven: identical
        // inputs always reproduce bit-identical rotations regardless of call order or buffer reuse. A resolved
        // scaling plan builds the variant's tables over the slice instead of the unscaled full-head tables.
        _cos = new float[maxContextTokens * _ropeHalf];
        _sin = new float[maxContextTokens * _ropeHalf];
        BuildRotationTables(Plan.RopeTheta, maxContextTokens, _cos, _sin);

        // A plan that resolves a second rotary base for the layers its window interleave narrows carries one more pair of
        // tables at that base, and each layer then reads whichever pair its own resolved window selects — which is how a
        // reference graph rotating its local-attention layers at one frequency and its full-attention layers at another is
        // read the way it computes. One base for every layer keeps the single pair above serving all of them, so nothing
        // extra is allocated or consulted for any model whose plan resolves no windowed rotation.
        int windowedLayerCount = CountWindowedLayers();
        bool splitsRopeBases = Plan.RopeWindowedTheta != Plan.RopeTheta && windowedLayerCount > 0;

        if (splitsRopeBases)
        {
            _cosWindowed = new float[maxContextTokens * _ropeHalf];
            _sinWindowed = new float[maxContextTokens * _ropeHalf];
            BuildRotationTables(Plan.RopeWindowedTheta, maxContextTokens, _cosWindowed, _sinWindowed);
        }

        // A positional plan that is not the plain full-head rotation is announced once at Open, in the style of the
        // tensor and profile lines: the variant, its multiplier and original context length, the attention factor
        // carried into the score path, and the dimension count the rotation actually covers. The plain shape stays
        // unannounced so the line means the same thing on every host.
        if (logLine is not null && (Plan.RopeScaling is not null || Plan.RopeRotaryDimensions > 0))
        {
            GgufMetadata metadata = file.Metadata;
            string modelName = metadata.Has("general.name") ? metadata.Get("general.name").AsString() : "";
            RopeScalingPlan? plan = Plan.RopeScaling;
            logLine(
                $"rope_scaling model='{(modelName.Length > 0 ? modelName : file.Path)}' " +
                $"architecture='{Plan.Architecture}' variant={(plan?.Variant ?? "none")} " +
                $"multiplier={(plan?.Multiplier ?? 1f).ToString(CultureInfo.InvariantCulture)} " +
                $"original_context_length={(plan?.OriginalContextLength ?? 0)} " +
                $"attention_factor={(plan?.AttentionFactor ?? 1f).ToString(CultureInfo.InvariantCulture)} " +
                $"rotary_dimensions={(Plan.RopeRotaryDimensions > 0 ? Plan.RopeRotaryDimensions : Plan.HeadDim)}");
        }

        // A model rotating its windowed layers at a base other than the one its unbounded layers rotate at announces that
        // split once at Open, in the same style: both bases and how many layers of each kind the resolved interleave puts
        // them on. A plan carrying one base for every layer stays unannounced so this line only ever names a real split.
        if (logLine is not null && splitsRopeBases)
        {
            GgufMetadata metadata = file.Metadata;
            string modelName = metadata.Has("general.name") ? metadata.Get("general.name").AsString() : "";
            logLine(
                $"rope_bases model='{(modelName.Length > 0 ? modelName : file.Path)}' " +
                $"architecture='{Plan.Architecture}' global_base={Plan.RopeTheta.ToString(CultureInfo.InvariantCulture)} " +
                $"windowed_base={Plan.RopeWindowedTheta.ToString(CultureInfo.InvariantCulture)} " +
                $"windowed_layers={windowedLayerCount} unwindowed_layers={Plan.LayerCount - windowedLayerCount}");
        }
    }

    /// <summary>
    /// Forwards a window of prompt positions in sub-batches of up to the plan's resolved <see cref="ModelPlan.PrefillRowWidth"/>
    /// rows and returns the final row's logits. Each row seeds its own residual stream from its token's embedding row and
    /// keeps its own norms, RoPE at its own absolute position, and its own causal attention, while every dense
    /// weight matrix (attention q/k/v/o, feed-forward gate/up/down) is traversed once per sub-batch through
    /// <see cref="IBackend.GemvBatch"/> instead of once per token — a window whose rows fill one sub-batch pays one traversal per
    /// matrix, not one per row. Within a layer the window's K/V is written for every row before any row attends, so a
    /// row's causal selection covers exactly the positions at or before its own: earlier rows of the same
    /// sub-batch are in it, later rows are not, and the read set is the same one the single-token forward made at
    /// that position. Each row appends at its own position into <paramref name="kv"/>, exactly as stepping the
    /// tokens one at a time would, and every batched result is bit-identical to that stepping — the same
    /// guarantee <see cref="DecodeBatch"/> holds per sequence. The returned logits are the window's final row:
    /// the sequence's next-token distribution, held in the model's last-logits buffer as <see cref="LastLogits"/>
    /// reports it, so sampling after a prefill needs no second forward. A window longer than the resolved row width is
    /// forwarded as consecutive sub-batches, each seeing the previous ones' KV, so sub-batch boundaries move nothing:
    /// the sequence's KV bytes and final logits are what the same tokens forwarded one at a time would leave, whatever
    /// the resolved width splits them at.
    /// </summary>
    /// <exception cref="ArgumentException">The window is empty.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A token id is outside the vocabulary.</exception>
    /// <exception cref="InvalidOperationException">
    /// Appending the window would pass this model's max context, checked before any slot is reserved.
    /// </exception>
    public Span<float> Prefill(IReadOnlyList<int> tokens, SequenceKv kv)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(kv);

        int n = tokens.Count;
        if (n == 0)
        {
            throw new ArgumentException("a prefill window must hold at least one token.", nameof(tokens));
        }

        int basePosition = kv.Length;
        if (basePosition + n > MaxContextTokens)
        {
            throw new InvalidOperationException(
                $"{n} prefill tokens starting at position {basePosition} exceed this model's max context of {MaxContextTokens} tokens.");
        }

        // The plan's resolved row count splits the window: every width carries the same positions, KV bytes, and logits,
        // so this is a throughput knob for batched weight traversals and nothing else. Read fresh per call like every
        // other forward-shape value.
        int prefillRows = Plan.PrefillRowWidth;

        for (int offset = 0; offset < n; offset += prefillRows)
        {
            RunPrefillWindow(tokens, offset, Math.Min(prefillRows, n - offset), basePosition, kv, emitLogits: offset + prefillRows >= n);
        }

        return _logits.AsSpan();
    }

    // The resolved kernel selection decides which window route runs, re-read at every sub-batch so a hot edit lands
    // from the next boundary: the batched row window, its row-tiled attention variant, and the dense-matrix weight
    // traversal all have kernels today, and a selection this build has no kernel for fails naming it rather than
    // running some other route behind it. The dense-matrix traversal is the row window with its weight passes handed
    // to the blocked matmul; attention stays on the same per-row causal kernel either way.
    private void RunPrefillWindow(IReadOnlyList<int> tokens, int offset, int rows, int basePosition, SequenceKv kv, bool emitLogits)
    {
        switch (_prefillKernel)
        {
            case PrefillKernel.GemV:
                PrefillWindow(tokens, offset, rows, basePosition, kv, emitLogits, denseMatmul: false);
                break;

            case PrefillKernel.Tiled:
                PrefillWindowTiled(tokens, offset, rows, basePosition, kv, emitLogits);
                break;

            case PrefillKernel.Gemm:
                PrefillWindow(tokens, offset, rows, basePosition, kv, emitLogits, denseMatmul: true);
                break;

            default:
                throw KernelRefusal(_prefillKernel);
        }
    }

    // One dense weight pass over a whole sub-batch, handed to the route this window resolved: either backend traverses
    // each matrix once for every activation vector of the window and writes the same <c>[batch * rows + row]</c> layout,
    // the batched GEMV walking one weight row per activation group and the blocked matmul dequantizing a weight tile
    // once for all of them. The choice is hoisted to this call so no per-row loop reads it as an operand.
    private void WeightPass(ReadOnlySpan<float> inputs, int count, WeightRef weight, Span<float> outputs, bool denseMatmul)
    {
        if (denseMatmul)
        {
            _backend.GemmBatch(inputs, count, weight, outputs);
        }
        else
        {
            _backend.GemvBatch(inputs, count, weight, outputs);
        }
    }

    // One sub-batch of a prefill window: the rows advance through the whole layer loop together, each holding its
    // own residual-stream row and its own absolute position — the same streams and positions DecodeBatch keeps per
    // sequence — over one sequence's growing KV. Attention runs row by row through the causal kernel, so a row
    // scores its own prefix only, including the earlier rows this sub-batch wrote at the same layer and excluding
    // every later one. Only the window's final row reaches the output head: the rows before it are kept for their
    // KV, not for logits, so the lm_head runs once per prefill rather than once per sub-batch. Every dense weight
    // matrix of the sub-batch is traversed once — batched across the window's activation vectors on the row route, or
    // through the blocked matmul when <paramref name="denseMatmul"/> selects it. The two routes differ only in how a
    // weight tile reaches the arithmetic, so each output element lands byte-identical to the other and to the
    // single-vector reference.
    private void PrefillWindow(IReadOnlyList<int> tokens, int offset, int rows, int basePosition, SequenceKv kv, bool emitLogits, bool denseMatmul)
    {
        int hidden = Plan.HiddenSize;
        int qkvRegion = Plan.HeadCount * Plan.HeadDim;
        int kvRegion = Plan.KvHeadCount * Plan.HeadDim;
        int inter = Plan.IntermediateSize;
        float attnScale = _attnScale;

        // Embedding lookup: each token's dequantized token_embd row seeds that row's input stream.
        ReadOnlySpan<byte> embBytes = _weights.Resolve(_tokenEmbd);
        WeightRef embRef = new(embBytes, _tokenEmbd.Shape, _tokenEmbd.Format);
        for (int r = 0; r < rows; r++)
        {
            int tokenId = tokens[offset + r];
            if ((uint)tokenId >= (uint)Plan.VocabSize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(tokens), $"token id {tokenId} at index {offset + r} is outside the vocabulary of {Plan.VocabSize}.");
            }

            _backend.DequantRow(embRef, tokenId, _batchCur.AsSpan(r * hidden, hidden));
        }

        ApplyInputEmbeddingScale(_batchCur.AsSpan(0, rows * hidden));

        // Reserve every row's slot in the sequence before any layer writes, mirroring the append each token made.
        for (int r = 0; r < rows; r++)
        {
            kv.Append();
        }

        for (int l = 0; l < Plan.LayerCount; l++)
        {
            int layerWindow = LayerAttentionWindow(l);

            // Pre-attention norm per row: identical op to the single-token forward.
            for (int r = 0; r < rows; r++)
            {
                ApplyNorm(_batchCur.AsSpan(r * hidden, hidden), _attnNormWeight[l], _attnNormBias[l], _batchNormed.AsSpan(r * hidden, hidden));
            }

            // One weight pass per matrix for the whole sub-batch; each row's result matches its single-vector GEMV on
            // either route this selection can take.
            WeightPass(_batchNormed.AsSpan(0, rows * hidden), rows, AttentionQueryWeight(l), _batchQ.AsSpan(0, rows * qkvRegion), denseMatmul);

            WeightPass(_batchNormed.AsSpan(0, rows * hidden), rows, AttentionKeyWeight(l), _batchK.AsSpan(0, rows * kvRegion), denseMatmul);

            WeightPass(_batchNormed.AsSpan(0, rows * hidden), rows, AttentionValueWeight(l), _batchV.AsSpan(0, rows * kvRegion), denseMatmul);

            // Bias, QK-norm, RoPE at the row's absolute position, and the K/V writes for every row complete
            // before any row attends, so each row's causal read sees exactly the prefix it would have alone.
            for (int r = 0; r < rows; r++)
            {
                int position = basePosition + offset + r;

                AddBias(_batchQ.AsSpan(r * qkvRegion, qkvRegion), _biasQ[l]);
                AddBias(_batchK.AsSpan(r * kvRegion, kvRegion), _biasK[l]);
                AddBias(_batchV.AsSpan(r * kvRegion, kvRegion), _biasV[l]);

                ApplyQkNorm(_batchQ.AsSpan(r * qkvRegion, qkvRegion), _attnQNorm[l]);
                ApplyQkNorm(_batchK.AsSpan(r * kvRegion, kvRegion), _attnKNorm[l]);

                ApplyRopeHeads(_batchQ.AsSpan(r * qkvRegion, qkvRegion), Plan.HeadCount, position, windowedLayer: layerWindow > 0);
                ApplyRopeHeads(_batchK.AsSpan(r * kvRegion, kvRegion), Plan.KvHeadCount, position, windowedLayer: layerWindow > 0);

                kv.WriteK(position, l, _batchK.AsSpan(r * kvRegion, kvRegion));
                kv.WriteV(position, l, _batchV.AsSpan(r * kvRegion, kvRegion));
            }

            for (int r = 0; r < rows; r++)
            {
                LayerOps.CausalAttention(_backend, _batchQ.AsSpan(r * qkvRegion, qkvRegion), Plan.HeadCount, Plan.KvHeadCount, Plan.HeadDim, kv, l, basePosition + offset + r, attnScale, Plan.AttentionSoftcap, layerWindow, _batchAttnOut.AsSpan(r * qkvRegion, qkvRegion));
            }

            ReadOnlySpan<byte> bytes = _weights.Resolve(_wo[l]);
            WeightPass(_batchAttnOut.AsSpan(0, rows * qkvRegion), rows, new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), _batchProj.AsSpan(0, rows * hidden), denseMatmul);

            for (int r = 0; r < rows; r++)
            {
                AddBias(_batchProj.AsSpan(r * hidden, hidden), _biasO[l]);

                // Post-sublayer placement: this row's attention output joins its stream only after the post-attention norm.
                ApplyPostSublayerNorm(_batchProj.AsSpan(r * hidden, hidden), _postAttnNorm[l]);

                // ffn_inp add on the row's own stream, same order as the single-token forward.
                Span<float> cur = _batchCur.AsSpan(r * hidden, hidden);
                ReadOnlySpan<float> proj = _batchProj.AsSpan(r * hidden, hidden);
                for (int i = 0; i < hidden; i++)
                {
                    cur[i] += proj[i];
                }

                ApplyNorm(cur, _ffnNormWeight[l], _ffnNormBias[l], _batchNormed.AsSpan(r * hidden, hidden));
            }

            if (_sparseFfn is { } routing)
            {
                ApplySparseFeedForwardBatch(routing, l, rows);
            }
            else
            {
                WeightPass(_batchNormed.AsSpan(0, rows * hidden), rows, FeedForwardGateWeight(l), _batchGateUp.AsSpan(0, rows * inter), denseMatmul);

                WeightPass(_batchNormed.AsSpan(0, rows * hidden), rows, FeedForwardUpWeight(l), _batchUp.AsSpan(0, rows * inter), denseMatmul);

                // act = activation(gate) * up in place on each row's gate buffer.
                for (int r = 0; r < rows; r++)
                {
                    ApplyActivation(_batchGateUp.AsSpan(r * inter, inter), _batchUp.AsSpan(r * inter, inter));
                }

                bytes = _weights.Resolve(_ffnDown[l]);
                WeightPass(_batchGateUp.AsSpan(0, rows * inter), rows, new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), _batchProj.AsSpan(0, rows * hidden), denseMatmul);
            }

            for (int r = 0; r < rows; r++)
            {
                // Post-sublayer placement: this row's feed-forward output joins its stream only after the post-feed-forward norm.
                ApplyPostSublayerNorm(_batchProj.AsSpan(r * hidden, hidden), _postFfnNorm[l]);

                // ffn_out + ffn_inp: the input stream for this row's next layer.
                Span<float> cur = _batchCur.AsSpan(r * hidden, hidden);
                ReadOnlySpan<float> proj = _batchProj.AsSpan(r * hidden, hidden);
                for (int i = 0; i < hidden; i++)
                {
                    cur[i] += proj[i];
                }
            }
        }

        if (emitLogits)
        {
            // The window's final row is the sequence's next-token distribution: its output norm and lm_head GEMV
            // run alone over the single-token buffers, exactly the tail the single-token forward produced.
            ApplyNorm(_batchCur.AsSpan((rows - 1) * hidden, hidden), _outputNormWeight, _outputNormBias, _normed);

            LogicalTensor lmHead = _output ?? _tokenEmbd; // tied weights reuse the token-embedding rows as logits
            ReadOnlySpan<byte> headBytes = _weights.Resolve(lmHead);
            _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(headBytes, lmHead.Shape, lmHead.Format), _logits);

            if (_outputBias is not null)
            {
                AddBias(_logits, _outputBias);
            }

            ApplyFinalLogitSoftcap(_logits);
        }
    }

    // One sub-batch of a prefill window on the row-tiled route: the same streams, positions, norms, RoPE, K/V writes,
    // batched weight passes, and residual adds the batched-row window runs — attention alone differs. Where the
    // batched-row route walks each row through the causal kernel on its own, this route hands the whole sub-batch to
    // LayerOps.CausalAttentionTiledRows: one walk per layer loads each selected position's K/V once per kv-head group
    // and applies it to every row whose own causal selection covers it, while each row advances its own online-softmax
    // state in the preallocated row-state buffers in exactly the order its per-row kernel would have run it. Only the
    // window's final row reaches the output head, exactly as on the batched-row route.
    private void PrefillWindowTiled(IReadOnlyList<int> tokens, int offset, int rows, int basePosition, SequenceKv kv, bool emitLogits)
    {
        int hidden = Plan.HiddenSize;
        int qkvRegion = Plan.HeadCount * Plan.HeadDim;
        int kvRegion = Plan.KvHeadCount * Plan.HeadDim;
        int inter = Plan.IntermediateSize;
        float attnScale = _attnScale;

        // Embedding lookup: each token's dequantized token_embd row seeds that row's input stream.
        ReadOnlySpan<byte> embBytes = _weights.Resolve(_tokenEmbd);
        WeightRef embRef = new(embBytes, _tokenEmbd.Shape, _tokenEmbd.Format);
        for (int r = 0; r < rows; r++)
        {
            int tokenId = tokens[offset + r];
            if ((uint)tokenId >= (uint)Plan.VocabSize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(tokens), $"token id {tokenId} at index {offset + r} is outside the vocabulary of {Plan.VocabSize}.");
            }

            _backend.DequantRow(embRef, tokenId, _batchCur.AsSpan(r * hidden, hidden));
        }

        ApplyInputEmbeddingScale(_batchCur.AsSpan(0, rows * hidden));

        // Reserve every row's slot in the sequence before any layer writes, mirroring the append each token made.
        for (int r = 0; r < rows; r++)
        {
            kv.Append();
        }

        Span<float> tiledRowMax = _tiledRowMax.AsSpan(0, rows * Plan.HeadCount); // every row's per-head online max across the tiled walk
        Span<float> tiledRowSumExp = _tiledRowSumExp.AsSpan(0, rows * Plan.HeadCount); // every row's per-head weight sum across the walk

        for (int l = 0; l < Plan.LayerCount; l++)
        {
            int layerWindow = LayerAttentionWindow(l);

            // Pre-attention norm per row: identical op to the single-token forward.
            for (int r = 0; r < rows; r++)
            {
                ApplyNorm(_batchCur.AsSpan(r * hidden, hidden), _attnNormWeight[l], _attnNormBias[l], _batchNormed.AsSpan(r * hidden, hidden));
            }

            // One weight pass per matrix for the whole sub-batch; each row's result matches its single-vector GEMV.
            _backend.GemvBatch(_batchNormed.AsSpan(0, rows * hidden), rows, AttentionQueryWeight(l), _batchQ.AsSpan(0, rows * qkvRegion));

            _backend.GemvBatch(_batchNormed.AsSpan(0, rows * hidden), rows, AttentionKeyWeight(l), _batchK.AsSpan(0, rows * kvRegion));

            _backend.GemvBatch(_batchNormed.AsSpan(0, rows * hidden), rows, AttentionValueWeight(l), _batchV.AsSpan(0, rows * kvRegion));

            // Bias, QK-norm, RoPE at the row's absolute position, and the K/V writes for every row complete
            // before any row attends, so each row's causal read sees exactly the prefix it would have alone.
            for (int r = 0; r < rows; r++)
            {
                int position = basePosition + offset + r;

                AddBias(_batchQ.AsSpan(r * qkvRegion, qkvRegion), _biasQ[l]);
                AddBias(_batchK.AsSpan(r * kvRegion, kvRegion), _biasK[l]);
                AddBias(_batchV.AsSpan(r * kvRegion, kvRegion), _biasV[l]);

                ApplyQkNorm(_batchQ.AsSpan(r * qkvRegion, qkvRegion), _attnQNorm[l]);
                ApplyQkNorm(_batchK.AsSpan(r * kvRegion, kvRegion), _attnKNorm[l]);

                ApplyRopeHeads(_batchQ.AsSpan(r * qkvRegion, qkvRegion), Plan.HeadCount, position, windowedLayer: layerWindow > 0);
                ApplyRopeHeads(_batchK.AsSpan(r * kvRegion, kvRegion), Plan.KvHeadCount, position, windowedLayer: layerWindow > 0);

                kv.WriteK(position, l, _batchK.AsSpan(r * kvRegion, kvRegion));
                kv.WriteV(position, l, _batchV.AsSpan(r * kvRegion, kvRegion));
            }

            // Row-tiled attention for the whole sub-batch in one walk per layer: each selected position's K/V is read
            // once per kv-head group and applied to every row whose own causal selection covers it — a row whose causal
            // limit or own selection excludes the position never touches that slot — while each row's online-softmax
            // state advances through the row-state buffers in exactly the ascending order its per-row kernel ran it.
            LayerOps.CausalAttentionTiledRows(
                _backend,
                _batchQ.AsSpan(0, rows * qkvRegion),
                basePosition + offset,
                rows,
                Plan.HeadCount,
                Plan.KvHeadCount,
                Plan.HeadDim,
                kv,
                l,
                attnScale,
                Plan.AttentionSoftcap,
                layerWindow,
                tiledRowMax,
                tiledRowSumExp,
                _batchAttnOut.AsSpan(0, rows * qkvRegion));

            ReadOnlySpan<byte> bytes = _weights.Resolve(_wo[l]);
            _backend.GemvBatch(_batchAttnOut.AsSpan(0, rows * qkvRegion), rows, new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), _batchProj.AsSpan(0, rows * hidden));

            for (int r = 0; r < rows; r++)
            {
                AddBias(_batchProj.AsSpan(r * hidden, hidden), _biasO[l]);

                // Post-sublayer placement: this row's attention output joins its stream only after the post-attention norm.
                ApplyPostSublayerNorm(_batchProj.AsSpan(r * hidden, hidden), _postAttnNorm[l]);

                // ffn_inp add on the row's own stream, same order as the single-token forward.
                Span<float> cur = _batchCur.AsSpan(r * hidden, hidden);
                ReadOnlySpan<float> proj = _batchProj.AsSpan(r * hidden, hidden);
                for (int i = 0; i < hidden; i++)
                {
                    cur[i] += proj[i];
                }

                ApplyNorm(cur, _ffnNormWeight[l], _ffnNormBias[l], _batchNormed.AsSpan(r * hidden, hidden));
            }

            if (_sparseFfn is { } routing)
            {
                ApplySparseFeedForwardBatch(routing, l, rows);
            }
            else
            {
                _backend.GemvBatch(_batchNormed.AsSpan(0, rows * hidden), rows, FeedForwardGateWeight(l), _batchGateUp.AsSpan(0, rows * inter));

                _backend.GemvBatch(_batchNormed.AsSpan(0, rows * hidden), rows, FeedForwardUpWeight(l), _batchUp.AsSpan(0, rows * inter));

                // act = activation(gate) * up in place on each row's gate buffer.
                for (int r = 0; r < rows; r++)
                {
                    ApplyActivation(_batchGateUp.AsSpan(r * inter, inter), _batchUp.AsSpan(r * inter, inter));
                }

                bytes = _weights.Resolve(_ffnDown[l]);
                _backend.GemvBatch(_batchGateUp.AsSpan(0, rows * inter), rows, new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), _batchProj.AsSpan(0, rows * hidden));
            }

            for (int r = 0; r < rows; r++)
            {
                // Post-sublayer placement: this row's feed-forward output joins its stream only after the post-feed-forward norm.
                ApplyPostSublayerNorm(_batchProj.AsSpan(r * hidden, hidden), _postFfnNorm[l]);

                // ffn_out + ffn_inp: the input stream for this row's next layer.
                Span<float> cur = _batchCur.AsSpan(r * hidden, hidden);
                ReadOnlySpan<float> proj = _batchProj.AsSpan(r * hidden, hidden);
                for (int i = 0; i < hidden; i++)
                {
                    cur[i] += proj[i];
                }
            }
        }

        if (emitLogits)
        {
            // The window's final row is the sequence's next-token distribution: its output norm and lm_head GEMV
            // run alone over the single-token buffers, exactly the tail the single-token forward produced.
            ApplyNorm(_batchCur.AsSpan((rows - 1) * hidden, hidden), _outputNormWeight, _outputNormBias, _normed);

            LogicalTensor lmHead = _output ?? _tokenEmbd; // tied weights reuse the token-embedding rows as logits
            ReadOnlySpan<byte> headBytes = _weights.Resolve(lmHead);
            _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(headBytes, lmHead.Shape, lmHead.Format), _logits);

            if (_outputBias is not null)
            {
                AddBias(_logits, _outputBias);
            }

            ApplyFinalLogitSoftcap(_logits);
        }
    }

    /// <summary>Forwards one token at the next free position and returns its vocab-sized logits.</summary>
    public Span<float> DecodeNext(int tokenId, SequenceKv kv)
    {
        ArgumentNullException.ThrowIfNull(kv);

        ForwardToken(tokenId, kv.Length, kv);
        return _logits.AsSpan();
    }

    /// <summary>
    /// Logits from the most recent forward: after a prefill these are already the next-token distribution for the
    /// sequence's final token — sampling them avoids re-forwarding (and double-appending) that token.
    /// </summary>
    public Span<float> LastLogits() => _logits.AsSpan();

    /// <summary>
    /// Advances up to <see cref="MaxBatchSequences"/> sequences by one token each in a single step: every dense
    /// weight matrix (attention q/k/v/o, feed-forward gate/up/down, lm_head) is traversed once for the whole batch
    /// through <see cref="IBackend.GemvBatch"/>, while norms, RoPE, causal attention, and the gated activation stay
    /// per-sequence because each sequence attends over its own KV history. Each sequence appends at its own <see cref="SequenceKv.Length"/>
    /// position, exactly as <see cref="DecodeNext"/> would, and every batched result is bit-identical to stepping that
    /// sequence alone. Sequences must belong to the paged KV pool this engine's forwards write into. The returned
    /// logits are owned by the model — valid until the next forward call — laid out as
    /// <c>logits[sequence * vocabSize + token]</c>.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// The two lists differ in length, fewer than one or more than <see cref="MaxBatchSequences"/> sequences were
    /// given, or a list entry is null.
    /// </exception>
    public Span<float> DecodeBatch(IReadOnlyList<int> tokenIds, IReadOnlyList<SequenceKv> kvs)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        ArgumentNullException.ThrowIfNull(kvs);

        int n = tokenIds.Count;
        if (n != kvs.Count)
        {
            throw new ArgumentException($"tokenIds has {tokenIds.Count} entries but kvs has {kvs.Count}: one token per sequence is required.", nameof(kvs));
        }

        if (n == 0 || n > MaxBatchSequences)
        {
            throw new ArgumentException($"a decode batch must hold between 1 and {MaxBatchSequences} sequences, got {n}.", nameof(tokenIds));
        }

        int[] positions = new int[n];
        for (int s = 0; s < n; s++)
        {
            if (tokenIds[s] < 0 || tokenIds[s] >= Plan.VocabSize)
            {
                throw new ArgumentException($"token id {tokenIds[s]} at sequence {s} is outside the vocabulary of {Plan.VocabSize}.", nameof(tokenIds));
            }

            SequenceKv kv = kvs[s] ?? throw new ArgumentException($"kvs entry {s} is null.", nameof(kvs));
            if (kv.Length >= MaxContextTokens)
            {
                throw new InvalidOperationException(
                    $"sequence {s} position {kv.Length} exceeds this model's max context of {MaxContextTokens} tokens.");
            }

            positions[s] = kv.Length;
        }

        // Embedding lookup: each token's dequantized token_embd row seeds its sequence's input stream.
        ReadOnlySpan<byte> embBytes = _weights.Resolve(_tokenEmbd);
        WeightRef embRef = new(embBytes, _tokenEmbd.Shape, _tokenEmbd.Format);
        for (int s = 0; s < n; s++)
        {
            _backend.DequantRow(embRef, tokenIds[s], _batchCur.AsSpan(s * Plan.HiddenSize, Plan.HiddenSize));
        }

        ApplyInputEmbeddingScale(_batchCur.AsSpan(0, n * Plan.HiddenSize));

        // Reserve this step's slot in every sequence before any layer writes, mirroring ForwardToken's append.
        for (int s = 0; s < n; s++)
        {
            kvs[s].Append();
        }

        int hidden = Plan.HiddenSize;
        int qkvRegion = Plan.HeadCount * Plan.HeadDim;
        int kvRegion = Plan.KvHeadCount * Plan.HeadDim;
        int inter = Plan.IntermediateSize;
        float attnScale = _attnScale;

        for (int l = 0; l < Plan.LayerCount; l++)
        {
            int layerWindow = LayerAttentionWindow(l);

            // Pre-attention norm per sequence: identical op to the single-token forward.
            for (int s = 0; s < n; s++)
            {
                ApplyNorm(_batchCur.AsSpan(s * hidden, hidden), _attnNormWeight[l], _attnNormBias[l], _batchNormed.AsSpan(s * hidden, hidden));
            }

            // One weight pass per matrix for the whole batch; each sequence's row result matches its single-vector GEMV.
            _backend.GemvBatch(_batchNormed.AsSpan(0, n * hidden), n, AttentionQueryWeight(l), _batchQ.AsSpan(0, n * qkvRegion));

            _backend.GemvBatch(_batchNormed.AsSpan(0, n * hidden), n, AttentionKeyWeight(l), _batchK.AsSpan(0, n * kvRegion));

            _backend.GemvBatch(_batchNormed.AsSpan(0, n * hidden), n, AttentionValueWeight(l), _batchV.AsSpan(0, n * kvRegion));

            for (int s = 0; s < n; s++)
            {
                AddBias(_batchQ.AsSpan(s * qkvRegion, qkvRegion), _biasQ[l]);
                AddBias(_batchK.AsSpan(s * kvRegion, kvRegion), _biasK[l]);
                AddBias(_batchV.AsSpan(s * kvRegion, kvRegion), _biasV[l]);

                ApplyQkNorm(_batchQ.AsSpan(s * qkvRegion, qkvRegion), _attnQNorm[l]);
                ApplyQkNorm(_batchK.AsSpan(s * kvRegion, kvRegion), _attnKNorm[l]);

                ApplyRopeHeads(_batchQ.AsSpan(s * qkvRegion, qkvRegion), Plan.HeadCount, positions[s], windowedLayer: layerWindow > 0);
                ApplyRopeHeads(_batchK.AsSpan(s * kvRegion, kvRegion), Plan.KvHeadCount, positions[s], windowedLayer: layerWindow > 0);

                kvs[s].WriteK(positions[s], l, _batchK.AsSpan(s * kvRegion, kvRegion));
                kvs[s].WriteV(positions[s], l, _batchV.AsSpan(s * kvRegion, kvRegion));

                LayerOps.CausalAttention(_backend, _batchQ.AsSpan(s * qkvRegion, qkvRegion), Plan.HeadCount, Plan.KvHeadCount, Plan.HeadDim, kvs[s], l, positions[s], attnScale, Plan.AttentionSoftcap, layerWindow, _batchAttnOut.AsSpan(s * qkvRegion, qkvRegion));
            }

            ReadOnlySpan<byte> bytes = _weights.Resolve(_wo[l]);
            _backend.GemvBatch(_batchAttnOut.AsSpan(0, n * qkvRegion), n, new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), _batchProj.AsSpan(0, n * hidden));

            for (int s = 0; s < n; s++)
            {
                AddBias(_batchProj.AsSpan(s * hidden, hidden), _biasO[l]);

                // Post-sublayer placement: this sequence's attention output joins its stream only after the post-attention norm.
                ApplyPostSublayerNorm(_batchProj.AsSpan(s * hidden, hidden), _postAttnNorm[l]);

                // ffn_inp add on the sequence's own stream, same order as the single-token forward.
                Span<float> cur = _batchCur.AsSpan(s * hidden, hidden);
                ReadOnlySpan<float> proj = _batchProj.AsSpan(s * hidden, hidden);
                for (int i = 0; i < hidden; i++)
                {
                    cur[i] += proj[i];
                }

                ApplyNorm(cur, _ffnNormWeight[l], _ffnNormBias[l], _batchNormed.AsSpan(s * hidden, hidden));
            }

            if (_sparseFfn is { } routing)
            {
                ApplySparseFeedForwardBatch(routing, l, n);
            }
            else
            {
                _backend.GemvBatch(_batchNormed.AsSpan(0, n * hidden), n, FeedForwardGateWeight(l), _batchGateUp.AsSpan(0, n * inter));

                _backend.GemvBatch(_batchNormed.AsSpan(0, n * hidden), n, FeedForwardUpWeight(l), _batchUp.AsSpan(0, n * inter));

                // act = activation(gate) * up in place on each sequence's gate buffer.
                for (int s = 0; s < n; s++)
                {
                    ApplyActivation(_batchGateUp.AsSpan(s * inter, inter), _batchUp.AsSpan(s * inter, inter));
                }

                bytes = _weights.Resolve(_ffnDown[l]);
                _backend.GemvBatch(_batchGateUp.AsSpan(0, n * inter), n, new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), _batchProj.AsSpan(0, n * hidden));
            }

            for (int s = 0; s < n; s++)
            {
                // Post-sublayer placement: this sequence's feed-forward output joins its stream only after the post-feed-forward norm.
                ApplyPostSublayerNorm(_batchProj.AsSpan(s * hidden, hidden), _postFfnNorm[l]);

                // ffn_out + ffn_inp: the input stream for this sequence's next layer.
                Span<float> cur = _batchCur.AsSpan(s * hidden, hidden);
                ReadOnlySpan<float> proj = _batchProj.AsSpan(s * hidden, hidden);
                for (int i = 0; i < hidden; i++)
                {
                    cur[i] += proj[i];
                }
            }
        }

        for (int s = 0; s < n; s++)
        {
            ApplyNorm(_batchCur.AsSpan(s * hidden, hidden), _outputNormWeight, _outputNormBias, _batchNormed.AsSpan(s * hidden, hidden));
        }

        LogicalTensor lmHead = _output ?? _tokenEmbd; // tied weights reuse the token-embedding rows as logits
        ReadOnlySpan<byte> headBytes = _weights.Resolve(lmHead);
        _backend.GemvBatch(_batchNormed.AsSpan(0, n * hidden), n, new WeightRef(headBytes, lmHead.Shape, lmHead.Format), _batchLogits.AsSpan(0, n * Plan.VocabSize));

        if (_outputBias is not null)
        {
            for (int s = 0; s < n; s++)
            {
                AddBias(_batchLogits.AsSpan(s * Plan.VocabSize, Plan.VocabSize), _outputBias);
            }
        }

        ApplyFinalLogitSoftcap(_batchLogits.AsSpan(0, n * Plan.VocabSize));

        return _batchLogits.AsSpan(0, n * Plan.VocabSize);
    }

    /// <summary>Token positions one encoder pass advances through the dense matrices per batched weight traversal.</summary>
    public const int EncodeChunkTokens = 8;

    /// <summary>Largest token count a single <see cref="Encode"/> call may forward.</summary>
    public const int MaxEncodeTokens = 256;

    /// <summary>
    /// Encoder-only forward: runs the whole token sequence through every layer with full-sequence (bidirectional)
    /// attention — each query position scores against all positions of its own sequence's KV contents instead of a
    /// causal prefix, narrowed per layer to that layer's resolved attention window read backward from the query's own
    /// position on the layers its resolved window interleave masks — and returns the final hidden states after the output norm,
    /// without lm_head logits. Per-position norms, RoPE at absolute positions, biases, residuals, and the gated
    /// activation are exactly the decode-path
    /// operations; K/V for
    /// every position is appended to <paramref name="kv"/> as usual (positions start at <see cref="SequenceKv.Length"/>,
    /// so encoding a fresh sequence keeps each token attending only to its own sequence). Dense matrices are traversed
    /// once per chunk of up to <see cref="EncodeChunkTokens"/> positions via <see cref="IBackend.GemvBatch"/>; every
    /// position then attends over the whole finished KV span, which is what distinguishes this path from incremental
    /// causal decode. All queries — whichever chunk they belong to — see exactly every position of their sequence after
    /// appending (0..<see cref="SequenceKv.Length"/>) in ascending order: within <c>[max(0, q - window + 1), q]</c> when the
    /// layer resolves a positive window, where no query sees past its own position, and over the whole sequence on an
    /// unbounded layer — so results are independent of chunking.
    /// This establishes the dense-LLM architecture running in encoder mode as groundwork for embedding extraction;
    /// dedicated encoder-native architectures (ModernBERT-class) require their own tensor layouts and arrive with the
    /// embeddings endpoint itself. The returned array is caller-owned: <c>hidden[position * HiddenSize + i]</c>, length
    /// <c>tokens.Count * HiddenSize</c>.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// The token list is empty or larger than <see cref="MaxEncodeTokens"/>, a token id is outside the vocabulary, or
    /// appending the sequence would exceed this model's max context.
    /// </exception>
    public float[] Encode(IReadOnlyList<int> tokens, SequenceKv kv)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(kv);

        int n = tokens.Count;
        if (n == 0 || n > MaxEncodeTokens)
        {
            throw new ArgumentException(
                $"an encoder forward must hold between 1 and {MaxEncodeTokens} tokens, got {n}.", nameof(tokens));
        }

        int hidden = Plan.HiddenSize;
        int basePosition = kv.Length;
        if (basePosition + n > MaxContextTokens)
        {
            throw new ArgumentException(
                $"{n} encoder tokens starting at position {basePosition} exceed this model's max context of {MaxContextTokens}.",
                nameof(tokens));
        }

        if (_sparseFfn is { } encoderSparse)
        {
            throw new ModelException(
                $"architecture '{Plan.Architecture}' declares a sparse-FFN block with {encoderSparse.ExpertCount} experts " +
                $"routing {encoderSparse.RoutedExpertsPerToken} per token: the encoder forward runs dense gate/up/down blocks only");
        }

        int qkvRegion = Plan.HeadCount * Plan.HeadDim;
        int kvRegion = Plan.KvHeadCount * Plan.HeadDim;
        int inter = Plan.IntermediateSize;
        float attnScale = _attnScale;

        ArrayPool<float> poolBuffers = ArrayPool<float>.Shared;

        // Position-major working streams for the whole sequence plus chunk-local scratch for batched GEMVs. The rented
        // arrays are returned before any throw past this point, so a mid-call failure cannot leak pool entries.
        float[] cur = poolBuffers.Rent(n * hidden);          // residual carrier per position; becomes the final stream
        float[] normed = poolBuffers.Rent(n * hidden);       // rmsnorm result feeding the QKV / FFN GEMVs
        float[] qAll = poolBuffers.Rent(n * qkvRegion);      // every position's rotated queries, kept for attention
        float[] attnOut = poolBuffers.Rent(n * qkvRegion);   // concatenated heads per position before wo
        float[] chunkNormed = poolBuffers.Rent(EncodeChunkTokens * hidden);
        float[] chunkQ = poolBuffers.Rent(EncodeChunkTokens * qkvRegion);
        float[] chunkK = poolBuffers.Rent(EncodeChunkTokens * kvRegion);
        float[] chunkV = poolBuffers.Rent(EncodeChunkTokens * kvRegion);
        float[] chunkProj = poolBuffers.Rent(EncodeChunkTokens * hidden);
        float[] chunkGateUp = poolBuffers.Rent(EncodeChunkTokens * inter);
        float[] chunkUp = poolBuffers.Rent(EncodeChunkTokens * inter);

        try
        {
            // Embedding lookup: each token's dequantized token_embd row seeds that position's input stream.
            ReadOnlySpan<byte> embBytes = _weights.Resolve(_tokenEmbd);
            WeightRef embRef = new(embBytes, _tokenEmbd.Shape, _tokenEmbd.Format);
            for (int i = 0; i < n; i++)
            {
                if ((uint)tokens[i] >= (uint)Plan.VocabSize)
                {
                    throw new ArgumentException($"token id {tokens[i]} at index {i} is outside the vocabulary of {Plan.VocabSize}.", nameof(tokens));
                }

                _backend.DequantRow(embRef, tokens[i], cur.AsSpan(i * hidden, hidden));
            }

            ApplyInputEmbeddingScale(cur.AsSpan(0, n * hidden));

            // Reserve every position's KV slot up front so later layer passes address them by absolute position.
            for (int i = 0; i < n; i++)
            {
                kv.Append();
            }

            int chunkCount = (n + EncodeChunkTokens - 1) / EncodeChunkTokens;

            for (int l = 0; l < Plan.LayerCount; l++)
            {
                int layerWindow = LayerAttentionWindow(l);

                // Phase 1: pre-attention norm, QKV GEMVs (+bias), RoPE at absolute positions, and the K/V writes — all
                // per chunk. Queries stay in qAll so no position's attention runs before every position's K/V exists.
                for (int c = 0; c < chunkCount; c++)
                {
                    int firstPos = basePosition + c * EncodeChunkTokens;
                    int m = Math.Min(EncodeChunkTokens, n - c * EncodeChunkTokens);

                    for (int i = 0; i < m; i++)
                    {
                        ApplyNorm(cur.AsSpan((c * EncodeChunkTokens + i) * hidden, hidden), _attnNormWeight[l], _attnNormBias[l], chunkNormed.AsSpan(i * hidden, hidden));
                    }

                    _backend.GemvBatch(chunkNormed.AsSpan(0, m * hidden), m, AttentionQueryWeight(l), chunkQ.AsSpan(0, m * qkvRegion));

                    _backend.GemvBatch(chunkNormed.AsSpan(0, m * hidden), m, AttentionKeyWeight(l), chunkK.AsSpan(0, m * kvRegion));

                    _backend.GemvBatch(chunkNormed.AsSpan(0, m * hidden), m, AttentionValueWeight(l), chunkV.AsSpan(0, m * kvRegion));

                    for (int i = 0; i < m; i++)
                    {
                        AddBias(chunkQ.AsSpan(i * qkvRegion, qkvRegion), _biasQ[l]);
                        AddBias(chunkK.AsSpan(i * kvRegion, kvRegion), _biasK[l]);
                        AddBias(chunkV.AsSpan(i * kvRegion, kvRegion), _biasV[l]);

                        ApplyQkNorm(chunkQ.AsSpan(i * qkvRegion, qkvRegion), _attnQNorm[l]);
                        ApplyQkNorm(chunkK.AsSpan(i * kvRegion, kvRegion), _attnKNorm[l]);

                        ApplyRopeHeads(chunkQ.AsSpan(i * qkvRegion, qkvRegion), Plan.HeadCount, firstPos + i, windowedLayer: layerWindow > 0);
                        ApplyRopeHeads(chunkK.AsSpan(i * kvRegion, kvRegion), Plan.KvHeadCount, firstPos + i, windowedLayer: layerWindow > 0);

                        int position = c * EncodeChunkTokens + i; // sequence-relative index into the streams
                        kv.WriteK(basePosition + position, l, chunkK.AsSpan(i * kvRegion, kvRegion));
                        kv.WriteV(basePosition + position, l, chunkV.AsSpan(i * kvRegion, kvRegion));

                        chunkQ.AsSpan(i * qkvRegion, qkvRegion).CopyTo(qAll.AsSpan(position * qkvRegion, qkvRegion));
                    }
                }

                // Phase 2: attention per chunk — every query scores against this sequence's finished KV, over all n
                // positions on a layer whose resolved window is zero and over [max(0, q - window + 1), q] on one that
                // resolves a positive window. Chunk-local outputs land at each position's slot in attnOut.
                for (int c = 0; c < chunkCount; c++)
                {
                    int m = Math.Min(EncodeChunkTokens, n - c * EncodeChunkTokens);

                    LayerOps.BidirectionalAttention(
                        _backend,
                        qAll.AsSpan(c * EncodeChunkTokens * qkvRegion, m * qkvRegion),
                        queryCount: m,
                        queryPositionOffset: basePosition + c * EncodeChunkTokens,
                        fromPosition: 0,
                        toPosition: basePosition + n - 1,
                        Plan.HeadCount,
                        Plan.KvHeadCount,
                        Plan.HeadDim,
                        kv,
                        l,
                        attnScale,
                        Plan.AttentionSoftcap,
                        layerWindow,
                        attnOut.AsSpan(c * EncodeChunkTokens * qkvRegion, m * qkvRegion));
                }

                // Phase 3: output projection (+bias) and the ffn_inp residual add on each position's own stream.
                for (int c = 0; c < chunkCount; c++)
                {
                    int m = Math.Min(EncodeChunkTokens, n - c * EncodeChunkTokens);

                    ReadOnlySpan<byte> bytes = _weights.Resolve(_wo[l]);
                    _backend.GemvBatch(attnOut.AsSpan(c * EncodeChunkTokens * qkvRegion, m * qkvRegion), m, new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), chunkProj.AsSpan(0, m * hidden));

                    for (int i = 0; i < m; i++)
                    {
                        AddBias(chunkProj.AsSpan(i * hidden, hidden), _biasO[l]);

                        // Post-sublayer placement: this position's attention output joins its stream only after the post-attention norm.
                        ApplyPostSublayerNorm(chunkProj.AsSpan(i * hidden, hidden), _postAttnNorm[l]);

                        Span<float> posCur = cur.AsSpan((c * EncodeChunkTokens + i) * hidden, hidden);
                        ReadOnlySpan<float> proj = chunkProj.AsSpan(i * hidden, hidden);
                        for (int x = 0; x < hidden; x++)
                        {
                            posCur[x] += proj[x];
                        }

                        ApplyNorm(posCur, _ffnNormWeight[l], _ffnNormBias[l], normed.AsSpan((c * EncodeChunkTokens + i) * hidden, hidden));
                    }
                }

                // Phase 4: gate/up GEMVs and the gated activation per chunk.
                for (int c = 0; c < chunkCount; c++)
                {
                    int m = Math.Min(EncodeChunkTokens, n - c * EncodeChunkTokens);

                    _backend.GemvBatch(normed.AsSpan(c * EncodeChunkTokens * hidden, m * hidden), m, FeedForwardGateWeight(l), chunkGateUp.AsSpan(0, m * inter));

                    _backend.GemvBatch(normed.AsSpan(c * EncodeChunkTokens * hidden, m * hidden), m, FeedForwardUpWeight(l), chunkUp.AsSpan(0, m * inter));

                    for (int i = 0; i < m; i++)
                    {
                        ApplyActivation(chunkGateUp.AsSpan(i * inter, inter), chunkUp.AsSpan(i * inter, inter));
                    }

                    // Phase 5: down GEMV and the ffn_out residual add, completing this layer for every position.
                    ReadOnlySpan<byte> bytes = _weights.Resolve(_ffnDown[l]);
                    _backend.GemvBatch(chunkGateUp.AsSpan(0, m * inter), m, new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), chunkProj.AsSpan(0, m * hidden));

                    for (int i = 0; i < m; i++)
                    {
                        // Post-sublayer placement: this position's feed-forward output joins its stream only after the post-feed-forward norm.
                        ApplyPostSublayerNorm(chunkProj.AsSpan(i * hidden, hidden), _postFfnNorm[l]);

                        Span<float> posCur = cur.AsSpan((c * EncodeChunkTokens + i) * hidden, hidden);
                        ReadOnlySpan<float> proj = chunkProj.AsSpan(i * hidden, hidden);
                        for (int x = 0; x < hidden; x++)
                        {
                            posCur[x] += proj[x]; // ffn_out + ffn_inp: this position's input stream for the next layer
                        }
                    }
                }
            }

            float[] encoded = new float[n * hidden]; // caller-owned result, exact size — pooling helpers consume it row-wise
            for (int i = 0; i < n; i++)
            {
                ApplyNorm(cur.AsSpan(i * hidden, hidden), _outputNormWeight, _outputNormBias, encoded.AsSpan(i * hidden, hidden));
            }

            return encoded;
        }
        finally
        {
            poolBuffers.Return(cur, clearArray: false);
            poolBuffers.Return(normed, clearArray: false);
            poolBuffers.Return(qAll, clearArray: false);
            poolBuffers.Return(attnOut, clearArray: false);
            poolBuffers.Return(chunkNormed, clearArray: false);
            poolBuffers.Return(chunkQ, clearArray: false);
            poolBuffers.Return(chunkK, clearArray: false);
            poolBuffers.Return(chunkV, clearArray: false);
            poolBuffers.Return(chunkProj, clearArray: false);
            poolBuffers.Return(chunkGateUp, clearArray: false);
            poolBuffers.Return(chunkUp, clearArray: false);
        }
    }

    public void Dispose()
    {
        _weights.Dispose();
    }

    private void ForwardToken(int tokenId, int position, SequenceKv kv)
    {
        if ((uint)tokenId >= (uint)Plan.VocabSize)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenId), $"token id {tokenId} is outside the vocabulary of {Plan.VocabSize}.");
        }

        if (position >= MaxContextTokens)
        {
            throw new InvalidOperationException(
                $"sequence position {position} exceeds this model's max context of {MaxContextTokens} tokens.");
        }

        // Embedding lookup: one dequantized row of token_embd becomes the layer input stream.
        ReadOnlySpan<byte> embBytes = _weights.Resolve(_tokenEmbd);
        _backend.DequantRow(new WeightRef(embBytes, _tokenEmbd.Shape, _tokenEmbd.Format), tokenId, _cur.AsSpan(0, Plan.HiddenSize));

        ApplyInputEmbeddingScale(_cur.AsSpan(0, Plan.HiddenSize));

        kv.Append(); // reserves this position's slot in every layer; WriteK/WriteV address it by logical position

        int hidden = Plan.HiddenSize;
        float attnScale = _attnScale;

        for (int l = 0; l < Plan.LayerCount; l++)
        {
            int layerWindow = LayerAttentionWindow(l);

            // Pre-attention norm on the stream: normed = rmsnorm(cur) * attn_norm weight.
            ApplyNorm(_cur.AsSpan(0, hidden), _attnNormWeight[l], _attnNormBias[l], _normed);

            _backend.Gemv(_normed.AsSpan(0, hidden), AttentionQueryWeight(l), _q);
            AddBias(_q, _biasQ[l]);

            _backend.Gemv(_normed.AsSpan(0, hidden), AttentionKeyWeight(l), _k);
            AddBias(_k, _biasK[l]);

            _backend.Gemv(_normed.AsSpan(0, hidden), AttentionValueWeight(l), _v);
            AddBias(_v, _biasV[l]);

            // Resolved QK-norm mode: each side's projected stream is normalized after the projection and bias and
            // before RoPE, either per head in its own head-dimension slice or across all its heads under one mean.
            ApplyQkNorm(_q, _attnQNorm[l]);
            ApplyQkNorm(_k, _attnKNorm[l]);

            ApplyRopeHeads(_q, Plan.HeadCount, position, windowedLayer: layerWindow > 0);
            ApplyRopeHeads(_k, Plan.KvHeadCount, position, windowedLayer: layerWindow > 0);

            kv.WriteK(position, l, _k);
            kv.WriteV(position, l, _v);

            LayerOps.CausalAttention(_backend, _q, Plan.HeadCount, Plan.KvHeadCount, Plan.HeadDim, kv, l, position, attnScale, Plan.AttentionSoftcap, layerWindow, _attnOut);

            // Output projection (+bias) into scratch; the residual added before the FFN norm is the pre-attention
            // stream value (ffn_inp = attn_out + inpSA), matching the reference graph. cur held that stream through
            // attention untouched, so adding proj in place turns it into ffn_inp. Under a post-sublayer-output
            // placement the attention output — including any output bias — is normalized with its own vector first.
            ReadOnlySpan<byte> bytes = _weights.Resolve(_wo[l]);
            _backend.Gemv(_attnOut.AsSpan(0, Plan.HeadCount * Plan.HeadDim), new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), _proj);
            AddBias(_proj, _biasO[l]);

            ApplyPostSublayerNorm(_proj, _postAttnNorm[l]);

            for (int i = 0; i < hidden; i++)
            {
                _cur[i] += _proj[i];
            }

            ApplyNorm(_cur.AsSpan(0, hidden), _ffnNormWeight[l], _ffnNormBias[l], _normed);

            if (_sparseFfn is { } routing)
            {
                ApplySparseFeedForward(routing, l);
            }
            else
            {
                _backend.Gemv(_normed.AsSpan(0, hidden), FeedForwardGateWeight(l), _gateUp);

                _backend.Gemv(_normed.AsSpan(0, hidden), FeedForwardUpWeight(l), _up);

                // act = activation(gate) * up with the plan's resolved activation, applied in place on the gate buffer.
                ApplyActivation(_gateUp.AsSpan(0, Plan.IntermediateSize), _up.AsSpan(0, Plan.IntermediateSize));

                bytes = _weights.Resolve(_ffnDown[l]);
                _backend.Gemv(_gateUp.AsSpan(0, Plan.IntermediateSize), new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), _proj);
            }

            ApplyPostSublayerNorm(_proj, _postFfnNorm[l]);

            for (int i = 0; i < hidden; i++)
            {
                _cur[i] += _proj[i]; // ffn_out + ffn_inp: the input stream for the next layer
            }
        }

        ApplyNorm(_cur.AsSpan(0, hidden), _outputNormWeight, _outputNormBias, _normed);

        LogicalTensor lmHead = _output ?? _tokenEmbd; // tied weights reuse the token-embedding rows as logits
        ReadOnlySpan<byte> headBytes = _weights.Resolve(lmHead);
        _backend.Gemv(_normed.AsSpan(0, hidden), new WeightRef(headBytes, lmHead.Shape, lmHead.Format), _logits);

        if (_outputBias is not null)
        {
            AddBias(_logits, _outputBias);
        }

        ApplyFinalLogitSoftcap(_logits);
    }

    /// <summary>Rotates every head region of data in place at the given position using the precomputed tables, with the pairing the plan's resolved RoPE strategy selects and the rotary base this layer resolves.</summary>
    private void ApplyRopeHeads(Span<float> data, int headCount, int position, bool windowedLayer)
    {
        // A model whose windowed layers resolve a second rotary base carries that pair here; every other model leaves it
        // null and rotates at its one base whatever the flag says.
        float[] cosTable = windowedLayer && _cosWindowed is { } windowedCos ? windowedCos : _cos;
        float[] sinTable = windowedLayer && _sinWindowed is { } windowedSin ? windowedSin : _sin;

        ReadOnlySpan<float> cosAtPosition = cosTable.AsSpan(position * _ropeHalf, _ropeHalf);
        ReadOnlySpan<float> sinAtPosition = sinTable.AsSpan(position * _ropeHalf, _ropeHalf);

        for (int h = 0; h < headCount; h++)
        {
            // Pairs are relative to each head's own region.
            Span<float> region = data.Slice(h * Plan.HeadDim, Plan.HeadDim);
            switch (Plan.Rope)
            {
                case RopeStrategy.StandardFullHead:
                    LayerOps.ApplyRopeToHead(region, _ropeHalf, cosAtPosition, sinAtPosition);
                    break;
                case RopeStrategy.RotateHalfFullHead:
                    LayerOps.ApplyRopeToSplitHalfHead(region, _ropeHalf, cosAtPosition, sinAtPosition);
                    break;
                default:
                    throw new ModelException($"rope strategy '{Plan.Rope}' is not implemented by the forward engine");
            }
        }
    }

    private static void AddBias(Span<float> values, float[]? bias)
    {
        if (bias is null)
        {
            return;
        }

        for (int i = 0; i < values.Length; i++)
        {
            values[i] += bias[i];
        }
    }

    // Norm and activation run as a direct call chosen from the plan's strategy values: no interface dispatch and
    // no allocation anywhere in the forward, so every path — single token, batched decode, and encode — applies
    // the resolved strategy identically. The bias operand is the layer's declared norm-bias vector when one was
    // loaded (an empty span otherwise) and is consumed only by the LayerNorm strategy, which takes its epsilon from
    // the plan's block-norm epsilon while the RMS strategy keeps the plan's RMS epsilon.
    private void ApplyNorm(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias, Span<float> dst)
    {
        switch (Plan.Norm)
        {
            case NormStrategy.RmsNorm:
                LayerOps.RmsNormWeighted(src, weight, Plan.EpsRmsNorm, dst);
                break;
            case NormStrategy.LayerNorm:
                LayerOps.LayerNormWeighted(src, weight, bias, Plan.EpsLayerNorm, dst);
                break;
            default:
                throw new ModelException($"norm strategy '{Plan.Norm}' is not implemented by the forward engine");
        }
    }

    // The resolved QK-norm form runs as a direct call chosen from the plan's mode, exactly as the norm and activation
    // strategies do: no interface dispatch and no allocation anywhere in the forward, so every path — single token,
    // batched decode, and encode — applies the resolved form identically between each head's projection (plus any bias)
    // and its rotation. The per-head weighted variant consumes the layer's head-dimension vector over each head region;
    // the stream-weighted variant consumes one whole-projection vector and normalizes every head of q (or of k) under a
    // single mean across that side's full width, which is what the reference does before splitting its projection into
    // heads. The unweighted variant normalizes with the plan epsilon alone, which is numerically identical to a weight
    // of ones and reads nothing from the file; off leaves every head region untouched. Each form normalizes in place: it finishes reading a value before writing back at that index.
    private void ApplyQkNorm(Span<float> heads, ReadOnlySpan<float> weight)
    {
        switch (Plan.QkNorm)
        {
            case QkNormMode.PerHeadWeighted:
                LayerOps.RmsNormWeightedHeads(heads, weight, Plan.HeadDim, Plan.EpsRmsNorm, heads);
                break;

            case QkNormMode.StreamWeighted:
                // One mean over the whole q (or k) projection and one element-by-element vector covering that same
                // width: every head is scaled by its own weights but normalized jointly with the other heads.
                LayerOps.RmsNormWeighted(heads, weight, Plan.EpsRmsNorm, heads);
                break;

            case QkNormMode.PerHeadUnweighted:
                LayerOps.RmsNormUnweightedHeads(heads, Plan.HeadDim, Plan.EpsRmsNorm, heads);
                break;

            case QkNormMode.Off:
                return;

            default:
                throw new ModelException($"QK-norm mode '{Plan.QkNorm}' is not implemented by the forward engine");
        }
    }

    // The second norm a sublayer output may receive runs as a direct branch on the plan's resolved placement, the way the
    // softcaps do. Under PreOnly nothing here reads anything and each output joins the residual stream unchanged; under
    // PostSublayerOutputs that sublayer's own norm vector normalizes its output in place — the attention output after its
    // projection and any output bias, or the feed-forward output after the down GEMV — before its residual add, through
    // the family's resolved norm kind with no bias operand, since no post-sublayer bias tensor exists in the canonical map.
    // In place is safe for both norm kinds: each finishes reading a value before writing back at that index.
    // The input-embedding gain runs as a direct branch on the plan's resolved value, the same way the norm and
    // activation strategies do: a resolved gain of zero leaves every embedding row untouched. A non-zero gain is
    // the sqrt(hidden size) the gemma-family reference graphs multiply each embedding row by before the first
    // layer, applied in place to the freshly dequantized rows on every forward path — single token, batched
    // decode, and encode — so each path seeds its residual stream identically.
    private void ApplyInputEmbeddingScale(Span<float> embeddings)
    {
        float scale = Plan.InputEmbeddingScale;
        if (scale == 0f)
        {
            return;
        }

        for (int i = 0; i < embeddings.Length; i++)
        {
            embeddings[i] *= scale;
        }
    }

    private void ApplyPostSublayerNorm(Span<float> output, ReadOnlySpan<float> weight)
    {
        switch (Plan.NormPlacement)
        {
            case NormPlacement.PreOnly:
                return;

            case NormPlacement.PostSublayerOutputs:
                ApplyNorm(output, weight, ReadOnlySpan<float>.Empty, output);
                break;

            default:
                throw new ModelException($"norm placement '{Plan.NormPlacement}' is not implemented by the forward engine");
        }
    }

    // The window one layer reads within comes from the plan's resolved per-layer value, called directly the way the
    // norm, activation, and softcap switches do: no dispatch and nothing resolved per token or per head. lair.kv then
    // turns that value into a block selection at each attention call — see LayerOps.CausalAttention and
    // LayerOps.BidirectionalAttention — so the windowed layers read only the blocks their backward span covers while
    // the unbounded ones select the whole table, on both the causal decode path and the encoder path. Storage is not
    // narrowed: every position's K/V is still written and stays readable for whichever layer attends it.
    private int LayerAttentionWindow(int layer) => Plan.AttentionWindowForLayer(layer);

    // How many layers the plan's resolved interleave narrows to its window: the only case in which a second rotary base
    // reaches anything, since an unbounded layer rotates at the plan's own base whatever that second base declares.
    private int CountWindowedLayers()
    {
        int count = 0;

        for (int layer = 0; layer < Plan.LayerCount; layer++)
        {
            if (Plan.AttentionWindowForLayer(layer) > 0)
            {
                count++;
            }
        }

        return count;
    }

    // The angle tables one rotary base rotates at, built the way this model's positional shape says — a resolved scaling
    // plan over the declared slice, an unscaled plan across the full head — so every pair this engine carries differs from
    // its neighbour only in the base it was built for.
    private void BuildRotationTables(float theta, int maxContextTokens, float[] cos, float[] sin)
    {
        if (Plan.RopeScaling is { } scaling)
        {
            LayerOps.BuildRopeScalingTables(
                theta,
                Plan.HeadDim,
                2 * _ropeHalf,
                maxContextTokens,
                scaling.Multiplier,
                scaling.OriginalContextLength,
                scaling.AttentionFactor,
                scaling.CorrectionScale,
                scaling.BetaFast,
                scaling.BetaSlow,
                cos,
                sin);
        }
        else
        {
            LayerOps.BuildRopeTables(theta, Plan.HeadDim, maxContextTokens, cos, sin);
        }
    }

    private void ApplyActivation(Span<float> gate, ReadOnlySpan<float> up)
    {
        switch (Plan.Activation)
        {
            case ActivationStrategy.SwiGLU:
                for (int i = 0; i < gate.Length; i++)
                {
                    gate[i] = LayerOps.Silu(gate[i]) * up[i];
                }

                break;
            case ActivationStrategy.GeGlu:
                for (int i = 0; i < gate.Length; i++)
                {
                    gate[i] = LayerOps.GeluTanh(gate[i]) * up[i];
                }

                break;
            case ActivationStrategy.GeGluExact:
                for (int i = 0; i < gate.Length; i++)
                {
                    gate[i] = LayerOps.Gelu(gate[i]) * up[i];
                }

                break;
            default:
                throw new ModelException($"activation strategy '{Plan.Activation}' is not implemented by the forward engine");
        }
    }

    // The final-logit softcap runs as a direct branch on the plan's resolved value, the same way the norm and
    // activation strategies do: a resolved cap of zero is disabled and leaves the vector untouched. It is applied
    // to every produced logit vector — untied output head or tied token-embedding rows, single-token or batched —
    // after the full vector including any output bias is computed and before any softmax, sampling, or copy-out.
    private void ApplyFinalLogitSoftcap(Span<float> logits)
    {
        float cap = Plan.FinalLogitSoftcap;
        if (cap <= 0f)
        {
            return;
        }

        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] = LayerOps.Softcap(logits[i], cap);
        }
    }

    // Loads one per-layer matrix role. When a fused qkv template is given for a q/k/v role and the layer
    // carries the fused tensor alongside the resolved split tensor, the fused layout is not consumed — the
    // split map already fills the role — and the name is recorded for the one structured tensors_ignored line.
    private LogicalTensor[] LoadMatrices(int layers, TensorRole role, string? fusedTemplate = null)
    {
        LogicalTensor[] tensors = new LogicalTensor[layers];
        for (int l = 0; l < layers; l++)
        {
            tensors[l] = RequireTensor(role, l);

            if (fusedTemplate is not null)
            {
                string fused = FusedName(fusedTemplate, l);
                if (_fileTensors.Find(fused) is not null)
                {
                    _ignoredFusedTensors.Add(fused);
                }
            }
        }

        return tensors;
    }

    // Loads every layer's fused qkv buffer and pins each head region's byte geometry against the declared dimensions: that
    // one matrix carries the query region (head_count × head_dim rows), then the key region (kv_head_count × head_dim
    // rows), then value again, over columns of the hidden size. Row packing is per row in every format these kernels
    // implement, so a region boundary lands on whole rows and each role reads its own window of one payload — no copy of
    // any side exists anywhere on this path. A buffer whose shape or payload disagrees with those regions fails here naming
    // the tensor and both sides of the mismatch; a layer carrying none of it while its three split matrices are present
    // fails there instead.
    private void LoadFusedQkvBuffers(int layers)
    {
        long queryRows = (long)Plan.HeadCount * Plan.HeadDim;
        long keyValueRows = (long)Plan.KvHeadCount * Plan.HeadDim;

        _qkvQueryShape = new TensorShape([checked((int)queryRows), Plan.HiddenSize]);
        _qkvKeyValueShape = new TensorShape([checked((int)keyValueRows), Plan.HiddenSize]);

        long totalRows = queryRows + 2 * keyValueRows;
        if (totalRows > int.MaxValue)
        {
            throw new ModelException(
                $"architecture '{Plan.Architecture}' declares head_count {Plan.HeadCount} and kv_head_count " +
                $"{Plan.KvHeadCount} at head dimension {Plan.HeadDim}: a fused qkv buffer of {totalRows} rows is beyond one addressable tensor");
        }

        _wqkv = new LogicalTensor[layers];
        long[] queryBytes = new long[layers];
        long[] keyBytes = new long[layers];

        for (int l = 0; l < layers; l++)
        {
            LogicalTensor tensor = RequireTensor(TensorRole.AttentionQkv, l);

            int[] dims = tensor.Shape.Dims;
            if (dims.Length != 2 || dims[0] != totalRows || dims[^1] != Plan.HiddenSize)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' must have shape [{totalRows}, {Plan.HiddenSize}] — the query region of " +
                    $"{queryRows} rows, then the key and value regions of {keyValueRows} rows each, over the hidden size — got [{string.Join(", ", dims)}]");
            }

            long stride = tensor.Shape.RowStrideBytes(tensor.Format);
            long payload = totalRows * stride;
            if (payload != tensor.SizeBytes)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' declares {tensor.SizeBytes} payload bytes but its shape and format require {payload}");
            }

            queryBytes[l] = checked(queryRows * stride);
            keyBytes[l] = checked(keyValueRows * stride);
            if (queryBytes[l] > int.MaxValue || keyBytes[l] > int.MaxValue)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' has a head region of {Math.Max(queryBytes[l], keyBytes[l])} bytes, beyond a single addressable slice");
            }

            _wqkv[l] = tensor;
        }

        _qkvQueryBytes = queryBytes;
        _qkvKeyBytes = keyBytes;
    }

    // Loads every layer's fused gate/up buffer for a block whose gate half lives inside it: one matrix carrying both
    // halves' rows — the gate side first, then the up side — over columns of the hidden size. Each half is validated as
    // exactly what the split triplet's gate and up matrices would have been, so the activation and down projection see the
    // same intermediate-sized stream either way, and both GEMVs read their own window of one payload.
    private void LoadFusedGateUpBuffers(int layers)
    {
        long halfRows = Plan.IntermediateSize;
        long totalRows = 2 * halfRows;

        _gateUpHalfShape = new TensorShape([Plan.IntermediateSize, Plan.HiddenSize]);

        _ffnUp = new LogicalTensor[layers];
        long[] halfBytes = new long[layers];

        for (int l = 0; l < layers; l++)
        {
            LogicalTensor tensor = RequireTensor(TensorRole.FeedForwardUp, l);

            int[] dims = tensor.Shape.Dims;
            if (dims.Length != 2 || dims[0] != totalRows || dims[^1] != Plan.HiddenSize)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' must have shape [{totalRows}, {Plan.HiddenSize}] — the gate half of " +
                    $"{halfRows} rows followed by the up half, over the hidden size — got [{string.Join(", ", dims)}]");
            }

            long stride = tensor.Shape.RowStrideBytes(tensor.Format);
            long payload = totalRows * stride;
            if (payload != tensor.SizeBytes)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' declares {tensor.SizeBytes} payload bytes but its shape and format require {payload}");
            }

            halfBytes[l] = checked(halfRows * stride);
            if (halfBytes[l] > int.MaxValue)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' has a gate/up half of {halfBytes[l]} bytes, beyond a single addressable slice");
            }

            _ffnUp[l] = tensor;
        }

        _gateUpHalfBytes = halfBytes;
    }

    // The weight operand one projection reads on this layer: its own tensor when the plan resolves a matrix per role, or
    // that role's region of the shared buffer when it resolves a fused layout. Both sides address packed bytes in place —
    // a slice is a window over the resolved payload, never a copy — and every byte range was pinned at Open against this
    // layer's own format and row stride.
    private WeightRef OwnWeight(LogicalTensor tensor) => new(_weights.Resolve(tensor), tensor.Shape, tensor.Format);

    private static WeightRef SliceWeight(ReadOnlySpan<byte> payload, LogicalTensor tensor, long offset, long length, TensorShape shape) =>
        new(payload.Slice((int)offset, (int)length), shape, tensor.Format);

    private WeightRef AttentionQueryWeight(int layer) =>
        Plan.AttentionWeights == QkvLayout.FusedBufferSlices
            ? SliceWeight(_weights.Resolve(_wqkv[layer]), _wqkv[layer], 0, _qkvQueryBytes[layer], _qkvQueryShape)
            : OwnWeight(_wq[layer]);

    private WeightRef AttentionKeyWeight(int layer) =>
        Plan.AttentionWeights == QkvLayout.FusedBufferSlices
            ? SliceWeight(_weights.Resolve(_wqkv[layer]), _wqkv[layer], _qkvQueryBytes[layer], _qkvKeyBytes[layer], _qkvKeyValueShape)
            : OwnWeight(_wk[layer]);

    // The value region follows the key region inside the same buffer, and both carry the same rows of the same format —
    // one head slice per kv head — so only its offset differs from the key side's.
    private WeightRef AttentionValueWeight(int layer) =>
        Plan.AttentionWeights == QkvLayout.FusedBufferSlices
            ? SliceWeight(_weights.Resolve(_wqkv[layer]), _wqkv[layer], _qkvQueryBytes[layer] + _qkvKeyBytes[layer], _qkvKeyBytes[layer], _qkvKeyValueShape)
            : OwnWeight(_wv[layer]);

    private WeightRef FeedForwardGateWeight(int layer) =>
        Plan.GateSource == FeedForwardGateSource.UpTensorHalves
            ? SliceWeight(_weights.Resolve(_ffnUp[layer]), _ffnUp[layer], 0, _gateUpHalfBytes[layer], _gateUpHalfShape)
            : OwnWeight(_ffnGate[layer]);

    private WeightRef FeedForwardUpWeight(int layer) =>
        Plan.GateSource == FeedForwardGateSource.UpTensorHalves
            ? SliceWeight(_weights.Resolve(_ffnUp[layer]), _ffnUp[layer], _gateUpHalfBytes[layer], _gateUpHalfBytes[layer], _gateUpHalfShape)
            : OwnWeight(_ffnUp[layer]);

    private float[][] LoadLayerVectors(int layers, TensorRole role, int expectedLength)
    {
        float[][] vectors = new float[layers][];
        for (int l = 0; l < layers; l++)
        {
            LogicalTensor tensor = RequireTensor(role, l);
            vectors[l] = LoadVector(tensor, expectedLength);
        }

        return vectors;
    }

    // Norm and bias tensors are one-dimensional in GGUF. Addressing them as a single-row matrix lets the backend's
    // row dequantization handle any stored format (F32 for most conversions, F16 elsewhere) with no extra API.
    private float[] LoadVector(LogicalTensor tensor, int expectedLength)
    {
        if (tensor.Shape.Dims.Length != 1 || tensor.Shape.Dims[0] != expectedLength)
        {
            throw new ModelException(
                $"tensor '{tensor.Name}' must have shape [{expectedLength}], got [{string.Join(", ", tensor.Shape.Dims)}]");
        }

        TensorShape row = new([1, expectedLength]);
        ReadOnlySpan<byte> bytes = _weights.Resolve(tensor);
        float[] values = new float[expectedLength];
        _backend.DequantRow(new WeightRef(bytes, row, tensor.Format), 0, values);
        return values;
    }

    // Loads one per-layer stacked expert role and validates it against the descriptor's counts: the logical shape
    // carries the declared expert count as its outermost dimension followed by this matrix's rows and columns, and
    // the payload holds exactly that many packed rows so a single expert's slice stays inside the tensor. A stack
    // whose per-expert block is too large for one addressable slice fails here rather than mis-slicing at forward time.
    private LogicalTensor[] LoadExpertStacks(int layers, TensorRole role, int experts, long rows, long cols)
    {
        LogicalTensor[] tensors = new LogicalTensor[layers];

        for (int l = 0; l < layers; l++)
        {
            LogicalTensor tensor = RequireTensor(role, l);
            int[] dims = tensor.Shape.Dims;
            if (dims.Length != 3 || dims[0] != experts || dims[1] != rows || dims[^1] != cols)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' must have shape [{experts}, {rows}, {cols}], got [{string.Join(", ", dims)}]");
            }

            long stride = tensor.Shape.RowStrideBytes(tensor.Format);
            long payload = (long)dims[0] * rows * stride;
            if (payload != tensor.SizeBytes)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' declares {tensor.SizeBytes} payload bytes but its shape and format require {payload}");
            }

            long expertBlock = rows * stride;
            if (expertBlock > int.MaxValue)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' has an expert block of {expertBlock} bytes, beyond a single addressable slice");
            }

            tensors[l] = tensor;
        }

        return tensors;
    }

    // A descriptor that declares a shared expert names all three matrices together at plan resolution. This reads the
    // declared role and fails on any half-set left unset, so a malformed descriptor never reaches execution; an
    // undeclared shared expert loads nothing here at all.
    private static TensorRole RequiredSparseRole(TensorRole? role, string side)
    {
        if (role is not { } resolved)
        {
            throw new ModelException($"the sparse-FFN descriptor declares a shared expert without naming its {side} matrix");
        }

        return resolved;
    }

    // Loads every layer's shared-expert selector vector, in the presence discipline the optional tensors use: a file
    // shipping none of them loads nothing and the shared block keeps its ungated accumulation, a file shipping one for
    // some layers only fails naming the first missing layer instead of gating half the block, and a complete table is
    // dequantized once here so forward time reads plain floats like every other vector role.
    private float[][] LoadSharedExpertSelectors(int layers, TensorDescriptor descriptor)
    {
        LogicalTensor?[] present = new LogicalTensor?[layers];
        bool anyPresent = false;
        int firstMissingLayer = -1;

        for (int l = 0; l < layers; l++)
        {
            present[l] = descriptor.Resolve(_fileTensors, l);
            if (present[l] is null)
            {
                if (firstMissingLayer < 0)
                {
                    firstMissingLayer = l;
                }
            }
            else
            {
                anyPresent = true;
            }
        }

        if (!anyPresent)
        {
            return [];
        }

        if (firstMissingLayer >= 0)
        {
            throw new ModelException(
                $"tensor '{descriptor.NameFor(firstMissingLayer)}' is missing: architecture '{Plan.Architecture}' " +
                "ships the shared-expert selector vector for some layers only");
        }

        float[][] vectors = new float[layers][];
        for (int l = 0; l < layers; l++)
        {
            vectors[l] = LoadVector(present[l]!, Plan.HiddenSize);
        }

        return vectors;
    }

    // A slice of one expert's resident stack as the weight operand a GEMV over that expert consumes: the contiguous
    // row range at this role's per-expert byte geometry, carrying the two-dimensional shape resolved for it at Open.
    private static WeightRef ExpertWeightRef(ReadOnlySpan<byte> payload, long expertBytes, int expertIndex, TensorShape shape, QuantFormat format) =>
        new(payload.Slice((int)(expertIndex * expertBytes), (int)expertBytes), shape, format);

    // Router scoring and top-k selection for one normalized activation vector, shared by both forward paths so a
    // batched step reproduces single-sequence routing exactly. The router emits one score per declared expert; every
    // score is softmaxed together because the reference routes over all experts, then RoutedExpertsPerToken picks run
    // as ascending-index scans using strict greater-than comparison — equal scores resolve to the lower expert index
    // deterministically, with no sort or dictionary stability assumption. A picked slot is poisoned so later passes
    // cannot re-pick it; selected slots are insertion-sorted into ascending expert-index execution order alongside
    // their weights. Normalization divides by the selected probability sum when declared, and a nonzero
    // ExpertWeightsScale multiplies every weight after that division. Scratch holds one routing at a time: batched
    // callers copy each selection before selecting the next sequence.
    private void RouteSparseExperts(SparseFfnDescriptor sparse, int layer, ReadOnlySpan<float> input)
    {
        LogicalTensor router = _moeRouter[layer];
        ReadOnlySpan<byte> bytes = _weights.Resolve(router);
        Span<float> scores = _moeLogits.AsSpan(0, sparse.ExpertCount);

        // The router emits one score per declared expert before any selection or softmax runs.
        _backend.Gemv(input, new WeightRef(bytes, router.Shape, router.Format), scores);

        float maxScore = float.NegativeInfinity;
        for (int e = 0; e < scores.Length; e++)
        {
            if (scores[e] > maxScore)
            {
                maxScore = scores[e];
            }
        }

        // All-expert softmax probabilities in place, summed and divided in ascending expert order: the reference
        // routes over every declared expert before selecting any of them. The max-subtraction guard keeps every
        // exponent in range.
        float allSum = 0f;
        for (int e = 0; e < scores.Length; e++)
        {
            scores[e] = MathF.Exp(scores[e] - maxScore);
            allSum += scores[e];
        }

        for (int e = 0; e < scores.Length; e++)
        {
            scores[e] /= allSum;
        }

        int routed = sparse.RoutedExpertsPerToken;
        float selectedSum = 0f;
        for (int pick = 0; pick < routed; pick++)
        {
            int best = -1;
            float bestScore = float.NegativeInfinity;
            for (int e = 0; e < scores.Length; e++)
            {
                // Strict '>' over ascending indexes: a tie keeps the earlier expert, so equal scores resolve to the
                // lower index regardless of call order or buffer reuse.
                if (scores[e] > bestScore)
                {
                    best = e;
                    bestScore = scores[e];
                }
            }

            _moeIds[pick] = best;
            _moeWeights[pick] = bestScore;
            selectedSum += bestScore;
            scores[best] = float.NegativeInfinity; // consumed: later passes cannot re-pick this expert
        }

        for (int i = 1; i < routed; i++)
        {
            int id = _moeIds[i];
            float weight = _moeWeights[i];
            int j = i - 1;
            while (j >= 0 && _moeIds[j] > id)
            {
                _moeIds[j + 1] = _moeIds[j];
                _moeWeights[j + 1] = _moeWeights[j];
                j--;
            }

            _moeIds[j + 1] = id;
            _moeWeights[j + 1] = weight;
        }

        if (sparse.NormalizeExpertWeights && selectedSum > 0f)
        {
            for (int i = 0; i < routed; i++)
            {
                _moeWeights[i] /= selectedSum;
            }
        }

        if (sparse.ExpertWeightsScale != 0f)
        {
            for (int i = 0; i < routed; i++)
            {
                _moeWeights[i] *= sparse.ExpertWeightsScale;
            }
        }
    }

    // Adds weight times source into target element-wise, in ascending order: the accumulation both forward paths use
    // for every selected expert's down output, so a batched step replays the single-token arithmetic exactly.
    private static void AccumulateScaled(Span<float> target, ReadOnlySpan<float> source, float weight)
    {
        for (int i = 0; i < source.Length; i++)
        {
            target[i] += weight * source[i];
        }
    }

    // The weight a declared shared expert contributes with. A block whose profile names no selector role, or whose
    // file ships none of those vectors, gets exactly 1f: the ungated accumulation this path has always used. When the
    // vectors are present the score is taken from them against this token's normalized stream and applied as
    // silu(score) / score to the shared output — the pinned qwen2moe graph builds it that way, as
    // `ggml_div(ggml_silu(cur_gate_inp), cur_gate_inp)` over a GEMV of ffn_gate_inp_shexp, which is sigmoid's value
    // read through its own silu rather than a separate node. The dot runs in ascending column order and both forward
    // paths reach it through this method with the same per-token stream, so batched steps stay bit-identical to
    // single-sequence ones. A score of exactly zero divides 0 by 0 as the reference graph does.
    private float SharedExpertWeight(int layer, ReadOnlySpan<float> normed)
    {
        if (_sharedInputGate.Length == 0)
        {
            return 1f; // no selector vector in this file: the shared block adds ungated, exactly as before
        }

        float[] selector = _sharedInputGate[layer];
        float score = 0f;
        for (int c = 0; c < selector.Length; c++)
        {
            score += selector[c] * normed[c];
        }

        return LayerOps.Silu(score) / score;
    }

    // Sparse feed-forward for the current token: routing selects, each selected expert's gate/up/down chain runs over
    // the normalized stream in ascending expert-index order, and every output accumulates into _proj scaled by its
    // routing weight — a declared shared expert contributing once under SharedExpertWeight, which is 1f unless the
    // file ships that block's selector vectors. The caller's post-feed-forward norm
    // and residual add then run exactly as they do on the dense path.
    private void ApplySparseFeedForward(SparseFfnDescriptor sparse, int layer)
    {
        RouteSparseExperts(sparse, layer, _normed.AsSpan(0, Plan.HiddenSize));

        int hidden = Plan.HiddenSize;
        Span<float> proj = _proj;
        proj.Clear(); // the routed and shared contributions accumulate from zero below

        for (int slot = 0; slot < sparse.RoutedExpertsPerToken; slot++)
        {
            ExecuteRoutedExpert(layer, _moeIds[slot], _moeSubsetIn.AsSpan(0, hidden));
            AccumulateScaled(proj, _moeSubsetIn.AsSpan(0, hidden), _moeWeights[slot]);
        }

        if (sparse.HasSharedExpert)
        {
            RunSharedExpertToken(layer, _moeSubsetIn.AsSpan(0, hidden));
            AccumulateScaled(proj, _moeSubsetIn.AsSpan(0, hidden), SharedExpertWeight(layer, _normed.AsSpan(0, hidden)));
        }
    }

    // One routed expert's gate/up/down chain over a single activation vector. The gate and up intermediates reuse the
    // dense token buffers; each GEMV reads this expert's contiguous row range of its resident stacked role at the
    // per-expert byte geometry resolved at Open, so no copy of any expert tensor exists anywhere.
    private void ExecuteRoutedExpert(int layer, int expertIndex, Span<float> output)
    {
        LogicalTensor gate = _moeExpertGate[layer];
        ReadOnlySpan<byte> payload = _weights.Resolve(gate);
        _backend.Gemv(_normed.AsSpan(0, Plan.HiddenSize), ExpertWeightRef(payload, _moeExpertGateBytes[layer], expertIndex, _moeExpertGateShape, gate.Format), _gateUp);

        LogicalTensor up = _moeExpertUp[layer];
        payload = _weights.Resolve(up);
        _backend.Gemv(_normed.AsSpan(0, Plan.HiddenSize), ExpertWeightRef(payload, _moeExpertUpBytes[layer], expertIndex, _moeExpertUpShape, up.Format), _up);

        ApplyActivation(_gateUp.AsSpan(0, Plan.IntermediateSize), _up.AsSpan(0, Plan.IntermediateSize));

        LogicalTensor down = _moeExpertDown[layer];
        payload = _weights.Resolve(down);
        _backend.Gemv(_gateUp.AsSpan(0, Plan.IntermediateSize), ExpertWeightRef(payload, _moeExpertDownBytes[layer], expertIndex, _moeExpertDownShape, down.Format), output);
    }

    // The shared expert's dense-shaped triplet over a single activation vector: the same gate/up/down GEMVs the dense
    // block runs, reading its own per-layer tensors and writing this token's contribution into the accumulation slot.
    private void RunSharedExpertToken(int layer, Span<float> output)
    {
        LogicalTensor gate = _sharedGate[layer];
        ReadOnlySpan<byte> bytes = _weights.Resolve(gate);
        _backend.Gemv(_normed.AsSpan(0, Plan.HiddenSize), new WeightRef(bytes, gate.Shape, gate.Format), _gateUp);

        LogicalTensor up = _sharedUp[layer];
        bytes = _weights.Resolve(up);
        _backend.Gemv(_normed.AsSpan(0, Plan.HiddenSize), new WeightRef(bytes, up.Shape, up.Format), _up);

        ApplyActivation(_gateUp.AsSpan(0, Plan.IntermediateSize), _up.AsSpan(0, Plan.IntermediateSize));

        LogicalTensor down = _sharedDown[layer];
        bytes = _weights.Resolve(down);
        _backend.Gemv(_gateUp.AsSpan(0, Plan.IntermediateSize), new WeightRef(bytes, down.Shape, down.Format), output);
    }

    // Sparse feed-forward for one batched step: every sequence routes first (one selection at a time in scratch,
    // copied into the per-sequence arrays before the next selection runs). Execution then walks selection slots in
    // ascending order and experts by index within each slot, packing that group's member rows of _batchNormed for one
    // GemvBatch pass through the expert's resident stack; results accumulate straight into each sequence's projection
    // stream under its routing weight. A declared shared expert runs last over all sequences, every sequence scaled by
    // its own SharedExpertWeight. The packed subset buffer carries activations and down results in place, so no copy of any expert tensor or whole-batch
    // output exists — and because every sequence accumulates its own contributions in the same slot order a single
    // token does, each batched result is bit-identical to stepping that sequence alone. The caller's post-feed-forward
    // norm and residual add then run exactly as they do on the dense path.
    private void ApplySparseFeedForwardBatch(SparseFfnDescriptor sparse, int layer, int n)
    {
        int hidden = Plan.HiddenSize;
        int routed = sparse.RoutedExpertsPerToken;

        for (int s = 0; s < n; s++)
        {
            RouteSparseExperts(sparse, layer, _batchNormed.AsSpan(s * hidden, hidden));

            for (int pick = 0; pick < routed; pick++)
            {
                _batchMoeIds[s * routed + pick] = _moeIds[pick];
                _batchMoeWeights[s * routed + pick] = _moeWeights[pick];
            }
        }

        Span<float> batchProj = _batchProj.AsSpan(0, n * hidden);
        batchProj.Clear(); // the routed and shared contributions accumulate from zero below

        for (int slot = 0; slot < routed; slot++)
        {
            for (int expertIndex = 0; expertIndex < sparse.ExpertCount; expertIndex++)
            {
                int count = 0;
                for (int s = 0; s < n; s++)
                {
                    if (_batchMoeIds[s * routed + slot] == expertIndex)
                    {
                        _expertMembers[count++] = s;
                    }
                }

                ExecuteRoutedExpertBatch(layer, expertIndex, count, slot, routed);
            }
        }

        if (sparse.HasSharedExpert)
        {
            RunSharedExpertBatch(layer, n, hidden);
        }
    }

    // One routed expert's gate/up/down chain over the packed member rows of one selection slot: a single batched pass
    // through its resident stack, bit-identical per row to the token path, with each member's down result accumulated
    // into that sequence's projection stream under its routing weight as soon as the group completes. _moeSubsetIn
    // carries this group's activations and then its contiguous down results in place; gate/up intermediates reuse the
    // dense batch buffers. A slot no sequence selected runs nothing at all.
    private void ExecuteRoutedExpertBatch(int layer, int expertIndex, int count, int slot, int routed)
    {
        if (count == 0)
        {
            return;
        }

        int hidden = Plan.HiddenSize;
        int inter = Plan.IntermediateSize;
        Span<float> subsetIn = _moeSubsetIn.AsSpan(0, count * hidden);

        for (int m = 0; m < count; m++)
        {
            _batchNormed.AsSpan(_expertMembers[m] * hidden, hidden).CopyTo(subsetIn.Slice(m * hidden, hidden));
        }

        LogicalTensor gate = _moeExpertGate[layer];
        ReadOnlySpan<byte> payload = _weights.Resolve(gate);
        _backend.GemvBatch(subsetIn, count, ExpertWeightRef(payload, _moeExpertGateBytes[layer], expertIndex, _moeExpertGateShape, gate.Format), _batchGateUp.AsSpan(0, count * inter));

        LogicalTensor up = _moeExpertUp[layer];
        payload = _weights.Resolve(up);
        _backend.GemvBatch(subsetIn, count, ExpertWeightRef(payload, _moeExpertUpBytes[layer], expertIndex, _moeExpertUpShape, up.Format), _batchUp.AsSpan(0, count * inter));

        for (int m = 0; m < count; m++)
        {
            ApplyActivation(_batchGateUp.AsSpan(m * inter, inter), _batchUp.AsSpan(m * inter, inter));
        }

        LogicalTensor down = _moeExpertDown[layer];
        payload = _weights.Resolve(down);
        _backend.GemvBatch(_batchGateUp.AsSpan(0, count * inter), count, ExpertWeightRef(payload, _moeExpertDownBytes[layer], expertIndex, _moeExpertDownShape, down.Format), subsetIn);

        for (int m = 0; m < count; m++)
        {
            int s = _expertMembers[m];
            AccumulateScaled(_batchProj.AsSpan(s * hidden, hidden), subsetIn.Slice(m * hidden, hidden), _batchMoeWeights[s * routed + slot]);
        }
    }

    // The shared expert's dense-shaped triplet over every sequence in one batched pass: the same gate/up/down
    // GemvBatch traversal the dense block runs, accumulating each sequence's output into its projection stream under
    // that sequence's SharedExpertWeight — 1f for a file shipping no selector vectors. An undeclared shared expert
    // never reaches this path and reads none of these roles at all.
    private void RunSharedExpertBatch(int layer, int n, int hidden)
    {
        LogicalTensor gate = _sharedGate[layer];
        ReadOnlySpan<byte> bytes = _weights.Resolve(gate);

        for (int s = 0; s < n; s++)
        {
            _batchNormed.AsSpan(s * hidden, hidden).CopyTo(_moeSubsetIn.AsSpan(s * hidden, hidden));
        }

        Span<float> subsetIn = _moeSubsetIn.AsSpan(0, n * hidden);
        int inter = Plan.IntermediateSize;

        _backend.GemvBatch(subsetIn, n, new WeightRef(bytes, gate.Shape, gate.Format), _batchGateUp.AsSpan(0, n * inter));

        LogicalTensor up = _sharedUp[layer];
        bytes = _weights.Resolve(up);
        _backend.GemvBatch(subsetIn, n, new WeightRef(bytes, up.Shape, up.Format), _batchUp.AsSpan(0, n * inter));

        for (int s = 0; s < n; s++)
        {
            ApplyActivation(_batchGateUp.AsSpan(s * inter, inter), _batchUp.AsSpan(s * inter, inter));
        }

        LogicalTensor down = _sharedDown[layer];
        bytes = _weights.Resolve(down);
        _backend.GemvBatch(_batchGateUp.AsSpan(0, n * inter), n, new WeightRef(bytes, down.Shape, down.Format), subsetIn);

        for (int s = 0; s < n; s++)
        {
            AccumulateScaled(_batchProj.AsSpan(s * hidden, hidden), _moeSubsetIn.AsSpan(s * hidden, hidden), SharedExpertWeight(layer, _batchNormed.AsSpan(s * hidden, hidden)));
        }
    }


    private static void RequireSingle(LogicalTensor tensor, int rows, long cols)
    {
        RequireShape([tensor], rows, cols);
    }

    private static void RequireShape(LogicalTensor[] tensors, long rows, long cols) => RequireShape(tensors.AsSpan(), rows, cols);

    private static void RequireShape(ReadOnlySpan<LogicalTensor> tensors, long rows, long cols)
    {
        foreach (LogicalTensor tensor in tensors)
        {
            int[] dims = tensor.Shape.Dims;
            if (dims.Length != 2 || dims[0] != rows || dims[^1] != cols)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' must have shape [{rows}, {cols}], got [{string.Join(", ", dims)}]");
            }

            long stride = tensor.Shape.RowStrideBytes(tensor.Format);
            if (dims[0] * stride != tensor.SizeBytes)
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' declares {tensor.SizeBytes} payload bytes but its shape and format require {dims[0] * stride}");
            }
        }
    }

    private LogicalTensor RequireTensor(TensorRole role, int layer = -1)
    {
        TensorDescriptor descriptor = Plan.Tensors[role];
        LogicalTensor? hit = descriptor.Resolve(_fileTensors, layer);
        if (hit is not null)
        {
            return hit;
        }

        // A required split role tensor absent while its fused qkv sibling is present: the file ships the fused
        // layout, which this engine does not de-interleave. The failure names both sides instead of only the
        // missing name.
        if (FusedQkvName(role) is string fusedTemplate)
        {
            string fused = FusedName(fusedTemplate, layer);
            if (_fileTensors.Find(fused) is not null)
            {
                throw new ModelException(
                    $"tensor '{descriptor.NameFor(layer)}' is missing and '{fused}' is present: the fused qkv " +
                    $"layout is not de-interleaved for architecture '{Plan.Architecture}'");
            }
        }

        // The other side of that mismatch: a plan reading one fused buffer against a file carrying its three matrices
        // instead. Every split tensor the table does carry gets named, so the failure says which layout the file ships.
        if (role == TensorRole.AttentionQkv)
        {
            List<string> present = [];
            foreach (TensorRole side in new TensorRole[] { TensorRole.AttentionQuery, TensorRole.AttentionKey, TensorRole.AttentionValue })
            {
                TensorDescriptor? sideDescriptor = Plan.Tensors[side];
                if (sideDescriptor?.Resolve(_fileTensors, layer) is not null)
                {
                    present.Add(sideDescriptor.NameFor(layer));
                }
            }

            if (present.Count > 0)
            {
                throw new ModelException(
                    $"tensor '{descriptor.NameFor(layer)}' is missing and '{string.Join("', '", present)}" +
                    $"' {(present.Count > 1 ? "are" : "is")} present: architecture '{Plan.Architecture}' reads one fused " +
                    "qkv buffer per layer, not the split three");
            }
        }

        throw new ModelException(
            $"tensor '{descriptor.NameFor(layer)}' is missing: required by architecture '{Plan.Architecture}'");
    }

    // The fused qkv name template behind a q/k/v role, or null for any other role. The tensor map resolves the
    // split names; the fused layout is recognized only to report it (alongside the split tensors) or to reject
    // a file that carries it as the only source of a role.
    private static string? FusedQkvName(TensorRole role) =>
        role is TensorRole.AttentionQuery or TensorRole.AttentionKey or TensorRole.AttentionValue
            ? "blk.{layer}.attn_qkv.weight"
            : null;

    private static string FusedName(string template, int layer) =>
        template.Replace("{layer}", layer.ToString(CultureInfo.InvariantCulture));

    // Quantization companion entries — a <tensor>.scale block or a <tensor>.input_scale scalar — of a tensor
    // the plan's map can name are known-optional: every format the dequantizers implement today carries its
    // own scales inside the block, so a companion carries no meaning for this engine and never triggers a
    // table rejection. A companion that is not a plain float scale would have to be decoded to be interpreted,
    // which is a load failure naming it rather than a guess. A companion of a tensor the map does not name is
    // simply not part of the forward.
    private void ValidateCompanionTensors()
    {
        foreach (LogicalTensor tensor in _fileTensors.Tensors)
        {
            if (!TryCompanionBase(tensor.Name, out string baseName) || !IsNameableTensorName(baseName))
            {
                continue;
            }

            if (tensor.Format is not (QuantFormat.F32 or QuantFormat.F16))
            {
                throw new ModelException(
                    $"tensor '{tensor.Name}' is a quantization companion of '{baseName}' in format {tensor.Format}: " +
                    $"companion-carried scales are not decoded for architecture '{Plan.Architecture}'");
            }
        }
    }

    // Splits a companion tensor name into the name of the tensor it accompanies: the recognized companion
    // suffixes are .scale (a per-block scale array) and .input_scale (a tensor-wide scalar).
    private static bool TryCompanionBase(string name, out string baseName)
    {
        if (name.EndsWith(".input_scale", StringComparison.Ordinal))
        {
            baseName = name[..^".input_scale".Length];
            return true;
        }

        if (name.EndsWith(".scale", StringComparison.Ordinal))
        {
            baseName = name[..^".scale".Length];
            return true;
        }

        baseName = string.Empty;
        return false;
    }

    // Whether the plan's tensor map can name this tensor: a primary or alias name of any descriptor, for a
    // global role or for any layer the plan covers. Built once and cached.
    private bool IsNameableTensorName(string name)
    {
        _nameableTensors ??= BuildNameableTensorNames();
        return _nameableTensors.Contains(name);
    }

    private HashSet<string> BuildNameableTensorNames()
    {
        HashSet<string> names = new(StringComparer.Ordinal);
        foreach (TensorDescriptor descriptor in Plan.Tensors.Descriptors)
        {
            if (descriptor.Scope == TensorScope.PerLayer)
            {
                for (int l = 0; l < Plan.LayerCount; l++)
                {
                    names.Add(descriptor.NameFor(l));
                    foreach (string alias in descriptor.Aliases)
                    {
                        names.Add(alias.Replace("{layer}", l.ToString(CultureInfo.InvariantCulture)));
                    }
                }
            }
            else
            {
                names.Add(descriptor.Name);
                foreach (string alias in descriptor.Aliases)
                {
                    names.Add(alias);
                }
            }
        }

        return names;
    }

    private float[]?[] LoadOptionalBias(int layers, TensorRole role, long expectedLength)
    {
        TensorDescriptor descriptor = Plan.Tensors[role];
        float[]?[] biases = new float[layers][]; // null entries for absent bias tensors
        for (int l = 0; l < layers; l++)
        {
            LogicalTensor? tensor = descriptor.Resolve(_fileTensors, l);
            if (tensor is not null)
            {
                biases[l] = LoadVector(tensor, checked((int)expectedLength));
            }
        }

        return biases;
    }

    // Attention-bias loading under the plan's bias policy: ByTensorPresence keeps the presence rule, while a
    // resolved-on attention bias makes every layer's q/k/v bias tensors required and names the first one the file
    // lacks — whether the resolution came from a declared use_bias or from the profile's feature default. The
    // output-projection bias is never demanded: it loads when present, under any policy.
    private float[]?[] LoadAttentionBias(int layers, TensorRole role, long expectedLength, bool required)
    {
        if (!required)
        {
            return LoadOptionalBias(layers, role, expectedLength);
        }

        TensorDescriptor descriptor = Plan.Tensors[role];
        float[]?[] biases = new float[layers][];
        for (int l = 0; l < layers; l++)
        {
            LogicalTensor? tensor = descriptor.Resolve(_fileTensors, l);
            if (tensor is null)
            {
                throw new ModelException(
                    $"tensor '{descriptor.NameFor(l)}' is missing: attention biases are resolved on for architecture " +
                    $"'{Plan.Architecture}' (a declared {Plan.Prefix}attention.use_bias or the profile default) and " +
                    "this tensor is required in every layer");
            }

            biases[l] = LoadVector(tensor, checked((int)expectedLength));
        }

        return biases;
    }

    // Post-sublayer norm vectors under a placement that consumes them: every layer needs both, and the failure names the
    // first one the file lacks together with the resolved placement. The map declares these roles optional because a
    // pre-only profile must accept files without them; only the plan decides which side of that rule applies.
    private float[][] LoadPostOutputNorms(int layers, TensorRole role)
    {
        TensorDescriptor descriptor = Plan.Tensors[role];
        float[][] vectors = new float[layers][];
        for (int l = 0; l < layers; l++)
        {
            LogicalTensor? tensor = descriptor.Resolve(_fileTensors, l);
            if (tensor is null)
            {
                throw new ModelException(
                    $"tensor '{descriptor.NameFor(l)}' is missing: norm placement '{Plan.NormPlacement}' needs a post-sublayer output norm in every layer of architecture '{Plan.Architecture}'");
            }

            vectors[l] = LoadVector(tensor, Plan.HiddenSize);
        }

        return vectors;
    }

    // The other side of the same rule: a pre-only placement reads no post-sublayer norm vector at all, and those tensors
    // are not ignorable material — carrying one describes a graph this profile does not implement. Every such tensor in
    // the table is named before the load fails.
    private void RejectUnconsumedPostNorms(int layers)
    {
        List<string> present = [];
        foreach (TensorRole role in new TensorRole[] { TensorRole.PostAttentionNorm, TensorRole.PostFeedForwardNorm })
        {
            TensorDescriptor descriptor = Plan.Tensors[role];
            for (int l = 0; l < layers; l++)
            {
                if (descriptor.Resolve(_fileTensors, l) is not null)
                {
                    present.Add(descriptor.NameFor(l));
                }
            }
        }

        if (present.Count == 0)
        {
            return;
        }

        throw new ModelException(
            $"tensor{(present.Count > 1 ? "s" : "")} '{string.Join("', '", present)}' " +
            $"{(present.Count > 1 ? "are" : "is")} present but norm placement '{Plan.NormPlacement}' reads no " +
            $"post-sublayer output norms for architecture '{Plan.Architecture}'");
    }

    // One structured line naming the model, the architecture, and every tensor of the given roles the file
    // carries although the resolved plan consumes none of them, in the same style as the tokenizer profile defaults
    // and the profile substitution line. No line when no such tensor is present.
    private void ReportIgnoredTensors(Action<string>? logLine, string prefix, GgufFile file, IReadOnlyList<TensorRole> roles)
    {
        if (logLine is null)
        {
            return;
        }

        List<string> names = [];
        foreach (TensorRole role in roles)
        {
            TensorDescriptor descriptor = Plan.Tensors[role];
            if (descriptor.Scope == TensorScope.PerLayer)
            {
                for (int l = 0; l < Plan.LayerCount; l++)
                {
                    if (descriptor.Resolve(_fileTensors, l) is not null)
                    {
                        names.Add(descriptor.NameFor(l));
                    }
                }
            }
            else if (descriptor.Resolve(_fileTensors) is not null)
            {
                names.Add(descriptor.NameFor(-1));
            }
        }

        if (names.Count == 0)
        {
            return;
        }

        GgufMetadata metadata = file.Metadata;
        string modelName = metadata.Has("general.name") ? metadata.Get("general.name").AsString() : "";
        logLine(
            $"{prefix} model='{(modelName.Length > 0 ? modelName : file.Path)}' " +
            $"architecture='{Plan.Architecture}' tensors={string.Join(",", names)}");
    }

    private float[]? LoadOptionalVector(LogicalTensor? tensor, int expectedLength) =>
        tensor is null ? null : LoadVector(tensor, expectedLength);
}
