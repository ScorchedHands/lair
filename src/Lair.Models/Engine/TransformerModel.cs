using System.Buffers;
using System.Diagnostics;
using System.Globalization;

using Lair.Contracts;
using Lair.Backends.Cpu;
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

    // The execution backend prompt windows run on: this model's own backend when prefill shares it, a second CPU
    // backend capped to its dedicated lane share when placement declares one, and an offload posture where a supplied
    // backend arrives without a layout — allowed only over the single DRAM-resident span every unsplit route resolves
    // to, because that shape reads host-resident weight bytes directly from whichever device it reports. Decode-side
    // forwards never consult the lane side.
    private readonly IBackend _prefillsBackend = null!;

    // The composed attention window this engine's prompt windows run their attention through when a device prefill
    // posture was opened for it: null everywhere else, which is every default path and the whole decode side. Supplied at
    // open by whoever opened the step — like an alternate weight source, its lifetime stays under that caller and model
    // disposal never releases it. Non-null only behind the internal seam no host configuration path reaches.
    private readonly IDeviceAttentionStep? _devicePrefill = null;

    private readonly IWeightSource _weights;

    // The bounded synthetic qwen4exp forward seam this engine opens on for a plan whose family geometry resolved, instead of
    // the standard dense loading: a dedicated internal engine that loads the file's own tensors through the plan's tensor map
    // and sequences every block in the reference order with explicit per-sequence state. Null on every other family — which is
    // every model served today — and never a public type: hosts reach this seam only through the assembly-internal chunk
    // forwards below, and the serving routes refuse against it because persistent production caches for this family do not
    // exist yet.
    private readonly Qwen4ExpEngine? _qwen4Exp = null;

    // The resolved prefill kernel selection: pinned at open, re-pointed only through ApplyPrefillKernel, and read
    // fresh at every window boundary so a hot edit lands from the next window — the same re-read discipline the
    // scheduler's chunk width keeps. Volatile because the apply path writes it from a caller thread while worker
    // threads consume it per forward.
    private volatile PrefillKernel _prefillKernel;

    // The quant formats this engine's dense weight matrices carry, resolved once from the plan's tensor map in role
    // order with repeats collapsed. It is what a blocked-matmul selection needs kernels for: every format it lists has
    // to have one before that selection can run this file, and they are the names an unsupported file reports back.
    private readonly IReadOnlyList<QuantFormat> _matmulFormats = [];

    // The name every load-time and re-point capability refusal this engine raises calls this model by — general.name when the file declares it, its path otherwise. Held so a hot re-point refused over resolved execution spans names the same identity an open refusal did.
    private readonly string _modelDisplay;

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
    private readonly LogicalTensor[] _wo = [];
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
    private readonly SparseFfnDescriptor? _sparseFfn = null;
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

    // The expert execution seam for sparse blocks, resolved once at Open: routed experts reach the forward as
    // contiguous per-expert slabs through an IWeightSource slab provider over this model's weight arena — logical
    // views, never copies — and every selected chain runs its GEMVs through the gather kernel. A weight source that
    // already answers slab resolutions is used directly; any other source gets a contiguous-slab view of it here, so
    // alternate providers keep working unchanged while slabs stay addressable by router top-K pointers and prefetched
    // before the GEMV consumes them. Null for a model with no sparse block.
    private IGatherKernel? _gatherKernel;
    private IExpertSlabSource? _expertSlabs;

    // The residency tier behind the slab path when an opened weight source answers as one: it folds each step's union active-expert set into its capacity/cost model before that step gathers, so hot experts resolve from resident copies and everything else stays a cold view. Null on every default posture — a plain slab source is not a tier, and nothing here pays for the check per token beyond one null test per sparse layer step.
    private IResidencyTier? _residencyTier;

    // Routing scratch, sized at Open for descriptor-bearing models only. The token path consumes one
    // routing at a time; the batch path stores every sequence's selection flat, sequence by sequence, and folds it
    // into the step's union active-expert set — each selected expert exactly once with its member rows grouped in
    // sequence order — so one gathered slab feeds every sequence that routed to it this step. Selection order is
    // ascending expert index, which is both the gather set's order and the accumulation order the dense-equivalent
    // forward uses per sequence.

    private float[] _moeLogits = [];      // experts: router scores for one sequence, consumed during selection
    private int[] _moeIds = [];           // routedExpertsPerToken: selected expert indexes, ascending order
    private float[] _moeWeights = [];     // routedExpertsPerToken: the weight of each selected expert
    private int[] _batchMoeIds = [];      // _rowCapacity * routedExpertsPerToken: per-sequence selections
    private float[] _batchMoeWeights = [];// same length, aligned with _batchMoeIds
    private int[] _moeUnion = [];         // _rowCapacity * routedExpertsPerToken upper bound: the batch's union active-expert set, ascending
    private int[] _moeUnionCounts = [];   // member rows per union position of the current step (overwritten with group starts during execution)
    private int[] _moeMembersFlat = [];   // same length: every selection slot's sequence grouped under its expert, in sequence order
    private float[] _moeMemberWeights = [];// aligned with _moeMembersFlat: each member's routing weight at its own slot
    private float[] _moeSubsetIn = []; // _rowCapacity * hidden: one expert group's activations and down results, packed in place

    // Small FP32 vectors preloaded at Open: per-layer norm weights and any biases present in the file. Null
    // entries mean the tensor is absent (llama files carry no attention bias, qwen2 files may).
    private readonly float[][] _attnNormWeight = [];
    private readonly float[][] _ffnNormWeight = [];
    private readonly float[]?[] _biasQ = [];
    private readonly float[]?[] _biasK = [];
    private readonly float[]?[] _biasV = [];
    private readonly float[]?[] _biasO = [];

    // Per-head QK-norm weight vectors (one headDim vector per layer, shared by every head of its side),
    // preloaded only when the plan resolves a weighted mode: every head normalizes its own headDim slice.
    private readonly float[][] _attnQNorm = [];
    private readonly float[][] _attnKNorm = [];

    // Post-sublayer norm weight vectors, preloaded only when the plan places norms on sublayer outputs: each holds one
    // hidden-sized vector per layer and normalizes that sublayer's own output before its residual add. A pre-only
    // placement loads neither and rejects a file carrying either tensor.
    private readonly float[][] _postAttnNorm = [];
    private readonly float[][] _postFfnNorm = [];

    // Norm-bias vectors preloaded only when the plan declares them: the LayerNorm path adds them after the
    // weighted normalization. Null entries mean the layer's tensor is absent.
    private readonly float[]?[] _attnNormBias = [];
    private readonly float[]?[] _ffnNormBias = [];
    private readonly float[]? _outputNormBias = null;

    private readonly LogicalTensor _tokenEmbd = null!;
    private readonly LogicalTensor? _output = null;
    private readonly float[] _outputNormWeight = [];
    private readonly float[]? _outputBias = null;

    // FP32 buffers reused across every forward call.
    private readonly float[] _cur = [];     // hidden: layer input stream (residual carrier) and FFN output accumulator
    private readonly float[] _normed = [];  // hidden: weighted norm before the QKV / FFN GEMVs
    private readonly float[] _q = [];       // headCount * headDim
    private readonly float[] _k = [];       // kvHeads * headDim (also the KV slot write source)
    private readonly float[] _v = [];       // kvHeads * headDim
    private readonly float[] _attnOut = []; // headCount * headDim: heads concatenated in order for wo GEMV
    private readonly float[] _proj = [];    // hidden: attn_output / ffn_down GEMV target before the residual add
    private readonly float[] _gateUp = [];  // intermediateSize: gate result, then activation(gate) * up written back into it
    private readonly float[] _up = [];      // intermediateSize: up GEMV result
    private readonly float[] _logits = [];  // vocabSize

    // Batched decode buffers: sequence-major blocks of _rowCapacity copies of the single-token streams, allocated once
    // at Open so a batched step adds no per-step heap pressure. Each sequence's block mirrors the same-sized buffer in
    // its own single-token forward; a wider prefill window widens these allocations and nothing else.
    private readonly int _rowCapacity = 0;

    // The row window a resolved prefill-buffer ceiling leaves this engine running: the clamp itself (null when nothing declared bounds it),
    // the width every prompt route splits at after it, the lane family's own capacity beside it, and one sub-batch row's activation cost —
    // what the clamp measures rows against. Resolved once at open like every other forward-shape value; no arithmetic in any forward reads them.
    private readonly PrefillRowWidthClamp? _prefillClamp = null;
    private readonly int _effectivePrefillRowWidth = 0;
    private readonly int _laneRowCapacity = 0;
    private readonly long _prefillRowActivationBytes = 0L;

    private readonly float[] _batchCur = [];     // _rowCapacity * hidden
    private readonly float[] _batchNormed = [];  // _rowCapacity * hidden
    private readonly float[] _batchQ = [];       // _rowCapacity * headCount * headDim
    private readonly float[] _batchK = [];       // _rowCapacity * kvHeads * headDim
    private readonly float[] _batchV = [];       // _rowCapacity * kvHeads * headDim
    private readonly float[] _batchAttnOut = []; // _rowCapacity * headCount * headDim
    private readonly float[] _batchProj = [];    // _rowCapacity * hidden
    private readonly float[] _batchGateUp = [];  // _rowCapacity * intermediateSize
    private readonly float[] _batchUp = [];      // _rowCapacity * intermediateSize
    private readonly float[] _batchLogits = [];  // _rowCapacity * vocabSize

    // Row-tiled prefill attention state: one online-softmax entry per head for every row of a sub-batch — running max,
    // then weight sum — held across the tiled walk so each row advances its own state in exactly the order its per-row
    // kernel would. Sized once at Open for the widest window; the accumulators ride on _batchAttnOut's per-row head
    // regions, so a tiled window allocates nothing per call.
    private readonly float[] _tiledRowMax = [];    // _rowCapacity * headCount: every row's per-head online max across the walk
    private readonly float[] _tiledRowSumExp = []; // same shape: every row's per-head weight sum across the walk

    // The dedicated prefill-lane posture this engine forwards under, resolved at open from placement values — null for
    // every shared-posture model. Worker pools, masks, activation buffers, and collision posture are load-time facts of it.
    private readonly PrefillLaneLayout? _prefillsLane = null;

    // The execution spans this engine's forwards advance over, resolved once at open from the route and the compute backends opened for it.
    // Every model served today resolves to one span holding every layer on the device its own backend reports — decode-side spans carry the
    // single-token and batched forwards, prompt-window spans carry the prefill routes — and a split route maps each ordered segment onto the
    // backend supplied for its position, so two spans agreeing on device and residency stay distinct ranges of layers. The two lists share one
    // geometry: only the first span's engine differs when placement declares a dedicated lane.
    private readonly ExecutionSpan[] _mainSpans = [];
    private readonly ExecutionSpan[] _prefillSpans = [];

    // The staging arenas a split route crosses through, one per resolved span beyond the first: each was allocated at open by that span's own backend and is freed here before this engine hands back. Empty for every single-span route — which leaves today's activation arrays as the only mutable streams any forward touches. A dedicated lane posture over a split route is refused above, so both routes of every crossing this build stages run through one activation family's arenas.
    private readonly ResidualCarrier?[] _stagedCarriers = [];

    // What each crossing staged at open, in segment order: the resolved position it feeds, the layers that span runs there, the device of the backend whose aligned allocator owns its arena, and that arena's byte size. Internal diagnostics — no host reports staging today.
    private readonly ExecutionCrossing[] _stagedCrossings = [];

    // The mutable-activation families each forward route writes into: decode-side paths (single-token forwards, batched
    // decode windows, encoder passes) always use the default family — every buffer this block has always allocated — while
    // prompt windows write through _laneAct. For a shared-posture model _laneAct is that same family, so nothing extra is
    // sized and behavior is byte-for-byte what it was before any lane vocabulary existed; for a dedicated posture the lane
    // gets its own buffers at the resolved prefill row width — separate storage per mutable path, while read-only weights,
    // norms, rotation tables, and KV bytes stay shared. ActivationBytes reports what the lane family holds in total.
    private readonly ActivationFamily _decodeAct = null!;
    private readonly ActivationFamily _laneAct = null!;

    private readonly float[] _cos = [];          // maxContext * rope pair count
    private readonly float[] _sin = [];
    private readonly float[]? _cosWindowed = null; // the same shape again at a second base, or null when every layer shares one
    private readonly float[]? _sinWindowed = null;
    private readonly int _ropeHalf = 0;         // pairs the rotation covers: the declared slice, or the full head
    private readonly float _attnScale = 0f;      // head-dimension score gain times the variant's attention factor

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
    /// The widest sub-batch this engine's prompt windows actually run: the resolved row count when no declared prefill-buffer ceiling binds,
    /// and the narrowest width the chain's own ceilings allow otherwise. A shared-posture engine keeps its batched streams sized for the
    /// resolved width whatever this narrows — the clamp bounds what a window runs, never reallocates shared storage; a dedicated lane sizes its
    /// own family beside this value.
    /// </summary>
    public int EffectivePrefillRowWidth => _effectivePrefillRowWidth;

    /// <summary>
    /// The clamp the resolved prefill-buffer chain applied to this engine's row window at open — null when no declared ceiling bound the
    /// resolved width, which is every model carrying no prefill-buffer budgets today. Public diagnostics: hosts and tests read what a clamp
    /// decided beside what the window runs.
    /// </summary>
    public PrefillRowWidthClamp? PrefillRowClamp => _prefillClamp;

    /// <summary>Total mutable activation bytes one prompt-window sub-batch row carries on this engine's prompt route — what a declared prefill-buffer ceiling measures rows against.</summary>
    public long PrefillRowActivationBytes => _prefillRowActivationBytes;

    // A measurement-only attention route for candidate KV-format kernels: null on every host path, so an untouched
    // engine forwards exactly the hand-written FP32 causal kernel it always has. Only bench and test opens install a
    // route through EnableKvFormatSpike below, and the resolved check at each single-token attention call is the only
    // thing the default posture pays for its existence.
    private KvAttentionSpikeRoute? _kvAttentionSpike;

    // The resolved KV format pair this engine's forwards serve, and the attention route closed for it: one fixed
    // instantiation chosen at open — the pinned FP32 causal kernel (and the measurement spike on the single-token path
    // when one is installed) for an FP32 pair, or one closed generic quantized kernel for a quantized pair. No forward
    // branches on a format per token: the pair is decided once here and every forward body runs the route its type names.
    private readonly KvFormatPair _kvFormats = default;
    private readonly FloatAttentionRoute? _floatAttention = null;
    private readonly QuantizedAttentionRoute<Q8Block, Q8Block>? _q8q8Attention = null;
    private readonly QuantizedAttentionRoute<Q8Block, Q4Block>? _q8q4Attention = null;

    /// <summary>The attention route a forward body runs at each attention call, closed over the sequence type its model opened with.</summary>
    private interface IKvAttentionRoute<TSeq> where TSeq : class
    {
        void Apply(IBackend engine, ReadOnlySpan<float> qHeads, int headCount, int kvHeadCount, int headDim, TSeq kv, int layer,
            int position, float attnScale, float softcap, int window, Span<float> outConcat);

        void ApplySingleToken(IBackend engine, ReadOnlySpan<float> qHeads, int headCount, int kvHeadCount, int headDim, TSeq kv, int layer,
            int position, float attnScale, float softcap, int window, Span<float> outConcat);
    }

    private sealed class FloatAttentionRoute(TransformerModel model) : IKvAttentionRoute<SequenceKv>
    {
        public void Apply(IBackend engine, ReadOnlySpan<float> qHeads, int headCount, int kvHeadCount, int headDim, SequenceKv kv, int layer,
            int position, float attnScale, float softcap, int window, Span<float> outConcat) =>
            LayerOps.CausalAttention(engine, qHeads, headCount, kvHeadCount, headDim, kv, layer, position, attnScale, softcap, window, outConcat);

        public void ApplySingleToken(IBackend engine, ReadOnlySpan<float> qHeads, int headCount, int kvHeadCount, int headDim, SequenceKv kv, int layer,
            int position, float attnScale, float softcap, int window, Span<float> outConcat)
        {
            if (model._kvAttentionSpike is { } spikeRoute)
            {
                // Measurement-only candidate route (EnableKvFormatSpike): the pinned FP32 kernel below runs whenever no route is installed.
                spikeRoute(qHeads, headCount, kvHeadCount, headDim, kv, layer, position, attnScale, softcap, window, outConcat);
                return;
            }

            Apply(engine, qHeads, headCount, kvHeadCount, headDim, kv, layer, position, attnScale, softcap, window, outConcat);
        }
    }

    private sealed class QuantizedAttentionRoute<TK, TV> : IKvAttentionRoute<SequenceKv<TK, TV>>
        where TK : struct, IKvFormat<TK>, IKvDotSide<TK>
        where TV : struct, IKvFormat<TV>, IKvValueSide<TV>
    {
        private readonly IntegerDotPath _dot;
        private readonly byte[] _queryScratch;
        private readonly float[] _queryScales;

        public QuantizedAttentionRoute(IntegerDotPath dot, int headCount, int headDim)
        {
            _dot = dot;
            _queryScratch = new byte[TK.DotScratchBytesPerHead(headDim) * headCount];
            _queryScales = new float[TK.DotScaleCount(headDim) * headCount];
        }

        public void Apply(IBackend engine, ReadOnlySpan<float> qHeads, int headCount, int kvHeadCount, int headDim, SequenceKv<TK, TV> kv, int layer,
            int position, float attnScale, float softcap, int window, Span<float> outConcat) =>
            LayerOps.CausalAttention(qHeads, headCount, kvHeadCount, headDim, kv, layer, position, attnScale, softcap, window, _dot, _queryScratch, _queryScales, outConcat);

        // The measurement spike is a float-kernel candidate route: a quantized engine has no spike posture, so its single-token
        // attention is exactly its closed quantized kernel.
        public void ApplySingleToken(IBackend engine, ReadOnlySpan<float> qHeads, int headCount, int kvHeadCount, int headDim, SequenceKv<TK, TV> kv, int layer,
            int position, float attnScale, float softcap, int window, Span<float> outConcat) =>
            Apply(engine, qHeads, headCount, kvHeadCount, headDim, kv, layer, position, attnScale, softcap, window, outConcat);
    }

    /// <summary>The resolved KV format pair this engine's forwards serve: <see cref="KvFormatPair.Default"/> for every float-posture model.</summary>
    public KvFormatPair ResolvedKvFormats => _kvFormats;

    /// <summary>The attention route this engine's forwards run, as the closed route type its pair resolved to.</summary>
    internal string ResolvedKvAttentionRoute => _q8q8Attention is not null ? "q8_0/q8_0" : _q8q4Attention is not null ? "q8_0/q4_0" : "f32/f32";

    /// <summary>Whether this engine composes its prompt windows' per-layer attention through an opened device step. Assembly-internal: only a bench or test open installs one, and every default posture answers false.</summary>
    internal bool DevicePrefillActive => _devicePrefill is not null;

    /// <summary>The KV storage the opened composition writes and walks its causal read from — "none" while no composition stands, the pool's resident device store when the step bound to one, and a private arena otherwise. Assembly-internal.</summary>
    internal string DevicePrefillKvResidency => _devicePrefill switch
    {
        null => "none",
        { KvStoreBacked: true } => "pool device store",
        _ => "private arena",
    };

    /// <summary>Whether this engine opened on the bounded synthetic qwen4exp forward seam instead of the standard dense loading. Assembly-internal: hosts never reach this family's fixture path.</summary>
    internal bool HasQwen4ExpForward => _qwen4Exp is not null;

    /// <summary>The leading dimensions the opened qwen4exp seam's positional rotations cover, per region kind — head regions and indexer component regions — or null on every engine that did not open the seam. Assembly-internal diagnostics behind this family's rope gates.</summary>
    internal (int HeadRegions, int IndexerRegions)? Qwen4ExpRotaryCoverage => _qwen4Exp?.RotaryCoverage;

    /// <summary>Fresh per-sequence state on the bounded qwen4exp seam: recurrent delta-rule slots, convolution history and synthetic whole-sequence caches, all sized at the opened context ceiling. Refused on every family that did not open the seam.</summary>
    internal Qwen4ExpSequence NewQwen4ExpSequence() =>
        _qwen4Exp is { } engine
            ? engine.NewSequence()
            : throw new InvalidOperationException($"model '{_modelDisplay}' did not open the bounded synthetic qwen4exp forward seam");

    /// <summary>
    /// Forward one chunk of one synthetic qwen4exp sequence through the seam: every block runs in the reference order over the
    /// sequence's carried state — recurrent slots and convolution history advance through the GDN chunk, whole-sequence KV rows
    /// and indexer selection advance through the QSA caches — and the returned span carries every token's logits. Splitting the
    /// same token run at any boundary lands bit-for-bit on the unsplit run. Refused on every family that did not open the seam.
    /// </summary>
    internal Span<float> ForwardQwen4ExpChunk(Qwen4ExpSequence sequence, ReadOnlySpan<int> tokens) =>
        _qwen4Exp is { } engine
            ? engine.ForwardChunk(sequence, tokens)
            : throw new InvalidOperationException($"model '{_modelDisplay}' did not open the bounded synthetic qwen4exp forward seam");

    /// <summary>The final-row logits the last chunk this sequence forwarded produced. Refused on every family that did not open the seam.</summary>
    internal Span<float> LastQwen4ExpLogits(Qwen4ExpSequence sequence) =>
        _qwen4Exp is { } engine
            ? engine.LastLogits(sequence)
            : throw new InvalidOperationException($"model '{_modelDisplay}' did not open the bounded synthetic qwen4exp forward seam");

    /// <summary>
    /// Turn on the bounded seam's stage/layer timing. Assembly-internal bench/diagnostic surface, not a public serving API:
    /// while off the seam's forwards read no clock and touch no profiling state, so measured logits stay bit-identical.
    /// Refused on every family that did not open the seam.
    /// </summary>
    internal void EnableQwen4ExpLayerProfiling()
    {
        if (_qwen4Exp is not { } engine)
        {
            throw SeamRefusal();
        }

        engine.EnableLayerProfiling();
    }

    /// <summary>Turn the seam's stage/layer timing back off. Refused on every family that did not open the seam.</summary>
    internal void DisableQwen4ExpLayerProfiling()
    {
        if (_qwen4Exp is not { } engine)
        {
            throw SeamRefusal();
        }

        engine.DisableLayerProfiling();
    }

    /// <summary>Zero the seam's stage and per-layer accumulators so the next measured window starts from zero. Refused on every family that did not open the seam.</summary>
    internal void ResetQwen4ExpLayerProfile()
    {
        if (_qwen4Exp is not { } engine)
        {
            throw SeamRefusal();
        }

        engine.ResetLayerProfile();
    }

    /// <summary>The seam's accumulated stage and per-layer times as milliseconds. Refused on every family that did not open the seam.</summary>
    internal Qwen4ExpLayerProfile Qwen4ExpLayerProfileSummary() =>
        _qwen4Exp is { } engine
            ? engine.LayerProfileSummary()
            : throw SeamRefusal();

    private InvalidOperationException SeamRefusal() =>
        new($"model '{_modelDisplay}' did not open the bounded synthetic qwen4exp forward seam");

    // Dense stage/layer profiling for this engine's own dense forwards — the batched-row and row-tiled prompt windows, the merged
    // decode step, and the single-token forward — off by default. A disabled forward reads no clock and touches none of this state:
    // BeginDenseProfile returns 0 without calling Stopwatch.GetTimestamp and AccumulateDenseProfile returns before touching an
    // accumulator, so every dense forward and its logits stay bit-identical unless a host explicitly turns the seam on (bench/test
    // diagnostics only — never a serving posture). Accumulators hold raw Stopwatch ticks; only the summary converts them, no clock
    // object is allocated per token or layer, and one forward at a time runs on an engine, so these tallies need no locking.
    private bool _denseProfileEnabled;

    // Stage partition of every dense prefill/decode window since the last reset: seed/embedding lookups with their input scale; each
    // sequence slot reservation the route makes before its layers write (cache interaction); each layer's attention block from its norms
    // through the sublayer fold and that block's feed-forward norm — partitioned by its dense weight traversals, its causal read spans, and
    // the K/V slot writes it performs; each layer's feed-forward block — partitioned by its gate/up/down traversals (or a sparse block's whole
    // routed/shared chain); the output head each window reaches; and each instrumented forward call as a whole. Parent stages cover their
    // substages, so a substage never adds cost beside its parent rather than inside it, exactly like the bounded seam's feed-forward split.
    // Each traversal parent is itself split: the attention weight tally partitions into the batched Q/K/V projection passes and the output
    // projection pass, and the feed-forward weight tally into the gate/up pair and the down pass, with a sparse block's chain reported as
    // its own named substage instead of a gate/up/down triple. A split site feeds one clock read into both its substage tally and its
    // parent tally, so a parent is exactly the sum of its traversal substages.
    private long _denseSeedTicks;
    private long _denseCacheAppendTicks;
    private long _denseKvWriteTicks;
    private long _denseAttentionTicks;
    private long _denseAttentionWeightTicks;
    private long _denseAttentionProjectionTicks;
    private long _denseAttentionOutputTicks;
    private long _denseKernelTicks;
    private long _denseFeedForwardTicks;
    private long _denseFeedForwardWeightTicks;
    private long _denseFeedForwardGateUpTicks;
    private long _denseFeedForwardDownTicks;
    private long _denseSparseChainTicks;
    private long _denseLogitsTicks;
    private long _denseTotalTicks;

    // Whole-block time of every layer the dense forwards ran, sized at enable like the bounded seam's per-layer array.
    private long[] _denseLayerTicks = [];

    /// <summary>
    /// Turn on stage/layer timing for this engine's dense prefill and decode forwards. Assembly-internal bench/diagnostic surface, not a
    /// public serving API: while off the dense forwards read no clock and touch no profiling state, so measured logits stay bit-identical to
    /// an unprofiled run. Refused on the bounded synthetic qwen4exp seam, whose forwards run through its own engine and carry their stage
    /// timings there instead. Encoder forwards (<see cref="Encode"/> / <see cref="EncodeBatch"/>) are not instrumented by this seam.
    /// </summary>
    internal void EnableDenseForwardProfiling()
    {
        if (_qwen4Exp is not null)
        {
            throw SeamRefusal();
        }

        if (_denseLayerTicks.Length != Plan.LayerCount)
        {
            _denseLayerTicks = new long[Plan.LayerCount];
        }

        _denseProfileEnabled = true;
    }

    /// <summary>Turn the dense stage/layer timing back off; accumulators keep their last values until <see cref="ResetDenseForwardProfile"/>.</summary>
    internal void DisableDenseForwardProfiling() => _denseProfileEnabled = false;

    /// <summary>Zero every dense stage and per-layer accumulator so the next measured window starts from zero.</summary>
    internal void ResetDenseForwardProfile()
    {
        _denseSeedTicks = 0;
        _denseCacheAppendTicks = 0;
        _denseKvWriteTicks = 0;
        _denseAttentionTicks = 0;
        _denseAttentionWeightTicks = 0;
        _denseAttentionProjectionTicks = 0;
        _denseAttentionOutputTicks = 0;
        _denseKernelTicks = 0;
        _denseFeedForwardTicks = 0;
        _denseFeedForwardWeightTicks = 0;
        _denseFeedForwardGateUpTicks = 0;
        _denseFeedForwardDownTicks = 0;
        _denseSparseChainTicks = 0;
        _denseLogitsTicks = 0;
        _denseTotalTicks = 0;

        if (_denseLayerTicks.Length > 0)
        {
            Array.Clear(_denseLayerTicks);
        }
    }

    /// <summary>
    /// The dense forwards' accumulated stage and per-layer times as milliseconds since the last reset, converted from raw Stopwatch ticks.
    /// Stages partition each instrumented window: seed/embedding lookups with their input scale; sequence slot reservations (cache interaction);
    /// attention blocks — split into their Q/K/V projection traversals and their output-projection traversal (which together are the attention
    /// weight substage), their causal read spans including a tiled walk or a composed device step, and the K/V slot writes they perform, with
    /// norms/bias/RoPE/residual folds riding inside the parent stage beside those substages; feed-forward blocks — split into their gate/up and
    /// down traversals (which together are the feed-forward weight substage), or a sparse block's whole routed/shared chain; the output head each
    /// window reaches; and each forward call as a whole. Parent stages cover their substages rather than adding to them, so summing parents stays
    /// inside the total while substage sums stay inside their own parent and each traversal parent is exactly the sum of its traversal substages.
    /// Per-layer totals cover each block's whole attention + feed-forward span.
    /// </summary>
    internal DenseForwardProfile DenseForwardProfileSummary()
    {
        List<DenseForwardLayerTiming> layers = new(Plan.LayerCount);
        for (int layer = 0; layer < Plan.LayerCount; layer++)
        {
            layers.Add(new DenseForwardLayerTiming(layer, TicksToMs(_denseLayerTicks.Length > layer ? _denseLayerTicks[layer] : 0L)));
        }

        return new DenseForwardProfile(
            TicksToMs(_denseSeedTicks),
            TicksToMs(_denseCacheAppendTicks),
            TicksToMs(_denseAttentionTicks),
            TicksToMs(_denseAttentionWeightTicks),
            TicksToMs(_denseAttentionProjectionTicks),
            TicksToMs(_denseAttentionOutputTicks),
            TicksToMs(_denseKernelTicks),
            TicksToMs(_denseKvWriteTicks),
            TicksToMs(_denseFeedForwardTicks),
            TicksToMs(_denseFeedForwardWeightTicks),
            TicksToMs(_denseFeedForwardGateUpTicks),
            TicksToMs(_denseFeedForwardDownTicks),
            TicksToMs(_denseSparseChainTicks),
            TicksToMs(_denseLogitsTicks),
            TicksToMs(_denseTotalTicks),
            layers);
    }

    // The dense profiling seams every instrumented site shares: a disabled forward gets 0 from BeginDenseProfile without ever reading the
    // clock, and AccumulateDenseProfile returns before touching its accumulator — so an unprofiled dense forward pays nothing beyond two
    // inlined flag checks around each span. No clock object is allocated per token or per layer.
    private long BeginDenseProfile() => _denseProfileEnabled ? Stopwatch.GetTimestamp() : 0L;

    private void AccumulateDenseProfile(ref long accumulator, long startTicks)
    {
        if (_denseProfileEnabled)
        {
            accumulator += Stopwatch.GetTimestamp() - startTicks;
        }
    }

    // A traversal span that is both a named substage and part of its parent stage: one clock read lands the elapsed ticks in both tallies,
    // so the parent is exactly the sum of its traversal substages rather than a second, independently measured pass over the same span.
    // A disabled forward still returns before touching either accumulator.
    private void AccumulateDenseSubstage(ref long substage, ref long parent, long startTicks)
    {
        if (_denseProfileEnabled)
        {
            long elapsed = Stopwatch.GetTimestamp() - startTicks;
            substage += elapsed;
            parent += elapsed;
        }
    }

    // The per-layer form of the same seam: an unprofiled forward skips before indexing the layer array at all, so a model that never enabled
    // the seam — whose array is still empty — pays one flag check and touches nothing.
    private void AccumulateDenseLayerTick(int layer, long startTicks)
    {
        if (_denseProfileEnabled)
        {
            _denseLayerTicks[layer] += Stopwatch.GetTimestamp() - startTicks;
        }
    }

    private static double TicksToMs(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    /// <summary>
    /// Re-points the engine's prefill path selection. The new value is read fresh at the next prefill window
    /// boundary, so sequences already mid-window finish that window on the route they started with — the same
    /// boundary discipline a chunk-width edit keeps. A selection this build has no kernel for is refused naming it
    /// and what is missing; nothing falls back to the route that works. The dense-matrix selection is also refused
    /// when any resolved prompt-window range carries a weight format its own execution backend has no blocked matmul
    /// for, naming that format beside the segment and backend it was resolved onto — the same gate model open runs.
    /// </summary>
    public void ApplyPrefillKernel(PrefillKernel kernel)
    {
        if (!KernelExecutes(kernel))
        {
            throw KernelRefusal(kernel);
        }

        // The bounded synthetic qwen4exp seam forwards its chunks through its own engine and reads no kernel selection, so a
        // hot re-point to another kernel would run some other route behind the selection — refused naming it exactly as open
        // did, and the running selection stays in force.
        if (_qwen4Exp is not null && kernel != PrefillKernel.GemV)
        {
            throw Qwen4ExpPostureRefusal($"prefill kernel '{kernel}'");
        }

        // A quantized engine forwards prompt windows through the batched-row window only: its attention route is the closed
        // quantized kernel, and the row-tiled and dense-matrix windows carry the FP32 attention kernels. The selection is
        // refused naming the pair, exactly as at open — nothing falls back to the route that works.
        if (!_kvFormats.IsFloatStorage && kernel != PrefillKernel.GemV)
        {
            throw QuantizedPrefillRefusal(kernel);
        }

        // The device prefill posture composes every prompt window's attention through the opened step and takes its dense weight passes through the blocked matmul: a hot re-point off that selection would run windows through some other route behind an opened composition, so it refuses naming both sides exactly as open did — the running selection stays in force.
        if (_devicePrefill is not null && kernel != PrefillKernel.Gemm)
        {
            throw DevicePrefillSelectionRefusal(kernel);
        }

        ValidateSpanMatmulCapability(kernel, _modelDisplay, Plan.Route, _fileTensors, _prefillSpans); // the re-point answers to the same per-span capability gate model open does; a refused selection leaves the running route untouched.

        _prefillKernel = kernel;
    }

    /// <summary>
    /// The attention route a candidate KV-format kernel runs on this engine's single-token decode step, taking the same
    /// operands the pinned FP32 causal kernel is called with. Assembly-internal: no host configuration path can install one.
    /// </summary>
    internal delegate void KvAttentionSpikeRoute(
        ReadOnlySpan<float> qHeads, int headCount, int kvHeadCount, int headDim, SequenceKv kv, int layer, int position,
        float attnScale, float softcap, int window, Span<float> outConcat);

    /// <summary>
    /// Routes this engine's single-token decode attention through a candidate kernel for measurement. Bench and test opens
    /// only: the hand-written FP32 causal kernel stays in force whenever no route is installed, prompt windows keep running
    /// their resolved prefill kernel whatever this sets, and every other forward step is untouched.
    /// </summary>
    internal void EnableKvFormatSpike(KvAttentionSpikeRoute route) =>
        _kvAttentionSpike = route ?? throw new ArgumentNullException(nameof(route));

    /// <summary>The candidate attention route installed on this engine, or null while it runs the pinned FP32 kernel.</summary>
    internal KvAttentionSpikeRoute? ResolvedKvFormatSpike => _kvAttentionSpike;

    /// <summary>The quant formats this engine's dense weight matrices carry, deduplicated in role order: the formats a blocked matmul needs kernels for before the dense-matrix selection can run this file.</summary>
    public IReadOnlyList<QuantFormat> MatmulWeightFormats => _matmulFormats;

    // Whether a selection has kernels for this engine's own weights. The batched-row window and its row-tiled variant
    // always do; the dense-matrix window needs this file to carry dense weight matrices at all, and every compute backend
    // that will multiply activation rows against them must report a blocked matmul for their formats — which is decided
    // per resolved execution span by ValidateSpanMatmulCapability, from the backend each prompt-window range actually runs on.
    private bool KernelExecutes(PrefillKernel kernel) => kernel switch
    {
        PrefillKernel.GemV or PrefillKernel.Tiled => true,
        PrefillKernel.Gemm => _matmulFormats.Count > 0,
        _ => false,
    };

    // The dense weight formats this engine carries that one compute backend has no blocked matmul for. Empty when every
    // one of them runs through the dense-matrix window on that backend; otherwise exactly the names a refusal reports.
    private List<QuantFormat> FormatsWithoutKernels(IBackend engine) => MatmulFormatsWithoutKernels(_matmulFormats, engine);

    // The formats one resolved prompt-window range carries through the dense-matrix window that its own resolved backend has no blocked matmul for: ResolveMatmulFormats' role list read over just this span's layers. Sparse FFN roles are excluded because expert matrices stay on batched row passes whatever selection is resolved — exactly as every forward of this engine runs them today.
    private static List<QuantFormat> SpanMatmulGaps(ModelPlan plan, LogicalTensorTable tensors, ExecutionSpan prefill) =>
        MatmulFormatsWithoutKernels(CollectWeightFormats(plan, tensors, prefill.LayerStart, prefill.LayerEnd, includeSparse: false), prefill.Backend);

    private static List<QuantFormat> MatmulFormatsWithoutKernels(IReadOnlyList<QuantFormat> formats, IBackend engine)
    {
        List<QuantFormat> missing = [];
        foreach (QuantFormat format in formats)
        {
            if (!engine.SupportsGemm(format))
            {
                missing.Add(format);
            }
        }

        return missing;
    }

    // The capability gate the dense-matrix selection answers to: when that selection is resolved, every prompt-window span's own layer range must be formats its own resolved backend multiplies through a blocked matmul. A gap refuses at load — before any window dispatches, and with no fall back to another kernel or device — naming the model, the segment and the backend resolved for it, the offending format(s), and that this build carries no dense-matrix path there. The batched-row windows answer empty whatever a file carries, so every other posture is untouched by this gate.
    private void ValidateSpanMatmulCapability(PrefillKernel kernel, string model, ModelRoute route, LogicalTensorTable tensors, ExecutionSpan[] prefill)
    {
        if (kernel != PrefillKernel.Gemm)
        {
            return;
        }

        for (int index = 0; index < prefill.Length && index < route.Segments.Count; index++)
        {
            List<QuantFormat> missing = SpanMatmulGaps(Plan, tensors, prefill[index]);

            if (missing.Count > 0)
            {
                throw MatmulGapRefusal(model, route.Segments[index], index, prefill[index].Backend, missing);
            }
        }
    }

    // Which wording a dense-matrix gap gets: when the backend answering it is this engine's own opened compute backend on its own device, the shape an unrunnable selection has always carried — the value beside this file's own formats and that backend; when a route resolved some other execution backend for that position, the segment and its backend are named instead.
    private ModelException MatmulGapRefusal(string model, DeviceSegment segment, int index, IBackend engine, List<QuantFormat> missing) =>
        ReferenceEquals(engine, _backend) && string.Equals(segment.DeviceId, _backend.DeviceId, StringComparison.Ordinal)
            ? KernelRefusal(PrefillKernel.Gemm)
            : MatmulSpanRefusal(model, segment, index, engine, missing);

    // The refusal wording a prompt-window span naming a format its own resolved compute backend cannot multiply in bulk answers: the model, the segment and the device resolved for it, the offending dense weight format(s), and that this build carries no blocked matmul there — beside the two windows that always run. It extends the load-time GEMV capability refusal shape to the window shape a split route or an offloaded prompt side selects, replacing any per-layer throw the same format would have raised after work dispatched.
    private static ModelException MatmulSpanRefusal(string model, DeviceSegment segment, int index, IBackend engine, List<QuantFormat> missing) => new(
        $"model '{model}', {segment.Describe(index)}: prefill kernel '{PrefillKernelNames.Gemm}' needs blocked matmul kernels for weight format(s) {string.Join(", ", missing)}, and the compute backend " +
        $"'{engine.DeviceId}' resolved for this segment reports none — no dense-matrix path exists in this build for them on that device, so this model is refused at load naming the selection, the format, " +
        $"and its backend rather than raised when a prompt window dispatches — resolve placement onto a segment whose backend multiplies these formats in bulk, re-quantize this file to one '{engine.DeviceId}' " +
        $"does, or run it through '{PrefillKernelNames.GemV}' / '{PrefillKernelNames.Tiled}', which forward every format through batched row passes");

    // The one refusal wording an unrunnable prefill selection gets, at open and on apply alike: it names the value, what
    // the forward engine lacks for it, and the selections that execute today. For the dense-matrix selection it also
    // names this file's own weight formats when they are the reason the selection cannot run, beside the compute backend
    // that reports no blocked matmul for them — which is why a route spanning two devices refuses per segment instead.
    private ModelException KernelRefusal(PrefillKernel kernel)
    {
        string name = PrefillKernelNames.Of(kernel);
        List<QuantFormat> missing = kernel == PrefillKernel.Gemm ? FormatsWithoutKernels(_backend) : [];

        if (missing.Count > 0)
        {
            return new ModelException(
                $"prefill kernel '{name}' is not implemented by the forward engine: architecture '{Plan.Architecture}' " +
                $"carries dense weight matrices of format(s) {string.Join(", ", missing)} and no {name} attention/weight path exists in this build for them on the compute backend " +
                $"'{_backend.DeviceId}' — the window shape that multiplies a whole sub-batch against one matrix runs only where every dense weight matrix has a blocked matmul kernel there, " +
                $"while every other family forwards its prompt windows through batched row passes — set engine.prefill_kernel to '{PrefillKernelNames.GemV}' or " +
                $"'{PrefillKernelNames.Tiled}' (or omit it) to run this file");
        }

        return new ModelException(
            $"prefill kernel '{name}' is not implemented by the forward engine: architecture " +
            $"'{Plan.Architecture}' selects its prompt windows through this selection and no {name} attention/weight path exists in this build — only " +
            $"{PrefillKernelNames.GemV} and {PrefillKernelNames.Tiled} execute today — set engine.prefill_kernel to '{PrefillKernelNames.GemV}' or " +
            $"'{PrefillKernelNames.Tiled}' (or omit it) to run this file");
    }

    // The refusal wording a quantized-KV engine gets for a prefill selection its closed attention route cannot serve: the
    // row-tiled and dense-matrix windows carry the FP32 attention kernels, so a quantized pair names its own pair and the
    // selection that runs the quantized kernel.
    private ModelException QuantizedPrefillRefusal(PrefillKernel kernel) => new(
        $"prefill kernel '{PrefillKernelNames.Of(kernel)}' cannot run this model's quantized KV pair k='" +
        $"{KvSideFormatNames.Of(_kvFormats.K)}' v='{KvSideFormatNames.Of(_kvFormats.V)}': its attention route is the closed quantized kernel, " +
        $"which the batched-row window forwards — set engine.prefill_kernel to '{PrefillKernelNames.GemV}' (or omit it) to run this pair");

    // The quantized-KV posture refusals one open shares: which posture the pair arrived beside, and what a quantized pair's
    // CPU-kernel-only, host-resident, batched-row route does not carry.
    // The opened device composition is installed once, against exactly what this engine resolved its prompt windows to run on: the dense-matrix selection it composes under, the span posture those windows stay inside, the KV storage shape the step's write and walk address, a plan whose head geometry the step walks unchanged, an arena that covers every position this context can reach, and staging wide enough for the sub-batch width these windows actually run. Every refusal names both sides — what was opened beside what this engine resolves — because nothing here re-points a composition onto some other posture behind it.
    private IDeviceAttentionStep InstallDevicePrefill(IDeviceAttentionStep step, int maxContextTokens)
    {
        ArgumentNullException.ThrowIfNull(step);

        if (_prefillKernel != PrefillKernel.Gemm)
        {
            throw DevicePrefillSelectionRefusal(_prefillKernel);
        }

        if (!ReferenceEquals(_prefillsBackend, _backend))
        {
            throw DevicePrefillPostureRefusal(_prefillsLane is not null ? "a dedicated prefill lane" : "an offloaded prefill backend");
        }

        // A quantized pair never reaches here: its own posture gate already refuses any selection but the batched-row window, and this composition resolves only the dense-matrix one — the refusal that fires names the formats and the route they run on.

        if (Plan.SparseFfn is { } routing)
        {
            throw new ModelException(
                $"model '{_modelDisplay}' opens a device attention step over a sparse-FFN block routing {routing.ExpertCount} experts: " +
                "the opened composition stages dense per-layer attention windows only — omit the device step to run this file's expert routing on its batched row passes");
        }

        if (step.LayerCount != Plan.LayerCount)
        {
            throw DevicePrefillGeometryRefusal("layers", step.LayerCount, Plan.LayerCount);
        }

        if (step.HeadDim != Plan.HeadDim)
        {
            throw DevicePrefillGeometryRefusal("head dimension", step.HeadDim, Plan.HeadDim);
        }

        if (step.KvHeadCount != Plan.KvHeadCount)
        {
            throw DevicePrefillGeometryRefusal("kv-head count", step.KvHeadCount, Plan.KvHeadCount);
        }

        if (step.HeadCount != Plan.HeadCount)
        {
            throw DevicePrefillGeometryRefusal("query head count", step.HeadCount, Plan.HeadCount);
        }

        if (step.MaxPositions < maxContextTokens)
        {
            throw new ModelException(
                $"model '{_modelDisplay}' resolves a context of {maxContextTokens} tokens, but the opened device attention step holds storage for {step.MaxPositions} positions: " +
                "a composed window may only read and write positions that storage covers — open a step over this context, or shorten it");
        }

        if (step.MaxWindowRows < _effectivePrefillRowWidth)
        {
            throw new ModelException(
                $"model '{_modelDisplay}' runs prompt windows in sub-batches of {_effectivePrefillRowWidth} rows, but the opened device attention step stages at most " +
                $"{step.MaxWindowRows} rows — open a step that stages this engine's resolved row width, or narrow the window");
        }

        return step;
    }

    private ModelException DevicePrefillSelectionRefusal(PrefillKernel kernel) => new(
        $"model '{_modelDisplay}': an opened device attention step composes prompt windows under prefill kernel '{PrefillKernelNames.Gemm}', and this engine resolves " +
        $"'{PrefillKernelNames.Of(kernel)}' — the composition takes its dense weight passes through the blocked matmul beside the device-side write and causal walk, so it opens with " +
        $"'{PrefillKernelNames.Gemm}' or not at all; a re-point away from it is refused rather than run behind an opened composition");

    private ModelException DevicePrefillPostureRefusal(string posture) => new(
        $"model '{_modelDisplay}': the opened device attention step composes prompt windows inside this engine's own resolved spans on device " +
        $"'{_backend.DeviceId}', and {posture} moves them onto execution backend '{_prefillsBackend.DeviceId}' it was not opened against — resolve placement onto one span beside the composition, or omit the device step");

    private ModelException DevicePrefillGeometryRefusal(string fact, int openedValue, int resolvedValue) => new(
        $"model '{_modelDisplay}' resolves {fact} {resolvedValue}, but the opened device attention step declares {openedValue}: a composed window walks this engine's own plan unchanged — " +
        "open a step for this model geometry or omit it");

    private ModelException QuantizedPostureRefusal(string posture) => new(
        $"this model opens quantized KV storage k='{KvSideFormatNames.Of(_kvFormats.K)}' v='{KvSideFormatNames.Of(_kvFormats.V)}' " +
        $"beside {posture}: quantized attention runs CPU kernels over host-resident KV through the batched-row window only — " +
        "resolve placement onto a CPU-only, host-resident posture with 'gemv' prefill to run this pair");

    // The quantized attention kernel dots whole 32-value groups per head region: a head dimension that is not a whole number
    // of groups has no block-aligned head slice to dot, so the pair refuses at open naming the geometry rather than failing
    // mid-forward.
    private ModelException QuantizedHeadDimRefusal() => new(
        $"this model opens quantized KV storage k='{KvSideFormatNames.Of(_kvFormats.K)}' v='{KvSideFormatNames.Of(_kvFormats.V)}' " +
        $"over a head dimension of {Plan.HeadDim}: the quantized attention kernel dots whole {KvFormatPair.GroupValues}-value groups per head region, " +
        $"and {Plan.HeadDim} is not a whole number of them.");

    // The quant format of every dense weight matrix a prompt window multiplies activation rows against: the attention
    // trio — or one fused qkv buffer, whichever layout the plan resolves — the attention output, and the feed-forward
    // block's gate/up/down as its layout names them. Deduplicated in role order because a family that keeps one format
    // across every layer should report it once. A sparse-FFN block contributes nothing here: its expert matrices stay on
    // batched row passes whatever selection is resolved, so they do not constrain which window shape can run.
    private static IReadOnlyList<QuantFormat> ResolveMatmulFormats(ModelPlan plan, LogicalTensorTable tensors) =>
        CollectWeightFormats(plan, tensors, 0, plan.LayerCount, includeSparse: false);

    // The weight formats one range of this file's layers carries through a GEMV weight pass: the dense families
    // ResolveMatmulFormats names, read over just that range, plus — when asked for them and the block declares a sparse
    // FFN — the router, the stacked expert matrices, and the shared triplet, because those reach GEMV through whatever
    // span runs the layer whatever prefill selection is resolved. Everything comes from the plan's tensor map and the
    // file's own table, so the names a refusal reports are exactly the names the file carries.
    private static List<QuantFormat> CollectWeightFormats(ModelPlan plan, LogicalTensorTable tensors, int layerStart, int layerEnd, bool includeSparse)
    {
        List<QuantFormat> formats = [];

        void AddRole(TensorRole role)
        {
            if (plan.Tensors[role] is not TensorDescriptor descriptor)
            {
                return;
            }

            for (int l = layerStart; l < layerEnd; l++)
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
        else if (includeSparse)
        {
            AddRole(plan.SparseFfn.RouterRole);
            AddRole(plan.SparseFfn.ExpertGateRole);
            AddRole(plan.SparseFfn.ExpertUpRole);
            AddRole(plan.SparseFfn.ExpertDownRole);

            if (plan.SparseFfn.SharedGateRole is { } sharedGate)
            {
                AddRole(sharedGate);
            }

            if (plan.SparseFfn.SharedUpRole is { } sharedUp)
            {
                AddRole(sharedUp);
            }

            if (plan.SparseFfn.SharedDownRole is { } sharedDown)
            {
                AddRole(sharedDown);
            }
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
    /// When placement declares a dedicated prefill lane, <paramref name="prefillsBackend"/> carries prompt windows on a second
    /// execution backend capped to its lane share and <paramref name="prefillsLane"/> pins the resolved split — masks, worker caps,
    /// and the activation buffers sized from it land at open; every decode-side forward stays on <paramref name="backend"/>. The two
    /// must arrive together and report the same device this engine runs on: a half-declared lane posture is refused naming what is missing.
    /// A backend arriving alone with no layout opens an offload posture instead — prompt windows run on it while decode stays on
    /// <paramref name="backend"/>, allowed only over the single DRAM-resident span every unsplit route resolves to, since that shape reads host-resident weight bytes across whatever device it reports; a split route refuses naming what lands short, and a weight format that backend reports no GEMV kernel for refuses at open naming the model, the format, and that backend rather than run on some other device.
    /// A host that resolved per-resource budgets against its own opened pools supplies them through <paramref name="budgets"/>, and they land
    /// on the plan unchanged beside the route — a declared prefill-buffer ceiling narrows this engine's row window to what it allows (requested
    /// width stays stated on the plan, effective width reported apart), while no capacity claim or relocation runs here. Omitting it keeps the
    /// empty set, which is what every model carrying no budget declarations runs on today.
    /// </summary>
    public static TransformerModel Open(
        GgufFile file, IBackend backend, int maxContextTokens, Action<string>? logLine, PrefillKernel prefillKernel, ModelRoute route,
        IBackend? prefillsBackend = null, PrefillLaneLayout? prefillsLane = null, ResolvedBudgetSet? budgets = null) =>
        new(file, backend, maxContextTokens, weights: null, logLine, profileOverride: null, prefillKernel, prefillRowWidth: 0, route, prefillsBackend, prefillsLane, crossingBackends: null, budgets);

    /// <summary>
    /// Opens the engine serving an explicit quantized KV format pair: every forward of this engine — single-token, batched
    /// decode, and prompt windows — runs the one attention instantiation closed for that pair, chosen once here and never
    /// re-resolved per token. The posture a quantized route carries is CPU-only, host-resident KV over the batched-row
    /// window: a split route, a dedicated lane or offload posture, a non-<c>gemv</c> prefill selection, or a head dimension
    /// that is not a whole number of quant groups refuses at open naming what the opened posture declares and what the
    /// quantized route lacks — never a silent fall back to the FP32 posture. <paramref name="quantizedDot"/> carries the
    /// integer-dot encoding the opened CPU backend resolved, which the quantized kernel runs per scored position.
    /// </summary>
    public static TransformerModel Open(
        GgufFile file, IBackend backend, int maxContextTokens, Action<string>? logLine, PrefillKernel prefillKernel, ModelRoute route,
        KvFormatPair kvFormats, IntegerDotPath quantizedDot, ResolvedBudgetSet? budgets = null) =>
        new(file, backend, maxContextTokens, weights: null, logLine, profileOverride: null, prefillKernel, prefillRowWidth: 0, route, prefillsBackend: null, prefillsLane: null, crossingBackends: null, budgets, kvFormats, quantizedDot);

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

    /// <summary>
    /// Opens the engine with a device attention composition installed on its prompt windows: every sub-batch's per-layer
    /// RoPE, KV write, and causal walk stay exactly as this engine runs them today for the host side, while each layer's
    /// whole window is additionally composed through <paramref name="devicePrefill"/> — one staged crossing in and one
    /// attention result back per layer instead of a score row, a softmax state, and every K/V read crossed per position.
    /// The composition arrives under the alternate-<see cref="IWeightSource"/> lifetime rule: whoever opened the step owns
    /// it, it must outlive this engine's forwards, and model disposal never releases it. Assembly-internal — no host
    /// configuration path reaches it; bench and test opens resolve candidate compositions through this seam only. A step
    /// whose geometry facts disagree with what this file's plan resolves is refused at open naming both sides, as is a
    /// posture the composition cannot run inside: anything but the dense-matrix selection, a lane or offload side that
    /// moves windows onto another backend, quantized KV storage, and expert-routed feed-forward blocks.
    /// </summary>
    internal static TransformerModel Open(
        GgufFile file, IBackend backend, int maxContextTokens, Action<string>? logLine, PrefillKernel prefillKernel, ModelRoute route,
        IDeviceAttentionStep devicePrefill, IBackend? prefillsBackend = null) =>
        new(file, backend, maxContextTokens, weights: null, logLine, profileOverride: null, prefillKernel, prefillRowWidth: 0, route, prefillsBackend, prefillsLane: null, crossingBackends: null, budgets: null, kvFormats: null, quantizedDot: null, devicePrefill);

    /// <summary>
    /// Opens the engine over a resolved route whose execution positions are answered by compute backends supplied for them. Element
    /// <c>i</c> of <paramref name="crossingBackends"/> answers position <c>i + 1</c> when the set covers every position after the first — so two
    /// spans that agree on device and residency stay distinct ranges of layers rather than collapsing into one pass, and the first span keeps
    /// running on <paramref name="backend"/>, which is what every unsplit route resolves to. A set covering every resolved segment in order answers
    /// position 0 too, which is how a host opens a split whose first span runs somewhere other than its decode-side backend: each segment
    /// is validated against the backend resolved for it, a boundary that fails to cover this file's layers once refuses naming what lands short,
    /// and a dedicated lane posture over a split route refuses naming the stage-side crossing this build does not carry rather than running the
    /// model on one span. Assembly-internal: hosts resolve placement onto these seams when they open a layer split.
    /// </summary>
    internal static TransformerModel Open(
        GgufFile file, IBackend backend, int maxContextTokens, Action<string>? logLine, PrefillKernel prefillKernel, ModelRoute route,
        IReadOnlyList<IBackend> crossingBackends) =>
        new(file, backend, maxContextTokens, weights: null, logLine, profileOverride: null, prefillKernel, prefillRowWidth: 0, route, prefillsBackend: null, prefillsLane: null, crossingBackends);

    /// <summary>
    /// The same split-route seam a host opens with, carrying the stage-side posture and resolved budgets the public open carries too: an ordered
    /// execution-backend set covering every resolved segment in order lets each span run on the backend its own placement named while this service's
    /// decode-side backend stays the engine's default. Assembly-internal for hosts that resolve a layer split at model open.
    /// </summary>
    internal static TransformerModel Open(
        GgufFile file, IBackend backend, int maxContextTokens, Action<string>? logLine, PrefillKernel prefillKernel, ModelRoute route,
        IReadOnlyList<IBackend> crossingBackends, ResolvedBudgetSet budgets) =>
        new(file, backend, maxContextTokens, weights: null, logLine, profileOverride: null, prefillKernel, prefillRowWidth: 0, route, prefillsBackend: null, prefillsLane: null, crossingBackends, budgets);

    /// <summary>
    /// The alternate-<see cref="IWeightSource"/> form of the split-route seam: every tensor read goes through the supplied
    /// source instead of a mmap view, while each resolved segment still runs on the backend supplied for its position and
    /// the first span's posture follows the same ordered-set rules as the seam above. Tensor metadata (names, shapes,
    /// formats, offsets) still comes from <paramref name="file"/>, and the caller keeps ownership of the source — it must
    /// outlive the model and is disposed by the caller, not here. Assembly-internal: hosts resolve placement onto the
    /// memory-mapped seam; this one lets a caller open a split over a weight source the engine never created.
    /// </summary>
    internal static TransformerModel Open(
        GgufFile file, IWeightSource weights, IBackend backend, int maxContextTokens, Action<string>? logLine, PrefillKernel prefillKernel,
        ModelRoute route, IReadOnlyList<IBackend> crossingBackends) =>
        new(file, backend, maxContextTokens, weights, logLine, profileOverride: null, prefillKernel, prefillRowWidth: 0, route, prefillsBackend: null, prefillsLane: null, crossingBackends);

    // The execution spans this engine runs its route over, resolved positionally from the segments and the compute backends opened for them.
    // One span holding every layer on the device its own backend reports is what every model served today resolves to; a segment naming another
    // device refuses naming that segment and the seam accelerator execution lands in rather than advancing part of the model on some device that
    // happens to work, exactly as it always did — where no crossing was supplied for a position, this engine's own opened backend is what answers
    // for it. A split route then needs boundaries that ascend from layer 0 and cover every layer exactly once; anything missing or out of order
    // refuses naming what lands short. The check lands before any weight view is spent on a shape that cannot run, exactly as an unrunnable kernel selection does.
    private (ExecutionSpan[] DecodeSide, ExecutionSpan[] PrefillSide) ResolveExecutionSpans(GgufFile file, IReadOnlyList<IBackend>? crossingBackends, IBackend? stageSidePrefills)
    {
        ModelRoute route = Plan.Route;

        if (_prefillsLane is not null && route.Segments.Count > 1)
        {
            throw new ModelException(
                $"model '{_prefillsLane.ModelName}' opens a dedicated prefill lane over a route of {route.Segments.Count} segments: this build carries one execution backend for the stage side, so its prompt windows have no crossing to run on — resolve placement onto one segment or open no lane");
        }

        // A position answers with the backend supplied for it, and with nothing where none was supplied yet. Each segment meets that answer first:
        // a named device this build cannot run is refused naming the segment before anything else about the shape is examined, exactly as an
        // unsupplied crossing to 'cuda' always was. A supplied set covers either every position after the first — where this engine's own backend answers position 0 — or every resolved position in order, which is how a host opens a split whose first span runs on a backend other than its decode-side one.
        bool fullSet = crossingBackends is not null && crossingBackends.Count == route.Segments.Count;

        IBackend? BackendAt(int position) =>
            fullSet ? (position < crossingBackends!.Count ? crossingBackends[position] : null)
            : position == 0 || (crossingBackends is not null && position - 1 < crossingBackends.Count)
                ? (position == 0 ? _backend : crossingBackends![position - 1])
                : null;

        for (int index = 0; index < route.Segments.Count; index++)
        {
            DeviceSegment segment = route.Segments[index];
            IBackend? resolved = BackendAt(index);
            IBackend engine = resolved ?? _backend;
            string owner = ReferenceEquals(engine, _backend) ? "the opened compute backend" : $"the execution backend resolved for position {index}";

            if (!string.Equals(segment.DeviceId, engine.DeviceId, StringComparison.Ordinal))
            {
                throw new ModelException(
                    $"{segment.Describe(index)} names a device this engine has no backend for: {owner} reports " +
                    $"'{engine.DeviceId}' and architecture '{Plan.Architecture}' resolves a route of {route.Segments.Count} segment(s) — " +
                    $"accelerator-backed segments run in '{ModelRoute.CudaBackendSeam}', which is not part of this build, so a crossing to " +
                    $"'{segment.DeviceId}' is refused rather than run on CPU");
            }

            // A segment resident in accelerator memory is answered only by the backend that reports reading weights from device memory it owns: a crossing whose backend states the other posture would run the span against a residency nothing here addresses, so it refuses naming the segment, both postures, and the seam device residency runs in. A host-resident segment is answered by any backend opened for its device, exactly as every unsplit route resolved until now.
            if (segment.WeightTier == WeightTier.Vram && engine.WeightTier != Lair.Contracts.WeightTier.Vram)
            {
                throw new ModelException(
                    $"{segment.Describe(index)} holds its weights at tier '{segment.WeightTier}': {owner} reports the " +
                    $"'{engine.WeightTier}' weight tier, so accelerator-resident weights run in '{ModelRoute.CudaBackendSeam}' only behind a backend opened with device-resident weight copies — architecture " +
                    $"'{Plan.Architecture}' is refused rather than run against a residency it cannot address");
            }
        }

        int supplied = crossingBackends?.Count ?? 0;
        if (supplied != route.Segments.Count - 1 && !fullSet)
        {
            throw new ModelException(
                $"architecture '{Plan.Architecture}' resolves a route of {route.Segments.Count} segment(s), which needs one execution backend for every position after the first, and this open supplied {supplied} — supply exactly {route.Segments.Count - 1}, or all {route.Segments.Count} backends covering every resolved position in order, or keep this model one span");
        }

        int layers = Plan.LayerCount;
        bool split = route.Segments.Count > 1;

        // The boundary chain is settled next: a split route names where each span's layers begin, ascending from layer 0 and ending inside this file's own range so the spans cover every layer exactly once — except that the final span may start exactly at the layer count and run nothing, which hands every transformer layer to an earlier span while the output head still runs on the backend that finished it.
        int[] boundaries = new int[route.Segments.Count];

        if (split)
        {
            for (int index = 0; index < route.Segments.Count; index++)
            {
                DeviceSegment segment = route.Segments[index];
                int boundary = segment.SplitPosition;

                if (boundary == ModelRoute.WholeModelSplitPosition || boundary < 0)
                {
                    throw new ModelException(
                        $"{segment.Describe(index)} names no split position: a route of {route.Segments.Count} segments must name where each span's layers begin, and architecture " +
                        $"'{Plan.Architecture}' opens over boundaries covering its {layers} layer(s) once");
                }

                if (index == 0 && boundary != 0)
                {
                    throw new ModelException(
                        $"{segment.Describe(index)} starts the split at boundary '{boundary}': a route's first span covers this model from layer 0 — name '0' to open it");
                }

                if (index > 0 && boundary <= boundaries[index - 1])
                {
                    throw new ModelException(
                        $"{segment.Describe(index)} starts at boundary '{boundary}', which does not ascend past the segment before it at " +
                        $"'{boundaries[index - 1]}': a split route hands every layer to exactly one span");
                }

                if (boundary > layers || (index < route.Segments.Count - 1 && boundary >= layers))
                {
                    throw new ModelException(
                        $"{segment.Describe(index)} starts at boundary '{boundary}', past this file's last layer {layers - 1}: architecture " +
                        $"'{Plan.Architecture}' resolves {layers} layer(s), and a span that begins outside them runs nothing — only the final span may start exactly at the layer count");
                }

                boundaries[index] = boundary;
            }
        }

        ExecutionSpan[] decodeSide = new ExecutionSpan[route.Segments.Count];
        ExecutionSpan[] prefillSide = new ExecutionSpan[route.Segments.Count];

        for (int index = 0; index < route.Segments.Count; index++)
        {
            IBackend engine = BackendAt(index)!; // every position has its own backend once the count above is satisfied, including this one

            int start = split ? boundaries[index] : 0;
            int end = split && index + 1 < route.Segments.Count ? boundaries[index + 1] : layers;

            decodeSide[index] = new ExecutionSpan { Backend = engine, LayerStart = start, LayerEnd = end };

            // Prompt windows run the first span on their stage-side backend where one was supplied — a lane's capped second backend or an offloaded prompt side — and otherwise on whatever execution answers this route's first position: this engine's own backend today, and the split's accelerator-backed first span when a full ordered set covers it.
            prefillSide[index] = new ExecutionSpan { Backend = index == 0 ? stageSidePrefills ?? engine : engine, LayerStart = start, LayerEnd = end };
        }

        // The load-time capability gate every resolved span passes before this engine opens: each span's own range of
        // weight formats must be formats the compute backend resolved for that position executes through GEMV. A span
        // naming a format its backend reports it has no GEMV kernel for refuses at open naming the model, the format, the
        // segment, and that backend — the posture every other unimplemented capability takes, and the same fact the
        // backend's own mid-forward refusal used to raise one weight pass late. Nothing falls back to CPU for a
        // device-named segment, and the check reads the tensor table alone: an unsupported route never spends a weight view.
        ValidateSpanWeightCapability(ModelDisplayName(file), file.Tensors, route, decodeSide, prefillSide);

        // The dense-matrix selection answers to the same per-span gate: a prompt-window range naming a weight format its own resolved backend has no blocked matmul for refuses here at load — before any window dispatches and with no fall back to another kernel or device — rather than raising one weight pass late.
        ValidateSpanMatmulCapability(_prefillKernel, ModelDisplayName(file), route, file.Tensors, prefillSide);

        return (decodeSide, prefillSide);
    }

    // The capability gate itself: the formats one span's layer range carries through GEMV, plus the output head the tail
    // span's own backend dots, each checked against the backend that span runs on for the decode side and the stage side
    // alike — the two differ only where a dedicated lane or an offloaded prompt backend answers the first position, and a
    // format either one would have to dot names the refusal.
    private void ValidateSpanWeightCapability(string model, LogicalTensorTable tensors, ModelRoute route, ExecutionSpan[] decodeSide, ExecutionSpan[] prefillSide)
    {
        for (int index = 0; index < route.Segments.Count; index++)
        {
            DeviceSegment segment = route.Segments[index];
            ExecutionSpan decode = decodeSide[index];
            ExecutionSpan prefill = prefillSide[index];

            List<QuantFormat> formats = CollectWeightFormats(Plan, tensors, decode.LayerStart, decode.LayerEnd, includeSparse: true);

            if (index == route.Segments.Count - 1)
            {
                // The output head is no layer: its GEMV runs on the backend that finished the last span of each side, and a
                // file tying its logits to the token embedding dots those rows there instead.
                LogicalTensor? lmHead = Plan.Tensors[TensorRole.OutputHead].Resolve(tensors)
                    ?? Plan.Tensors[TensorRole.TokenEmbedding].Resolve(tensors);

                if (lmHead is not null && !formats.Contains(lmHead.Format))
                {
                    formats.Add(lmHead.Format);
                }
            }

            ValidateSpanCapability(model, segment, index, decode.Backend, formats);

            if (!ReferenceEquals(prefill.Backend, decode.Backend))
            {
                ValidateSpanCapability(model, segment, index, prefill.Backend, formats);
            }
        }
    }

    private void ValidateSpanCapability(string model, DeviceSegment segment, int index, IBackend engine, List<QuantFormat> formats)
    {
        List<QuantFormat> missing = [];
        foreach (QuantFormat format in formats)
        {
            if (!engine.SupportsGemv(format))
            {
                missing.Add(format);
            }
        }

        if (missing.Count > 0)
        {
            throw WeightCapabilityRefusal(model, segment, index, engine, missing);
        }
    }

    // The refusal wording a span naming a format its own compute backend cannot dot gets: the model, the offending
    // format(s), the segment and the backend answering it, and the reason — that backend reports no GEMV kernel for them —
    // beside the two ways to run the file instead. It extends the FormatsWithoutKernels refusal shape for the route's
    // other side, and replaces the mid-decode Unsupported throw the same format used to raise after work had dispatched.
    private static ModelException WeightCapabilityRefusal(string model, DeviceSegment segment, int index, IBackend engine, List<QuantFormat> missing) => new(
        $"model '{model}', {segment.Describe(index)}: weight format(s) {string.Join(", ", missing)} have no GEMV kernel in the compute backend " +
        $"'{engine.DeviceId}' resolved for this segment: that backend reports it executes GEMV for other formats only, so this model is refused at load " +
        $"naming the format and the backend rather than raised mid-decode — resolve placement onto '{ModelRoute.CpuDeviceId}' or re-quantize this file " +
        $"to a format '{engine.DeviceId}' executes");

    // The refusal the bounded synthetic qwen4exp seam raises for a serving posture it has no composition over: the model, the
    // posture, and what the seam runs instead — named rather than silently dropped, so an open never serves some other route
    // behind the selection that happens to work.
    private ModelException Qwen4ExpPostureRefusal(string posture) => new(
        $"model '{_modelDisplay}' architecture 'qwen4exp' opens its forward on the bounded synthetic qwen4exp engine seam, " +
        $"which runs CPU-only GEMV and slab kernels over the file's own packed weight views with per-sequence state sized at the opened context ceiling: " +
        $"{posture} has no composition over that seam — omit it to run the fixture through the assembly-internal qwen4exp " +
        $"chunk forwards, or resolve placement onto a standard family");

    // The refusal a serving forward gets when this engine opened on the qwen4exp seam instead of the standard loading: the seam
    // forwards chunks against its own per-sequence state, and the persistent production caches these routes hand a sequence
    // object to do not exist for this family yet.
    private InvalidOperationException Qwen4ExpServingRefusal(string routeName) => new(
        $"model '{_modelDisplay}' opened the bounded synthetic qwen4exp forward seam: {routeName} runs against persistent " +
        $"production caches this family does not have yet — forward the fixture through the assembly-internal qwen4exp " +
        $"chunk seam instead");

    // The name every load-time refusal this engine raises calls the model by: general.name when the file declares it, and
    // the file path when it does not, the same identity the structured load lines report.
    private static string ModelDisplayName(GgufFile file)
    {
        GgufMetadata metadata = file.Metadata;
        string name = metadata.Has("general.name") ? metadata.Get("general.name").AsString() : "";
        return name.Length > 0 ? name : file.Path;
    }

    private TransformerModel(GgufFile file, IBackend backend, int maxContextTokens, IWeightSource? weights = null, Action<string>? logLine = null, ModelProfile? profileOverride = null, PrefillKernel prefillKernel = PrefillKernel.GemV, int prefillRowWidth = 0, ModelRoute? route = null, IBackend? prefillsBackend = null, PrefillLaneLayout? prefillsLane = null, IReadOnlyList<IBackend>? crossingBackends = null, ResolvedBudgetSet? budgets = null, KvFormatPair? kvFormats = null, IntegerDotPath? quantizedDot = null, IDeviceAttentionStep? devicePrefill = null)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxContextTokens);

        _backend = backend;
        Plan = profileOverride is null ? ModelPlans.Resolve(file, logLine) : ModelPlans.Resolve(file, profileOverride, logLine);
        Desc = Plan.ToDesc();
        MaxContextTokens = maxContextTokens;
        _modelDisplay = ModelDisplayName(file); // every capability refusal this engine raises — at open and on a re-point alike — calls this model by the same name the structured load lines report.

        // A qwen4exp plan resolves its per-layer interleave, recurrent-state geometry, indexer budgets and hyper-connection
        // mixing from the file's own declarations — that resolution is what residency accounting sizes against today, and it
        // also decides this engine's forward seam: a valid synthetic fixture opens on the bounded CPU-only qwen4exp engine
        // instead of the standard dense loading. This family's linear layers carry no dense q/k/v triplet and its blocks carry
        // none of the standard norms the dense loader requires, so running that loader against it would refuse material the
        // family legitimately does not ship; the dedicated engine reads the file's own tensors through the plan's tensor map,
        // sequences every block in the reference order — hyper-connection mix, gated-delta-net chunk or indexer-selected
        // restricted attention, combine, mix, feed-forward, combine, then the global output mix before the head — with explicit
        // per-sequence state advanced across chunk splits, and sizes every cache at the opened context ceiling. That is a
        // bounded synthetic path, not target serving: the postures that compose against the standard machinery (quantized KV,
        // dedicated prefill lanes, offloaded or crossing backends, device prefill steps, split routes, non-GemV kernels) have
        // no composition over this seam and refuse naming it — like every unimplemented capability before it — rather than
        // half-forwarding a model whose logits would be wrong.
        if (Plan.Qwen4Exp is { } qwen4expGeometry)
        {
            if (!string.Equals(backend.DeviceId, ModelRoute.CpuDeviceId, StringComparison.Ordinal))
            {
                throw Qwen4ExpPostureRefusal($"compute backend '{backend.DeviceId}'");
            }

            if (route is not null && route.Segments.Count > 1)
            {
                throw Qwen4ExpPostureRefusal($"a route of {route.Segments.Count} segments");
            }

            if (kvFormats is { } declaredPair && !declaredPair.IsFloatStorage)
            {
                throw Qwen4ExpPostureRefusal($"quantized KV storage k='{KvSideFormatNames.Of(declaredPair.K)}' v='{KvSideFormatNames.Of(declaredPair.V)}'");
            }

            if (prefillsLane is not null || prefillsBackend is not null || crossingBackends is not null || devicePrefill is not null)
            {
                throw Qwen4ExpPostureRefusal("a dedicated prefill lane, an offloaded prefill backend, crossing backends or a device prefill step");
            }

            if (prefillKernel != PrefillKernel.GemV)
            {
                throw Qwen4ExpPostureRefusal($"prefill kernel '{prefillKernel}'");
            }

            Plan = route is null ? Plan : Plan with { Route = route };
            Plan = budgets is null ? Plan : Plan with { Budgets = budgets };

            _prefillKernel = prefillKernel;
            _kvFormats = kvFormats ?? KvFormatPair.Default;

            // The weight view lands here rather than further down the standard path: this seam reads nothing but the file's
            // own tensors, and a posture refused above never spends one.
            _ownsWeights = weights is null;
            _weights = weights ?? new GgufMmapWeightSource(file);
            _fileTensors = file.Tensors;

            _qwen4Exp = new Qwen4ExpEngine(Plan, file.Tensors, _weights, backend, maxContextTokens, _modelDisplay);

            if (logLine is not null)
            {
                GgufMetadata metadata = file.Metadata;
                string modelName = metadata.Has("general.name") ? metadata.Get("general.name").AsString() : "";
                logLine(
                    $"qwen4exp_engine model='{(modelName.Length > 0 ? modelName : file.Path)}' " +
                    $"architecture='{Plan.Architecture}' linear_layers={qwen4expGeometry.LinearLayerCount} " +
                    $"sparse_full_layers={qwen4expGeometry.FullLayerCount} hyper_streams={qwen4expGeometry.HyperConnections.StreamCount} " +
                    $"hyper_rank={qwen4expGeometry.HyperConnections.LowRank} context_ceiling={maxContextTokens}");
            }

            return;
        }

        // The formats this file's dense weight matrices carry, read from the plan's tensor map before any payload is
        // touched: it decides whether a selection that multiplies a whole sub-batch against one matrix can run here at
        // all, and it names the formats a refusal reports.
        _matmulFormats = ResolveMatmulFormats(Plan, file.Tensors);

        // A prefill kernel selection this engine has no kernel for is refused at open, the posture every other
        // unimplemented capability takes: the selection stays visible in the resolved value's vocabulary, the
        // refusal names it and what is missing, and nothing falls back to the route that happens to work. The batched-
        // row window and its row-tiled variant run for every family; the dense-matrix shape needs this file to carry
        // dense weight matrices at all, and which compute backend must report a blocked matmul for each of their
        // formats is decided per resolved execution span below. Both checks land before any tensor maps or weight views
        // are built, so an unrunnable selection never spends either on it.
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
        // put there — which is where every model served today runs. Resolved budget sets land the same way: a host resolves what
        // it declares against the pools it actually opened, and this engine carries that shape unchanged rather than re-deriving
        // ceilings from some other source while the model stays open. A file opened without budgets keeps the empty set — today's
        // behavior, where only KV block counts bound anything.
                Plan = route is null ? Plan : Plan with { Route = route };

        // The execution spans this engine runs that route over resolve below, once the stage-side posture has landed — each segment is checked against the backend resolved to run it.

        // Budgets land on the plan exactly as the route did above: what a host resolved against its opened pools arrives unchanged here.
        Plan = budgets is null ? Plan : Plan with { Budgets = budgets };

        // The KV format pair this engine's forwards serve lands here, once, before any allocation: a quantized pair closes
        // one fixed attention instantiation for every forward of this engine, and a posture that instantiation cannot serve
        // refuses at open naming what the opened posture declares and what the quantized route lacks — never a silent fall
        // back to the FP32 posture. Quantized attention runs CPU kernels over host-resident KV through the batched-row
        // window, dotting whole group-aligned head regions.
        _kvFormats = kvFormats ?? KvFormatPair.Default;

        if (_kvFormats.IsFloatStorage)
        {
            _floatAttention = new FloatAttentionRoute(this);
        }
        else
        {
            if (quantizedDot is null)
            {
                throw new ModelException(
                    $"opening quantized KV storage k='{KvSideFormatNames.Of(_kvFormats.K)}' v='{KvSideFormatNames.Of(_kvFormats.V)}' needs the integer-dot encoding the opened backend resolved: " +
                    "pass the opened CPU backend's resolved dot path beside the pair — the quantized attention kernel runs it per scored position and never re-resolves it.");
            }

            if (Plan.Route.Segments.Count > 1 || crossingBackends is not null)
            {
                throw QuantizedPostureRefusal($"a route of {Plan.Route.Segments.Count} segments");
            }

            if (prefillsLane is not null || prefillsBackend is not null)
            {
                throw QuantizedPostureRefusal(prefillsLane is not null ? "a dedicated prefill lane" : "an offloaded prefill backend");
            }

            if (!string.Equals(backend.DeviceId, ModelRoute.CpuDeviceId, StringComparison.Ordinal))
            {
                throw QuantizedPostureRefusal($"device '{backend.DeviceId}'");
            }

            if (prefillKernel != PrefillKernel.GemV)
            {
                throw QuantizedPrefillRefusal(prefillKernel);
            }

            if (Plan.HeadDim % KvFormatPair.GroupValues != 0)
            {
                throw QuantizedHeadDimRefusal();
            }

            if (_kvFormats.K == KvSideFormat.Q8_0 && _kvFormats.V == KvSideFormat.Q8_0)
            {
                _q8q8Attention = new QuantizedAttentionRoute<Q8Block, Q8Block>(quantizedDot.Value, Plan.HeadCount, Plan.HeadDim);
            }
            else if (_kvFormats.K == KvSideFormat.Q8_0 && _kvFormats.V == KvSideFormat.Q4_0)
            {
                _q8q4Attention = new QuantizedAttentionRoute<Q8Block, Q4Block>(quantizedDot.Value, Plan.HeadCount, Plan.HeadDim);
            }
            else
            {
                throw new ModelException(
                    $"quantized KV pair k='{KvSideFormatNames.Of(_kvFormats.K)}' v='{KvSideFormatNames.Of(_kvFormats.V)}' has no attention route in this build: " +
                    "the quantized kernel instantiates over 'q8_0' on the dot side and 'q8_0' or 'q4_0' on the value side — declare 'q8_0'/'q8_0' or 'q8_0'/'q4_0', or omit kv.quant for the FP32 posture");
            }
        }


        // A dedicated lane posture arrives as a pair — the second execution backend prompt windows run on, plus the resolved split it was capped from.
        // Half of it is refused at open naming what is missing; the backend must report this engine's own device, because weights, norms, rotation tables, and KV
        // bytes are shared by construction and only exist resident where the route says they live — a lane on some other device would read tensors that
        // were never mapped there. The split itself carries load-resolved worker caps; an engine refuses to open over shares one of which holds no worker.
        if (prefillsLane is { } declaredLane)
        {
            if (prefillsBackend is null)
            {
                throw new ModelException(
                    $"a dedicated prefill lane for model '{declaredLane.ModelName}' arrives with only half of its pair: " +
                    "no execution backend for the lane to run on — pass both together, or neither.");
            }

            if (prefillsBackend.DeviceId != backend.DeviceId)
            {
                throw new ModelException(
                    $"model '{declaredLane.ModelName}' declares a dedicated prefill lane on device '{prefillsBackend.DeviceId}', but this engine runs its weights and KV on " +
                    $"'{backend.DeviceId}': the lane shares those tensors, so it must run on the same device — resolve placement onto one device or open no lane.");
            }

            if (declaredLane.PrefillShareThreads < 1 || declaredLane.DecodeShareThreads < 1)
            {
                throw new ModelException(
                    $"model '{declaredLane.ModelName}' opens a dedicated prefill lane over shares of {declaredLane.PrefillShareThreads} prefill and " +
                    $"{declaredLane.DecodeShareThreads} decode workers: every lane needs at least one worker — resolve the split wider, or open no lane.");
            }

            _prefillsLane = declaredLane;
        }
        else if (prefillsBackend is not null)
        {
            // A prefill offload posture arrives as a backend alone: prompt windows run on it while every decode-side forward stays on this engine's own. It caps no worker share, so it carries no layout — and this build runs that shape only over the single DRAM-resident span every unsplit route resolves to, which is where reading host-resident weight bytes from another device holds. A split route refuses naming what lands short rather than staging an unsized lane through crossings; a residency other than DRAM was already refused against each segment's own backend below, and a weight format this backend reports no GEMV kernel for refused at open naming the model, the format, and this backend, never run on some other device behind the selection that happens to work.
            ModelRoute offloadRoute = Plan.Route;

            if (offloadRoute.Segments.Count > 1)
            {
                throw new ModelException(
                    $"an offloaded prefill backend on device '{prefillsBackend.DeviceId}' arrives with no lane layout over a route of {offloadRoute.Segments.Count} segments: " +
                    "prompt windows run across host-resident weights only where the model resolves to one DRAM-resident span — declare placement.prefill_lane 'dedicated' for a same-device lane pair, or supply crossing backends as their own posture.");
            }
        }

        _prefillsBackend = prefillsBackend ?? backend;

        // The resolved segments become the execution spans every forward walks: one range holding this model's whole layer count on this engine's own device is what every route served today resolves to, and a supplied crossing backend takes each position it covers. Each segment lands against its own backend here — before any weight view or activation storage exists behind it.
        (_mainSpans, _prefillSpans) = ResolveExecutionSpans(file, crossingBackends, prefillsBackend);

        // Row capacity of every batched stream: whichever forward family needs the wider sub-batch — never below what a
        // decode batch may advance, and above that only when the plan resolves wider prompt windows. Allocation is the
        // one thing width moves; no arithmetic in any forward reads it as an operand.
        _rowCapacity = Math.Max(MaxBatchSequences, Plan.PrefillRowWidth);

        // A resolved prefill-buffer ceiling clamps the row window this engine runs: the widest sub-batch whose per-row activation cost fits
        // what this model's own declared ceilings allow. No chain, or a chain whose every capped position already fits the resolved width,
        // clamps nothing — today's shape exactly. A binding that allows no positive row still narrows to the engine's minimum one-row window
        // and reports itself unfit: what capacity accounting then seats names why, rather than a window running over its ceiling quietly.
        _prefillRowActivationBytes = PrefillRouteRowActivationBytes(Plan);
        _prefillClamp = Plan.Budgets.For(BudgetResource.PrefillBuffers) is { } prefillChain
            ? PrefillBufferClamp.Compute(prefillChain, _prefillRowActivationBytes, Plan.PrefillRowWidth)
            : null;
        _effectivePrefillRowWidth = _prefillClamp is null ? Plan.PrefillRowWidth : Math.Max(1, _prefillClamp.AllowedRowWidth);

        // A clamped dedicated lane sizes its own family beside its own ceiling rather than holding storage wider than anything may run on
        // it; an unclamped posture sizes at this engine's row capacity exactly as before any clamp existed. Shared-posture streams stay the
        // shared batched arrays sized for the resolved width — a clamp bounds what a window runs there, it never reallocates shared storage.
        _laneRowCapacity = _prefillsLane is null
            ? 0
            : _prefillClamp is null ? _rowCapacity : Math.Min(_rowCapacity, Math.Max(1, _prefillClamp.AllowedRowWidth));

        // The device prefill posture lands beside the resolved spans and row width it composes against: every fact of the opened step is checked against this engine's own plan here, a refused composition spends no weight view, and nothing about the decode side changes — host attention still answers single-token forwards exactly as before.
        if (devicePrefill is not null)
        {
            _devicePrefill = InstallDevicePrefill(devicePrefill, maxContextTokens);
        }

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
            // a time for routing plus per-sequence copies of every batched selection, the step's union active-expert
            // set with its grouped member rows and weights (an upper bound of one entry per selection slot), and one
            // hidden-sized buffer carrying each group's activations and down results in place.
            _moeLogits = new float[_sparseFfn.ExpertCount];
            _moeIds = new int[_sparseFfn.RoutedExpertsPerToken];
            _moeWeights = new float[_sparseFfn.RoutedExpertsPerToken];
            _batchMoeIds = new int[_rowCapacity * _sparseFfn.RoutedExpertsPerToken];
            _batchMoeWeights = new float[_rowCapacity * _sparseFfn.RoutedExpertsPerToken];
            _moeUnion = new int[_rowCapacity * _sparseFfn.RoutedExpertsPerToken];
            _moeUnionCounts = new int[_rowCapacity * _sparseFfn.RoutedExpertsPerToken];
            _moeMembersFlat = new int[_rowCapacity * _sparseFfn.RoutedExpertsPerToken];
            _moeMemberWeights = new float[_rowCapacity * _sparseFfn.RoutedExpertsPerToken];
            _moeSubsetIn = new float[_rowCapacity * Plan.HiddenSize];

            // The slab provider over this model's weight arena and the CPU gather kernel every selected chain runs
            // through, resolved once here so no forward re-resolves either. An injected source that already answers
            // slab resolutions (a residency or repacked provider) is used directly; anything else — the mmap arena
            // today, an arena-copied alternate — gets a contiguous-slab view of its own payloads.
            _expertSlabs = _weights as IExpertSlabSource ?? new ExpertSlabWeightSource(_weights);
            _gatherKernel = new CpuGatherKernel();

            // A slab source that is itself a residency tier answers this model's expert reads from its hot set: the forward folds every layer step's union into the tier before gathering, so capacity/cost-selected experts resolve resident and everything else falls through cold. Any other source keeps today's prefetch-only posture untouched — no tier runs without one being opened here.
            _residencyTier = _expertSlabs as IResidencyTier;

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

        // The activation families land here, at open: the default family aliases every mutable buffer this block has just sized — decode-side forwards keep writing through exactly those arrays — and a dedicated posture gives prompt windows their own family on top of it. A lane's streams are sized to the engine's whole window ceiling so any legal sub-batch fits its own storage whatever width placement resolved, while weights, norms, rotation tables, and KV bytes stay shared between the two routes.
        _decodeAct = new ActivationFamily
        {
            BatchCur = _batchCur,
            BatchNormed = _batchNormed,
            BatchQ = _batchQ,
            BatchK = _batchK,
            BatchV = _batchV,
            BatchAttnOut = _batchAttnOut,
            BatchProj = _batchProj,
            BatchGateUp = _batchGateUp,
            BatchUp = _batchUp,
            TiledRowMax = _tiledRowMax,
            TiledRowSumExp = _tiledRowSumExp,
            MoeScores = _moeLogits,
            MoeIds = _moeIds,
            MoeWeights = _moeWeights,
            BatchMoeIds = _batchMoeIds,
            BatchMoeWeights = _batchMoeWeights,
            MoeUnion = _moeUnion,
            MoeUnionCounts = _moeUnionCounts,
            MoeMembersFlat = _moeMembersFlat,
            MoeMemberWeights = _moeMemberWeights,
            SubsetIn = _moeSubsetIn,
            FinalNormed = _normed,
            FinalLogits = _logits,
        };

        if (_prefillsLane is not null)
        {
            // A dedicated lane's own mutable storage: the same shapes as the default family at the window width this engine runs — the resolved
            // capacity unclamped, a clamped lane sized to its ceiling instead — allocated once and never grown. An overlapped prompt window
            // writes here while decode-side forwards read the arrays above.
            _laneAct = new ActivationFamily
            {
                BatchCur = new float[_laneRowCapacity * hidden],
                BatchNormed = new float[_laneRowCapacity * hidden],
                BatchQ = new float[_laneRowCapacity * Plan.HeadCount * Plan.HeadDim],
                BatchK = new float[_laneRowCapacity * Plan.KvHeadCount * Plan.HeadDim],
                BatchV = new float[_laneRowCapacity * Plan.KvHeadCount * Plan.HeadDim],
                BatchAttnOut = new float[_laneRowCapacity * Plan.HeadCount * Plan.HeadDim],
                BatchProj = new float[_laneRowCapacity * hidden],
                BatchGateUp = new float[_laneRowCapacity * Plan.IntermediateSize],
                BatchUp = new float[_laneRowCapacity * Plan.IntermediateSize],
                TiledRowMax = new float[_laneRowCapacity * Plan.HeadCount],
                TiledRowSumExp = new float[_laneRowCapacity * Plan.HeadCount],
                MoeScores = _moeLogits.Length > 0 ? new float[_moeLogits.Length] : [],
                MoeIds = _moeIds.Length > 0 ? new int[_moeIds.Length] : [],
                MoeWeights = _moeWeights.Length > 0 ? new float[_moeWeights.Length] : [],
                BatchMoeIds = new int[_laneRowCapacity * (_sparseFfn?.RoutedExpertsPerToken ?? 0)],
                BatchMoeWeights = new float[_laneRowCapacity * (_sparseFfn?.RoutedExpertsPerToken ?? 0)],
                MoeUnion = new int[_laneRowCapacity * (_sparseFfn?.RoutedExpertsPerToken ?? 0)],
                MoeUnionCounts = new int[_laneRowCapacity * (_sparseFfn?.RoutedExpertsPerToken ?? 0)],
                MoeMembersFlat = new int[_laneRowCapacity * (_sparseFfn?.RoutedExpertsPerToken ?? 0)],
                MoeMemberWeights = new float[_laneRowCapacity * (_sparseFfn?.RoutedExpertsPerToken ?? 0)],
                SubsetIn = new float[_laneRowCapacity * hidden],
                FinalNormed = new float[hidden],
                FinalLogits = new float[Plan.VocabSize],
            };
        }
        else
        {
            // Shared posture: prompt windows write through the very family every forward has always used — nothing extra is sized, and behavior is byte-for-byte what it was before any lane vocabulary existed.
            _laneAct = _decodeAct;
        }

        // Crossing carriers land with the families they stage between. Each arena belongs to the backend its own span runs on — allocated here through that backend's aligned allocator at this engine's whole row capacity, so every legal sub-batch crosses through storage sized once and never grown. A dedicated lane posture over a split route was refused above, which is why both routes of any crossing share one family's arenas today: under a shared posture they are the same mutable streams either way. A single-span route — every model served today — stages nothing at all.
        if (_mainSpans.Length > 1)
        {
            _stagedCarriers = new ResidualCarrier[_mainSpans.Length];
            ExecutionCrossing[] staged = new ExecutionCrossing[_mainSpans.Length - 1];

            for (int index = 1; index < _mainSpans.Length; index++)
            {
                ExecutionSpan span = _mainSpans[index];
                ResidualCarrier carrier = new(span.Backend, _rowCapacity, hidden);
                _stagedCarriers[index] = carrier;

                staged[index - 1] = new ExecutionCrossing(
                    SegmentIndex: index,
                    LayerStart: span.LayerStart,
                    LayerEnd: span.LayerEnd,
                    StagingOwnerDeviceId: carrier.OwnerDeviceId,
                    StagingBytes: carrier.Bytes);
            }

            _stagedCrossings = staged;
        }
        else
        {
            _stagedCarriers = [];
            _stagedCrossings = [];
        }

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
    /// Forwards a window of prompt positions in sub-batches of up to this engine's effective row width — the plan's resolved
    /// <see cref="ModelPlan.PrefillRowWidth"/> narrowed by a declared prefill-buffer ceiling when one binds — and returns the final row's logits. Each row seeds its own residual stream from its token's embedding row and
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
    public Span<float> Prefill(IReadOnlyList<int> tokens, IKvSequence kv)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(kv);

        if (_qwen4Exp is not null)
        {
            throw Qwen4ExpServingRefusal(nameof(Prefill));
        }

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

        // The window-total span covers every sub-batch this call runs, so a profiled prompt reports one total beside the stage totals its windows accumulated.
        long forwardStart = BeginDenseProfile();

        if (_q8q8Attention is { } q8q8)
        {
            PrefillBody(tokens, ServeSequence<SequenceKv<Q8Block, Q8Block>>(kv), q8q8);
        }
        else if (_q8q4Attention is { } q8q4)
        {
            PrefillBody(tokens, ServeSequence<SequenceKv<Q8Block, Q4Block>>(kv), q8q4);
        }
        else
        {
            PrefillBody(tokens, ServeSequence<SequenceKv>(kv), _floatAttention!);
        }

        AccumulateDenseProfile(ref _denseTotalTicks, forwardStart);

        // The logits left by the window's final row live in this route's family — an alias of the model-owned buffer for a shared posture, lane-owned storage when dedicated. Valid on whichever thread ran the forward until its next forward overwrites it.
        return _laneAct.FinalLogits.AsSpan();
    }

    // The window loop over one closed (sequence, attention-route) pair: the row window this engine runs splits at the plan's
    // resolved count when no declared prefill-buffer ceiling binds, or the widest sub-batch those ceilings allow otherwise.
    // Every width carries the same positions, KV bytes, and logits — a clamp narrows batched weight traversals and nothing
    // else. Read fresh per call like every other forward-shape value.
    private void PrefillBody<TSeq, TR>(IReadOnlyList<int> tokens, TSeq kv, TR attention)
        where TSeq : class, IKvSequence
        where TR : IKvAttentionRoute<TSeq>
    {
        int n = tokens.Count;
        int basePosition = kv.Length;
        int prefillRows = _effectivePrefillRowWidth;

        for (int offset = 0; offset < n; offset += prefillRows)
        {
            // Prompt windows advance through the resolved execution spans in order: their first span runs on this engine's own backend (the dedicated lane's capped second backend when placement declares one), writing through that route's activation family either way.
            RunPrefillWindow(tokens, offset, Math.Min(prefillRows, n - offset), basePosition, kv, emitLogits: offset + prefillRows >= n, _laneAct, attention);
        }
    }

    // The sequence a forward may run on this engine: the concrete type its opened pair hands out. A host that passes some
    // other storage's sequence is refused naming the pair rather than forwarding against bytes this engine never wrote.
    private TSeq ServeSequence<TSeq>(IKvSequence kv) where TSeq : class =>
        kv as TSeq ?? throw new InvalidOperationException(
            $"this model opened kv k='{KvSideFormatNames.Of(_kvFormats.K)}' v='{KvSideFormatNames.Of(_kvFormats.V)}': its forwards run over the sequence type that pool hands out — open the sequence from the opened pool.");

    // The resolved kernel selection decides which window route runs, re-read at every sub-batch so a hot edit lands
    // from the next boundary: the batched row window, its row-tiled attention variant, and the dense-matrix weight
    // traversal all have kernels today, and a selection this build has no kernel for fails naming it rather than
    // running some other route behind it. The dense-matrix traversal is the row window with its weight passes handed
    // to the blocked matmul; attention stays on the same per-row causal kernel either way. The row-tiled variant
    // carries the FP32 attention kernels, so a quantized pair — whose selection is pinned to gemv at open — never
    // reaches that arm.
    private void RunPrefillWindow<TSeq, TR>(IReadOnlyList<int> tokens, int offset, int rows, int basePosition, TSeq kv, bool emitLogits, ActivationFamily act, TR attention)
        where TSeq : class, IKvSequence
        where TR : IKvAttentionRoute<TSeq>
    {
        switch (_prefillKernel)
        {
            case PrefillKernel.GemV:
                PrefillWindow(tokens, offset, rows, basePosition, kv, emitLogits, denseMatmul: false, act, attention);
                break;

            case PrefillKernel.Tiled:
                if (kv is SequenceKv tiledKv)
                {
                    PrefillWindowTiled(tokens, offset, rows, basePosition, tiledKv, emitLogits, act);
                }
                else
                {
                    throw QuantizedPrefillRefusal(_prefillKernel);
                }

                break;

            case PrefillKernel.Gemm:
                PrefillWindow(tokens, offset, rows, basePosition, kv, emitLogits, denseMatmul: true, act, attention);
                break;

            default:
                throw KernelRefusal(_prefillKernel);
        }
    }

    // One dense weight pass over a whole sub-batch, handed the execution backend this window runs on: either route traverses
    // each matrix once for every activation vector of the window and writes the same <c>[batch * rows + row]</c> layout,
    // the batched GEMV walking one weight row per activation group and the blocked matmul dequantizing a weight tile
    // once for all of them. The choice is hoisted to this call so no per-row loop reads it as an operand.
    private void WeightPass(IBackend engine, ReadOnlySpan<float> inputs, int count, WeightRef weight, Span<float> outputs, bool denseMatmul)
    {
        if (denseMatmul)
        {
            engine.GemmBatch(inputs, count, weight, outputs);
        }
        else
        {
            engine.GemvBatch(inputs, count, weight, outputs);
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
    // Dense feed-forward gate/up traversal for one batch: both matrices consume the same normed activation batch, so a backend that
    // offers shared-input pair fusion walks them in one call and keeps those activations hot across both traversals. Fusion is gated
    // per posture by the backend itself (formats it reproduces bit-identically only), the gemm route never fuses, and every backend
    // without the capability — CUDA included — runs exactly the two independent calls this seam replaced.
    private void RunDenseFeedForwardPair(IBackend engine, ReadOnlySpan<float> inputs, int count, WeightRef gate, Span<float> outGate, WeightRef up, Span<float> outUp, bool denseMatmul)
    {
        if (!denseMatmul && engine is IDenseWeightPairFusion fusion &&
            fusion.TryGemvBatchSharedInput(inputs, count, gate, outGate, up, outUp))
        {
            return;
        }

        WeightPass(engine, inputs, count, gate, outGate, denseMatmul);
        WeightPass(engine, inputs, count, up, outUp, denseMatmul);
    }

    // One batched attention Q/K/V projection pass: the three matrices consume the same normed activation batch and share its column count, so a
    // backend that offers shared-input triple fusion walks them in one call and keeps those activations hot across all three traversals — the
    // activation-side win the gate/up pair got, extended to the trio. Fusion is gated per posture by the backend itself (formats it reproduces
    // bit-identically only), the gemm route never fuses, and every backend without the capability — CUDA included — runs exactly the three
    // independent calls this seam replaced. Unlike the pair seam the trio may hold different row counts (the query region is wider than the
    // kv region), and the fused executor walks each matrix's own rows.
    private void RunAttentionQKVProjection(IBackend engine, ReadOnlySpan<float> inputs, int count, WeightRef query, Span<float> outQ, WeightRef key, Span<float> outK, WeightRef value, Span<float> outV, bool denseMatmul)
    {
        if (!denseMatmul && engine is IDenseWeightTripleFusion fusion &&
            fusion.TryGemvBatchSharedInput(inputs, count, query, outQ, key, outK, value, outV))
        {
            return;
        }

        WeightPass(engine, inputs, count, query, outQ, denseMatmul);
        WeightPass(engine, inputs, count, key, outK, denseMatmul);
        WeightPass(engine, inputs, count, value, outV, denseMatmul);
    }

    private void PrefillWindow<TSeq, TR>(IReadOnlyList<int> tokens, int offset, int rows, int basePosition, TSeq kv, bool emitLogits, bool denseMatmul, ActivationFamily act, TR attention)
        where TSeq : class, IKvSequence
        where TR : IKvAttentionRoute<TSeq>
    {
        int hidden = Plan.HiddenSize;
        int qkvRegion = Plan.HeadCount * Plan.HeadDim;
        int kvRegion = Plan.KvHeadCount * Plan.HeadDim;
        int inter = Plan.IntermediateSize;
        float attnScale = _attnScale;

        // The window's mutable streams ride the activation family its route writes through — this model's default family for a shared posture, the dedicated lane's own when placement declares one. Read-only weights, norms, rotation tables, and KV bytes stay outside both families.
        float[] batchNormed = act.BatchNormed;
        float[] batchQ = act.BatchQ;
        float[] batchK = act.BatchK;
        float[] batchV = act.BatchV;
        float[] batchAttnOut = act.BatchAttnOut;
        float[] batchProj = act.BatchProj;
        float[] batchGateUp = act.BatchGateUp;
        float[] batchUp = act.BatchUp;

        // The rows' residual streams start on the family's own storage and move to a span's staging arena wherever that backend takes over below.
        Span<float> curStream = act.BatchCur.AsSpan(0, rows * hidden);

        long seedStart = BeginDenseProfile();

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

            _prefillSpans[0].Backend.DequantRow(embRef, tokenId, curStream.Slice(r * hidden, hidden));
        }

        ApplyInputEmbeddingScale(curStream);

        AccumulateDenseProfile(ref _denseSeedTicks, seedStart);

        long appendStart = BeginDenseProfile();

        // Reserve every row's slot in the sequence before any layer writes, mirroring the append each token made.
        for (int r = 0; r < rows; r++)
        {
            kv.Append();
        }

        AccumulateDenseProfile(ref _denseCacheAppendTicks, appendStart);

        // The resolved execution spans carry this sub-batch through the model in ordered ranges: every weight pass below runs on the backend its own span was resolved to, and a crossing hands only these residual rows to that backend's arena — each layer recomputes its norms, projections, attention scratch, and expert buffers inside whichever span runs it.
        for (int s = 0; s < _prefillSpans.Length; s++)
        {
            ExecutionSpan span = _prefillSpans[s];
            IBackend engine = span.Backend;

            if (s > 0)
            {
                // Every resolved window span beyond the first owns an arena assigned at open.
                ResidualCarrier staging = _stagedCarriers[s]!;
                curStream.CopyTo(staging.View(rows));
                curStream = staging.View(rows);
            }

            for (int l = span.LayerStart; l < span.LayerEnd; l++)
            {
                int layerWindow = LayerAttentionWindow(l);

                long blockStart = BeginDenseProfile();       // this dense block's whole attention + feed-forward span
                long attentionStart = BeginDenseProfile();   // the attention stage: its norms through the sublayer fold and that block's feed-forward norm

                // Pre-attention norm per row: identical op to the single-token forward.
                for (int r = 0; r < rows; r++)
                {
                    ApplyNorm(curStream.Slice(r * hidden, hidden), _attnNormWeight[l], _attnNormBias[l], batchNormed.AsSpan(r * hidden, hidden));
                }

                long projectionStart = BeginDenseProfile();   // the attention side's dense weight traversals: Q, K, V for every row at once

                // One weight pass per matrix for the whole sub-batch; each row's result matches its single-vector GEMV on
                // either route this selection can take.
                RunAttentionQKVProjection(engine, batchNormed.AsSpan(0, rows * hidden), rows, AttentionQueryWeight(l), batchQ.AsSpan(0, rows * qkvRegion), AttentionKeyWeight(l), batchK.AsSpan(0, rows * kvRegion), AttentionValueWeight(l), batchV.AsSpan(0, rows * kvRegion), denseMatmul);

                AccumulateDenseSubstage(ref _denseAttentionProjectionTicks, ref _denseAttentionWeightTicks, projectionStart);

                // Bias, QK-norm, RoPE at the row's absolute position, and the K/V writes for every row complete
                // before any row attends, so each row's causal read sees exactly the prefix it would have alone.
                for (int r = 0; r < rows; r++)
                {
                    int position = basePosition + offset + r;

                    AddBias(batchQ.AsSpan(r * qkvRegion, qkvRegion), _biasQ[l]);
                    AddBias(batchK.AsSpan(r * kvRegion, kvRegion), _biasK[l]);
                    AddBias(batchV.AsSpan(r * kvRegion, kvRegion), _biasV[l]);

                    ApplyQkNorm(batchQ.AsSpan(r * qkvRegion, qkvRegion), _attnQNorm[l]);
                    ApplyQkNorm(batchK.AsSpan(r * kvRegion, kvRegion), _attnKNorm[l]);

                    ApplyRopeHeads(batchQ.AsSpan(r * qkvRegion, qkvRegion), Plan.HeadCount, position, windowedLayer: layerWindow > 0);
                    ApplyRopeHeads(batchK.AsSpan(r * kvRegion, kvRegion), Plan.KvHeadCount, position, windowedLayer: layerWindow > 0);

                    long writeStart = BeginDenseProfile();   // this row's K/V slot writes into the sequence's cache
                    kv.WriteK(position, l, batchK.AsSpan(r * kvRegion, kvRegion));
                    kv.WriteV(position, l, batchV.AsSpan(r * kvRegion, kvRegion));
                    AccumulateDenseProfile(ref _denseKvWriteTicks, writeStart);
                }

                long kernelStart = BeginDenseProfile();       // the causal read spans: one walk per row here, or the composed device step below

                if (_devicePrefill is { } deviceStep)
                {
                    // The composed posture hands one layer of the whole sub-batch to the opened step in a single call: the rows land in the KV storage that walk reads, and every score row, softmax state, and K/V read it resolves stays on the far side of the seam — exactly what the per-row host kernel below does position by position over this sequence's blocks. The host RoPE pass and the host KV writes above stay in force: they keep this engine's own decode path reading bytes that never moved device-side.
                    deviceStep.RunAttentionWindow(
                        l, basePosition + offset, batchQ.AsSpan(0, rows * qkvRegion), batchK.AsSpan(0, rows * kvRegion), batchV.AsSpan(0, rows * kvRegion),
                        attnScale, Plan.AttentionSoftcap, layerWindow, batchAttnOut.AsSpan(0, rows * qkvRegion));
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        attention.Apply(engine, batchQ.AsSpan(r * qkvRegion, qkvRegion), Plan.HeadCount, Plan.KvHeadCount, Plan.HeadDim, kv, l, basePosition + offset + r, attnScale, Plan.AttentionSoftcap, layerWindow, batchAttnOut.AsSpan(r * qkvRegion, qkvRegion));
                    }
                }

                AccumulateDenseProfile(ref _denseKernelTicks, kernelStart);

                ReadOnlySpan<byte> bytes = _weights.Resolve(_wo[l]);

                long outputPassStart = BeginDenseProfile();   // the attention side's remaining dense traversal: the output projection for every row
                WeightPass(engine, batchAttnOut.AsSpan(0, rows * qkvRegion), rows, new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), batchProj.AsSpan(0, rows * hidden), denseMatmul);
                AccumulateDenseSubstage(ref _denseAttentionOutputTicks, ref _denseAttentionWeightTicks, outputPassStart);

                for (int r = 0; r < rows; r++)
                {
                    AddBias(batchProj.AsSpan(r * hidden, hidden), _biasO[l]);

                    // Post-sublayer placement: this row's attention output joins its stream only after the post-attention norm.
                    ApplyPostSublayerNorm(batchProj.AsSpan(r * hidden, hidden), _postAttnNorm[l]);

                    // ffn_inp add on the row's own stream, same order as the single-token forward.
                    Span<float> cur = curStream.Slice(r * hidden, hidden);
                    ReadOnlySpan<float> proj = batchProj.AsSpan(r * hidden, hidden);
                    for (int i = 0; i < hidden; i++)
                    {
                        cur[i] += proj[i];
                    }

                    ApplyNorm(cur, _ffnNormWeight[l], _ffnNormBias[l], batchNormed.AsSpan(r * hidden, hidden));
                }

                AccumulateDenseProfile(ref _denseAttentionTicks, attentionStart);

                long feedForwardStart = BeginDenseProfile();   // the feed-forward stage: its traversals plus this block's activation fold and output norm/residual tail

                if (_sparseFfn is { } routing)
                {
                    long chainStart = BeginDenseProfile();   // a sparse block reports its whole routed/shared chain as this stage's traversal substage
                    ApplySparseFeedForwardBatch(routing, l, rows, act, engine);
                    AccumulateDenseSubstage(ref _denseSparseChainTicks, ref _denseFeedForwardWeightTicks, chainStart);
                }
                else
                {
                    long gateUpStart = BeginDenseProfile();   // the dense feed-forward's gate and up traversals for every row at once

                    RunDenseFeedForwardPair(engine, batchNormed.AsSpan(0, rows * hidden), rows, FeedForwardGateWeight(l), batchGateUp.AsSpan(0, rows * inter), FeedForwardUpWeight(l), batchUp.AsSpan(0, rows * inter), denseMatmul);

                    AccumulateDenseSubstage(ref _denseFeedForwardGateUpTicks, ref _denseFeedForwardWeightTicks, gateUpStart);

                    // act = activation(gate) * up in place on each row's gate buffer.
                    for (int r = 0; r < rows; r++)
                    {
                        ApplyActivation(batchGateUp.AsSpan(r * inter, inter), batchUp.AsSpan(r * inter, inter));
                    }

                    bytes = _weights.Resolve(_ffnDown[l]);

                    long downStart = BeginDenseProfile();   // the dense feed-forward's down traversal for every row at once
                    WeightPass(engine, batchGateUp.AsSpan(0, rows * inter), rows, new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), batchProj.AsSpan(0, rows * hidden), denseMatmul);
                    AccumulateDenseSubstage(ref _denseFeedForwardDownTicks, ref _denseFeedForwardWeightTicks, downStart);
                }

                for (int r = 0; r < rows; r++)
                {
                    // Post-sublayer placement: this row's feed-forward output joins its stream only after the post-feed-forward norm.
                    ApplyPostSublayerNorm(batchProj.AsSpan(r * hidden, hidden), _postFfnNorm[l]);

                    // ffn_out + ffn_inp: the input stream for this row's next layer.
                    Span<float> cur = curStream.Slice(r * hidden, hidden);
                    ReadOnlySpan<float> proj = batchProj.AsSpan(r * hidden, hidden);
                    for (int i = 0; i < hidden; i++)
                    {
                        cur[i] += proj[i];
                    }
                }

                AccumulateDenseProfile(ref _denseFeedForwardTicks, feedForwardStart);
                AccumulateDenseLayerTick(l, blockStart);
            }
        }

        if (emitLogits)
        {
            long logitsStart = BeginDenseProfile();   // the window's output head: final-row norm plus the lm_head traversal with its bias and softcap tail

            // The window's final row is the sequence's next-token distribution: its output norm and lm_head GEMV run alone over this route's tail buffers — the default family's for a shared posture, the lane family's own when dedicated — exactly the tail the single-token forward produced. The head itself runs on the backend that finished the last span.
            ApplyNorm(curStream.Slice((rows - 1) * hidden, hidden), _outputNormWeight, _outputNormBias, act.FinalNormed);

            LogicalTensor lmHead = _output ?? _tokenEmbd; // tied weights reuse the token-embedding rows as logits
            ReadOnlySpan<byte> headBytes = _weights.Resolve(lmHead);
            IBackend finalEngine = _prefillSpans[^1].Backend;
            finalEngine.Gemv(act.FinalNormed.AsSpan(0, hidden), new WeightRef(headBytes, lmHead.Shape, lmHead.Format), act.FinalLogits);

            if (_outputBias is not null)
            {
                AddBias(act.FinalLogits, _outputBias);
            }

            ApplyFinalLogitSoftcap(act.FinalLogits);

            AccumulateDenseProfile(ref _denseLogitsTicks, logitsStart);
        }
    }

    // One sub-batch of a prefill window on the row-tiled route: the same streams, positions, norms, RoPE, K/V writes,
    // batched weight passes, and residual adds the batched-row window runs — attention alone differs. Where the
    // batched-row route walks each row through the causal kernel on its own, this route hands the whole sub-batch to
    // LayerOps.CausalAttentionTiledRows: one walk per layer loads each selected position's K/V once per kv-head group
    // and applies it to every row whose own causal selection covers it, while each row advances its own online-softmax
    // state in the preallocated row-state buffers in exactly the order its per-row kernel would have run it. Only the
    // window's final row reaches the output head, exactly as on the batched-row route.
    private void PrefillWindowTiled(IReadOnlyList<int> tokens, int offset, int rows, int basePosition, SequenceKv kv, bool emitLogits, ActivationFamily act)
    {
        int hidden = Plan.HiddenSize;
        int qkvRegion = Plan.HeadCount * Plan.HeadDim;
        int kvRegion = Plan.KvHeadCount * Plan.HeadDim;
        int inter = Plan.IntermediateSize;
        float attnScale = _attnScale;

        // The window's mutable streams ride the activation family its route writes through — this model's default family for a shared posture, the dedicated lane's own when placement declares one. Read-only weights, norms, rotation tables, and KV bytes stay outside both families.
        float[] batchNormed = act.BatchNormed;
        float[] batchQ = act.BatchQ;
        float[] batchK = act.BatchK;
        float[] batchV = act.BatchV;
        float[] batchAttnOut = act.BatchAttnOut;
        float[] batchProj = act.BatchProj;
        float[] batchGateUp = act.BatchGateUp;
        float[] batchUp = act.BatchUp;

        // The rows' residual streams start on the family's own storage and move to a span's staging arena wherever that backend takes over below.
        Span<float> curStream = act.BatchCur.AsSpan(0, rows * hidden);

        long seedStart = BeginDenseProfile();

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

            _prefillSpans[0].Backend.DequantRow(embRef, tokenId, curStream.Slice(r * hidden, hidden));
        }

        ApplyInputEmbeddingScale(curStream);

        AccumulateDenseProfile(ref _denseSeedTicks, seedStart);

        long appendStart = BeginDenseProfile();

        // Reserve every row's slot in the sequence before any layer writes, mirroring the append each token made.
        for (int r = 0; r < rows; r++)
        {
            kv.Append();
        }

        AccumulateDenseProfile(ref _denseCacheAppendTicks, appendStart);

        Span<float> tiledRowMax = act.TiledRowMax.AsSpan(0, rows * Plan.HeadCount); // every row's per-head online max across the tiled walk
        Span<float> tiledRowSumExp = act.TiledRowSumExp.AsSpan(0, rows * Plan.HeadCount); // every row's per-head weight sum across the walk

        // The resolved execution spans carry this sub-batch through the model in ordered ranges: a crossing hands only these residual rows to the next backend's arena, and each layer recomputes its own norms, projections, tiled row state, and expert buffers inside whichever span runs it.
        for (int s = 0; s < _prefillSpans.Length; s++)
        {
            ExecutionSpan span = _prefillSpans[s];
            IBackend engine = span.Backend;

            if (s > 0)
            {
                // Every resolved window span beyond the first owns an arena assigned at open.
                ResidualCarrier staging = _stagedCarriers[s]!;
                curStream.CopyTo(staging.View(rows));
                curStream = staging.View(rows);
            }

            for (int l = span.LayerStart; l < span.LayerEnd; l++)
            {
                int layerWindow = LayerAttentionWindow(l);

                long blockStart = BeginDenseProfile();       // this dense block's whole attention + feed-forward span
                long attentionStart = BeginDenseProfile();   // the attention stage: its norms through the sublayer fold and that block's feed-forward norm

                // Pre-attention norm per row: identical op to the single-token forward.
                for (int r = 0; r < rows; r++)
                {
                    ApplyNorm(curStream.Slice(r * hidden, hidden), _attnNormWeight[l], _attnNormBias[l], batchNormed.AsSpan(r * hidden, hidden));
                }

                long projectionStart = BeginDenseProfile();   // the attention side's dense weight traversals: Q, K, V for every row at once

                // One weight pass per matrix for the whole sub-batch; each row's result matches its single-vector GEMV.
                RunAttentionQKVProjection(engine, batchNormed.AsSpan(0, rows * hidden), rows, AttentionQueryWeight(l), batchQ.AsSpan(0, rows * qkvRegion), AttentionKeyWeight(l), batchK.AsSpan(0, rows * kvRegion), AttentionValueWeight(l), batchV.AsSpan(0, rows * kvRegion), denseMatmul: false);

                AccumulateDenseSubstage(ref _denseAttentionProjectionTicks, ref _denseAttentionWeightTicks, projectionStart);

                // Bias, QK-norm, RoPE at the row's absolute position, and the K/V writes for every row complete
                // before any row attends, so each row's causal read sees exactly the prefix it would have alone.
                for (int r = 0; r < rows; r++)
                {
                    int position = basePosition + offset + r;

                    AddBias(batchQ.AsSpan(r * qkvRegion, qkvRegion), _biasQ[l]);
                    AddBias(batchK.AsSpan(r * kvRegion, kvRegion), _biasK[l]);
                    AddBias(batchV.AsSpan(r * kvRegion, kvRegion), _biasV[l]);

                    ApplyQkNorm(batchQ.AsSpan(r * qkvRegion, qkvRegion), _attnQNorm[l]);
                    ApplyQkNorm(batchK.AsSpan(r * kvRegion, kvRegion), _attnKNorm[l]);

                    ApplyRopeHeads(batchQ.AsSpan(r * qkvRegion, qkvRegion), Plan.HeadCount, position, windowedLayer: layerWindow > 0);
                    ApplyRopeHeads(batchK.AsSpan(r * kvRegion, kvRegion), Plan.KvHeadCount, position, windowedLayer: layerWindow > 0);

                    long writeStart = BeginDenseProfile();   // this row's K/V slot writes into the sequence's cache
                    kv.WriteK(position, l, batchK.AsSpan(r * kvRegion, kvRegion));
                    kv.WriteV(position, l, batchV.AsSpan(r * kvRegion, kvRegion));
                    AccumulateDenseProfile(ref _denseKvWriteTicks, writeStart);
                }

                long kernelStart = BeginDenseProfile();       // the causal read span: this route's one tiled walk over every row of the sub-batch

                // Row-tiled attention for the whole sub-batch in one walk per layer: each selected position's K/V is read
                // once per kv-head group and applied to every row whose own causal selection covers it — a row whose causal
                // limit or own selection excludes the position never touches that slot — while each row's online-softmax
                // state advances through the row-state buffers in exactly the ascending order its per-row kernel ran it.
                LayerOps.CausalAttentionTiledRows(
                    engine,
                    batchQ.AsSpan(0, rows * qkvRegion),
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
                    batchAttnOut.AsSpan(0, rows * qkvRegion));

                AccumulateDenseProfile(ref _denseKernelTicks, kernelStart);

                ReadOnlySpan<byte> bytes = _weights.Resolve(_wo[l]);

                long outputPassStart = BeginDenseProfile();   // the attention side's remaining dense traversal: the output projection for every row
                engine.GemvBatch(batchAttnOut.AsSpan(0, rows * qkvRegion), rows, new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), batchProj.AsSpan(0, rows * hidden));
                AccumulateDenseSubstage(ref _denseAttentionOutputTicks, ref _denseAttentionWeightTicks, outputPassStart);

                for (int r = 0; r < rows; r++)
                {
                    AddBias(batchProj.AsSpan(r * hidden, hidden), _biasO[l]);

                    // Post-sublayer placement: this row's attention output joins its stream only after the post-attention norm.
                    ApplyPostSublayerNorm(batchProj.AsSpan(r * hidden, hidden), _postAttnNorm[l]);

                    // ffn_inp add on the row's own stream, same order as the single-token forward.
                    Span<float> cur = curStream.Slice(r * hidden, hidden);
                    ReadOnlySpan<float> proj = batchProj.AsSpan(r * hidden, hidden);
                    for (int i = 0; i < hidden; i++)
                    {
                        cur[i] += proj[i];
                    }

                    ApplyNorm(cur, _ffnNormWeight[l], _ffnNormBias[l], batchNormed.AsSpan(r * hidden, hidden));
                }

                AccumulateDenseProfile(ref _denseAttentionTicks, attentionStart);

                long feedForwardStart = BeginDenseProfile();   // the feed-forward stage: its traversals plus this block's activation fold and output norm/residual tail

                if (_sparseFfn is { } routing)
                {
                    long chainStart = BeginDenseProfile();   // a sparse block reports its whole routed/shared chain as this stage's traversal substage
                    ApplySparseFeedForwardBatch(routing, l, rows, act, engine);
                    AccumulateDenseSubstage(ref _denseSparseChainTicks, ref _denseFeedForwardWeightTicks, chainStart);
                }
                else
                {
                    long gateUpStart = BeginDenseProfile();   // the dense feed-forward's gate and up traversals for every row at once

                RunDenseFeedForwardPair(engine, batchNormed.AsSpan(0, rows * hidden), rows, FeedForwardGateWeight(l), batchGateUp.AsSpan(0, rows * inter), FeedForwardUpWeight(l), batchUp.AsSpan(0, rows * inter), denseMatmul: false);

                    AccumulateDenseSubstage(ref _denseFeedForwardGateUpTicks, ref _denseFeedForwardWeightTicks, gateUpStart);

                    // act = activation(gate) * up in place on each row's gate buffer.
                    for (int r = 0; r < rows; r++)
                    {
                        ApplyActivation(batchGateUp.AsSpan(r * inter, inter), batchUp.AsSpan(r * inter, inter));
                    }

                    bytes = _weights.Resolve(_ffnDown[l]);

                    long downStart = BeginDenseProfile();   // the dense feed-forward's down traversal for every row at once
                    engine.GemvBatch(batchGateUp.AsSpan(0, rows * inter), rows, new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), batchProj.AsSpan(0, rows * hidden));
                    AccumulateDenseSubstage(ref _denseFeedForwardDownTicks, ref _denseFeedForwardWeightTicks, downStart);
                }

                for (int r = 0; r < rows; r++)
                {
                    // Post-sublayer placement: this row's feed-forward output joins its stream only after the post-feed-forward norm.
                    ApplyPostSublayerNorm(batchProj.AsSpan(r * hidden, hidden), _postFfnNorm[l]);

                    // ffn_out + ffn_inp: the input stream for this row's next layer.
                    Span<float> cur = curStream.Slice(r * hidden, hidden);
                    ReadOnlySpan<float> proj = batchProj.AsSpan(r * hidden, hidden);
                    for (int i = 0; i < hidden; i++)
                    {
                        cur[i] += proj[i];
                    }
                }

                AccumulateDenseProfile(ref _denseFeedForwardTicks, feedForwardStart);
                AccumulateDenseLayerTick(l, blockStart);
            }
        }

        if (emitLogits)
        {
            long logitsStart = BeginDenseProfile();   // the window's output head: final-row norm plus the lm_head traversal with its bias and softcap tail

            // The window's final row is the sequence's next-token distribution: its output norm and lm_head GEMV run alone over this route's tail buffers — the default family's for a shared posture, the lane family's own when dedicated — exactly the tail the single-token forward produced. The head itself runs on the backend that finished the last span.
            ApplyNorm(curStream.Slice((rows - 1) * hidden, hidden), _outputNormWeight, _outputNormBias, act.FinalNormed);

            LogicalTensor lmHead = _output ?? _tokenEmbd; // tied weights reuse the token-embedding rows as logits
            ReadOnlySpan<byte> headBytes = _weights.Resolve(lmHead);
            IBackend finalEngine = _prefillSpans[^1].Backend;
            finalEngine.Gemv(act.FinalNormed.AsSpan(0, hidden), new WeightRef(headBytes, lmHead.Shape, lmHead.Format), act.FinalLogits);

            if (_outputBias is not null)
            {
                AddBias(act.FinalLogits, _outputBias);
            }

            ApplyFinalLogitSoftcap(act.FinalLogits);

            AccumulateDenseProfile(ref _denseLogitsTicks, logitsStart);
        }
    }

    /// <summary>Forwards one token at the next free position and returns its vocab-sized logits.</summary>
    public Span<float> DecodeNext(int tokenId, IKvSequence kv)
    {
        ArgumentNullException.ThrowIfNull(kv);

        if (_qwen4Exp is not null)
        {
            throw Qwen4ExpServingRefusal(nameof(DecodeNext));
        }

        long forwardStart = BeginDenseProfile();   // one single-token decode step as a whole, beside the stage totals its layers accumulated

        if (_q8q8Attention is { } q8q8)
        {
            ForwardToken(tokenId, kv.Length, ServeSequence<SequenceKv<Q8Block, Q8Block>>(kv), q8q8);
        }
        else if (_q8q4Attention is { } q8q4)
        {
            ForwardToken(tokenId, kv.Length, ServeSequence<SequenceKv<Q8Block, Q4Block>>(kv), q8q4);
        }
        else
        {
            ForwardToken(tokenId, kv.Length, ServeSequence<SequenceKv>(kv), _floatAttention!);
        }

        AccumulateDenseProfile(ref _denseTotalTicks, forwardStart);

        return _logits.AsSpan();
    }

    /// <summary>
    /// Logits from the most recent forward: after a prefill these are already the next-token distribution for the
    /// sequence's final token — sampling them avoids re-forwarding (and double-appending) that token.
    /// </summary>
    public Span<float> LastLogits()
    {
        if (_qwen4Exp is not null)
        {
            throw Qwen4ExpServingRefusal(nameof(LastLogits));
        }

        return _logits.AsSpan();
    }

    /// <summary>
    /// The logits left by this engine's most recent prompt window, read through the prefill route's activation family:
    /// identical to <see cref="LastLogits"/> for a shared posture, and lane-owned storage that no decode-side forward ever writes when dedicated.
    /// </summary>
    internal Span<float> PrefillRouteLogits() => _laneAct.FinalLogits.AsSpan();

    /// <summary>The resolved dedicated prefill-lane layout this engine opened with — null for every shared-posture model. Internal diagnostics for hosts and tests.</summary>
    internal PrefillLaneLayout? DedicatedPrefillLane => _prefillsLane;

    /// <summary>Total bytes of mutable activation storage the dedicated lane family holds at open — weights, norms, rotation tables, and KV live outside it, a shared-posture model sizes none of it, and a resolved prefill-buffer ceiling narrows what this engine allocates for the lane beside its own window. Public diagnostics for hosts that report lane state.</summary>
    public long PrefillLaneActivationBytes => _prefillsLane is null ? 0 : _laneAct.ActivationBytes;

    /// <summary>
    /// The activation bytes this engine's prompt route holds for the working set a declared prefill-buffer ceiling bounds: what the dedicated
    /// lane family carries when one opened, and the effective row window's per-row cost beside its fixed tail buffers when the posture shares
    /// one backend. What capacity accounting claims against a chain in this resource's unit; zero is never reported for an engine that runs
    /// prompt windows at all.
    /// </summary>
    public long PrefillWorkingSetBytes => _prefillsLane is null
        ? (long)_effectivePrefillRowWidth * _prefillRowActivationBytes + sizeof(float) * (Plan.HiddenSize + Plan.VocabSize)
        : _laneAct.ActivationBytes;

    /// <summary>The working set stated in the megabyte unit prefill-buffer claims and ceilings count in, rounded up so a claim never under-reports what it holds against a ceiling.</summary>
    public long PrefillWorkingSetMegabytes => (PrefillWorkingSetBytes + PrefillBufferClamp.Megabytes - 1) / PrefillBufferClamp.Megabytes;

    /// <summary>
    /// The residency tier attached to this engine's expert slab path when the opened weight source answers as one: hot-set size, hits/misses,
    /// and saved-bytes tallies read from here for monitor facts. Null is today's prefetch-only posture — a model whose slabs resolve only through
    /// cold views reports nothing residency-side. Public diagnostics; hosts attach what open resolved rather than re-deriving it.
    /// </summary>
    public IResidencyTier? AttachedResidencyTier => _residencyTier;

    /// <summary>
    /// The mutable activation bytes one prompt-window sub-batch row costs on this engine's route: the per-row streams every batched forward
    /// family carries — residual, norm, projection, and expert-group carrier; query/key/value/head regions for both attention shapes; gate/up
    /// halves; online-softmax state; and routing scratch beside a sparse block — counted per row rather than across any sized allocation. A
    /// declared prefill-buffer ceiling measures rows against this figure; the window's single fixed tail pair rides beside it whatever width splits.
    /// </summary>
    private static long PrefillRouteRowActivationBytes(ModelPlan plan)
    {
        int routed = plan.SparseFfn?.RoutedExpertsPerToken ?? 0;

        // Each row-scaled stream of every batched family, in element counts per sub-batch row: four hidden-sized carriers (residual, normed,
        // projection, packed expert group), the q/attn-out pair over head regions, the k/v pair over kv-head regions, gate/up halves, and the
        // tiled walk's per-row online-softmax state. Routing rides one selection per row for sparse blocks — its scratch arrays as a batched step carries them.
        long floats = 4L * plan.HiddenSize
            + 2L * plan.HeadCount * plan.HeadDim
            + 2L * plan.KvHeadCount * plan.HeadDim
            + 2L * plan.IntermediateSize
            + 2L * plan.HeadCount
            + routed;

        long ints = (long)routed + 1; // one member-row selector and its expert ids per row, whatever the block routes

        return floats * sizeof(float) + ints * sizeof(int);
    }

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
    public Span<float> DecodeBatch(IReadOnlyList<int> tokenIds, IReadOnlyList<IKvSequence> kvs)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        ArgumentNullException.ThrowIfNull(kvs);

        if (_qwen4Exp is not null)
        {
            throw Qwen4ExpServingRefusal(nameof(DecodeBatch));
        }

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

            IKvSequence kv = kvs[s] ?? throw new ArgumentException($"kvs entry {s} is null.", nameof(kvs));
            if (kv.Length >= MaxContextTokens)
            {
                throw new InvalidOperationException(
                    $"sequence {s} position {kv.Length} exceeds this model's max context of {MaxContextTokens} tokens.");
            }

            positions[s] = kv.Length;
        }

        long forwardStart = BeginDenseProfile();   // one merged batched step as a whole, beside the stage totals its layers accumulated

        Span<float> logits = _q8q8Attention is { } q8q8
            ? DecodeBatchBody<SequenceKv<Q8Block, Q8Block>, QuantizedAttentionRoute<Q8Block, Q8Block>>(tokenIds, kvs, positions, q8q8)
            : _q8q4Attention is { } q8q4
                ? DecodeBatchBody<SequenceKv<Q8Block, Q4Block>, QuantizedAttentionRoute<Q8Block, Q4Block>>(tokenIds, kvs, positions, q8q4)
                : DecodeBatchBody<SequenceKv, FloatAttentionRoute>(tokenIds, kvs, positions, _floatAttention!);

        AccumulateDenseProfile(ref _denseTotalTicks, forwardStart);

        return logits;
    }

    // The batched step over one closed (sequence, attention-route) pair: each table is cast to the pair's concrete
    // sequence once here — the same once-per-forward addressing the pair's route is closed at — and every per-sequence
    // op below addresses it directly, with no format branch anywhere in the loop.
    private Span<float> DecodeBatchBody<TSeq, TR>(IReadOnlyList<int> tokenIds, IReadOnlyList<IKvSequence> kvs, int[] positions, TR attention)
        where TSeq : class, IKvSequence
        where TR : IKvAttentionRoute<TSeq>
    {
        int n = tokenIds.Count;

        TSeq[] sequences = new TSeq[n];
        for (int s = 0; s < n; s++)
        {
            sequences[s] = ServeSequence<TSeq>(kvs[s]);
        }

        long seedStart = BeginDenseProfile();

        // Embedding lookup: each token's dequantized token_embd row seeds its sequence's input stream.
        ReadOnlySpan<byte> embBytes = _weights.Resolve(_tokenEmbd);
        WeightRef embRef = new(embBytes, _tokenEmbd.Shape, _tokenEmbd.Format);
        for (int s = 0; s < n; s++)
        {
            _backend.DequantRow(embRef, tokenIds[s], _batchCur.AsSpan(s * Plan.HiddenSize, Plan.HiddenSize));
        }

        ApplyInputEmbeddingScale(_batchCur.AsSpan(0, n * Plan.HiddenSize));

        AccumulateDenseProfile(ref _denseSeedTicks, seedStart);

        long appendStart = BeginDenseProfile();

        // Reserve this step's slot in every sequence before any layer writes, mirroring ForwardToken's append.
        for (int s = 0; s < n; s++)
        {
            sequences[s].Append();
        }

        AccumulateDenseProfile(ref _denseCacheAppendTicks, appendStart);

        int hidden = Plan.HiddenSize;
        int qkvRegion = Plan.HeadCount * Plan.HeadDim;
        int kvRegion = Plan.KvHeadCount * Plan.HeadDim;
        int inter = Plan.IntermediateSize;
        float attnScale = _attnScale;

        // The sequences' residual streams start on this engine's batched storage and move to a span's staging arena wherever that backend takes over below.
        Span<float> curStream = _batchCur.AsSpan(0, n * hidden);

        // The resolved execution spans carry the merged step through the model in ordered ranges: each weight pass runs on the backend its own span was resolved to, and a crossing hands only these residual rows onward — every norm, projection, attention scratch, and expert buffer is recomputed inside whichever span runs that layer.
        for (int spanIndex = 0; spanIndex < _mainSpans.Length; spanIndex++)
        {
            ExecutionSpan span = _mainSpans[spanIndex];
            IBackend engine = span.Backend;

            if (spanIndex > 0)
            {
                // Every resolved span beyond the first owns an arena assigned at open.
                ResidualCarrier staging = _stagedCarriers[spanIndex]!;
                curStream.CopyTo(staging.View(n));
                curStream = staging.View(n);
            }

            for (int l = span.LayerStart; l < span.LayerEnd; l++)
            {
                int layerWindow = LayerAttentionWindow(l);

                long blockStart = BeginDenseProfile();       // this dense block's whole attention + feed-forward span
                long attentionStart = BeginDenseProfile();   // the attention stage: its norms through the sublayer fold and that block's feed-forward norm

                // Pre-attention norm per sequence: identical op to the single-token forward.
                for (int s = 0; s < n; s++)
                {
                    ApplyNorm(curStream.Slice(s * hidden, hidden), _attnNormWeight[l], _attnNormBias[l], _batchNormed.AsSpan(s * hidden, hidden));
                }

                long projectionStart = BeginDenseProfile();   // the attention side's dense weight traversals: Q, K, V for every sequence at once

                // One weight pass per matrix for the whole batch; each sequence's row result matches its single-vector GEMV.
                RunAttentionQKVProjection(engine, _batchNormed.AsSpan(0, n * hidden), n, AttentionQueryWeight(l), _batchQ.AsSpan(0, n * qkvRegion), AttentionKeyWeight(l), _batchK.AsSpan(0, n * kvRegion), AttentionValueWeight(l), _batchV.AsSpan(0, n * kvRegion), denseMatmul: false);

                AccumulateDenseSubstage(ref _denseAttentionProjectionTicks, ref _denseAttentionWeightTicks, projectionStart);

                for (int s = 0; s < n; s++)
                {
                    AddBias(_batchQ.AsSpan(s * qkvRegion, qkvRegion), _biasQ[l]);
                    AddBias(_batchK.AsSpan(s * kvRegion, kvRegion), _biasK[l]);
                    AddBias(_batchV.AsSpan(s * kvRegion, kvRegion), _biasV[l]);

                    ApplyQkNorm(_batchQ.AsSpan(s * qkvRegion, qkvRegion), _attnQNorm[l]);
                    ApplyQkNorm(_batchK.AsSpan(s * kvRegion, kvRegion), _attnKNorm[l]);

                    ApplyRopeHeads(_batchQ.AsSpan(s * qkvRegion, qkvRegion), Plan.HeadCount, positions[s], windowedLayer: layerWindow > 0);
                    ApplyRopeHeads(_batchK.AsSpan(s * kvRegion, kvRegion), Plan.KvHeadCount, positions[s], windowedLayer: layerWindow > 0);

                    long writeStart = BeginDenseProfile();   // this sequence's K/V slot writes into its own cache
                    sequences[s].WriteK(positions[s], l, _batchK.AsSpan(s * kvRegion, kvRegion));
                    sequences[s].WriteV(positions[s], l, _batchV.AsSpan(s * kvRegion, kvRegion));
                    AccumulateDenseProfile(ref _denseKvWriteTicks, writeStart);

                    long kernelStart = BeginDenseProfile();   // this sequence's causal read over its own history
                    attention.Apply(engine, _batchQ.AsSpan(s * qkvRegion, qkvRegion), Plan.HeadCount, Plan.KvHeadCount, Plan.HeadDim, sequences[s], l, positions[s], attnScale, Plan.AttentionSoftcap, layerWindow, _batchAttnOut.AsSpan(s * qkvRegion, qkvRegion));
                    AccumulateDenseProfile(ref _denseKernelTicks, kernelStart);
                }

                ReadOnlySpan<byte> bytes = _weights.Resolve(_wo[l]);

                long outputPassStart = BeginDenseProfile();   // the attention side's remaining dense traversal: the output projection for every sequence
                engine.GemvBatch(_batchAttnOut.AsSpan(0, n * qkvRegion), n, new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), _batchProj.AsSpan(0, n * hidden));
                AccumulateDenseSubstage(ref _denseAttentionOutputTicks, ref _denseAttentionWeightTicks, outputPassStart);

                for (int s = 0; s < n; s++)
                {
                    AddBias(_batchProj.AsSpan(s * hidden, hidden), _biasO[l]);

                    // Post-sublayer placement: this sequence's attention output joins its stream only after the post-attention norm.
                    ApplyPostSublayerNorm(_batchProj.AsSpan(s * hidden, hidden), _postAttnNorm[l]);

                    // ffn_inp add on the sequence's own stream, same order as the single-token forward.
                    Span<float> cur = curStream.Slice(s * hidden, hidden);
                    ReadOnlySpan<float> proj = _batchProj.AsSpan(s * hidden, hidden);
                    for (int i = 0; i < hidden; i++)
                    {
                        cur[i] += proj[i];
                    }

                    ApplyNorm(cur, _ffnNormWeight[l], _ffnNormBias[l], _batchNormed.AsSpan(s * hidden, hidden));
                }

                AccumulateDenseProfile(ref _denseAttentionTicks, attentionStart);

                long feedForwardStart = BeginDenseProfile();   // the feed-forward stage: its traversals plus this block's activation fold and output norm/residual tail

                if (_sparseFfn is { } routing)
                {
                    long chainStart = BeginDenseProfile();   // a sparse block reports its whole routed/shared chain as this stage's traversal substage
                    ApplySparseFeedForwardBatch(routing, l, n, _decodeAct, engine);
                    AccumulateDenseSubstage(ref _denseSparseChainTicks, ref _denseFeedForwardWeightTicks, chainStart);
                }
                else
                {
                    long gateUpStart = BeginDenseProfile();   // the dense feed-forward's gate and up traversals for every sequence at once

                RunDenseFeedForwardPair(engine, _batchNormed.AsSpan(0, n * hidden), n, FeedForwardGateWeight(l), _batchGateUp.AsSpan(0, n * inter), FeedForwardUpWeight(l), _batchUp.AsSpan(0, n * inter), denseMatmul: false);

                    AccumulateDenseSubstage(ref _denseFeedForwardGateUpTicks, ref _denseFeedForwardWeightTicks, gateUpStart);

                    // act = activation(gate) * up in place on each sequence's gate buffer.
                    for (int s = 0; s < n; s++)
                    {
                        ApplyActivation(_batchGateUp.AsSpan(s * inter, inter), _batchUp.AsSpan(s * inter, inter));
                    }

                    bytes = _weights.Resolve(_ffnDown[l]);

                    long downStart = BeginDenseProfile();   // the dense feed-forward's down traversal for every sequence at once
                    engine.GemvBatch(_batchGateUp.AsSpan(0, n * inter), n, new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), _batchProj.AsSpan(0, n * hidden));
                    AccumulateDenseSubstage(ref _denseFeedForwardDownTicks, ref _denseFeedForwardWeightTicks, downStart);
                }

                for (int s = 0; s < n; s++)
                {
                    // Post-sublayer placement: this sequence's feed-forward output joins its stream only after the post-feed-forward norm.
                    ApplyPostSublayerNorm(_batchProj.AsSpan(s * hidden, hidden), _postFfnNorm[l]);

                    // ffn_out + ffn_inp: the input stream for this sequence's next layer.
                    Span<float> cur = curStream.Slice(s * hidden, hidden);
                    ReadOnlySpan<float> proj = _batchProj.AsSpan(s * hidden, hidden);
                    for (int i = 0; i < hidden; i++)
                    {
                        cur[i] += proj[i];
                    }
                }

                AccumulateDenseProfile(ref _denseFeedForwardTicks, feedForwardStart);
                AccumulateDenseLayerTick(l, blockStart);
            }
        }

        long logitsStart = BeginDenseProfile();   // this step's output head: every sequence's norm plus the batched lm_head traversal with its bias and softcap tail

        for (int s = 0; s < n; s++)
        {
            ApplyNorm(curStream.Slice(s * hidden, hidden), _outputNormWeight, _outputNormBias, _batchNormed.AsSpan(s * hidden, hidden));
        }

        LogicalTensor lmHead = _output ?? _tokenEmbd; // tied weights reuse the token-embedding rows as logits
        ReadOnlySpan<byte> headBytes = _weights.Resolve(lmHead);
        IBackend finalEngine = _mainSpans[^1].Backend;
        finalEngine.GemvBatch(_batchNormed.AsSpan(0, n * hidden), n, new WeightRef(headBytes, lmHead.Shape, lmHead.Format), _batchLogits.AsSpan(0, n * Plan.VocabSize));

        if (_outputBias is not null)
        {
            for (int s = 0; s < n; s++)
            {
                AddBias(_batchLogits.AsSpan(s * Plan.VocabSize, Plan.VocabSize), _outputBias);
            }
        }

        ApplyFinalLogitSoftcap(_batchLogits.AsSpan(0, n * Plan.VocabSize));

        AccumulateDenseProfile(ref _denseLogitsTicks, logitsStart);

        return _batchLogits.AsSpan(0, n * Plan.VocabSize);
    }

    /// <summary>Token positions one encoder pass advances through the dense matrices per batched weight traversal.</summary>
    public const int EncodeChunkTokens = 8;

    /// <summary>Largest token count a single <see cref="Encode"/> call may forward.</summary>
    public const int MaxEncodeTokens = 256;

    /// <summary>Row ceiling of the engine's merged encoder forwards: the largest number of sequences one <see cref="EncodeBatch"/> call composes, mirroring what one decode batch advances.</summary>
    public const int MaxEncodeSequences = MaxBatchSequences;

    /// <summary>Largest flattened token row count one <see cref="EncodeBatch"/> call rents its position-major streams for: four whole encoder sequences' worth of rows. A wider set of inputs composes as several calls, in order.</summary>
    public const int MaxEncodeBatchRows = 4 * MaxEncodeTokens;

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

        // The encoder pass rents chunked streams beyond this engine's batched row capacity, and a crossing stages rows only through storage sized for the batched forwards. A split route refuses here naming what it does not carry rather than forwarding part of an encoder sequence whole-model on one backend behind the step.
        EnsureEncoderForwardPosture();

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

        EnsureDenseEncoderBlocks();

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

                    RunAttentionQKVProjection(_backend, chunkNormed.AsSpan(0, m * hidden), m, AttentionQueryWeight(l), chunkQ.AsSpan(0, m * qkvRegion), AttentionKeyWeight(l), chunkK.AsSpan(0, m * kvRegion), AttentionValueWeight(l), chunkV.AsSpan(0, m * kvRegion), denseMatmul: false);

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

    /// <summary>
    /// Merged encoder forward over several sequences in one composition: every dense weight matrix of a layer is traversed
    /// once for the whole batch's rows through <see cref="IBackend.GemvBatch"/>, exactly as <see cref="DecodeBatch"/> merges
    /// decode steps, while each sequence still attends only over its own finished KV span — attention stays per-sequence
    /// because a bidirectional query scores against that sequence's positions and nobody else's. The flattened stream is
    /// position-major across the whole batch: chunk-row windows of up to <see cref="EncodeChunkTokens"/> rows run per layer
    /// for norms, projections, RoPE at each row's own absolute position, and the gated activation, so a window can carry the
    /// tail of one sequence beside the head of the next. Every phase runs the same operations, in the same order, over the
    /// same values as an independent <see cref="Encode"/> call per sequence: batching changes which rows share a weight pass,
    /// not what any row computes. Results are returned per input sequence, each array laid out exactly like that sequence's
    /// single-call result (<c>hidden[position * HiddenSize + i]</c>) and caller-owned, in the order the inputs were given.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// The two lists differ in length, fewer than one or more than <see cref="MaxEncodeSequences"/> sequences were given, a
    /// list entry is null, any sequence holds no tokens or more than <see cref="MaxEncodeTokens"/>, the batch's rows exceed
    /// <see cref="MaxEncodeBatchRows"/>, a token id is outside the vocabulary, or appending a sequence would exceed this
    /// model's max context. A rejected call touches no sequence's KV.
    /// </exception>
    public IReadOnlyList<float[]> EncodeBatch(IReadOnlyList<IReadOnlyList<int>> tokenLists, IReadOnlyList<SequenceKv> kvs)
    {
        ArgumentNullException.ThrowIfNull(tokenLists);
        ArgumentNullException.ThrowIfNull(kvs);

        int sequences = tokenLists.Count;
        if (sequences != kvs.Count)
        {
            throw new ArgumentException($"tokenLists has {tokenLists.Count} entries but kvs has {kvs.Count}: one token list per sequence is required.", nameof(kvs));
        }

        if (sequences == 0 || sequences > MaxEncodeSequences)
        {
            throw new ArgumentException($"an encoder batch must hold between 1 and {MaxEncodeSequences} sequences, got {sequences}.", nameof(tokenLists));
        }

        EnsureEncoderForwardPosture();

        int[] lengths = new int[sequences];
        long totalRows = 0;
        for (int s = 0; s < sequences; s++)
        {
            IReadOnlyList<int> tokens = tokenLists[s] ?? throw new ArgumentException($"tokenLists entry {s} is null.", nameof(tokenLists));
            SequenceKv kv = kvs[s] ?? throw new ArgumentException($"kvs entry {s} is null.", nameof(kvs));

            int n = tokens.Count;
            if (n == 0 || n > MaxEncodeTokens)
            {
                throw new ArgumentException(
                    $"sequence {s} holds {n} encoder tokens: an encoder forward must hold between 1 and {MaxEncodeTokens} tokens.",
                    nameof(tokenLists));
            }

            if (kv.Length + n > MaxContextTokens)
            {
                throw new ArgumentException(
                    $"{n} encoder tokens starting at position {kv.Length} exceed this model's max context of {MaxContextTokens}.",
                    nameof(kvs));
            }

            lengths[s] = n;
            totalRows += n;
        }

        if (totalRows > MaxEncodeBatchRows)
        {
            throw new ArgumentException(
                $"a merged encoder batch carries between 1 and {MaxEncodeBatchRows} token rows across its sequences, got {totalRows}: compose these inputs as several calls.",
                nameof(tokenLists));
        }

        EnsureDenseEncoderBlocks();

        return EncodeBatchBody(tokenLists, kvs, lengths);
    }

    // The merged encoder composition: flattened position-major streams over the whole batch plus chunk-local scratch for each
    // window's batched GEMVs. Row ownership and absolute positions resolve once here — every window below addresses rows through
    // that map instead of re-deriving sequence boundaries, and attention groups queries by their own sequence.
    private IReadOnlyList<float[]> EncodeBatchBody(IReadOnlyList<IReadOnlyList<int>> tokenLists, IReadOnlyList<SequenceKv> kvs, int[] lengths)
    {
        int sequences = tokenLists.Count;

        int hidden = Plan.HiddenSize;
        int qkvRegion = Plan.HeadCount * Plan.HeadDim;
        int kvRegion = Plan.KvHeadCount * Plan.HeadDim;
        int inter = Plan.IntermediateSize;
        float attnScale = _attnScale;

        int totalRows = 0;
        int[] rowStarts = new int[sequences];
        for (int s = 0; s < sequences; s++)
        {
            rowStarts[s] = totalRows;
            totalRows += lengths[s];
        }

        // Which sequence each flattened row belongs to, and the absolute position its K/V lands at: RoPE, KV addressing, and
        // attention grouping all read this map, so no window below re-derives a sequence boundary.
        int[] rowSequence = new int[totalRows];
        int[] rowPosition = new int[totalRows];
        int[] basePositions = new int[sequences];

        ArrayPool<float> poolBuffers = ArrayPool<float>.Shared;

        float[] cur = poolBuffers.Rent(totalRows * hidden);          // residual carrier per row; becomes the final stream
        float[] normed = poolBuffers.Rent(totalRows * hidden);       // rmsnorm result feeding the QKV / FFN GEMVs
        float[] qAll = poolBuffers.Rent(totalRows * qkvRegion);      // every position's rotated queries, kept for attention
        float[] attnOut = poolBuffers.Rent(totalRows * qkvRegion);   // concatenated heads per position before wo
        float[] chunkNormed = poolBuffers.Rent(EncodeChunkTokens * hidden);
        float[] chunkQ = poolBuffers.Rent(EncodeChunkTokens * qkvRegion);
        float[] chunkK = poolBuffers.Rent(EncodeChunkTokens * kvRegion);
        float[] chunkV = poolBuffers.Rent(EncodeChunkTokens * kvRegion);
        float[] chunkProj = poolBuffers.Rent(EncodeChunkTokens * hidden);
        float[] chunkGateUp = poolBuffers.Rent(EncodeChunkTokens * inter);
        float[] chunkUp = poolBuffers.Rent(EncodeChunkTokens * inter);

        try
        {
            // Embedding lookup across the flattened stream: each token's dequantized row seeds that position's input stream.
            ReadOnlySpan<byte> embBytes = _weights.Resolve(_tokenEmbd);
            WeightRef embRef = new(embBytes, _tokenEmbd.Shape, _tokenEmbd.Format);
            for (int s = 0; s < sequences; s++)
            {
                IReadOnlyList<int> tokens = tokenLists[s];
                for (int i = 0; i < lengths[s]; i++)
                {
                    if ((uint)tokens[i] >= (uint)Plan.VocabSize)
                    {
                        throw new ArgumentException($"token id {tokens[i]} at index {i} of sequence {s} is outside the vocabulary of {Plan.VocabSize}.", nameof(tokenLists));
                    }

                    _backend.DequantRow(embRef, tokens[i], cur.AsSpan((rowStarts[s] + i) * hidden, hidden));
                }
            }

            ApplyInputEmbeddingScale(cur.AsSpan(0, totalRows * hidden));

            // Reserve every sequence's positions up front so later layer passes address them by absolute position.
            for (int s = 0; s < sequences; s++)
            {
                int basePosition = kvs[s].Length;
                basePositions[s] = basePosition;

                for (int i = 0; i < lengths[s]; i++)
                {
                    rowSequence[rowStarts[s] + i] = s;
                    rowPosition[rowStarts[s] + i] = basePosition + i;
                    kvs[s].Append();
                }
            }

            int windowCount = (totalRows + EncodeChunkTokens - 1) / EncodeChunkTokens;

            for (int l = 0; l < Plan.LayerCount; l++)
            {
                int layerWindow = LayerAttentionWindow(l);

                // Phase 1: pre-attention norm, QKV GEMVs (+bias), RoPE at each row's absolute position, and the K/V writes —
                // one window of rows per weight pass across the whole batch. Queries stay in qAll so no position's attention
                // runs before every position's K/V exists.
                for (int w = 0; w < windowCount; w++)
                {
                    int firstRow = w * EncodeChunkTokens;
                    int m = Math.Min(EncodeChunkTokens, totalRows - firstRow);

                    for (int i = 0; i < m; i++)
                    {
                        ApplyNorm(cur.AsSpan((firstRow + i) * hidden, hidden), _attnNormWeight[l], _attnNormBias[l], chunkNormed.AsSpan(i * hidden, hidden));
                    }

                    RunAttentionQKVProjection(_backend, chunkNormed.AsSpan(0, m * hidden), m, AttentionQueryWeight(l), chunkQ.AsSpan(0, m * qkvRegion), AttentionKeyWeight(l), chunkK.AsSpan(0, m * kvRegion), AttentionValueWeight(l), chunkV.AsSpan(0, m * kvRegion), denseMatmul: false);

                    for (int i = 0; i < m; i++)
                    {
                        int row = firstRow + i;
                        SequenceKv owner = kvs[rowSequence[row]];
                        int position = rowPosition[row];

                        AddBias(chunkQ.AsSpan(i * qkvRegion, qkvRegion), _biasQ[l]);
                        AddBias(chunkK.AsSpan(i * kvRegion, kvRegion), _biasK[l]);
                        AddBias(chunkV.AsSpan(i * kvRegion, kvRegion), _biasV[l]);

                        ApplyQkNorm(chunkQ.AsSpan(i * qkvRegion, qkvRegion), _attnQNorm[l]);
                        ApplyQkNorm(chunkK.AsSpan(i * kvRegion, kvRegion), _attnKNorm[l]);

                        ApplyRopeHeads(chunkQ.AsSpan(i * qkvRegion, qkvRegion), Plan.HeadCount, position, windowedLayer: layerWindow > 0);
                        ApplyRopeHeads(chunkK.AsSpan(i * kvRegion, kvRegion), Plan.KvHeadCount, position, windowedLayer: layerWindow > 0);

                        owner.WriteK(position, l, chunkK.AsSpan(i * kvRegion, kvRegion));
                        owner.WriteV(position, l, chunkV.AsSpan(i * kvRegion, kvRegion));

                        // Sequence-relative slot in the batch-wide query stream: attention reads queries where they live.
                        chunkQ.AsSpan(i * qkvRegion, qkvRegion).CopyTo(qAll.AsSpan(row * qkvRegion, qkvRegion));
                    }
                }

                // Phase 2: attention per sequence — each window of that sequence's own queries scores against its finished KV,
                // over every position on a layer whose resolved window is zero and over [max(0, q - window + 1), q] on one that
                // resolves a positive window. No query ever reads another sequence's K/V.
                for (int s = 0; s < sequences; s++)
                {
                    int start = rowStarts[s];
                    int length = lengths[s];

                    for (int c = 0, chunkCount = (length + EncodeChunkTokens - 1) / EncodeChunkTokens; c < chunkCount; c++)
                    {
                        int m = Math.Min(EncodeChunkTokens, length - c * EncodeChunkTokens);

                        LayerOps.BidirectionalAttention(
                            _backend,
                            qAll.AsSpan((start + c * EncodeChunkTokens) * qkvRegion, m * qkvRegion),
                            queryCount: m,
                            queryPositionOffset: basePositions[s] + c * EncodeChunkTokens,
                            fromPosition: 0,
                            toPosition: basePositions[s] + length - 1,
                            Plan.HeadCount,
                            Plan.KvHeadCount,
                            Plan.HeadDim,
                            kvs[s],
                            l,
                            attnScale,
                            Plan.AttentionSoftcap,
                            layerWindow,
                            attnOut.AsSpan((start + c * EncodeChunkTokens) * qkvRegion, m * qkvRegion));
                    }
                }

                // Phase 3: output projection (+bias) and the ffn_inp residual add on each row's own stream.
                for (int w = 0; w < windowCount; w++)
                {
                    int firstRow = w * EncodeChunkTokens;
                    int m = Math.Min(EncodeChunkTokens, totalRows - firstRow);

                    ReadOnlySpan<byte> bytes = _weights.Resolve(_wo[l]);
                    _backend.GemvBatch(attnOut.AsSpan(firstRow * qkvRegion, m * qkvRegion), m, new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), chunkProj.AsSpan(0, m * hidden));

                    for (int i = 0; i < m; i++)
                    {
                        AddBias(chunkProj.AsSpan(i * hidden, hidden), _biasO[l]);

                        // Post-sublayer placement: this position's attention output joins its stream only after the post-attention norm.
                        ApplyPostSublayerNorm(chunkProj.AsSpan(i * hidden, hidden), _postAttnNorm[l]);

                        Span<float> posCur = cur.AsSpan((firstRow + i) * hidden, hidden);
                        ReadOnlySpan<float> proj = chunkProj.AsSpan(i * hidden, hidden);
                        for (int x = 0; x < hidden; x++)
                        {
                            posCur[x] += proj[x];
                        }

                        ApplyNorm(posCur, _ffnNormWeight[l], _ffnNormBias[l], normed.AsSpan((firstRow + i) * hidden, hidden));
                    }
                }

                // Phase 4: gate/up GEMVs and the gated activation per window.
                for (int w = 0; w < windowCount; w++)
                {
                    int firstRow = w * EncodeChunkTokens;
                    int m = Math.Min(EncodeChunkTokens, totalRows - firstRow);

                    _backend.GemvBatch(normed.AsSpan(firstRow * hidden, m * hidden), m, FeedForwardGateWeight(l), chunkGateUp.AsSpan(0, m * inter));

                    _backend.GemvBatch(normed.AsSpan(firstRow * hidden, m * hidden), m, FeedForwardUpWeight(l), chunkUp.AsSpan(0, m * inter));

                    for (int i = 0; i < m; i++)
                    {
                        ApplyActivation(chunkGateUp.AsSpan(i * inter, inter), chunkUp.AsSpan(i * inter, inter));
                    }

                    // Phase 5: down GEMV and the ffn_out residual add, completing this layer for every row of the window.
                    ReadOnlySpan<byte> bytes = _weights.Resolve(_ffnDown[l]);
                    _backend.GemvBatch(chunkGateUp.AsSpan(0, m * inter), m, new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), chunkProj.AsSpan(0, m * hidden));

                    for (int i = 0; i < m; i++)
                    {
                        // Post-sublayer placement: this position's feed-forward output joins its stream only after the post-feed-forward norm.
                        ApplyPostSublayerNorm(chunkProj.AsSpan(i * hidden, hidden), _postFfnNorm[l]);

                        Span<float> posCur = cur.AsSpan((firstRow + i) * hidden, hidden);
                        ReadOnlySpan<float> proj = chunkProj.AsSpan(i * hidden, hidden);
                        for (int x = 0; x < hidden; x++)
                        {
                            posCur[x] += proj[x]; // ffn_out + ffn_inp: this position's input stream for the next layer
                        }
                    }
                }
            }

            float[][] encoded = new float[sequences][]; // one caller-owned buffer per sequence, exact size — pooling helpers consume each row-wise
            for (int s = 0; s < sequences; s++)
            {
                int length = lengths[s];
                float[] rows = new float[length * hidden];

                for (int i = 0; i < length; i++)
                {
                    ApplyNorm(cur.AsSpan((rowStarts[s] + i) * hidden, hidden), _outputNormWeight, _outputNormBias, rows.AsSpan(i * hidden, hidden));
                }

                encoded[s] = rows;
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

    // The posture every encoder forward needs, stated once for both the single-sequence and merged-batch entry points: a route this engine crosses
    // between backends cannot stage an encoder's streams, and a quantized pair closes only causal kernels. Both refusals name what they lack rather
    // than forwarding part of a sequence on one backend or reading quantized groups as floats.
    private void EnsureEncoderForwardPosture()
    {
        if (_mainSpans.Length > 1)
        {
            throw new ModelException(
                $"architecture '{Plan.Architecture}' resolves a route this engine crosses between backends, and its encoder forward rents streams past the row capacity those crossings stage through — open this model as one span to encode it");
        }

        if (!_kvFormats.IsFloatStorage)
        {
            throw new ModelException(
                $"this model opened kv k='{KvSideFormatNames.Of(_kvFormats.K)}' v='{KvSideFormatNames.Of(_kvFormats.V)}': the encoder pass runs the FP32 bidirectional attention kernels over float KV reads, " +
                "and no quantized encoder route exists in this build — omit kv.quant to encode this model in the FP32 posture");
        }
    }

    // Encoder forwards run dense gate/up/down blocks only: an expert-routed block has no encoder-side routing pass. Shared by both encoder entry points.
    private void EnsureDenseEncoderBlocks()
    {
        if (_sparseFfn is { } encoderSparse)
        {
            throw new ModelException(
                $"architecture '{Plan.Architecture}' declares a sparse-FFN block with {encoderSparse.ExpertCount} experts " +
                $"routing {encoderSparse.RoutedExpertsPerToken} per token: the encoder forward runs dense gate/up/down blocks only");
        }
    }

    /// <summary>The staging arenas this engine allocated at open, in segment order — empty for every single-span route. Assembly-internal diagnostics: what each crossing staged is the resolved position it feeds, its layer range, and the device whose backend allocated that arena.</summary>
    internal IReadOnlyList<ExecutionCrossing> ExecutionStaging => _stagedCrossings;

    public void Dispose()
    {
        // The bounded synthetic qwen4exp seam releases nothing it did not allocate as arrays, but it holds the family's operand
        // reads, so it lands before the weight view it resolved against.
        _qwen4Exp?.Dispose();

        // Model-owned staging arenas return before anything else: each was allocated through a compute backend this host disposes after the model, so every handle lands while its allocator is still live. Repeated disposal never frees one twice.
        foreach (ResidualCarrier? carrier in _stagedCarriers)
        {
            carrier?.Dispose();
        }

        _weights.Dispose();
    }

    private void ForwardToken<TSeq, TR>(int tokenId, int position, TSeq kv, TR attention)
        where TSeq : class, IKvSequence
        where TR : IKvAttentionRoute<TSeq>
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

        long seedStart = BeginDenseProfile();

        // Embedding lookup: one dequantized row of token_embd becomes the layer input stream.
        ReadOnlySpan<byte> embBytes = _weights.Resolve(_tokenEmbd);
        _backend.DequantRow(new WeightRef(embBytes, _tokenEmbd.Shape, _tokenEmbd.Format), tokenId, _cur.AsSpan(0, Plan.HiddenSize));

        ApplyInputEmbeddingScale(_cur.AsSpan(0, Plan.HiddenSize));

        AccumulateDenseProfile(ref _denseSeedTicks, seedStart);

        long appendStart = BeginDenseProfile();

        kv.Append(); // reserves this position's slot in every layer; WriteK/WriteV address it by logical position

        AccumulateDenseProfile(ref _denseCacheAppendTicks, appendStart);

        int hidden = Plan.HiddenSize;
        float attnScale = _attnScale;

        // The token's residual stream starts on this engine's single-token buffer and moves to a span's staging arena wherever that backend takes over below.
        Span<float> curStream = _cur.AsSpan(0, hidden);

        // The resolved execution spans carry the token through the model in ordered ranges: each GEMV runs on the backend its own span was resolved to, and a crossing hands only this stream onward — norms, projections, attention scratch, and expert buffers are recomputed inside whichever span runs that layer.
        for (int spanIndex = 0; spanIndex < _mainSpans.Length; spanIndex++)
        {
            ExecutionSpan span = _mainSpans[spanIndex];
            IBackend engine = span.Backend;

            if (spanIndex > 0)
            {
                // Every resolved span beyond the first owns an arena assigned at open.
                ResidualCarrier staging = _stagedCarriers[spanIndex]!;
                curStream.CopyTo(staging.View(1));
                curStream = staging.View(1);
            }

            for (int l = span.LayerStart; l < span.LayerEnd; l++)
            {
                int layerWindow = LayerAttentionWindow(l);

                long blockStart = BeginDenseProfile();       // this dense block's whole attention + feed-forward span
                long attentionStart = BeginDenseProfile();   // the attention stage: its norms through the sublayer fold and that block's feed-forward norm

                // Pre-attention norm on the stream: normed = rmsnorm(cur) * attn_norm weight.
                ApplyNorm(curStream, _attnNormWeight[l], _attnNormBias[l], _normed);

                long projectionStart = BeginDenseProfile();   // the attention side's dense traversals with their bias adds riding inside this span: Q, K, V

                // Q, K, and V all consume this block's one normed vector. A backend that offers single-vector shared-input
                // fusion walks the three matrices in one dispatch and keeps that vector hot across all three traversals;
                // every element lands bit-identical to the three independent calls the fallback runs.
                if (!(engine is IDenseWeightSharedInputFusion sharedFusion &&
                      sharedFusion.TryGemvSharedInput(_normed.AsSpan(0, hidden), AttentionQueryWeight(l), _q, AttentionKeyWeight(l), _k, AttentionValueWeight(l), _v)))
                {
                    engine.Gemv(_normed.AsSpan(0, hidden), AttentionQueryWeight(l), _q);
                    engine.Gemv(_normed.AsSpan(0, hidden), AttentionKeyWeight(l), _k);
                    engine.Gemv(_normed.AsSpan(0, hidden), AttentionValueWeight(l), _v);
                }

                AddBias(_q, _biasQ[l]);
                AddBias(_k, _biasK[l]);
                AddBias(_v, _biasV[l]);

                AccumulateDenseSubstage(ref _denseAttentionProjectionTicks, ref _denseAttentionWeightTicks, projectionStart);

                // Resolved QK-norm mode: each side's projected stream is normalized after the projection and bias and
                // before RoPE, either per head in its own head-dimension slice or across all its heads under one mean.
                ApplyQkNorm(_q, _attnQNorm[l]);
                ApplyQkNorm(_k, _attnKNorm[l]);

                ApplyRopeHeads(_q, Plan.HeadCount, position, windowedLayer: layerWindow > 0);
                ApplyRopeHeads(_k, Plan.KvHeadCount, position, windowedLayer: layerWindow > 0);

                long writeStart = BeginDenseProfile();   // this token's K/V slot writes into the sequence's cache
                kv.WriteK(position, l, _k);
                kv.WriteV(position, l, _v);
                AccumulateDenseProfile(ref _denseKvWriteTicks, writeStart);

                long kernelStart = BeginDenseProfile();   // this token's causal read over its own history

                // Single-token attention runs the opened pair's route: the pinned FP32 kernel — or the measurement spike
                // route when one is installed on a float engine (EnableKvFormatSpike) — for an FP32 pair, and the closed
                // quantized kernel for a quantized pair, which carries no spike posture.
                attention.ApplySingleToken(engine, _q, Plan.HeadCount, Plan.KvHeadCount, Plan.HeadDim, kv, l, position, attnScale, Plan.AttentionSoftcap, layerWindow, _attnOut);

                AccumulateDenseProfile(ref _denseKernelTicks, kernelStart);

                // Output projection (+bias) into scratch; the residual added before the FFN norm is the pre-attention
                // stream value (ffn_inp = attn_out + inpSA), matching the reference graph. cur held that stream through
                // attention untouched, so adding proj in place turns it into ffn_inp. Under a post-sublayer-output
                // placement the attention output — including any output bias — is normalized with its own vector first.
                ReadOnlySpan<byte> bytes = _weights.Resolve(_wo[l]);

                long outputPassStart = BeginDenseProfile();   // the attention side's remaining dense traversal: the output projection
                engine.Gemv(_attnOut.AsSpan(0, Plan.HeadCount * Plan.HeadDim), new WeightRef(bytes, _wo[l].Shape, _wo[l].Format), _proj);
                AccumulateDenseSubstage(ref _denseAttentionOutputTicks, ref _denseAttentionWeightTicks, outputPassStart);

                AddBias(_proj, _biasO[l]);

                ApplyPostSublayerNorm(_proj, _postAttnNorm[l]);

                for (int i = 0; i < hidden; i++)
                {
                    curStream[i] += _proj[i];
                }

                ApplyNorm(curStream, _ffnNormWeight[l], _ffnNormBias[l], _normed);

                AccumulateDenseProfile(ref _denseAttentionTicks, attentionStart);

                long feedForwardStart = BeginDenseProfile();   // the feed-forward stage: its traversals plus this block's activation fold and output norm/residual tail

                if (_sparseFfn is { } routing)
                {
                    long chainStart = BeginDenseProfile();   // a sparse block reports its whole routed/shared chain as this stage's traversal substage
                    ApplySparseFeedForward(routing, l, engine);
                    AccumulateDenseSubstage(ref _denseSparseChainTicks, ref _denseFeedForwardWeightTicks, chainStart);
                }
                else
                {
                    long gateUpStart = BeginDenseProfile();   // the dense feed-forward's gate and up traversals

                    // Gate and up consume the same normed vector: the same single-vector shared-input fusion seam walks
                    // them in one dispatch when the backend covers the pair, bit-identical to the two independent calls.
                    if (!(engine is IDenseWeightSharedInputFusion ffnFusion &&
                          ffnFusion.TryGemvSharedInput(_normed.AsSpan(0, hidden), FeedForwardGateWeight(l), _gateUp, FeedForwardUpWeight(l), _up)))
                    {
                        engine.Gemv(_normed.AsSpan(0, hidden), FeedForwardGateWeight(l), _gateUp);
                        engine.Gemv(_normed.AsSpan(0, hidden), FeedForwardUpWeight(l), _up);
                    }

                    AccumulateDenseSubstage(ref _denseFeedForwardGateUpTicks, ref _denseFeedForwardWeightTicks, gateUpStart);

                    // act = activation(gate) * up with the plan's resolved activation, applied in place on the gate buffer.
                    ApplyActivation(_gateUp.AsSpan(0, Plan.IntermediateSize), _up.AsSpan(0, Plan.IntermediateSize));

                    bytes = _weights.Resolve(_ffnDown[l]);

                    long downStart = BeginDenseProfile();   // the dense feed-forward's down traversal
                    engine.Gemv(_gateUp.AsSpan(0, Plan.IntermediateSize), new WeightRef(bytes, _ffnDown[l].Shape, _ffnDown[l].Format), _proj);
                    AccumulateDenseSubstage(ref _denseFeedForwardDownTicks, ref _denseFeedForwardWeightTicks, downStart);
                }

                ApplyPostSublayerNorm(_proj, _postFfnNorm[l]);

                for (int i = 0; i < hidden; i++)
                {
                    curStream[i] += _proj[i]; // ffn_out + ffn_inp: the input stream for the next layer
                }

                AccumulateDenseProfile(ref _denseFeedForwardTicks, feedForwardStart);
                AccumulateDenseLayerTick(l, blockStart);
            }
        }

        long logitsStart = BeginDenseProfile();   // this step's output head: its norm plus the lm_head traversal with its bias and softcap tail

        ApplyNorm(curStream, _outputNormWeight, _outputNormBias, _normed);

        LogicalTensor lmHead = _output ?? _tokenEmbd; // tied weights reuse the token-embedding rows as logits
        ReadOnlySpan<byte> headBytes = _weights.Resolve(lmHead);
        IBackend finalEngine = _mainSpans[^1].Backend;
        finalEngine.Gemv(_normed.AsSpan(0, hidden), new WeightRef(headBytes, lmHead.Shape, lmHead.Format), _logits);

        if (_outputBias is not null)
        {
            AddBias(_logits, _outputBias);
        }

        ApplyFinalLogitSoftcap(_logits);

        AccumulateDenseProfile(ref _denseLogitsTicks, logitsStart);
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

    // Router scoring and top-k selection for one normalized activation vector, shared by both forward paths so a
    // batched step reproduces single-sequence routing exactly. The router emits one score per declared expert; every
    // score is softmaxed together because the reference routes over all experts, then RoutedExpertsPerToken picks run
    // as ascending-index scans using strict greater-than comparison — equal scores resolve to the lower expert index
    // deterministically, with no sort or dictionary stability assumption. A picked slot is poisoned so later passes
    // cannot re-pick it; selected slots are insertion-sorted into ascending expert-index execution order alongside
    // their weights. Normalization divides by the selected probability sum when declared, and a nonzero
    // ExpertWeightsScale multiplies every weight after that division. Scratch holds one routing at a time: batched
    // callers copy each selection before selecting the next sequence.
    private void RouteSparseExperts(SparseFfnDescriptor sparse, int layer, ReadOnlySpan<float> input, ActivationFamily act, IBackend engine)
    {
        LogicalTensor router = _moeRouter[layer];
        ReadOnlySpan<byte> bytes = _weights.Resolve(router);
        Span<float> scores = act.MoeScores.AsSpan(0, sparse.ExpertCount);

        // Scratch for one routing at a time lives in the calling route's family: batched callers copy each selection out of it before selecting the next sequence.
        int[] moeIds = act.MoeIds;
        float[] moeWeights = act.MoeWeights;

        // The router emits one score per declared expert before any selection or softmax runs.
        engine.Gemv(input, new WeightRef(bytes, router.Shape, router.Format), scores);

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

            moeIds[pick] = best;
            moeWeights[pick] = bestScore;
            selectedSum += bestScore;
            scores[best] = float.NegativeInfinity; // consumed: later passes cannot re-pick this expert
        }

        for (int i = 1; i < routed; i++)
        {
            int id = moeIds[i];
            float weight = moeWeights[i];
            int j = i - 1;
            while (j >= 0 && moeIds[j] > id)
            {
                moeIds[j + 1] = moeIds[j];
                moeWeights[j + 1] = moeWeights[j];
                j--;
            }

            moeIds[j + 1] = id;
            moeWeights[j + 1] = weight;
        }

        if (sparse.NormalizeExpertWeights && selectedSum > 0f)
        {
            for (int i = 0; i < routed; i++)
            {
                moeWeights[i] /= selectedSum;
            }
        }

        if (sparse.ExpertWeightsScale != 0f)
        {
            for (int i = 0; i < routed; i++)
            {
                moeWeights[i] *= sparse.ExpertWeightsScale;
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
    // vectors are present the score is taken from them against this token's normalized stream and gated by the family
    // the plan resolved: a qwen4exp plan composes plain sigmoid(score) — the reference gate its own bounded forward
    // reads, finite at every score including exactly zero — while every other routed family keeps silu(score) / score,
    // how the pinned qwen2moe graph builds it as `ggml_div(ggml_silu(cur_gate_inp), cur_gate_inp)` over a GEMV of
    // ffn_gate_inp_shexp, sigmoid's value read through its own silu rather than a separate node. The dot runs in
    // ascending column order and both forward paths reach it through this method with the same per-token stream, so
    // batched steps stay bit-identical to single-sequence ones. A qwen2moe score of exactly zero divides 0 by 0 as its
    // reference graph does; a qwen4exp one reads 0.5.
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

        // The gate arm is chosen by the family the plan resolved, never probed from the score: qwen2moe-shaped plans keep
        // their pinned composition unchanged, and a qwen4exp plan reads the plain sigmoid its reference composes.
        return LayerOps.SharedExpertGateWeight(plainSigmoidFamily: Plan.Qwen4Exp is not null, score);
    }

    // Sparse feed-forward for the current token: routing selects, then every selected expert's gate/up/down chain runs
    // over the normalized stream in ascending expert-index order through the gather kernel — each expert's slabs read as
    // contiguous views of its role stack and prefetched before the GEMVs consume them — and every output accumulates into
    // _proj scaled by its routing weight; a declared shared expert contributing once under SharedExpertWeight, which is
    // 1f unless the file ships that block's selector vectors. The gate/up intermediates reuse the dense token buffers; no
    // copy of any expert tensor exists anywhere. The caller's post-feed-forward norm and residual add then run exactly as
    // they do on the dense path.
    private void ApplySparseFeedForward(SparseFfnDescriptor sparse, int layer, IBackend engine)
    {
        RouteSparseExperts(sparse, layer, _normed.AsSpan(0, Plan.HiddenSize), _decodeAct, engine);

        int hidden = Plan.HiddenSize;
        _proj.AsSpan().Clear(); // the routed and shared contributions accumulate from zero below

        ExpertSlabRole gate = new(_moeExpertGate[layer], _moeExpertGateShape, _moeExpertGateBytes[layer]);
        ExpertSlabRole up = new(_moeExpertUp[layer], _moeExpertUpShape, _moeExpertUpBytes[layer]);
        ExpertSlabRole down = new(_moeExpertDown[layer], _moeExpertDownShape, _moeExpertDownBytes[layer]);

        // The token path feeds the tier one selection per routed slot — its own union of one sequence, every count 1.
        _residencyTier?.NoteActiveExperts(gate, up, down, _moeIds.AsSpan(0, sparse.RoutedExpertsPerToken), ReadOnlySpan<int>.Empty);

        GatherRoutedChain(gate, up, down, _moeIds.AsSpan(0, sparse.RoutedExpertsPerToken), (int expertIndex, ReadOnlySpan<byte> gateSlab, ReadOnlySpan<byte> upSlab, ReadOnlySpan<byte> downSlab) =>
        {
            engine.Gemv(_normed.AsSpan(0, Plan.HiddenSize), new WeightRef(gateSlab, _moeExpertGateShape, gate.Stack.Format), _gateUp);

            engine.Gemv(_normed.AsSpan(0, Plan.HiddenSize), new WeightRef(upSlab, _moeExpertUpShape, up.Stack.Format), _up);

            ApplyActivation(_gateUp.AsSpan(0, Plan.IntermediateSize), _up.AsSpan(0, Plan.IntermediateSize));

            engine.Gemv(_gateUp.AsSpan(0, Plan.IntermediateSize), new WeightRef(downSlab, _moeExpertDownShape, down.Stack.Format), _moeSubsetIn.AsSpan(0, hidden));

            for (int slot = 0; slot < sparse.RoutedExpertsPerToken; slot++)
            {
                if (_moeIds[slot] == expertIndex)
                {
                    AccumulateScaled(_proj, _moeSubsetIn.AsSpan(0, hidden), _moeWeights[slot]);
                    break;
                }
            }
        });

        if (sparse.HasSharedExpert)
        {
            RunSharedExpertToken(layer, _moeSubsetIn.AsSpan(0, hidden), engine);
            AccumulateScaled(_proj, _moeSubsetIn.AsSpan(0, hidden), SharedExpertWeight(layer, _normed.AsSpan(0, hidden)));
        }
    }

    // The shared expert's dense-shaped triplet over a single activation vector: the same gate/up/down GEMVs the dense
    // block runs, reading its own per-layer tensors and writing this token's contribution into the accumulation slot.
    private void RunSharedExpertToken(int layer, Span<float> output, IBackend engine)
    {
        LogicalTensor gate = _sharedGate[layer];
        ReadOnlySpan<byte> bytes = _weights.Resolve(gate);
        engine.Gemv(_normed.AsSpan(0, Plan.HiddenSize), new WeightRef(bytes, gate.Shape, gate.Format), _gateUp);

        LogicalTensor up = _sharedUp[layer];
        bytes = _weights.Resolve(up);
        engine.Gemv(_normed.AsSpan(0, Plan.HiddenSize), new WeightRef(bytes, up.Shape, up.Format), _up);

        ApplyActivation(_gateUp.AsSpan(0, Plan.IntermediateSize), _up.AsSpan(0, Plan.IntermediateSize));

        LogicalTensor down = _sharedDown[layer];
        bytes = _weights.Resolve(down);
        engine.Gemv(_gateUp.AsSpan(0, Plan.IntermediateSize), new WeightRef(bytes, down.Shape, down.Format), output);
    }

    // Sparse feed-forward for one merged decode step: every sequence routes first (one selection at a time in scratch,
    // copied into the per-sequence arrays before the next selection runs). The selections then fold into this step's
    // union active-expert set — each selected expert exactly once, ascending, with its member sequences counted — and one
    // gather pass over that set feeds every sequence: an expert several sequences routed to has its slabs resolved from
    // the slab provider and prefetched once for the step. Each gathered expert packs its members' rows of _batchNormed
    // into the subset buffer, runs gate/up/down as batched passes through its slabs, and accumulates each member's down
    // result straight into that sequence's projection stream under its routing weight. Selections are ascending-sorted
    // per sequence, so walking the union ascending replays exactly the slot-order accumulation a single token performs:
    // every batched row reads the same slab bytes and activation vector it read before, and each batched result stays
    // bit-identical to stepping that sequence alone. A declared shared expert runs last over all sequences, every
    // sequence scaled by its own SharedExpertWeight. The caller's post-feed-forward norm and residual add then run
    // exactly as they do on the dense path.
    private void ApplySparseFeedForwardBatch(SparseFfnDescriptor sparse, int layer, int n, ActivationFamily act, IBackend engine)
    {
        int hidden = Plan.HiddenSize;
        int routed = sparse.RoutedExpertsPerToken;

        for (int s = 0; s < n; s++)
        {
            RouteSparseExperts(sparse, layer, act.BatchNormed.AsSpan(s * hidden, hidden), act, engine);

            for (int pick = 0; pick < routed; pick++)
            {
                act.BatchMoeIds[s * routed + pick] = act.MoeIds[pick];
                act.BatchMoeWeights[s * routed + pick] = act.MoeWeights[pick];
            }
        }

        Span<float> batchProj = act.BatchProj.AsSpan(0, n * hidden);
        batchProj.Clear(); // the routed and shared contributions accumulate from zero below

        int unionCount = BuildUnionActiveExperts(n, routed, act);

        ExpertSlabRole gate = new(_moeExpertGate[layer], _moeExpertGateShape, _moeExpertGateBytes[layer]);
        ExpertSlabRole up = new(_moeExpertUp[layer], _moeExpertUpShape, _moeExpertUpBytes[layer]);
        ExpertSlabRole down = new(_moeExpertDown[layer], _moeExpertDownShape, _moeExpertDownBytes[layer]);

        // The tier sees this step's union before the gather walks it — selections with their member counts (DN-M7-3: the union set is the unit of residency work), and any swap round lands here, ahead of the reads its decisions steer. Gather arithmetic stays untouched either way.
        _residencyTier?.NoteActiveExperts(gate, up, down, act.MoeUnion.AsSpan(0, unionCount), act.MoeUnionCounts.AsSpan(0, unionCount));

        int selectionSlots = n * routed; // only this step's selections are live in the per-sequence arrays
        GatherRoutedChain(gate, up, down, act.MoeUnion.AsSpan(0, unionCount), (int expertIndex, ReadOnlySpan<byte> gateSlab, ReadOnlySpan<byte> upSlab, ReadOnlySpan<byte> downSlab) =>
            ExecuteGatheredExpertBatch(expertIndex, selectionSlots, gate.Stack.Format, gateSlab, up.Stack.Format, upSlab, down.Stack.Format, downSlab, act, engine));

        if (sparse.HasSharedExpert)
        {
            RunSharedExpertBatch(layer, n, hidden, act, engine);
        }
    }

    // Folds this step's per-sequence selections into the union active-expert set: ascending distinct expert indexes with
    // each expert's member count beside it. Selections are already sorted ascending within a sequence; the insertion
    // keeps the set itself ascending so the gather walks experts exactly in the order the dense-equivalent forward
    // accumulates them, and every selection slot lands in exactly one group.
    private int BuildUnionActiveExperts(int n, int routed, ActivationFamily act)
    {
        int[] union = act.MoeUnion;
        int[] counts = act.MoeUnionCounts;

        int unionCount = 0;
        for (int s = 0; s < n; s++)
        {
            for (int pick = 0; pick < routed; pick++)
            {
                int expertIndex = act.BatchMoeIds[s * routed + pick];

                int position = 0;
                while (position < unionCount && union[position] < expertIndex)
                {
                    position++;
                }

                if (position == unionCount || union[position] != expertIndex)
                {
                    for (int shift = unionCount; shift > position; shift--)
                    {
                        union[shift] = union[shift - 1];
                        counts[shift] = counts[shift - 1];
                    }

                    union[position] = expertIndex;
                    counts[position] = 0;
                    unionCount++;
                }

                counts[position]++;
            }
        }

        return unionCount;
    }

    // One gathered expert's chain over every sequence that selected it this step: its members are collected from the
    // per-sequence selections in sequence order, their normalized rows packed into the subset buffer, and gate/up/down
    // run as batched passes through the slabs the gather kernel just prefetched — bit-identical per row to the token
    // path, with each member's down result accumulated into that sequence's projection stream under its routing weight
    // as soon as the group completes. The family this route writes through carries the packed subset buffer and gate/up
    // intermediates — the default family's arrays for decode-side steps, the lane's own when a dedicated window runs them.
    private void ExecuteGatheredExpertBatch(
        int expertIndex,
        int selectionSlots,
        QuantFormat gateFormat,
        ReadOnlySpan<byte> gateSlab,
        QuantFormat upFormat,
        ReadOnlySpan<byte> upSlab,
        QuantFormat downFormat,
        ReadOnlySpan<byte> downSlab,
        ActivationFamily act,
        IBackend engine)
    {
        int hidden = Plan.HiddenSize;
        int inter = Plan.IntermediateSize;

        int count = 0;
        for (int s = 0; s < selectionSlots; s++)
        {
            if (act.BatchMoeIds[s] != expertIndex)
            {
                continue;
            }

            int sequence = s / act.MoeIds.Length;
            act.BatchNormed.AsSpan(sequence * hidden, hidden).CopyTo(act.SubsetIn.AsSpan(count * hidden, hidden));
            act.MoeMembersFlat[count] = sequence;
            act.MoeMemberWeights[count] = act.BatchMoeWeights[s];
            count++;
        }

        Span<float> subsetIn = act.SubsetIn.AsSpan(0, count * hidden);

        engine.GemvBatch(subsetIn, count, new WeightRef(gateSlab, _moeExpertGateShape, gateFormat), act.BatchGateUp.AsSpan(0, count * inter));
        engine.GemvBatch(subsetIn, count, new WeightRef(upSlab, _moeExpertUpShape, upFormat), act.BatchUp.AsSpan(0, count * inter));

        for (int m = 0; m < count; m++)
        {
            ApplyActivation(act.BatchGateUp.AsSpan(m * inter, inter), act.BatchUp.AsSpan(m * inter, inter));
        }

        engine.GemvBatch(act.BatchGateUp.AsSpan(0, count * inter), count, new WeightRef(downSlab, _moeExpertDownShape, downFormat), subsetIn);

        for (int m = 0; m < count; m++)
        {
            AccumulateScaled(act.BatchProj.AsSpan(act.MoeMembersFlat[m] * hidden, hidden), subsetIn.Slice(m * hidden, hidden), act.MoeMemberWeights[m]);
        }
    }

    private void GatherRoutedChain(ExpertSlabRole gate, ExpertSlabRole up, ExpertSlabRole down, ReadOnlySpan<int> expertIndexes, GatheredExpertChain handle) =>
        _gatherKernel!.Gather(_expertSlabs!, gate, up, down, expertIndexes, handle);

    // The shared expert's dense-shaped triplet over every sequence in one batched pass: the same gate/up/down
    // GemvBatch traversal the dense block runs, accumulating each sequence's output into its projection stream under
    // that sequence's SharedExpertWeight — 1f for a file shipping no selector vectors. An undeclared shared expert
    // never reaches this path and reads none of these roles at all.
    private void RunSharedExpertBatch(int layer, int n, int hidden, ActivationFamily act, IBackend engine)
    {
        LogicalTensor gate = _sharedGate[layer];
        ReadOnlySpan<byte> bytes = _weights.Resolve(gate);

        for (int s = 0; s < n; s++)
        {
            act.BatchNormed.AsSpan(s * hidden, hidden).CopyTo(act.SubsetIn.AsSpan(s * hidden, hidden));
        }

        Span<float> subsetIn = act.SubsetIn.AsSpan(0, n * hidden);
        int inter = Plan.IntermediateSize;

        engine.GemvBatch(subsetIn, n, new WeightRef(bytes, gate.Shape, gate.Format), act.BatchGateUp.AsSpan(0, n * inter));

        LogicalTensor up = _sharedUp[layer];
        bytes = _weights.Resolve(up);
        engine.GemvBatch(subsetIn, n, new WeightRef(bytes, up.Shape, up.Format), act.BatchUp.AsSpan(0, n * inter));

        for (int s = 0; s < n; s++)
        {
            ApplyActivation(act.BatchGateUp.AsSpan(s * inter, inter), act.BatchUp.AsSpan(s * inter, inter));
        }

        LogicalTensor down = _sharedDown[layer];
        bytes = _weights.Resolve(down);
        engine.GemvBatch(act.BatchGateUp.AsSpan(0, n * inter), n, new WeightRef(bytes, down.Shape, down.Format), subsetIn);

        for (int s = 0; s < n; s++)
        {
            AccumulateScaled(act.BatchProj.AsSpan(s * hidden, hidden), act.SubsetIn.AsSpan(s * hidden, hidden), SharedExpertWeight(layer, act.BatchNormed.AsSpan(s * hidden, hidden)));
        }
    }


    // One contiguous range of this model's layers, executed on one compute backend: a resolved route position as every forward walks it. Spans are settled once at open and read in order from here — nothing resolves them per token or per layer.
    private sealed class ExecutionSpan
    {
        public required IBackend Backend;

        /// <summary>This span's first layer, inclusive.</summary>
        public required int LayerStart;

        /// <summary>One past this span's last layer, exclusive — where its forward loop stops instead of at the model's whole layer count.</summary>
        public required int LayerEnd;
    }

    // The residual-stream storage one execution span beyond the first runs its rows through: allocated at open by that span's own backend through its aligned allocator, sized once to this engine's batched row capacity, and freed here before this engine hands back. Every row a forward writes into an arena was written by the crossing feeding it before any layer reads it; a view reached after disposal refuses rather than reading released memory.
    private sealed class ResidualCarrier : IDisposable
    {
        public const int Alignment = 64;

        // The host owns every backend this engine opened with: its allocator is asked to release this arena, and nothing here disposes it.
        private readonly IBackend _owner;
        private MemoryHandle _arena;
        private readonly unsafe float* _base;
        private readonly int _rowStride;
        private bool _released;

        /// <summary>Bytes this arena holds — rows × hidden, settled at open and never grown.</summary>
        public long Bytes { get; }

        /// <summary>The device of the backend whose aligned allocator owns this arena.</summary>
        public string OwnerDeviceId => _owner.DeviceId;

        public unsafe ResidualCarrier(IBackend owner, int rows, int hidden)
        {
            ArgumentNullException.ThrowIfNull(owner);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hidden);

            long floats = checked((long)rows * hidden);
            if (floats > int.MaxValue)
            {
                throw new ModelException(
                    $"a crossing arena of {rows} rows at {hidden} values wide passes what one span can address: staging storage sizes within a single view");
            }

            Bytes = floats * sizeof(float);
            _owner = owner;
            _rowStride = hidden;

            MemoryHandle arena = owner.AllocateAligned(Bytes, Alignment);
            if (arena.Pointer is null)
            {
                throw new OutOfMemoryException(
                    $"the execution backend on device '{owner.DeviceId}' allocated no {Bytes}-byte staging arena for this model's crossings.");
            }

            _arena = arena;
            _base = (float*)arena.Pointer;
        }

        /// <summary>The rows-wide view of this arena, laid out exactly as the family's own batched residual stream is.</summary>
        public unsafe Span<float> View(int rows)
        {
            if (_released)
            {
                throw new ObjectDisposedException(nameof(TransformerModel), "this engine released its staging arenas before a forward reached them");
            }

            return new Span<float>(_base, checked(rows * _rowStride));
        }

        public void Dispose()
        {
            if (_released)
            {
                return; // already returned to the backend that allocated it — repeated disposal never frees twice
            }

            _owner.FreeAligned(_arena);
            _arena = default;
            _released = true;
        }
    }

    /// <summary>One crossing this engine staged at open: the resolved segment position it feeds, the layers its span runs there, the device of the backend whose aligned allocator owns its arena, and that arena's byte size. Assembly-internal diagnostics — no host reports staging today.</summary>
    internal sealed record ExecutionCrossing(int SegmentIndex, int LayerStart, int LayerEnd, string StagingOwnerDeviceId, long StagingBytes);

    // The mutable activation storage one forward route writes through: every batched stream, routing scratch, and tail buffer a window or a batch step touches. The default family aliases the buffers this block has always allocated, so decode-side forwards read and write the very same arrays they did before any lane vocabulary existed; a dedicated posture gives prompt windows a second family of their own instead — separate storage per mutable path while weights, norms, rotation tables, KV bytes, and every other read-only structure stay shared between the two routes.
    private sealed class ActivationFamily
    {
        public required float[] BatchCur;
        public required float[] BatchNormed;
        public required float[] BatchQ;
        public required float[] BatchK;
        public required float[] BatchV;
        public required float[] BatchAttnOut;
        public required float[] BatchProj;
        public required float[] BatchGateUp;
        public required float[] BatchUp;

        /// <summary>Per-row online-softmax state for the row-tiled walk, sized to this family's widest window.</summary>
        public required float[] TiledRowMax;
        public required float[] TiledRowSumExp;

        // Routing scratch for sparse feed-forward blocks: one selection at a time plus per-sequence copies across a batched step, and the step's union active-expert set with its grouped member rows and weights — every selected expert once, members of each expert in sequence order. Empty arrays when the model runs no expert routing at all — nothing reads them then.
        public required float[] MoeScores;
        public required int[] MoeIds;
        public required float[] MoeWeights;
        public required int[] BatchMoeIds;
        public required float[] BatchMoeWeights;
        public required int[] MoeUnion;
        public required int[] MoeUnionCounts;
        public required int[] MoeMembersFlat;
        public required float[] MoeMemberWeights;

        /// <summary>Packed expert-group activations and down results, held in place across one group.</summary>
        public required float[] SubsetIn;

        // The window's tail buffers: the final row's normalized stream and its vocab-sized logits. For the default family these are the single-token buffers every forward has always used — the same arrays LastLogits reads — and a dedicated lane gets its own pair so an overlapped prompt window never writes where decode is reading.
        public required float[] FinalNormed;
        public required float[] FinalLogits;

        /// <summary>Total bytes of mutable activation storage this family holds.</summary>
        public long ActivationBytes =>
            (long)(BatchCur.Length + BatchNormed.Length + BatchQ.Length + BatchK.Length + BatchV.Length + BatchAttnOut.Length + BatchProj.Length + BatchGateUp.Length + BatchUp.Length
                + TiledRowMax.Length + TiledRowSumExp.Length + MoeScores.Length + MoeWeights.Length + BatchMoeWeights.Length + MoeMemberWeights.Length + SubsetIn.Length + FinalNormed.Length + FinalLogits.Length) * sizeof(float)
            + (long)(MoeIds.Length + BatchMoeIds.Length + MoeUnion.Length + MoeUnionCounts.Length + MoeMembersFlat.Length) * sizeof(int);
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

/// <summary>
/// One measured window's dense stage and per-layer times, in milliseconds. Stages partition every instrumented prefill or decode forward:
/// seed/embedding lookups with their input-embedding scale; the sequence slot reservations that route makes before its layers write (cache
/// interaction); each layer's attention block from its norms through its sublayer fold and that block's feed-forward norm — split into its Q/K/V
/// projection traversals and its output-projection traversal (which together are <see cref="AttentionWeightMs"/>), its causal read spans including
/// a tiled walk or a composed device step, and the K/V slot writes it performs (bias/QK-norm/RoPE/residual folds ride inside this stage beside those
/// substages); each layer's feed-forward block — split into its gate/up pair traversal and its down traversal (which together are
/// <see cref="FeedForwardWeightMs"/>), or a sparse block's whole routed/shared chain; the output head each window reaches; and each instrumented
/// forward call as a whole. Parent stages cover their substages rather than adding to them: attention weight + kernel + KV write times sum inside
/// the attention stage, feed-forward traversal time inside the feed-forward stage, each traversal parent is exactly the sum of its traversal
/// substages, and every parent beside seed and cache interaction stays inside the total apart from cross-span staging copies and per-call argument
/// checks. Assembly-internal diagnostic shape behind the engine's dense profiling seam.
/// </summary>
internal readonly record struct DenseForwardProfile(
    double SeedMs,
    double CacheAppendMs,
    double AttentionMs,
    double AttentionWeightMs,
    double AttentionProjectionMs,
    double AttentionOutputMs,
    double AttentionKernelMs,
    double KvWriteMs,
    double FeedForwardMs,
    double FeedForwardWeightMs,
    double FeedForwardGateUpMs,
    double FeedForwardDownMs,
    double SparseChainMs,
    double LogitsHeadMs,
    double TotalMs,
    IReadOnlyList<DenseForwardLayerTiming> Layers);

/// <summary>One dense block's whole attention + feed-forward span, in milliseconds.</summary>
internal readonly record struct DenseForwardLayerTiming(int Layer, double TotalMs);
