using Lair.Kv;
using Lair.Sampling;

namespace Lair.Models.Scheduling;

/// <summary>
/// Where one pipeline stands in its own state machine: admit → transform → prefill → decode → finish, never backwards. A
/// pipeline is admitted with its capacity reservation and no engine work issued for it yet; it takes an advancement slot to
/// start forwarding prompt chunks; the pass that turns finished-prefill logits into its first generated token runs on its own
/// logits row before that row reaches its sampler; from there it advances one token per shared engine step, each sampled from
/// its own row; and it lands in <see cref="PipelineStage.Finished"/> exactly once. Stages are the control plane's progress
/// record: they decide only whether a pipeline may advance, never which batch its math rides in — a merged step hands this
/// pipeline the very logits row an unmerged step would have produced for it, so merging is invisible to its tokens.
/// </summary>
public enum PipelineStage
{
    /// <summary>Admitted and holding its capacity reservation, with no engine work issued for it yet: no advancement slot was free when it arrived.</summary>
    Admitted,

    /// <summary>Holding an advancement slot while its remaining prompt chunks are forwarded through the step executor. A pipeline whose every block was adopted at admission never occupies this stage: it has no chunk left and is promoted to decoding when it takes a slot.</summary>
    Prefilling,

    /// <summary>Prefill logits are being transformed and sampled into this pipeline's first generated token; the same per-pipeline transform-and-sample authority serves every later step while decoding.</summary>
    Transforming,

    /// <summary>Advancing one token per shared engine step, each from its own logits row.</summary>
    Decoding,

    /// <summary>The outcome has been published and this pipeline's blocks are released; the scheduler drops it from its working set at this boundary.</summary>
    Finished,
}

/// <summary>
/// One generation request as the control plane sees it: a logical state machine advanced on the scheduler thread — never an OS
/// thread of its own — owning its sampler (so its RNG stream never interleaves with another pipeline's), its paged KV cache, and
/// its admission state. The step executor decides how this pipeline's math runs; whether it may advance at a given boundary, what
/// blocks it reserved for the whole run, and where it stands all live here instead of in the batch composition around it. The
/// scheduler thread is the only writer of the mutable fields; the cancellation flag is the one field any thread may set.
/// </summary>
internal sealed class RequestPipeline
{
    private int _cancelRequested;

    public RequestPipeline(ScheduledRequest request, CancellationToken ct, IKvPool pool, long id)
    {
        Request = request;
        Cancellation = ct;
        Id = id;
        Sampler = new Sampler(request.Sampling);
        Kv = pool.NewSequence(); // the sequence type this pool's storage is addressed by — float slots or a quantized pair's groups
        Outcome = new TaskCompletionSource<SequenceOutcome>(TaskCreationOptions.RunContinuationsAsynchronously);
        Handle = new PipelineHandle(this);
    }

    public long Id { get; }

    /// <summary>The request this pipeline runs: prompt tokens, resolved token budgets, sampling knobs, stop id, and streaming hook.</summary>
    public ScheduledRequest Request { get; }

    public CancellationToken Cancellation { get; }

    /// <summary>This pipeline's own sampler, built once from its sampling params; no other pipeline shares it.</summary>
    public Sampler Sampler { get; }

    public IKvSequence Kv { get; }

    public TaskCompletionSource<SequenceOutcome> Outcome { get; }

    public PipelineHandle Handle { get; }

    /// <summary>This pipeline's position in its state machine, as of the last stage transition on the scheduler thread.</summary>
    public PipelineStage Stage { get; private set; } = PipelineStage.Admitted;

    /// <summary>True while this pipeline holds one advancement slot. Only a slotted pipeline is ever chosen for step work, so an over-cap pipeline waits at <see cref="PipelineStage.Admitted"/> rather than being preempted after it started.</summary>
    public bool HoldsSlot;

    /// <summary>True while this pipeline's pool admission window is open: its whole-lifetime block reservation may spend above a hot-shrunk soft cap until it finishes, including the stretch where it waits for an advancement slot. Scheduler-thread only (every mutation runs under the gate).</summary>
    public bool AdmissionWindowOpened;

    /// <summary>Prompt tokens forwarded so far; chunks remain while this is below the prompt length.</summary>
    public int PromptOffset;

    /// <summary>Prompt tokens whose blocks were adopted from the prefix cache instead of prefilled; set once at admission.</summary>
    public int PrefillCachedTokens;

    /// <summary>The sampled token awaiting its next decode step; valid once decoding starts.</summary>
    public int PendingToken;

    public readonly List<int> Generated = [];

    /// <summary>Per-generated-token log-probability captures, appended in generation order; empty when the request did not ask for logprobs.</summary>
    public readonly List<TokenLogprob> LogProbEntries = [];

    // ------------------------------------------------------------------ stage transitions (scheduler thread only)

    /// <summary>Takes this pipeline out of admission and onto prompt chunks. The caller has just given it an advancement slot.</summary>
    public void StartPrefilling() => Stage = PipelineStage.Prefilling;

    /// <summary>Marks the transform-and-sample pass that consumes prefill logits into this pipeline's first generated token.</summary>
    public void StartTransforming() => Stage = PipelineStage.Transforming;

    /// <summary>Promotes this pipeline to one token per engine step, each from its own logits row. The caller has forwarded or adopted every prompt chunk by now; a pipeline whose whole prefix was adopted enters decode with no chunk ever issued for it.</summary>
    public void StartDecoding() => Stage = PipelineStage.Decoding;

    /// <summary>Records this pipeline's terminal stage and takes it out of every progress query. Idempotent: a second finish — an eos in the sampling loop followed by a batch fault, or a shutdown sweep over a pipeline that already finished — leaves the stage where it is.</summary>
    public void MarkFinished() => Stage = PipelineStage.Finished;

    // ------------------------------------------------------------------ progress queries the scheduler and executor read

    /// <summary>True while this pipeline is eligible for step work: it holds a slot and has not finished. An over-cap pipeline stays out of every batch until a slot frees.</summary>
    public bool IsAdvancing => HoldsSlot && Stage != PipelineStage.Admitted && Stage != PipelineStage.Finished;

    /// <summary>True while this pipeline is slotted but its prompt still has chunks to forward.</summary>
    public bool IsPrefilling => IsAdvancing && PromptOffset < Request.PromptTokens.Length;

    public bool IsDecoding => IsAdvancing && Stage == PipelineStage.Decoding;

    /// <summary>True once the outcome is published; the scheduler removes finished pipelines from its working set.</summary>
    public bool IsFinished => Outcome.Task.IsCompleted;

    /// <summary>True while this pipeline still has token budget left to advance.</summary>
    public bool HasBudget => Generated.Count < Request.GenerationTokenCeiling;

    // ------------------------------------------------------------------ per-pipeline sampling authority

    /// <summary>
    /// Transforms and samples a logits row that belongs to this pipeline alone. When its sampling params carry penalties or a logit
    /// bias, the row is adjusted in place first — penalties over this pipeline's prompt plus already-generated history, biases added to
    /// their tokens' logits — so each pipeline's transform stays on its own row and the sampler itself sees only the adjusted logits.
    /// When the request asked for logprobs, the chosen token's entry is captured from that same transformed row; the stop token ends a
    /// sequence without being generated, so it gets no entry. Which batch produced this row — one solo forward or one position inside a
    /// merged step — changes nothing here: the row holds exactly the values an unmerged step would have produced for this pipeline, and
    /// its RNG stream is its own sampler's either way, which is what makes a pipeline's token stream identical alone or merged.
    /// </summary>
    public int SampleFrom(Span<float> row)
    {
        SamplingParams p = Request.Sampling;
        if (p.HasLogitTransform)
        {
            SamplingTransforms.Apply(row, p, Request.PromptTokens, Generated);
        }

        int token = Sampler.Sample(row);
        if (Request.LogProbTopN is int topN && token != Request.EosId)
        {
            (double logProb, IReadOnlyList<(int TokenId, double LogProb)> top) = Sampler.LogProbsFor(row, token, topN);
            LogProbEntries.Add(new TokenLogprob(token, logProb, top));
        }

        return token;
    }

    public void RequestCancel() => Volatile.Write(ref _cancelRequested, 1);

    public bool CancelRequested => Volatile.Read(ref _cancelRequested) != 0;
}
