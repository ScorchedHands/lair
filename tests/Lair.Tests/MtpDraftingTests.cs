using Lair.Backends.Cpu;
using Lair.Kv;
using Lair.Loader;
using Lair.Models.Engine;
using Lair.Models.Scheduling;
using Lair.Sampling;
using Lair.Sampling.Drafting;

namespace Lair.Tests;

/// <summary>
/// CPU-only speculative-draft seam gates on the pinned SmolLM fixture and a synthetic deterministic target: drafted output is
/// token-identical to non-drafted greedy (and seeded) decoding under every scripted-head posture — oracle, adversarial, mixed
/// window — rejected drafts never advance sequence positions or write KV bytes an undrafted run would not have written, the
/// accept-rate tallies report exactly the proposed/accepted/rejected counts each boundary produced, and a synthetic draft head
/// serves families whose artifacts carry no model-native multi-token tensors while an absent head refuses drafting outright.
/// Verify forwards here are one-for-one ordinary decode steps of the same sequence (accept-prefix semantics), so these gates
/// prove output equivalence; single-forward batched verification over an unaccepted window is not what this build's engine
/// exposes yet and stays outside these gates rather than being approximated by a widened band. Runs in the serialized real-model
/// lane with the other serve suites.
/// </summary>
[Trait("Category", "Mtp")]
[Collection("model-loaded")] // shares one serialized lane with the other real-model suites: two engines mapping 270 MB at once is not what this box should be doing
public sealed class MtpDraftingTests
{
    private const int PromptTokens = 16;

    /// <summary>An id no vocabulary can produce, so fixture runs finish on their generation ceiling rather than a tokenizer stop token.</summary>
    private const int NeverEos = -1;

    // A fixed out-of-continuation token: it cannot be what the model samples at any position of the recorded stream under the
    // same posture, so heads proposing it are provably wrong exactly where they propose.
    private static int ForbiddenIdIn(IReadOnlyList<int> continuation) => ContinuationIds(continuation).Contains(0) ? 1 : 0;

    private static HashSet<int> ContinuationIds(IReadOnlyList<int> continuation) => [.. continuation];

    // ---------------------------------------------------------------- output equivalence against non-drafted greedy decode

    [Theory]
    [InlineData(0f, 4)]       // greedy posture, window of four drafts per boundary
    [InlineData(0.7f, 2)]     // seeded sampling posture (seed fixed below), window of two
    public void DraftedRun_OracleScriptedHead_OutputMatchesNonDraftedGreedyStream(float temperature, int draftWindow)
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        SamplingParams sampling = new() { Temperature = temperature, TopK = 40, TopP = 1.0f, Seed = 20261122 };
        const int ceiling = 8;

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(ModelFixture.Resolve()!);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128);
        using BlockPool pool = NewPool(model, backend);

        int[] prompt = [.. Enumerable.Repeat(7, PromptTokens)];

        // Non-drafted reference through the engine's own decode seam.
        List<int> baseline;
        long undraftedAppendsSeen;
        using (IKvSequence kv = pool.NewSequence())
        {
            baseline = UndraftedRun(model, kv, prompt, sampling, ceiling);
            undraftedAppendsSeen = kv.Length - PromptTokens; // Prefill wrote the prompt positions; each decode step fed one token more
        }

        Assert.Equal(ceiling, baseline.Count); // a deterministic fixture stream fills its ceiling before any sentinel stop id here

        SpeculativeDecoder decoder = new(sampling, OracleHead(baseline, draftWindow, PromptTokens), model.Desc.VocabSize);
        using IKvSequence draftedKv = pool.NewSequence();

        DraftedRunResult run = RunDrafted(decoder, model, draftedKv, prompt, ceiling);

        Assert.Equal(FinishReason.Length, run.Reason); // the scripted continuation never emits the sentinel stop id
        Assert.Equal(baseline, run.Tokens);
        Assert.Equal(undraftedAppendsSeen + PromptTokens, draftedKv.Length); // same appended positions as the reference: drafts added nothing extra

        int expectedProposals = ExpectedWindowedProposalCount(ceiling - 1, draftWindow); // the first token came from prefill logits before any window ran
        Assert.Equal(expectedProposals, run.Tallies.DraftsProposed);
        Assert.Equal(expectedProposals, run.Tallies.DraftTokensAccepted); // oracle head: nothing rejected anywhere
        Assert.True(run.Tallies.AcceptRate > 0.999);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void DraftedRun_AlwaysWrongHead_RejectsEverything_AndOutputStaysIdenticalToNonDrafted(int draftWindow)
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        SamplingParams sampling = new() { Temperature = 0f };
        const int ceiling = 6;

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(ModelFixture.Resolve()!);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128);
        using BlockPool pool = NewPool(model, backend);

        int[] prompt = [.. Enumerable.Repeat(7, PromptTokens)];

        List<int> baseline;
        long undraftedAppendsSeen;
        using (IKvSequence kv = pool.NewSequence())
        {
            baseline = UndraftedRun(model, kv, prompt, sampling, ceiling);
            undraftedAppendsSeen = kv.Length - PromptTokens;
        }

        // Adversarial head: a provably out-of-continuation id at every position. Acceptance dies at each boundary's first
        // proposal, whose target sample is still authoritative output — the stream stays identical while nothing extra reaches KV.
        SpeculativeDecoder decoder = new(sampling, WrongHead(baseline, draftWindow, PromptTokens), model.Desc.VocabSize);
        using IKvSequence draftedKv = pool.NewSequence();

        DraftedRunResult run = RunDrafted(decoder, model, draftedKv, prompt, ceiling);

        Assert.Equal(baseline, run.Tokens); // rejected drafts changed no token of the output stream
        Assert.Equal(undraftedAppendsSeen + PromptTokens, draftedKv.Length); // identical append footprint to the undrafted reference
        Assert.Equal(ceiling - 1, run.Tallies.DraftsProposed); // consultation stops at each boundary's first mismatch, so only index 0 of every window is ever consulted regardless of its size
        Assert.Equal(0, run.Tallies.DraftTokensAccepted);
        Assert.Equal(0.0, run.Tallies.AcceptRate);
        Assert.Equal(ceiling - 1, run.Tallies.MismatchedSteps); // one mismatched boundary per decode step after the prefill-sampled first token
    }

    [Fact]
    public void DraftedRun_MixedWindowHead_AcceptsPrefixAndRejectsTail_OutputMatchesNonDrafted()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        SamplingParams sampling = new() { Temperature = 0f };
        const int ceiling = 6;

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(ModelFixture.Resolve()!);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128);
        using BlockPool pool = NewPool(model, backend);

        int[] prompt = [.. Enumerable.Repeat(7, PromptTokens)];

        List<int> baseline;
        using (IKvSequence kv = pool.NewSequence())
        {
            baseline = UndraftedRun(model, kv, prompt, sampling, ceiling);
        }

        // Mixed head: matches the reference for one position per boundary, then proposes a provably wrong token — acceptance
        // takes the prefix and rejects exactly that tail on every boundary.
        SpeculativeDecoder decoder = new(sampling, PrefixThenWrongHead(baseline, PromptTokens), model.Desc.VocabSize);
        using IKvSequence draftedKv = pool.NewSequence();

        DraftedRunResult run = RunDrafted(decoder, model, draftedKv, prompt, ceiling);

        Assert.Equal(baseline, run.Tokens); // the accepted prefix plus each mismatch position's own target sample is exactly the reference stream
        Assert.Equal(PromptTokens + baseline.Count - 1, draftedKv.Length); // one append per fed token — like the undrafted run, where the last sampled token stays pending and unforwarded

        // Boundary trace for a window of two against six generated tokens: [match,reject] consumes two steps twice, then the
        // final boundary's window clamps to its single remaining proposal and that match still records as accepted even though
        // this very token finishes the ceiling. Consultation stops at each mismatch, so rejected tails past index 1 never count.
        Assert.Equal(5, run.Tallies.DraftsProposed); // 2 + 2 + 1 (clamped tail window)
        Assert.Equal(3, run.Tallies.DraftTokensAccepted); // the accepted prefix of each boundary: match at index 0 twice plus the clamped single-match tail
        Assert.Equal(0.6, run.Tallies.AcceptRate);
        Assert.Equal(2, run.Tallies.MismatchedSteps); // only the two full-window boundaries reached a rejection position
        Assert.Equal(1, run.Tallies.FullyAcceptedSteps); // the clamped boundary consumed its whole (single-proposal) window by accepting it
    }

    // ---------------------------------------------------------------- accept-rate accounting reports expected counts

    [Fact]
    public void DraftedRun_OracleHead_TallyCountersMatchWindowedByBoundary()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        SamplingParams sampling = new() { Temperature = 0f };
        const int ceiling = 6;
        const int window = 3;

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(ModelFixture.Resolve()!);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128);
        using BlockPool pool = NewPool(model, backend);

        int[] prompt = [.. Enumerable.Repeat(7, PromptTokens)];

        List<int> baseline;
        using (IKvSequence kv = pool.NewSequence())
        {
            baseline = UndraftedRun(model, kv, prompt, sampling, ceiling);
        }

        SpeculativeDecoder decoder = new(sampling, OracleHead(baseline, window, PromptTokens), model.Desc.VocabSize);
        using IKvSequence kv2 = pool.NewSequence();

        DraftedRunResult run = RunDrafted(decoder, model, kv2, prompt, ceiling);
        MtpTallySnapshot tallies = run.Tallies;

        Assert.Equal(tallies.DraftsProposed, tallies.DraftTokensAccepted + tallies.DraftsRejected);
        Assert.Equal(0, tallies.NoProposalSteps); // an oracle head always proposes while budget remains
        Assert.Equal(0, tallies.MismatchedSteps);

        // Every boundary consumed its whole covered window: full windows while budget allows, then the truncated tail — and a
        // fully consumed window ends the boundary, so each covered window is one advanced step.
        int expectedSteps = WindowedStepCount(ceiling - 1, window);
        Assert.Equal(expectedSteps, tallies.StepsAdvanced);
        Assert.Equal(expectedSteps, tallies.FullyAcceptedSteps);
        Assert.Equal(ExpectedWindowedProposalCount(ceiling - 1, window), tallies.DraftsProposed);
    }

    [Fact]
    public void DraftedRun_EmptyWindowHead_AdvancesExactlyOneOrdinaryStepPerBoundary()
    {
        if (ModelFixture.Resolve() is null)
        {
            return; // real-model tests skip when the fixture is absent
        }

        SamplingParams sampling = new() { Temperature = 0f };
        const int ceiling = 5;

        using CpuBackend backend = new();
        using GgufFile gguf = GgufFile.Open(ModelFixture.Resolve()!);
        using TransformerModel model = TransformerModel.Open(gguf, backend, maxContextTokens: 128);
        using BlockPool pool = NewPool(model, backend);

        int[] prompt = [.. Enumerable.Repeat(7, PromptTokens)];

        List<int> baseline;
        using (IKvSequence kv = pool.NewSequence())
        {
            baseline = UndraftedRun(model, kv, prompt, sampling, ceiling);
        }

        // A head that proposes nothing turns every boundary into one ordinary decode step: output stays identical and the
        // tallies say so — proposed zero, no acceptances, a no-proposal count per boundary.
        SpeculativeDecoder decoder = new(sampling, EmptyHead(), model.Desc.VocabSize);
        using IKvSequence kv2 = pool.NewSequence();

        DraftedRunResult run = RunDrafted(decoder, model, kv2, prompt, ceiling);

        Assert.Equal(baseline, run.Tokens);
        Assert.Equal(0, run.Tallies.DraftsProposed);
        Assert.Equal(0.0, run.Tallies.AcceptRate);
        Assert.Equal(ceiling - 1, run.Tallies.NoProposalSteps); // the first token came from prefill logits; each later boundary advanced one ordinary step
        Assert.Equal(ceiling - 1, run.Tallies.StepsAdvanced);
    }

    // ---------------------------------------------------------------- synthetic heads with no model-native tensors

    [Fact]
    public void SyntheticDraftHead_ServesSyntheticDeterministicTarget_DraftedStreamMatchesUndrafted()
    {
        const int vocab = 32;
        const int promptLen = 3;

        // A deterministic closed-form "target": each fed token scores highest at a pseudo-logit row derived from it. No GGUF,
        // no tensors — exactly the posture a family without draft-head artifacts needs the seam to support through a supplied head.
        SamplingParams sampling = new() { Temperature = 0f }; // greedy: argmax of each row is fully deterministic here
        const int ceiling = 6;

        List<int> baseline = UndraftedSyntheticRun(vocab, promptLen, sampling, ceiling);

        SpeculativeDecoder oracle = new(sampling, OracleHead(baseline, 3, promptLen), vocab);
        DraftedRunResult draftedOracle = RunDraftedSynthetic(oracle, vocab, promptLen, ceiling);
        Assert.Equal(baseline, draftedOracle.Tokens);
        Assert.True(draftedOracle.Tallies.AcceptRate > 0.999);

        SpeculativeDecoder wrong = new(sampling, WrongHead(baseline, 3, promptLen), vocab);
        DraftedRunResult draftedWrong = RunDraftedSynthetic(wrong, vocab, promptLen, ceiling);
        Assert.Equal(baseline, draftedWrong.Tokens); // rejected drafts preserved the undrafted stream on the synthetic posture too
        Assert.Equal(0.0, draftedWrong.Tallies.AcceptRate);

        SpeculativeDecoder mixed = new(sampling, PrefixThenWrongHead(baseline, promptLen), vocab);
        DraftedRunResult draftedMixed = RunDraftedSynthetic(mixed, vocab, promptLen, ceiling);
        Assert.Equal(baseline, draftedMixed.Tokens); // prefix accepted, tail rejected, stream unchanged — no model tensors involved anywhere
    }

    [Fact]
    public void SyntheticTarget_RejectedDrafts_NeverAppendSequencePositionsOrWriteKvBytes()
    {
        const int vocab = 32;
        const int promptLen = 2;
        const int ceiling = 4;

        // The synthetic target writes a marker into the KV slot it appends for each fed token, so append counts and written
        // bytes are observable: an undrafted run's positions must be exactly reproduced by a fully-rejecting drafted run.
        using CpuBackend backend = new();
        using BlockPool pool = new(blockTokens: 4, layerCount: 1, kvHeads: 1, headDim: 2, capacityBlocks: 32, backend);

        List<int> baseline;
        float[] markersUndrafted;
        int undraftedPositions;
        using (SequenceKv kv = new(pool))
        {
            baseline = UndraftedSyntheticRun(vocab, promptLen, new SamplingParams { Temperature = 0f }, ceiling, kv);
            undraftedPositions = kv.Length;
            markersUndrafted = ReadMarkers(kv);
        }

        using SequenceKv draftedKv = new(pool);
        {
            SpeculativeDecoder wrong = new(new SamplingParams { Temperature = 0f }, WrongHead(baseline, 3, promptLen), vocab);
            DraftedRunResult run = RunDraftedSynthetic(wrong, vocab, promptLen, ceiling, draftedKv);

            Assert.Equal(baseline, run.Tokens); // output unchanged by rejection...
            Assert.Equal(undraftedPositions, draftedKv.Length); // ...and the same positions were appended as in the undrafted run — rejected drafts added none.
            Assert.Equal(markersUndrafted, ReadMarkers(draftedKv)); // every written slot holds what the undrafted run wrote there
        }

        Assert.True(baseline.Count > 0);
    }

    [Fact]
    public void MtpDraftPolicy_NoHeadNoNativeTensors_RefusesNamingModelAndRemedy()
    {
        DraftHeadDecision decision = MtpDraftPolicy.Resolve("qwen4exp-flash-next", hasNativeDraftTensors: false, configuredHead: null);

        Assert.True(decision.Refused); // the target family carries no draft-head tensors — drafting must not silently free-run
        Assert.Contains("MTP drafting is refused", decision.Reason);
    }

    [Fact]
    public void MtpDraftPolicy_ConfiguredSyntheticHead_EnablesDraftingWithoutNativeTensors()
    {
        DraftHeadDecision decision = MtpDraftPolicy.Resolve(
            "qwen4exp-flash-next",
            hasNativeDraftTensors: false,
            configuredHead: new ScriptedDraftHead(2, _ => []));

        Assert.True(decision.Enabled); // a supplied stand-in head is what the seam requires for tensor-less families...
        Assert.Contains("host-supplied draft head", decision.Reason);

        DraftHeadDecision nativeDeclared = MtpDraftPolicy.Resolve("some-family", hasNativeDraftTensors: true, configuredHead: null);
        Assert.True(nativeDeclared.Refused); // declaring tensors alone does not enable drafting until a provider binds them to the seam
    }

    [Fact]
    public void ScriptedDraftHead_OutOfVocabularyProposal_FailsNamingPositionAndVocab()
    {
        ScriptedDraftHead head = new(2, ctx => [0, ctx.VocabSize]); // second proposal sits outside the target vocabulary

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(
            () => head.Propose(new DraftProposalContext(vocabSize: 32, history: [1], lastSampledLogits: null)));

        Assert.Contains("outside the target vocabulary", error.Message);
    }

    // ---------------------------------------------------------------- helpers

    private static BlockPool NewPool(TransformerModel model, CpuBackend backend) =>
        new(blockTokens: 8, layerCount: model.Desc.LayerCount, kvHeads: model.Desc.KvHeadCount, headDim: model.Desc.HeadDim, capacityBlocks: 64, backend);

    // The undrafted reference: prefill's logits row samples the first token without a forward (the same seam a served pipeline
    // uses), then DecodeNext advances one authoritative token per step — greedy or seeded via this run's own Sampler.
    private static List<int> UndraftedRun(TransformerModel model, IKvSequence kv, int[] prompt, SamplingParams sampling, int ceiling)
    {
        Sampler sampler = new(sampling);
        List<int> generated = [];

        Span<float> logits = model.Prefill(prompt, kv);
        int first = sampler.Sample(logits);
        generated.Add(first); // the sentinel id cannot appear in a real vocabulary, so the first sample is always generated
        int pending = first;
        while (generated.Count < ceiling)
        {
            logits = model.DecodeNext(pending, kv);
            int next = sampler.Sample(logits);
            if (next == NeverEos)
            {
                break;
            }

            generated.Add(next);
            pending = next;
        }

        return generated;
    }

    private static DraftedRunResult RunDrafted(SpeculativeDecoder decoder, TransformerModel model, IKvSequence kv, int[] prompt, int ceiling)
        => decoder.RunDrafted(
            (token, seq) => model.DecodeNext(token, seq),
            // Same boundary a served pipeline starts from: the drafted sequence's own prompt window runs first, and its final logits row seeds the first sample.
            () =>
            {
                _ = model.Prefill(prompt, kv);
                return model.LastLogits();
            },
            kv,
            prompt,
            generationTokenCeiling: ceiling,
            eosId: NeverEos);

    private static int PredictedNext(int fedTokenOrSeed, int vocab) => (fedTokenOrSeed * 7 + 3) % vocab;

    private static Span<float> SyntheticRow(int fedTokenOrSeed, int vocab, float[] scratch)
    {
        for (int i = 0; i < vocab; i++)
        {
            scratch[i] = -1f;
        }

        scratch[PredictedNext(fedTokenOrSeed, vocab)] = 5f;
        return scratch.AsSpan();
    }

    // Synthetic-target run: greedy sampler over closed-form rows. When a sequence is supplied, each forward appends its fed
    // token and writes a marker there — the same append shape DecodeNext produces for authoritative tokens only.
    private static List<int> UndraftedSyntheticRun(int vocab, int promptLen, SamplingParams sampling, int ceiling, SequenceKv? kvWithMarkers = null)
    {
        float[] scratch = new float[vocab];
        Sampler sampler = new(sampling);
        List<int> generated = [];

        int first = sampler.Sample(SyntheticRow(1, vocab, scratch)); // the synthetic prompt's last token seeds the first boundary
        generated.Add(first);
        int pending = first;
        while (generated.Count < ceiling)
        {
            if (kvWithMarkers is not null)
            {
                _ = kvWithMarkers.Append(); // one append per fed token, like the engine's decode step
                kvWithMarkers.WriteK(kvWithMarkers.Length - 1, layer: 0, [pending + 0.5f, -pending]); // marker the fed token wrote into its slot
            }

            Span<float> row = SyntheticRow(pending, vocab, scratch);
            int next = sampler.Sample(row);
            generated.Add(next);
            pending = next;
        }

        return generated;
    }

    private static DraftedRunResult RunDraftedSynthetic(
        SpeculativeDecoder decoder,
        int vocab,
        int promptLen,
        int ceiling,
        SequenceKv? kvWithMarkers = null)
    {
        float[] scratch = new float[vocab];

        Func<int, IKvSequence, Span<float>> forward = (fedToken, seq) =>
        {
            if (kvWithMarkers is not null && ReferenceEquals(seq, kvWithMarkers))
            {
                _ = kvWithMarkers.Append(); // verify forwards feed authoritative tokens only, so these appends mirror the undrafted run exactly
                kvWithMarkers.WriteK(kvWithMarkers.Length - 1, layer: 0, [fedToken + 0.5f, -fedToken]);
            }

            return SyntheticRow(fedToken, vocab, scratch);
        };

        Func<Span<float>> prefillLogits = () => SyntheticRow(1, vocab, scratch); // the synthetic prompt's last token seeds the first boundary

        int[] prompt = [.. Enumerable.Repeat(1, promptLen)]; // history shape only: the synthetic target keys off fed tokens, not prompt content
        IKvSequence sequence = kvWithMarkers is null ? NullSequence.Instance : kvWithMarkers;
        return decoder.RunDrafted(forward, prefillLogits, sequence, prompt, generationTokenCeiling: ceiling, eosId: NeverEos);
    }

    private static float[] ReadMarkers(SequenceKv kv)
    {
        List<float> values = [];
        for (int pos = 0; pos < kv.Length; pos++)
        {
            ReadOnlySpan<float> k = kv.ReadK(pos, layer: 0);
            if (!k.IsEmpty)
            {
                values.Add(k[0]);
                values.Add(k[1]);
            }
        }

        return [.. values];
    }

    // A head that replays the reference continuation from the history it is given: proposals equal what a non-drafted run would
    // sample next at each position, so acceptance covers whole windows. Heads see prompt + accepted tokens only, exactly like
    // real drafting sees its sequence state.
    private static ScriptedDraftHead OracleHead(IReadOnlyList<int> continuation, int window, int promptLen) => new(window, ctx =>
    {
        List<int> drafts = [];
        int generatedSoFar = ctx.History.Length - promptLen;
        for (int i = 0; i < window && generatedSoFar + i < continuation.Count; i++)
        {
            drafts.Add(continuation[generatedSoFar + i]);
        }

        return drafts;
    });

    // Proposes a provably out-of-continuation id at every position: under the same posture the target can only sample
    // continuation tokens at those positions, so each proposal mismatches by construction.
    private static ScriptedDraftHead WrongHead(IReadOnlyList<int> continuation, int window, int promptLen)
    {
        int forbidden = ForbiddenIdIn(continuation);
        return new ScriptedDraftHead(window, ctx =>
        {
            List<int> drafts = [];
            for (int i = 0; i < window; i++)
            {
                drafts.Add(forbidden);
            }

            return drafts;
        });
    }

    // First proposal replays the known continuation (accepted); the second is provably wrong, so acceptance stops at index 1
    // on every boundary.
    private static ScriptedDraftHead PrefixThenWrongHead(IReadOnlyList<int> continuation, int promptLen)
    {
        int forbidden = ForbiddenIdIn(continuation);
        return new ScriptedDraftHead(2, ctx =>
        {
            List<int> drafts = [];
            int generatedSoFar = ctx.History.Length - promptLen;
            if (generatedSoFar < continuation.Count)
            {
                drafts.Add(continuation[generatedSoFar]); // the match this boundary accepts
            }

            drafts.Add(forbidden); // the rejection this boundary stops at
            return drafts;
        });
    }

    private static ScriptedDraftHead EmptyHead() => new(4, _ => []);

    // A sequence stand-in for synthetic-target forwards that never touch KV: every IKvSequence member answers inertly.
    private sealed class NullSequence : IKvSequence
    {
        public static readonly NullSequence Instance = new();

        public int BlockTokens => 1;
        public int Length => 0;
        public int TokenCapacity => 0;
        public IReadOnlyList<int> Blocks => [];

        public SlotRef Append() => throw new InvalidOperationException("the synthetic target must not append positions without a pool.");
        public void MarkAdopted(int physicalBlockId, int startLogicalPos) => throw new NotSupportedException();
        public void SetLength(int length) => throw new NotSupportedException();
        public bool IsAdopted(int logicalPos) => false;
        public int BlockFor(int logicalPos) => -1;
        public void WriteK(int logicalPos, int layer, ReadOnlySpan<float> values) => throw new NotSupportedException();
        public void WriteV(int logicalPos, int layer, ReadOnlySpan<float> values) => throw new NotSupportedException();
        public BlockSelection SelectCausalRead(int queryPosition, int layerWindow) => default;
        public BlockSelection SelectRangeRead(int fromPosition, int toPosition) => default;
        public void Dispose() { }
    }

    // Window arithmetic mirroring how SpeculativeDecoder clamps proposals to remaining generation budget: full windows while
    // budget allows, then one truncated tail; each covered window is one advanced boundary.
    private static int ExpectedWindowedProposalCount(int tokensAfterFirst, int window)
    {
        int proposed = 0;
        int left = tokensAfterFirst;
        while (left > 0)
        {
            proposed += Math.Min(window, left);
            left -= Math.Min(window, left);
        }

        return proposed;
    }

    private static int WindowedStepCount(int tokensAfterFirst, int window)
    {
        int steps = 0;
        int left = tokensAfterFirst;
        while (left > 0)
        {
            left -= Math.Min(window, left);
            steps++;
        }

        return steps;
    }
}
