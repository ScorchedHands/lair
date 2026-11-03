using System.Diagnostics;

namespace Lair.Http.Monitoring;

/// <summary>
/// Per-generation tok/s tracker for the monitor stream: one fixed ring of step-boundary elapsed-time samples (no
/// per-token allocation — a stopwatch tick and two array writes ride the token hook that already runs on every
/// generated token), plus throttled lifecycle publishing to the metrics bus. A sequence's <c>request_update</c>
/// frames fire at its first sampled token and then every eighth, so short requests still show progress while long
/// ones stay event-cheap; completion always publishes a final frame carrying the full retained series. Sample
/// points record each Nth token's total elapsed milliseconds since generation opened — sparkline-ready pairs whose
/// latest point is the request's current tok/s rate. Series are newest-last over a fixed window (64 samples): older
/// points drop first, so long generations report recent speed rather than their whole history. Views handed to
/// resyncs read only volatile fields, never the sequence itself.
/// </summary>
internal sealed class RequestTracker
{
    /// <summary>Tok/s sample points retained per request; beyond it the oldest point drops.</summary>
    internal const int SampleCapacity = 64;

    /// <summary>Lifecycle update cadence: every Nth generated token (the first always publishes).</summary>
    private const int UpdateEveryTokens = 8;

    private readonly EngineMetricsBus _bus;
    private readonly string _requestId;
    private readonly string _model;
    private readonly string _endpoint;
    private readonly int _promptTokens;
    private readonly long _startTick = Stopwatch.GetTimestamp();

    // Ring of elapsed-milliseconds values, indexed by (token count - 1) % capacity: the token index is implied by
    // position against the volatile generated total, so each point stays a single stored number. Written only from
    // the scheduler-thread token hook; volatile reads let resync views see a consistent-enough window without
    // locking it out.
    private readonly long[] _elapsedMs = new long[SampleCapacity];

    private int _generated; // generated-so-far count (volatile)

    // Start-frame serialization: the facade publishes start right after Submit while the scheduler thread's first
    // token hook may reach EnsureStarted from the other side. The gate makes exactly one start frame win, and its
    // bus publish completes inside the critical section, so every update or completion — whichever thread emits it —
    // enters the bus only after that start frame did. Per-request cost only; once started, later updates read the
    // flag volatile and skip the lock entirely.
    private readonly object _startGate = new();

    private int _started; // 0/1 under _startGate: exactly one start frame per request on the stream

    public RequestTracker(EngineMetricsBus bus, string requestId, string model, string endpoint, int promptTokens)
    {
        ArgumentNullException.ThrowIfNull(bus);
        ArgumentException.ThrowIfNullOrWhiteSpace(requestId);
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(endpoint);

        _bus = bus;
        _requestId = requestId;
        _model = model;
        _endpoint = endpoint;
        _promptTokens = promptTokens;
    }

    public string RequestId => _requestId;

    /// <summary>Token hook body: sample one step-boundary point, then publish a throttled update once the start frame is out.</summary>
    public void ObserveToken()
    {
        int count = Volatile.Read(ref _generated) + 1;

        long elapsedMs = ElapsedMs(_startTick);
        _elapsedMs[(count - 1) % SampleCapacity] = elapsedMs; // linear fill until wrap-around begins, then oldest point is overwritten by the newest

        Volatile.Write(ref _generated, count);

        if (count == 1 || count % UpdateEveryTokens == 0)
        {
            EnsureStarted(); // the token hook can beat Submit's start frame; ordering is repaired here so a subscriber never sees an update before its request exists on the stream.
            _bus.PublishRequest(MonitorEventNames.RequestUpdate, BuildPayload("running", null, null));
        }
    }

    /// <summary>Publishes the admitted-request start frame: id, model, endpoint kind, and prompt size are known from here on. Idempotent — a token hook that beat this call already published it.</summary>
    public void PublishStart()
    {
        lock (_startGate)
        {
            if (_started != 0)
            {
                return; // exactly one start frame per request, whichever thread got there first
            }

            _started = 1;
            _bus.PublishRequest(MonitorEventNames.RequestStart, BuildPayload("running", null, null));
        }
    }


    /// <summary>Publishes the final lifecycle frame — status plus finish reason, cached tokens, and the full retained sample series.</summary>
    public RequestPayload Complete(string status, int cachedTokens, string? finishReason)
    {
        EnsureStarted(); // a generation that never produced an update token still gets its start frame ahead of completion

        RequestPayload payload = BuildPayload(status, cachedTokens, finishReason);
        _bus.PublishRequest(MonitorEventNames.RequestComplete, payload);
        return payload;
    }

    /// <summary>The current in-flight view of this request — volatile reads only, used when a resync state dump is built.</summary>
    public RequestPayload View() => BuildPayload("running", null, null);

    private void EnsureStarted()
    {
        if (Volatile.Read(ref _started) == 0)
        {
            PublishStart(); // idempotent through the gate: a racing pair still yields one start frame
        }
    }

    private RequestPayload BuildPayload(string status, int? cachedTokens, string? finishReason)
    {
        int generated = Volatile.Read(ref _generated);
        IReadOnlyList<TokenSample>? samples = BuildSamples(generated);

        double? tokensPerSec = null;
        if (generated > 0 && samples is not null && samples.Count > 0)
        {
            long lastElapsedMs = samples[^1].ElapsedMs;
            if (lastElapsedMs > 0)
            {
                tokensPerSec = Math.Round(generated * 1000.0 / lastElapsedMs, 2);
            }
        }

        return new RequestPayload(
            MonitorEventSchema.Version,
            _requestId,
            _model,
            _endpoint,
            _promptTokens,
            status,
            GeneratedTokens: generated,
            ElapsedMs: ElapsedMs(_startTick),
            TokensPerSec: tokensPerSec,
            CachedTokens: cachedTokens,
            FinishReason: finishReason,
            Samples: samples);
    }

    // Chronological oldest→newest sample points over the retained window: token indices run from max(1, generated -
    // capacity + 1) through the volatile total, each read straight from its implied ring slot. A reader racing a mid-
    // wrap write may catch one torn point — transient views and stream frames tolerate that, completion payloads are
    // built after the sequence has stopped advancing.
    private IReadOnlyList<TokenSample>? BuildSamples(int generated)
    {
        if (generated == 0)
        {
            return null;
        }

        int oldestToken = Math.Max(1, generated - SampleCapacity + 1); // first token still inside the retained window

        List<TokenSample> samples = new(generated - oldestToken + 1);
        for (int atToken = oldestToken; atToken <= generated; atToken++)
        {
            samples.Add(new TokenSample(atToken, _elapsedMs[(atToken - 1) % SampleCapacity]));
        }

        return samples;
    }

    private static long ElapsedMs(long startTick) => (Stopwatch.GetTimestamp() - startTick) * 1000 / Stopwatch.Frequency;
}
