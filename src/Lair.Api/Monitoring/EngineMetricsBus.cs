using System.Diagnostics;
using System.Threading.Channels;
using Lair.Kv;
using Lair.Models.Scheduling;
using Microsoft.Extensions.Logging;

namespace Lair.Api.Monitoring;

/// <summary>
/// In-process metrics bus for one engine: cheap scheduler counters plus pool and cache state are sampled at a
/// fixed cadence into immutable <see cref="MetricsSnapshot"/>s, kept newest-first in a bounded ring (oldest drop),
/// fanned out to independent per-subscriber queues that live streams read without ever blocking the publisher,
/// and written as structured JSON log lines. While the engine idles — no sequences active and nothing completed
/// since the previous tick — ticks are quiet: no snapshot, no log line until something moves again; any interval
/// that saw completions publishes even if load ended mid-window, so a burst's tail is never lost from logs or
/// stream history. Collection cost per published tick is a handful of volatile reads plus process facts, and the
/// counters themselves ride step boundaries regardless of consumers: with nobody subscribed this changes nothing
/// measurable about generation paths. The baseline window rolls forward on every tick so each snapshot's rates
/// describe its own interval rather than since engine open.
/// </summary>
/// <remarks>
/// Every stream event — snapshots, request lifecycle frames, and resync dumps alike — carries one server-scoped
/// id from a single incrementing counter: the SSE frame's <c>id:</c>, also embedded in snapshot payloads as
/// <c>metrics_event_seq</c>. Ids are assigned under the fan-out lock so publish order matches id order, and each
/// event kind keeps its own per-subscriber queue. A slow consumer never blocks publishing: snapshots coalesce to a
/// latest-pending slot (the subscriber only ever sees the newest one), request frames ride a bounded queue whose
/// oldest entries drop under pressure — both counted in <c>metrics_dropped_events_total</c>. Completed requests
/// also land in a recent-request ring sized by the live <see cref="SetRecentRequestCapacity"/> knob. Ids survive
/// coalescing: an id can vanish from delivery while newer ids flow, so reconnect replay treats any header id at
/// or above the newest known event as already caught up (live tail only) instead of trying to re-deliver it.
/// </remarks>
public sealed class EngineMetricsBus : IAsyncDisposable
{
    /// <summary>Snapshots retained in the ring; older ones are dropped as newer arrive.</summary>
    public const int RingCapacity = 300;

    /// <summary>Largest recent-request ring capacity the configuration may ask for (the loader enforces the same bound).</summary>
    public const int MaxRecentRequestsCapacity = 1_000;

    /// <summary>Request frames queued per subscriber before the oldest drop.</summary>
    internal const int RequestQueueCapacity = 200;

    private readonly string _modelId;
    private readonly SchedulerStats _stats;
    private readonly BlockPool _pool;
    private readonly PrefixCache? _cache;
    private readonly ILogger<EngineMetricsBus> _logger;
    private readonly TimeSpan _interval;
    private readonly TimeProvider _time;

    // Ring and baseline are touched only by the sampling task (or after it has joined, during dispose).
    private readonly MetricsSnapshot?[] _ring = new MetricsSnapshot?[RingCapacity];
    private readonly long[] _ringIds = new long[RingCapacity];
    private long _ringTotal;
    private EngineSample _previous;

    // Request completions that ended on a stop literal — the one finish fact only the facade can see — written by
    // request threads and read as volatile longs at each tick.
    private long _requestsStop;

    // Server-scoped event id: every frame this bus delivers takes the next value, so Last-Event-ID names one
    // point in one total stream regardless of which subscriber or event kind produced it. Assigned under the
    // fan-out lock together with delivery, keeping publish order and id order aligned.
    private long _eventSeq;

    // Fan-out serialization: snapshot publishing (sampler thread) and request lifecycle publishing (request
    // threads) share one short critical section — assign id, store to rings, hand frames to subscriber queues.
    // Queue writes never block (coalescing slot + bounded drop-oldest channels), so the lock is held only for
    // bookkeeping even when a consumer stalls.
    private readonly object _fanout = new();

    // Recent completed requests: newest-first ring sized by the live ui.recent_requests knob. Shrinking just
    // narrows what reads see (oldest entries drop); growing admits more again, re-seating only the retained tail.
    private RequestPayload[] _recentRequests;
    private int _recentCapacity = 25;
    private long _recentTotal;

    // Live in-flight request views, pulled from the owning facade when a resync is built. Null until attached —
    // hosts that never wire a request source simply replay empty in-flight sets.
    private Func<IReadOnlyList<RequestPayload>>? _inFlightProvider;

    // Stream frames dropped at subscriber queues since open (slow consumers); surfaced on every snapshot.
    private long _droppedEvents;

    // Model facts attached by the owning service; null until then, so snapshots carry nulls in those slots — the
    // v1 shape untouched. Re-pushed on hot reload when a live-applied field changes them (worker quota today).
    private ModelFacts? _modelFacts;

    private readonly List<MonitorSubscription> _subscriptions = [];
    private long _subscriptionSeq;

    private readonly Task _sampler;
    private bool _disposed;

    public EngineMetricsBus(
        string modelId,
        GenerationScheduler scheduler,
        BlockPool pool,
        PrefixCache? cache,
        ILoggerFactory loggerFactory,
        TimeSpan interval,
        TimeProvider? timeProvider = null)
    {
        ArgumentNullException.ThrowIfNull(modelId);
        ArgumentNullException.ThrowIfNull(scheduler);
        ArgumentNullException.ThrowIfNull(pool);
        ArgumentNullException.ThrowIfNull(loggerFactory);
        ArgumentOutOfRangeException.ThrowIfLessThan(interval.TotalMilliseconds, 25); // guards against absurd cadences; snapshots are not meant to fire per-token

        _modelId = modelId;
        _stats = scheduler.Stats;
        _pool = pool;
        _cache = cache;
        _interval = interval;
        _time = timeProvider ?? TimeProvider.System;
        _logger = loggerFactory.CreateLogger<EngineMetricsBus>();
        _recentRequests = new RequestPayload[_recentCapacity];

        EngineSample baseline = Sample();
        _previous = baseline with { TimestampMs = _time.GetUtcNow().ToUnixTimeMilliseconds() };

        _sampler = Task.Factory.StartNew(
            SampleLoopAsync,
            CancellationToken.None,
            TaskCreationOptions.LongRunning | TaskCreationOptions.PreferFairness,
            TaskScheduler.Default).Unwrap();
    }

    /// <summary>How many snapshots the ring currently holds (at most <see cref="RingCapacity"/>).</summary>
    public long RingCount => Math.Min(Volatile.Read(ref _ringTotal), RingCapacity);

    /// <summary>Total snapshots published since the bus opened — larger than the ring count once it has wrapped.</summary>
    public long PublishedTotal => Volatile.Read(ref _ringTotal);

    /// <summary>The retained snapshots, newest first.</summary>
    public IReadOnlyList<MetricsSnapshot> SnapshotRing()
    {
        List<MetricsSnapshot> list = [];
        for (long index = _ringTotal - 1; index >= Math.Max(0, _ringTotal - RingCapacity); index--)
        {
            if (_ring[index % RingCapacity] is MetricsSnapshot snapshot)
            {
                list.Add(snapshot);
            }
        }

        return list;
    }

    /// <summary>How many completed requests the recent-request ring currently holds (at most its live capacity). Shrinks drop entries permanently, so this counts what is actually retained rather than a total-and-capacity bound.</summary>
    public int RecentRequestCount
    {
        get
        {
            int capacity = Volatile.Read(ref _recentCapacity);
            long floor = Math.Max(0, Volatile.Read(ref _recentTotal) - capacity);

            int held = 0;
            for (long index = floor; index < Volatile.Read(ref _recentTotal); index++)
            {
                if (_recentRequests[index % capacity] is not null)
                {
                    held++;
                }
            }

            return held;
        }
    }

    /// <summary>The retained completed requests, newest first — what a resync hands to a reconnecting client.</summary>
    public IReadOnlyList<RequestPayload> RecentRequests()
    {
        int capacity = Volatile.Read(ref _recentCapacity);
        List<RequestPayload> list = new((int)Math.Min(Volatile.Read(ref _recentTotal), (long)capacity));
        for (long index = _recentTotal - 1; index >= Math.Max(0, _recentTotal - capacity); index--)
        {
            if (_recentRequests[index % capacity] is RequestPayload request)
            {
                list.Add(request);
            }
        }

        return list;
    }

    /// <summary>Stream frames dropped at subscriber queues since the bus opened — how often slow consumers lost events.</summary>
    public long DroppedEventsTotal => Volatile.Read(ref _droppedEvents);

    /// <summary>The highest event id any frame this bus has delivered carries (0 before the first).</summary>
    public long LatestEventId => Volatile.Read(ref _eventSeq);

    /// <summary>The id of the oldest snapshot still in the ring — 0 when the ring holds none. Reconnect ids below it predate retained history.</summary>
    public long OldestRetainedSnapshotId => _ringTotal == 0
        ? 0
        : _ringIds[Math.Max(0, _ringTotal - RingCapacity) % RingCapacity];

    /// <summary>Records a request that ended on a stop literal, so finish-reason tallies match HTTP outcomes.</summary>
    public void RecordStopCompletion() => Interlocked.Increment(ref _requestsStop);

    /// <summary>
    /// Resizes the recent-request ring live. Shrinking drops the oldest entries by simply narrowing what reads
    /// see; growing re-seats only the retained tail into a wider buffer so older completed requests admit again.
    /// Out-of-range capacities clamp to [1, <see cref="MaxRecentRequestsCapacity"/>] — the loader already rejects
    /// such values before they reach here.
    /// </summary>
    public void SetRecentRequestCapacity(int capacity)
    {
        if (capacity is < 1 or > MaxRecentRequestsCapacity)
        {
            throw new ArgumentOutOfRangeException(
                nameof(capacity),
                $"recent-request ring capacity must be between 1 and {MaxRecentRequestsCapacity}, got {capacity}.");
        }

        lock (_fanout)
        {
            if (capacity == _recentCapacity)
            {
                return;
            }

            int retained = (int)Math.Min(_recentTotal, Math.Min((long)capacity, _recentCapacity));
            RequestPayload[] grown = new RequestPayload[capacity];
            for (int i = 0; i < retained; i++)
            {
                // Re-seat each logical entry at its ring position under the new capacity: reads pair monotonic totals
                // with `index % capacity`, so a resize must land every retained payload exactly where that walk expects it.
                long logicalIndex = _recentTotal - retained + i;
                grown[(int)(logicalIndex % capacity)] = _recentRequests[logicalIndex % _recentCapacity];
            }

            _recentRequests = grown;
            Volatile.Write(ref _recentCapacity, capacity); // published last: readers pair the new array with the new size
        }
    }

    /// <summary>
    /// Wires the source of live in-flight request views used when a resync payload is built. The owning facade
    /// attaches it alongside its lifecycle publishing; without one, resyncs report an empty in-flight set.
    /// </summary>
    public void AttachInFlightProvider(Func<IReadOnlyList<RequestPayload>> provider)
    {
        ArgumentNullException.ThrowIfNull(provider);
        Volatile.Write(ref _inFlightProvider, provider);
    }

    /// <summary>
    /// Attaches — or re-places, e.g. after a hot reload moved worker participation — the configuration and backend
    /// facts embedded additively in every snapshot from now on. One volatile record swap; sampling reads it once per tick.
    /// </summary>
    public void AttachModelFacts(ModelFacts facts)
    {
        ArgumentNullException.ThrowIfNull(facts);
        Volatile.Write(ref _modelFacts, facts);
    }

    /// <summary>
    /// Opens an independent subscription: snapshots published from now on coalesce into the subscriber's latest-
    /// pending slot (a slow reader only ever sees the newest one) and request frames queue on a bounded channel.
    /// Callers hand it back through <see cref="Unsubscribe"/> when their stream ends; the bus completes every open
    /// subscription once it disposes.
    /// </summary>
    public MonitorSubscription Subscribe()
    {
        ObjectDisposedException.ThrowIf(_disposed, typeof(EngineMetricsBus));

        MonitorSubscription subscription = new(Interlocked.Increment(ref _subscriptionSeq));
        lock (_subscriptions)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(EngineMetricsBus));
            }

            _subscriptions.Add(subscription);
        }

        return subscription;
    }

    /// <summary>Completes and forgets a subscription so the bus stops publishing to it.</summary>
    public void Unsubscribe(MonitorSubscription subscription)
    {
        ArgumentNullException.ThrowIfNull(subscription);
        lock (_subscriptions)
        {
            if (_subscriptions.Remove(subscription))
            {
                subscription.Close();
            }
        }
    }

    /// <summary>Builds one resync payload: the live in-flight request set plus the recent completed-request ring.</summary>
    public ResyncPayload BuildResync() => new(
        MonitorEventSchema.Version,
        Volatile.Read(ref _inFlightProvider) is Func<IReadOnlyList<RequestPayload>> provider ? provider() : [],
        RecentRequests());

    /// <summary>
    /// The snapshot frames in the ring with ids strictly newer than <paramref name="lastEventId"/>, oldest first —
    /// what a reconnecting client missed. Replay answers only from snapshots, so an id that names any non-snapshot
    /// frame (a request or resync event) can never match: callers treat "no frames" as the signal to fall back to
    /// a fresh <c>resync</c>. An empty result also means nothing was missed when the header already sits at or
    /// above the newest known event id — see <see cref="LatestEventId"/>.
    /// </summary>
    public IReadOnlyList<MonitorFrame> SnapshotReplayAfter(long lastEventId)
    {
        List<MonitorFrame> frames = [];
        for (long index = Math.Max(0, _ringTotal - RingCapacity); index < _ringTotal; index++)
        {
            long id = _ringIds[index % RingCapacity];
            if (id > lastEventId && _ring[index % RingCapacity] is MetricsSnapshot snapshot)
            {
                frames.Add(new MonitorFrame(id, MonitorEventNames.Snapshot, snapshot.ToJson()));
            }
        }

        return frames;
    }

    /// <summary>
    /// Publishes one request lifecycle frame (a <see cref="MonitorEventNames"/> event name with its payload) to every
    /// subscriber's bounded request queue. A completed frame additionally lands in the recent-request ring before
    /// fan-out, so a client reconnecting right after sees it through resync even if live delivery had dropped it.
    /// </summary>
    public void PublishRequest(string eventName, RequestPayload payload)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(eventName);
        ArgumentNullException.ThrowIfNull(payload);

        MonitorFrame frame;
        lock (_fanout)
        {
            long id = ++_eventSeq;
            frame = new MonitorFrame(id, eventName, payload.ToJson());

            if (eventName == MonitorEventNames.RequestComplete)
            {
                AppendRecentRequest(payload);
            }

            foreach (MonitorSubscription subscription in _subscriptions)
            {
                if (!subscription.PublishRequest(frame))
                {
                    Interlocked.Increment(ref _droppedEvents); // oldest queued frame goes out to make room
                }
            }
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        // The sampler exits after its current delay, then a closing tick captures any burst that ended mid-window.
        await _sampler.ConfigureAwait(false);
        FlushFinal();

        lock (_subscriptions)
        {
            foreach (MonitorSubscription subscription in _subscriptions)
            {
                subscription.Close();
            }

            _subscriptions.Clear();
        }
    }

    private async Task SampleLoopAsync()
    {
        while (!_disposed)
        {
            await Task.Delay(_interval, CancellationToken.None).ConfigureAwait(false);
            if (_disposed)
            {
                return;
            }

            try
            {
                Tick();
            }
            catch (Exception ex)
            {
                // Observability must never take the host down with it.
                _logger.LogError(ex, "metrics sampling tick failed");
            }
        }
    }

    private void Tick()
    {
        EngineSample current = Sample();
        long nowMs = _time.GetUtcNow().ToUnixTimeMilliseconds();

        if (IsActive(current))
        {
            PublishSnapshot(current, nowMs);
        }

        // The window rolls forward on every tick — quiet ones included — so rates always describe the interval since
        // the previous check instead of accumulating idle stretches into a diluted average.
        _previous = current with { TimestampMs = nowMs };
    }

    private void FlushFinal()
    {
        EngineSample current = Sample();
        long nowMs = _time.GetUtcNow().ToUnixTimeMilliseconds();
        if (IsActive(current))
        {
            PublishSnapshot(current, nowMs);
        }
    }

    // Busy means sequences are advancing right now or something completed since the previous tick — a burst that
    // ended between ticks still gets its closing snapshot. Otherwise the engine is idle and stays quiet.
    private bool IsActive(EngineSample current) =>
        Volatile.Read(ref _stats.ActiveSequenceCount) > 0 || CurrentsDiffer(current, _previous);

    private static bool CurrentsDiffer(EngineSample a, EngineSample b) =>
        a.DecodeTokensAdvanced != b.DecodeTokensAdvanced
        || a.FinishedEos != b.FinishedEos
        || a.FinishedLength != b.FinishedLength
        || a.FinishedCanceled != b.FinishedCanceled
        || a.FinishedFailed != b.FinishedFailed;

    private void PublishSnapshot(EngineSample current, long nowMs)
    {
        double windowMs = Math.Max(1, nowMs - _previous.TimestampMs);
        ModelFacts? facts = Volatile.Read(ref _modelFacts);

        long decodeDelta = current.DecodeTokensAdvanced - _previous.DecodeTokensAdvanced;
        long stepsDelta = current.DecodeSteps - _previous.DecodeSteps;
        long occupancySumDelta = current.WindowSequenceSum - _previous.WindowSequenceSum;
        double cpuMs = current.ProcessCpuMs - _previous.ProcessCpuMs;

        MonitorFrame frame;
        lock (_fanout)
        {
            MetricsSnapshot snapshot = new(
                IsoUtc(nowMs),
                (long)Math.Round(windowMs),
                Model: _modelId,
                TokensPerSecDecodeWindowed: Math.Round(decodeDelta * 1000.0 / windowMs, 2),
                AvgStepTimeMsWindowed: stepsDelta > 0 ? Math.Round((current.DecodeWindowMicrosTotal - _previous.DecodeWindowMicrosTotal) / (double)stepsDelta / 1000.0, 3) : 0.0,
                DecodeStepsTotal: current.DecodeSteps,
                BatchOccupancyAvgSequencesPerStep: stepsDelta > 0 ? Math.Round(occupancySumDelta / (double)stepsDelta, 2) : 0.0,
                PrefillChunksTotal: current.PrefillChunksIssued,
                PrefillTokensForwardedTotal: current.PrefillTokensForwarded,

                CacheHitSequencesTotal: current.CacheHitSequences,
                CacheSkippedTokensTotal: current.CacheSkippedTokens,

                KvUsedBlocks: _pool.UsedBlocks,
                KvCapacityBlocks: _pool.CapacityBlocks,
                KvCachedBlocks: _cache?.CachedBlocks ?? 0,
                ActiveSequences: Volatile.Read(ref _stats.ActiveSequenceCount),

                RequestsTotal: current.FinishedEos + current.FinishedLength + current.FinishedCanceled + current.FinishedFailed + Volatile.Read(ref _requestsStop),
                RequestsEosTotal: current.FinishedEos,
                RequestsLengthTotal: current.FinishedLength,
                RequestsStopTotal: Volatile.Read(ref _requestsStop),
                RequestsCancelledTotal: current.FinishedCanceled,
                RequestsFailedTotal: current.FinishedFailed,

                ProcessWorkingSetMb: Math.Round(current.WorkingSetMb, 1),
                ProcessCpuPercent: windowMs > 0 ? Math.Round(cpuMs / windowMs * 100.0, 2) : 0.0,
                EventsVersion: MonitorEventSchema.Version,
                EventSeq: _eventSeq + 1, // the id this snapshot is about to take — embedded so payloads and SSE frames agree
                DroppedEventsTotal: Volatile.Read(ref _droppedEvents),

                // Additive operator-view fields (v1 shape untouched): pool geometry always known here; identity and
                // placement ride along only once the owning service has attached facts.
                ModelType: facts?.Type,
                KvBlockTokens: _pool.BlockTokens,
                KvBudgetBlocks: _pool.EffectiveBudgetBlocks,
                KvPoolMb: Math.Round(_pool.BytesPerBlock * (double)_pool.CapacityBlocks / (1024.0 * 1024.0), 1),
                WorkerThreads: facts?.WorkerThreads,
                CoreSet: facts?.CoreSet,
                ThreadQuota: facts?.ThreadQuota,
                DecodeDevice: facts?.DecodeDevice,
                PrefillDevice: facts?.PrefillDevice,
                KvResidency: facts?.KvResidency);

            long id = ++_eventSeq;
            frame = new MonitorFrame(id, MonitorEventNames.Snapshot, snapshot.ToJson());

            _ring[_ringTotal % RingCapacity] = snapshot;
            _ringIds[_ringTotal % RingCapacity] = id;
            Volatile.Write(ref _ringTotal, _ringTotal + 1);

            foreach (MonitorSubscription subscription in _subscriptions)
            {
                if (!subscription.PublishSnapshotCoalesced(frame))
                {
                    Interlocked.Increment(ref _droppedEvents); // a newer snapshot replaced the one this subscriber had not read yet
                }
            }
        }

        _logger.LogInformation("metrics_snapshot={SnapshotJson}", frame.Json);
    }

    // Completed payloads land in the recent ring as the same immutable payload objects that were published, so a
    // resync hands out exactly what live delivery carried. The ring's capacity bound (live knob) is what trims them.
    private void AppendRecentRequest(RequestPayload payload)
    {
        int capacity = _recentCapacity;
        _recentRequests[_recentTotal % capacity] = payload;
        Volatile.Write(ref _recentTotal, _recentTotal + 1);
    }

    private EngineSample Sample()
    {
        Process process = Process.GetCurrentProcess();
        return new EngineSample(
            Volatile.Read(ref _stats.DecodeSteps),
            Volatile.Read(ref _stats.DecodeWindowMicrosTotal),
            Volatile.Read(ref _stats.WindowSequenceSum),
            Volatile.Read(ref _stats.PrefillChunksIssued),
            Volatile.Read(ref _stats.PrefillTokensForwarded),
            Volatile.Read(ref _stats.DecodeTokensAdvanced),
            Volatile.Read(ref _stats.CacheHitSequences),
            Volatile.Read(ref _stats.CacheSkippedTokens),
            Volatile.Read(ref _stats.FinishedEos),
            Volatile.Read(ref _stats.FinishedLength),
            Volatile.Read(ref _stats.FinishedCanceled),
            Volatile.Read(ref _stats.FinishedFailed))
        {
            TimestampMs = 0,
            WorkingSetMb = process.WorkingSet64 / (1024.0 * 1024.0),

            // CPU time through System.Diagnostics only; OS- and hardware-level telemetry beyond working set and
            // process CPU share is out of scope for this surface.
            ProcessCpuMs = process.TotalProcessorTime.TotalMilliseconds,
        };
    }

    private static string IsoUtc(long unixMs) => DateTimeOffset.FromUnixTimeMilliseconds(unixMs).ToString("yyyy-MM-ddTHH:mm:ss.fffZ");

    // One volatile read per counter plus the process facts, taken as a unit for windowed deltas.
    private readonly record struct EngineSample(
        long DecodeSteps,
        long DecodeWindowMicrosTotal,
        long WindowSequenceSum,
        long PrefillChunksIssued,
        long PrefillTokensForwarded,
        long DecodeTokensAdvanced,
        long CacheHitSequences,
        long CacheSkippedTokens,
        long FinishedEos,
        long FinishedLength,
        long FinishedCanceled,
        long FinishedFailed)
    {
        public long TimestampMs { get; init; }

        public double WorkingSetMb { get; init; }

        public double ProcessCpuMs { get; init; }
    }
}
