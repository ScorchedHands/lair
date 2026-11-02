using System.Diagnostics;
using System.Threading.Channels;
using Lair.Kv;
using Lair.Models.Scheduling;
using Microsoft.Extensions.Logging;

namespace Lair.Api.Monitoring;

/// <summary>
/// In-process metrics bus for one engine: cheap scheduler counters plus pool and cache state are sampled at a
/// fixed cadence into immutable <see cref="MetricsSnapshot"/>s, kept newest-first in a bounded ring (oldest drop),
/// fanned out to independent per-subscriber channels that live streams read without ever blocking the publisher,
/// and written as structured JSON log lines. While the engine idles — no sequences active and nothing completed
/// since the previous tick — ticks are quiet: no snapshot, no log line until something moves again; any interval
/// that saw completions publishes even if load ended mid-window, so a burst's tail is never lost from logs or
/// stream history. Collection cost per published tick is a handful of volatile reads plus process facts, and the
/// counters themselves ride step boundaries regardless of consumers: with nobody subscribed this changes nothing
/// measurable about generation paths. The baseline window rolls forward on every tick so each snapshot's rates
/// describe its own interval rather than since engine open.
/// </summary>
public sealed class EngineMetricsBus : IAsyncDisposable
{
    /// <summary>Snapshots retained in the ring; older ones are dropped as newer arrive.</summary>
    public const int RingCapacity = 300;

    private readonly string _modelId;
    private readonly SchedulerStats _stats;
    private readonly BlockPool _pool;
    private readonly PrefixCache? _cache;
    private readonly ILogger<EngineMetricsBus> _logger;
    private readonly TimeSpan _interval;
    private readonly TimeProvider _time;

    // Ring and baseline are touched only by the sampling task (or after it has joined, during dispose).
    private readonly MetricsSnapshot?[] _ring = new MetricsSnapshot?[RingCapacity];
    private long _ringTotal;
    private EngineSample _previous;

    // Request completions that ended on a stop literal — the one finish fact only the facade can see — written by
    // request threads and read as volatile longs at each tick.
    private long _requestsStop;

    // One unbounded channel per live subscriber, dropped when its stream ends or the bus disposes. Writing to an
    // already-completed writer is a no-op, so publishing never blocks on a slow consumer's reader.
    private readonly List<Channel<MetricsSnapshot>> _channels = [];

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

    /// <summary>Records a request that ended on a stop literal, so finish-reason tallies match HTTP outcomes.</summary>
    public void RecordStopCompletion() => Interlocked.Increment(ref _requestsStop);

    /// <summary>
    /// Opens an independent subscription: snapshots published from now on arrive on the returned channel reader.
    /// Callers hand it back through <see cref="Unsubscribe"/> when their stream ends; the bus completes every open
    /// subscription once it disposes.
    /// </summary>
    public ChannelReader<MetricsSnapshot> Subscribe()
    {
        ObjectDisposedException.ThrowIf(_disposed, typeof(EngineMetricsBus));

        Channel<MetricsSnapshot> channel = Channel.CreateUnbounded<MetricsSnapshot>();
        lock (_channels)
        {
            _channels.Add(channel);
        }

        return channel.Reader;
    }

    /// <summary>Completes and forgets a subscription so the bus stops publishing to it.</summary>
    public void Unsubscribe(ChannelReader<MetricsSnapshot> reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        lock (_channels)
        {
            for (int i = 0; i < _channels.Count; i++)
            {
                if (ReferenceEquals(_channels[i].Reader, reader))
                {
                    Channel<MetricsSnapshot> channel = _channels[i];
                    _channels.RemoveAt(i);
                    channel.Writer.TryComplete();
                    return;
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

        lock (_channels)
        {
            foreach (Channel<MetricsSnapshot> channel in _channels)
            {
                channel.Writer.TryComplete();
            }

            _channels.Clear();
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

        long decodeDelta = current.DecodeTokensAdvanced - _previous.DecodeTokensAdvanced;
        long stepsDelta = current.DecodeSteps - _previous.DecodeSteps;
        long occupancySumDelta = current.WindowSequenceSum - _previous.WindowSequenceSum;
        double cpuMs = current.ProcessCpuMs - _previous.ProcessCpuMs;

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
            ProcessCpuPercent: windowMs > 0 ? Math.Round(cpuMs / windowMs * 100.0, 2) : 0.0);

        _ring[_ringTotal % RingCapacity] = snapshot;
        Volatile.Write(ref _ringTotal, _ringTotal + 1);

        lock (_channels)
        {
            foreach (Channel<MetricsSnapshot> channel in _channels)
            {
                channel.Writer.TryWrite(snapshot); // completed writers drop the write; a slow reader never blocks publishing
            }
        }

        _logger.LogInformation("metrics_snapshot={SnapshotJson}", snapshot.ToJson());
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
