using Lair.Http;
using Lair.Host.Config;

namespace Lair.Host;

/// <summary>
/// Hot-reload watcher for one served configuration file: a <see cref="FileSystemWatcher"/> with a debounce window
/// turns every edit burst into exactly one reload pass — the coalescing timer restarts on each event, so atomic-save
/// patterns (temp-file write then rename onto the config name) and editors that touch the file repeatedly settle into
/// a single attempt once writes go quiet. Each pass re-reads through the normal loader (full validation), diffs the
/// candidate against what is in force, and dispatches every difference through <see cref="IConfigApplyTarget.ApplyConfiguration"/>
/// by mutation class — whichever shape is serving, one live engine or a multi-model registry. A broken edit never touches
/// what is running: validation failure logs a loud rejection — with per-field refusals naming each changed key's mutation
/// class when the file still parses enough to name them — and the candidate is dropped whole. Applied and refused outcomes
/// are mirrored as structured <c>config_reload field=... class=... outcome=...</c> log lines plus one compact JSON summary
/// line, and every completed attempt lands on the served target's reload summary store for programmatic observation. Only
/// the serve verb wires this; chat and monitor hosts never watch anything.
/// </summary>
public sealed class ConfigWatcher : IDisposable
{
    /// <summary>Silence required before a burst of file events counts as one edit: every event restarts this window.</summary>
    private const int DebounceMs = 250;

    /// <summary>After the debounce window, how long to keep retrying an exclusive open — covers renames still landing and editors holding the file briefly.</summary>
    private const int SettleTimeoutMs = 4_000;

    private const int SettleRetryMs = 100;

    private readonly string _configPath;
    private readonly IConfigApplyTarget _target;
    private readonly Action<string> _logLine;
    private readonly object _stateGate = new();

    private FileSystemWatcher? _watcher;
    private Timer? _debounceTimer;
    private bool _eventsPending;
    private bool _reloadRunning;
    private bool _disposed;

    /// <param name="configPath">The served configuration file to watch.</param>
    /// <param name="engine">The live engine that owns the change-application pipeline and the reload summary store.</param>
    /// <param name="logLine">Sink for each structured log line; defaults to console output. Hosts (and tests) may capture instead.</param>
    public ConfigWatcher(string configPath, GenerationService engine, Action<string>? logLine = null)
        : this(configPath, (IConfigApplyTarget)engine ?? throw new ArgumentNullException(nameof(engine)), logLine)
    {
    }

    /// <param name="configPath">The served configuration file to watch.</param>
    /// <param name="registry">The live model registry whose instances own the change-application pipelines and the reload summary store; every declared model's edits, additions, and removals ride this watcher exactly as a single engine's do.</param>
    /// <param name="logLine">Sink for each structured log line; defaults to console output. Hosts (and tests) may capture instead.</param>
    public ConfigWatcher(string configPath, ModelRegistry registry, Action<string>? logLine = null)
        : this(configPath, (IConfigApplyTarget)registry ?? throw new ArgumentNullException(nameof(registry)), logLine)
    {
    }

    private ConfigWatcher(string configPath, IConfigApplyTarget target, Action<string>? logLine)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(configPath);

        _configPath = Path.GetFullPath(configPath);
        _target = target;
        _logLine = logLine ?? Console.WriteLine;
    }

    /// <summary>Starts watching. The config file's directory is watched (not the bare path) so rename-onto-name atomic saves are seen.</summary>
    public void Start()
    {
        ObjectDisposedException.ThrowIf(_disposed, typeof(ConfigWatcher));

        string? directory = Path.GetDirectoryName(_configPath);
        if (directory is null or { Length: 0 })
        {
            throw new ArgumentException($"'{_configPath}' has no containing directory to watch.");
        }

        FileSystemWatcher watcher = new(directory, Path.GetFileName(_configPath))
        {
            NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.FileName | NotifyFilters.CreationTime | NotifyFilters.Size,
        };

        watcher.Changed += OnFileEvent;
        watcher.Created += OnFileEvent;
        watcher.Renamed += OnFileEvent;
        watcher.Deleted += OnFileEvent;
        watcher.Error += (_, eventArgs) => Log($"config_reload outcome=failed error={Quote(eventArgs.GetException()?.Message ?? "watcher failure")}");

        _debounceTimer = new Timer(_ => OnDebounceFired(), null, Timeout.Infinite, Timeout.Infinite);
        _watcher = watcher; // publish only after the timer exists: an event landing mid-Start still finds its coalescing window armed
        watcher.EnableRaisingEvents = true;
    }

    private void OnFileEvent(object sender, FileSystemEventArgs eventArgs)
    {
        lock (_stateGate)
        {
            if (_disposed)
            {
                return;
            }

            _eventsPending = true;

            // Restart the coalescing window on every event: a burst (or an atomic rename's Created+Changed pair) collapses into one pass once quiet. A pass already running rechecks pending events when it finishes, so nothing arriving mid-reload is lost.
            _debounceTimer?.Change(DebounceMs, Timeout.Infinite);
        }
    }

    private void OnDebounceFired()
    {
        lock (_stateGate)
        {
            if (_disposed || _reloadRunning || !_eventsPending)
            {
                return; // a running pass owns the pending flag and reschedules this window itself when it ends
            }

            _reloadRunning = true;
            _eventsPending = false;
        }

        try
        {
            ReloadPass();
        }
        catch (Exception ex)
        {
            Log($"config_reload outcome=failed error={Quote(ex.Message)}"); // a watcher fault never kills the host loop
        }
        finally
        {
            lock (_stateGate)
            {
                _reloadRunning = false;

                if (_eventsPending && !_disposed)
                {
                    _debounceTimer?.Change(DebounceMs, Timeout.Infinite); // edits landed mid-pass: run again after the next quiet window
                }
            }
        }
    }

    private void ReloadPass()
    {
        if (!TryWaitForSettledRead(out string? settleError))
        {
            Log($"config_reload outcome=rejected error={Quote(settleError ?? "the configuration file stayed unreadable")}");
            _target.ReportReload(ReloadSummary.FromResults([], false, settleError ?? "the configuration file stayed unreadable"));
            return;
        }

        LairConfig candidate = null!; // assigned exactly when validationOk is true
        bool validationOk = true;
        string? validationError = null;
        try
        {
            candidate = YamlConfigLoader.Load(_configPath);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            // Anything the loader (or its YAML layer) surfaces on a broken edit counts as validation failure: loud, structured rejection; nothing downstream of here applies.
            validationOk = false;
            validationError = ex.Message;
        }

        if (!validationOk)
        {
            // Loud rejection, candidate dropped whole: nothing was applied and the running engine keeps every current value.
            // When the file still parsed enough to name its changed keys, each one is reported with its mutation class so a
            // refused restart-only edit says exactly that even though validation rejected it before application.
            IReadOnlyList<FieldResult> refusals = ConfigMutationTable.ExtractEdits(_target.EffectiveConfiguration, _configPath)
                .Select(change => new FieldResult(
                    change,
                    ConfigApplyOutcome.Refused,
                    "not applied: the reloaded configuration failed validation and was dropped whole"))
                .ToList();

            foreach (FieldResult result in refusals)
            {
                Log(FieldLine(result));
            }

            Log($"config_reload outcome=rejected error={Quote(validationError!)}");
            _target.ReportReload(ReloadSummary.FromResults(refusals, false, validationError));
            return;
        }

        IReadOnlyList<FieldResult> results = _target.ApplyConfiguration(candidate);
        foreach (FieldResult result in results)
        {
            Log(FieldLine(result));
        }

        if (results.Count > 0)
        {
            Log($"config_reload_summary={ReloadSummary.FromResults(results, true, null).ToJson()}");
        }

        _target.ReportReload(ReloadSummary.FromResults(results, true, null));
    }

    // Exclusive-open retry: an editor mid-save (or a rename still landing) surfaces as a sharing violation; keep
    // trying until the quiet window's file is genuinely readable or the settle timeout gives up.
    private bool TryWaitForSettledRead(out string? error)
    {
        long deadline = Environment.TickCount64 + SettleTimeoutMs;
        while (true)
        {
            try
            {
                using (FileStream probe = File.Open(_configPath, FileMode.Open, FileAccess.Read, FileShare.None))
                {
                    error = null;
                    return true;
                }
            }
            catch (IOException ex)
            {
                if (Environment.TickCount64 >= deadline)
                {
                    error = $"the configuration file stayed locked: {ex.Message}";
                    return false;
                }

                Thread.Sleep(SettleRetryMs);
            }
        }
    }

    private string FieldLine(FieldResult result) =>
        $"config_reload field={result.Change.Field} class={MutationClassNames.Of(result.Change.MutationClass)} " +
        $"outcome={(result.Outcome == ConfigApplyOutcome.Applied ? "applied" : "refused")} " +
        $"old={Quote(result.Change.OldValue)} new={Quote(result.Change.NewValue)} detail={Quote(result.Detail)}";

    private void Log(string line) => _logLine(line);

    // Values carry raw config text (paths included), so structured lines quote every value and escape embedded quotes.
    private static string Quote(string value) => "\"" + value.Replace("\"", "\\\"") + "\"";

    public void Dispose()
    {
        lock (_stateGate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
        }

        FileSystemWatcher? watcher = Interlocked.Exchange(ref _watcher, null);
        Timer? timer = Interlocked.Exchange(ref _debounceTimer, null);

        if (watcher is not null)
        {
            watcher.EnableRaisingEvents = false; // event handlers see _disposed and drop late callbacks from here on
            watcher.Changed -= OnFileEvent;
            watcher.Created -= OnFileEvent;
            watcher.Renamed -= OnFileEvent;
            watcher.Deleted -= OnFileEvent;
            watcher.Dispose();
        }

        timer?.Dispose();
    }
}
