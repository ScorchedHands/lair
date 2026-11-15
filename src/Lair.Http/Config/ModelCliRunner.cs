using System.Globalization;

namespace Lair.Host.Config;

/// <summary>
/// The <c>lair model</c> CLI surface behind one testable runner: <c>fetch</c> lands a declared file set into its
/// pack directory through the transport seam and records the pin, no-oping on an already verified final file so a
/// repeated fetch never double-streams; <c>check</c> reports candidate deltas against the hub's current HEAD —
/// revision, per-file hash and size, and the file list — without downloading, repinning, or touching a manifest or
/// the load path; <c>list</c> prints what a pack pins. Exit codes are bounded: 0 when the command did its job (a
/// candidate delta is still a completed check), 1 when a fetch refused, a pin disagreed with what was asked for, or
/// HEAD could not be consulted, and 2 when the arguments do not form a usable command.
/// </summary>
public sealed class ModelCliRunner
{
    /// <summary>The command completed its job.</summary>
    public const int ExitSuccess = 0;

    /// <summary>A fetch refused, a pin disagreed with what was asked for, or the hub could not be consulted.</summary>
    public const int ExitRefused = 1;

    /// <summary>The arguments do not form a usable command.</summary>
    public const int ExitUsage = 2;

    private readonly IModelHubTransport _transport;
    private readonly string _packDirectory;
    private readonly TextWriter _output;
    private readonly LairConfig? _config;

    /// <summary>Builds the runner against one pack directory — the store used when no config names model paths — an output writer, and optionally a loaded config whose model entries name their own packs.</summary>
    public ModelCliRunner(IModelHubTransport transport, string packDirectory, TextWriter output, LairConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(transport);
        ArgumentException.ThrowIfNullOrWhiteSpace(packDirectory);
        ArgumentNullException.ThrowIfNull(output);

        _transport = transport;
        _packDirectory = packDirectory;
        _output = output;
        _config = config;
    }

    /// <summary>Runs one parsed argument tail — everything after the <c>model</c> verb — and returns the process exit code.</summary>
    public async Task<int> RunAsync(IReadOnlyList<string> args, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(args);

        try
        {
            return args.Count switch
            {
                0 => Usage(),
                _ when args[0] == "fetch" => await FetchAsync(args, cancellationToken),
                _ when args[0] == "check" => await CheckAsync(args, cancellationToken),
                _ when args[0] == "list" => await ListManifests(args),
                _ => Usage(),
            };
        }
        catch (ModelFetchException ex)
        {
            // The fetch path already refuses naming repo, revision, and file; the CLI reports it as one actionable line.
            await _output.WriteLineAsync($"error: {ex.Message}");
            return ExitRefused;
        }
        catch (ConfigException ex)
        {
            await _output.WriteLineAsync($"error: {ex.Message}");
            return ExitRefused;
        }
        catch (Exception ex) when (ex is HttpRequestException or IOException or UnauthorizedAccessException)
        {
            // A hub failure while reporting HEAD refuses the check naming what was consulted — never a substitute revision.
            await _output.WriteLineAsync($"error: {ex.Message}");
            return ExitRefused;
        }
    }

    private int Usage()
    {
        _output.WriteLine("usage:");
        _output.WriteLine("  lair model fetch <repo> [--revision <rev>] [--file <path>]...   fetch the declared files into their pack");
        _output.WriteLine("  lair model check [model]                                        report candidate deltas against hub HEAD");
        _output.WriteLine("  lair model list                                                 print what a pack pins");
        return ExitUsage;
    }

    private async Task<int> FetchAsync(IReadOnlyList<string> args, CancellationToken cancellationToken)
    {
        string? repo = null;
        string? revision = null;
        string? model = null;
        List<string> files = [];

        for (int i = 1; i < args.Count; i++)
        {
            switch (args[i])
            {
                case "--revision":
                    if (i + 1 >= args.Count || revision is not null)
                    {
                        await _output.WriteLineAsync("error: fetch --revision needs exactly one value");
                        return ExitUsage;
                    }

                    revision = args[++i];
                    break;

                case "--file":
                    if (i + 1 >= args.Count || string.IsNullOrWhiteSpace(args[i + 1]))
                    {
                        await _output.WriteLineAsync("error: fetch --file needs a path");
                        return ExitUsage;
                    }

                    files.Add(args[++i]);
                    break;

                case "--model":
                    if (i + 1 >= args.Count || model is not null)
                    {
                        await _output.WriteLineAsync("error: fetch --model needs exactly one value");
                        return ExitUsage;
                    }

                    model = args[++i];
                    break;

                default:
                    if (args[i].StartsWith("-", StringComparison.Ordinal))
                    {
                        await _output.WriteLineAsync($"error: unknown fetch option '{args[i]}'");
                        return ExitUsage;
                    }

                    if (repo is not null)
                    {
                        await _output.WriteLineAsync($"error: fetch takes exactly one repo; unexpected argument '{args[i]}'");
                        return ExitUsage;
                    }

                    repo = args[i];
                    break;
            }
        }

        if (repo is null)
        {
            await _output.WriteLineAsync("error: fetch needs a repo to fetch from");
            return ExitUsage;
        }

        ModelConfig? declared = null;
        if (_config is not null)
        {
            // With a config in hand, fetch only ever fetches what a model entry declares: an undeclared repo or
            // name refuses rather than widening the pin to whatever the operator typed.
            declared = FindDeclaredModel(model, repo);
            if (declared is null)
            {
                if (model is not null)
                {
                    await _output.WriteLineAsync($"error: no config model '{model}' declares a source for repo '{repo}'.");
                    return ExitRefused;
                }

                List<ModelConfig> candidates = [.. _config.Models.Where(candidate => candidate.Source is { } source && string.Equals(source.Repo, repo, StringComparison.Ordinal))];
                if (candidates.Count > 1)
                {
                    await _output.WriteLineAsync($"error: repo '{repo}' is declared by multiple models ({string.Join(", ", candidates.Select(candidate => $"'{candidate.Name}'"))}); pass --model to pick one.");
                    return ExitRefused;
                }

                await _output.WriteLineAsync($"error: no config model declares a source for repo '{repo}'.");
                return ExitRefused;
            }
        }

        string packDirectory;
        ModelSourceConfig source;
        string modelName;

        if (declared is not null)
        {
            ModelSourceConfig pinned = declared.Source!;

            if (revision is not null && !string.Equals(revision, pinned.Revision, StringComparison.Ordinal))
            {
                await _output.WriteLineAsync($"error: model '{declared.Name}' pins repo '{pinned.Repo}' at revision '{pinned.Revision}'; refusing to fetch it at '{revision}'.");
                return ExitRefused;
            }

            List<SourceFileConfig> declaredFiles = [.. pinned.Files];
            if (files.Count > 0)
            {
                List<string> undeclared = [.. files.Where(requested => !declaredFiles.Any(known => string.Equals(known.Path, requested, StringComparison.Ordinal)))];
                if (undeclared.Count > 0)
                {
                    await _output.WriteLineAsync($"error: {string.Join(", ", undeclared.Select(f => $"'{f}'"))} is not declared by model '{declared.Name}' in repo '{pinned.Repo}'.");
                    return ExitRefused;
                }

                declaredFiles = [.. declaredFiles.Where(known => files.Any(requested => string.Equals(known.Path, requested, StringComparison.Ordinal)))];
            }

            packDirectory = PackDirectoryOf(declared);
            source = new ModelSourceConfig { Repo = pinned.Repo, Revision = pinned.Revision, Files = [.. declaredFiles] };
            modelName = declared.Name;
        }
        else
        {
            if (files.Count == 0)
            {
                await _output.WriteLineAsync("error: fetch needs at least one --file when no config declares the source");
                return ExitUsage;
            }

            packDirectory = _packDirectory;
            source = new ModelSourceConfig
            {
                Repo = repo,
                Revision = revision ?? "main",
                Files = [.. files.Select(f => new SourceFileConfig { Path = f })],
            };
            modelName = model ?? RepoName(repo);
        }

        ModelConfig target = new()
        {
            Name = modelName,
            Path = Path.Combine(packDirectory, source.Files.Count > 0 ? source.Files[0].Path : modelName),
            Source = source,
        };

        ModelManifestEntry entry = await new ModelFetcher(_transport).FetchAsync(target, packDirectory, cancellationToken, modelName);

        await _output.WriteLineAsync($"fetched {entry.Model} from {entry.Repo} at revision {entry.ResolvedRevision}:");
        foreach (ModelManifestFile file in entry.Files)
        {
            await _output.WriteLineAsync($"  {file.Path}  sha256={file.Sha256}  size={RenderSize(file.Size)}");
        }

        return ExitSuccess;
    }

    private async Task<int> CheckAsync(IReadOnlyList<string> args, CancellationToken cancellationToken)
    {
        string? model = null;

        for (int i = 1; i < args.Count; i++)
        {
            if (args[i].StartsWith("-", StringComparison.Ordinal))
            {
                await _output.WriteLineAsync($"error: unknown check option '{args[i]}'");
                return ExitUsage;
            }

            if (model is not null)
            {
                await _output.WriteLineAsync($"error: check takes at most one model; unexpected argument '{args[i]}'");
                return ExitUsage;
            }

            model = args[i];
        }

        List<ModelManifestEntry> entries = [];
        foreach (string packDirectory in PackDirectories(model))
        {
            ModelManifest? manifest = ModelManifestStore.Load(packDirectory); // an unreadable manifest throws and refuses at 1.
            if (manifest is null)
            {
                continue; // no manifest pins anything here: nothing to report a candidate against.
            }

            foreach (ModelManifestEntry entry in manifest.Models)
            {
                if (model is null || string.Equals(entry.Model, model, StringComparison.Ordinal))
                {
                    entries.Add(entry);
                }
            }
        }

        if (entries.Count == 0)
        {
            if (model is not null)
            {
                await _output.WriteLineAsync($"error: no manifest entry pins model '{model}'.");
                return ExitRefused;
            }

            await _output.WriteLineAsync("no manifest entries to check.");
            return ExitSuccess;
        }

        // Reporting only: the shared checker consults HEAD through the seam and diffs the pinned file set against it,
        // and nothing here downloads, rewrites a manifest, or runs the resolver — a candidate stays a candidate until
        // an operator adopts it behind a restart.
        ModelUpdateChecker checker = new(_transport);
        int candidates = 0;
        foreach (ModelManifestEntry entry in entries)
        {
            candidates += await ReportCandidateAsync(await checker.CheckAsync(entry, cancellationToken));
        }

        await _output.WriteLineAsync(candidates > 0
            ? $"candidate deltas: {candidates}"
            : "all pinned revisions current.");

        return ExitSuccess;
    }

    private async Task<int> ReportCandidateAsync(ModelUpdateCandidate candidate)
    {
        await _output.WriteLineAsync($"model: {candidate.Model}");
        await _output.WriteLineAsync($"  repo: {candidate.Repo}");
        await _output.WriteLineAsync($"  pinned: {candidate.PinnedRevision}");
        await _output.WriteLineAsync($"  head: {candidate.CandidateRevision}");

        foreach (UpdateDelta delta in candidate.Deltas)
        {
            await _output.WriteLineAsync($"  {delta.Detail}");
        }

        await _output.WriteLineAsync(candidate.HasCandidate
            ? $"  candidate: {candidate.DeltaCount} delta(s) against head"
            : "  current: pinned matches head");

        return candidate.DeltaCount;
    }

    private async Task<int> ListManifests(IReadOnlyList<string> args)
    {
        if (args.Count > 1)
        {
            await _output.WriteLineAsync($"error: list takes no arguments; unexpected argument '{args[1]}'");
            return ExitUsage;
        }

        int printed = 0;
        bool anyManifest = false;
        foreach (string packDirectory in PackDirectories(model: null))
        {
            ModelManifest? manifest = ModelManifestStore.Load(packDirectory); // an unreadable manifest throws and refuses at 1.
            if (manifest is null)
            {
                await _output.WriteLineAsync($"no manifest in pack '{packDirectory}'");
                continue;
            }

            anyManifest = true;
            foreach (ModelManifestEntry entry in manifest.Models)
            {
                printed++;
                await PrintEntryAsync(entry);
            }
        }

        if (anyManifest && printed == 0)
        {
            await _output.WriteLineAsync("no model entries in manifest.");
        }

        return ExitSuccess;
    }

    private async Task PrintEntryAsync(ModelManifestEntry entry)
    {
        await _output.WriteLineAsync($"model: {entry.Model}");
        await _output.WriteLineAsync($"  repo: {entry.Repo}");
        await _output.WriteLineAsync($"  revision: {entry.ResolvedRevision}");
        foreach (ModelManifestFile file in entry.Files)
        {
            await _output.WriteLineAsync($"  file: {file.Path}  sha256={file.Sha256}  size={RenderSize(file.Size)}  fetched_at={file.FetchedAt.ToString("O", CultureInfo.InvariantCulture)}");
        }
    }

    // With a config in hand, a fetch only ever fetches what a model entry declares: a named model must declare this
    // repo, and an unnamed repo must be declared by exactly one model.
    private ModelConfig? FindDeclaredModel(string? model, string repo)
    {
        if (_config is null)
        {
            return null;
        }

        if (model is not null)
        {
            ModelConfig? named = _config.Models.FirstOrDefault(candidate => string.Equals(candidate.Name, model, StringComparison.Ordinal));
            return named is not null && named.Source is { } source && string.Equals(source.Repo, repo, StringComparison.Ordinal)
                ? named
                : null;
        }

        List<ModelConfig> candidates = [.. _config.Models.Where(candidate => candidate.Source is { } source && string.Equals(source.Repo, repo, StringComparison.Ordinal))];
        return candidates.Count == 1 ? candidates[0] : null;
    }

    // The packs a command reads: the directories beside each config model's files, or the injected pack directory
    // when no config names any. A named model narrows the search to its own pack.
    private IReadOnlyList<string> PackDirectories(string? model)
    {
        if (_config is null)
        {
            return [_packDirectory];
        }

        if (model is not null)
        {
            ModelConfig? named = _config.Models.FirstOrDefault(candidate => string.Equals(candidate.Name, model, StringComparison.Ordinal));
            if (named is not null)
            {
                return [PackDirectoryOf(named)];
            }
        }

        List<string> directories = [];
        foreach (ModelConfig candidate in _config.Models)
        {
            string pack = PackDirectoryOf(candidate);
            if (!directories.Contains(pack, StringComparer.Ordinal))
            {
                directories.Add(pack);
            }
        }

        return directories.Count > 0 ? directories : [_packDirectory];
    }

    // Mirrors the resolver's pack layout: the pack directory is the directory beside the model's declared file.
    private static string PackDirectoryOf(ModelConfig model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model.Path);

        string? directory = Path.GetDirectoryName(Path.GetFullPath(model.Path));
        if (string.IsNullOrWhiteSpace(directory))
        {
            throw new ModelFetchException($"Model '{model.Name}' path '{model.Path}' names no pack directory to read its manifest from.");
        }

        return directory;
    }

    private static string RepoName(string repo)
    {
        string[] segments = repo.Split('/', StringSplitOptions.RemoveEmptyEntries);
        return segments.Length > 0 ? segments[^1] : repo;
    }

    private static string RenderSize(long? size) =>
        size is long value ? value.ToString(CultureInfo.InvariantCulture) : "unknown";
}
