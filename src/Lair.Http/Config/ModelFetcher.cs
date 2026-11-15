using System.Collections.Concurrent;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using Lair.Loader;

namespace Lair.Host.Config;

/// <summary>
/// Fetches a model's declared source into its pack directory through the <see cref="IModelHubTransport"/> seam.
/// A declared revision is used directly when it is already a commit sha and resolved once through the seam when
/// it is a branch. Each file downloads to a <c>.part</c> sibling that resumes from its current length, and only
/// after the whole file lands and its LAIR-computed sha256 agrees with the pin does it move to the final,
/// loadable name — a <c>.part</c> is never a loadable path. A declared file set is fetched as one coherent unit
/// at a single resolved revision: any missing or refused file refuses the whole set, so a shard set is never
/// recorded as loadable in pieces. A per-<c>(repo, revision, path)</c> lock marker serializes concurrent fetches
/// and reclaims a marker left by a killed fetch.
/// </summary>
public sealed class ModelFetcher
{
    private const string LockDirectory = ".fetch-locks";

    private static readonly ConcurrentDictionary<string, SemaphoreSlim> ProcessGates = new(StringComparer.Ordinal);

    private readonly IModelHubTransport _transport;

    public ModelFetcher(IModelHubTransport transport)
    {
        ArgumentNullException.ThrowIfNull(transport);
        _transport = transport;
    }

    /// <summary>
    /// Fetches <paramref name="model"/>'s declared source into <paramref name="packDirectory"/> and records the
    /// fetched model in the pack manifest. Returns the manifest entry the fetch recorded. A refusal leaves the
    /// manifest untouched: no partial entry is written, so nothing half-fetched is loadable.
    /// </summary>
    public async Task<ModelManifestEntry> FetchAsync(ModelConfig model, string packDirectory, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(packDirectory);

        ModelSourceConfig source = model.Source
            ?? throw new ModelFetchException($"Model '{model.Name}' declares no source to fetch.");

        string resolvedRevision = await ResolveRevisionAsync(source.Repo, source.Revision, cancellationToken);

        // Every declared file lands before the manifest records anything, so the set is coherent at one revision.
        List<ModelManifestFile> fetched = [];
        foreach (SourceFileConfig file in source.Files)
        {
            fetched.Add(await FetchFileAsync(source.Repo, resolvedRevision, file, packDirectory, cancellationToken));
        }

        ModelManifestEntry entry = new()
        {
            Model = model.Name,
            Repo = source.Repo,
            ResolvedRevision = resolvedRevision,
            Files = [.. fetched],
            Sharded = source.Files.Count > 1,
        };

        RecordManifestEntry(packDirectory, entry);
        return entry;
    }

    private async Task<string> ResolveRevisionAsync(string repo, string revision, CancellationToken cancellationToken)
    {
        // A declared commit sha is already the pin: it is used as-is and never sent to the seam. Only a branch
        // name resolves, and it resolves once for the whole fetch.
        if (IsCommitSha(revision))
        {
            return revision;
        }

        string resolved = await _transport.ResolveRevisionAsync(repo, revision, cancellationToken);
        if (!IsCommitSha(resolved))
        {
            throw new ModelFetchException($"Revision '{revision}' for repo '{repo}' resolved to '{resolved}', which is not a 40-character commit sha.");
        }

        return resolved;
    }

    private async Task<ModelManifestFile> FetchFileAsync(string repo, string resolvedRevision, SourceFileConfig file, string packDirectory, CancellationToken cancellationToken)
    {
        (string finalPath, string partPath) = ResolvePaths(packDirectory, file.Path, repo, resolvedRevision);

        if (File.Exists(finalPath))
        {
            // The bytes already landed in an earlier fetch: verify the pin against them and record without
            // re-streaming, so a concurrent or repeated fetch never double-streams or renames over a loadable file.
            return RecordFromExisting(finalPath, file, repo, resolvedRevision);
        }

        using LockHandle handle = AcquireLock(packDirectory, repo, resolvedRevision, file.Path);

        // Re-check under the lock: a concurrent fetch may have finalized this file while this one waited on the
        // gate, so it records the landed bytes instead of streaming a second copy.
        if (File.Exists(finalPath))
        {
            return RecordFromExisting(finalPath, file, repo, resolvedRevision);
        }

        long resumeFrom = File.Exists(partPath) ? new FileInfo(partPath).Length : 0;

        HubFile? hub = await _transport.OpenFileAsync(repo, resolvedRevision, file.Path, resumeFrom, cancellationToken)
            ?? throw new ModelFetchException($"File '{file.Path}' is not present in repo '{repo}' at revision {resolvedRevision}; the fetch is refused.");

        try
        {
            using FileStream part = new(partPath, FileMode.Append, FileAccess.Write, FileShare.None);
            await hub.Content.CopyToAsync(part, cancellationToken);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new ModelFetchException($"File '{file.Path}' from repo '{repo}' at revision {resolvedRevision} could not be written to the pack: {ex.Message}", ex);
        }

        long partLength = new FileInfo(partPath).Length;
        if (hub.TotalSize is long total && partLength != total)
        {
            // The stream ended short of the whole file: keep the partial bytes for a resume and refuse to finalize.
            throw new ModelFetchException($"File '{file.Path}' from repo '{repo}' at revision {resolvedRevision} downloaded {partLength} of {total} bytes; the transfer is incomplete.");
        }

        string actual = ComputeSha256(partPath);
        if (file.Sha256 is string expected && !string.Equals(expected, actual, StringComparison.OrdinalIgnoreCase))
        {
            // The bytes disagree with the pin: drop the partial file so a retry re-downloads fresh rather than
            // resuming from bytes already known to be wrong.
            TryDeleteQuiet(partPath);
            throw new ModelFetchException($"File '{file.Path}' from repo '{repo}' at revision {resolvedRevision} does not match its pinned sha256: expected {expected}, actual {actual}.");
        }

        try
        {
            File.Move(partPath, finalPath, overwrite: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new ModelFetchException($"File '{file.Path}' from repo '{repo}' at revision {resolvedRevision} could not be finalized in the pack: {ex.Message}", ex);
        }

        return new ModelManifestFile
        {
            Path = file.Path,
            Sha256 = actual,
            Size = partLength,
            FetchedAt = DateTimeOffset.UtcNow,
            SplitTensorsCount = ReadSplitTensorsCount(finalPath),
        };
    }

    private static ModelManifestFile RecordFromExisting(string finalPath, SourceFileConfig file, string repo, string resolvedRevision)
    {
        string actual = ComputeSha256(finalPath);
        if (file.Sha256 is string expected && !string.Equals(expected, actual, StringComparison.OrdinalIgnoreCase))
        {
            throw new ModelFetchException($"File '{file.Path}' from repo '{repo}' at revision {resolvedRevision} does not match its pinned sha256: expected {expected}, actual {actual}.");
        }

        return new ModelManifestFile
        {
            Path = file.Path,
            Sha256 = actual,
            Size = new FileInfo(finalPath).Length,
            FetchedAt = DateTimeOffset.UtcNow,
            SplitTensorsCount = ReadSplitTensorsCount(finalPath),
        };
    }

    private static void RecordManifestEntry(string packDirectory, ModelManifestEntry entry)
    {
        ModelManifest? existing = ModelManifestStore.Load(packDirectory);
        List<ModelManifestEntry> models = existing?.Models is { } carried ? [.. carried] : [];
        models.RemoveAll(model => string.Equals(model.Model, entry.Model, StringComparison.Ordinal));
        models.Add(entry);
        ModelManifestStore.Save(packDirectory, new ModelManifest { Models = [.. models] });
    }

    private static (string Final, string Part) ResolvePaths(string packDirectory, string declaredPath, string repo, string resolvedRevision)
    {
        if (string.IsNullOrWhiteSpace(declaredPath) || declaredPath.Contains('\0'))
        {
            throw new ModelFetchException($"File '{declaredPath}' from repo '{repo}' at revision {resolvedRevision} is not a usable file name.");
        }

        if (Path.IsPathRooted(declaredPath))
        {
            throw new ModelFetchException($"File '{declaredPath}' from repo '{repo}' at revision {resolvedRevision} must be a repo-relative path, not an absolute one.");
        }

        string relative = declaredPath.Replace('\\', '/');
        foreach (string segment in relative.Split('/'))
        {
            if (segment == "..")
            {
                throw new ModelFetchException($"File '{declaredPath}' from repo '{repo}' at revision {resolvedRevision} must not escape the pack directory.");
            }
        }

        string packRoot = Path.GetFullPath(packDirectory);
        string finalPath = Path.GetFullPath(Path.Combine(packRoot, relative));
        if (!finalPath.StartsWith(packRoot + Path.DirectorySeparatorChar, StringComparison.Ordinal))
        {
            throw new ModelFetchException($"File '{declaredPath}' from repo '{repo}' at revision {resolvedRevision} resolves outside the pack directory.");
        }

        Directory.CreateDirectory(packRoot);
        Directory.CreateDirectory(Path.GetDirectoryName(finalPath)!);
        return (finalPath, finalPath + ".part");
    }

    private static string ComputeSha256(string path)
    {
        using FileStream file = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant();
    }

    // Reads the GGUF split count from a finalized file when it is a readable GGUF carrying the key, and reports
    // nothing when it is not — an absent value is left absent rather than invented.
    private static long? ReadSplitTensorsCount(string path)
    {
        try
        {
            using GgufFile gguf = GgufFile.Open(path);
            return gguf.Metadata.Has("split.tensors.count") ? gguf.Metadata.Get("split.tensors.count").AsLong() : null;
        }
        catch (Exception ex) when (ex is GgufException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    private static bool IsCommitSha(string value) =>
        value.Length == 40 && value.All(ch => ch is >= '0' and <= '9' or >= 'a' and <= 'f');

    // Lock machinery: a per-(repo, revision, path) marker file inside the pack directory holds one fetch at a
    // time. An in-process gate serializes same-process fetches so a concurrent fetch waits for the first to
    // finish instead of racing the .part rename; a marker whose recorded process is gone is reclaimed before the
    // retry, so a fetch killed mid-download does not wedge the artifact.
    private static LockHandle AcquireLock(string packDirectory, string repo, string revision, string path)
    {
        string gateKey = $"{packDirectory}\n{repo}\n{revision}\n{path}";
        SemaphoreSlim gate = ProcessGates.GetOrAdd(gateKey, _ => new SemaphoreSlim(1, 1));
        gate.Wait();

        try
        {
            string lockFile = LockFilePath(packDirectory, repo, revision, path);
            Directory.CreateDirectory(Path.GetDirectoryName(lockFile)!);

            for (int attempt = 0; attempt < 200; attempt++)
            {
                try
                {
                    using FileStream marker = new(lockFile, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                    using StreamWriter writer = new(marker);
                    writer.Write(Environment.ProcessId.ToString(System.Globalization.CultureInfo.InvariantCulture));
                    return new LockHandle(gate, lockFile);
                }
                catch (IOException)
                {
                    if (IsStaleLock(lockFile))
                    {
                        TryDeleteQuiet(lockFile);
                        continue; // reclaimed a marker left by a killed fetch.
                    }

                    Thread.Sleep(25); // another live fetch holds this artifact: wait for it to finish.
                }
            }

            throw new ModelFetchException($"Another fetch holds the lock for repo '{repo}' at revision {revision} file '{path}'; refusing to double-fetch it.");
        }
        catch
        {
            // Every path that did not hand the gate to a LockHandle releases it exactly once here.
            gate.Release();
            throw;
        }
    }

    private static bool IsStaleLock(string lockFile)
    {
        try
        {
            using FileStream file = File.OpenRead(lockFile);
            using StreamReader reader = new(file);
            string? head = reader.ReadLine();
            if (!int.TryParse(head, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out int pid))
            {
                return true; // a marker this fetch did not write is not one it can trust.
            }

            try
            {
                using Process process = Process.GetProcessById(pid);
                return false; // a live process still holds this artifact.
            }
            catch (ArgumentException)
            {
                return true; // the recorded process is gone: the marker is stale.
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return true; // an unreadable marker cannot be holding a fetch we can see.
        }
    }

    private static void TryDeleteQuiet(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    internal static string LockFilePath(string packDirectory, string repo, string revision, string path)
    {
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes($"{repo}\n{revision}\n{path}"));
        return Path.Combine(packDirectory, LockDirectory, Convert.ToHexString(digest).ToLowerInvariant() + ".lock");
    }

    private sealed class LockHandle : IDisposable
    {
        private readonly SemaphoreSlim _gate;
        private readonly string _lockFile;
        private int _disposed;

        internal LockHandle(SemaphoreSlim gate, string lockFile)
        {
            _gate = gate;
            _lockFile = lockFile;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            TryDeleteQuiet(_lockFile);
            _gate.Release();
        }
    }
}
