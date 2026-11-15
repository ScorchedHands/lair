using System.Security.Cryptography;

namespace Lair.Host.Config;

/// <summary>
/// The load-time gate a model entry with a declared source passes through before its GGUF is opened: the pack
/// directory beside the model file carries the pin record, and what that record declares decides whether the bytes
/// already on disk are the pinned bytes. A present artifact is verified with no transport call at all — a sha that
/// disagrees with the manifest refuses naming repo, revision, file, expected, and actual. An absent pinned file
/// fetches exactly the revision the manifest already resolved, never a freshly re-resolved branch, and the fetched
/// bytes are verified before load continues. A pack with no manifest entry yet but a declared <c>source:</c> block
/// takes the config as the pin, fetches-if-absent, and records the resolved sha in the manifest — a branch resolves
/// once, during that fetch. A transport or network failure while a fetch is required refuses naming repo, revision,
/// file, and reason rather than substituting a different revision or falling back to an unverified file. A model
/// with no source is today's local-path posture: nothing here reads a manifest or reaches the transport.
/// </summary>
public sealed class ModelArtifactResolver
{
    private readonly IModelHubTransport _transport;

    public ModelArtifactResolver(IModelHubTransport transport)
    {
        ArgumentNullException.ThrowIfNull(transport);
        _transport = transport;
    }

    /// <summary>
    /// Resolves <paramref name="model"/>'s declared source against its pack manifest before the model file is opened.
    /// Returns once the artifact on disk is verified (or freshly fetched and verified); refuses with a
    /// <see cref="ModelFetchException">manifest refusal</see> when a present artifact disagrees with its pin or a
    /// required fetch fails. A model with no <c>source:</c> block returns without touching the manifest or the transport.
    /// </summary>
    public async Task ResolveAsync(ModelConfig model, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(model);

        if (model.Source is null)
        {
            return; // no declared provenance: the local path is served exactly as it always was.
        }

        string packDirectory = PackDirectoryOf(model);
        ModelManifest? manifest = ModelManifestStore.Load(packDirectory);
        ModelManifestEntry? entry = manifest?.Models.FirstOrDefault(candidate => string.Equals(candidate.Model, model.Name, StringComparison.Ordinal));

        if (entry is null)
        {
            await FetchFromConfigSourceAsync(model, packDirectory, cancellationToken);
            return;
        }

        List<string> missing = [];
        foreach (ModelManifestFile file in entry.Files)
        {
            string finalPath = FinalPathOf(packDirectory, file.Path, entry.Repo, entry.ResolvedRevision);
            if (!File.Exists(finalPath))
            {
                missing.Add(file.Path);
                continue;
            }

            VerifyAgainstManifest(entry, file, finalPath);
        }

        if (missing.Count == 0)
        {
            return; // every pinned file is present and matches: a verify-only load never reaches the transport.
        }

        await FetchManifestRevisionAsync(entry, packDirectory, cancellationToken);

        // The fetch wrote the missing files; re-verify the whole declared set against the manifest before load continues.
        foreach (ModelManifestFile file in entry.Files)
        {
            string finalPath = FinalPathOf(packDirectory, file.Path, entry.Repo, entry.ResolvedRevision);
            if (!File.Exists(finalPath))
            {
                throw new ModelFetchException($"File '{file.Path}' from repo '{entry.Repo}' at revision {entry.ResolvedRevision} is still absent after the fetch; the artifact is refused.");
            }

            VerifyAgainstManifest(entry, file, finalPath);
        }
    }

    // Fetches exactly the revision the manifest already resolved, for the files that manifest entry declares. A
    // manifest resolved revision is a commit sha, so the fetch path uses it as-is and never re-resolves a branch.
    private async Task FetchManifestRevisionAsync(ModelManifestEntry entry, string packDirectory, CancellationToken cancellationToken)
    {
        ModelConfig pinned = new()
        {
            Name = entry.Model,
            Path = Path.Combine(packDirectory, entry.Files.Count > 0 ? entry.Files[0].Path : entry.Model),
            Source = new ModelSourceConfig
            {
                Repo = entry.Repo,
                Revision = entry.ResolvedRevision,
                Files = [.. entry.Files.Select(file => new SourceFileConfig { Path = file.Path, Sha256 = file.Sha256, Size = file.Size })],
            },
        };

        await FetchThroughSeamAsync(pinned, packDirectory, entry.Repo, entry.ResolvedRevision, cancellationToken);
    }

    // No manifest entry yet: the config source is the pin, the fetch records the resolved sha, and a declared branch
    // resolves exactly once inside the fetch.
    private async Task FetchFromConfigSourceAsync(ModelConfig model, string packDirectory, CancellationToken cancellationToken)
    {
        ModelSourceConfig source = model.Source!;
        await FetchThroughSeamAsync(model, packDirectory, source.Repo, source.Revision, cancellationToken);
    }

    private async Task FetchThroughSeamAsync(ModelConfig model, string packDirectory, string repo, string revision, CancellationToken cancellationToken)
    {
        string declaredFiles = string.Join(", ", model.Source!.Files.Select(file => $"'{file.Path}'"));

        try
        {
            await new ModelFetcher(_transport).FetchAsync(model, packDirectory, cancellationToken);
        }
        catch (ModelFetchException)
        {
            // The fetch already refused naming repo, revision, and file; a manifest refusal stands as written.
            throw;
        }
        catch (Exception ex) when (ex is HttpRequestException or IOException or UnauthorizedAccessException or TaskCanceledException)
        {
            // A transport failure while a fetch is required refuses the artifact naming what was being fetched and why —
            // never a substitute revision and never an unverified fallback.
            throw new ModelFetchException($"Fetch of {declaredFiles} from repo '{repo}' at revision {revision} failed: {ex.Message}", ex);
        }
    }

    private static void VerifyAgainstManifest(ModelManifestEntry entry, ModelManifestFile file, string finalPath)
    {
        string actual = ComputeSha256(finalPath);
        if (!string.Equals(file.Sha256, actual, StringComparison.OrdinalIgnoreCase))
        {
            throw new ModelFetchException($"File '{file.Path}' from repo '{entry.Repo}' at revision {entry.ResolvedRevision} does not match its manifest sha256: expected {file.Sha256}, actual {actual}.");
        }
    }

    private static string PackDirectoryOf(ModelConfig model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model.Path);

        string full = Path.GetFullPath(model.Path);
        string? directory = Path.GetDirectoryName(full);
        if (string.IsNullOrWhiteSpace(directory))
        {
            throw new ModelFetchException($"Model '{model.Name}' path '{model.Path}' names no pack directory to resolve its source against.");
        }

        return directory;
    }

    // Mirrors the fetch path's pack-relative resolution: a declared file is a repo-relative name kept under the pack
    // directory, never an absolute path and never one that escapes it.
    private static string FinalPathOf(string packDirectory, string declaredPath, string repo, string resolvedRevision)
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

        return finalPath;
    }

    private static string ComputeSha256(string path)
    {
        using FileStream file = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant();
    }
}
