namespace Lair.Host.Config;

/// <summary>
/// The report-only candidate check one pinned model pack runs against the hub's current HEAD: resolve HEAD, list the
/// tree it carries at that revision, and diff the pinned file set against it — a revision delta when HEAD moved, a
/// removed-file delta for a pinned file no longer present at HEAD, a per-file hash delta where the hub states one, a
/// per-file size delta where both sides state one, and an added-file delta for a HEAD file outside the pinned set.
/// The result is a <see cref="ModelUpdateCandidate">candidate record</see> carrying the candidate revision, the
/// bounded delta list, and a one-line summary suitable for monitor facts and CLI output. Nothing here downloads,
/// stages, repins, rewrites a manifest, or runs the resolver — a candidate stays a candidate until an operator
/// adopts it behind a restart, and the check reaches the transport only through the read-only HEAD/tree seam.
/// </summary>
public sealed class ModelUpdateChecker
{
    private readonly IModelHubTransport _transport;

    public ModelUpdateChecker(IModelHubTransport transport)
    {
        ArgumentNullException.ThrowIfNull(transport);
        _transport = transport;
    }

    /// <summary>
    /// Consults HEAD through the seam, lists the tree at it, and returns the candidate the pinned
    /// <paramref name="entry"/> would move to. A transport failure (offline, gated, unreachable) propagates to the
    /// caller unchanged — reporting decides whether a failed check is a refusal or an omitted candidate, and this
    /// checker never substitutes a revision or a fallback.
    /// </summary>
    public async Task<ModelUpdateCandidate> CheckAsync(ModelManifestEntry entry, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(entry);

        string head = await _transport.ResolveHeadRevisionAsync(entry.Repo, cancellationToken);
        IReadOnlyList<HubTreeEntry> tree = await _transport.ListTreeAsync(entry.Repo, head, cancellationToken);
        return Compute(entry, head, tree);
    }

    /// <summary>
    /// Diffs a pinned entry against an already-consulted HEAD and its tree. Exposed separately so a host that has
    /// already read HEAD and the tree (or a test that supplies them directly) shares one delta computation with the
    /// transport-backed path — the candidate record is identical either way.
    /// </summary>
    public static ModelUpdateCandidate Compute(ModelManifestEntry entry, string head, IReadOnlyList<HubTreeEntry> tree)
    {
        ArgumentNullException.ThrowIfNull(entry);
        ArgumentException.ThrowIfNullOrWhiteSpace(head);
        ArgumentNullException.ThrowIfNull(tree);

        List<UpdateDelta> deltas = [];

        if (!string.Equals(entry.ResolvedRevision, head, StringComparison.OrdinalIgnoreCase))
        {
            deltas.Add(new UpdateDelta(
                UpdateDeltaKind.Revision,
                Path: null,
                $"revision delta: {entry.ResolvedRevision} -> {head}"));
        }

        Dictionary<string, HubTreeEntry> headFiles = new(StringComparer.Ordinal);
        foreach (HubTreeEntry file in tree)
        {
            headFiles[Normalize(file.Path)] = file;
        }

        HashSet<string> pinnedPaths = [];
        foreach (ModelManifestFile file in entry.Files)
        {
            string path = Normalize(file.Path);
            pinnedPaths.Add(path);

            if (!headFiles.TryGetValue(path, out HubTreeEntry? headFile))
            {
                deltas.Add(new UpdateDelta(
                    UpdateDeltaKind.Removed,
                    path,
                    $"file delta: '{path}' pinned in the manifest but no longer present at head"));
                continue;
            }

            if (headFile.HubHash is string hubHash && !string.Equals(hubHash, file.Sha256, StringComparison.OrdinalIgnoreCase))
            {
                deltas.Add(new UpdateDelta(
                    UpdateDeltaKind.Hash,
                    path,
                    $"hash delta: '{path}' pinned {file.Sha256} -> head {hubHash}"));
            }

            if (headFile.Size is long headSize && file.Size is long pinnedSize && headSize != pinnedSize)
            {
                deltas.Add(new UpdateDelta(
                    UpdateDeltaKind.Size,
                    path,
                    $"size delta: '{path}' pinned {pinnedSize} -> head {headSize}"));
            }
        }

        foreach (HubTreeEntry headFile in tree)
        {
            string path = Normalize(headFile.Path);
            if (!pinnedPaths.Contains(path))
            {
                deltas.Add(new UpdateDelta(
                    UpdateDeltaKind.Added,
                    path,
                    $"file delta: '{path}' present at head but not in the pinned file set"));
            }
        }

        return new ModelUpdateCandidate
        {
            Model = entry.Model,
            Repo = entry.Repo,
            PinnedRevision = entry.ResolvedRevision,
            CandidateRevision = head,
            Deltas = [.. deltas],
        };
    }

    private static string Normalize(string path) => path.Replace('\\', '/');
}

/// <summary>
/// The candidate one pinned entry would move to: the candidate revision HEAD currently reports, the bounded delta
/// list against the pinned revision, and a one-line summary naming the candidate revision and the delta count. The
/// summary is what a monitor fact or a CLI candidate line carries; the delta list is what a surface that reports each
/// delta renders. A candidate with no deltas means HEAD already matches the pin.
/// </summary>
public sealed record ModelUpdateCandidate
{
    /// <summary>The model name this candidate is about, matching the manifest entry it pins.</summary>
    public required string Model { get; init; }

    /// <summary>The repo the pinned entry came from, as the manifest recorded it.</summary>
    public required string Repo { get; init; }

    /// <summary>The commit sha the pack pins — what the candidate is measured against.</summary>
    public required string PinnedRevision { get; init; }

    /// <summary>The commit sha HEAD currently points at — the revision a candidate would move to.</summary>
    public required string CandidateRevision { get; init; }

    /// <summary>The bounded delta list against the pinned revision, in report order.</summary>
    public required IReadOnlyList<UpdateDelta> Deltas { get; init; }

    /// <summary>How many deltas separate the pin from HEAD.</summary>
    public int DeltaCount => Deltas.Count;

    /// <summary>True when HEAD separates from the pin by at least one delta.</summary>
    public bool HasCandidate => Deltas.Count > 0;

    /// <summary>The candidate revision and delta summary, one line, for monitor facts and CLI output.</summary>
    public string Summary => HasCandidate
        ? $"{CandidateRevision} ({DeltaCount} delta(s) against {PinnedRevision})"
        : $"{CandidateRevision} matches {PinnedRevision}";
}

/// <summary>One candidate delta against a pinned entry: its kind, the file it concerns (null for the revision delta), and the readable line a reporting surface renders.</summary>
public sealed record UpdateDelta(UpdateDeltaKind Kind, string? Path, string Detail);

/// <summary>The candidate delta kinds: HEAD moved the revision, a pinned file left the tree, a hub-stated hash changed, a stated size changed, or a HEAD file sits outside the pinned set.</summary>
public enum UpdateDeltaKind
{
    Revision,
    Removed,
    Hash,
    Size,
    Added,
}
