namespace Lair.Host.Config;

/// <summary>
/// The published-source seam a model fetch runs against: resolve a declared revision to a pinned commit
/// sha, and open a file's bytes at that revision with an optional resume offset. Everything the fetch path
/// reaches lives behind this seam, so a fetch runs against a local stand-in in tests and against the live
/// hub only in production — no fetch path needs a live network to be exercised.
/// </summary>
public interface IModelHubTransport
{
    /// <summary>
    /// The 40-character commit sha the declared <paramref name="revision"/> resolves to in
    /// <paramref name="repo"/>. A declared sha is returned unchanged by the fetch path without ever calling
    /// this; only a branch name reaches it, and it is resolved once per fetch.
    /// </summary>
    Task<string> ResolveRevisionAsync(string repo, string revision, CancellationToken cancellationToken);

    /// <summary>
    /// The bytes of <paramref name="path"/> in <paramref name="repo"/> at the pinned <paramref name="revision"/>,
    /// starting at <paramref name="resumeFrom"/> (zero for a fresh download). Returns null when the file is not
    /// present at that revision — the fetch path treats a null as a refused artifact, not a partial one. The
    /// returned <see cref="HubFile"/> carries the full file size when the hub reports it, so the fetch path can
    /// confirm a resumed transfer reached the end before it records anything as loadable.
    /// </summary>
    Task<HubFile?> OpenFileAsync(string repo, string revision, string path, long resumeFrom, CancellationToken cancellationToken);
}

/// <summary>
/// One opened hub file: a stream of its bytes from the requested offset, and the whole-file size when the hub
/// reports it (null when it does not). The stream yields exactly the bytes at and after the resume offset.
/// </summary>
public sealed record HubFile(Stream Content, long? TotalSize);
