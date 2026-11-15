using System.Net;
using System.Net.Http.Headers;
using System.Text.RegularExpressions;

namespace Lair.Host.Config;

/// <summary>
/// The live hub seam: it reads a published model's revision, tree, and file bytes straight from
/// <c>huggingface.co</c>. A declared revision resolves through the commit page's canonical location; a tree
/// listing follows the hub's <c>Link</c> pagination; and a file download opens the resolve endpoint with an
/// HTTP <c>Range</c> so an interrupted transfer resumes from its current offset. This is the production
/// implementation — tests run the fetch path against a local stand-in instead and never reach this type.
/// </summary>
public sealed class HuggingFaceHubTransport : IModelHubTransport
{
    private const string HubBase = "https://huggingface.co";

    private static readonly Regex NextLink = new(@"<([^>]+)>;\s*rel=""?next""?", RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private readonly HttpClient _http;

    public HuggingFaceHubTransport(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
    }

    /// <inheritdoc />
    public async Task<string> ResolveRevisionAsync(string repo, string revision, CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repo);
        ArgumentException.ThrowIfNullOrWhiteSpace(revision);

        // The commit page answers with the canonical commit sha in its redirect location, whether the declared
        // revision was a branch or already a sha — the fetch path only routes a branch here.
        string url = $"{HubBase}/{repo}/commit/{Uri.EscapeDataString(revision)}";
        using HttpRequestMessage request = new(HttpMethod.Head, url);
        using HttpResponseMessage response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);

        if (!response.IsSuccessStatusCode && response.StatusCode != HttpStatusCode.Redirect && response.StatusCode != HttpStatusCode.Found)
        {
            throw new ModelFetchException($"Revision '{revision}' for repo '{repo}' could not be resolved from the hub (HTTP {(int)response.StatusCode}).");
        }

        Uri? location = response.Headers.Location;
        if (location is null)
        {
            throw new ModelFetchException($"Revision '{revision}' for repo '{repo}' resolved to no commit location.");
        }

        Uri absolute = location.IsAbsoluteUri ? location : new Uri(new Uri(HubBase), location);
        string resolved = absolute.AbsolutePath.Split('/', StringSplitOptions.RemoveEmptyEntries) switch
        {
            [_, "commit", var sha, ..] => sha,
            _ => throw new ModelFetchException($"Revision '{revision}' for repo '{repo}' resolved to an unexpected commit location '{absolute.AbsolutePath}'."),
        };

        if (!IsCommitSha(resolved))
        {
            throw new ModelFetchException($"Revision '{revision}' for repo '{repo}' resolved to '{resolved}', which is not a 40-character commit sha.");
        }

        return resolved;
    }

    /// <inheritdoc />
    public async Task<HubFile?> OpenFileAsync(string repo, string revision, string path, long resumeFrom, CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repo);
        ArgumentException.ThrowIfNullOrWhiteSpace(revision);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentOutOfRangeException.ThrowIfNegative(resumeFrom);

        string url = $"{HubBase}/{repo}/resolve/{revision}/{path}";
        using HttpRequestMessage request = new(HttpMethod.Get, url);
        if (resumeFrom > 0)
        {
            request.Headers.Range = new RangeHeaderValue(resumeFrom, null);
        }

        using HttpResponseMessage response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);

        if (response.StatusCode == HttpStatusCode.NotFound)
        {
            return null; // not present at this revision — the fetch path refuses the artifact.
        }

        if (response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            // The resume offset is at or past the end: the bytes already on disk are the whole file.
            long? total = ParseContentRangeTotal(response.Content.Headers.ContentRange);
            return new HubFile(Stream.Null, total);
        }

        if (!response.IsSuccessStatusCode)
        {
            throw new ModelFetchException($"File '{path}' in repo '{repo}' at revision {revision} could not be fetched (HTTP {(int)response.StatusCode}).");
        }

        long? totalSize = response.Content.Headers.ContentRange is { } range
            ? range.Length
            : response.Content.Headers.ContentLength;

        Stream content = await response.Content.ReadAsStreamAsync(cancellationToken);
        return new HubFile(content, totalSize);
    }

    /// <inheritdoc />
    public async Task<string> ResolveHeadRevisionAsync(string repo, CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repo);

        // The model-info endpoint reports the default branch's current commit sha — the tip a candidate report
        // compares against. Nothing on the fetch or load path calls this.
        string url = $"{HubBase}/api/models/{repo}";
        using HttpResponseMessage response = await _http.GetAsync(url, cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            throw new ModelFetchException($"HEAD for repo '{repo}' could not be resolved from the hub (HTTP {(int)response.StatusCode}).");
        }

        string body = await response.Content.ReadAsStringAsync(cancellationToken);
        string head;
        try
        {
            using System.Text.Json.JsonDocument document = System.Text.Json.JsonDocument.Parse(body);
            head = document.RootElement.TryGetProperty("sha", out System.Text.Json.JsonElement shaElement)
                ? shaElement.GetString() ?? string.Empty
                : string.Empty;
        }
        catch (System.Text.Json.JsonException ex)
        {
            throw new ModelFetchException($"HEAD for repo '{repo}' could not be read from the hub response: {ex.Message}", ex);
        }

        if (!IsCommitSha(head))
        {
            throw new ModelFetchException($"HEAD for repo '{repo}' reported '{head}', which is not a 40-character commit sha.");
        }

        return head;
    }

    /// <summary>
    /// The files <paramref name="repo"/> carries at <paramref name="revision"/>, following the hub's <c>Link</c>
    /// pagination so a large tree is read whole rather than one page at a time. Candidate reporting only: the fetch
    /// path never lists a tree.
    /// </summary>
    public async Task<IReadOnlyList<HubTreeEntry>> ListTreeAsync(string repo, string revision, CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repo);
        ArgumentException.ThrowIfNullOrWhiteSpace(revision);

        List<HubTreeEntry> entries = [];
        string? pageUrl = $"{HubBase}/api/models/{repo}/tree/{Uri.EscapeDataString(revision)}?recursive=true";

        while (pageUrl is not null)
        {
            using HttpResponseMessage response = await _http.GetAsync(pageUrl, cancellationToken);
            if (!response.IsSuccessStatusCode)
            {
                throw new ModelFetchException($"Tree for repo '{repo}' at revision {revision} could not be listed (HTTP {(int)response.StatusCode}).");
            }

            string body = await response.Content.ReadAsStringAsync(cancellationToken);
            entries.AddRange(ParseTreeEntries(body));

            pageUrl = NextPageUrl(response.Headers);
        }

        return entries;
    }

    private static string? NextPageUrl(HttpResponseHeaders headers)
    {
        foreach (string header in headers.GetValues("Link"))
        {
            Match match = NextLink.Match(header);
            if (match.Success)
            {
                string link = match.Groups[1].Value;
                return link.StartsWith("http", StringComparison.Ordinal) ? link : new Uri(new Uri(HubBase), link).ToString();
            }
        }

        return null;
    }

    private static IEnumerable<HubTreeEntry> ParseTreeEntries(string body)
    {
        // The hub returns a JSON array of { type, path, size, lfs } objects; the path and size are read here, plus
        // the LFS pointer's oid as an advisory hash — LAIR never gates loadability on a hub-reported hash.
        using System.Text.Json.JsonDocument document = System.Text.Json.JsonDocument.Parse(body);
        List<HubTreeEntry> entries = [];

        foreach (System.Text.Json.JsonElement item in document.RootElement.EnumerateArray())
        {
            if (item.TryGetProperty("path", out System.Text.Json.JsonElement pathElement) &&
                pathElement.GetString() is string path &&
                !string.IsNullOrWhiteSpace(path))
            {
                long? size = item.TryGetProperty("size", out System.Text.Json.JsonElement sizeElement) &&
                             sizeElement.TryGetInt64(out long declared)
                    ? declared
                    : null;

                string? hubHash = item.TryGetProperty("lfs", out System.Text.Json.JsonElement lfs) &&
                                  lfs.ValueKind == System.Text.Json.JsonValueKind.Object &&
                                  lfs.TryGetProperty("oid", out System.Text.Json.JsonElement oid) &&
                                  oid.GetString() is string hash &&
                                  !string.IsNullOrWhiteSpace(hash)
                    ? hash.ToLowerInvariant()
                    : null;

                entries.Add(new HubTreeEntry(path, size, hubHash));
            }
        }

        return entries;
    }

    private static long? ParseContentRangeTotal(System.Net.Http.Headers.ContentRangeHeaderValue? range) =>
        range is { HasRange: false } && range.Length is long total ? total : null;

    private static bool IsCommitSha(string value) =>
        value.Length == 40 && value.All(ch => ch is >= '0' and <= '9' or >= 'a' and <= 'f');
}

/// <summary>
/// One file a tree listing reports at a revision: its repo-relative path, its size when the hub states one, and
/// the hub-stated hash when the hub states one (an LFS pointer's oid — advisory for candidate reporting, never a
/// loadability gate). A null is an unreported value, not a zero.
/// </summary>
public sealed record HubTreeEntry(string Path, long? Size, string? HubHash = null);
