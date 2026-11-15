using System.Net;
using Lair.Host.Config;

namespace Lair.Tests;

/// <summary>
/// The environment-token seam a gated hub read runs through: the production hub seam presents the operator's token as
/// a bearer authorization on hub requests only, omits it entirely when none is supplied, never sends it to a host
/// other than the hub, and turns a gated 401/403 into an actionable refusal that names the repo and the reason while
/// keeping the token value out of the refusal text, the CLI output, and the recorded manifest. Every case runs the
/// production seam against a stand-in HTTP handler and a stand-in provider, never the live hub.
/// </summary>
[Trait("Category", "Loader")]
[Trait("Category", "Config")]
public sealed class HubTokenTests : IDisposable
{
    private const string CommitSha = "0123456789abcdef0123456789abcdef01234567";
    private const string Repo = "gated-owner/Token-Fixture";
    private const string ModelName = "token-model";
    private const string Token = "secret-hf-token-9f3b7c1d4e";

    private readonly List<string> _tempDirs = [];

    public void Dispose()
    {
        foreach (string dir in _tempDirs)
        {
            try
            {
                Directory.Delete(dir, recursive: true);
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }
        }
    }

    [Fact]
    public async Task HubTransport_SendsBearerAuthorization_WhenTokenProviderReturnsToken()
    {
        (HuggingFaceHubTransport transport, StubHandler handler) = Hub(new StubTokenProvider(Token), _ => Ok("payload"u8.ToArray()));

        HubFile? file = await transport.OpenFileAsync(Repo, CommitSha, "model.gguf", 0, CancellationToken.None);

        Assert.NotNull(file);
        (string host, string? authorization) = Assert.Single(handler.Requests);
        Assert.Equal("huggingface.co", host);
        Assert.Equal($"Bearer {Token}", authorization);
    }

    [Fact]
    public async Task HubTransport_OmitsAuthorizationHeader_WhenNoTokenIsPresent()
    {
        (HuggingFaceHubTransport transport, StubHandler handler) = Hub(new StubTokenProvider(null), _ => Ok("payload"u8.ToArray()));

        HubFile? file = await transport.OpenFileAsync(Repo, CommitSha, "model.gguf", 0, CancellationToken.None);

        Assert.NotNull(file);
        (string host, string? authorization) = Assert.Single(handler.Requests);
        Assert.Equal("huggingface.co", host);
        Assert.Null(authorization);
    }

    [Fact]
    public async Task HubTransport_NeverSendsToken_ToANonHubPaginationHost()
    {
        (HuggingFaceHubTransport transport, StubHandler handler) = Hub(new StubTokenProvider(Token), request =>
        {
            HttpResponseMessage response = Ok("[]"u8.ToArray());
            if (string.Equals(request.RequestUri!.Host, "huggingface.co", StringComparison.Ordinal))
            {
                response.Headers.TryAddWithoutValidation("Link", "<https://cdn.example.invalid/next>; rel=\"next\"");
            }

            return response;
        });

        IReadOnlyList<HubTreeEntry> entries = await transport.ListTreeAsync(Repo, CommitSha, CancellationToken.None);

        Assert.Empty(entries);
        Assert.Equal(2, handler.Requests.Count);
        Assert.Equal($"Bearer {Token}", handler.Requests[0].authorization); // the hub page carried the credential.
        Assert.Null(handler.Requests[1].authorization); // the off-hub pagination link did not.
    }

    [Fact]
    public async Task GatedFetch_SucceedsThroughTheSeam_WhenAuthIsPresent()
    {
        byte[] content = "gated model bytes fetched with a token"u8.ToArray();
        string expected = Sha256Hex(content);
        string packDir = TempPackDir();

        (HuggingFaceHubTransport transport, StubHandler handler) = Hub(new StubTokenProvider(Token), _ => Ok(content));

        ModelManifestEntry entry = await new ModelFetcher(transport).FetchAsync(
            ModelWithSource(packDir, CommitSha, new SourceFileConfig { Path = "model.gguf", Sha256 = expected, Size = content.Length }),
            packDir,
            CancellationToken.None);

        Assert.Equal(content, File.ReadAllBytes(Path.Combine(packDir, "model.gguf")));
        Assert.Equal(expected, Assert.Single(entry.Files).Sha256);
        Assert.Equal($"Bearer {Token}", handler.Requests[0].authorization); // the authenticated read carried the credential.
    }

    [Fact]
    public async Task GatedFetch_Refuses_WhenAuthIsAbsent_NamingRepoAndReason()
    {
        string packDir = TempPackDir();

        (HuggingFaceHubTransport transport, _) = Hub(new StubTokenProvider(null), _ => new HttpResponseMessage(HttpStatusCode.Unauthorized));

        ModelFetchException error = await Assert.ThrowsAsync<ModelFetchException>(() => new ModelFetcher(transport).FetchAsync(
            ModelWithSource(packDir, CommitSha, new SourceFileConfig { Path = "model.gguf" }),
            packDir,
            CancellationToken.None));

        Assert.Contains(Repo, error.Message); // the refusal names the repo it was refused on.
        Assert.Contains("gated", error.Message); // and the reason the hub turned the request away.
        Assert.Contains(EnvironmentHubTokenProvider.LairVariable, error.Message); // and points at the variable that grants access.
        Assert.Null(ModelManifestStore.Load(packDir)); // a refused fetch records no pin.
    }

    [Fact]
    public async Task GatedFetch_RefusalAndCliOutput_NeverContainTheTokenValue()
    {
        string packDir = TempPackDir();

        (HuggingFaceHubTransport transport, _) = Hub(new StubTokenProvider(Token), _ => new HttpResponseMessage(HttpStatusCode.Forbidden));

        StringWriter output = new();
        int exit = await new ModelCliRunner(transport, packDir, output, ConfigWith(packDir))
            .RunAsync(["fetch", Repo, "--model", ModelName], CancellationToken.None);

        Assert.Equal(ModelCliRunner.ExitRefused, exit);
        Assert.Contains(Repo, output.ToString()); // the CLI reports the repo and reason.
        Assert.Contains("gated", output.ToString());
        Assert.DoesNotContain(Token, output.ToString()); // the offered token value never reaches the output.
    }

    [Fact]
    public async Task Manifest_AfterSuccessfulGatedFetch_NeverContainsTheTokenValue()
    {
        byte[] content = "gated bytes that landed after an authenticated fetch"u8.ToArray();
        string expected = Sha256Hex(content);
        string packDir = TempPackDir();

        (HuggingFaceHubTransport transport, _) = Hub(new StubTokenProvider(Token), _ => Ok(content));

        await new ModelFetcher(transport).FetchAsync(
            ModelWithSource(packDir, CommitSha, new SourceFileConfig { Path = "model.gguf", Sha256 = expected, Size = content.Length }),
            packDir,
            CancellationToken.None);

        string manifestText = File.ReadAllText(Path.Combine(packDir, ModelManifestStore.FileName));
        Assert.DoesNotContain(Token, manifestText); // the manifest records pins, never the credential.
    }

    private static (HuggingFaceHubTransport Transport, StubHandler Handler) Hub(IHubTokenProvider? provider, Func<HttpRequestMessage, HttpResponseMessage> responder)
    {
        StubHandler handler = new(responder);
        return (new HuggingFaceHubTransport(provider, new HttpClient(handler)), handler);
    }

    private static HttpResponseMessage Ok(byte[] body)
    {
        HttpResponseMessage response = new(HttpStatusCode.OK) { Content = new LiveContent(body) };
        response.Content.Headers.ContentLength = body.Length;
        return response;
    }

    // A stand-in body whose handed-out stream outlives the response, mirroring how the fetch path consumes a hub
    // download: the seam returns a live stream and the fetcher reads it after the request completes.
    private sealed class LiveContent : HttpContent
    {
        private readonly byte[] _body;

        public LiveContent(byte[] body) => _body = body;

        protected override async Task SerializeToStreamAsync(Stream stream, TransportContext? context) =>
            await stream.WriteAsync(_body);

        protected override bool TryComputeLength(out long length)
        {
            length = _body.Length;
            return true;
        }

        protected override void Dispose(bool disposing)
        {
        }
    }

    private static ModelConfig ModelWithSource(string packDir, string revision, params SourceFileConfig[] files) =>
        new()
        {
            Name = ModelName,
            Path = Path.Combine(packDir, "model.gguf"),
            Source = new ModelSourceConfig { Repo = Repo, Revision = revision, Files = [.. files] },
        };

    private static LairConfig ConfigWith(string packDir) =>
        new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig(),
            Models =
            [
                ModelWithSource(packDir, CommitSha, new SourceFileConfig { Path = "model.gguf" }),
            ],
        };

    private string TempPackDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"lair-hub-token-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);
        return dir;
    }

    private static string Sha256Hex(byte[] bytes) => Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes)).ToLowerInvariant();

    private sealed class StubTokenProvider : IHubTokenProvider
    {
        private readonly string? _token;

        public StubTokenProvider(string? token) => _token = token;

        public string? GetToken() => _token;
    }

    private sealed class StubHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, HttpResponseMessage> _responder;
        public List<(string host, string? authorization)> Requests { get; } = [];

        public StubHandler(Func<HttpRequestMessage, HttpResponseMessage> responder) => _responder = responder;

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests.Add((request.RequestUri!.Host, request.Headers.Authorization?.ToString()));
            return Task.FromResult(_responder(request));
        }
    }
}
