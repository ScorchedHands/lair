using System.Net;
using Lair.Http;
using Lair.Host.Config;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

namespace Lair.Tests;

/// <summary>
/// Serves the host app over the real SmolLM fixture purely to exercise the embedded web UI surface. The fixture
/// opens nothing when wwwroot/ui was not produced next to the test assembly (a SkipWebUi=true build).
/// </summary>
public sealed class WebUiFixture : IAsyncLifetime
{
    public string BaseUrl { get; private set; } = string.Empty;

    public static bool EmbeddedUiExists() => Directory.Exists(Path.Combine(AppContext.BaseDirectory, "wwwroot", "ui"));

    private GenerationService? _engine;
    private WebApplication? _app;

    public async Task InitializeAsync()
    {
        if (!EmbeddedUiExists())
        {
            // SkipWebUi build: no embedded UI to serve, so nothing is opened.
            return;
        }

        string modelPath = ModelFixture.Resolve() ?? throw new InvalidOperationException(
            $"GGUF fixture '{ModelFixture.SmolLmRelativePath}' was not found walking up from the test assembly directory.");

        LairConfig config = new()
        {
            Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
            Engine = new EngineConfig(),
            Models =
            [
                new ModelConfig
                {
                    Name = "webui-test",
                    Path = modelPath,
                    MaxContextTokens = 512,
                    KvBlocks = 32,
                },
            ],
        };

        _engine = GenerationService.Open(config);
        _app = ApiApp.Build(config, _engine);
        await ((IHost)_app).StartAsync(CancellationToken.None);
        BaseUrl = _app.Urls.Single();
    }

    public async Task DisposeAsync()
    {
        if (_app is not null)
        {
            await ((IHost)_app).StopAsync(TimeSpan.FromSeconds(5));
            await _app.DisposeAsync();
        }

        _engine?.Dispose();
    }
}

/// <summary>
/// Embedded-UI checks: /ui and /ui/index.html return the shell HTML with its stable marker, the assets the HTML
/// references are served from the same directory, and an unmatched client route falls back to the shell.
/// </summary>
[Trait("Category", "WebUi")]
public class WebUiTests : IClassFixture<WebUiFixture>
{
    private static readonly HttpClient Http = new();

    // Redirects must be asserted as responses, not followed transparently.
    private static readonly HttpClient NoRedirectHttp = new(new SocketsHttpHandler { AllowAutoRedirect = false });

    private const string ShellMarker = "data-lair-shell=\"tabbed-shell/rev-b\"";

    private static readonly string[] AssetAttributes = ["src", "href"];

    private readonly WebUiFixture _fixture;

    public WebUiTests(WebUiFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public async Task UiShell_ServesHtmlWithMarkerAndLiveAssets()
    {
        if (!WebUiFixture.EmbeddedUiExists())
        {
            return; // SkipWebUi build: no embedded UI to assert against.
        }

        (HttpResponseMessage shell, string html) = await GetHtmlAsync("/ui");
        Assert.Contains(ShellMarker, html);

        (HttpResponseMessage direct, string directHtml) = await GetHtmlAsync("/ui/index.html");
        Assert.Single(shell.Headers.GetValues("x-lair-request-id")); // host middleware still decorates UI responses.
        Assert.Contains(ShellMarker, directHtml);

        foreach (string assetPath in AssetPaths(html))
        {
            using HttpResponseMessage asset = await Http.GetAsync(_fixture.BaseUrl + assetPath);
            Assert.Equal(HttpStatusCode.OK, asset.StatusCode);
            Assert.True((await asset.Content.ReadAsByteArrayAsync()).Length > 0, $"asset {assetPath} served empty");
        }

        // SPA fallback: an unmatched client route under /ui serves the shell instead of a 404.
        (HttpResponseMessage fallback, string fallbackHtml) = await GetHtmlAsync("/ui/monitor/models");
        Assert.Contains(ShellMarker, fallbackHtml);
    }

    [Fact]
    public async Task Root_RedirectsPermanentlyToUi()
    {
        if (!WebUiFixture.EmbeddedUiExists())
        {
            return; // SkipWebUi build: no UI to redirect to.
        }

        using HttpResponseMessage redirect = await NoRedirectHttp.GetAsync(_fixture.BaseUrl + "/");
        Assert.Equal(HttpStatusCode.MovedPermanently, redirect.StatusCode);
        Assert.Equal("/ui/", redirect.Headers.Location?.ToString());

        // The redirect target really serves the shell.
        (HttpResponseMessage landing, string landingHtml) = await GetHtmlAsync("/ui/");
        Assert.Contains(ShellMarker, landingHtml);
    }

    private async Task<(HttpResponseMessage Response, string Html)> GetHtmlAsync(string path)
    {
        using HttpResponseMessage response = await Http.GetAsync(_fixture.BaseUrl + path);
        string body = await response.Content.ReadAsStringAsync();
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/html", response.Content.Headers.ContentType?.MediaType);
        return (response, body);
    }

    // Collects src/href asset references that point at files in the same directory as the shell.
    private static List<string> AssetPaths(string html)
    {
        List<string> paths = [];
        foreach (string attributeName in AssetAttributes)
        {
            string needle = attributeName + "=\"";
            int searchFrom = 0;
            while ((searchFrom = html.IndexOf(needle, searchFrom, StringComparison.Ordinal)) >= 0)
            {
                int valueStart = searchFrom + needle.Length;
                int valueEnd = html.IndexOf('"', valueStart);
                if (valueEnd < 0)
                {
                    break;
                }

                string reference = html[valueStart..valueEnd];
                bool sameDirectoryAsset = !reference.StartsWith("http", StringComparison.OrdinalIgnoreCase) && !reference.StartsWith("/");
                if (sameDirectoryAsset)
                {
                    paths.Add("/ui/" + reference);
                }

                searchFrom = valueEnd;
            }
        }

        return paths;
    }
}
