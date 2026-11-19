using System.Diagnostics;
using System.Net;
using System.Text.Json;
using Lair.Http;
using Lair.Host.Config;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

namespace Lair.Tests;

/// <summary>
/// The model registry over tiny synthetic GGUFs: a multi-model serve opens every declared entry and advertises them in
/// declared order, requests route by name exactly as the single engine answers an unknown id, a reload adds a model that
/// loads in the background while everything already served keeps answering, and a removed model stops accepting new work,
/// refuses further submissions with the retirement code until its admitted requests finish, releases its ledger claims and
/// compute grant only after that drain, and then leaves the advertised set. Reload bookkeeping re-keys per-model fields to
/// each entry's declared position while host-wide keys report once.
/// </summary>
[Trait("Category", "Registry")]
public sealed class RegistryTests : IDisposable
{
    private readonly List<string> _tempPaths = [];

    public void Dispose()
    {
        foreach (string path in _tempPaths)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
                // A temp file the test already released is gone; nothing else depends on it.
            }
        }
    }

    [Fact]
    public void Open_TwoDeclaredModels_AdvertisesBothInDeclaredOrder_RoutesByName()
    {
        string firstPath = SyntheticModel();
        string secondPath = SyntheticModel();

        using ModelRegistry registry = ModelRegistry.Open(LoadConfig($"""
            models:
              - name: alpha
                path: '{firstPath}'
              - name: beta
                path: '{secondPath}'
            """));

        Assert.Equal(["alpha", "beta"], registry.AdvertisedIds());

        using ModelRegistry.Lease first = registry.BeginRequest("alpha");
        Assert.Equal("alpha", first.Engine.Model.Id);

        using ModelRegistry.Lease second = registry.BeginRequest("beta");
        Assert.Equal("beta", second.Engine.Model.Id);
    }

    [Fact]
    public void BeginRequest_UnknownOrUnnamedModel_RefusesLikeTheSingleEngine()
    {
        string path = SyntheticModel();

        using ModelRegistry registry = ModelRegistry.Open(LoadConfig($"models:\n  - name: alpha\n    path: '{path}'"));

        ModelNotFoundException unknown = Assert.Throws<ModelNotFoundException>(() => registry.BeginRequest("nope"));
        Assert.Contains("nope", unknown.Message);

        Assert.Throws<ModelNotFoundException>(() => registry.BeginRequest(string.Empty));
    }

    [Fact]
    public void ApplyConfiguration_AddsModel_LoadsInBackgroundWhileServingContinues()
    {
        string firstPath = SyntheticModel();
        string secondPath = SyntheticModel();

        LairConfig start = LoadConfig($"models:\n  - name: alpha\n    path: '{firstPath}'");
        using ModelRegistry registry = ModelRegistry.Open(start);

        IReadOnlyList<FieldResult> results = registry.ApplyConfiguration(LoadConfig($"""
            models:
              - name: alpha
                path: '{firstPath}'
              - name: beta
                path: '{secondPath}'
            """));

        FieldResult addition = Assert.Single(results, result => result.Change.Field == "models[1]");
        Assert.Equal(ConfigApplyOutcome.Applied, addition.Outcome);
        Assert.Contains("'beta' added", addition.Detail);

        // The load runs in the background behind every claim already held; alpha keeps answering while it warms up.
        using ModelRegistry.Lease serving = registry.BeginRequest("alpha");
        Assert.Equal("alpha", serving.Engine.Model.Id);

        WaitUntil(() => registry.AdvertisedIds().Contains("beta"), "'beta' to advertise after its background open");
        Assert.Equal(["alpha", "beta"], registry.AdvertisedIds());
    }

    [Fact]
    public void ApplyConfiguration_AddOverlappingCoreSet_RefusesNamingTheLiveModel_AndNothingLoads()
    {
        string firstPath = SyntheticModel();
        string secondPath = SyntheticModel();

        LairConfig start = LoadConfig("""
            models:
              - name: alpha
                path: 'PATH_A'
                placement:
                  core_set: "1,2"
            """.Replace("PATH_A", firstPath));

        using ModelRegistry registry = ModelRegistry.Open(start);

        // The candidate is built whole rather than loaded: the loader's own startup validation already refuses two declared entries sharing a core, so this test reaches the registry rule that guards an addition against what live instances still pin.
        ModelConfig overlappingAddition = start.Models[0] with
        {
            Name = "beta",
            Path = secondPath,
            Placement = new PlacementConfig { CoreSet = new HashSet<int>([2, 3]) },
        };

        IReadOnlyList<FieldResult> results = registry.ApplyConfiguration(start with { Models = [.. start.Models, overlappingAddition] });

        FieldResult refusal = Assert.Single(results);
        Assert.Equal(ConfigApplyOutcome.Refused, refusal.Outcome);
        Assert.Contains("both pin core 2", refusal.Detail);
        Assert.Contains("'alpha'", refusal.Detail);
        Assert.Contains("'beta'", refusal.Detail);

        // Nothing opened for the refused entry: the live posture is untouched.
        Thread.Sleep(TimeSpan.FromMilliseconds(400));
        Assert.Equal(["alpha"], registry.AdvertisedIds());
    }

    [Fact]
    public void ApplyConfiguration_RemovesModel_RefusesRetiringRequests_DrainsThenReleasesClaimsAndGrant()
    {
        string firstPath = SyntheticModel();
        string secondPath = SyntheticModel();

        using ModelRegistry registry = ModelRegistry.Open(LoadConfig($"""
            models:
              - name: alpha
                path: '{firstPath}'
              - name: beta
                path: '{secondPath}'
            """));

        WaitUntil(() => registry.Compute?.GrantFor("beta") is not null, "the opened model's compute grant to seat");

        // Hold one admitted request open so the drain window stays observable.
        ModelRegistry.Lease inFlight = registry.BeginRequest("beta");

        IReadOnlyList<FieldResult> results = registry.ApplyConfiguration(LoadConfig($"models:\n  - name: alpha\n    path: '{firstPath}'"));

        FieldResult removal = Assert.Single(results, result => result.Change.Field == "models[1]"); // beta held declared position 1 in the live posture.
        Assert.Equal(ConfigApplyOutcome.Applied, removal.Outcome);
        Assert.Contains("'beta' left the declared set", removal.Detail);

        // While it drains, new submissions are refused with the retirement code...
        EngineBusyException retiring = Assert.Throws<EngineBusyException>(() => registry.BeginRequest("beta"));
        Assert.Equal("model_retiring", retiring.Code);

        // ...and nothing is released while an admitted request still runs.
        Thread.Sleep(TimeSpan.FromMilliseconds(300));
        Assert.NotNull(registry.Compute?.GrantFor("beta"));

        inFlight.Dispose(); // the last admitted generation finished: its release may proceed now.

        WaitUntil(() => registry.AdvertisedIds().SequenceEqual(["alpha"]), "'beta' to leave the advertised set after its drain");
        Assert.Null(registry.Compute?.GrantFor("beta"));

        // The removed model's ledger claims are gone from host accounting — released before its pools were freed, and never resurrected afterward.
        Assert.DoesNotContain(registry.Ledger!.Claims, claim => claim.Model == "beta");

        // The model that never left keeps serving throughout.
        using ModelRegistry.Lease stillServing = registry.BeginRequest("alpha");
        Assert.Equal("alpha", stillServing.Engine.Model.Id);
    }

    private string SyntheticModel()
    {
        string path = SyntheticLlamaGguf.Write("registry-test-model");
        _tempPaths.Add(path);
        return path;
    }

    private LairConfig LoadConfig(string yamlBody) => YamlConfigLoader.Load(WriteYaml(yamlBody));

    private string WriteYaml(string body)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-registry-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, body.Replace('\r', ' ').Trim()); // keep absolute Windows paths inside quoted YAML scalars.
        _tempPaths.Add(path);
        return path;
    }

    private static void WaitUntil(Func<bool> condition, string what)
    {
        Stopwatch clock = Stopwatch.StartNew();
        while (!condition())
        {
            if (clock.Elapsed > TimeSpan.FromSeconds(30))
            {
                throw new TimeoutException($"timed out waiting for {what}");
            }

            Thread.Sleep(50);
        }
    }
}

/// <summary>
/// The registry's HTTP surface: one live two-model serve on an ephemeral loopback port. Every advertised instance appears
/// in <c>/v1/models</c>; named requests route to their own engine; a model no instance answers gets today's 404 shape; and
/// a removed model refuses new work with the retirement code while it drains, then leaves the advertised list — all without
/// disturbing the model that stayed.
/// </summary>
[Trait("Category", "Registry")]
public sealed class RegistryHttpTests : IAsyncLifetime
{
    private static readonly HttpClient Http = new();

    private string _firstPath = string.Empty;
    private string _secondPath = string.Empty;
    private LairConfig? _startConfig;
    private ModelRegistry? _registry;
    private WebApplication? _app;

    public string BaseUrl { get; private set; } = string.Empty;

    public async Task InitializeAsync()
    {
        _firstPath = SyntheticLlamaGguf.Write("alpha");
        _secondPath = SyntheticLlamaGguf.Write("beta");

        // Built whole like the single-engine API fixture does: an ephemeral port is a build-time posture, not something the config loader accepts.
        _startConfig = new LairConfig
        {
            Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
            Engine = new EngineConfig(),
            Models =
            [
                new ModelConfig { Name = "alpha", Path = _firstPath, MaxContextTokens = 512, KvBlocks = 32 },
                new ModelConfig { Name = "beta", Path = _secondPath, MaxContextTokens = 512, KvBlocks = 32 },
            ],
        };

        _registry = ModelRegistry.Open(_startConfig);
        _app = ApiApp.Build(_startConfig, _registry);
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

        _registry?.Dispose();

        foreach (string path in new[] { _firstPath, _secondPath })
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
                // already gone; nothing depends on it.
            }
        }
    }

    [Fact]
    public async Task Models_ReportsEveryAdvertisedInstanceInDeclaredOrder()
    {
        using HttpResponseMessage response = await Http.GetAsync(Url("/v1/models"));
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        JsonDocument body = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        List<string> ids = [.. body.RootElement.GetProperty("data").EnumerateArray().Select(entry => entry.GetProperty("id").GetString()!)];

        Assert.Equal(["alpha", "beta"], ids);
    }

    [Fact]
    public async Task Chat_RoutesToTheNamedModel_UnknownModelGetsNotFoundShape()
    {
        (HttpResponseMessage routed, JsonElement routedBody) = await PostJsonAsync(
            "/v1/chat/completions",
            """{"model":"beta","messages":[{"role":"user","content":"hello world"}],"max_tokens":4,"temperature":0}""");

        Assert.Equal(HttpStatusCode.OK, routed.StatusCode);
        Assert.Equal("beta", routedBody.GetProperty("model").GetString());

        (HttpResponseMessage unknown, JsonElement unknownBody) = await PostJsonAsync(
            "/v1/chat/completions",
            """{"model":"nope","messages":[{"role":"user","content":"hello"}],"max_tokens":4}""");

        Assert.Equal(HttpStatusCode.NotFound, unknown.StatusCode);
        Assert.Equal("model_not_found", unknownBody.GetProperty("error").GetProperty("code").GetString());
    }

    [Fact]
    public async Task RemovedModel_RefusesNewWorkWithRetirementCode_ThenLeavesTheAdvertisedListWhileAlphaKeepsServing()
    {
        ModelRegistry registry = _registry!;

        // Hold one admitted request open so the drain window is observable from the wire.
        ModelRegistry.Lease inFlight = registry.BeginRequest("beta");

        registry.ApplyConfiguration(_startConfig! with { Models = [_startConfig!.Models[0]] });

        (HttpResponseMessage retiring, JsonElement retiringBody) = await PostJsonAsync(
            "/v1/chat/completions",
            """{"model":"beta","messages":[{"role":"user","content":"hello world"}],"max_tokens":4,"temperature":0}""");

        Assert.Equal(HttpStatusCode.ServiceUnavailable, retiring.StatusCode);
        Assert.Equal("engine_overloaded_error", retiringBody.GetProperty("error").GetProperty("type").GetString());
        Assert.Equal("model_retiring", retiringBody.GetProperty("error").GetProperty("code").GetString());

        inFlight.Dispose(); // the held admission finishes: beta's release may proceed.

        WaitUntil(async () => (await AdvertisedIdsAsync()).SequenceEqual(["alpha"]), "'beta' to leave /v1/models after its drain");

        // The model that stayed keeps answering end to end over HTTP.
        (HttpResponseMessage stillServing, JsonElement body) = await PostJsonAsync(
            "/v1/chat/completions",
            """{"model":"alpha","messages":[{"role":"user","content":"hello world"}],"max_tokens":4,"temperature":0}""");

        Assert.Equal(HttpStatusCode.OK, stillServing.StatusCode);
        Assert.Equal("alpha", body.GetProperty("model").GetString());
    }

    private async Task<List<string>> AdvertisedIdsAsync()
    {
        using HttpResponseMessage response = await Http.GetAsync(Url("/v1/models"));
        JsonDocument body = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        return [.. body.RootElement.GetProperty("data").EnumerateArray().Select(entry => entry.GetProperty("id").GetString()!)];
    }

    private async Task<(HttpResponseMessage Response, JsonElement Body)> PostJsonAsync(string path, string json)
    {
        using HttpRequestMessage request = new(HttpMethod.Post, Url(path))
        {
            Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json"),
        };

        HttpResponseMessage response = await Http.SendAsync(request);
        return (response, JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement.Clone());
    }

    private string Url(string path) => BaseUrl + path;

    private static void WaitUntil(Func<Task<bool>> condition, string what)
    {
        Stopwatch clock = Stopwatch.StartNew();
        while (!condition().GetAwaiter().GetResult())
        {
            if (clock.Elapsed > TimeSpan.FromSeconds(30))
            {
                throw new TimeoutException($"timed out waiting for {what}");
            }

            Thread.Sleep(50);
        }
    }
}
