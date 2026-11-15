using System.Net;
using System.Text;
using System.Text.Json;
using Lair.Host.Config;
using Lair.Http;
using Lair.Models;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

namespace Lair.Tests;

/// <summary>
/// The wire shape of a budget exhaustion and the boundary it must not cross: an admission the shared host ledger's chain
/// cannot seat answers 503 with type <c>engine_overloaded_error</c> and code <c>budget_exhausted</c> inside the error
/// envelope — while the completion payload field of the same name, which reports how a token budget bounded a successful
/// generation, keeps carrying its attribution values untouched. The capacity refusal that predates this keeps answering
/// with its own existing code beside it.
/// </summary>
[Trait("Category", "Api")]
[Trait("Category", "Budget")]
public sealed class BudgetExhaustionApiTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try
            {
                File.Delete(path);
            }
            catch (IOException)
            {
                // A model still holding its mmap view keeps the file on Windows; the temp directory absorbs it.
            }
        }

        _tempFiles.Clear();
    }

    [Fact]
    public async Task AdmissionExhaustion_Returns503EngineOverloadedError_WithBudgetExhaustedCode()
    {
        string path = SyntheticModel("api-budget-exhaust");

        // One shared ledger across two programmatically opened engines: both reservations seat it, and then the host re-reports a smaller total than what already stands there.
        DeviceLedger ledger = new([new CapacitySeam { Tier = BudgetTier.Dram, Resource = BudgetResource.Kv, Capacity = 8 }]);

        await using ServedEngine neighbour = OpenServed(path, "budget-neighbour", kvBlocks: 4, sharedLedger: ledger); // four blocks of the tier seated
        await using ServedEngine served = OpenServed(path, "budget-served", kvBlocks: 4, sharedLedger: ledger);       // its own pool has room for this request

        ledger.UpdateCapacity(BudgetTier.Dram, BudgetResource.Kv, capacity: 4); // the re-reported total leaves no landing room beside what already stands — nothing seated is revoked

        (HttpResponseMessage response, JsonElement body) = await PostCompletionAsync(served, prompt: "hello world", maxTokens: 8);

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        Assert.True(response.Headers.TryGetValues(ApiApp.RequestIdHeader, out _)); // the correlation header every error carries

        JsonElement error = body.GetProperty("error");
        Assert.Equal("budget_exhausted", error.GetProperty("code").GetString());
        Assert.Equal("engine_overloaded_error", error.GetProperty("type").GetString());
        Assert.Contains("budget-served", error.GetProperty("message").GetString()); // the refusal names the model it belongs to

        // The refused admission never entered a completion payload: an exhaustion answer carries no choice and no usage.
        Assert.False(body.TryGetProperty("choices", out _));
    }

    [Fact]
    public async Task CompletionFieldNamedBudgetExhausted_KeepsItsAttribution_AndStaysOutOfTheErrorEnvelope()
    {
        string path = SyntheticModel("api-budget-field");

        // A response ceiling is what bounds this run: the completion payload's budget attribution must read exactly as before.
        await using ServedEngine served = OpenServed(path, "budget-attribution", kvBlocks: 8, declaresKvBudgets: false, maxResponseTokens: 2);

        (HttpResponseMessage ok, JsonElement body) = await PostCompletionAsync(served, prompt: "hello world", maxTokens: 8);
        Assert.Equal(HttpStatusCode.OK, ok.StatusCode);
        Assert.Equal("length", body.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        Assert.Equal("response", body.GetProperty("budget_exhausted").GetString()); // the success-side field of the same name is untouched

        // And on this engine — whose chain seats everything it admits — no error envelope appears at all: the code lives only in errors.
        Assert.False(body.TryGetProperty("error", out _));
    }

    [Fact]
    public void OverloadFamily_KeepsExistingCodes_ForCapacityAndQuotaRefusals_AndAddsOnlyTheNewOne()
    {
        // The two refusals that predate the ledger answer exactly as they always have; only a chain exhaustion takes the new code.
        Assert.Equal("engine_overloaded", EngineBusyException.ForKvCapacity("some-model", 0).Code);
        Assert.Equal("engine_overloaded", EngineBusyException.ForConcurrencyQuota("some-model", 1).Code);

        EngineBusyException exhausted = EngineBusyException.ForBudgetExhaustion(
            "some-model", BudgetResource.Kv, BudgetTier.Dram, "'dram' holds 8 of 8 KV blocks ('other-model' 8 KV blocks)");
        Assert.Equal("budget_exhausted", exhausted.Code);

        // The exhaustion message carries the actionable breakdown beside the model it names.
        Assert.Contains("some-model", exhausted.Message);
        Assert.Contains("'dram' holds 8 of 8 KV blocks ('other-model' 8 KV blocks)", exhausted.Message);
    }

    private string SyntheticModel(string modelName)
    {
        string path = SyntheticLlamaGguf.Write(modelName);
        _tempFiles.Add(path);
        return path;
    }

    // One served engine on an ephemeral port: the same facade + API host a real serve uses. A supplied ledger is the
    // shared-host seam (a programmatic one — no configuration key carries it); omitting it opens the engine against its
    // own ledger exactly as every configured host does today.
    private ServedEngine OpenServed(string modelPath, string modelName, int kvBlocks, bool declaresKvBudgets = true, int? maxResponseTokens = null, DeviceLedger? sharedLedger = null)
    {
        LairConfig config = new()
        {
            Server = new ServerConfig { Bind = "127.0.0.1", Port = 0 },
            Engine = new EngineConfig { Threads = 1 },
            Models =
            [
                new ModelConfig
                {
                    Name = modelName,
                    Path = modelPath,
                    MaxContextTokens = 512,
                    KvBlocks = kvBlocks,
                    MaxResponseTokens = maxResponseTokens,
                    Budgets = declaresKvBudgets
                        ? [new BudgetChain { Resource = BudgetResource.Kv, Order = [BudgetTier.Dram], Caps = new Dictionary<BudgetTier, long> { [BudgetTier.Dram] = kvBlocks } }]
                        : null,
                },
            ],
        };

        GenerationService engine = GenerationService.Open(config, logger: null, ledger: sharedLedger);
        try
        {
            WebApplication app = ApiApp.Build(config, engine);
            ((IHost)app).StartAsync(CancellationToken.None).GetAwaiter().GetResult();
            return new ServedEngine(app, engine);
        }
        catch
        {
            engine.Dispose();
            throw;
        }
    }

    private static async Task<(HttpResponseMessage Response, JsonElement Body)> PostCompletionAsync(ServedEngine served, string prompt, int maxTokens)
    {
        string bodyJson = JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["model"] = served.ModelName,
            ["prompt"] = prompt,
            ["max_tokens"] = maxTokens,
            ["temperature"] = 0f,
            ["stream"] = false,
        });

        HttpResponseMessage response = await served.Http.PostAsync(
            $"{served.Url}/v1/completions", new StringContent(bodyJson, Encoding.UTF8, "application/json"));

        using JsonDocument document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        return (response, document.RootElement.Clone()); // detached: assertions outlive the parsed document
    }

    private sealed class ServedEngine : IAsyncDisposable
    {
        public WebApplication App { get; }
        public GenerationService Engine { get; }
        public HttpClient Http { get; } = new();
        public string Url { get; }
        public string ModelName => Engine.Model.Id;

        public ServedEngine(WebApplication app, GenerationService engine)
        {
            App = app;
            Engine = engine;
            Url = app.Urls.Single();
        }

        public async ValueTask DisposeAsync()
        {
            Http.Dispose();
            await ((IHost)App).StopAsync(TimeSpan.FromSeconds(5));
            await App.DisposeAsync();
            Engine.Dispose();
        }
    }
}
