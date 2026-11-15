using System.Net.Http;
using System.Security.Cryptography;
using System.Text.Json;
using Lair.Host.Config;
using Lair.Http;
using Lair.Http.Monitoring;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// Candidate reporting on the monitor surface: a served model that declares a published source gets one report-only
/// HEAD/tree consult against its pinned pack at open, and a candidate that separates from the pin rides the same
/// attached-facts path every other additive operator view uses — the candidate revision and its delta summary on a
/// trailing optional <c>metrics_update_candidate</c> field. A model already at HEAD, a model with no source, and a
/// check the hub could not answer all leave the snapshot exactly as it read before the candidate fact existed: the
/// field is null, the load path and the manifest pin are untouched, and nothing downloads. Every case runs against a
/// local stand-in transport, never the live hub.
/// </summary>
[Trait("Category", "Api")]
[Trait("Category", "Metrics")]
public sealed class ModelUpdateCandidateMonitorTests : IDisposable
{
    private const string Repo = "bartowski/Candidate-Fixture";
    private const string ModelName = "candidate-model";
    private const string PinnedCommit = "0123456789abcdef0123456789abcdef01234567";
    private const string HeadCommit = "ffffffffffffffffffffffffffffffffffffffff";

    private readonly List<string> _tempDirs = [];
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
            }
            catch (UnauthorizedAccessException)
            {
            }
        }

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
    public async Task Snapshot_CarriesUpdateCandidate_WhenHeadMovesPastPin()
    {
        string packDir = TempPackDir();
        byte[] modelBytes = StageVerifiedPack(packDir, PinnedCommit);

        FakeTransport transport = new();
        transport.HeadSha = HeadCommit;
        transport.HeadTree =
        [
            new HubTreeEntry("model.gguf", modelBytes.Length + 1, Sha256Hex("changed bytes"u8.ToArray())),
            new HubTreeEntry("extra.gguf", 7),
        ];

        using GenerationService engine = OpenServedModel(packDir, transport);
        MetricsSnapshot snapshot = NewestSnapshotWithCandidate(engine);

        Assert.NotNull(snapshot.UpdateCandidate);
        Assert.Contains(HeadCommit, snapshot.UpdateCandidate); // the candidate revision rides the fact
        Assert.Contains("delta", snapshot.UpdateCandidate);    // and so does the delta summary against the pin

        using JsonDocument document = JsonDocument.Parse(snapshot.ToJson());
        Assert.True(document.RootElement.TryGetProperty("metrics_update_candidate", out JsonElement candidate));
        Assert.Equal(JsonValueKind.String, candidate.ValueKind);
        Assert.Contains(HeadCommit, candidate.GetString());
        Assert.Contains(PinnedCommit, candidate.GetString());
    }

    [Fact]
    public async Task Snapshot_OmitsUpdateCandidate_WhenHeadMatchesPin()
    {
        string packDir = TempPackDir();
        byte[] modelBytes = StageVerifiedPack(packDir, PinnedCommit);

        FakeTransport transport = new();
        transport.HeadSha = PinnedCommit; // HEAD already points at the pin
        transport.HeadTree = [new HubTreeEntry("model.gguf", modelBytes.Length, Sha256Hex(modelBytes))];

        using GenerationService engine = OpenServedModel(packDir, transport);
        MetricsSnapshot snapshot = NewestSnapshotAfterLoad(engine);

        Assert.Null(snapshot.UpdateCandidate); // a candidate-free check leaves the additive slot inert

        using JsonDocument document = JsonDocument.Parse(snapshot.ToJson());
        JsonElement root = document.RootElement;
        Assert.True(root.TryGetProperty("metrics_update_candidate", out JsonElement candidate)); // a trailing optional field, serialized null like its neighbors
        Assert.Equal(JsonValueKind.Null, candidate.ValueKind);

        // Every operator-view field an existing consumer already reads is still there with the same meaning.
        Assert.Equal(ModelName, root.GetProperty("metrics_model").GetString());
        Assert.True(root.TryGetProperty("metrics_kv_capacity_blocks", out _));
        Assert.True(root.TryGetProperty("metrics_requests_total", out _));
    }

    [Fact]
    public async Task Serve_UpdateCandidateCheckFailure_KeepsServing_AndLeavesManifestAndLoadPathUntouched()
    {
        string packDir = TempPackDir();
        StageVerifiedPack(packDir, PinnedCommit);
        string manifestBefore = File.ReadAllText(Path.Combine(packDir, ModelManifestStore.FileName));

        FakeTransport transport = new();
        transport.HeadFails = true; // the hub cannot answer the report-only HEAD/tree consult

        using GenerationService engine = OpenServedModel(packDir, transport); // the open is not blocked by a failed report

        GenerationResult served = await engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = "hello world",
            MaxTokens = 8,
            Temperature = 0f,
        }, CancellationToken.None);

        Assert.False(string.IsNullOrEmpty(served.Text)); // serving proceeds exactly as it did before the candidate seam existed
        Assert.Equal(0, transport.OpenCalls);            // a failed report never downloads
        Assert.Equal(manifestBefore, File.ReadAllText(Path.Combine(packDir, ModelManifestStore.FileName))); // the pin stands untouched
        Assert.Equal(PinnedCommit, Assert.Single(ModelManifestStore.Load(packDir)!.Models).ResolvedRevision);
        Assert.Null(engine.Metrics!.SnapshotRing().FirstOrDefault()?.UpdateCandidate); // no candidate fact from a failed check
    }

    [Fact]
    public async Task UpdateCandidateReporting_LeavesManifestUntouched_AndDownloadsNothing()
    {
        string packDir = TempPackDir();
        byte[] modelBytes = StageVerifiedPack(packDir, PinnedCommit);
        string manifestBefore = File.ReadAllText(Path.Combine(packDir, ModelManifestStore.FileName));

        FakeTransport transport = new();
        transport.HeadSha = HeadCommit;
        transport.HeadTree =
        [
            new HubTreeEntry("model.gguf", modelBytes.Length + 1, Sha256Hex("changed bytes"u8.ToArray())),
            new HubTreeEntry("extra.gguf", 7),
        ];

        using GenerationService engine = OpenServedModel(packDir, transport);
        _ = NewestSnapshotWithCandidate(engine);

        Assert.Equal(0, transport.OpenCalls); // candidate reporting never opens a download
        Assert.Equal(0, transport.ResolveCalls); // nor does it re-resolve a pin
        Assert.Equal(manifestBefore, File.ReadAllText(Path.Combine(packDir, ModelManifestStore.FileName))); // the manifest pin is byte-for-byte unchanged
        Assert.Equal(PinnedCommit, Assert.Single(ModelManifestStore.Load(packDir)!.Models).ResolvedRevision);
        Assert.False(File.Exists(Path.Combine(packDir, "extra.gguf"))); // a head-only file never lands in the pack
    }

    [Fact]
    public async Task CliCheck_UsesSharedCandidateSummary_WithoutBehaviorRegression()
    {
        string packDir = TempPackDir();
        byte[] modelBytes = StageVerifiedPack(packDir, PinnedCommit);
        string manifestBefore = File.ReadAllText(Path.Combine(packDir, ModelManifestStore.FileName));

        FakeTransport transport = new();
        transport.HeadSha = HeadCommit;
        transport.HeadTree =
        [
            new HubTreeEntry("model.gguf", modelBytes.Length + 1, Sha256Hex("changed bytes"u8.ToArray())),
            new HubTreeEntry("extra.gguf", 7),
        ];

        StringWriter output = new();
        int exit = await new ModelCliRunner(transport, packDir, output)
            .RunAsync(["check"], CancellationToken.None);

        Assert.Equal(0, exit); // a candidate delta is a completed check, not a refusal
        string text = output.ToString();
        Assert.Contains($"revision delta: {PinnedCommit} -> {HeadCommit}", text);
        Assert.Contains("hash delta: 'model.gguf'", text);
        Assert.Contains("size delta: 'model.gguf'", text);
        Assert.Contains("file delta: 'extra.gguf'", text);
        Assert.Contains("candidate:", text);
        Assert.Contains("candidate deltas: 4", text);

        Assert.Equal(0, transport.OpenCalls); // reporting never downloads through the CLI either
        Assert.Equal(manifestBefore, File.ReadAllText(Path.Combine(packDir, ModelManifestStore.FileName)));
    }

    // Opens the served model through the live load gate + candidate seam against the stand-in transport, then attaches
    // a metrics bus on the same facts path a serving host wires through ApiApp.Build.
    private static GenerationService OpenServedModel(string packDir, FakeTransport transport)
    {
        LairConfig config = new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig { Threads = 1 },
            Models =
            [
                new ModelConfig
                {
                    Name = ModelName,
                    Path = Path.Combine(packDir, "model.gguf"),
                    MaxContextTokens = 512,
                    KvBlocks = 8,
                    Source = new ModelSourceConfig
                    {
                        Repo = Repo,
                        Revision = PinnedCommit,
                        Files = [new SourceFileConfig { Path = "model.gguf" }],
                    },
                },
            ],
        };

        GenerationService engine = GenerationService.Open(config, NullLogger.Instance, ledger: null, compute: null, transport);
        engine.AttachMetrics(NullLoggerFactory.Instance, TimeSpan.FromMilliseconds(50));
        return engine;
    }

    // Stages a pack whose pinned file is present on disk and matches its manifest sha, so the load gate verifies with
    // no transport call and only the report-only candidate check reaches the seam. Returns the model bytes.
    private byte[] StageVerifiedPack(string packDir, string resolvedRevision)
    {
        string syntheticPath = SyntheticLlamaGguf.Write(ModelName);
        _tempFiles.Add(syntheticPath);
        byte[] modelBytes = File.ReadAllBytes(syntheticPath);

        string modelPath = Path.Combine(packDir, "model.gguf");
        File.WriteAllBytes(modelPath, modelBytes);

        ModelManifestStore.Save(packDir, new ModelManifest
        {
            Models =
            [
                new ModelManifestEntry
                {
                    Model = ModelName,
                    Repo = Repo,
                    ResolvedRevision = resolvedRevision,
                    Files = [new ModelManifestFile { Path = "model.gguf", Sha256 = Sha256Hex(modelBytes), Size = modelBytes.Length, FetchedAt = DateTimeOffset.UnixEpoch }],
                },
            ],
        });

        return modelBytes;
    }

    private static MetricsSnapshot NewestSnapshotAfterLoad(GenerationService engine)
    {
        _ = engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = "hello world",
            MaxTokens = 8,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult();

        EngineMetricsBus bus = engine.Metrics!;
        for (int attempt = 0; attempt < 60; attempt++)
        {
            IReadOnlyList<MetricsSnapshot> ring = bus.SnapshotRing();
            if (ring.Count > 0)
            {
                return ring[0]; // newest first: the closing tick after a completed load publishes even once the engine goes quiet
            }

            Thread.Sleep(50);
        }

        throw new InvalidOperationException("the bus published no snapshots after a completed generation");
    }

    private static MetricsSnapshot NewestSnapshotWithCandidate(GenerationService engine)
    {
        _ = engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = "hello world",
            MaxTokens = 8,
            Temperature = 0f,
        }, CancellationToken.None).GetAwaiter().GetResult();

        EngineMetricsBus bus = engine.Metrics!;
        for (int attempt = 0; attempt < 60; attempt++)
        {
            foreach (MetricsSnapshot snapshot in bus.SnapshotRing())
            {
                if (snapshot.UpdateCandidate is not null)
                {
                    return snapshot;
                }
            }

            Thread.Sleep(50);
        }

        throw new InvalidOperationException("the bus published no snapshot carrying an update candidate");
    }

    private string TempPackDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"lair-candidate-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);
        return dir;
    }

    private static string Sha256Hex(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private sealed class FakeTransport : IModelHubTransport
    {
        public string HeadSha { get; set; } = HeadCommit;
        public List<HubTreeEntry> HeadTree { get; set; } = [];
        public bool HeadFails { get; set; }
        public int ResolveCalls { get; private set; }
        public int OpenCalls { get; private set; }
        public int HeadCalls { get; private set; }

        public Task<string> ResolveRevisionAsync(string repo, string revision, CancellationToken cancellationToken)
        {
            ResolveCalls++;
            return Task.FromResult(revision);
        }

        public Task<HubFile?> OpenFileAsync(string repo, string revision, string path, long resumeFrom, CancellationToken cancellationToken)
        {
            OpenCalls++;
            return Task.FromResult<HubFile?>(null);
        }

        public Task<string> ResolveHeadRevisionAsync(string repo, CancellationToken cancellationToken)
        {
            HeadCalls++;

            if (HeadFails)
            {
                throw new HttpRequestException("hub unreachable");
            }

            return Task.FromResult(HeadSha);
        }

        public Task<IReadOnlyList<HubTreeEntry>> ListTreeAsync(string repo, string revision, CancellationToken cancellationToken) =>
            Task.FromResult<IReadOnlyList<HubTreeEntry>>(HeadTree);
    }
}
