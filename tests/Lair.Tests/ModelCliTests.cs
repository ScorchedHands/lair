using System.Net.Http;
using System.Security.Cryptography;
using Lair.Host.Config;

namespace Lair.Tests;

/// <summary>
/// The model CLI surface driven through an injected transport, pack directory, and output writer: <c>list</c>
/// prints what a pack pins, <c>fetch</c> lands declared files through the fetch path and records the pin —
/// idempotently, no-oping on an already verified file rather than double-streaming — and <c>check</c> reports
/// candidate deltas against a mocked newer HEAD without downloading, repinning, or rewriting a manifest. Exit
/// codes are bounded 0/1/2, and every case runs against a local stand-in transport, never the live hub.
/// </summary>
[Trait("Category", "Cli")]
[Trait("Category", "Config")]
public sealed class ModelCliTests : IDisposable
{
    private const string Repo = "bartowski/Cli-Fixture";
    private const string RepoName = "Cli-Fixture";
    private const string ModelName = "cli-model";
    private const string PinnedCommit = "0123456789abcdef0123456789abcdef01234567";
    private const string HeadCommit = "ffffffffffffffffffffffffffffffffffffffff";
    private const string WrongPin = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

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
    public async Task List_NoManifest_ExitsZeroWithClearStatus()
    {
        string packDir = TempPackDir();

        (int exit, string output) = await RunAsync(packDir, new FakeTransport(), "list");

        Assert.Equal(0, exit);
        Assert.Contains("no manifest", output);
    }

    [Fact]
    public async Task List_EmptyManifest_ExitsZeroWithClearStatus()
    {
        string packDir = TempPackDir();
        ModelManifestStore.Save(packDir, new ModelManifest { Models = [] });

        (int exit, string output) = await RunAsync(packDir, new FakeTransport(), "list");

        Assert.Equal(0, exit);
        Assert.Contains("no model entries", output);
    }

    [Fact]
    public async Task List_PrintsManifestFields()
    {
        string packDir = TempPackDir();
        DateTimeOffset fetchedAt = new(2026, 11, 14, 8, 30, 0, TimeSpan.Zero);
        WriteManifest(packDir, PinnedCommit, new ModelManifestFile
        {
            Path = "model.gguf",
            Sha256 = WrongPin,
            Size = 1024,
            FetchedAt = fetchedAt,
        });

        (int exit, string output) = await RunAsync(packDir, new FakeTransport(), "list");

        Assert.Equal(0, exit);
        Assert.Contains($"model: {ModelName}", output);
        Assert.Contains($"repo: {Repo}", output);
        Assert.Contains($"revision: {PinnedCommit}", output);
        Assert.Contains("file: model.gguf", output);
        Assert.Contains($"sha256={WrongPin}", output);
        Assert.Contains("size=1024", output);
        Assert.Contains($"fetched_at={fetchedAt.ToString("O")}", output);
    }

    [Fact]
    public async Task Fetch_RecordsManifestEntry()
    {
        byte[] content = "hello cli fetch"u8.ToArray();
        string expected = Sha256Hex(content);
        string packDir = TempPackDir();

        FakeTransport transport = new();
        transport.Files["model.gguf"] = content;

        (int exit, string output) = await RunAsync(packDir, transport, $"fetch {Repo} --file model.gguf");

        Assert.Equal(0, exit);
        Assert.Contains("fetched", output);
        Assert.True(File.Exists(Path.Combine(packDir, "model.gguf")));

        ModelManifestEntry entry = Assert.Single(ModelManifestStore.Load(packDir)!.Models);
        Assert.Equal(RepoName, entry.Model); // a standalone fetch names the pack entry after the repo.
        Assert.Equal(Repo, entry.Repo);
        Assert.Equal(PinnedCommit, entry.ResolvedRevision); // the declared branch resolved through the seam.
        Assert.Equal(expected, Assert.Single(entry.Files).Sha256);
    }

    [Fact]
    public async Task Fetch_IdempotentOnExistingVerifiedFile_NoDoubleDownload()
    {
        byte[] content = "already fetched bytes"u8.ToArray();
        string packDir = TempPackDir();

        FakeTransport transport = new();
        transport.Files["model.gguf"] = content;

        (int first, _) = await RunAsync(packDir, transport, $"fetch {Repo} --file model.gguf");
        (int second, _) = await RunAsync(packDir, transport, $"fetch {Repo} --file model.gguf");

        Assert.Equal(0, first);
        Assert.Equal(0, second);
        Assert.Equal(1, transport.OpenCalls); // the second run verified the landed file instead of re-streaming it.
        Assert.Equal(content, File.ReadAllBytes(Path.Combine(packDir, "model.gguf")));
    }

    [Fact]
    public async Task Fetch_PinMismatch_ExitsOneAndRecordsNothing()
    {
        byte[] content = "wrong bytes entirely"u8.ToArray();
        string packDir = TempPackDir();

        FakeTransport transport = new();
        transport.Files["model.gguf"] = content;

        (int exit, string output) = await RunAsync(packDir, transport, $"fetch {Repo} --model {ModelName}", ConfigWith(packDir, "main", WrongPin, "model.gguf"));

        Assert.Equal(1, exit);
        Assert.Contains("does not match its pinned sha256", output);
        Assert.False(File.Exists(Path.Combine(packDir, "model.gguf")));
        Assert.Null(ModelManifestStore.Load(packDir));
    }

    [Fact]
    public async Task Fetch_RevisionOverrideAgainstDeclaredPin_ExitsOne()
    {
        string packDir = TempPackDir();

        FakeTransport transport = new();
        transport.Files["model.gguf"] = "irrelevant"u8.ToArray();

        (int exit, string output) = await RunAsync(packDir, transport, $"fetch {Repo} --model {ModelName} --revision {PinnedCommit}", ConfigWith(packDir, "main", null, "model.gguf"));

        Assert.Equal(1, exit);
        Assert.Contains($"refusing to fetch it at '{PinnedCommit}'", output);
        Assert.Equal(0, transport.OpenCalls); // refused before anything streamed.
    }

    [Fact]
    public async Task Fetch_UndeclaredFile_ExitsOne()
    {
        string packDir = TempPackDir();

        FakeTransport transport = new();
        transport.Files["other.gguf"] = "irrelevant"u8.ToArray();

        (int exit, string output) = await RunAsync(packDir, transport, $"fetch {Repo} --model {ModelName} --file other.gguf", ConfigWith(packDir, "main", null, "model.gguf"));

        Assert.Equal(1, exit);
        Assert.Contains("not declared", output);
        Assert.Equal(0, transport.OpenCalls);
    }

    [Fact]
    public async Task Check_ReportsCandidateDeltas_AndLeavesManifestUntouched()
    {
        byte[] pinned = "pinned bytes"u8.ToArray();
        string pinnedHash = Sha256Hex(pinned);
        string packDir = TempPackDir();
        WriteManifest(packDir, PinnedCommit, new ModelManifestFile
        {
            Path = "model.gguf",
            Sha256 = pinnedHash,
            Size = pinned.Length,
            FetchedAt = DateTimeOffset.UtcNow,
        });

        string manifestBefore = File.ReadAllText(Path.Combine(packDir, ModelManifestStore.FileName));

        FakeTransport transport = new();
        transport.HeadSha = HeadCommit;
        transport.HeadTree =
        [
            new HubTreeEntry("model.gguf", pinned.Length + 1, Sha256Hex("changed bytes"u8.ToArray())),
            new HubTreeEntry("extra.gguf", 7),
        ];

        (int exit, string output) = await RunAsync(packDir, transport, "check");

        Assert.Equal(0, exit); // a candidate delta is a completed check, not a refusal.
        Assert.Contains($"revision delta: {PinnedCommit} -> {HeadCommit}", output);
        Assert.Contains("hash delta: 'model.gguf'", output);
        Assert.Contains("size delta: 'model.gguf'", output);
        Assert.Contains("file delta: 'extra.gguf'", output);
        Assert.Contains("candidate:", output);

        Assert.Equal(0, transport.OpenCalls); // reporting never opens a download.
        Assert.False(File.Exists(Path.Combine(packDir, "model.gguf"))); // nothing landed in the pack.
        Assert.Equal(manifestBefore, File.ReadAllText(Path.Combine(packDir, ModelManifestStore.FileName))); // the pin stands untouched.
        Assert.Equal(PinnedCommit, Assert.Single(ModelManifestStore.Load(packDir)!.Models).ResolvedRevision);
    }

    [Fact]
    public async Task Check_CurrentHead_ReportsNoCandidate()
    {
        byte[] pinned = "pinned bytes"u8.ToArray();
        string pinnedHash = Sha256Hex(pinned);
        string packDir = TempPackDir();
        WriteManifest(packDir, PinnedCommit, new ModelManifestFile
        {
            Path = "model.gguf",
            Sha256 = pinnedHash,
            Size = pinned.Length,
            FetchedAt = DateTimeOffset.UtcNow,
        });

        FakeTransport transport = new();
        transport.HeadSha = PinnedCommit;
        transport.HeadTree = [new HubTreeEntry("model.gguf", pinned.Length, pinnedHash)];

        (int exit, string output) = await RunAsync(packDir, transport, "check");

        Assert.Equal(0, exit);
        Assert.Contains("current: pinned matches head", output);
        Assert.DoesNotContain("delta", output);
        Assert.Contains("all pinned revisions current", output);
    }

    [Fact]
    public async Task Check_UnknownModel_ExitsOne()
    {
        string packDir = TempPackDir();
        WriteManifest(packDir, PinnedCommit, new ModelManifestFile
        {
            Path = "model.gguf",
            Sha256 = WrongPin,
            Size = 4,
            FetchedAt = DateTimeOffset.UtcNow,
        });

        (int exit, string output) = await RunAsync(packDir, new FakeTransport(), "check missing-model");

        Assert.Equal(1, exit);
        Assert.Contains("no manifest entry pins model 'missing-model'", output);
    }

    [Fact]
    public async Task Check_TransportFailure_ExitsOne()
    {
        string packDir = TempPackDir();
        WriteManifest(packDir, PinnedCommit, new ModelManifestFile
        {
            Path = "model.gguf",
            Sha256 = WrongPin,
            Size = 4,
            FetchedAt = DateTimeOffset.UtcNow,
        });

        FakeTransport transport = new();
        transport.HeadFails = true;

        (int exit, string output) = await RunAsync(packDir, transport, "check");

        Assert.Equal(1, exit);
        Assert.Contains("error:", output);
    }

    [Theory]
    [InlineData("")]
    [InlineData("bogus")]
    [InlineData("fetch")]
    [InlineData("fetch bartowski/Cli-Fixture --file")]
    [InlineData("fetch bartowski/Cli-Fixture --bogus")]
    [InlineData("fetch bartowski/Cli-Fixture extra")]
    [InlineData("list extra")]
    [InlineData("check model-a model-b")]
    public async Task UsageErrors_ExitTwo(string command)
    {
        string packDir = TempPackDir();

        (int exit, string output) = await RunAsync(packDir, new FakeTransport(), command);

        Assert.Equal(2, exit);
        Assert.True(output.Trim().Length > 0, $"expected a usage or error line for '{command}'");
    }

    private static async Task<(int Exit, string Output)> RunAsync(string packDir, FakeTransport transport, string command, LairConfig? config = null)
    {
        StringWriter output = new();
        int exit = await new ModelCliRunner(transport, packDir, output, config)
            .RunAsync(command.Split(' ', StringSplitOptions.RemoveEmptyEntries), CancellationToken.None);
        return (exit, output.ToString());
    }

    private static void WriteManifest(string packDir, string resolvedRevision, params ModelManifestFile[] files) =>
        ModelManifestStore.Save(packDir, new ModelManifest
        {
            Models =
            [
                new ModelManifestEntry
                {
                    Model = ModelName,
                    Repo = Repo,
                    ResolvedRevision = resolvedRevision,
                    Files = [.. files],
                },
            ],
        });

    private static LairConfig ConfigWith(string packDir, string revision, string? sha256, params string[] files) =>
        new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig(),
            Models =
            [
                new ModelConfig
                {
                    Name = ModelName,
                    Path = Path.Combine(packDir, files[0]),
                    Source = new ModelSourceConfig
                    {
                        Repo = Repo,
                        Revision = revision,
                        Files = [.. files.Select(f => new SourceFileConfig { Path = f, Sha256 = sha256 })],
                    },
                },
            ],
        };

    private string TempPackDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"lair-model-cli-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);
        return dir;
    }

    private static string Sha256Hex(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private sealed class FakeTransport : IModelHubTransport
    {
        public Dictionary<string, byte[]> Files { get; } = new(StringComparer.Ordinal);
        public string ResolvedSha { get; set; } = PinnedCommit;
        public string HeadSha { get; set; } = HeadCommit;
        public List<HubTreeEntry> HeadTree { get; set; } = [];
        public bool HeadFails { get; set; }
        public int OpenCalls { get; private set; }
        public int HeadCalls { get; private set; }

        public Task<string> ResolveRevisionAsync(string repo, string revision, CancellationToken cancellationToken) =>
            Task.FromResult(ResolvedSha);

        public Task<HubFile?> OpenFileAsync(string repo, string revision, string path, long resumeFrom, CancellationToken cancellationToken)
        {
            OpenCalls++;

            if (!Files.TryGetValue(path, out byte[]? full))
            {
                return Task.FromResult<HubFile?>(null);
            }

            if (resumeFrom >= full.Length)
            {
                return Task.FromResult<HubFile?>(new HubFile(Stream.Null, full.Length));
            }

            return Task.FromResult<HubFile?>(new HubFile(new MemoryStream(full[(int)resumeFrom..]), full.Length));
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
