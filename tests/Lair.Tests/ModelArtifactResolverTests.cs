using System.Security.Cryptography;
using Lair.Host.Config;
using Lair.Http;
using Microsoft.Extensions.Logging.Abstractions;

namespace Lair.Tests;

/// <summary>
/// The load-time gate a model with a declared source passes through before its GGUF is opened: a present artifact
/// verifies against its pack manifest with no transport call, a disagreement refuses naming repo, revision, file,
/// expected, and actual; an absent pinned file fetches exactly the revision the manifest already resolved and the
/// fetched bytes are verified; a pack with no manifest entry yet takes the config source as the pin, fetches-if-absent,
/// and records the resolved sha — a branch resolving exactly once inside that fetch; and a transport failure during a
/// required fetch refuses naming repo, revision, file, and reason rather than substituting a revision or falling back
/// to an unverified file. A model with no source keeps today's local-path posture even when a transport seam is
/// supplied. Every case runs against a local stand-in transport, never the live hub.
/// </summary>
[Trait("Category", "Loader")]
[Trait("Category", "Config")]
public sealed class ModelArtifactResolverTests : IDisposable
{
    private const string CommitSha = "0123456789abcdef0123456789abcdef01234567";
    private const string PinnedSha = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    private const string ResolvedCommit = "1234567890123456789012345678901234567890";
    private const string Repo = "bartowski/Resolver-Fixture";

    private readonly List<string> _tempDirs = [];
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
    public async Task Resolve_PresentVerifiedArtifact_SucceedsWithNoTransportCall()
    {
        byte[] content = "verified pack bytes"u8.ToArray();
        string actual = Sha256Hex(content);
        string packDir = TempPackDir();
        File.WriteAllBytes(Path.Combine(packDir, "model.gguf"), content);

        ModelManifestStore.Save(packDir, ManifestWith(CommitSha, new ModelManifestFile { Path = "model.gguf", Sha256 = actual, Size = content.Length, FetchedAt = DateTimeOffset.UnixEpoch }));

        FakeTransport transport = new();
        transport.Files["model.gguf"] = content;

        await new ModelArtifactResolver(transport).ResolveAsync(ModelWithSource(packDir, CommitSha, new SourceFileConfig { Path = "model.gguf", Sha256 = actual, Size = content.Length }), CancellationToken.None);

        Assert.Equal(0, transport.ResolveCalls); // a present, matching artifact never reaches the seam.
        Assert.Equal(0, transport.OpenCalls);
    }

    [Fact]
    public async Task Resolve_PresentArtifactWrongSha_RefusesNamingRepoRevisionFileExpectedAndActual()
    {
        byte[] content = "bytes that disagree with the pin"u8.ToArray();
        string actual = Sha256Hex(content);
        string packDir = TempPackDir();
        File.WriteAllBytes(Path.Combine(packDir, "model.gguf"), content);

        ModelManifestStore.Save(packDir, ManifestWith(CommitSha, new ModelManifestFile { Path = "model.gguf", Sha256 = PinnedSha, Size = content.Length, FetchedAt = DateTimeOffset.UnixEpoch }));

        FakeTransport transport = new();
        transport.Files["model.gguf"] = content;

        ModelFetchException error = await Assert.ThrowsAsync<ModelFetchException>(() => new ModelArtifactResolver(transport).ResolveAsync(
            ModelWithSource(packDir, CommitSha, new SourceFileConfig { Path = "model.gguf", Sha256 = PinnedSha, Size = content.Length }),
            CancellationToken.None));

        Assert.Contains(Repo, error.Message);
        Assert.Contains(CommitSha, error.Message);
        Assert.Contains("model.gguf", error.Message);
        Assert.Contains($"expected {PinnedSha}", error.Message);
        Assert.Contains($"actual {actual}", error.Message);

        Assert.Equal(0, transport.ResolveCalls); // a manifest disagreement is refused before any fetch is attempted.
        Assert.Equal(0, transport.OpenCalls);
    }

    [Fact]
    public async Task Resolve_AbsentPinnedFile_FetchesManifestRevisionAndRecordsEntry()
    {
        byte[] content = "the pinned bytes fetched from the hub"u8.ToArray();
        string expected = Sha256Hex(content);
        string packDir = TempPackDir(); // the pack carries the pin but not the bytes.

        ModelManifestStore.Save(packDir, ManifestWith(CommitSha, new ModelManifestFile { Path = "model.gguf", Sha256 = expected, Size = content.Length, FetchedAt = DateTimeOffset.UnixEpoch }));

        FakeTransport transport = new();
        transport.Files["model.gguf"] = content;

        await new ModelArtifactResolver(transport).ResolveAsync(
            ModelWithSource(packDir, CommitSha, new SourceFileConfig { Path = "model.gguf", Sha256 = expected, Size = content.Length }),
            CancellationToken.None);

        string finalPath = Path.Combine(packDir, "model.gguf");
        Assert.True(File.Exists(finalPath));
        Assert.Equal(content, File.ReadAllBytes(finalPath));

        Assert.Equal(1, transport.OpenCalls); // the missing file was fetched.
        Assert.Equal(0, transport.ResolveCalls); // the manifest already resolved a commit sha: it is used as-is.
        Assert.Equal(CommitSha, transport.Opens[0].Revision); // fetched at exactly the manifest revision.

        ModelManifestEntry entry = Assert.Single(ModelManifestStore.Load(packDir)!.Models);
        Assert.Equal(CommitSha, entry.ResolvedRevision);
        Assert.Equal(expected, Assert.Single(entry.Files).Sha256);
    }

    [Fact]
    public async Task Resolve_BranchRevisionInConfig_ResolvesOnceAndRecordsResolvedCommitSha()
    {
        byte[] content = "bytes fetched behind a branch pin"u8.ToArray();
        string expected = Sha256Hex(content);
        string packDir = TempPackDir(); // no manifest yet: the config source is the pin.

        FakeTransport transport = new();
        transport.ResolvedSha = ResolvedCommit;
        transport.Files["model.gguf"] = content;

        await new ModelArtifactResolver(transport).ResolveAsync(
            ModelWithSource(packDir, "main", new SourceFileConfig { Path = "model.gguf", Sha256 = expected, Size = content.Length }),
            CancellationToken.None);

        Assert.Equal(1, transport.ResolveCalls); // the declared branch resolves exactly once for the whole fetch.
        Assert.Equal(ResolvedCommit, transport.Opens[0].Revision); // the download ran against the resolved commit.

        ModelManifestEntry entry = Assert.Single(ModelManifestStore.Load(packDir)!.Models);
        Assert.Equal(ResolvedCommit, entry.ResolvedRevision); // the manifest records the resolved sha, not the branch.
        Assert.Equal(expected, Assert.Single(entry.Files).Sha256);
    }

    [Fact]
    public async Task Resolve_TransportFailureDuringRequiredFetch_RefusesNamingRepoRevisionFileAndReason()
    {
        string packDir = TempPackDir(); // no manifest, no bytes: a fetch is required and the seam fails.

        FakeTransport transport = new();
        transport.ThrowOnOpen = new HttpRequestException("connection reset by peer");

        ModelFetchException error = await Assert.ThrowsAsync<ModelFetchException>(() => new ModelArtifactResolver(transport).ResolveAsync(
            ModelWithSource(packDir, CommitSha, new SourceFileConfig { Path = "model.gguf" }),
            CancellationToken.None));

        Assert.Contains(Repo, error.Message);
        Assert.Contains(CommitSha, error.Message);
        Assert.Contains("model.gguf", error.Message);
        Assert.Contains("connection reset by peer", error.Message); // the reason the fetch refused.

        Assert.False(File.Exists(Path.Combine(packDir, "model.gguf"))); // nothing half-fetched is left loadable.
        Assert.Null(ModelManifestStore.Load(packDir)); // a refused fetch records no pin.
    }

    [Fact]
    public async Task Resolve_ModelWithoutSource_SkipsManifestAndTransportEntirely()
    {
        string packDir = TempPackDir();
        string modelPath = Path.Combine(packDir, "plain.gguf");
        File.WriteAllBytes(modelPath, "local bytes"u8.ToArray());

        FakeTransport transport = new();

        ModelConfig model = new() { Name = "plain", Path = modelPath };
        await new ModelArtifactResolver(transport).ResolveAsync(model, CancellationToken.None);

        Assert.Equal(0, transport.ResolveCalls);
        Assert.Equal(0, transport.OpenCalls); // no source block: no manifest read, no seam.
    }

    [Fact]
    public async Task Serve_PresentVerifiedArtifact_SucceedsWithNoTransportCall()
    {
        string packDir = TempPackDir();
        byte[] content = SyntheticLlamaBytes();
        string actual = Sha256Hex(content);
        string modelPath = Path.Combine(packDir, "model.gguf");
        File.WriteAllBytes(modelPath, content);

        ModelManifestStore.Save(packDir, ManifestWith(CommitSha, new ModelManifestFile { Path = "model.gguf", Sha256 = actual, Size = content.Length, FetchedAt = DateTimeOffset.UnixEpoch }));

        FakeTransport transport = new();
        transport.Files["model.gguf"] = content;

        LairConfig config = new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig(),
            Models =
            [
                new ModelConfig
                {
                    Name = "served-verified",
                    Path = modelPath,
                    MaxContextTokens = 512,
                    KvBlocks = 8,
                    Source = new ModelSourceConfig { Repo = Repo, Revision = CommitSha, Files = [new SourceFileConfig { Path = "model.gguf", Sha256 = actual, Size = content.Length }] },
                },
            ],
        };

        using GenerationService engine = GenerationService.Open(config, NullLogger.Instance, ledger: null, compute: null, transport);

        Assert.Equal(0, transport.ResolveCalls); // the open verified the present artifact without reaching the hub.
        Assert.Equal(0, transport.OpenCalls);

        GenerationResult served = await engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = "hello world",
            MaxTokens = 8,
            Temperature = 0f,
        }, CancellationToken.None);

        Assert.False(string.IsNullOrEmpty(served.Text));
    }

    [Fact]
    public async Task Serve_ModelWithoutSource_IgnoresTheTransportSeam()
    {
        string packDir = TempPackDir();
        byte[] content = SyntheticLlamaBytes();
        string modelPath = Path.Combine(packDir, "plain.gguf");
        File.WriteAllBytes(modelPath, content);

        FakeTransport transport = new();

        LairConfig config = new()
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig(),
            Models =
            [
                new ModelConfig { Name = "served-plain", Path = modelPath, MaxContextTokens = 512, KvBlocks = 8 },
            ],
        };

        using GenerationService engine = GenerationService.Open(config, NullLogger.Instance, ledger: null, compute: null, transport);

        Assert.Equal(0, transport.ResolveCalls);
        Assert.Equal(0, transport.OpenCalls); // a source-less model loads exactly as before, even with a seam supplied.

        GenerationResult served = await engine.GenerateAsync(new GenerationRequest
        {
            Model = engine.Model.Id,
            IsChat = false,
            Prompt = "hello world",
            MaxTokens = 8,
            Temperature = 0f,
        }, CancellationToken.None);

        Assert.False(string.IsNullOrEmpty(served.Text));
    }

    private static ModelConfig ModelWithSource(string packDir, string revision, params SourceFileConfig[] files)
    {
        // The pack directory is the directory holding the model file, so the model path lives beside the manifest.
        return new ModelConfig
        {
            Name = "resolver-model",
            Path = Path.Combine(packDir, "model.gguf"),
            Source = new ModelSourceConfig { Repo = Repo, Revision = revision, Files = [.. files] },
        };
    }

    private static ModelManifest ManifestWith(string resolvedRevision, params ModelManifestFile[] files) =>
        new()
        {
            Models =
            [
                new ModelManifestEntry
                {
                    Model = "resolver-model",
                    Repo = Repo,
                    ResolvedRevision = resolvedRevision,
                    Files = [.. files],
                },
            ],
        };

    private string TempPackDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"lair-resolver-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);
        return dir;
    }

    // A synthetic llama-shaped GGUF written into a temp file, returned as bytes so a test can stage it inside its own pack directory.
    private byte[] SyntheticLlamaBytes()
    {
        string path = SyntheticLlamaGguf.Write("resolver-serve");
        _tempPaths.Add(path);
        return File.ReadAllBytes(path);
    }

    private static string Sha256Hex(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private sealed class FakeTransport : IModelHubTransport
    {
        public Dictionary<string, byte[]> Files { get; } = new(StringComparer.Ordinal);
        public string ResolvedSha { get; set; } = ResolvedCommit;
        public Exception? ThrowOnOpen { get; set; }
        public int ResolveCalls { get; private set; }
        public int OpenCalls { get; private set; }
        public List<(string Path, long ResumeFrom, string Revision)> Opens { get; } = [];

        public Task<string> ResolveRevisionAsync(string repo, string revision, CancellationToken cancellationToken)
        {
            ResolveCalls++;
            return Task.FromResult(ResolvedSha);
        }

        public Task<HubFile?> OpenFileAsync(string repo, string revision, string path, long resumeFrom, CancellationToken cancellationToken)
        {
            OpenCalls++;
            Opens.Add((path, resumeFrom, revision));

            if (ThrowOnOpen is not null)
            {
                throw ThrowOnOpen;
            }

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

        // The resolver never consults HEAD or a tree listing; these stand in for the seam's reporting-only members.
        public Task<string> ResolveHeadRevisionAsync(string repo, CancellationToken cancellationToken) => Task.FromResult(ResolvedSha);

        public Task<IReadOnlyList<HubTreeEntry>> ListTreeAsync(string repo, string revision, CancellationToken cancellationToken) =>
            Task.FromResult<IReadOnlyList<HubTreeEntry>>([.. Files.Select(file => new HubTreeEntry(file.Key, file.Value.Length))]);
    }
}
