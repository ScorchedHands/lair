using System.Security.Cryptography;
using Lair.Host.Config;

namespace Lair.Tests;

/// <summary>
/// The fetch path over the transport seam: a declared revision is used directly when it is already a commit sha
/// and resolved once through the seam when it is a branch; a file downloads to a <c>.part</c> that resumes from
/// its current length and only becomes the final, loadable name after its whole body lands and LAIR's own sha256
/// agrees with the pin; a hash disagreement refuses naming repo, revision, file, expected, and actual; a
/// per-<c>(repo, revision, path)</c> lock marker serializes concurrent fetches and reclaims a marker left by a
/// killed fetch; and a declared file set is fetched coherently so a missing shard refuses the whole set rather
/// than recording a partial entry as loadable. Every case runs against a local stand-in transport, never the live
/// hub.
/// </summary>
[Trait("Category", "Loader")]
[Trait("Category", "Config")]
public sealed class ModelFetchTests : IDisposable
{
    private const string CommitSha = "0123456789abcdef0123456789abcdef01234567";
    private const string PinnedSha = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    private const string ResolvedCommit = "1234567890123456789012345678901234567890";
    private const string Repo = "bartowski/Fetch-Fixture";

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
    public async Task Fetch_SingleFile_WritesFinalFileAndManifestEntry()
    {
        byte[] content = "hello lair fetch"u8.ToArray();
        string expected = Sha256Hex(content);
        string packDir = TempPackDir();

        FakeTransport transport = new();
        transport.Files["model.gguf"] = content;

        ModelManifestEntry entry = await new ModelFetcher(transport).FetchAsync(
            ModelWithSource(CommitSha, new SourceFileConfig { Path = "model.gguf", Sha256 = expected, Size = content.Length }),
            packDir,
            CancellationToken.None);

        string finalPath = Path.Combine(packDir, "model.gguf");
        Assert.True(File.Exists(finalPath));
        Assert.Equal(content, File.ReadAllBytes(finalPath));
        Assert.False(File.Exists(finalPath + ".part"));

        Assert.Equal("fetch-model", entry.Model);
        Assert.Equal(Repo, entry.Repo);
        Assert.Equal(CommitSha, entry.ResolvedRevision);

        ModelManifestFile file = Assert.Single(entry.Files);
        Assert.Equal("model.gguf", file.Path);
        Assert.Equal(expected, file.Sha256);
        Assert.Equal((long)content.Length, file.Size);
        Assert.Null(file.SplitTensorsCount); // arbitrary bytes are not a GGUF: no split count is invented.

        ModelManifest recorded = ModelManifestStore.Load(packDir)!;
        Assert.Equal(expected, Assert.Single(Assert.Single(recorded.Models).Files).Sha256);
    }

    [Fact]
    public async Task Fetch_ResumesFromExistingPartLength()
    {
        byte[] content = DeterministicBytes(1000);
        string expected = Sha256Hex(content);
        string packDir = TempPackDir();

        string finalPath = Path.Combine(packDir, "model.gguf");
        string partPath = finalPath + ".part";
        int prefix = 400;
        File.WriteAllBytes(partPath, content[..prefix]); // an interrupted transfer left a partial file behind.

        FakeTransport transport = new();
        transport.Files["model.gguf"] = content;

        ModelManifestEntry entry = await new ModelFetcher(transport).FetchAsync(
            ModelWithSource(CommitSha, new SourceFileConfig { Path = "model.gguf", Sha256 = expected, Size = content.Length }),
            packDir,
            CancellationToken.None);

        Assert.Equal(content, File.ReadAllBytes(finalPath));
        Assert.False(File.Exists(partPath));
        Assert.Equal(1, transport.OpenCalls);
        Assert.Equal(prefix, transport.Opens[0].ResumeFrom); // the seam was asked to resume from the existing length.
        Assert.Equal(expected, entry.Files[0].Sha256);
    }

    [Fact]
    public async Task Fetch_ShaMismatch_RefusesNamingRepoRevisionFileExpectedAndActual()
    {
        byte[] content = "wrong bytes entirely"u8.ToArray();
        string actual = Sha256Hex(content);
        string packDir = TempPackDir();

        FakeTransport transport = new();
        transport.Files["model.gguf"] = content;

        ModelFetchException error = await Assert.ThrowsAsync<ModelFetchException>(() => new ModelFetcher(transport).FetchAsync(
            ModelWithSource(CommitSha, new SourceFileConfig { Path = "model.gguf", Sha256 = PinnedSha, Size = content.Length }),
            packDir,
            CancellationToken.None));

        Assert.Contains(Repo, error.Message);
        Assert.Contains(CommitSha, error.Message); // the resolved revision the bytes came from.
        Assert.Contains("model.gguf", error.Message);
        Assert.Contains($"expected {PinnedSha}", error.Message);
        Assert.Contains($"actual {actual}", error.Message);

        Assert.False(File.Exists(Path.Combine(packDir, "model.gguf"))); // never finalized to a loadable name.
        Assert.False(File.Exists(Path.Combine(packDir, "model.gguf") + ".part")); // the disagreeing partial is dropped.
        Assert.Null(ModelManifestStore.Load(packDir));
    }

    [Fact]
    public async Task Fetch_NeverRecordsPartAsTheLoadablePath()
    {
        byte[] content = "complete file body"u8.ToArray();
        string expected = Sha256Hex(content);
        string packDir = TempPackDir();

        string finalPath = Path.Combine(packDir, "model.gguf");
        File.WriteAllBytes(finalPath + ".part", content[..3]); // only a partial exists before the fetch.

        FakeTransport transport = new();
        transport.Files["model.gguf"] = content;

        ModelManifestEntry entry = await new ModelFetcher(transport).FetchAsync(
            ModelWithSource(CommitSha, new SourceFileConfig { Path = "model.gguf", Sha256 = expected, Size = content.Length }),
            packDir,
            CancellationToken.None);

        Assert.Equal("model.gguf", entry.Files[0].Path); // the manifest names the final file, never the .part sibling.
        Assert.False(entry.Files[0].Path.EndsWith(".part", StringComparison.Ordinal));
        Assert.True(File.Exists(finalPath));
        Assert.False(File.Exists(finalPath + ".part"));
        Assert.Equal(content, File.ReadAllBytes(finalPath));
    }

    [Fact]
    public async Task Fetch_ReclaimsStaleLockFromAKilledFetch()
    {
        byte[] content = DeterministicBytes(64);
        string expected = Sha256Hex(content);
        string packDir = TempPackDir();
        const string path = "model.gguf";

        // A prior fetch was killed mid-download and left its marker behind under a process that no longer exists.
        string lockFile = ModelFetcher.LockFilePath(packDir, Repo, CommitSha, path);
        Directory.CreateDirectory(Path.GetDirectoryName(lockFile)!);
        File.WriteAllText(lockFile, "2147483646"); // a pid no live process carries.

        FakeTransport transport = new();
        transport.Files[path] = content;

        ModelManifestEntry entry = await new ModelFetcher(transport).FetchAsync(
            ModelWithSource(CommitSha, new SourceFileConfig { Path = path, Sha256 = expected, Size = content.Length }),
            packDir,
            CancellationToken.None);

        Assert.Equal(content, File.ReadAllBytes(Path.Combine(packDir, path)));
        Assert.Equal(1, transport.OpenCalls); // reclaimed, then fetched once — no double-stream.
        Assert.False(File.Exists(lockFile)); // the reclaimed marker is released at the end of the fetch.
        Assert.Equal(expected, entry.Files[0].Sha256);
    }

    [Fact]
    public async Task Fetch_PartialShardSet_RefusesTheWholeSet()
    {
        string packDir = TempPackDir();

        FakeTransport transport = new();
        transport.Files["shard-00001-of-00002.gguf"] = "first shard bytes"u8.ToArray();
        transport.Missing.Add("shard-00002-of-00002.gguf"); // the hub carries no such file at this revision.

        ModelConfig model = ModelWithSource(
            CommitSha,
            new SourceFileConfig { Path = "shard-00001-of-00002.gguf" },
            new SourceFileConfig { Path = "shard-00002-of-00002.gguf" });

        ModelFetchException error = await Assert.ThrowsAsync<ModelFetchException>(() => new ModelFetcher(transport).FetchAsync(
            model,
            packDir,
            CancellationToken.None));

        Assert.Contains("shard-00002-of-00002.gguf", error.Message);
        Assert.Contains("not present", error.Message);

        // The first shard landed on disk, but the set is refused as a whole: no partial entry is left loadable.
        Assert.Null(ModelManifestStore.Load(packDir));
    }

    [Fact]
    public async Task Fetch_CompleteShardSet_RecordsTheShardListAndSplitCount()
    {
        string packDir = TempPackDir();

        byte[] first = GgufWithSplitCount(2);
        byte[] second = GgufWithSplitCount(2);

        FakeTransport transport = new();
        transport.Files["model-00001-of-00002.gguf"] = first;
        transport.Files["model-00002-of-00002.gguf"] = second;

        ModelManifestEntry entry = await new ModelFetcher(transport).FetchAsync(
            ModelWithSource(CommitSha, new SourceFileConfig { Path = "model-00001-of-00002.gguf" }, new SourceFileConfig { Path = "model-00002-of-00002.gguf" }),
            packDir,
            CancellationToken.None);

        Assert.True(entry.Sharded);
        Assert.Equal(2, entry.Files.Count);
        Assert.Equal(2L, entry.Files[0].SplitTensorsCount); // read straight from each shard's GGUF metadata.
        Assert.Equal(2L, entry.Files[1].SplitTensorsCount);
        Assert.True(File.Exists(Path.Combine(packDir, "model-00001-of-00002.gguf")));
        Assert.True(File.Exists(Path.Combine(packDir, "model-00002-of-00002.gguf")));

        ModelManifest recorded = ModelManifestStore.Load(packDir)!;
        Assert.True(Assert.Single(recorded.Models).Sharded);
    }

    [Fact]
    public async Task Fetch_BranchRevision_ResolvesOnceAndRecordsResolvedCommitSha()
    {
        byte[] content = DeterministicBytes(48);
        string expected = Sha256Hex(content);
        string packDir = TempPackDir();

        FakeTransport transport = new();
        transport.ResolvedSha = ResolvedCommit;
        transport.Files["model.gguf"] = content;

        ModelManifestEntry entry = await new ModelFetcher(transport).FetchAsync(
            ModelWithSource("main", new SourceFileConfig { Path = "model.gguf", Sha256 = expected, Size = content.Length }),
            packDir,
            CancellationToken.None);

        Assert.Equal(1, transport.ResolveCalls); // a branch resolves exactly once for the whole fetch.
        Assert.Equal(ResolvedCommit, entry.ResolvedRevision); // the manifest records the resolved commit, not the branch.
        Assert.Equal(ResolvedCommit, transport.Opens[0].Revision); // the download ran against the resolved commit.
        Assert.Equal(expected, entry.Files[0].Sha256);
    }

    [Fact]
    public async Task Fetch_CommitShaRevision_NeverResolvesThroughTheSeam()
    {
        byte[] content = DeterministicBytes(48);
        string expected = Sha256Hex(content);
        string packDir = TempPackDir();

        FakeTransport transport = new();
        transport.Files["model.gguf"] = content;

        ModelManifestEntry entry = await new ModelFetcher(transport).FetchAsync(
            ModelWithSource(CommitSha, new SourceFileConfig { Path = "model.gguf", Sha256 = expected, Size = content.Length }),
            packDir,
            CancellationToken.None);

        Assert.Equal(0, transport.ResolveCalls); // a declared sha is already the pin: it is never sent to the seam.
        Assert.Equal(CommitSha, entry.ResolvedRevision);
    }

    [Fact]
    public async Task Fetch_ConcurrentFetchesForOneArtifact_DoNotDoubleStream()
    {
        byte[] content = DeterministicBytes(512);
        string expected = Sha256Hex(content);
        string packDir = TempPackDir();

        FakeTransport transport = new();
        transport.Files["model.gguf"] = content;

        ModelConfig model = ModelWithSource(CommitSha, new SourceFileConfig { Path = "model.gguf", Sha256 = expected, Size = content.Length });
        ModelFetcher fetcher = new(transport);

        Task<ModelManifestEntry>[] concurrent =
        [
            fetcher.FetchAsync(model, packDir, CancellationToken.None),
            fetcher.FetchAsync(model, packDir, CancellationToken.None),
        ];
        await Task.WhenAll(concurrent);

        Assert.Equal(1, transport.OpenCalls); // the second fetch waited on the lock and recorded the landed file.
        Assert.Equal(content, File.ReadAllBytes(Path.Combine(packDir, "model.gguf")));
        Assert.False(File.Exists(Path.Combine(packDir, "model.gguf") + ".part"));
    }

    private static ModelConfig ModelWithSource(string revision, params SourceFileConfig[] files)
    {
        string unusedPath = Path.Combine(Path.GetTempPath(), $"lair-fetch-unused-{Guid.NewGuid():N}.gguf");
        return new ModelConfig
        {
            Name = "fetch-model",
            Path = unusedPath,
            Source = new ModelSourceConfig { Repo = Repo, Revision = revision, Files = [.. files] },
        };
    }

    private string TempPackDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"lair-fetch-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        _tempDirs.Add(dir);
        return dir;
    }

    private static byte[] DeterministicBytes(int length)
    {
        byte[] bytes = new byte[length];
        for (int i = 0; i < length; i++)
        {
            bytes[i] = (byte)((i * 131 + 7) % 251);
        }

        return bytes;
    }

    private static string Sha256Hex(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    // A minimal valid GGUF carrying only the split.tensors.count key, so the fetch path can read it back.
    private static byte[] GgufWithSplitCount(ulong splitCount)
    {
        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, System.Text.Encoding.UTF8, leaveOpen: true);

        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u);          // version
        writer.Write(0ul);         // no tensors
        writer.Write(1ul);         // one metadata key

        byte[] key = System.Text.Encoding.UTF8.GetBytes("split.tensors.count");
        writer.Write((ulong)key.Length);
        writer.Write(key);
        writer.Write(10u);         // UINT64
        writer.Write(splitCount);

        writer.Flush();
        return stream.ToArray();
    }

    private sealed class FakeTransport : IModelHubTransport
    {
        public Dictionary<string, byte[]> Files { get; } = new(StringComparer.Ordinal);
        public HashSet<string> Missing { get; } = new(StringComparer.Ordinal);
        public string ResolvedSha { get; set; } = ResolvedCommit;
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

            if (Missing.Contains(path) || !Files.TryGetValue(path, out byte[]? full))
            {
                return Task.FromResult<HubFile?>(null);
            }

            if (resumeFrom >= full.Length)
            {
                return Task.FromResult<HubFile?>(new HubFile(Stream.Null, full.Length));
            }

            return Task.FromResult<HubFile?>(new HubFile(new MemoryStream(full[(int)resumeFrom..]), full.Length));
        }

        public Task<string> ResolveHeadRevisionAsync(string repo, CancellationToken cancellationToken) => Task.FromResult(ResolvedSha);

        public Task<IReadOnlyList<HubTreeEntry>> ListTreeAsync(string repo, string revision, CancellationToken cancellationToken) =>
            Task.FromResult<IReadOnlyList<HubTreeEntry>>([.. Files.Select(file => new HubTreeEntry(file.Key, file.Value.Length))]);
    }
}
