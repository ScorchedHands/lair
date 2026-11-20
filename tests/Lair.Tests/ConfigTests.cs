using Lair.Host.Config;
using Lair.Models;

namespace Lair.Tests;

[Trait("Category", "Config")]
public sealed class ConfigTests : IDisposable
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
                // Best-effort cleanup of temp fixtures.
            }
        }
    }

    [Fact]
    public void Load_ValidConfig_ParsesEverySection()
    {
        string yaml = $"""
            server:
              bind: 0.0.0.0
              port: 8081
            engine:
              threads: 12
              block_size: 16
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                type: chat
                max_context_tokens: 8192
                kv_blocks: 512
                placement:
                  device: cpu
                  kv_residency: dram
                  prefill: cpu
                  core_set: "0-15"
            """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));

        Assert.Equal("0.0.0.0", config.Server.Bind);
        Assert.Equal(8081, config.Server.Port);
        Assert.Equal(12, config.Engine.Threads);
        Assert.Equal(16, config.Engine.BlockSize);

        Assert.Single(config.Models);
        ModelConfig model = config.Models[0];
        Assert.Equal("qwen-test", model.Name);
        Assert.Equal("chat", model.Type);
        Assert.Equal(8192, model.MaxContextTokens);
        Assert.Equal(512, model.KvBlocks);
        Assert.Equal("cpu", model.Placement.Device);
        Assert.Equal("dram", model.Placement.KvResidency);
        Assert.Equal("cpu", model.Placement.Prefill);
        Assert.Equal(16, model.Placement.CoreSet.Count);
        Assert.Contains(0, model.Placement.CoreSet);
        Assert.Contains(15, model.Placement.CoreSet);
        Assert.Null(model.Placement.ThreadQuota);
    }

    [Fact]
    public void Load_OmittedOptionalKeys_AppliesDefaults()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));

        Assert.Equal("127.0.0.1", config.Server.Bind);
        Assert.Equal(8080, config.Server.Port);
        Assert.Equal(0, config.Engine.Threads);
        Assert.Equal(16, config.Engine.BlockSize);

        ModelConfig model = Assert.Single(config.Models);
        Assert.Equal("chat", model.Type);
        Assert.Equal(4096, model.MaxContextTokens);
        Assert.Equal(256, model.KvBlocks);
        Assert.Equal("cpu", model.Placement.Device);
        Assert.Equal("dram", model.Placement.KvResidency);
        Assert.Equal("cpu", model.Placement.Prefill);
        Assert.Empty(model.Placement.CoreSet);
        Assert.Null(model.Placement.ThreadQuota);
    }

    [Fact]
    public void Load_TwoModels_ParsesBothInDeclaredOrder()
    {
        string modelPath = TempModelFile();
        string yaml = $"""
            models:
              - name: first
                path: '{modelPath}'
              - name: second
                path: '{modelPath}'
            """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));

        Assert.Equal(["first", "second"], config.Models.Select(model => model.Name));
    }

    [Fact]
    public void Load_DuplicateModelNames_Fails()
    {
        string modelPath = TempModelFile();
        string yaml = $"""
            models:
              - name: same
                path: '{modelPath}'
              - name: same
                path: '{modelPath}'
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("duplicates", error.Message);
        Assert.Contains("'same'", error.Message);
    }

    [Fact]
    public void Load_EmptyModelList_Fails()
    {
        ConfigException error = Assert.Throws<ConfigException>(
            () => YamlConfigLoader.Load(WriteConfig("models: []")));

        Assert.Contains("at least one model", error.Message);
    }

    [Fact]
    public void Load_MissingModelsKey_Fails()
    {
        ConfigException error = Assert.Throws<ConfigException>(
            () => YamlConfigLoader.Load(WriteConfig("server:\n  port: 8080\n")));

        Assert.Contains("Missing required key 'models'", error.Message);
        Assert.Contains("server, engine, models", error.Message);
    }

    [Fact]
    public void Load_ModelPathMustExistOnDisk_Fails()
    {
        string missing = Path.Combine(Path.GetTempPath(), $"lair-missing-{Guid.NewGuid():N}.gguf");
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{missing}'
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].path", error.Message);
        Assert.Contains("does not exist on disk", error.Message);
    }

    [Fact]
    public void Load_SourceModelPathMayBeAbsentForFetch()
    {
        string missing = Path.Combine(Path.GetTempPath(), $"lair-source-{Guid.NewGuid():N}.gguf");
        string yaml = $"""
            models:
              - name: source-test
                path: '{missing}'
                source:
                  repo: owner/repo
                  revision: main
                  file: model.gguf
            """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));

        Assert.NotNull(config.Models[0].Source);
        Assert.Equal("owner/repo", config.Models[0].Source!.Repo);
    }

    [Fact]
    public void Load_EmptyModelPath_Fails()
    {
        string yaml = """
            models:
              - name: qwen-test
                path: "   "
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].path", error.Message);
        Assert.Contains("non-empty path", error.Message);
    }

    [Fact]
    public void Load_MissingModelPath_Fails()
    {
        string yaml = """
            models:
              - name: qwen-test
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("Missing required key 'path'", error.Message);
        Assert.Contains("name, path, type, max_context_tokens, kv_blocks, kv, placement", error.Message);
    }

    [Fact]
    public void Load_EmbedModelType_Parses_AndOtherTypesFailNamingTheAcceptedSet()
    {
        string yaml = $"""
            models:
              - name: embed-test
                path: '{TempModelFile()}'
                type: embed
            """;

        LairConfig embed = YamlConfigLoader.Load(WriteConfig(yaml));
        Assert.Equal("embed", embed.Models[0].Type); // the encode-only serving posture the embeddings endpoint answers with

        LairConfig chat = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: chat-test
                path: '{TempModelFile()}'
            """));
        Assert.Equal("chat", chat.Models[0].Type); // an absent key keeps the sampling posture the only default this loader has ever resolved

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: vector-test
                path: '{TempModelFile()}'
                type: vector
            """)));
        Assert.Contains("models[0].type 'vector' is invalid for model 'vector-test'", error.Message);
        Assert.Contains("'type' accepts only 'chat' or 'embed'", error.Message);
    }

    [Fact]
    public void Load_PoolingAndNormalize_ParseDefaultAndValidate()
    {
        LairConfig declared = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: embed-test
                path: '{TempModelFile()}'
                type: embed
                pooling: cls
                normalize: true
            """));
        Assert.Equal("cls", declared.Models[0].Pooling);
        Assert.True(declared.Models[0].Normalize);

        LairConfig defaulted = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: embed-test
                path: '{TempModelFile()}'
                type: embed
            """));
        Assert.Equal("mean", defaulted.Models[0].Pooling); // the average-over-positions recipe an absent key resolves to
        Assert.False(defaulted.Models[0].Normalize);       // raw pooled vectors answer unless the config asks for unit length

        ConfigException pooling = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: embed-test
                path: '{TempModelFile()}'
                type: embed
                pooling: max
            """)));
        Assert.Contains("models[0].pooling 'max' is invalid for model 'embed-test'", pooling.Message);
        Assert.Contains("'pooling' accepts only 'mean' or 'cls'", pooling.Message);

        ConfigException normalize = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: embed-test
                path: '{TempModelFile()}'
                type: embed
                normalize: sometimes
            """)));
        Assert.Contains("models[0].normalize", normalize.Message);
    }

    [Fact]
    public void Load_PlacementDeviceAuto_ResolvesHostCompute()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig(PlacementYaml("device: auto")));

        Assert.Equal("cpu", config.Models[0].Placement.Device); // the posture-neutral declaration resolves once at load onto this build's opened compute backend.
    }

    [Fact]
    public void Load_PlacementDeviceAccelerator_FailsNamingAcceptedVocabularyAndSeam()
    {
        ConfigException error = Assert.Throws<ConfigException>(
            () => YamlConfigLoader.Load(WriteConfig(PlacementYaml("device: cuda"))));

        Assert.Contains("placement.device 'cuda'", error.Message);
        Assert.Contains("'device' accepts only 'auto' or 'cpu'", error.Message, StringComparison.Ordinal);
        Assert.Contains(ModelRoute.CudaBackendSeam, error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_PlacementKvResidencyOutsideTheDeclaredSet_FailsNamingModelKeyAndAcceptedSet()
    {
        ConfigException error = Assert.Throws<ConfigException>(
            () => YamlConfigLoader.Load(WriteConfig(PlacementYaml("kv_residency: gpu"))));

        Assert.Contains("models[0].placement.kv_residency 'gpu' is invalid for model 'qwen-test'", error.Message);
        Assert.Contains("'kv_residency' accepts only 'auto', 'dram', 'vram', 'split'", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_PlacementPrefillAuto_ResolvesHostCompute()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig(PlacementYaml("prefill: auto")));

        Assert.Equal("cpu", config.Models[0].Placement.Prefill); // the posture-neutral declaration resolves once at load onto this build's opened prefill backend.
    }

    [Fact]
    public void Load_PlacementPrefillAccelerator_FailsNamingAcceptedVocabularyAndSeam()
    {
        ConfigException error = Assert.Throws<ConfigException>(
            () => YamlConfigLoader.Load(WriteConfig(PlacementYaml("prefill: gpu"))));

        Assert.Contains("placement.prefill 'gpu'", error.Message);
        Assert.Contains("'prefill' accepts only 'auto' or 'cpu'", error.Message, StringComparison.Ordinal);
        Assert.Contains(ModelRoute.CudaBackendSeam, error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_BlockSizeIsFixedTo16_Fails()
    {
        string yaml = $"""
            engine:
              block_size: 32
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("engine.block_size 32", error.Message);
        Assert.Contains("block size is fixed to 16", error.Message);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(65536)]
    [InlineData(-1)]
    public void Load_PortOutOfRange_Fails(int port)
    {
        string yaml = $"""
            server:
              port: {port}
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"server.port {port}", error.Message);
        Assert.Contains("must be between 1 and 65535", error.Message);
    }

    [Fact]
    public void Load_NegativeThreads_Fails()
    {
        string yaml = $"""
            engine:
              threads: -4
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("engine.threads -4", error.Message);
        Assert.Contains("'threads' must be 0 or greater", error.Message);
    }

    [Theory]
    [InlineData(7)]
    [InlineData(0)]
    public void Load_KvBlocksTooSmall_Fails(int kvBlocks)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                kv_blocks: {kvBlocks}
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"models[0].kv_blocks {kvBlocks}", error.Message);
        Assert.Contains("'kv_blocks' must be 8 or greater", error.Message);
    }

    [Theory]
    [InlineData(127)]
    [InlineData(0)]
    public void Load_MaxContextTokensTooSmall_Fails(int maxContextTokens)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                max_context_tokens: {maxContextTokens}
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"models[0].max_context_tokens {maxContextTokens}", error.Message);
        Assert.Contains("'max_context_tokens' must be 128 or greater", error.Message);
    }

    [Fact]
    public void Load_MaxBatchTokens_DefaultsValidatesAndRejects()
    {
        LairConfig defaulted = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));
        Assert.Equal(512, defaulted.Engine.MaxBatchTokens);

        LairConfig configured = YamlConfigLoader.Load(WriteConfig($"""
            engine:
              max_batch_tokens: 256
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));
        Assert.Equal(256, configured.Engine.MaxBatchTokens);

        foreach (int invalid in new[] { 0, -3 })
        {
            ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
                engine:
                  max_batch_tokens: {invalid}
                models:
                  - name: qwen-test
                    path: '{TempModelFile()}'
                """)));
            Assert.Contains($"engine.max_batch_tokens {invalid}", error.Message);
            Assert.Contains("'max_batch_tokens' must be 1 or greater", error.Message);
        }
    }

    [Fact]
    public void Load_PrefillStepBudgetMs_DefaultsValidatesAndRejects()
    {
        LairConfig defaulted = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));
        Assert.Null(defaulted.Engine.PrefillStepBudgetMs); // absent means chunk width keeps coming from max_batch_tokens alone

        LairConfig configured = YamlConfigLoader.Load(WriteConfig($"""
            engine:
              prefill_step_budget_ms: 120
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));
        Assert.Equal(120, configured.Engine.PrefillStepBudgetMs);

        foreach (int invalid in new[] { 0, -3 })
        {
            ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
                engine:
                  prefill_step_budget_ms: {invalid}
                models:
                  - name: qwen-test
                    path: '{TempModelFile()}'
                """)));
            Assert.Contains($"engine.prefill_step_budget_ms {invalid}", error.Message);
            Assert.Contains("'prefill_step_budget_ms' must be 1 or greater", error.Message);
        }
    }

    [Fact]
    public void Load_UnknownKeys_NamedWithAllowedKeysAtThatLevel()
    {
        string modelPath = TempModelFile();

        (string Yaml, string UnknownKey, string Where, string AllowedKeys)[] cases =
        [
            (
                "bogus: 1\nmodels:\n  - name: qwen-test\n    path: '" + modelPath + "'\n",
                "bogus",
                "the top level",
                "server, engine, models"
            ),
            (
                "server:\n  bindd: 127.0.0.1\nmodels:\n  - name: qwen-test\n    path: '" + modelPath + "'\n",
                "bindd",
                "'server'",
                "bind, port"
            ),
            (
                "engine:\n  thread: 8\nmodels:\n  - name: qwen-test\n    path: '" + modelPath + "'\n",
                "thread",
                "'engine'",
                "threads, block_size, max_batch_tokens"
            ),
            (
                "models:\n  - name: qwen-test\n    path: '" + modelPath + "'\n    context: 4096\n",
                "context",
                "models[0]",
                "name, path, type, max_context_tokens, kv_blocks, kv, placement"
            ),
            (
                "models:\n  - name: qwen-test\n    path: '" + modelPath + "'\n    placement:\n      device_id: 0\n",
                "device_id",
                "models[0].placement",
                "device, kv_residency, prefill, core_set, thread_quota"
            ),
        ];

        foreach ((string yaml, string unknownKey, string where, string allowedKeys) in cases)
        {
            ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
            Assert.Contains($"Unknown key '{unknownKey}'", error.Message);
            Assert.Contains(where, error.Message);
            Assert.Contains($"allowed keys are: {allowedKeys}", error.Message);
        }
    }

    [Fact]
    public void Load_ForceFencedTools_ParsesBoolean_AndDefaultsFalse()
    {
        LairConfig forced = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                force_fenced_tools: true
            """));
        Assert.True(forced.Models[0].ForceFencedTools);

        LairConfig absent = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));
        Assert.False(absent.Models[0].ForceFencedTools);

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                force_fenced_tools: sometimes
            """)));
        Assert.Contains("models[0].force_fenced_tools", error.Message);
        Assert.Contains("must be a true or false value", error.Message);
    }

    [Fact]
    public void Load_CoreSetRange_ParsesInclusiveRange()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  core_set: "0-15"
            """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));
        IReadOnlySet<int> coreSet = config.Models[0].Placement.CoreSet;

        Assert.Equal(16, coreSet.Count);
        Assert.True(coreSet.IsSupersetOf([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]));
    }

    [Fact]
    public void Load_CoreSetMixedIdsAndRanges_ParsesToExpectedSet()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  core_set: "0,2,8-11"
            """;

        LairConfig config = YamlConfigLoader.Load(WriteConfig(yaml));
        IReadOnlySet<int> coreSet = config.Models[0].Placement.CoreSet;

        Assert.Equal(6, coreSet.Count);
        int[] expected = [0, 2, 8, 9, 10, 11];
        Assert.Equal(expected, coreSet.OrderBy(core => core).ToArray());
    }

    [Theory]
    [InlineData("0-")]
    [InlineData("-5")]
    [InlineData("15-0")]
    [InlineData("abc")]
    [InlineData("0,,")]
    public void Load_CoreSetInvalidSyntax_Fails(string coreSet)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  core_set: "{coreSet}"
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].placement.core_set", error.Message);
        Assert.Contains("core_set", error.Message);
    }

    [Fact]
    public void Load_MissingFile_Fails()
    {
        string missing = Path.Combine(Path.GetTempPath(), $"lair-absent-{Guid.NewGuid():N}.yaml");

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(missing));
        Assert.Contains("does not exist", error.Message);
    }

    [Fact]
    public void Load_ThreadQuota_ParsesAndDefaultsToNull()
    {
        LairConfig config = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  thread_quota: 4
            """));

        Assert.Equal(4, config.Models[0].Placement.ThreadQuota);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-2)]
    public void Load_ThreadQuotaTooSmall_Fails(int quota)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                placement:
                  thread_quota: {quota}
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"models[0].placement.thread_quota {quota}", error.Message);
        Assert.Contains("'thread_quota' must be 1 or greater", error.Message);
    }

    [Fact]
    public void Load_MaxConcurrentRequests_ParsesAndDefaultsToNull()
    {
        LairConfig capped = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                max_concurrent_requests: 3
            """));

        Assert.Equal(3, capped.Models[0].MaxConcurrentRequests);

        LairConfig absent = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));

        Assert.Null(absent.Models[0].MaxConcurrentRequests); // absent bounds nothing, exactly as before this key existed
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-2)]
    public void Load_MaxConcurrentRequestsTooSmall_Fails(int cap)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                max_concurrent_requests: {cap}
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"models[0].max_concurrent_requests {cap}", error.Message);
        Assert.Contains("'max_concurrent_requests' must be 1 or greater", error.Message);
    }

    [Fact]
    public void Load_MaxConcurrentRequestsNonInteger_Fails()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                max_concurrent_requests: two
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].max_concurrent_requests must be an integer value.", error.Message);
    }

    [Fact]
    public void Load_RequestQueueKeys_ParseAndDefaultToNothingQueued()
    {
        LairConfig bounded = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                max_queued_requests: 4
                queue_timeout_ms: 12500
            """));

        Assert.Equal(4, bounded.Models[0].MaxQueuedRequests);
        Assert.Equal(12_500, bounded.Models[0].QueueTimeoutMs);

        LairConfig absent = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
            """));

        // Absent keys keep today's posture outright: nothing queues and no bounded wait is declared for something that never waits.
        Assert.Null(absent.Models[0].MaxQueuedRequests);
        Assert.Null(absent.Models[0].QueueTimeoutMs);

        LairConfig explicitZero = YamlConfigLoader.Load(WriteConfig($"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                max_queued_requests: 0
            """));

        Assert.Equal(0, explicitZero.Models[0].MaxQueuedRequests); // an explicit zero states the same immediate-refusal posture the absent key carries
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(-7)]
    public void Load_MaxQueuedRequestsNegative_Fails(int depth)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                max_queued_requests: {depth}
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"models[0].max_queued_requests {depth}", error.Message);
        Assert.Contains("'max_queued_requests' must be 0 or greater", error.Message); // zero is the documented disabled posture; only a negative number is refused
    }

    [Fact]
    public void Load_MaxQueuedRequestsNonInteger_Fails()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                max_queued_requests: two
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].max_queued_requests must be an integer value.", error.Message);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-2)]
    public void Load_QueueTimeoutTooSmall_Fails(int ceiling)
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                queue_timeout_ms: {ceiling}
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains($"models[0].queue_timeout_ms {ceiling}", error.Message);
        Assert.Contains("'queue_timeout_ms' must be 1 or greater", error.Message); // a present ceiling is a real wait: zero would say nothing without saying so
    }

    [Fact]
    public void Load_QueueTimeoutNonInteger_Fails()
    {
        string yaml = $"""
            models:
              - name: qwen-test
                path: '{TempModelFile()}'
                queue_timeout_ms: two
            """;

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.Load(WriteConfig(yaml)));
        Assert.Contains("models[0].queue_timeout_ms must be an integer value.", error.Message);
    }

    [Fact]
    public void ValidateCoreSetsDisjoint_OverlappingModels_NamesBothModels()
    {
        List<ModelConfig> models =
        [
            new ModelConfig { Name = "first", Path = "a.gguf", Placement = new PlacementConfig { CoreSet = new HashSet<int>([0, 1, 2]) } },
            new ModelConfig { Name = "second", Path = "b.gguf", Placement = new PlacementConfig { CoreSet = new HashSet<int>([2, 3]) } },
        ];

        ConfigException error = Assert.Throws<ConfigException>(() => YamlConfigLoader.ValidateCoreSetsDisjoint(models));
        Assert.Contains("both pin core 2", error.Message);
        Assert.Contains("'first'", error.Message);
        Assert.Contains("'second'", error.Message);
    }

    [Fact]
    public void ValidateCoreSetsDisjoint_DisjointAndUnpinnedModels_Pass()
    {
        List<ModelConfig> models =
        [
            new ModelConfig { Name = "first", Path = "a.gguf", Placement = new PlacementConfig { CoreSet = new HashSet<int>([0, 1]) } },
            new ModelConfig { Name = "second", Path = "b.gguf", Placement = new PlacementConfig { CoreSet = new HashSet<int>([2, 3]) } },
            new ModelConfig { Name = "third", Path = "c.gguf" }, // unpinned: an empty core set shares nothing
        ];

        YamlConfigLoader.ValidateCoreSetsDisjoint(models);
    }

    [Fact]
    public void Load_MalformedYaml_Fails()
    {
        ConfigException error = Assert.Throws<ConfigException>(
            () => YamlConfigLoader.Load(WriteConfig("server: [1, 2\n")));

        Assert.Contains("could not be read as YAML", error.Message);
    }

    private string PlacementYaml(string placementLine) =>
        "models:\n" +
        "  - name: qwen-test\n" +
        $"    path: '{TempModelFile()}'\n" +
        "    placement:\n" +
        $"      {placementLine}\n";

    private string WriteConfig(string yaml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-config-{Guid.NewGuid():N}.yaml");
        File.WriteAllText(path, yaml);
        _tempPaths.Add(path);
        return path;
    }

    private string TempModelFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"lair-model-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, [0x47, 0x47, 0x55, 0x46]);
        _tempPaths.Add(path);
        return path;
    }
}
