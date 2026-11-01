using Lair.Host;
using Lair.Host.Config;

namespace Lair.Tests;

/// <summary>
/// CLI chat REPL tests driven through in-memory readers and writers against the real SmolLM fixture: banner,
/// streamed replies, slash commands, context-window trimming errors, and cancellation. Each test opens its own
/// engine from a small-context config so overflow paths stay cheap.
/// </summary>
[Trait("Category", "Cli")]
public class CliTests
{
    private const string ModelName = "smollm-chat";

    // Measured against the fixture's 512-token window: 2 pangram repetitions render to ~110 prompt tokens,
    // 13 render to 604 and overflow even after the REPL trims the oldest exchange.
    private static readonly string ModeratePrompt = Repeat("the quick brown fox jumps over the lazy dog. ", 2);
    private static readonly string OversizedPrompt = Repeat("the quick brown fox jumps over the lazy dog. ", 13);

    [Fact]
    public async Task RunAsync_HelloThenExit_PrintsBannerAndReply_AndExitsZero()
    {
        StringReader input = new("hello\n/exit\n");
        StringWriter output = new();

        int exitCode = await ChatCommand.RunAsync(FixtureConfig(), input, output, CancellationToken.None, maxTokens: 24);

        Assert.Equal(0, exitCode);
        string text = output.ToString();
        Assert.Contains(ModelName, text);
        Assert.Contains("context: 512", text);
        Assert.Contains("quant: F16", text);
        Assert.DoesNotContain("error:", text);

        List<string> lines = Lines(text);
        Assert.StartsWith("model: ", lines[0]);
        Assert.True(ReplyLines(lines).Count >= 1, $"expected a streamed assistant reply line in:{Environment.NewLine}{text}");
    }

    [Fact]
    public async Task RunAsync_ResetAfterExchange_ClearsHistory_AndGeneratesFreshReply()
    {
        StringReader input = new("hello\n/reset\nhi again\n/exit\n");
        StringWriter output = new();

        int exitCode = await ChatCommand.RunAsync(FixtureConfig(), input, output, CancellationToken.None, maxTokens: 24);

        Assert.Equal(0, exitCode);
        string text = output.ToString();
        Assert.DoesNotContain("error:", text);
        Assert.DoesNotContain("interrupted", text);

        // Both turns answered: one reply before /reset and a fresh one after it.
        Assert.True(ReplyLines(Lines(text)).Count >= 2, $"expected two assistant replies in:{Environment.NewLine}{text}");
    }

    [Fact]
    public async Task RunAsync_OversizedTurnAfterHistory_ReportsSingleLineError_AndKeepsReadingUntilExit()
    {
        StringReader input = new($"{ModeratePrompt}\n{OversizedPrompt}\n/exit\n");
        StringWriter output = new();

        int exitCode = await ChatCommand.RunAsync(FixtureConfig(), input, output, CancellationToken.None, maxTokens: 24);

        Assert.Equal(0, exitCode);
        string text = output.ToString();

        List<string> lines = Lines(text);
        List<string> errors = [.. lines.Where(line => line.StartsWith("error: "))];
        Assert.True(errors.Count >= 1, $"expected a single-line context error in:{Environment.NewLine}{text}");

        // Failures are reported as one plain line each: no stack frames or exception type names leak out.
        Assert.DoesNotContain("   at ", text);
        Assert.DoesNotContain("Exception", string.Concat(errors));

        List<string> replies = ReplyLines(lines);
        Assert.True(replies.Count >= 1, $"expected the first turn to answer before the overflow in:{Environment.NewLine}{text}");
    }

    [Fact]
    public async Task RunAsync_CancellationBeforeGeneration_PrintsInterrupted_AndExitsZero()
    {
        using CancellationTokenSource cancel = new();
        cancel.Cancel();

        StringReader input = new("hello\n/exit\n");
        StringWriter output = new();

        int exitCode = await ChatCommand.RunAsync(FixtureConfig(), input, output, cancel.Token, maxTokens: 24);

        Assert.Equal(0, exitCode);
        string text = output.ToString();
        Assert.Contains("interrupted", text);
        Assert.DoesNotContain("error:", text);
    }

    private static LairConfig FixtureConfig()
    {
        string modelPath = ModelFixture.Resolve() ?? throw new InvalidOperationException(
            $"GGUF fixture '{ModelFixture.SmolLmRelativePath}' was not found walking up from the test assembly directory.");

        return new LairConfig
        {
            Server = new ServerConfig(),
            Engine = new EngineConfig(),
            Models =
            [
                new ModelConfig
                {
                    Name = ModelName,
                    Path = modelPath,
                    MaxContextTokens = 512,
                    KvBlocks = 32,
                },
            ],
        };
    }

    private static string Repeat(string unit, int times) => string.Concat(Enumerable.Repeat(unit, times));

    private static List<string> Lines(string text) =>
        [.. text.Split('\n').Select(line => line.TrimEnd('\r'))];

    // A reply line is a prompt-prefixed turn with streamed content after the "> " marker. Continuation lines of
    // multi-line replies carry plain text and count too; blank prompts are turns that produced nothing.
    private static List<string> ReplyLines(List<string> lines) =>
        [.. lines.Skip(1).Where(line => line.StartsWith("> ") ? line.Length > 2 : line.Trim().Length > 0)];
}
