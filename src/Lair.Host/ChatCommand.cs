using Lair.Http;
using Lair.Host.Config;
using Lair.Tokenizer.Templating;

namespace Lair.Host;

/// <summary>
/// Interactive chat REPL over the first configured model: streams each reply token by token, keeps a rolling
/// conversation history, and drops the oldest exchange when the history outgrows the model's context window.
/// Slash commands end or clear the conversation; any other line is a user message.
/// </summary>
public static class ChatCommand
{
    private const string ExitCommand = "/exit";
    private const string QuitCommand = "/quit";
    private const string ResetCommand = "/reset";

    /// <summary>
    /// Runs the REPL until <see cref="ExitCommand"/>/<see cref="QuitCommand"/>, end of input, or cancellation.
    /// Returns 0 on a clean or cancelled exit, 1 when the engine cannot be opened. Failures during a turn are
    /// reported as a single <c>error:</c> line and the loop keeps reading.
    /// </summary>
    public static async Task<int> RunAsync(
        LairConfig config,
        TextReader input,
        TextWriter output,
        CancellationToken ct,
        int maxTokens = 256)
    {
        GenerationService engine;
        try
        {
            engine = GenerationService.Open(config);
        }
        catch (Exception ex)
        {
            output.WriteLine($"error: {ex.Message}");
            return 1;
        }

        using (engine)
        {
            output.WriteLine($"model: {engine.Model.Id}  context: {engine.Model.ContextLength}  quant: {engine.Model.Quant}");

            List<ChatMessage> history = [];

            while (true)
            {
                if (ct.IsCancellationRequested)
                {
                    output.WriteLine("interrupted");
                    return 0;
                }

                output.Write("> ");
                output.Flush();

                string? line = input.ReadLine();
                if (line is null)
                {
                    output.WriteLine();
                    break;
                }

                string text = line.Trim();
                if (text.Length == 0)
                {
                    continue;
                }

                if (text is ExitCommand or QuitCommand)
                {
                    break;
                }

                if (text == ResetCommand)
                {
                    history.Clear();
                    continue;
                }

                history.Add(new ChatMessage("user", text));

                bool trimmed = false;
                while (true)
                {
                    try
                    {
                        GenerationResult result = await engine.GenerateAsync(
                            new GenerationRequest
                            {
                                Model = engine.Model.Id,
                                IsChat = true,
                                Messages = history,
                                MaxTokens = maxTokens,
                                Temperature = 1.0f,
                                Seed = null,
                                Stream = true,
                                OnToken = piece =>
                                {
                                    output.Write(piece);
                                    output.Flush();
                                },
                            },
                            ct).ConfigureAwait(false);

                        output.WriteLine();
                        history.Add(new ChatMessage("assistant", result.Text));
                        break;
                    }
                    catch (ContextLengthExceededException ex)
                    {
                        // The rolling history outgrew the window; drop the oldest exchange and try once more.
                        if (!trimmed && history.Count >= 2)
                        {
                            history.RemoveRange(0, 2);
                            trimmed = true;
                            continue;
                        }

                        // Even trimmed the turn does not fit; drop it so the next turn starts from usable history.
                        history.RemoveAt(history.Count - 1);
                        output.WriteLine();
                        output.WriteLine($"error: {ex.Message}");
                        break;
                    }
                    catch (OperationCanceledException)
                    {
                        output.WriteLine();
                        output.WriteLine("interrupted");
                        return 0;
                    }
                    catch (Exception ex)
                    {
                        history.RemoveAt(history.Count - 1);
                        output.WriteLine();
                        output.WriteLine($"error: {ex.Message}");
                        break;
                    }
                }
            }

            return 0;
        }
    }
}
