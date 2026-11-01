using System.Reflection;
using Lair.Api;
using Lair.Host;
using Lair.Host.Config;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;

if (args.Length == 1 && (args[0] is "--version" or "-v"))
{
    string? version = typeof(Program).Assembly.GetName().Version?.ToString(3);
    Console.WriteLine($"lair {version ?? "0.0.0-dev"}");
    return 0;
}

if (args.Length == 3 && args[0] == "serve" && args[1] == "-c")
{
    LairConfig config;
    GenerationService engine;
    try
    {
        config = YamlConfigLoader.Load(args[2]);
        engine = GenerationService.Open(config);
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"lair: {ex.Message}");
        return 1;
    }

    using (engine)
    using (CancellationTokenSource shutdown = new())
    {
        int interrupts = 0;
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            // First Ctrl+C shuts the server down gracefully; a second one exits immediately.
            if (++interrupts == 1)
            {
                eventArgs.Cancel = true;
                shutdown.Cancel();
            }
        };

        WebApplication app = ApiApp.Build(config, engine);
        Console.WriteLine($"lair serving '{engine.Model.Id}' on http://{config.Server.Bind}:{config.Server.Port}");
        await ((IHost)app).RunAsync(shutdown.Token);
    }

    return 0;
}

if (args.Length == 3 && args[0] == "chat" && args[1] == "-c")
{
    LairConfig chatConfig;
    try
    {
        chatConfig = YamlConfigLoader.Load(args[2]);
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"lair: {ex.Message}");
        return 1;
    }

    using (CancellationTokenSource cancel = new())
    {
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            // Ctrl+C aborts the current generation and ends the REPL cleanly.
            eventArgs.Cancel = true;
            cancel.Cancel();
        };

        return await ChatCommand.RunAsync(chatConfig, Console.In, Console.Out, cancel.Token);
    }
}

Console.Error.WriteLine("""
    usage:
      lair serve -c <configPath>   serve the configured model over the OpenAI-compatible HTTP API
      lair chat -c <configPath>    chat with the configured model in an interactive REPL
      lair --version               print the lair version
    """);
return 1;
