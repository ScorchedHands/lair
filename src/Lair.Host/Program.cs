using System.Reflection;
using Lair.Http;
using Lair.Host;
using Lair.Host.Config;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

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
    using Microsoft.Extensions.Logging.ILoggerFactory loggerFactory = Microsoft.Extensions.Logging.LoggerFactory.Create(builder =>
        builder.AddSimpleConsole(options => options.SingleLine = true).SetMinimumLevel(LogLevel.Information));
    try
    {
        config = YamlConfigLoader.Load(args[2]);
        // A model that declares a published source is verified against its pack manifest (and fetched if absent) through
        // the live hub seam before its GGUF opens; a model without a source is untouched by this seam. A gated repo is
        // read with the operator's environment-supplied token, which the seam keeps out of the manifest and the logs.
        engine = GenerationService.Open(config, loggerFactory.CreateLogger("Lair.Http.GenerationService"), ledger: null, compute: null, transport: new HuggingFaceHubTransport(new EnvironmentHubTokenProvider()));
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

        // Hot reload belongs to the serving posture only: this watcher coalesces config edits and pushes validated changes through the engine's change-application bus. Declared last so it disposes before the engine it watches.
        using ConfigWatcher hotReload = new(args[2], engine);
        hotReload.Start();

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

if (args.Length >= 2 && args[0] == "monitor")
{
    Uri baseUri = new("http://127.0.0.1:8080");
    int intervalMs = 0;

    for (int i = 1; i < args.Length; i += 2)
    {
        switch (args[i])
        {
            case "-u":
                if (i + 1 >= args.Length || !Uri.TryCreate(args[i + 1], UriKind.Absolute, out Uri? parsed))
                {
                    Console.Error.WriteLine("lair: monitor -u needs an absolute server URL, e.g. http://127.0.0.1:8080");
                    return 1;
                }

                baseUri = parsed;
                break;
            case "--interval":
                if (i + 1 >= args.Length || !int.TryParse(args[i + 1], out intervalMs) || intervalMs < 0)
                {
                    Console.Error.WriteLine("lair: monitor --interval needs a non-negative number of milliseconds");
                    return 1;
                }

                break;
            default:
                Console.Error.WriteLine($"lair: unknown monitor option '{args[i]}'");
                return 1;
        }
    }

    using (CancellationTokenSource cancel = new())
    {
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            // Ctrl+C ends the live table cleanly.
            eventArgs.Cancel = true;
            cancel.Cancel();
        };

        return await MonitorCommand.RunAsync(baseUri, Console.Out, cancel.Token, intervalMs);
    }
}

if (args.Length >= 1 && args[0] == "model")
{
    // The model CLI reads pack manifests beside the model files it names: with -c it takes each config model's own
    // pack directory, and without one it defaults to the models directory beside the executable. HEAD is consulted
    // through the live hub seam for candidate reporting only. A gated repo is read with the operator's
    // environment-supplied token, which the seam keeps out of the manifest and the CLI output.
    string? modelConfigPath = null;
    List<string> modelArgs = [];

    for (int i = 1; i < args.Length; i++)
    {
        if (args[i] == "-c")
        {
            if (i + 1 >= args.Length)
            {
                Console.Error.WriteLine("lair: model -c needs a config path");
                return 2;
            }

            modelConfigPath = args[++i];
            continue;
        }

        modelArgs.Add(args[i]);
    }

    LairConfig? modelConfig = null;
    if (modelConfigPath is not null)
    {
        try
        {
            modelConfig = YamlConfigLoader.Load(modelConfigPath);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"lair: {ex.Message}");
            return 1;
        }
    }

    ModelCliRunner modelRunner = new(new HuggingFaceHubTransport(new EnvironmentHubTokenProvider()), Path.Combine(AppContext.BaseDirectory, "models"), Console.Out, modelConfig);
    return await modelRunner.RunAsync(modelArgs, CancellationToken.None);
}

Console.Error.WriteLine("""
    usage:
      lair serve -c <configPath>   serve the configured model over the OpenAI-compatible HTTP API
      lair chat -c <configPath>    chat with the configured model in an interactive REPL
      lair monitor [-u URL] [--interval ms]  render live engine stats from a serving server (default http://127.0.0.1:8080)
      lair model fetch|check|list [-c <configPath>]  fetch, report update candidates for, and list pinned model packs
      lair --version               print the lair version
    """);
return 1;
