# LAIR CLI reference

LAIR's command surface is the native executable produced by `dotnet build`: on Windows, `src\Lair.Host\bin\Release\net10.0\lair.exe`; on Linux/macOS, `./src/Lair.Host/bin/Release/net10.0/lair`. The repository does not currently provide an install script that places the binary on `PATH`, so examples below use a PowerShell variable:

```powershell
$LAIR = ".\src\Lair.Host\bin\Release\net10.0\lair.exe"
```

Run LAIR from the directory you want relative model/config paths to resolve against, or use absolute paths in config files.

## Global forms

| Command | Behavior | Exit behavior |
|---|---|---|
| `lair --version` / `lair -v` | Prints `lair <version>` and exits successfully. | 0 |
| Unknown top-level command | Prints usage to stderr. | 1 |

## Serve

```powershell
& $LAIR serve -c lair.yaml
```

The accepted syntax is exactly `serve -c <configPath>`. LAIR validates the YAML file, opens and verifies/fetches declared model sources when present, starts the HTTP server from `server.bind`/`server.port`, serves `/ui`, and watches the same config file for hot-reload changes. First Ctrl+C shuts down gracefully; a second Ctrl+C exits immediately. Config or engine-open failures exit 1 with an actionable message.

## Chat REPL

```powershell
& $LAIR chat -c lair.yaml
```

The accepted syntax is exactly `chat -c <configPath>`. The REPL supports `/exit`, `/quit`, and `/reset`; Ctrl+C interrupts the current generation and ends cleanly. Overlong turns are reported after an automatic trim-and-retry attempt.

## Monitor CLI

```powershell
& $LAIR monitor
& $LAIR monitor -u http://127.0.0.1:8181 --interval 1000
```

Accepted syntax is `monitor [-u <absolute URL>] [--interval <milliseconds>]`. The default base URL is `http://127.0.0.1:8080`, so pass `-u` when the server uses a different bind or port. An invalid absolute URL, negative interval, or unknown monitor option exits 1.

Monitor rows show tok/s, step time, batch occupancy, cache-hit share, KV usage/capacity, and active sequences. When a served engine has opened a CUDA backend, each row appends an accelerator tail: `gpu <kernel busy %>`, runtime PCIe copy bytes/calls (`pcie … over N copies`), and persistent device weight residency (`persistent-vram … MiB`). CPU-only serves omit the tail rather than showing idle-hardware zeros. The same facts arrive additively on `/monitor/stream` as optional snapshot fields: `metrics_gpu_utilization_percent`, `metrics_pcie_transfer_bytes_total`, `metrics_pcie_transfers_total`, and `metrics_persistent_vram_bytes`; they are omitted when no CUDA backend is open.

## Model CLI

```powershell
& $LAIR model [-c <configPath>] fetch <repo> [--revision <rev>] [--file <path>]... [--model <name>]
& $LAIR model [-c <configPath>] check [model]
& $LAIR model [-c <configPath>] list
```

The `-c` option may appear after `model`; it loads the config whose model entries name their own pack directories. Without `-c`, commands use the default models directory beside the executable.

### fetch

- With no config, a repo is required and at least one `--file` is required; omitted revision defaults to `main`.
- With a config, fetch only operates on declared sources: an unnamed repo must be declared by exactly one model entry, or pass `--model <name>` when multiple entries declare the same repo.
- A configured source's pinned revision cannot be overridden with a different `--revision` value.
- Requested files must already be declared by that config model; undeclared file names are refused.

Fetch downloads/verifies only the declared file set, records pins in the pack manifest, and no-ops on an already verified final file so repeated fetches do not double-stream. Gated repos authenticate through `LAIR_HF_TOKEN` or fallback `HUGGING_FACE_HUB_TOKEN`; tokens are sent only to `huggingface.co` and never written to logs/errors/manifests.

### check

```powershell
& $LAIR model check [model]
```

Check consults the hub's current HEAD through the transport seam and reports candidate deltas — revision, per-file hash/size changes, and file-list changes — without downloading, repinning, or changing load behavior. Candidate deltas are still a completed report. Exit codes: 0 on success, 1 for config/manifest/hub failures or no manifest entry naming the requested model, 2 for malformed arguments.

### list

```powershell
& $LAIR model list [-c lair.yaml]
```

List prints what each relevant pack pins: model name, repo, resolved revision, and file entries with hash/size/fetched timestamp when recorded. Exit codes follow the same bounded model CLI pattern above.
