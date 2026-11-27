# LAIR CLI Reference

Every command line LAIR accepts, what each flag does, and what the command prints. All examples are Windows PowerShell.

## Invocation

LAIR is one executable with one verb. Every example below runs the compiled executable — `lair` (`lair.exe` on Windows):

```powershell
lair serve -c lair.yaml
```

### Getting the executable

A release build produces a self-contained single-file executable that carries the .NET runtime, so nothing else has to be installed to run it:

```powershell
dotnet publish src/Lair.Host -c Release -r win-x64 --self-contained -p:PublishSingleFile=true -o release
```

That writes `release\lair.exe` (about 100 MB) with `wwwroot\` beside it — the Web UI bundle the host serves at `/ui` — plus `aspnetcorev2_inprocess.dll` (a native library that stays unpacked beside the executable) and `.pdb` sidecar files. Publishing with `-p:SkipWebUi=true` omits the Web UI and needs no Node; publishing with the UI bundled needs Node 22 / npm 11 on the build machine.

A checkout can also be driven without publishing:

```powershell
dotnet build Lair.sln                                  # src\Lair.Host\bin\Release\net10.0\lair.exe (framework-dependent)
dotnet run --project src/Lair.Host -- <command> <args> # the same program, run from source
```

No publish profile is committed in the repository yet: the publish command above is the packaging shape the project intends to ship, and CI packaging is planned (see "Planned CLI surface" at the end). Everything below writes `lair <args>` as shorthand for whichever executable you are running.

### Choosing the config file

| Flag | Applies to | Meaning |
|---|---|---|
| `-c <path>` | `serve`, `chat`, `model` | The `lair.yaml` file to load. |

- The spelling is exactly `-c`. There is no `--config` long form: `lair serve --config lair.yaml` is rejected as a usage error.
- There is no default config path. `serve` and `chat` require `-c <path>`; without it they print the usage block and exit `1`.
- `model` accepts `-c` anywhere after the verb, and it is optional (see the `model` section for what changes when it is present).
- `monitor` takes no config file: it talks to a running server over HTTP.
- A missing or unreadable file is reported as one line, `lair: Configuration file '<path>' does not exist.`, and exits `1`.

### Usage and `--help`

There is no `--help` flag. Any argument set the host does not recognise prints the usage block and exits `1`:

```text
usage:
  lair serve -c <configPath>   serve the configured model over the OpenAI-compatible HTTP API
  lair chat -c <configPath>    chat with the configured model in an interactive REPL
  lair monitor [-u URL] [--interval ms]  render live engine stats from a serving server (default http://127.0.0.1:8080)
  lair model fetch|check|list [-c <configPath>]  fetch, report update candidates for, and list pinned model packs
  lair --version               print the lair version
```

`lair model` with an unusable argument tail prints its own, narrower usage block (only the three `model` subcommands) and exits `2`.

### Global options

| Flag | Default | Meaning |
|---|---|---|
| `--version`, `-v` | — | Print `lair <version>` (for example `lair 0.4.0`) and exit `0`. Recognised only when it is the single argument. |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | The command did its job (for `model check`, a candidate delta is a completed check, not a failure). |
| `1` | A configuration or load failure, a refused operation, a pin mismatch, an unreachable server, or an unrecognised command line. |
| `2` | `model` arguments that do not form a usable command (usage error). |

---

## `serve`

```powershell
lair serve -c lair.yaml
```

Loads the config, opens the model, and serves the OpenAI-compatible HTTP API until interrupted.

| Argument | Default | Meaning |
|---|---|---|
| `-c <path>` | required | Config file to load. |

No other flags are accepted: `serve` matches exactly `serve -c <path>`, so extra arguments print the usage block and exit `1`.

### What it prints at load

Load decisions are logged as structured lines before the server starts, so an operator can see what the engine actually chose rather than what the config asked for. The host prints them through a single-line console formatter (level and category prefix), and the structured fields below are what each line carries. These appear on every load:

```text
tool_parser model='smollm-135m-instruct-q8_0' choice=fenced
reasoning model='smollm-135m-instruct-q8_0' policy=keep
prefill_kernel model='smollm-135m-instruct-q8_0' kernel=gemv
compute_topology model='smollm-135m-instruct-q8_0' nodes=1 logical_processors=32 node_mb_total=65536
numa_placement model='smollm-135m-instruct-q8_0' node=0 spread=auto→local core_set='0-7'
compute_grant model='smollm-135m-instruct-q8_0' requested=8 granted=8 exclusive_cores=8 shared_cores=0 node_allocations=… nodes=1 logical_processors=32
lair serving 'smollm-135m-instruct-q8_0' on http://127.0.0.1:8080
```

Conditional lines appear only when the model or config carries the matching facts:

| Line | Printed when |
|---|---|
| `model_profile model='…' architecture='…' profile=…` | The file's declared architecture had no registered profile and was accepted on an exact llama-shaped tensor match. |
| `tokenizer_default model='…' key=… default=…` | A profile-scoped tokenizer default filled a special-token id the GGUF does not declare. |
| `reasoning_region model='…' open_token_id=… close_token_id=…` | The tokenizer carries `<think>` / `</think>` as whole tokens, so a reasoning budget can bind. |
| `rope_scaling model='…' variant=… multiplier=… original_context_length=…` | The GGUF declares rope scaling or a partial rotary slice. |
| `sparse_ffn model='…' layout=… rank=… strategy=…` | The file declares routed experts. |
| `ledger_claim model='…' resource='weights'\|'kv'\|'prefill_buffers' …` | The model entry declares `budgets:`. |
| `host_capacity model='…' … accepted against reported topology` | Declared `host:` totals disagree with what the host reports; this is a warning, and startup continues. |
| `kv_blocks model='…' value=… deprecated: …` | Both `kv_blocks` and `budgets.kv.caps.dram_blocks` state the same ceiling. |
| `update_candidate model='…' check skipped: …` | A `source:` model's HEAD check could not run (offline, gated, transport failure). Serving continues. |

A model that declares `max_reasoning_tokens` on a vocabulary that cannot carry `<think>` / `</think>` as whole tokens is refused at open, naming the model, the key, and the marker.

### Posture and endpoints

- Binding comes from the config, not the command line: `server.bind` (default `127.0.0.1`) and `server.port` (default `8080`). Loopback only unless you deliberately change `server.bind`.
- `GET /healthz` answers `{"status":"ok","model_ready":true}` — useful as a readiness probe:

  ```powershell
  curl http://127.0.0.1:8080/healthz
  ```

- `GET /ui` serves the Web UI (a tabbed monitor plus a browser chat client). `GET /` redirects permanently to `/ui/`. The UI is only served when the host build bundled it; a build with `-p:SkipWebUi=true` serves no UI and `/ui` returns 404.
- `GET /monitor/stream` is the SSE engine snapshot stream that `monitor` and the Web UI both consume. Integer-dot CPU snapshots report resolved `metrics_cpu_isa` and `metrics_cpu_gather`; CUDA-backed serves also publish GPU utilization, PCIe transfer totals, and persistent device VRAM bytes.

### Hot reload

`serve` is the only verb that watches anything. The watched file is the one named by `-c`: edits are coalesced (250 ms quiet window, with up to 4 s of retries while an editor holds the file), re-validated through the normal loader, and applied field by field. Applied and refused fields are logged as `config_reload field=… class=… outcome=… old=… new=…` lines plus one JSON summary line. A broken edit never touches the running engine: the candidate is dropped whole and every current value stays in force. See `CONFIG.md` for which keys are live, new-requests-only, or restart-only.

### Ctrl+C

The first `Ctrl+C` shuts the server down gracefully (in-flight work is cancelled, pools freed). A second `Ctrl+C` exits immediately without waiting.

### Exit codes

`0` after a clean run, `1` when the config fails to load or the model cannot be opened (`lair: <reason>` on stderr).

---

## `chat`

```powershell
lair chat -c lair.yaml
```

An interactive REPL over the configured model. It opens its own engine — it does not talk to a running `serve` process, and it never watches the config file.

| Argument | Default | Meaning |
|---|---|---|
| `-c <path>` | required | Config file to load. |

On start it prints one banner line:

```text
model: smollm-135m-instruct-q8_0  context: 4096  quant: Q8_0
```

then a `> ` prompt per turn. Replies stream token by token. Each reply is capped at 256 tokens; that cap is fixed in the REPL and is not a CLI flag or a config key.

### REPL commands

| Input | Effect |
|---|---|
| any other line | A user message; the assistant answers and the exchange is appended to the rolling history. |
| `/exit` or `/quit` | End the REPL, exit `0`. |
| `/reset` | Clear the conversation history; the next turn starts fresh. |
| empty line | Ignored. |
| end of input (EOF) | Ends the REPL cleanly, exit `0`. |

Conversation history is rolling: when a turn no longer fits the model's context window, the REPL drops the oldest exchange and retries once. If the turn still does not fit, it is discarded and reported as a single `error: …` line, and the REPL keeps reading. Failures never print stack frames.

### Ctrl+C

`Ctrl+C` aborts the generation in flight and ends the REPL cleanly: it prints `interrupted` and exits `0`. It does not just cancel the turn.

### Exit codes

`0` on a clean or interrupted exit, `1` when the engine cannot be opened (reported as `error: …`).

---

## `monitor`

```powershell
lair monitor -u http://127.0.0.1:8080 --interval 1000
```

Follows a running server's `GET /monitor/stream` SSE snapshots and re-renders one table row per snapshot. It reads a live server; it does not read a config file.

| Flag | Default | Accepted values | Meaning |
|---|---|---|---|
| `-u <URL>` | `http://127.0.0.1:8080` | An absolute URL only | Base URL of the serving server. `/monitor/stream` is appended to it. A relative or missing value is refused: `lair: monitor -u needs an absolute server URL, e.g. http://127.0.0.1:8080`. |
| `--interval <ms>` | `0` | A non-negative integer | Minimum milliseconds between rendered rows. Snapshots arriving faster than this are folded (dropped, not queued), so a chatty server cannot flood the console. `0` renders every snapshot. The server itself emits a snapshot every 2 seconds. |

An unrecognised option is refused as `lair: unknown monitor option '<flag>'`.

Note: `monitor` requires at least one option in the current build — a bare `monitor` with no flags falls through to the usage block and exits `1`. Pass `-u` or `--interval` explicitly (for example `monitor --interval 0`).

### The table

```text
model                 tok/s   step ms  seqs/step  cache hit    KV used/cap  active
smollm-135m-instruct  11.62     86.0      1.00     42.9%       17/256       1
smollm-135m-f16        9.80     94.0      1.00      0.0%       17/256       1  gpu 2.64%  pcie 1.5 MiB over 224 copies  persistent-vram 13.5 MiB
```

| Column | Meaning |
|---|---|
| `tok/s` | Aggregate decode tokens per second. |
| `step ms` | Average step time in milliseconds. |
| `seqs/step` | Average number of sequences merged into each batch step (batch occupancy). |
| `cache hit` | Share of admitted requests that reused cached prefix KV blocks. |
| `KV used/cap` | KV blocks in use against the model's live cap. |
| `active` | Sequences currently admitted. |
| `gpu` | CUDA utilization percentage for the served model, printed when nonzero. |
| `pcie` | PCIe transfer bytes and copy count for the served model, printed when nonzero. |
| `persistent-vram` | Persistent device VRAM held by the served model, printed when nonzero. |
| `cpu isa` | Resolved integer-dot ISA (`scalar`, `avx2`, or `vnni`) for the served model. |
| `gather` | Grid staging encoding paired with the resolved ISA (`scalar` or `gather`). |

Keep-alive frames and malformed snapshots are skipped rather than ending the table. Integer-dot CPU rows carry `cpu isa=<scalar|avx2|vnni> gather=<scalar|gather>`; the accelerator tail appears only for CUDA-backed serves with nonzero accelerator facts, and CPU-only serves omit it.

### Ctrl+C and exit codes

`Ctrl+C` ends the table cleanly with exit `0`. A server that cannot be reached prints exactly one actionable line and exits `1`:

```text
error: cannot reach monitor stream at http://127.0.0.1:8099/monitor/stream (No connection could be made …)
error: server answered 404 for http://127.0.0.1:8080/monitor/stream — is this a lair serve host?
```

Known limitation: if the serving process is killed while `monitor` is attached, the torn stream surfaces as an unhandled `System.IO.IOException` instead of the clean notice above. This is a tracked defect, not intended behaviour.

---

## `model`

```powershell
lair model fetch QuantFactory/SmolLM-135M-Instruct-GGUF -c lair.yaml
lair model check manual-fetch-smollm -c lair.yaml
lair model list -c lair.yaml
```

Fetch, report update candidates for, and list pinned model packs. `-c <path>` is optional and changes which pack directories the command works on.

| Flag | Meaning |
|---|---|
| `-c <configPath>` | Read the pack directory beside each config model's `path`. Without it, the pack directory is a `models` directory beside the executable you ran — the publish directory for a released `lair.exe`, or `src/Lair.Host/bin/<Configuration>/net10.0/models` for a built checkout. A missing config is reported as `lair: Configuration file '<path>' does not exist.` and exits `1`; a `-c` with no value exits `2`. |

### `model fetch`

```text
lair model fetch <repo> [--revision <rev>] [--file <path>]... [--model <name>]
```

Lands a declared file set in its pack directory and records the pin in that pack's `manifest.json`.

| Argument | Default | Meaning |
|---|---|---|
| `<repo>` | required | Exactly one Hugging Face repo slug. A second bare argument is a usage error. |
| `--revision <rev>` | `main` when no config declares the source | A commit sha or a branch name. A branch resolves once, at fetch, and the resolved 40-character sha is what the manifest records. Only one `--revision` is accepted. |
| `--file <path>` | none | A repo-relative file to fetch; repeatable. Absolute paths and paths containing `..` are refused so nothing lands outside the pack directory. |
| `--model <name>` | — | Which config entry to fetch for, when the config declares the repo on more than one model. |

Behaviour worth knowing:

- **With `-c`, fetch only ever fetches what a model entry declares.** An undeclared repo, an undeclared file, or a `--revision` that disagrees with the entry's pinned revision is refused (exit `1`) rather than widening the pin. If several models declare the same repo, `--model` is required.
- **Without `-c`, at least one `--file` is required**, and the pack entry is named after the last path segment of the repo unless `--model` names it.
- **Downloads land in a `<file>.part` sibling.** A `.part` resumes from its current length, is finalised (renamed to the loadable name) only after the whole declared size has arrived and LAIR's own sha256 agrees with the pin, and is deleted when the hash disagrees so a retry re-downloads fresh. A `.part` file is never a loadable path.
- **A declared file set is fetched as one coherent unit at one resolved revision.** A missing or refused file refuses the whole set, so a shard set is never recorded as loadable in pieces.
- **Fetch is idempotent.** An already-verified final file is hashed and recorded without re-streaming, so a repeated fetch never double-downloads.
- **One fetch per artifact.** A per-`(repo, revision, path)` lock marker under `.fetch-locks/` in the pack directory serialises concurrent fetches, and `serve` uses the same lock, so a killed fetch does not wedge the artifact (a marker whose recorded process is gone is reclaimed).
- Hashes are computed by LAIR after download; the hub's own reported hash is never trusted as the pin.

Output on success (exit `0`):

```text
fetched SmolLM-135M-Instruct-GGUF from QuantFactory/SmolLM-135M-Instruct-GGUF at revision d36054e030c66b4be24b0c65513ece348db06ba5:
  SmolLM-135M-Instruct.Q8_0.gguf  sha256=76520babb0daebccb6e17d2f38504ece61356a0ca958d8e8795ef4d23c23c1f0  size=144810912
```

A refusal leaves the manifest untouched: nothing half-fetched is loadable.

### `model check`

```text
lair model check [model]
```

Report-only. It consults the hub's current HEAD, lists the file tree at that revision, and diffs the pinned file set against it. It never downloads, never re-pins, never rewrites a manifest, and never touches the load path — a candidate stays a candidate until an operator adopts it behind a restart.

| Argument | Default | Meaning |
|---|---|---|
| `[model]` | all pinned entries | At most one model name. With `-c`, a named model narrows the search to that model's own pack directory. |

Reported delta kinds: revision moved, a pinned file no longer present at HEAD, a per-file hash difference, a per-file size difference, and a HEAD file outside the pinned set.

```text
model: manual-fetch-smollm
  repo: QuantFactory/SmolLM-135M-Instruct-GGUF
  pinned: a0c73bc9f01121c33dee05af504d753fed4089b3
  head: d36054e030c66b4be24b0c65513ece348db06ba5
  revision delta: a0c73bc9f01121c33dee05af504d753fed4089b3 -> d36054e030c66b4be24b0c65513ece348db06ba5
  hash delta: 'SmolLM-135M-Instruct.Q8_0.gguf' pinned 76520b… -> head 8f3c1a…
  candidate: 16 delta(s) against head
candidate deltas: 16
```

When HEAD already matches the pin the entry reports `current: pinned matches head` and the run ends with `all pinned revisions current.`. Exit `0` in both cases: a candidate is a completed check. Exit `1` when the named model has no manifest entry (`error: no manifest entry pins model '<name>'.`) or when HEAD could not be consulted (offline, gated, unreachable) — a failed check names what was consulted and never substitutes a revision. With no pinned entries at all it prints `no manifest entries to check.` and exits `0`.

The same candidate is available to a running server as the optional `metrics_update_candidate` field on `/monitor/stream`.

### `model list`

```text
lair model list
```

Prints what each pack pins. It takes no arguments (any argument is a usage error, exit `2`).

```text
model: manual-fetch-smollm
  repo: QuantFactory/SmolLM-135M-Instruct-GGUF
  revision: d36054e030c66b4be24b0c65513ece348db06ba5
  file: SmolLM-135M-Instruct.Q8_0.gguf  sha256=76520babb0daebccb6e17d2f38504ece61356a0ca958d8e8795ef4d23c23c1f0  size=144810912  fetched_at=2026-11-15T08:30:00.0000000+00:00
```

A pack with no manifest prints `no manifest in pack '<directory>'`; a manifest with no entries prints `no model entries in manifest.`. Both exit `0`. A manifest that exists but cannot be parsed is refused (exit `1`) rather than guessed at.

### Gated repositories and tokens

| Variable | Meaning |
|---|---|
| `LAIR_HF_TOKEN` | The token LAIR reads first for gated Hugging Face repositories. |
| `HUGGING_FACE_HUB_TOKEN` | Read as a fallback when `LAIR_HF_TOKEN` is unset, for operators already exporting the conventional variable. |

```powershell
$env:LAIR_HF_TOKEN = "hf_..."
lair model fetch org/gated-model -c lair.yaml
```

- The token is read from the process environment only. It is never accepted in `lair.yaml`, never accepted as a command-line argument, and never written to a pack manifest.
- It is attached only to requests to `huggingface.co`.
- It is never echoed in logs, error text, or CLI output. A gated refusal names the repo and the reason and points at the environment variable without printing the token.
- A whitespace-only value counts as unset rather than as a credential.

### `model` exit codes

| Code | Meaning |
|---|---|
| `0` | Fetch landed or verified, check completed (candidate or no candidate), list completed. |
| `1` | A fetch refused, a pin disagreed with what was asked for, HEAD could not be consulted, an unreadable manifest, or a bad config path. |
| `2` | Arguments that do not form a usable command. |

---

## What the CLI deliberately does not have

LAIR's admin surface is the config file, not the command line. There is no admin API and no CLI command that mutates a running server: `serve` applies changes only through the watched config file, and `monitor` is read-only.

---

## Planned CLI surface

These are **not available on this branch**. Each is listed with its status so a reader cannot mistake it for a working option.

| Feature | Status | Notes |
|---|---|---|
| Packaged release artifacts: a committed publish profile, CI packaging, and a distributable archive | planned | The self-contained single-file `win-x64` publish documented under "Getting the executable" works today, but no publish profile or packaging step is committed in the repository, so a release has to be built locally. |
| Native AOT packaging | planned | Backlog idea (startup cost reduction); nothing implemented. |
| Concurrent GPU-compute overlap across models on the same CUDA device | planned | Multi-model serving is capacity/admission accounting only; `serve`, chat, and model-listing already operate across declared `chat`/`embed` entries today. |
| Whole-model CUDA monitor/tier counters beyond the shipped hybrid split | planned | GPU utilization, PCIe transfer bytes/copies, and persistent device VRAM are shipped for CUDA-backed serves; whole-model CUDA execution, broader tier occupancy, and CUDA kernels across all loader-accepted quant formats remain planned. |
| `embed` model serving (`type: embed`) reachable from `serve` | planned | The config key is validated today but only `chat` is accepted. |

Known open defect, not a feature: `monitor` on a torn event stream (server killed while attached) exits with an unhandled `IOException` instead of a clean notice.
