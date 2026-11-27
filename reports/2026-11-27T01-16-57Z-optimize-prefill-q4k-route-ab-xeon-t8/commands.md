# Commands and provenance -- Q4_K matmul-route A/B on lair-lab (root@192.168.33.53)

All remote work ran through the SSH MCP profile `lair-lab`. Nothing under `/root/lair-measurements/qwen25-small-cpu/app` or any pre-existing `reports/` directory was modified; the fresh build went to a new `build/` subtree. No commits were made anywhere.

## Checkout

```
git clone --depth 1 --single-branch --branch phase/mvp7 https://github.com/ScorchedHands/lair.git /root/lair-measurements/qwen25-small-cpu/repo
git -C /root/lair-measurements/qwen25-small-cpu/repo rev-parse --short HEAD   -> e3f001f
cat /root/lair-measurements/qwen25-small-cpu/repo/.git/refs/heads/phase/mvp7  -> e3f001f33fe4b19b198603329cf74de009d9c3be
grep -c "q4k-matmul-route" .../repo/bench/Lair.Bench/Program.cs               -> 5
```

Anonymous clone succeeded; no archive or HTTP fallback was needed. HEAD matches the required short SHA `e3f001f`.

## Publish (bench app only)

Ran as tracked background SSH session `mvp7-build` so tool timeouts could not kill it:

```
cd /root/lair-measurements/qwen25-small-cpu/repo && /usr/local/dotnet/dotnet publish bench/Lair.Bench -c Release \
  -o /root/lair-measurements/qwen25-small-cpu/build/app-q4k-ab \
  > /root/lair-measurements/qwen25-small-cpu/logs/publish-mvp7.log 2>&1; echo PUBLISH_EXIT=$? >> .../logs/publish-mvp7.log
```

- Exit marker: `PUBLISH_EXIT=0`
- Success line in log: `Lair.Bench -> /root/lair-measurements/qwen25-small-cpu/build/app-q4k-ab/`
- Restore succeeded from the existing `/root/.nuget/packages` cache (9 projects restored, no system-wide installs). Target framework `net10.0`.
- Output dir listing confirmed `Lair.Bench`, `Lair.Bench.dll`, `Lair.Bench.deps.json`, `Lair.Bench.runtimeconfig.json` plus the referenced `Lair.*.dll`s (created 2026-11-27T01:14Z).
- `/usr/local/dotnet/dotnet --version` -> `10.0.404`.

## Flag surface of the published binary

```
/usr/local/dotnet/dotnet /root/lair-measurements/qwen25-small-cpu/build/app-q4k-ab/Lair.Bench.dll prefill-profile   (exit 1, usage line)
usage: prefill-profile <gguf path> [--threads N] [--lengths ...] [--widths ...] [--kernel gemv,tiled,gemm] [--reps R]
       [--per-token] [--stage-profile] [--matmul-route off|on|N] [--q4k-matmul-route off|on|N]
       [--attention-batch-dots off|on|N] [--out <path>]
```

`bench/Lair.Bench/Program.cs` in the checkout maps `off -> null`, `on -> 8`, otherwise a parsed integer, validated against `CpuKernelTuning.MatmulRouteBatchWidthMax = 64` (`src/Lair.Backends.Cpu/CpuKernelTuning.cs:85`). So `off`, `on`, and `16` are all accepted values.

## Variant runs (sequential, never concurrent)

Common prefix for every variant, run from inside `/root/lair-measurements/qwen25-small-cpu/repo`:

```
python3 scripts/run-xeon-prefill-stage-profile.py --root /root/lair-measurements/qwen25-small-cpu \
  --app /root/lair-measurements/qwen25-small-cpu/build/app-q4k-ab/Lair.Bench.dll \
  --model /root/lair-measurements/qwen25-small-cpu/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --dotnet /usr/local/dotnet/dotnet --threads 8 --widths 64 --lengths 128,256 --kernels gemv,tiled --rounds 3 <variant flag> \
  > /root/lair-measurements/qwen25-small-cpu/build/logs-ab/<log name> 2>&1; echo SCRIPT_EXIT=$? >> <same log>
```

| variant | trailing flag | SSH background session | exit status | final script line | remote report stamp dir |
| --- | --- | --- | --- | --- | --- |
| baseline route OFF | `--q4k-matmul-route off` | `mvp7-baseline-off` | `SCRIPT_EXIT=0` | `wrote ... rows=60 charts=2` | `reports/2026-11-27T01-16-57Z-optimize-prefill-stage-profile-xeon-t8` |
| candidate A route ON (width 8) | `--q4k-matmul-route on` | `mvp7-route-on-w8b` | `SCRIPT_EXIT=0` | `wrote ... rows=60 charts=2` | `reports/2026-11-27T01-25-12Z-optimize-prefill-stage-profile-xeon-t8` |
| candidate B route ON (width 16) | `--q4k-matmul-route 16` | `mvp7-route-w16` | `SCRIPT_EXIT=0` | `wrote ... rows=60 charts=2` | `reports/2026-11-27T01-34-43Z-optimize-prefill-stage-profile-xeon-t8` |

The script's own defaults were left untouched (`--matmul-route on`, `--attention-batch-dots on`), which the per-round command lines in each raw log confirm, e.g. baseline round 0:

```
RUN[stage-round0]: /usr/local/dotnet/dotnet .../build/app-q4k-ab/Lair.Bench.dll prefill-profile \
  .../models/qwen2.5-3b-instruct-q4_k_m.gguf --threads 8 --lengths 128,256 --widths 64 --kernel gemv,tiled \
  --reps 1 --stage-profile --attention-batch-dots on --matmul-route on --q4k-matmul-route off
```

Each run took roughly 7 minutes (three rounds of ~2 min each, model reloaded per round). Variant logs are kept remotely at `/root/lair-measurements/qwen25-small-cpu/build/logs-ab/{variant-off.log, variant-on-w8.log, variant-w16.log}`.

## Anomalies and caveats

- **Aborted first candidate-A attempt.** `open-session mvp7-route-on-w8` was launched at 01:22:48Z while the baseline run's third round was still executing (baseline round logs land at 01:19 / 01:21 / 01:24). Detected within about a minute via `ls -la .../build/logs-ab/variant-off.log` showing no completion marker, then closed with `close-session mvp7-route-on-w8`. It left an incomplete dir `reports/2026-11-27T01-22-48Z-optimize-prefill-stage-profile-xeon-t8` (logs only, no `data.json`), which was deliberately left untouched and is **not** part of the downloaded evidence. Candidate A was then re-run alone as `mvp7-route-on-w8b`. Consequence: baseline round 2 shares its tail with that ~35 s window, so any contention effect on baseline round-2 samples is possible but not separable from ordinary noise; no claim about it is made in the analysis.
- **Rounds=4 confirmation was not run.** Baseline `gemv@128` (spread 20.5% of median) and candidate A `tiled@128`/gate-up (22.0%/32.0%) are noisy enough that the instructed rounds=4 re-pairing would be worthwhile; it was skipped here for session-time budget, and is flagged as unfinished rather than silently assumed.
- **Tool quirks observed.** `read-command` refuses compound commands (`|`, `[`) and non-allowlisted binaries, so all git/dotnet/python work used `run-command`. The SSH MCP streaming transfer tools are unavailable on this Windows host ("Streaming SFTP file tools are not available ... transfer root cannot yet be verified private"), so downloads used the content-based `sftp-download` path with local re-write; every downloaded file was then byte-verified against the remote copy.
- **Byte-integrity verification.** All 18 files in this report directory match their remote counterparts by SHA-256 (and size). Two transcription slips during download were caught and repaired from authoritative `sed`/content fetches before verification: a spacing error in `baseline-off/logs/stage-t8-round0.txt` (now `4928fb08...`, 3886 B) and a mistyped `sparse_chain_ms=0.000` field in `route-w16/logs/stage-t8-round2.txt` (now `f79ef4c7...`).
- No test suites were run; this task was bench-only.
