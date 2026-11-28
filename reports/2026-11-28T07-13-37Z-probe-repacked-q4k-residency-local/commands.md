# Repacked Q4_K residency probe commands — 2026-11-28T07:13:37Z

Working directory for all commands: `C:\source\gitlab.scorchedhands.com\personal\LAIR\.agent-worktrees\mvp7`.

Host facts are recorded in `host.json` beside this file. The full-model fixture was read from the main repository because the worktree does not contain a local copy under `models/`:
`C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf`.

## Synthetic Q4_K wide-batch arms

```powershell
dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll q4k-widebatch --threads 8 --reps 5 --counts 32,64 --integer-dot --repacked-residency 8 | Tee-Object -FilePath 'reports/2026-11-28T07-13-37Z-probe-repacked-q4k-residency-local/logs/q4k-widebatch-threads8.log' > $null
```

```powershell
dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll q4k-widebatch --threads 16 --reps 5 --counts 32,64 --integer-dot --repacked-residency 8 | Tee-Object -FilePath 'reports/2026-11-28T07-13-37Z-probe-repacked-q4k-residency-local/logs/q4k-widebatch-threads16.log' > $null
```

## Budget refusal demo

```powershell
dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll q4k-widebatch --threads 1 --reps 3 --counts 8,64 --shapes 5000x256 --integer-dot --repacked-residency 8 --repacked-cap-mb 1 | Tee-Object -FilePath 'reports/2026-11-28T07-13-37Z-probe-repacked-q4k-residency-local/logs/q4k-widebatch-refusal.log' > $null
```

## Full-model prefill stage-profile arms

`--stage-profile` is enabled so each window group publishes substage totals. `forwardCalls=3` means the printed substage values are sums across three timed forward calls, not per-substage medians/min/max samples.

### Threads 8 — off arm

```powershell
dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --threads 8 --lengths 256,1024 --reps 3 --stage-profile --repacked-residency off | Tee-Object -FilePath 'reports/2026-11-28T07-13-37Z-probe-repacked-q4k-residency-local/logs/prefill-off-threads8.log' > $null
```

### Threads 8 — on arm

```powershell
dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --threads 8 --lengths 256,1024 --reps 3 --stage-profile --repacked-residency on | Tee-Object -FilePath 'reports/2026-11-28T07-13-37Z-probe-repacked-q4k-residency-local/logs/prefill-on-threads8.log' > $null
```

### Threads 16 — off arm

```powershell
dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --threads 16 --lengths 256,1024 --reps 3 --stage-profile --repacked-residency off | Tee-Object -FilePath 'reports/2026-11-28T07-13-37Z-probe-repacked-q4k-residency-local/logs/prefill-off-threads16.log' > $null
```

### Threads 16 — on arm

```powershell
dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --threads 16 --lengths 256,1024 --reps 3 --stage-profile --repacked-residency on | Tee-Object -FilePath 'reports/2026-11-28T07-13-37Z-probe-repacked-q4k-residency-local/logs/prefill-on-threads16.log' > $null
```

## Key-line extraction used for this report family

```powershell
Select-String -Path 'reports/2026-11-28T07-13-37Z-probe-repacked-q4k-residency-local/logs/*.log' -Pattern 'repacked_residency|shape=.*arm=|arms_bit_parity|stage window|refusal=' | ForEach-Object { $_.Line } | Out-File -FilePath 'reports/2026-11-28T07-13-37Z-probe-repacked-q4k-residency-local/data/key-measurement-lines.txt' -Encoding utf8
```
