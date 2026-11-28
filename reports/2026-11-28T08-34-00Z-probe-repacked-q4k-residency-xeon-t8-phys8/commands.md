# Commands for Xeon repacked Q4_K residency attribution 2026-11-28

Host profile: `lair-lab` (`root@192.168.33.53`). Host worktree was created from pushed commit `47efa42f7a289fe5fc61b45448c74d558a6f71fe` (`perf(cpu): land budgeted repacked q4k residency probe`) at `/root/lair-measurements/qwen25-small-cpu/repo-q4k-residency`.

## Host worktree setup
```bash
cd /root/lair-measurements/qwen25-small-cpu/repo && git fetch origin phase/mvp7
git -C /root/lair-measurements/qwen25-small-cpu/repo worktree add --detach /root/lair-measurements/qwen25-small-cpu/repo-q4k-residency FETCH_HEAD
```

## Build posture used for measurement
Full solution build is avoided on this host because `Lair.Host` requires npm frontend setup that the measurement lane does not provide. Bench-only Release build succeeded:
```bash
cd /root/lair-measurements/qwen25-small-cpu/repo-q4k-residency && stamp=$(date -u +%Y-%m-%dT%H-%M-%SZ) && report=/root/lair-measurements/qwen25-small-cpu/reports/${stamp}-probe-repacked-q4k-residency-xeon-t8-phys8 && mkdir -p "$report/logs" && /usr/local/dotnet/dotnet build bench/Lair.Bench/Lair.Bench.csproj -c Release > "$report/logs/build-bench.log" 2>&1; echo BUILD_EXIT=$? REPORT_DIR="$report"
```

## Stage-profile A/B measurement
Report directory on host: `/root/lair-measurements/qwen25-small-cpu/reports/2026-11-28T08-34-00Z-probe-repacked-q4k-residency-xeon-t8-phys8`. Full raw logs remain there; this committed report includes extracted key facts in `logs/prefill-off-on-t8-phys-key-facts.log`.

```bash
cd /root/lair-measurements/qwen25-small-cpu/repo-q4k-residency && model=/root/lair-measurements/qwen25-small-cpu/models/qwen2.5-3b-instruct-q4_k_m.gguf && report=/root/lair-measurements/qwen25-small-cpu/reports/2026-11-28T08-34-00Z-probe-repacked-q4k-residency-xeon-t8-phys8 && taskset -c 0-15 /usr/local/dotnet/dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll prefill-profile "$model" --threads 8 --lengths 128,256 --widths 64 --kernel gemv --reps 3 --stage-profile --repacked-residency off --repacked-cap-mb auto > "$report/logs/prefill-off-t8-phys.log" 2>&1; echo OFF_EXIT=$? && taskset -c 0-15 /usr/local/dotnet/dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll prefill-profile "$model" --threads 8 --lengths 128,256 --widths 64 --kernel gemv --reps 3 --stage-profile --repacked-residency on --repacked-cap-mb auto > "$report/logs/prefill-on-t8-phys.log" 2>&1; echo ON_EXIT=$? REPORT_DIR="$report"
```

Notes:
- `taskset -c 0-15` is node0 physical-core pinning, matching the prior Xeon stage-profile baseline posture.
- The stated repacked arm pins the profile backend to AVX2 integer-dot posture for comparable off/on arms; it prepares dense Q4_K profile weights under per-tensor auto caps and prints residency facts rather than narrowing over-cap copies.
