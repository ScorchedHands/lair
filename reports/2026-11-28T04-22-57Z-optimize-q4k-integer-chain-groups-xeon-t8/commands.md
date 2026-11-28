# Commands for Xeon candidate B attribution 2026-11-28

Host profile: `lair-lab` (`root@192.168.33.53`). Host measurement directory remains on the target at `/root/lair-measurements/qwen25-small-cpu/reports/2026-11-28T04-22-57Z-optimize-q4k-integer-chain-groups-xeon-t8`; this committed report includes the relevant bench arm lines in `logs/q4k-intchain-target-arms.log` and parsed values in `data.json`.

## Host worktree setup
```bash
git -C /root/lair-measurements/qwen25-small-cpu/repo fetch origin phase/mvp7
git -C /root/lair-measurements/qwen25-small-cpu/repo worktree add --detach /root/lair-measurements/qwen25-small-cpu/repo-q4k-intchain FETCH_HEAD
```

## Build posture used for measurement
Full solution build failed on this host because `Lair.Host` runs an npm frontend install and the host lacks npm. Bench-only Release build succeeded:
```bash
cd /root/lair-measurements/qwen25-small-cpu/repo-q4k-intchain
/usr/local/dotnet/dotnet build bench/Lair.Bench/Lair.Bench.csproj -c Release > logs/build-bench.log 2>&1; echo BUILD_EXIT=$?
```

## Bench seam measurement
```bash
cd /root/lair-measurements/qwen25-small-cpu/repo-q4k-intchain
stamp=$(date -u +%Y-%m-%dT%H-%M-%SZ) && report=/root/lair-measurements/qwen25-small-cpu/reports/${stamp}-optimize-q4k-integer-chain-groups-xeon-t8 && mkdir -p "$report/logs" && taskset -c 0-15 /usr/local/dotnet/dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll q4k-widebatch --threads 8 --counts 8,16,32,64 --integer-dot --integer-chain-group 2,4 --reps 5 > "$report/logs/q4k-intchain-t8.log" 2>&1
```

Notes:
- `taskset -c 0-15` is the node0 physical-core posture used for prior Xeon kernel attribution.
- The harness records rounds=5 per arm with settle passes and rotated timing arms; committed target-arm lines preserve those medians/min/max values.
- Float arms are bit-parity-checked by the bench guard; integer arms are divergence-classified because Q8_K activation staging changes numeric operands relative to current float batched dot paths.
