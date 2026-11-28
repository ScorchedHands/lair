# Commands for Xeon static row-partition attribution 2026-11-28

Host profile: `lair-lab` (`root@192.168.33.53`). Host worktree was created from pushed commit `5b26bfe8eb59c90c219c67207770636a282d9e24` (`perf(cpu): land unpromoted static row partition probe`) at `/root/lair-measurements/qwen25-small-cpu/repo-static-row-partition`.

## Host worktree setup
```bash
cd /root/lair-measurements/qwen25-small-cpu/repo && git fetch origin phase/mvp7
git -C /root/lair-measurements/qwen25-small-cpu/repo worktree add --detach /root/lair-measurements/qwen25-small-cpu/repo-static-row-partition FETCH_HEAD
```

## Build posture used for measurement
Full solution build is avoided on this host because `Lair.Host` requires npm frontend setup that the measurement lane does not provide. Bench-only Release build succeeded:
```bash
cd /root/lair-measurements/qwen25-small-cpu/repo-static-row-partition && stamp=$(date -u +%Y-%m-%dT%H-%M-%SZ) && report=/root/lair-measurements/qwen25-small-cpu/reports/${stamp}-probe-static-row-partitioning-xeon-t8-ht32 && mkdir -p "$report/logs" && /usr/local/dotnet/dotnet build bench/Lair.Bench/Lair.Bench.csproj -c Release > "$report/logs/build-bench.log" 2>&1; echo BUILD_EXIT=$? REPORT_DIR="$report"
```

## Bench seam measurement
Report directory on host: `/root/lair-measurements/qwen25-small-cpu/reports/2026-11-28T06-03-03Z-probe-static-row-partitioning-xeon-t8-ht32`. Full raw logs remain there; this committed report includes the target-arm lines parsed into `data.json`.

### Unpinned threads=32 arm
```bash
cd /root/lair-measurements/qwen25-small-cpu/repo-static-row-partition && report=/root/lair-measurements/qwen25-small-cpu/reports/2026-11-28T06-03-03Z-probe-static-row-partitioning-xeon-t8-ht32 && /usr/local/dotnet/dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll q4k-widebatch --threads 32 --counts 8,16,32,64 --static-partition 8 --reps 7 > "$report/logs/wb-t32-unpinned-static.log" 2>&1
```

### Node0 HT-pinned threads=32 arm
Host topology recorded by the earlier pinning report: node0 physical CPUs `0-15`, siblings `32-47`; node1 physical CPUs `16-31`, siblings `48-63`. This arm isolates node0 and oversubscribes its 32 logical processors.
```bash
cd /root/lair-measurements/qwen25-small-cpu/repo-static-row-partition && report=/root/lair-measurements/qwen25-small-cpu/reports/2026-11-28T06-03-03Z-probe-static-row-partitioning-xeon-t8-ht32 && taskset -c 0-15,32-47 /usr/local/dotnet/dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll q4k-widebatch --threads 32 --counts 8,16,32,64 --static-partition 8 --reps 7 > "$report/logs/wb-t32-node0ht-static.log" 2>&1
```

Notes:
- The initial combined command used malformed `taskset -c0-15,32-47`; the successful HT arm was rerun with a space after `-c`, as shown above.
- Float-class arms (`auto`, `streaming`, `tile`, `static-w8`) are bit-parity-checked per count by the bench harness; no integer-domain Q4_K arm was enabled for this dispatch-schedule probe.
