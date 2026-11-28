# Commands for interleaved Xeon repacked Q4_K residency length 1024 — 2026-11-28

Host profile: `lair-lab`. Host worktree `/root/lair-measurements/qwen25-small-cpu/repo-q4k-residency` was already built from pushed commit `47efa42f7a289fe5fc61b45448c74d558a6f71fe`; this run reused the existing bench-only Release build.

Host report directory: `/root/lair-measurements/qwen25-small-cpu/reports/2026-11-28T08-59-17Z-probe-repacked-q4k-residency-xeon-t8-phys8-interleaved-l1024`. Full raw logs remain there; this committed report includes extracted key facts in `logs/interleaved-l1024-key-facts.log`.

## Measurement loop
```bash
cd /root/lair-measurements/qwen25-small-cpu/repo-q4k-residency && model=/root/lair-measurements/qwen25-small-cpu/models/qwen2.5-3b-instruct-q4_k_m.gguf && report=/root/lair-measurements/qwen25-small-cpu/reports/2026-11-28T08-59-17Z-probe-repacked-q4k-residency-xeon-t8-phys8-interleaved-l1024 && mkdir -p "$report/logs" && for r in 1 2 3; do taskset -c 0-15 /usr/local/dotnet/dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll prefill-profile "$model" --threads 8 --lengths 1024 --widths 64 --kernel gemv --reps 3 --stage-profile --repacked-residency off --repacked-cap-mb auto > "$report/logs/off-r$r.log" 2>&1; echo "OFF_R${r}_EXIT=$?"; taskset -c 0-15 /usr/local/dotnet/dotnet bench/Lair.Bench/bin/Release/net10.0/Lair.Bench.dll prefill-profile "$model" --threads 8 --lengths 1024 --widths 64 --kernel gemv --reps 3 --stage-profile --repacked-residency on --repacked-cap-mb auto > "$report/logs/on-r$r.log" 2>&1; echo "ON_R${r}_EXIT=$?"; done && ls -la "$report/logs"
```

## Extraction and band parsing used for the committed summary
Key facts were extracted from each raw log with:
```bash
cd /root/lair-measurements/qwen25-small-cpu/reports/2026-11-28T08-59-17Z-probe-repacked-q4k-residency-xeon-t8-phys8-interleaved-l1024/logs && awk '/repacked_residency|stage window=window kernel=gemv rowWidth=64 length=1024|peakWorkingSetMb/ {print FILENAME": "$0}' off-r*.log on-r*.log
```

Sorted per-arm bands were obtained from the same logs with a generic metric extractor. Example for `total_ms`:
```bash
cd /root/lair-measurements/qwen25-small-cpu/reports/2026-11-28T08-59-17Z-probe-repacked-q4k-residency-xeon-t8-phys8-interleaved-l1024/logs && for arm in off on; do echo "ARM=$arm"; for key in total_ms attention_output_ms feed_forward_down_ms peakWorkingSetMb; do grep -hE "(^| )$key=" $arm-r*.log | awk "{for(i=1;i<=NF;i++){split(\$i,a,\"=\"); if(a[1]==\"$key\") print a[2]}}" | sort -n | tr '\n' ' '; echo; done; done
```
