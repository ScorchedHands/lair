# Provenance — 2026-11-27T23-07-15Z-optimize-affinity-selfplacement-xeon-t8

Background SSH session `affinity_ab` on profile `lair-lab`, launched after host idle check and `git fetch origin phase/mvp7` (fetched range `016dd9c..1dcb354`; local branch pushed to `origin/phase/mvp7` before this run). Chain as executed:

```
cd /root/lair-measurements/qwen25-small-cpu/repo \
  && git worktree add ../repo-affinity --detach 1dcb354 \
  && cd /root/lair-measurements/qwen25-small-cpu/repo-affinity \
  && /usr/local/dotnet/dotnet publish bench/Lair.Bench -c Release -o /root/lair-measurements/qwen25-small-cpu/build/app-affinity-2026-11-27T23Z   # published clean, no warnings reported in tail
  && python3 /root/lair-tools/run-xeon-alternating-build-ab.py \
       --old-app .../build/app-posture-2026-11-27T06-53-30Z/Lair.Bench.dll \
       --new-app .../build/app-affinity-2026-11-27T23Z/Lair.Bench.dll \
       --model /root/lair-measurements/qwen25-small-cpu/models/qwen2.5-3b-instruct-q4_k_m.gguf \
       --dotnet /usr/local/dotnet/dotnet --threads 8 --lengths 256 --widths 64 \
       --rounds 13 --warmup-rounds 3 \
       --out-root /root/lair-measurements/qwen25-small-cpu/reports \
       --name optimize-affinity-selfplacement-xeon-t8
```

Harness script uploaded byte-exact from this worktree's `scripts/run-xeon-alternating-build-ab.py` (local SHA-256 `F3088DC2495DA0E32F0E37F7F627AE37002BB08913D96B111137C80A4BB5949D` = remote file hash, verified with `sha256sum` and `python3 -m py_compile`). Invocation template per arm: `<dotnet> <app> prefill-profile <model> --threads 8 --lengths 256 --widths 64 --kernel gemv,tiled --reps 1 --stage-profile --attention-batch-dots on --matmul-route on` (no route pins; both apps see identical flags). Rounds ran strictly alternating with odd-round order reversal.

## Remote artifact hashes
- `data.json` `8567dea6a0294193ddcdd110bdb650605f767415c4bf1e55e174cf1d3af04ca4`
- `summary.md` (host version) `9d364a5be9c637642e9ce1f079d7479dc3ec7850db8dc133e5b78425829482a3`
- No `report.html` was produced by this harness (headline extract only); per-arm raw logs remain under `…/logs/w*.txt`, `r*-old.txt`, `r*-new.txt`.

## Anomalies / caveats
None observed in execution: worktree add, publish, and all 32 invocations completed. The old app predates the affinity commit (built from the pre-`1dcb354` checkout whose only uncommitted deltas were bench-seam files that do not alter `prefill-profile`). A new host worktree directory `/root/lair-measurements/qwen25-small-cpu/repo-affinity` and build dir `app-affinity-2026-11-27T23Z` now exist additively; nothing was deleted, and the original dirty checkout remains untouched.
