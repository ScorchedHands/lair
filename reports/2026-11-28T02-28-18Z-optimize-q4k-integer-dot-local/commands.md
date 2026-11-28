# Commands for this attribution pass 2026-11-28

Run from the worktree `C:\source\gitlab.scorchedhands.com\personal\LAIR\.agent-worktrees\mvp7` (branch `phase/mvp7`). The authoritative per-arm lines are in `logs/q4k-widebatch-integerdot-t8.txt`; this file records how they were produced.

```powershell
# Gates (main-agent verification)
dotnet build Lair.sln --nologo
dotnet test Lair.sln --filter "Category=DenseWeightFusion|Kernel|CpuTuning|Profile&Category!=Cuda" -c Release --nologo
dotnet test Lair.sln --filter Category=Parity -c Release --nologo

# Seam measurement (bench arms recorded rounds=5 in the raw log)
dotnet run --project bench/Lair.Bench -c Release -- q4k-widebatch --integer-dot   # plus the harness's existing float arms auto/streaming/tile for comparison, threads=8, counts 8/16/32/64 on shapes 11008x2048 and 2048x11008
```

Notes:
- `--integer-dot` is the bench pin added by this change (`bench/Lair.Bench/Q4KWideBatchBench.cs`); it forces an AVX2-only posture for that arm and classifies divergence from float arms instead of asserting global bit-parity. Auto-resolved posture leaves `Q4KIntegerDotBatchWidth = 0`, so the variant is never selected outside pins.
- No golden files under Parity fixtures were regenerated or edited in this pass; the gate reported unchanged goldens because the variant was not auto-selected on this AVX-VNNI host.
