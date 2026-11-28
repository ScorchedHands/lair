# Commands for candidate B local attribution 2026-11-28

Run from the worktree `C:\source\gitlab.scorchedhands.com\personal\LAIR\.agent-worktrees\mvp7` (branch `phase/mvp7`). Raw logs are committed under `logs/`; parsed machine-readable summaries are in `data.json` and `measurements.csv`.

## Gates
```powershell
dotnet build Lair.sln --nologo
dotnet test Lair.sln --filter "Category=DenseWeightFusion|Kernel|CpuTuning|Profile&Category!=Cuda" -c Release
dotnet test Lair.sln --filter "Category=Kernel|CpuTuning&Category!=Cuda" -c Release
dotnet test Lair.sln --filter Category=Parity -c Release
```

## Bench seam measurement
Candidate A baseline rerun and candidate B default-arm rerun used the same command posture:
```powershell
dotnet run --project bench/Lair.Bench -c Release -- q4k-widebatch --threads 8 --counts 8,16,32,64 --integer-dot --reps 5
```

Candidate B chain-group width arms added `--integer-chain-group`:
```powershell
dotnet run --project bench/Lair.Bench -c Release -- q4k-widebatch --threads 8 --counts 8,16,32,64 --integer-dot --integer-chain-group 2,4 --reps 5
```

Notes:
- The historical candidate-A original log is copied into `logs/candidateA-original.log` from the earlier committed report; its command posture is recorded in `reports/2026-11-28T02-28-18Z-optimize-q4k-integer-dot-local/commands.md`.
- The bench harness adds integer arms named `integer-dot`, `integer-g2`, and `integer-g4`; each stated chain group shares the same settle passes and rotated timing schedule as the float arms.
- Float arms are bit-parity-checked; integer arms are divergence-classified because Q8_K activation staging changes numeric operands relative to the current float batched dot paths.
