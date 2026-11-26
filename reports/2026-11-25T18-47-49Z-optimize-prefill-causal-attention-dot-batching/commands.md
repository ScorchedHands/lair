# Commands used — optimize-prefill-causal-attention-dot-batching (2026-11-25T18-47-49Z host-clock UTC)

All commands run in `.agent-worktrees/mvp7` on branch `phase/mvp7`. Model: `C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf`. CPU-only lane; no CUDA benchmarks, no remote hosts. Threads were left at the auto posture (resolved threads=32 in every run header).

## Build and tests (final kernel state: register-resident chain arms)

```powershell
dotnet build -c Release
# -> succeeded, 0 warnings, 0 errors

dotnet test tests/Lair.Tests -c Release --no-build --filter "Category=Attention|Kernel|CpuTuning&Category!=Cuda"
# -> 496 passed / 0 failed / 0 skipped (32 s)   [run right after the chain-arm rewrite]

dotnet test tests/Lair.Tests -c Release --no-build --filter "Category=Attention|Kernel|CpuTuning|Engine|Prefill&Category!=Cuda"
# -> 681 passed / 0 failed / 0 skipped (1 m 29 s)

dotnet test tests/Lair.Tests -c Release --no-build --filter "FullyQualifiedName~AttentionSharedDot|FullyQualifiedName~AttentionAccumulatorTests|FullyQualifiedName~GemmKernelTests|FullyQualifiedName~Q6KGemmRouteTests"
# -> 114 passed / 0 failed / 0 skipped (1 s)

dotnet test tests/Lair.Tests -c Release --no-build --filter "Category!=Cuda"
# -> 2534 passed / 0 failed / 0 skipped (5 m 48 s)
```

## Stage-profile A/B — shipped kernel, 3 alternating rounds per arm (`ab-stage-profile-v2-registerchains.txt`)

```powershell
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 1 --stage-profile
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 1 --stage-profile --attention-batch-dots off
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 1 --stage-profile --attention-batch-dots on
```

## Throughput A/B — shipped kernel, 3 alternating rounds per arm (`ab-throughput-v2-registerchains.txt`)

```powershell
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 3
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 3 --attention-batch-dots off
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 3 --attention-batch-dots on
```

## Superseded first-generation runs (kept for provenance)

`ab-stage-profile.txt` and `ab-throughput.txt` in this directory are the earlier scratch-chain accumulator version of the batch kernel, measured before the rewrite. That generation was measurably *slower* than individual dots (`attention_kernel_ms` medians: gemv128 off 119.3 vs on 134.3; gemv256 off 603.4 vs default 650.2/on 712.0; tiled128 off 158.6 vs on 184.8; tiled256 off 618.5 vs on 671.9) and was replaced by the register-resident arms shipped in `src/Lair.Backends.Cpu/CpuBackend+SharedInputDot.cs`. One further discarded probe (a stage-profile rerun that omitted `--widths 64` and therefore ran rowWidth=32) is not kept as a raw log.

Arms: `default` = posture-resolved auto chain group (8 on this AVX2 host, identical behavior to pinned `on`); `off` = every shared-K score dot stays an individual backend Dot; `on` = batched shared-input dots pinned at chain group 8. Spread between the two behaviorally-identical arms calibrates run noise (~±3-5% at medians here).
