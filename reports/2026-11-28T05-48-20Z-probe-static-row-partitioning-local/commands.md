# Probe: static contiguous row partitioning for CPU worker dispatch — commands

Host posture: dev box (not the pinned Xeon), AVX2-only kernel selection, `threads=32` explicit quota (`workers=32`, oversubscribed to logical processors), unpinned thread placement. Model fixture for e2e arm: `models/qwen2.5-3b-instruct-q4_k_m.gguf` (Qwen2.5-3B Instruct, Q4_K_M). Kernel-level arms run on synthetic pinned tensors at the same feed-forward widths.

## Build and gates

```
dotnet build Lair.sln --nologo                                   # 0 warning(s), 0 error(s)
dotnet build Lair.sln -c Release --nologo                        # 0 warning(s), 0 error(s)
dotnet test Lair.sln -c Release --filter "FullyQualifiedName~StaticRowDispatchTests"   # Passed: 95
dotnet test Lair.sln -c Release --filter "Category=Kernel|CpuTuning&Category!=Cuda"    # Passed: 805
dotnet test Lair.sln -c Release --filter "Category=DenseWeightFusion|Kernel|CpuTuning|Profile&Category!=Cuda"   # Passed: 510
dotnet test Lair.sln -c Release --filter "Category=Parity"                             # Passed: 154 (1 m 32 s)
```

## Kernel-level A/B arms (logs/srp-runA-t32.log, logs/srp-runB-t32.log)

Arms share one run's settle passes and rotated timing rounds; `auto`/`streaming`/`tile`/`static-w8` are bit-parity-checked per count.

```
dotnet run --project bench/Lair.Bench -c Release -- q4k-widebatch --threads 32 \
  --shapes 11008x2048 --counts 8,32 --static-partition 8,16 --reps 3      # plumbing smoke, parity confirmed

dotnet run --project bench/Lair.Bench -c Release -- q4k-widebatch --threads 32 \
  --counts 8,16,32,64 --static-partition 8 --reps 7                        # run A (both shapes)
dotnet run --project bench/Lair.Bench -c Release -- q4k-widebatch --threads 32 \
  --counts 8,16,32,64 --static-partition 8 --reps 7                        # run B (repeat of run A)
```

Arm posture line published by each run: `static-w8_tuning=[static_row_partition_batch_width=8,row_block=16,parallel_row_threshold=256,workers=32]`.

## End-to-end wide-batch dense prefill arms (logs/srp-e2e-off.log, logs/srp-e2e-on.log)

```
dotnet run --project bench/Lair.Bench -c Release -- prefill-profile <model.gguf> \
  --threads 32 --lengths 1024,2048 --reps 2 --stage-profile --static-row-partition off
dotnet run --project bench/Lair.Bench -c Release -- prefill-profile <model.gguf> \
  --threads 32 --lengths 1024,2048 --reps 2 --stage-profile --static-row-partition on   # 'on' = width 8
```
