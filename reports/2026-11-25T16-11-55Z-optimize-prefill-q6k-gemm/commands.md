# Commands used — optimize-prefill-q6k-gemm (2026-11-25T16-11-55Z host-clock UTC)

All commands run in `.agent-worktrees/mvp7` on branch `phase/mvp7`. Model: `C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf`. CPU-only lane; no CUDA benchmarks, no remote hosts.

## Build and tests

```powershell
dotnet build -c Release
# -> succeeded, 0 warnings, 0 errors

dotnet test tests/Lair.Tests -c Release --no-build --filter "Category=Kernel|Category=Backend|Category=DenseWeightFusion|Category=CpuTuning|Category=Profile"
# -> 591 passed / 1 failed: the exact filter also matched CudaBlockQuantGemvTests (a real-GPU lane that fails on this host's contended GPU). CPU-only-lane equivalent below.

dotnet test tests/Lair.Tests -c Release --no-build --filter "(Category=Kernel|Category=Backend|Category=DenseWeightFusion|Category=CpuTuning|Category=Profile)&Category!=Cuda"
# -> 449 passed / 0 failed / 0 skipped (39 s)

dotnet test tests/Lair.Tests -c Release --no-build --filter "FullyQualifiedName~Q6KGemmRouteTests|FullyQualifiedName~GemmKernelTests|FullyQualifiedName~PrefillGemmCapabilityGateTests"
# -> 101 passed / 0 failed / 0 skipped (2 s)

dotnet test tests/Lair.Tests -c Release --no-build --filter "FullyQualifiedName~CudaPrefillGemmCapabilityGateTests"
# -> 37 passed / 0 failed / 0 skipped (host-side capability-gate lanes; no driver/device opened by these tests)

dotnet test tests/Lair.Tests -c Release --no-build --filter "Category!=Cuda"
# -> 2512 passed / 0 failed / 0 skipped (7 m 50 s)
```

## Throughput A/B (3 alternating rounds: default, off, on; raw outputs in `ab-throughput.txt`)

```powershell
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --threads 32 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 3
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --threads 32 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 3 --matmul-route off
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --threads 32 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 3 --matmul-route on
```

## Stage-profile A/B (3 alternating rounds: default, off; raw outputs in `ab-stage-profile.txt`)

```powershell
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --threads 32 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 1 --stage-profile
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\qwen2.5-3b-instruct-q4_k_m.gguf" --threads 32 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 1 --stage-profile --matmul-route off
```
