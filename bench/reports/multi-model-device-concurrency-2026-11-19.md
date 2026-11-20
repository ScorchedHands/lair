# MVP6.1 CUDA co-scheduling payoff measurement — 2026-11-19

## Command and posture

```powershell
dotnet run --project bench/Lair.Bench -c Release --no-build -- cuda-concurrency --reps 15 --sweeps 3 --out "C:\Users\Chuck\AppData\Local\Temp\kilo\cuda-concurrency-v3-run1.json"
dotnet run --project bench/Lair.Bench -c Release --no-build -- cuda-concurrency --reps 15 --sweeps 3 --out "C:\Users\Chuck\AppData\Local\Temp\kilo\cuda-concurrency-v3-run2.json"
dotnet run --project bench/Lair.Bench -c Release --no-build -- cuda-concurrency --reps 15 --sweeps 3 --out "C:\Users\Chuck\AppData\Local\Temp\kilo\cuda-concurrency-v3-run3.json"
```

Pinned fixture:

```text
shape rows=2048 cols=2048 batch=32 repsPerArm=15 sweeps=3 pairs=3
device ordinal=0 name="NVIDIA GeForce RTX 4090" driverApiVersion=13030
host osDescription="Microsoft Windows 10.0.26200" logicalProcessors=32
fixture=deterministic pinned block patterns from cuda-gemv's builder; activation family Sin(0.11*c+7*v)*1.3
arms: serialized legacy pipeline + twoStreamSerialized control + overlapped two non-blocking CUstreams
basis: serialized arm = sync H2D -> launch on legacy NULL stream -> cuCtxSynchronize -> sync D2H, workload A then B; twoStreamSerialized control = same non-blocking lanes and async staging, but each lane is awaited before the next workload is submitted; overlapped arm = async H2D + launch + async D2H on each lane, both submitted before either is awaited
```

The bench harness hand-rolls launches against the module handles already loaded by `CudaBackend`; it does not add a production stream seam, does not change `CudaBackend` defaults, and does not alter serving/config posture. This is the measurement gate for whether MVP6.1 should build per-model CUDA streams.

## Three-run paired-ratio summary

`ratioSerializedOverOverlapped > 1` means today's serialized two-workload pipeline took longer than the overlapped two-stream arm. `ratioTwoStreamOverOverlapped` isolates concurrency from the cost of lane machinery/async staging. `busyConcurrencyRatio > 1` means the device did more kernel-busy work than the overlapped wall time had available, so the lanes ran together.

| Pair | Run | Serialized med ms | Two-stream med ms | Overlapped med ms | Serialized/Overlapped | Two-stream/Overlapped | Serialized/Two-stream | busySum/overlapped wall | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mixed_gemv_q4k__gemm_f16` | 1 | 10.9538 | 9.8325 | 8.2039 | 1.38 | 1.39 | 1.03 | 1.29 | inconclusive |
| `mixed_gemv_q4k__gemm_f16` | 2 | 17.6337 | 10.8647 | 8.9626 | 1.66 | 1.32 | 1.14 | 1.94 | positive |
| `mixed_gemv_q4k__gemm_f16` | 3 | 8.3977 | 7.8725 | 7.4592 | 1.28 | 1.09 | 1.06 | 1.09 | positive |
| `two_gemv_q8_0__q6_k` | 1 | 23.0051 | 25.5809 | 11.1589 | 1.57 | 2.05 | 1.00 | 2.03 | positive |
| `two_gemv_q8_0__q6_k` | 2 | 20.0827 | 17.9961 | 10.3500 | 1.94 | 2.13 | 0.99 | 1.90 | positive |
| `two_gemv_q8_0__q6_k` | 3 | 23.4904 | 21.8141 | 14.2374 | 2.00 | 1.98 | 1.06 | 1.56 | positive |
| `two_gemm_q8_0__bf16` | 1 | 13.9433 | 14.0805 | 2.7224 | 2.15 | 2.31 | 1.03 | 5.01 | positive |
| `two_gemm_q8_0__bf16` | 2 | 17.8322 | 14.3925 | 3.2674 | 2.17 | 1.92 | 1.14 | 5.30 | positive |
| `two_gemm_q8_0__bf16` | 3 | 3.7677 | 3.4174 | 2.0388 | 1.89 | 1.96 | 1.02 | 1.65 | positive |

Bit-for-bit serialized-vs-overlapped output equivalence was true for every pair in every run.

## Gate result

The measured payoff is not a no-win. Two-GEMV and two-GEMM pairs were positive in all three runs, with serialized/overlapped paired medians from `1.57` to `2.17`. The mixed GEMV/GEMM pair was positive in two of three runs and inconclusive once (`1.28` paired median in run 3), so the honest posture is: CUDA stream overlap pays off clearly for same-kernel-family pairs on this 4090, while mixed GEMV/GEMM payoff is less stable under host contention.

The two-stream control shows that the lane machinery and async staging alone are close to today's serialized pipeline (`serialized/two-stream` medians `0.99`–`1.14`). Essentially all measured payoff comes from submitting both lanes before awaiting either, not from the stream/async staging machinery by itself.

## Caveats

Absolute millisecond medians are contention-contaminated on this host: other GPU compute processes were active during the runs, and per-rep ratio spreads were large. The verdicts therefore rest on rep-by-rep paired ratios, not on the absolute wall medians. The raw JSON outputs are outside the repo at `C:\Users\Chuck\AppData\Local\Temp\kilo\cuda-concurrency-v3-run{1,2,3}.json`.

## Verification

```powershell
dotnet build -c Release --nologo
dotnet test tests/Lair.Tests -c Release --no-build --nologo --filter "Category=Cuda"
```

Result:

```text
Build succeeded.
    0 Warning(s)
    0 Error(s)

Passed!  - Failed: 0, Passed: 248, Skipped: 0, Total: 248, Duration: 6 s
```
