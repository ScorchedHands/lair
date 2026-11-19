# CUDA dense prefill payoff measurement — 2026-11-19

## Command and posture

```powershell
dotnet run --project bench/Lair.Bench -c Release -- cuda-prefill --lengths 32,64,128 --reps 3
```

Pinned fixture:

```text
file=models/SmolLM-135M-Instruct-F16.gguf bytes=270885792 lengths=32,64,128 reps=3 cpuThreads=auto
arch=llama layers=30 hidden=576 heads=9 kvHeads=3 headDim=64 vocab=49152 resolvedPrefillRowWidth=32
arms: cpu(weightPass=gemv, attention=host per-row over pool blocks) + cuda(device step composed per layer, kvStore=pool device store resident on the device)
basis: both arms forward identical token windows through TransformerModel.Prefill; wall time brackets each whole prefill (one untimed warm-up rep first); CUDA tallies are cumulative deltas across that arm's timed reps and count compute crossings only — pool residency moves print apart from them
```

## Median speed summary

`wallRatioCpuOverGpu > 1` means the CPU median wall time was larger than the CUDA median wall time, so CUDA prefill won on that length. Values below `1` mean CUDA lost on that measured run.

| Length | CPU median ms (min/max) | CPU tok/s | CUDA median ms (min/max) | CUDA tok/s | Ratio CPU/CUDA | GPU wins | Max abs logit drift | Argmax match |
|---:|---:|---:|---:|---:|---:|:--:|---:|:--:|
| 32 | 143.51 (137.97/157.05) | 223.0 | 120.11 (118.19/124.19) | 266.4 | 1.195 | yes | 1.512E-003 | cpu=29 cuda=29 |
| 64 | 232.43 (230.99/239.21) | 275.3 | 240.61 (239.45/247.63) | 266.0 | 0.966 | no | 2.019E-003 | cpu=65 cuda=65 |
| 128 | 554.38 (544.61/595.84) | 230.9 | 492.37 (490.48/498.83) | 260.0 | 1.126 | yes | 1.945E-003 | cpu=8547 cuda=8547 |

## CUDA tallies and residency facts

| Length | Kernel busy us | Runtime transfer calls | Runtime transfer bytes | Avg bytes/call | Device-written blocks | Crossed blocks/bytes | Resident device KV bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 442509 | 2710 | 233237760 | 86066 | 2 | 16 / 20643840 | 11796480 |
| 64 | 859483 | 5410 | 465480960 | 86041 | 4 | 16 / 32440320 | 11796480 |
| 128 | 1903216 | 10810 | 929967360 | 86028 | 8 | 16 / 56033280 | 11796480 |

CUDA facts for the full process run:

```text
weightCopiesToDevice=211 persistentDeviceWeightBytes=268959744 peakWorkingSetMb=576.0 across this process run
```

## Interpretation

- The payoff is mixed rather than universal on this fixture and host. CUDA composed dense prefill won at lengths `32` (`1.195`) and `128` (`1.126`) in this measured run, but lost at length `64` (`0.966`). CPU medians can vary with threading noise across runs, so the honest gate is "length-dependent payoff", not a blanket fast-prefill claim.
- The device path composes CUDA matmul with on-device attention/KV steps for dense prompt windows and keeps KV resident in the pool's device store during measured prefill. Scores, softmax state, and causal KV reads do not cross as per-layer attention staging; q/k/v rows are still staged into the step once per window/layer as part of this first payoff seam.
- Logit drift values above are bench basis only: they do not assert or widen parity bands in tests. The composition-only test band recorded for MVP5.4 is `4e-3`, with measured worst absolute logit drift `2.019E-3` on these pinned cases and argmax membership matching the CPU arm across all three lengths.
- GPU prefill remains internal/non-default. A public default flip would require a stable payoff posture across serving shapes, not one mixed length-dependent bench result.
