# MVP5.1 quantized KV recommendation bench - 2026-11-18

## Question

MVP5.1's Manual Quant Recommendation asks whether the documented KV default should change now that CPU `q8_0/q8_0` and `k8v4` (`q8_0/q4_0`) quantized KV routes are measurable in production serving.

## Command

```powershell
dotnet run --project bench/Lair.Bench -c Release --no-build -- kv-quant "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\SmolLM-135M-Instruct-F16.gguf" --threads 8 --context 1024 --decode 32 --windows 5
```

The bench uses the production CPU serving seams: `TransformerModel.Open(...)` with a resolved `KvFormatPair` and `PrefillKernel.GemV`, typed `BlockPool<Q8Block,Q8Block>` / `BlockPool<Q8Block,Q4Block>`, `SequenceKv<TK,TV>` / `IKvSequence`, and the closed quantized attention route. The baseline mode uses the untouched float `f32/f32` route.

## Environment

- Model: `SmolLM-135M-Instruct-F16.gguf`
- Architecture: `llama`
- Layers: `30`
- Hidden: `576`
- Heads: `9`
- KV heads: `3`
- Head dim: `64`
- Vocab: `49152`
- Threads: `8`
- ISA: `vnni`
- Context: `1024`
- Decode tokens: `32`
- Windows: `5`
- Process peak working set: `381.1 MB`

## Measured decode throughput

| Mode | Median tok/s | Min tok/s | Max tok/s | Spread | Prefill setup ms | Reserved KV |
|---|---:|---:|---:|---:|---:|---:|
| `f32/f32` baseline | `23.659` | `20.949` | `27.111` | `26.05%` | `14410.1` | `56.25 MiB` |
| `q8_0/q8_0` | `8.200` | `7.839` | `8.503` | `8.10%` | `51903.1` | `14.94 MiB` |
| `q8_0/q4_0` (`k8v4`) | `8.205` | `7.961` | `8.564` | `7.35%` | `49744.1` | `11.43 MiB` |

Per-window decode values:

| Mode | Window tok/s |
|---|---|
| `f32/f32` | `22.255`, `23.659`, `26.190`, `27.111`, `20.949` |
| `q8_0/q8_0` | `8.503`, `8.426`, `8.200`, `7.955`, `7.839` |
| `q8_0/q4_0` | `8.564`, `8.343`, `8.204`, `8.205`, `7.961` |

## Footprint

| Mode | Bytes per block | Reserved KV |
|---|---:|---:|
| `f32/f32` | `737280` | `56.25 MiB` |
| `q8_0/q8_0` | `195840` | `14.94 MiB` |
| `q8_0/q4_0` | `149760` | `11.43 MiB` |

## Equivalence sanity

The bench compares the final forced-token distribution from each mode against the same run's FP32 baseline probe. This is a production-route drift sanity check, not the pinned llama.cpp parity gate.

| Mode | Probe top id | Argmax match vs baseline | Max abs logit diff vs baseline |
|---|---:|---:|---:|
| `q8_0/q8_0` | `504/49152` | true | `0.1299` |
| `q8_0/q4_0` | `504/49152` | true | `1.5525` |

Pinned cross-tool parity bands remain the authoritative quality gate:

- `q8_0/q8_0`: max relevant delta `0.1347` nats, band `0.25` nats.
- `q8_0/q4_0`: max relevant delta `0.3856` nats, band `0.50` nats.

## Recommendation

Keep `f32/f32` as the documented/default KV posture.

The measured quantized routes save KV footprint, but they are not a better default on this geometry:

- Decode is materially slower than the FP32 baseline (`8.200` and `8.205` median tok/s vs `23.659` median tok/s).
- Prefill setup is materially slower than FP32 (`51903.1` and `49744.1` ms vs `14410.1` ms).
- `q8_0/q8_0` has the smaller measured production drift (`0.1299` max abs logit diff vs baseline) and a tighter pinned band (`0.25` nats), but still costs speed.
- `k8v4` has the smaller footprint (`11.43 MiB` reserved KV) but the larger production drift (`1.5525` max abs logit diff vs baseline) and a looser pinned band (`0.50` nats).

Practical posture:

- Default: `f32/f32`.
- Conservative memory-saving quantized option: `q8_0/q8_0`.
- Memory-pressure option only: `k8v4`, when the smaller KV footprint justifies the larger measured drift.

The default does not move to a quantized mode because the recommendation is parity/quality and production posture first, not smallest footprint.

## Caveats

- This machine was carrying other workloads, so baseline spread is high (`26.05%`). Quantized medians are still clearly below baseline.
- The bench measures a fixed forced-token continuation; it is not a full generation-quality suite.
- The production max-abs-logit diff is not the same as pinned llama.cpp cache-type parity drift; the pinned parity report remains the quality gate.
