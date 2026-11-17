# KV format abstraction spike — 2026-11-17

Decision recorded by this measurement: **keep the static-abstract generic `LayerOps<TK,TV>` seam as primary** for MVP5.1 attention work. The hand-specialized fallback in DN-M51-6 is not selected because the measured generic FP32 control was faster than today's hand-written FP32 attention path across micro and short-context decode runs on pinned fixtures.

## Harness

The spike uses three modes through the engine's real single-token decode route:

- `baseline`: unchanged production `LayerOps.CausalAttention` over `ReadOnlySpan<float>` KV reads.
- `f32block`: generic static-abstract control with FP32 block structs and bit-identical attention math. This is the abstraction-tax measurement for DN-M51-6.
- `q8k`: candidate quantized-dot shape using Q8_K K-side integer dot plus signed-Q8 V accumulation through existing CPU activation-quant machinery.

Important caveat: `f32block` and `q8k` keep production FP32 KV writes and add a mirror storage arena for the candidate read path, so these numbers measure attention/kernel seam cost. They do not yet prove final storage replacement footprint or append-time savings.

## Commands

```powershell
dotnet build Lair.sln -c Release --no-restore
.\bench\Lair.Bench\bin\Release\net10.0\Lair.Bench.exe kv-micro --threads 8 --context 256
.\bench\Lair.Bench\bin\Release\net10.0\Lair.Bench.exe kv-spike --threads 8 --context 2048 --decode 64 --windows 5 --modes baseline,f32block,q8k
.\bench\Lair.Bench\bin\Release\net10.0\Lair.Bench.exe kv-spike "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\Llama-3.2-1B-Instruct-IQ4_XS.gguf" --threads 8 --context 2048 --decode 64 --windows 5 --modes baseline,f32block,q8k
```

## Micro result — Q4_K geometry, threads=8, positions=256

Resolved CPU path: `isaResolved=vnni avx2Float=True`.

| Mode | Median per call | Range | Delta vs hand |
|---|---:|---:|---:|
| `hand` FP32 kernel | 443.986 us/call | 437.60–451.39 us across five samples after warm-up | baseline |
| `f32block` generic control | 429.359 us/call | 335.99–439.25 us | -3.3% |
| `q8k` candidate quantized dot | 270.455 us/call | 256.02–296.86 us | -38.0% |

Attention equivalence for the generic control was exact: `maxAbsDiff=0`, `meanRelDiff=0`. The candidate quantized shape reported micro attention error `maxAbsDiff=7.091e-4`, `meanRelDiff=2.872e-2`; this is not a parity-band result and does not validate final KV mode quality.

## Decode result — Q4_K_M pinned fixture, threads=8, context=2048

| Mode | Median tok/s | Window range | Delta vs baseline median |
|---|---:|---:|---:|
| `baseline` FP32 production attention | 7.133 | 6.710–7.428 | baseline |
| `f32block` generic control | 8.229 | 7.979–8.552 | +15.4% |
| `q8k` candidate quantized dot | 12.319 | 12.129–12.642 | +72.7% |

Probe continuity: all modes kept the same forced continuation argmax (`probeTopId=791/128256`). `f32block` logit drift vs baseline was `0.0000`; `q8k` probe drift was `0.3664`, reported as candidate-shape observation only.

## Decode result — IQ4_XS pinned fixture, threads=8, context=2048

| Mode | Median tok/s | Window range | Delta vs baseline median |
|---|---:|---:|---:|
| `baseline` FP32 production attention | 4.779 | 4.658–5.052 | baseline |
| `f32block` generic control | 8.670 | 8.539–9.019 | +81.4% |
| `q8k` candidate quantized dot | 13.613 | 13.210–13.891 | +184.8% |

Probe continuity: all modes kept the same forced continuation argmax (`probeTopId=791/128256`). `f32block` logit drift vs baseline was `0.0000`; `q8k` probe drift was `0.3947`, reported as candidate-shape observation only.

## Earlier noisy long-context runs

An initial subagent run used longer context (`ctx3840`) and showed unstable sign for the FP32 generic control because baseline windows had a 37% spread and one inverted outlier. Those short-window reruns above use `context=2048`, five timed windows per mode, sequential processes, no competing LAIR/bench work, and are used as the decision evidence.

## Decision consequence

DN-M51-6's primary path remains: cold descriptor + generic static-abstract block structs (`IKvFormat<TSelf>`, `BlockPool<TK,TV>`, `LayerOps<TK,TV>`). The fallback hand-specialized per-combo hot kernels are not adopted for the abstraction-tax reason. Later storage work must still prove final copy-less KV replacement behavior and quantized-mode parity bands; this spike does not decide q4-K viability or footprint savings.
