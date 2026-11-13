# Prefill cost profile - 2026-11-12

`Lair.Bench prefill-profile` forwards prompt windows through the engine's batched-prefill path over a fresh KV per
measurement, then fits measured cost as `weight = ms/traversal x ceil(L / 16)` plus
`attention = us/position-pair x L(L+1)/2` over every length at once. Raw tables for the runs quoted here are in
`prefill-llama-batched.txt` and `prefill-qwen3-batched.txt` beside this file.

## Commands

```
dotnet run --project bench/Lair.Bench -c Release -- prefill-profile models/Llama-3.2-1B-Instruct-IQ4_XS.gguf --threads 8 --reps 3
dotnet run --project bench/Lair.Bench -c Release -- prefill-profile models/Qwen3-0.6B-Q8_0.gguf --threads 8 --reps 2 --lengths 16,32,64,128,256,512,1024
```

## Fitted split

| fixture | weight ms/traversal | implied weight GB/s | attention us/position-pair | crossover |
|---|---|---|---|---|
| Llama-3.2-1B IQ4_XS (run 1) | 252.07 | 2.06 | 57.3 | 555 |
| Llama-3.2-1B IQ4_XS (run 2) | 344.79 | 1.51 | 52.4 | 827 |
| Qwen3-0.6B Q8_0 | 126.7 | 3.70 | 61.0 | 266 |

Llama run 2, the quieter of the two, in full:

```
      length    subBatches     batchedMs  batchedTok/s      weightMs   attentionMs  fitResidualMs    weightGB/s
          16             1         296.8         53.91         344.8           7.1         -55.1          1.51
          32             2         778.6         41.10         689.6          27.7          61.4          1.51
          64             4        1559.0         41.05        1379.1         109.0          70.8          1.51
         128             8        3279.4         39.03        2758.3         432.7          88.4          1.51
         256            16        7261.5         35.25        5516.6        1724.2          20.7          1.51
         512            32       17635.4         29.03       11033.2        6883.5        -281.3          1.51
        1024            64       49286.9         20.78       22066.4       27507.3        -286.8          1.51
        1536            96       95580.6         16.07       33099.5       61871.3         609.8          1.51
        2048           128      153853.7         13.31       44132.7      109975.5        -254.5          1.51
```

## Reading

- **The weight term is compute-bound, not bandwidth-bound.** One traversal streams 496 MB of packed attention/FFN
  bytes for Llama in 252-345 ms (1.5-2.1 GB/s) and 446 MB for Qwen3 in 127 ms (3.7 GB/s). A core pair on this machine
  sustains several times that against sequential reads, so the batched GEMV dequant arithmetic - not the memory
  system - is what the weight term pays for. Amortizing dequant across many rows is the lever, which is exactly what
  the pinned llama.cpp reference does at 412-464 tok/s on the same fixture.
- **Attention becomes the larger term far earlier than the serve-surface numbers suggested.** The crossover is 266
  tokens for Qwen3 (kvDim 1024 over 28 layers) and 555-827 for Llama (kvDim 1024 over 16 layers). At 2048 tokens on
  Llama, attention is 110 s against 44 s of weight reads - 2.5x. Windowed families push the crossover later by
  bounding the causal span; both fixtures here resolve `windowedLayers = 0`.
- **Batching already removed the weight term's growth with prompt length.** Per-token prefill on the same fixture
  measured 9.19 and 7.60 tok/s through the serve surface against 13.3-15.4 tok/s batched at 1751-2048 tokens, and the
  batched curve now degrades with length only because of the attention term.

## Measurement quality

The box carries other workloads during these runs, and the fit exposes it. Two Llama runs on identical commands
disagree on the weight term by 1.4x (252 vs 345 ms/traversal) while agreeing on the shape of the curve; a run whose
measured costs stop rising with the causal span prints
`UNRELIABLE: measured costs are not separable into the two terms on this run` instead of a fitted split, and such a
run is not quoted here. Crossover values are floors: the quadratic attention term is only valid up to the model's
resolved attention windows.
