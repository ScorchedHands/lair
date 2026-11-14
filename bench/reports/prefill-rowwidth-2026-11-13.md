# Prefill row-width profile - 2026-11-13

`Lair.Bench prefill-profile --widths` measures prompt windows through engines opened at candidate resolved row counts.
Each candidate opens its own engine, sizes its batched streams for that row count, and gets its own sub-batch table and
fitted split. Raw tables for every measured width are in `prefill-rowwidth-llama.txt` and `prefill-rowwidth-qwen3.txt`.

## Commands

```
dotnet run --project bench/Lair.Bench -c Release -- prefill-profile models/Qwen3-0.6B-Q8_0.gguf --threads 8 --reps 1 --lengths 512,1024,2048 --widths 16,32,64 --out bench/reports/prefill-rowwidth-qwen3.txt
dotnet run --project bench/Lair.Bench -c Release -- prefill-profile models/Llama-3.2-1B-Instruct-IQ4_XS.gguf --threads 8 --reps 1 --lengths 512,1024,2048 --widths 16,32,64 --out bench/reports/prefill-rowwidth-llama.txt
```

## Chosen resolved row count: `32`

The default remains `32`. It is the balance point across both fixtures rather than a single-width speed winner.

| fixture | window | 16 rows tok/s | 32 rows tok/s | 64 rows tok/s |
|---|---:|---:|---:|---:|
| Llama-3.2-1B IQ4_XS | 512 | 34.90 | 39.84 | 41.91 |
| Llama-3.2-1B IQ4_XS | 1024 | 28.62 | 30.23 | 31.43 |
| Llama-3.2-1B IQ4_XS | 2048 | 19.57 | 20.31 | 20.57 |
| Qwen3-0.6B Q8_0 | 512 | 55.93 | 55.16 | 55.30 |
| Qwen3-0.6B Q8_0 | 1024 | 34.93 | 34.56 | 33.84 |
| Qwen3-0.6B Q8_0 | 2048 | 18.44 | 18.52 | 18.37 |

Estimated activation bytes for one full sub-batch:

| fixture | 16 rows | 32 rows | 64 rows |
|---|---:|---:|---:|
| Llama-3.2-1B IQ4_XS | 1,769,472 | 3,538,944 | 7,077,888 |
| Qwen3-0.6B Q8_0 | 983,040 | 1,966,080 | 3,932,160 |

For the wider-hidden Llama fixture, `32` rows beats `16` at every measured window and captures most of the gain from
raising to `64`. For Qwen, `16`, `32`, and `64` are within measurement noise except that `64` is slightly worse at
`1024`; doubling activation bytes buys nothing useful there. The engine ceiling stays available for measured candidate
widths, but the default resolved width does not pay twice the stream memory for a small gain on one fixture only.

## Default-width tables from this run

Qwen3-0.6B Q8_0 at row count `32`:

```
      length    subBatches     batchedMs  batchedTok/s      weightMs   attentionMs  fitResidualMs    weightGB/s
         512            16        9282.6         55.16        2406.4        6316.3         559.9          3.11
        1024            32       29633.5         34.56        4812.8       25240.6        -419.9          3.11
        2048            64      110608.8         18.52        9625.6      100913.2          70.0          3.11
```

Llama-3.2-1B IQ4_XS at row count `32`:

```
      length    subBatches     batchedMs  batchedTok/s      weightMs   attentionMs  fitResidualMs    weightGB/s
         512            16       12850.2         39.84        8675.0        4140.7          34.5          0.96
        1024            32       33870.7         30.23       17350.0       16546.6         -25.9          0.96
        2048            64      100858.4         20.31       34699.9       66154.2           4.3          0.96
```

## Reading

- Row width changes sub-batch count and activation allocation; it does not change the attention term materially. At
  `2048`, fitted attention still dominates both fixtures: about `100.9 s` against `9.6 s` of fitted weight cost for Qwen,
  and about `66.2 s` against `34.7 s` for Llama on this run.
- The fitted crossovers at row count `32` are floors: length `>= 209` for Qwen and length `>= 1081` for Llama under the
  quadratic attention fit used by the bench verb.

## Measurement quality

These runs use `--reps 1`, warm plus best-of-one, because each candidate width reopens a real model engine across three
long windows. Short-window Qwen differences should be read as noise; the decision is driven by the repeated shape of the
Llama curve and by activation bytes rather than by one-tenths tok/s on Qwen short windows.
