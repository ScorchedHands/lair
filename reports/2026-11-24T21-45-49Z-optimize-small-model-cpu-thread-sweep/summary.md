# Small CPU thread/kernel sweep 2026-11-24T21-45-49Z

Model: `C:\Users\Chuck\AppData\Local\Temp\kilo\optimize-small-cpu\models\qwen2.5-3b-instruct-q4_k_m.gguf`
llama.cpp tool root: `C:\Users\Chuck\AppData\Local\Temp\kilo\optimize-small-cpu\tools\llama-b12359-bin-win-cpu-x64`
Host lane: local Windows 11 machine with an Intel i9-13900K-class CPU; this sweep tested LAIR and llama.cpp thread scaling on the small Qwen2.5-3B model only, not CUDA.

## llama.cpp thread sweep (`llama-thread-sweep.txt`)
```text
threads=8:  pp128 = 185.31 ± 6.84 tok/s, tg32 = 21.08 ± 0.31 tok/s
threads=16: pp128 = 252.71 ± 9.36 tok/s, tg32 = 22.72 ± 0.15 tok/s
threads=32: pp128 = 279.94 ± 2.69 tok/s, tg32 = 19.27 ± 0.10 tok/s
```

## LAIR prefill sweep (`lair-prefill-t*.txt`)
The matrix used `prefill-profile --lengths 128` only for each thread posture and swept widths/kernels; single-length fits are marked unreliable by the bench because weight/attention costs cannot be separated from one length, but the measured `batchedTok/s` values remain useful.

Best observed short-window LAIR prefill throughput in the matrix:
```text
threads=8:  tiled rowWidth=16 = 16.59 tok/s at length 128; gemv rowWidth=32 = 15.29 tok/s
threads=16: gemv rowWidth=64 = 23.30 tok/s, tiled rowWidth=16 = 20.76 tok/s
threads=32: gemv rowWidth=64 = 29.07 tok/s peak in the single-length matrix; tiled rowWidth=32 = 26.43 tok/s
```

## Stable confirmation at best-looking thread/width (`prefill-best-config.txt`)
Command posture: `--threads 32 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 3`.
```text
gemv rowWidth=64 length128 = 25.31 tok/s; length256 = 22.76 tok/s
tiled rowWidth=64 length128 = 25.18 tok/s; length256 = 22.55 tok/s
weightFit ≈ 2244 ms/traversal, impliedWeightGB/s ≈ 0.74 for both kernels at this posture
```

## LAIR production KV decode baseline (`lair-kv-quant-baseline-t*.txt`)
Command posture: `kv-quant --context 256 --decode 32 --windows 2 --modes baseline`.
```text
threads=8:  prefillSetupMs = 19436.7, median decode = 5.947 tok/s, spread = 8.43%
threads=16: prefillSetupMs = 13429.7, median decode = 6.796 tok/s, spread = 0.06%
threads=32: prefillSetupMs = 10791.2, median decode = 7.029 tok/s, spread = 0.26%
```

## Interpretation
Thread scaling and row-width tuning provide a real LAIR improvement for the small dense model: short-window prefill improves from roughly `15-17 tok/s` at threads=8 to about `23-29 tok/s` in the sweep, with stable confirmation near `25 tok/s` at threads=32 / rowWidth=64. Production decode also improves modestly from `~5.9` to `~7.0 tok/s`.

This is not parity yet: llama.cpp reaches roughly `185-280 pp128 tok/s` and `19-23 tg32 tok/s` across the same thread postures, while LAIR remains around `25 prefill tok/s` and `7 decode tok/s`. The next concrete optimization path is not more thread tuning alone; it is dense CPU GEMV/dequant traversal/staging profiling and kernel-level optimization for qwen2/llama dense weight matrices.
