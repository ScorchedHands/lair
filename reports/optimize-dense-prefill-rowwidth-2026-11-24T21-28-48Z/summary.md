# Dense prefill row-width sweep 2026-11-24T21-28-48Z

Model: `C:\Users\Chuck\AppData\Local\Temp\kilo\optimize-small-cpu\models\qwen2.5-3b-instruct-q4_k_m.gguf`
Command posture: local CPU lane, `threads=8`, `--lengths 128,256 --widths 16,32,64 --kernel gemv,tiled,gemm --reps 3`.

The first attempt requested row width 128 and was refused before measurement because the engine's resolved prefill row ceiling is 64:
```text
prefill-profile: row width 128 exceeds the engine's resolved prefill row ceiling of 64
```

Valid sweep results from `prefill-rowwidth-sweep.txt`:
```text
gemv rowWidth=16 length128 = 14.91 tok/s, length256 = 14.25 tok/s
gemv rowWidth=32 length128 = 14.79 tok/s, length256 = 13.83 tok/s
gemv rowWidth=64 length128 = 16.28 tok/s, length256 = 15.73 tok/s

tiled rowWidth=16 length128 = 19.18 tok/s, length256 = 17.80 tok/s
tiled rowWidth=32 length128 = 15.41 tok/s, length256 = 14.16 tok/s
tiled rowWidth=64 length128 = 13.27 tok/s, length256 = 12.16 tok/s

kernel=gemm status=refused: qwen2 dense Q6_K weight matrices have no CPU blocked matmul path in this build
```

Best stable `threads=8` row-width configuration was `tiled rowWidth=16`, reaching `19.18 tok/s` at length 128 and `17.80 tok/s` at length 256. The cumulative peak working set across the process run was about `1981.8 MB`; per-kernel peaks should be measured in separate processes if needed.

Interpretation: row-width tuning improves LAIR short-window prefill versus the earlier default posture, but it remains far below llama.cpp prompt-processing throughput for this model. This supports deeper dense CPU GEMV/dequant traversal and stage profiling as the next optimization path rather than treating row width alone as sufficient.
