# Dense CPU forward stage profile 2026-11-24T23-13-13Z

Model: `C:\Users\Chuck\AppData\Local\Temp\kilo\optimize-small-cpu\models\qwen2.5-3b-instruct-q4_k_m.gguf`

Commands:
```text
prefill-profile --threads 32 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 3 --stage-profile
kv-quant --threads 32 --context 256 --decode 32 --windows 2 --modes baseline --stage-profile
```

These are diagnostic stage-profile runs. The enabled stage-profile path adds clock spans around dense forward substages, so raw throughput in this report is lower than the unprofiled tuning sweep. The stage ratios are still the useful result.

## Prefill stage ratios
```text
gemv length128: total_ms=20442.234, feed_forward_weight_ms=14323.066, attention_weight_ms=3327.713, attention_kernel_ms=1946.241, logits_head_ms=58.759
gemv length256: total_ms=45574.336, feed_forward_weight_ms=29074.244, attention_weight_ms=6836.688, attention_kernel_ms=8013.103, logits_head_ms=45.089

tiled length128: total_ms=19562.276, feed_forward_weight_ms=13090.476, attention_weight_ms=3037.431, attention_kernel_ms=2610.849, logits_head_ms=34.825
tiled length256: total_ms=44766.371, feed_forward_weight_ms=26399.487, attention_weight_ms=6072.431, attention_kernel_ms=10649.852, logits_head_ms=48.833
```

## Decode stage ratios
```text
decode window 1: total_ms=6169.627, feed_forward_weight_ms=3259.475, attention_weight_ms=1613.982, attention_kernel_ms=813.480, logits_head_ms=404.515
decode window 2: total_ms=6245.701, feed_forward_weight_ms=3245.660, attention_weight_ms=1617.511, attention_kernel_ms=896.783, logits_head_ms=408.799
```

## Interpretation
Dense CPU feed-forward weight traversal is the dominant substage for both prefill and decode on Qwen2.5-3B:
- Prefill length128: feed-forward weight traversal is about 67-70% of the profiled total.
- Prefill length256: feed-forward weight traversal is about 59-64% of the profiled total.
- Decode windows: feed-forward weight traversal is about 52-53% of the profiled total.

The next optimization target is therefore dense feed-forward GEMV/dequant staging, especially gate/up/down traversal for qwen2/llama dense weight matrices, before trying broader attention or KV/cache changes.
