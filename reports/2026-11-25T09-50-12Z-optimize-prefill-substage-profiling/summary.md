# Dense CPU prefill substage profiling 2026-11-25T09-50-12Z

Model: `C:\Users\Chuck\AppData\Local\Temp\kilo\optimize-small-cpu\models\qwen2.5-3b-instruct-q4_k_m.gguf`
Lane: local CPU lane, `threads=32`, `rowWidth=64`, Qwen2.5-3B Q4_K_M, no CUDA benchmarking.

This run splits the existing dense stage profile into substages: `attention_projection_ms` (batched Q/K/V weight passes) vs `attention_output_ms` (output projection), `feed_forward_gate_up_ms` vs `feed_forward_down_ms`, plus a named `sparse_chain_ms` for routed blocks. `attention_weight_ms` and `feed_forward_weight_ms` stay as the parent traversal sums. No kernel arithmetic changed: profiled logits/KV bytes remain bit-identical to unprofiled runs (proven in `DenseForwardProfileTests` and `MoeTests`).

Raw outputs: `prefill-stage-profile-gemv.txt`, `prefill-stage-profile-tiled.txt`, `kv-decode-stage-profile.txt`.

## Commands

```text
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel gemv --reps 1 --stage-profile
dotnet run --project bench/Lair.Bench -c Release --no-build -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel tiled --reps 1 --stage-profile
dotnet run --project bench/Lair.Bench -c Release --no-build -- kv-quant <model> --threads 32 --context 256 --decode 32 --windows 1 --modes baseline --stage-profile
```

## Prefill substage split (ms, share of profiled total)

```text
gemv  length128  total=2223.8  tok/s=57.56
  feed_forward_gate_up_ms=679.8 (30.6%)  feed_forward_down_ms=469.1 (21.1%)  [feed_forward_weight_ms=1148.9, 51.7%]
  attention_kernel_ms=543.5 (24.4%)      attention_projection_ms=165.6 (7.4%)  attention_output_ms=101.4 (4.6%)  [attention_weight_ms=267.0, 12.0%]
  logits_head_ms=11.9 (0.5%)             sparse_chain_ms=0.0

gemv  length256  total=6273.1  tok/s=40.81
  attention_kernel_ms=2616.4 (41.7%)
  feed_forward_gate_up_ms=1508.5 (24.0%)  feed_forward_down_ms=1058.1 (16.9%)  [feed_forward_weight_ms=2566.6, 40.9%]
  attention_projection_ms=364.1 (5.8%)    attention_output_ms=217.3 (3.5%)     [attention_weight_ms=581.4, 9.3%]
  logits_head_ms=13.2 (0.2%)              sparse_chain_ms=0.0

tiled length128  total=2652.7  tok/s=48.25
  attention_kernel_ms=773.8 (29.2%)      feed_forward_gate_up_ms=769.3 (29.0%)  feed_forward_down_ms=539.0 (20.3%)  [feed_forward_weight_ms=1308.3, 49.3%]
  attention_projection_ms=185.4 (7.0%)   attention_output_ms=116.9 (4.4%)       [attention_weight_ms=302.3, 11.4%]
  logits_head_ms=19.7 (0.7%)             sparse_chain_ms=0.0

tiled length256  total=5284.4  tok/s=48.44
  attention_kernel_ms=2092.0 (39.6%)     feed_forward_gate_up_ms=1323.1 (25.0%)  feed_forward_down_ms=900.2 (17.0%)  [feed_forward_weight_ms=2223.4, 42.1%]
  attention_projection_ms=318.1 (6.0%)   attention_output_ms=197.0 (3.7%)        [attention_weight_ms=515.1, 9.7%]
  logits_head_ms=9.6 (0.2%)              sparse_chain_ms=0.0
```

## Decode substage split (kv-quant baseline, FP32 KV, 32 single-token steps at context 256)

```text
prefill_setup tokens=256 total=6941.8
  feed_forward_gate_up_ms=1824.8 (26.3%)  feed_forward_down_ms=1046.2 (15.1%)  [feed_forward_weight_ms=2871.0, 41.4%]
  attention_kernel_ms=2621.6 (37.8%)      attention_projection_ms=545.4 (7.9%)  attention_output_ms=303.8 (4.4%)  [attention_weight_ms=849.2, 12.2%]
  logits_head_ms=69.1 (1.0%)              sparse_chain_ms=0.0

decode window=1 tokens=32 forwardCalls=32 total=5232.8 tok/s=6.115
  feed_forward_gate_up_ms=1558.4 (29.8%)  feed_forward_down_ms=1347.8 (25.8%)  [feed_forward_weight_ms=2906.1, 55.5%]
  attention_projection_ms=525.8 (10.0%)   attention_output_ms=516.5 (9.9%)      [attention_weight_ms=1042.3, 19.9%]
  attention_kernel_ms=794.6 (15.2%)       logits_head_ms=418.9 (8.0%)           sparse_chain_ms=0.0
```

## llama.cpp reference (same host, from `reports/optimize-dense-cpu-traversal-2026-11-25T01-23-00Z`)

```text
rerun-llama-bench-t32.txt: pp128 = 265.35 ± 6.62 tok/s, pp256 = 164.06 ± 112.40 tok/s, tg32 = 13.79 ± 1.14 tok/s
older baseline:            pp128 = 188 tok/s, pp256 = 195 tok/s
```

LAIR at this posture (profiled): gemv `57.56`/`40.81` tok/s, tiled `48.25`/`48.44` tok/s — about 4.6x below llama.cpp `pp128` and 3.4-4x below the noisy `pp256` band. The prior unprofiled sweep in the traversal report recorded gemv `66.00`/`47.15` and tiled `48.30`/`41.96` tok/s, so the stage-profile seam is not measurably distorting throughput.

## Interpretation

- The dominant prefill substage is the **feed-forward gate/up traversal** at length 128 on both kernels (29-31% of the window), and it stays the dominant *weight traversal* at length 256 (24-25%).
- At length 256 the **attention causal kernel** (KV read + softmax + V accumulation) is the single largest substage overall: 41.7% on gemv and 39.6% on tiled, slightly ahead of gate/up. The tiled kernel is what cuts that cost (2092 vs 2616 ms), while paying more at length 128 (774 vs 544 ms).
- **Attention output projection is the cheapest traversal everywhere** (3.5-4.6% of the window); the batched Q/K/V projection runs about 1.6-1.7x the output projection, as expected from its larger row count (16 heads vs 2 kv heads for q, 2 for k/v) — attention weight traversal is never the prefill bottleneck.
- gate/up vs down lands at about 1.4-1.5x even though gate and up are two matrices against down's one, so the batched gate/up pair (`RunDenseFeedForwardPair`) is not getting the per-matrix efficiency the down pass gets — that gap is the traversal target.
- Decode is the same shape with a bigger FF share (gate_up 29.8% + down 25.8% = 55.5%) and a large output-head cost (8.0%) that prefill barely pays.

Next optimization order for the follow-up item: (1) batched dense feed-forward gate/up traversal — the pair pass is the largest weight substage at every length; (2) the causal attention kernel's KV-read/softmax/V-accumulation span, which dominates at 256 tokens and is the gap the tiled route only partly closes; (3) attention output projection and Q/K/V projection last — together they are under 10% of the window.

## Validation

```text
dotnet build Lair.sln -c Release --nologo -> 0 warnings, 0 errors
dotnet test tests/Lair.Tests -c Release --no-build --filter "FullyQualifiedName~DenseForwardProfileTests|FullyQualifiedName~MoeTests" -> 30/30 passed
dotnet test tests/Lair.Tests -c Release --no-build --filter "(Category=Prefill|Engine|Kv|Attention|Kernel|Serve|Config|HotReload)&Category!=Cuda" -> 891/891 passed, 0 failed, 0 skipped (5m56s)
```
