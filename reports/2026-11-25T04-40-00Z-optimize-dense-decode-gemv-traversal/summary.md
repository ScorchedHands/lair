# Dense single-token decode GEMV traversal optimization

> Superseded by `reports/optimize-cpu-tuning-decode-gemm-2026-11-25T06-56-44Z/` for the committed implementation. This candidate report records the pre-tuning measurements that motivated the detection-driven/configurable CPU tuning rule.

Branch `phase/mvp7`, worktree `.agent-worktrees/mvp7`, on top of `01dfa65 perf(cpu): speed dense batched traversal`.
Model: `qwen2.5-3b-instruct-q4_k_m.gguf` (qwen2, 36 layers, hidden 2048, heads 16, kvHeads 2, headDim 128, vocab 151936, intermediate 11008).
Local host, threads 32. No CUDA category was run; no remote hosts touched.

## What changed in the single-vector dense decode path

1. **Row prefetch hints for every dense format.** `PrefetchRowWindow` previously ran only on Q8_K i-quant row loops; it now runs on every non-threaded and threaded dense row loop (single-vector `Gemv`, batched `GemvBatch`, pair jobs) as a pure hint over the call's own row region. No output semantics change: prefetch touches nothing that affects arithmetic.
2. **Row-block partitioning tuned from 64 to 16 rows.** A full real-mmap single-token weight-pass probe on this model showed small attention matrices paying worker-dispatch cost at coarse blocks: `RowBlock=64` 167.4 ms/token, `32` → 149.4, `16` → 133.3 (threads 32); `RowBlock=8` regressed to 178.6. `ParallelRowThreshold` was briefly raised 256→512 and regressed the probe (169.6 at RowBlock 16), so it stays 256.
3. **Single-vector shared-input dense fusion (`IDenseWeightSharedInputFusion`).** New contract in `src/Lair.Contracts/IDenseWeightSharedInputFusion.cs` (two- and three-weight overloads) with CPU implementation `src/Lair.Backends.Cpu/CpuBackend+SharedInputGemv.cs`. The decode forward in `TransformerModel.ForwardToken` now walks attention Q/K/V in one dispatch over the one normed vector and FF gate/up in one dispatch; backends that do not implement the seam or decline a posture fall back to the independent `Gemv` calls. Each fused row runs the identical `RowDot`/`RowDotQ8K` evaluator with identical operands (including one shared Q8_K activation quantization for sub-4-bit i-quant matrices), so every element is bit-identical to the independent calls. This removes two of the five per-layer single-vector dispatches and keeps the activation vector hot across three/four matrix traversals.

## Decode before/after

Command: `dotnet run --project bench/Lair.Bench -c Release --no-build -- kv-quant <model> --threads 32 --context 256 --decode 64 --windows 4 --modes baseline`

| posture | medians (tok/s) | window spread |
|---|---|---|
| before (`lair-decode-before-1/2.txt`) | 5.127, 5.052 | 10.9% / 24.2% |
| after (`lair-decode-after-1/2/3.txt`) | 6.608, 6.384, 6.820 | 4.4% / 4.1% / 7.5% |

Median gain ≈ +27%, and the post-change runs are markedly more stable (the late-window degradation seen before is gone). Output identity: every 64-token run before and after reports `probeTopId=85047/151936 probeTopLogit=9.4356`, identical bit-for-bit.

## Stage profile before/after (32 tokens, `--stage-profile`)

| stage | before ms/token | after ms/token |
|---|---|---|
| attention_weight | ~48 | ~31 |
| attention_kernel | ~25 | ~24 |
| feed_forward_weight | ~97 | ~78 |
| logits_head | ~12 | ~12 |
| total | ~184 | ~147 |

## Prefill posture (no regression)

Command: `... -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 3`

| kernel | before batchedTok/s (128 / 256) | after batchedTok/s (128 / 256) |
|---|---|---|
| gemv | 49.43 / 46.39 (1.38 GB/s) | 59.90 / 46.03 (2.24 GB/s) |
| tiled | 53.12 / 41.59 (1.92 GB/s) | 52.92 / 44.91 (1.68 GB/s) |

gemv length-128 improved; everything else is within this machine's prefill noise band. The prior committed report's gemv 66.00 figure sits inside the same band; no posture regressed.

## llama.cpp generation baselines, same machine, same session

`llama-bench -m <model> -t 32 -p 0 -n 32 -r 5` → tg32 **18.86 ± 2.48** (`llama-bench-tg32.txt`)
`llama-bench -m <model> -t 32 -p 0 -n 64 -r 5` → tg64 **13.06 ± 3.79** (`llama-bench-tg64.txt`)

LAIR after this item: ~6.4–6.8 tok/s median. The remaining gap is the attention kernel stage (~24 ms/token), engine/overhead, and the Q6_K `ffn_down`/logits traversal (~15 GB/s vs ~22 GB/s for Q4_K), not the dispatch structure this item targeted.

## Verification

- `dotnet build -c Release --nologo` — 0 warnings, 0 errors.
- `dotnet test tests/Lair.Tests -c Release --no-build --filter "Category=Kernel|BatchKernel"` — 484 passed.
- `dotnet test ... --filter "Category=Prefill|Engine|DenseForwardProfile"` — 179 passed.
- `dotnet test ... --filter "FullyQualifiedName~SharedInputGemvTests"` — 14 passed (new: fused pair/triple bit-identical to independent `Gemv` for Q4_K/Q6_K/Q8_0/IQ3_S/F32/F16, block-aligned and ragged shapes, single-threaded and threaded, refusal on mismatched columns leaves outputs untouched).
- `dotnet test ... --filter "(Category=Prefill|Engine|Kv|Attention|Kernel|Serve)&Category!=Cuda"` — 682 passed, 0 failed.

## Limitations / notes

- Decode medians on this host are noisy (thermal/sustained-load); the after-posture spread (4–7%) is far tighter than the before-posture spread (11–24%), so the gain direction is robust even though individual window numbers wobble.
- The fused seam covers one shared input vector only; `attn_output` (different input vector) and `ffn_down` (activation input) stay independent calls. Fusing across two different activation vectors in one dispatch was not attempted.
- Q6_K unpack kernels (`ffn_down`, logits head) remain the slowest dense evaluators; a Q6_K group-pair unpack restructure is the obvious next traversal candidate but was not attempted in this item.
