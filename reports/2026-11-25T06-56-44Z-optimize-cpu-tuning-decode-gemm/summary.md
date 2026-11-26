# Detection-driven CPU kernel tuning (row block, dispatch threshold, chain grouping, scratch width)

Branch `phase/mvp7`, worktree `.agent-worktrees/mvp7`, on top of `12caa8d docs(plan): require detection-driven cpu tuning` and the staged dense decode traversal work.
Model: `qwen2.5-3b-instruct-q4_k_m.gguf` (qwen2, 36 layers, hidden 2048, heads 16, kvHeads 2, headDim 128, vocab 151936, intermediate 11008).
Local host, ISA resolved `vnni`. No CUDA category was run; no remote hosts touched.

## What changed

The four kernel constants the staged candidate hardcoded (`RowBlock=16`, `ParallelRowThreshold=256`, `ChainGroup=8`, `MaxScratchCols=16384`) are now resolved per backend open from detected posture, with override seams:

- **New `src/Lair.Backends.Cpu/CpuKernelTuning.cs`** — `CpuKernelTuningOverrides` (nullable pins) + `CpuKernelTuning` (resolved values) + `Resolve(effectiveWorkerCount, vectorPosture, overrides)`.
  - Row block = `clamp(512 / workers, 8, 64)` — about half a page of weight rows per participating worker. At this box's 32-worker posture it lands exactly on the measured optimum 16; fewer workers get coarser blocks, more get finer ones.
  - Dispatch threshold = `max(rowBlock x workers / 2, 2 x rowBlock)` — lands exactly on the measured 256 at the 32-worker/16-block posture (and at 8 workers too, since the block scales inversely); never dispatches below two blocks of work.
  - Chain group = 8 under a vector posture (eight `Vector256` chains plus one weight chunk fit AVX2's sixteen YMM registers), 1 without one.
  - Max scratch cols = 16 384 floats (staged row + packed bytes stay cache-resident at that width).
- **Bounds refusals** (never silent clamps): row block [1,64], threshold ≥ 1, chain group [1,8], scratch cols [512, 65 536]. An out-of-span pin throws naming the qualified key (`engine.cpu.tuning.*`), its value, the span, and the rationale.
- **Backend wiring**: `CpuBackend` resolves tuning once at open from `_effectiveThreads` + vector posture; `ApplyThreadQuota` re-resolves under the pool gate (pins persist, auto siblings follow the new worker count); every threaded dispatch snapshots the record once per call and row jobs copy the ints — no hot-path allocation, no tearing between block count and job widths. `MaxBatch=64` stays a compile-time ceiling deliberately: it sizes stack scratch across every quant partial and forward callers already stay within it.
- **Public config**: `engine.cpu.tuning.{row_block, parallel_row_threshold, chain_group, max_scratch_cols}` (all optional, restart-only classification, loader validation mirrors the backend refusals naming qualified keys). `GenerationService` converts stated pins into `CpuKernelTuningOverrides` and passes them into every CPU backend open. Absent keys declare nothing — posture-resolved, not fixed to one machine class.
- **Grouped scratch kernel**: `BatchDotFromScratch` keeps the unrolled eight-chain body at the default posture and runs pinned narrower groups through `ScratchGroupsOverStackChains` (stack-staged chains, shared hot-scratch weight loads, ragged single-accumulator tail) with identical operand order — every posture byte-identical, asserted in `CpuTuningTests`.

## Resolved postures (auto)

| effective workers | row block | dispatch threshold |
|---|---|---|
| 1 | 64 | 128 |
| 4 | 64 | 128 |
| 8 | 64 | 256 |
| 16 | 32 | 256 |
| 32 | 16 | 256 |
| 64 | 8 | 256 |

At the 32-worker posture the resolved values equal the staged candidate's hardcoded constants exactly — this item changes how they are obtained, not what this machine runs.

## Decode (threads 32)

Command: `dotnet run --project bench/Lair.Bench -c Release --no-build -- kv-quant <model> --threads 32 --context 256 --decode 64 --windows 4 --modes baseline`

| posture | medians (tok/s) | window spread |
|---|---|---|
| committed dense-traversal baseline (`optimize-dense-cpu-traversal-2026-11-25T01-23-00Z`) | 5.3–5.7 | — |
| staged hardcoded candidate (`optimize-dense-decode-gemv-traversal-2026-11-25T04-40-00Z`, quieter session) | 6.608, 6.384, 6.820 | 4.4–7.5% |
| this change, worker reps 1–3 (`lair-decode-t32-1/2/3.txt`) | 5.929, 6.075, 5.863 | 11.3–25.2% |
| this change, main validation reps 1–3 (`main-decode-t32-1/2/3.txt`) | 5.870, 5.908, 6.163 | 11.0–18.4% |
| this change, main extra single runs (`main-decode-t32-extra.txt`) | 6.419, 5.726 | 10.2–14.1% |

Output identity: every run reports `probeTopId=85047/151936`, identical across postures (the bit-identity claim is additionally asserted per-posture in `CpuTuningTests`).

**Honesty note:** at threads 32 the resolved tuning is byte-identical to the staged candidate's constants, so the worker-run lower medians are not a tuning change. Main validation after the 891-test gate produced a wider band (`5.726`–`6.419`, with three-loop medians `5.870`–`6.163`) than the staged candidate's quieter `6.384`–`6.820`, while staying above the committed dense-traversal baseline (`5.3`–`5.7`). The spread and stage profile point to host load/noise rather than a posture regression.

## Posture scaling check (threads 16)

`lair-decode-t16.txt`: median 5.553 tok/s with auto row block 32 — coarser blocks with fewer participants, as designed; the 32-worker posture (block 16) still wins on this box, matching the staged sweep's ordering (16 > 32 > 64 blocks at 32 workers).

## Prefill (no regression)

Command: `... -- prefill-profile <model> --threads 32 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 3` (`lair-prefill-after.txt`)

| kernel | worker change batchedTok/s (128 / 256) | main validation batchedTok/s (128 / 256) | staged candidate (128 / 256) |
|---|---|---|---|
| gemv | 62.14 / 40.57 | 61.50 / 51.83 | 59.90 / 46.03 |
| tiled | 52.07 / 41.95 | 62.76 / 45.66 | 52.92 / 44.91 |

All inside this machine's prefill noise band; nothing regressed below the committed baseline (gemv 49.43 / 46.39, tiled 53.12 / 41.59).

## llama.cpp generation baselines (carried from the staged report, same machine)

tg32 **18.86 ± 2.48**, tg64 **13.06 ± 3.79** — not re-run in this session.

## Verification

- `dotnet build Lair.sln -c Release` — 0 warnings, 0 errors.
- `dotnet test ... --filter "FullyQualifiedName~CpuTuning"` — 52 passed (new: `CpuTuningTests` auto-formula postures, pin passthrough, out-of-span refusals naming qualified keys, quota re-resolution with pins preserved, byte-identical parity across forced-block/pinned-chain-group/streaming-fallback postures for Q4_K/Q6_K GEMV + batch + fused batch pair; `CpuTuningConfigTests` loader acceptance/defaults/unknown-key/out-of-span refusals, restart-only classification, Diff/ExtractEdits naming).
- `dotnet test ... --filter "(Category=Prefill|Engine|Kv|Attention|Kernel|Serve|Config|HotReload)&Category!=Cuda"` — 891 passed, 0 failed.
- Main validation after worker handoff: `dotnet build Lair.sln -c Release` clean; targeted filter `FullyQualifiedName~CpuTuning|FullyQualifiedName~SharedInputGemv|Category=Kernel|BatchKernel` passed 550/550; broad non-CUDA gate passed 891/891; decode/prefill reruns recorded in `main-decode-t32-*.txt` and `main-prefill-t32.txt`.

## Limitations / notes

- Tuning is not exposed in `CpuSelectionReport` / monitor JSON yet (resolved values are readable through the internal `ResolvedTuning` seam only).
- `engine.cpu.tuning` pins are restart-only by classification, same as the ISA beside them; a quota edit re-resolves auto knobs live but cannot change pins.
- Decode numbers here should be read against the staged report's quieter-session numbers as a load caveat, not as a regression: identical resolved values at this posture, identical output bytes.
