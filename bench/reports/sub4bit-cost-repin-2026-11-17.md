# MVP5 measured-cost re-pin — 2026-11-17

## Scope and posture

This report re-measures the sub-4-bit i-quant cost after the MVP5 integer-dot rewrite, against the pinned dense baselines:

- `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`
- `models/Llama-3.2-1B-Instruct-IQ4_XS.gguf`
- `models/Llama-3.2-1B-Instruct-UD-IQ3_XXS.gguf`

The GEMV runs used the current Release bench build with settled high-repetition posture:

```powershell
dotnet bench\Lair.Bench\bin\Release\net10.0\Lair.Bench.dll gemv-file "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf" --threads 8 --reps 200 --max-rows 256 > bench/reports/gemv-mvp5-q4k-reps200-2026-11-17.txt
dotnet bench\Lair.Bench\bin\Release\net10.0\Lair.Bench.dll gemv-file "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\Llama-3.2-1B-Instruct-IQ4_XS.gguf" --threads 8 --reps 200 --max-rows 256 > bench/reports/gemv-mvp5-iq4xs-reps200-2026-11-17.txt
dotnet bench\Lair.Bench\bin\Release\net10.0\Lair.Bench.dll gemv-file "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\Llama-3.2-1B-Instruct-UD-IQ3_XXS.gguf" --threads 8 --reps 200 --max-rows 256 > bench/reports/gemv-mvp5-udiq3xxs-reps200-2026-11-17.txt
```

All three GEMV processes reported:

```text
sub4bit axis: forcedGrid=auto forcedPrefetch=auto resolvedGrid=gather prefetchHints=on
```

Raw tables are retained beside this report. The live-decode reruns used temporary configs outside the repo with `max_context_tokens: 4096`, `kv_blocks: 256`, engine `threads: 0`, and model-level `placement.thread_quota: 8`. One server ran at a time on ports `59101`, `59102`, then `59103`; startup logs confirmed each run with `compute_grant ... requested=8 granted=8`. A first pass that omitted `thread_quota` is excluded from the reported live numbers because it did not match the documented serving posture.

## GEMV cost

The primary kernel-cost comparison is resident single-threaded GEMV (`threads=1`). Ratios use the pinned `Q4_K_M` fixture summary value of `0.164 ns/elt`.

| Format group | Summary resident 1T | Settled tensor samples | Cost vs Q4_K baseline |
|---|---:|---:|---:|
| `IQ2_S` in UD file | `553.0 ns/GEMV`, `0.270 ns/elt` | `504.7–507.0 ns/GEMV`; one first-measured sample at `693.4 ns/GEMV` inflates the group summary | settled ≈1.50–1.51×; summary ≈1.65× |
| `IQ2_XS` in UD file | `655.3 ns/GEMV`, `0.320 ns/elt` | `654.7–655.5 ns/GEMV` | ≈1.95× |
| `IQ3_S` in UD file | `692.1 ns/GEMV`, `0.338 ns/elt` | `691.0–693.4 ns/GEMV` | ≈2.06–2.07× |
| `IQ3_XXS` in UD file | `783.9 ns/GEMV`, `0.383 ns/elt` | `778.9–788.7 ns/GEMV` | ≈2.32–2.35× |
| `Q4_K` anchor in UD file | `340.9 ns/GEMV`, `0.166 ns/elt` | `335.9–350.4 ns/GEMV` | same-file baseline, ≈1.01–1.02× vs Q4_K_M summary |
| `IQ4_XS` mixed wide rows in UD file | `1381.3 ns/GEMV`, `0.169 ns/elt` | `1379.3–1382.8 ns/GEMV`; contraction width is 8192 columns | per-element cost comparable to dense baselines despite higher raw GEMV time |

Baseline fixture summaries:

- `Q4_K_M`: resident 1T `335.8 ns/GEMV`, `0.164 ns/elt`.
- `IQ4_XS`: resident 1T `334.6 ns/GEMV`, `0.163 ns/elt`.

The old ≈6.5× sub-4-bit penalty is not present on this build. The measured stable per-element costs are roughly `0.25–0.39 ns/elt` against the Q4_K baseline, i.e. about 1.5–2.3× depending on format and whether warm-up skew is included in a group summary. `IQ3_XXS` modestly exceeds the intended ≈1–2× band at about 2.3× on this posture; that premise miss is recorded rather than smoothed into an assertion of universal ≤2× cost.

## Live decode measurements

Live throughput uses `/monitor/stream` snapshots captured while serving fixed chat requests with `temperature: 0`, `max_tokens: 256`, one warm-up request, then two measured runs over the same three prompts. Completion token counts differ by model/quant (`337`, `239`, and `204` tokens for Q4_K_M, IQ4_XS, and UD respectively), so client-aggregated tok/s is confounded by output length; monitor-windowed decode tok/s is the primary live comparison.

| Fixture | Startup CPU selection | Monitor peak decode | Peak working set | Client aggregate runs |
|---|---|---:|---:|---|
| `Q4_K_M` | `isa=vnni gather=gather` | 12.43 tok/s | 1430.9 MB | run1 `337 tokens / 41.33 s = 8.153 tok/s`; run2 `337 / 38.22 = 8.818` |
| `IQ4_XS` | `isa=vnni gather=scalar` | 16.46 tok/s | 1344.2 MB | run1 `239 tokens / 16.49 s = 14.491 tok/s`; run2 `239 / 15.54 = 15.377` |
| UD mixed-tier file | `isa=vnni gather=gather` | 12.97 tok/s | 1222.0 MB | run1 `204 tokens / 18.90 s = 10.794 tok/s`; run2 `204 / 17.27 = 11.811` |

The recommendation remains speed-based: `IQ4_XS` is the fastest measured fixture on this runtime at ≈16.5 monitor-windowed decode tok/s, ahead of Q4_K_M at ≈12.4 and ahead of the mixed-tier sub-4-bit file at ≈13.0 while staying under the 4 GB working-set ceiling.

The mixed-tier `UD-IQ3_XXS` fixture no longer shows the old severe penalty: its live peak is about 0.79× Q4_K_M speed and about 0.85× IQ4_XS speed, not ≈0.38× as previously recorded. It remains slower than either dense baseline on this file because it contains a mixture of formats and produces different output lengths; it should not be read as a per-format live measurement for every sub-4-bit quant.

## Selection variance and fixture coverage

The process-wide synthetic grid probe is noisy across separate server starts: in the corrected quota8 runs, Q4_K_M and UD resolved `gather`, while IQ4_XS resolved `scalar`. The GEMV bench processes all resolved `gather`. This report therefore records each live run's actual startup selection rather than treating one global encoding as a fixed posture for every process.

`IQ2_XXS` remains correctness-covered by Kernel/Parity fixtures, but the pinned servable and bench files do not carry an individual measurable `IQ2_XXS` fixture set; no speed claim is made for it beyond its inclusion in the mixed-tier file family coverage.
