# MVP5.1 q4-K viability spike — 2026-11-17

## Question

Measure LAIR `q4/q4` KV cache output against llama.cpp b12032 server mode run with matching `--cache-type-k q4_0 --cache-type-v q4_0`, then decide whether the named combo stays in MVP5.1's public option set or is dropped without widening an existing parity band.

## Measurement shape

- Reference engine: pinned llama.cpp b12032 server mode, `-ctk q4_0 -ctv q4_0`.
- LAIR candidate: `q4/q4` KV mirror through the bench spike seam against production attention output.
- Prompts: standard parity set `p1..p4`, including the chat-template prompt that has historically exposed near-tie drift.
- Runtime shape: context 1024, threads 8, greedy sampling with seed 0 and top-5 logprobs captured from llama.cpp server mode.
- Sequencing rule: for each fixture, llama-server is stopped before LAIR starts so the two engines do not contend for CPU cores during measurement.

Corrected LAIR candidate semantics used in this run follow pinned b12032 `q4_0` behavior: signed-max scale selection with FP32 reciprocal quantization (`id = 1/d`), stored half-scale, C-style truncating code rounding after offset (`(int)(z + 8.5f)` clamped to `[0,15]`), and reference q4_0 nibble layout where byte `j` stores low element `start+j` plus high element `start+16+j`. All-zero groups follow the faithful C path rather than a special-cased positive zero: scale negative zero (`half(0)` as `0x8000`), reciprocal id 0, codes truncating to 8, and nibbles `0x88`.

## Gate posture

The comparison uses the existing quantized-file cross-tool gate shape: candidates within 6 nats of the reference top-1 must stay inside the quantized-file band (1 nat) and LAIR's argmax must remain among the reference top-5. No dense-family band is widened, and no q4-specific tolerance is adopted unless it clears measurement first.

## Results

| Fixture | Worst `q4` vs llama `q4_0/q4_0` relevant delta | Worst `q4` vs LAIR batched baseline relevant delta | Notes |
|---|---:|---:|---|
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (primary serving fixture) | 4.0019 nats | 3.3946 nats | Primary p4 stays argmax-in-top5 but misses one relevant token and exceeds the quantized band by 4×. |
| `Llama-3.2-1B-Instruct-IQ4_XS.gguf` | 1.3559 nats | 1.3125 nats | Exceeds the existing quantized-file band on p4; baseline cross-tool drift is also above band on that prompt. |
| `Qwen3-0.6B-Q8_0.gguf` | 3.7929 nats | 4.3014 nats | Argmax moves outside the reference top-5 on p1, p2, and p4; this is a hard membership failure as well as drift. |
| `Qwen3-0.6B-Q8_0-YaRN.gguf` | 2.0977 nats | 2.6298 nats | p4 moves outside the reference top-5 and all relevant tokens are absent from the q4 candidate window. |
| `SmolLM-135M-Instruct-F16.gguf` | 1.0671 nats | 3.0457 nats | Marginal cross-tool failure, but LAIR-to-LAIR q4 cost is far above the baseline posture. |
| `smollm-135m-instruct-q8_0.gguf` | 1.7457 nats | 1.8973 nats | Exceeds band on p3/p4 despite argmax membership remaining in top-5. |
| `smollm-135m-instruct-q4_k_m.gguf` | 2.8091 nats | 3.0266 nats | Broad drift across prompts; not a near-tie-only failure. |
| `Llama-3.2-1B-Instruct-Q3_K_S.gguf` | 1.8759 nats | 1.1218 nats | p4 misses two relevant tokens from the q4 candidate window while argmax remains in top-5. |
| `Llama-3.2-1B-Instruct-Q2_K.gguf` | 4.8785 nats | 4.2921 nats | Primary failure mode: large drift plus missing relevant token on p4. |
| `Llama-3.2-1B-Instruct-UD-IQ3_XXS.gguf` | 1.1745 nats | 0.9247 nats | Exceeds band on p2; p1 also misses one relevant token despite its max delta staying below band. |
| `gemma-2-2b-it.Q4_K_M.gguf` | 0.4972 nats | 0.2726 nats | Clears the measured gate, but a single fixture does not rescue the named combo when primary and broader pinned fixtures fail. |
| `Phi-3-mini-4k-instruct-Q2_K.gguf` | 1.7777 nats | 1.3713 nats | Exceeds band on p1/p4 while argmax membership remains in top-5. |

## Decision

Drop `q4/q4` from MVP5.1's named KV option set. The measured drift is not a narrow near-tie artifact: the primary serving fixture shows 4.0019 nats worst relevant delta with one missing relevant token, Qwen/YaRN fixtures produce argmax-membership failures, and multiple other pinned fixtures exceed the existing quantized-file band independently of dense-family tolerance changes.

The spike does not validate `k8v4` by inference from this failure: `q4_0` remains a per-side type needed for asymmetric KV work only where its own mode validation can justify it later. The public named combo set proceeds as `f32/f32`, `f16/f16`, `q8/q8`/int8/k8v8, and `k8v4`.

## Verification commands

```powershell
dotnet build Lair.sln -c Release --no-restore
```

Result: `0 Warning(s) / 0 Error(s)`.

```powershell
dotnet test tests/Lair.Tests -c Release --no-build --nologo --filter "Category=Parity|Kv"
```

Result: `Failed: 0, Passed: 169, Skipped: 0`.

## Evidence paths

Raw measurement captures used local temp output directory `%TEMP%\kilo\q4-parity-corrected`, including per-fixture llama reference JSON files and LAIR measurement text outputs such as `llama32-q4km.llama-q4-cache.json` and `llama32-q4km.lair-measurements.txt`.
