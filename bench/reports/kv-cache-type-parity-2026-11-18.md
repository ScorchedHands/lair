# MVP5.1 CPU quantized KV cache-type parity — 2026-11-18

## Question

Measure LAIR's public CPU quantized KV serving postures against llama.cpp b12032 server mode run with matching cache types, then record per-mode bands without widening existing dense-family parity posture.

Named combos measured in this pass:

- `q8_0/q8_0` (public names include `q8`, `int8`, and `k8v8`)
- `q8_0/q4_0` (public name `k8v4`)

## Measurement shape

- Model fixture: `models/SmolLM-135M-Instruct-F16.gguf`
  - sha256: `1fc02c21fba7874b15955d21dc59182aeae382abea412419ffd2fbaa84861790`
- Reference engine: pinned llama.cpp b12032 server mode, `/completion`, temperature 0, `n_probs=5`, threads 8, context 1024.
- LAIR candidate posture: production CPU quantized KV route opened through the same seams serving uses — a resolved `KvFormatPair` at model open plus typed block pools and generic `LayerOps<TK,TV>` attention.
- Prompts: standard parity set `p1..p4`, including the ChatML chat-template prompt that historically exposes near-tie drift.
- Sequencing rule used by the new harness: capture all llama-server references first, stop/kill llama-server, then run LAIR calibration so the two engines do not contend for CPU cores during measurement.

## Gate shape

The new quantized KV parity tests mirror the existing cross-tool parity gate:

- LAIR's argmax must appear among the pinned reference top-5 candidates.
- Every calibrated LAIR logprob must reproduce within the normal 0.2 nat reproduction tolerance, including documented absences.
- Candidates inside the 6-nat sampling window below the reference top-1 must stay inside that mode's recorded quantized-KV band.
- Existing dense-family bands are not widened.

## Results

| Mode | Worst relevant candidate drift | Max relevant delta | Recorded band |
|---|---|---:|---:|
| `q8_0/q8_0` | `Once upon a time` candidate `(` (`id 365`) | 0.1347 nats | 0.25 nats |
| `q8_0/q4_0` (`k8v4`) | `Once upon a time` candidate ` there` (`id 665`) | 0.3856 nats | 0.50 nats |

Both modes kept LAIR's argmax inside the pinned llama.cpp top-5 candidates for all four prompts. No candidate was calibrated as present in LAIR and later absent.

## Fixture outputs

Generated fixtures:

- `tests/fixtures/parity-kv-q8q8-smollm-f16.json`
- `tests/fixtures/parity-kv-k8v4-smollm-f16.json`

Generation commands used from the MVP5.1 worktree:

```powershell
.\scripts\generate-kv-cache-type-parity.ps1 -ModelPath "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\SmolLM-135M-Instruct-F16.gguf" -CacheTypeK q8_0 -CacheTypeV q8_0 -Out "tests/fixtures/parity-kv-q8q8-smollm-f16.json"

.\scripts\generate-kv-cache-type-parity.ps1 -ModelPath "C:\source\gitlab.scorchedhands.com\personal\LAIR\models\SmolLM-135M-Instruct-F16.gguf" -CacheTypeK q8_0 -CacheTypeV q4_0 -Out "tests/fixtures/parity-kv-k8v4-smollm-f16.json"
```

## Verification commands

```powershell
dotnet build Lair.sln -c Release --no-restore
```

Result: `0 Warning(s) / 0 Error(s)`.

```powershell
dotnet test tests/Lair.Tests -c Release --no-build --nologo --filter "Category=Attention"
dotnet test tests/Lair.Tests -c Release --no-build --nologo --filter "Category=Kv"
dotnet test tests/Lair.Tests -c Release --no-build --nologo --filter "Category=Config|Kv"
dotnet test tests/Lair.Tests -c Release --no-build --nologo --filter "Category=Serve"
dotnet test tests/Lair.Tests -c Release --no-build --nologo --filter "Category=Layers|Kernel"
dotnet test tests/Lair.Tests -c Release --no-build --nologo --filter "Category=Parity"
```

Results:

- `Category=Attention`: 10 passed.
- `Category=Kv`: 77 passed.
- `Category=Config|Kv`: 439 passed.
- `Category=Serve`: 26 passed.
- `Category=Layers|Kernel`: 427 passed.
- `Category=Parity` after adding the quantized KV fixtures: 72 passed.

`Category=Api` was rerun in isolation after a full-suite contention flake in `TokenBudgetAttributionApiTests.ResponseCappedExchange_ReportsResponseExhausted_AndKeepsLengthFinishReason`; the isolated API lane passed 101/101. The full-suite failure is consistent with the existing whole-suite real-model load caveat and is not attributable to the quantized KV parity harness.
