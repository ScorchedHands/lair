# MVP7 Flash-Next reality check — 2026-11-21

## Command and posture

Xeon CPU-only lane (`lair` / Intel Xeon E5-2683 v4 @ 2.10GHz, 64 logical cores, ~252 GiB RAM, AVX2 only, no AVX-512). Pinned model: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` (qwen4exp, IQ3_XXS, 3.0625 bpw, 70.62 GiB, 176.94 B params) assembled from two shards in `/tmp/lair-models`.

Harnesses:

```text
llama.cpp b12032          /tmp/lair-tools/llama.cpp/b12032/llama-b12032/
official ik_llama.cpp main 7c00f8788e303bfba2886138e288effb0fd5fbbf (built from source)
```

Commands (CPU-only, 8 threads, batch 512, ubatch 512, no GPU layers):

```bash
# mainline batched-bench
llama-batched-bench -m <model> --verbose -ngl 0 -t 8 -tb 8 -c 2048 -b 512 -ub 512 -npp 128 -ntg 256 -npl 1 -fa off

# mainline llama-bench (default and -ncmoe 999)
llama-bench -m <model> -t 8 -b 512 -ub 512 -ngl 0 -p 128 -n 256 -r 1 -o md --verbose [-ncmoe 999]

# official ik main llama-bench (-fa off and -fa auto)
llama-bench -m <model> -t 8 -tb 8 -b 512 -ub 512 -ngl 0 -p 128 -n 256 -r 1 -o md --verbose -fa off|auto
```

## Bench results

| Harness | Config | pp128 tok/s | tg256 tok/s |
|---|---|---:|---:|
| llama.cpp b12032 batched-bench | -fa off | 13.40 | 3.18 |
| llama.cpp b12032 llama-bench | default | 13.86 | 3.33 |
| llama.cpp b12032 llama-bench | -ncmoe 999 | 14.85 | 3.27 |
| official ik_llama.cpp main llama-bench | -fa off | 24.07 | 4.31 |
| official ik_llama.cpp main llama-bench | -fa auto | 23.50 | 4.33 |

Mainline CLI (greedy, seed 1, 256 tokens): prompt 10.55 tok/s, generation 3.259 tok/s.

## Bytes/token estimate

Active MoE FFN weight traffic per token from GGUF geometry:

- routed experts: 10 × 3 × 2560 × 640 × 48 = 2,359,296,000 params
- shared expert: 3 × 2560 × 640 × 48 = 235,929,600 params
- total active MoE FFN params/token = 2,595,225,600
- at 3.0625 bpw = 993,484,800 bytes/token (~947 MiB/token)

Derived decode bandwidth: mainline ≈ 3.16–3.31 GB/s, official ik main ≈ 4.28–4.30 GB/s. This is an active-FFN estimate, not a measured DRAM byte counter.

## Expert-gather observations

- Mainline b12032: `n_expert=512`, `n_expert_used=10`, `n_expert_groups=0`.
- Official ik main: `fused_moe=1`, graph nodes 4750, graph splits 1, "enabling only_active_experts scheduling".
- Neither harness logged per-layer selected expert ids; LAIR gather design relies on the recorded geometry (512 total / 10 active).

## Harness failures

- Official pinned `ik_llama.cpp` commit `4e87a0e7fe75f906e2fcfa4d7671494d43a46ed5`: built successfully but SIGSEGV (139) on the target model; SIGFPE (136) under `-fa auto`. Rejected as a baseline harness.
- Thireus unofficial prebuilts `main-b5431-4e87a0e` and `main-b5574-45acbfc`: `undefined symbol: llama_get_dflash_draft_lattice_top_k` during generation. Rejected.
- Official main `ik_llama.cpp` `llama-cli` loads the model but emits EOS after 1 token even with chat/reasoning flags; `llama-bench` is the usable official-main evidence.

## Gate verdict

- Usable baselines recorded: mainline llama.cpp b12032 (`pp128 ≈ 13.9–14.9`, `tg256 ≈ 3.2–3.3`) and official ik_llama.cpp main (`pp128 ≈ 24`, `tg256 ≈ 4.3`).
- The ik/main advantage over mainline (~1.7× prefill, ~1.3× decode) is attributable to its MoE-specific CPU optimizations (only-active-expert scheduling, fused MoE). This sets the bar LAIR's gather/residency work must meet or beat on this Xeon.
- CUDA observations remain deferred to a scripted idle-accelerator run per DN-M7-13.

## Artifacts

Machine-readable data and HTML report: `reports/2026-11-21T07-04-13Z/` (`manifest.json`, `host.json`, `bench.json`, `cli.json`, `report.html`). Raw harness logs remain on the Xeon under `/tmp/lair-reality-check/`.
