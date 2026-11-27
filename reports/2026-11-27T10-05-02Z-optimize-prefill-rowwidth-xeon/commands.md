# Commands log — LAIR prefill row-width probe on lair-lab (2026-11-27T10-05-02Z start)

All remote paths under `R = /root/lair-measurements/qwen25-small-cpu/reports/2026-11-27T10-05-02Z-optimize-prefill-rowwidth-xeon`. Raw logs were **not** copied to this local directory (token economy); they remain only at `R/logs/` on the host. Local artifacts: `data.json`, `summary.md`, `report.html` (byte-identical both sides, hashes below), plus this file authored locally.

## Environment verification (exit 0 unless noted)

| # | Command (run via MCP profile lair-lab; pipes/redirects required run-command) | Exit | Notes |
|---|---|---|---|
| V1 | `pgrep -af 'prefill-profile\|llama-bench'; echo "PGREP_EXIT=$?"; git -C /root/lair-measurements/qwen25-small-cpu/repo rev-parse --short HEAD; git -C …/repo branch --show-current; ls -la "<app>/Lair.Bench.dll" <model> <llama-bench>` | 0 | No bench procs running (only the pgrep wrapper self-matched); **HEAD `016dd9c`, branch `phase/mvp7`** confirmed without fetch; app dll 457,216 B; model 2,104,932,768 B; llama-bench present |
| V2 | `mkdir -p $R/logs && echo MKDIR_OK $?` | 0 | scratch dir created |

## Bench arms (sequential; each exact command run with `> logs/<name>.log 2>&1`, exit captured to `<name>.exit`)

APP=`/root/lair-measurements/qwen25-small-cpu/build/app-posture-2026-11-27T06-53-30Z/Lair.Bench.dll`; MODEL=`/root/lair-measurements/qwen25-small-cpu/models/qwen2.5-3b-instruct-q4_k_m.gguf`; DOTNET=`/usr/local/dotnet/dotnet`. Arms 1 ran as background session `arm1_t8`; arms 2–5+help chained sequentially (no `set -e`) via uploaded scratch scripts `run_arms.sh` / `run_arms345.sh`.

| # | Exact command | Exit | Log (`R/logs/…`, size, SHA-256) |
|---|---|---|---|
| 1 (spec run 1) | `/usr/local/dotnet/dotnet $APP prefill-profile $MODEL --threads 8 --lengths 128,256 --widths 32,64 --kernel gemv,tiled,gemm --reps 3` | **0** | `arm1_t8_gemv-tiled-gemm_w32-64_l128-256_r3.log` 4,246 B · `7efc7b9fc280857f15c7e607e2e23e597bec79cbec855c5db6e3604a26e3a89a` |
| 2 (spec run 2) | same as #1 with `--threads 32 … --reps 3` | **0** | `arm2_t32_gemv-tiled-gemm_w32-64_l128-256_r3.log` 4,290 B · `abd0cced707d21eabee11c6d10ac795bc53ea7cac4a39d57ea72f59a949601d5` |
| — (aborted attempt) | same as #1 with `--reps 3 --per-token` (spec run 3 at spec reps; killed after ~10 min wall / ~71 min CPU, no table emitted before first per-token block finished computing; process tree terminated by closing its SSH session) | n/a — aborted, not a normal exit | partial kept: `arm3a_t8_perToken_gemv-tiled-gemm_w32-64_l128-256_r3.log` 501 B · `076f13a9c7ff170368f7ed34ed2df02d2b1b1d99a5198154923ae5aa62155fb9` |
| 3a (spec run 3, **deviation: `--reps 1`**) | `/usr/local/dotnet/dotnet $APP prefill-profile $MODEL --threads 8 --lengths 128,256 --widths 32,64 --kernel gemv,tiled,gemm --reps 1 --per-token` | **0** | `arm3a_t8_perToken_gemv-tiled-gemm_w32-64_l128-256_r1.log` 4,754 B · `1c58de5cc35a254b5e92427367bc5cc2f3132649b9faf9eb396a18462c8edf27` |
| 3b (spec run 3, **deviation: `--reps 1`**) | same as #3a with `--threads 32 --per-token … --reps 1` | **0** | `arm3b_t32_perToken_gemv-tiled-gemm_w32-64_l128-256_r1.log` 4,755 B · `7b0b633b32214427cce48bdcfddb0571833bf7de24408560b8e2c8194ec8d3c7` |
| 4 (spec run 4) | `/usr/local/dotnet/dotnet $APP prefill-profile $MODEL --threads 32 --lengths 128,256 --widths 64 --kernel gemv,tiled,gemm --reps 2 --stage-profile` | **0** | `arm4_t32_stageProfile_gemv-tiled-gemm_w64_l128-256_r2.log` 5,553 B · `ac620da2037576db7e73b392951b113074a71f0cc4e6180749b8ac0592f9f6f0` — all six `prefill-profile stage … feed_forward_gate_up_ms=…/feed_forward_down_ms=… attention_*_ms=… total_ms=… top_layers=…` lines captured (see data.json `stage_lines`) |
| 5a (spec run 5) | `/root/lair-tools/llama-b12032/llama-bench -m $MODEL -ngl 0 -t 8 -p 128,256,512 -n 0 -r 3` | **0** | `arm5a_llama-bench_t8_p128-256-512_r3.log` 828 B · `0c50c6b5823fc5779ddf3bf3baf7c3ee61ab1facb1bcbf4034b4e44b45139a43` |
| 5b (spec run 5) | same as #5a with `-t 32` | **0** | `arm5b_llama-bench_t32_p128-256-512_r3.log` 828 B · `b43eff651d358d38d5ab630bf2ed718d3701c6c4641276d5b00e151fd6ebe176` |
| 5c (spec run 5, help) | `/root/lair-tools/llama-b12032/llama-bench --help` | **0** | `llama-bench_help.log` 4,623 B · `303755885617e2bb0611bb611e7b9a1a946757057d172072ddbacf1c046f26d3` — defaults captured verbatim: `-b, --batch-size <n>  (default: 2048)`, `-ub, --ubatch-size <n>  (default: 512)` |

Remnant artifact from the aborted chain session wrapper: `logs/chain_runner.log` 0 B · `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty sha).

## Analysis pass (remote, read-only over logs)

- `python3 analyze.py .` at `$R` (script uploaded as scratch file `analyze.py`, 5,488 B; one parser fix applied in place via sed): wrote `data.json`; printed width-ratio/amortization/stage digests consumed into summary.md. Exit **0**.
- Host-side raw-log manifest: `logs/CHECKSUMS.remote.txt` (sha256sum of all 10 logs + data.json at the time).

## Curated artifacts — byte-identical both sides, non-empty ✓

| File | Size | SHA-256 (remote == local) |
|---|---|---|
| `data.json` | 34,842 B | `476d1bf83e5454ed5be0559c62fd78d6381262b08b5e130a177b3fe16b01f6d5` |
| `summary.md` | 12,227 B | `2532180fcb466401589cafa901ecae7daa6588fb4805618abc3499e7968d4dd0` |
| `report.html` | 7,818 B | `5b166e65fb9ae694dc6399b29e30058ba19e0696d83898b5e3ef3ff7fa1edddb` |

Transfer note: streaming SFTP file tools are refused on this Windows client ("transfer root cannot yet be verified private"), so `data.json` was fetched content-based and byte-verified against the remote hash (two transcription slips — both a stray `-0.0` vs `0.0` in `fitResidualMs` fields at data.json lines 567/840 — were caught by slice-hash bisection, corrected locally, then re-hashed to equality). `summary.md` and `report.html` were authored locally, uploaded via sftp-upload content mode, and hash-matched remote↔local.

## Refusal / anomaly lines (verbatim)

- **No kernel refusals anywhere.** `grep -inE 'refus|unsupported|not supported' logs/arm*.log` → `NONE`. gemm executed every span.
- Engine-reported fit warnings (quoted verbatim from the raw tables):
  - arm2 (`arm2_t32_…_r3.log`, gemv rowWidth=32): `weightFit=1931.02 ms/traversal (UNRELIABLE: measured costs are not separable into the two terms on this run) impliedWeightGB/s=0.86` — rows show negative attentionMs `-688.3`, `-2742.7`.
  - arm4 (`arm4_t32_stageProfile_…_r2.log`, gemv rowWidth=64): `weightFit=3155.20 ms/traversal (UNRELIABLE: measured costs are not separable into the two terms on this run) impliedWeightGB/s=0.53` — rows show negative attentionMs `-238.8`, `-951.6`.
- Run-to-run variance, tiled@threads=32@w64 batchedTok/s: 23.47/22.47 (arm2) vs 31.10/29.12 (arm3b). Documented in summary §5; conclusions hold across both.
- Aborted per-token r3 attempt: process killed by SSH session close after ~10 min wall (~71 CPU-min); partial log retained as evidence (row above).

## State at end of run

Host idle (`pgrep` → IDLE), chain marker `logs/CHAIN_DONE.marker` present, all eight arm exit files = 0. Nothing committed or pushed anywhere; no repo source touched locally or remotely; nothing deleted on the host. Scratch helper scripts left in place: `$R/run_arms.sh`, `$R/run_arms345.sh`, `$R/analyze.py`.
