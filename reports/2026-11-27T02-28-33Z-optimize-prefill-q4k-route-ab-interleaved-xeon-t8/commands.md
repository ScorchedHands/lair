# Commands and evidence — interleaved q4k route A/B (Xeon t8)

Run window: 2026-11-27T02:28:33Z -> ~02:55 UTC (remote clock UTC), host root@192.168.33.53 via SSH MCP profile `lair-lab`.
No git commit/push was performed anywhere; no remote files deleted.

## Remote checkout update

- Before: branch `phase/mvp7` at `e3f001f`, clean tree (`git status --short --branch`).
- Update command (via ssh run-command):
  `cd /root/lair-measurements/qwen25-small-cpu/repo && git fetch origin phase/mvp7 && git merge --ff-only origin/phase/mvp7`
- Result: Fast-forward `e3f001f..f5ea30c`. HEAD after update (short): **f5ea30c** (`git rev-parse --short HEAD`).

## Reuse-binary verification (no rebuild)

- Command: `git diff --stat e3f001f..HEAD | tail -n 40` and
  `git diff --name-only e3f001f..HEAD | grep -cE '(^|/)(src|bench)/'` -> **0**.
- Diff-stat: 23 files changed, all under `reports/2026-11-27T01-16-57Z-optimize-prefill-q4k-route-ab-xeon-t8/**`, plus `NOTES.md` and `PLAN.md`. No `src/` or `bench/` files changed since `e3f001f`.
- Bench binary reused as-is: `/root/lair-measurements/qwen25-small-cpu/build/app-q4k-ab/Lair.Bench.dll`.

## Script transfer (local -> remote)

- Local file: `.agent-worktrees/mvp7/scripts/run-xeon-prefill-stage-ab.py`
  - size **17190** bytes, LF-only line endings (CR_count=0), local syntax check `python -m py_compile` OK.
  - SHA-256: `A5DC93ABA6F1952683612EDCE7FADE7B733810E9CFD1F2B9BF712D2B37657544`
- Transfer method: SSH MCP `sftp-upload-file` was refused on this host ("Streaming SFTP file tools are not available on Windows"), so the instructed fallback was used: short-lived local HTTP server (`python -m http.server 8017`, bound 192.168.33.90, stopped immediately after use) + remote `curl` into staging path `/root/lair-measurements/qwen25-small-cpu/stage/stage-ab-transfer.py`.
- Remote staged copy: SHA-256 `a5dc93aba6f1952683612edce7fade7b733810e9cfd1f2b9bf712d2b37657544`, 17190 bytes (identical to local).
- Installed copy: `cp` staged -> `/root/lair-measurements/qwen25-small-cpu/repo/scripts/run-xeon-prefill-stage-ab.py`; SHA-256 there identical (`a5dc93ab...`, 17190 bytes). Remote in-memory syntax check passed (`python3 -c "compile(...)"` -> REMOTE_SYNTAX_OK, no bytecode files written on remote).
- Installed script is untracked in git (`git status --short scripts/` -> `?? scripts/run-xeon-prefill-stage-ab.py`) — deliberately not committed; main session reviews it.
- Leftover staging artifacts (not deleted per policy): `/root/lair-measurements/qwen25-small-cpu/stage/stage-ab-transfer.py`, `/root/lair-measurements/qwen25-small-cpu/stage/ab-manifest.txt`.

## Exact measurement invocation

`cd /root/lair-measurements/qwen25-small-cpu/repo && python3 scripts/run-xeon-prefill-stage-ab.py --root /root/lair-measurements/qwen25-small-cpu --app /root/lair-measurements/qwen25-small-cpu/build/app-q4k-ab/Lair.Bench.dll --model /root/lair-measurements/qwen25-small-cpu/models/qwen2.5-3b-instruct-q4_k_m.gguf --dotnet /usr/local/dotnet/dotnet --threads 8 --widths 64 --lengths 128,256 --kernels gemv,tiled --rounds 4 --arms off,on,16`

Launched as SSH MCP background session `mvp7-ab-interleaved`. Engine flags per invocation:
`<dotnet> <app> prefill-profile <model> --threads 8 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 1 --stage-profile --attention-batch-dots on --matmul-route on --q4k-matmul-route <arm>`

## Run completion evidence

- All 12 `RUN[ab-round{0..3}-arm-{off,on,16}]` lines appeared in session output with exact flags; no failure or exception line observed.
- Remote report dir: `/root/lair-measurements/qwen25-small-cpu/reports/2026-11-27T02-28-33Z-optimize-prefill-q4k-route-ab-interleaved-xeon-t8` (directory-name identity verified by md5 comparison of `ls -d reports/*interleaved*` output vs the reconstructed name — equal; MCP redactor masks long names in plaintext).
- All 12 raw logs present remotely (`logs/*.txt | wc -l` -> 12), sizes 3842–3888 bytes, timestamps 02:30 -> 02:55 UTC monotonic per round/arm alternation order; `data.json`, `summary.md`, `report.html` written at 02:55.
- Anomaly (honest note): the MCP background session was auto-closed when its command finished before my final poll, so the script's final `wrote ... rows=...` print and an explicit exit-code echo were not captured in-session output. Completion is evidenced instead by the fail-fast design: any non-zero bench exit aborts without writing later logs or report files; since all 12 logs plus all three generated artifacts exist, the run completed through its final write step (exit 0 path).

## Artifact download and verification

- `ssh_sftp-download-file` unavailable on Windows (same streaming-SFTP refusal); downloads used a short-lived remote HTTP server (`python3 -m http.server 8019`, rooted at the report dir, session closed immediately after) + local `curl.exe --fail`. Byte-for-byte transfer; integrity verified against a manifest computed on the remote host (`sha256sum` of all 15 files -> `stage/ab-manifest.txt`, fetched via MCP content download).
- Local verification: every downloaded file non-empty and SHA-256-equal to the remote manifest entry — **PASS 15/15, failures=0**.

| file | bytes (remote = local) | SHA-256 (manifest = local) |
| --- | ---: | --- |
| logs/stage-ab-r0-off.txt | 3842 | 0841b40f97a5f3aac04d23b84175c9e07fcedd388008e08fe6dd0f29c2bf4634 |
| logs/stage-ab-r0-on.txt | 3886 | 83b71d5f1ef0a3f238abb3ac6c9a7b8a39b667c8c4f7d52eb625f8c0cfa705d8 |
| logs/stage-ab-r0-16.txt | 3845 | a61cf08e0709d4787ec5f18093bb274d6d09ee23966d842227402c6b99741115 |
| logs/stage-ab-r1-off.txt | 3845 | edcb4539a8997f101c9c5afa7120ebab02d1d3b4fdd4dfbc325e91f5683ca76f |
| logs/stage-ab-r1-on.txt | 3839 | 877c15751b7387ab4a5a4ae61d74ace6675597d14c6ca128403b82521e9d8d0e |
| logs/stage-ab-r1-16.txt | 3887 | 39436de0e103ee1ed85e5036419242b6f7f07a6efd528f6893a09325a18962d2 |
| logs/stage-ab-r2-off.txt | 3887 | d0007c5ea25f065fbb080306fef803005f46b914916a86ba51968469f2147128 |
| logs/stage-ab-r2-on.txt | 3888 | 0ef130dc935743eac22e5909f14f4a9d844374972841542866e509e3a2f4a6d0 |
| logs/stage-ab-r2-16.txt | 3847 | 40934b9392e94fba7af48ae4b719b06e611fe93493ba3dd0a009cbc8fb45d7af |
| logs/stage-ab-r3-off.txt | 3846 | 3793a7022522fa326750f95eb20c7155db448ad3f5bac69c0924d1b8de32d5e3 |
| logs/stage-ab-r3-on.txt | 3846 | b08f8633539e04069c0c934e83bab5f2ec3084e1b5fca13e0651a6f1ca3163cd |
| logs/stage-ab-r3-16.txt | 3844 | bddd5cc2052f7a613f10ad3abe6282cffdea716550361fc2435c28d7f7098db5 |
| data.json | 74927 | 869da0c87edeecf8427b22a6b4f4022b2cd7877667ada78910f19b6338e36399 |
| summary.md | 2632 | a51bd0c2bc6becc8e48456dc08a37a7a906d9274752a0152657dedc0ef36068a |
| report.html | 2887 | 957d8d0733b2ef87981876070b903ff725ac25cd84bb78a82b04c497ef4329f8 |

## Anomalies / notes

- MCP output sanitizer masked long directory paths in plaintext tool results; identity was confirmed by md5 comparison instead (see above). No effect on artifact bytes.
- Inter-arm alternation worked as designed (round logs written strictly off -> on -> 16 per round), but intra-arm total_ms spreads remained high (worst pooled spread: `on` 26.57%, `16` 23.89%, baseline `off` 20.22%), so same-session alternation reduced drift-stacking yet did not push host variance under the deltas observed at 4 rounds / reps=1.
- No bench invocation failed; no partial-run recovery was needed.
