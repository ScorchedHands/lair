# commands.md — measurement run 2026-11-27T14-14-58Z (local record only)

Chronological record of commands executed for this measurement, with exit statuses and verification hashes. Host = lair-lab profile (dual Xeon E5-2683 v4). Paths abbreviated: `R` = `/root/lair-measurements/qwen25-small-cpu/reports/2026-11-27T14-14-58Z-optimize-q4k-widebatch-xeon-pinning`, `S` = `/root/lair-measurements/qwen25-small-cpu/staging-widebatch-xeonpin`.

## Display-sanitizer caveat

The MCP transport redacts long absolute paths and high-entropy strings in tool results (`[REDACTED:entropy:N]`). Hash values themselves survived every gate; path names quoted below are the true host paths. One consequence recorded during measurement: `publish` success line display was redacted, but the proof exists verbatim in `R/logs/publish.log`.

## 1. Host idle / posture probes (read-only)

- pgrep sweep before and between arms — matched only our own wrapper processes; load averages ~0.28–1.27 attributable to our runs. No external contention observed at any point.
- `cat /sys/.../cpufreq/scaling_governor` → performance on all cores.
- taskset probe: `Cpus_allowed_list: 0-15` for restricted mask; numactl absent on host (pinning applied via taskset cpu lists only).

## 2. Bench seam install (local blobs → host)

Local git blobs extracted from branch tip `phase/mvp7` = commit `50a4ef0` into `%TEMP%\kilo\xeonpin`:

- Q4KWideBatchBench.cs len=16012 sha=35ac30424a19b4a32918e7e17a2ac99a9b38095ae1e36d035e44b66219ab49c4 (identical copy as wb-candidate.cs)
- Program.cs len=87849 sha=58e8e8ff1569d03a03b1b4a4cb6eaee872e716a3390c7b3fd0c0fd141d8ffd34
- program.patch len=3033 sha=653958c807559d2fc1ca9cffb6551b015854d9883af629251c70e34560a00f44
- wb-line.txt len=479 sha=9adb4de794b8cc763078439b611d66b69db870d8f82bb3968c763610f553dedd

Attempted transfer: local HTTP server + remote curl (`serve.ps1` len=706) — FAILED, `curl: (28) Connection timed out`, exit 28; host firewall blocks inbound from host to workstation. Fallback used instead: sftp-upload of Q4KWideBatchBench.cs with sha gate → matched blob hash on host ✓. Program.cs was NOT re-uploaded wholesale; anchor-guarded python3 edit inserted the dispatch line using `S/wb-line.txt` (uploaded, verified 9adb4de7…) into the repo copy at commit state — remote status shows ` M bench/Lair.Bench/Program.cs`, `?? bench/Lair.Bench/Q4KWideBatchBench.cs`. Transcription slips during earlier sftp uploads of Program.cs candidates were caught by hash gates and repaired via sed (dropped `/`; hallucinated placeholder token) before the anchor-edit approach replaced them.

Remote repo after install: HEAD 016dd9c; nothing committed on host.

## 3. Publish

- `dotnet publish` into app dir `app-widebatch-2026-11-27T14-14-58Z` → PUBLISH_EXIT=0 (success line display redacted by sanitizer; log retained at R/logs/publish.log, sha dff18c9853114905327d6199fe5faec99f4d86a03663c65413ecfb5c5a65bd72).

## 4. Measurement arms (sequential, all exit-coded)

Kernel wide-batch (`--threads T --reps 5 --counts 8,16,32,64`), each arm `WB_EXIT=0`, every cell reported bit_identical parity lines; no NaN warnings:

| # | log | posture | sha256 of raw log on host |
|---|---|---|---|
| arm1 | arm1_wb-t8-unpinned.log | threads 8, unpinned | af7ab1d6cda37123a44f22542ae6c93bf3421ed81317fb8694d90b2bbfba0761 |
| arm2 | arm2_wb-t8-pin-node0phys.log | threads 8, taskset 0-15 | 8bf75c455daa2d4fcf01bc9c303c4088a061c3152551489b59f901b0abf05832 |
| arm3 | arm3_wb-t8-pin-node0ht.log | threads 8, taskset 0-15,32-47 | fc3228b51c93939852363eea34ecc301134f5fd38cb7225ad0bf9fed6d87ac65 |
| arm4 | arm4_wb-t32-unpinned.log | threads 32, unpinned | f1691c173202bb011465b4c9831952cdc792e3b63d832f9489bafe5949b53daa |

E2E prefill (`--threads 8 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 3`), each `PF_EXIT=0`:

| # | log | posture | sha256 of raw log on host |
|---|---|---|---|
| e2e1 | e2e1_pf-t8-unpinned.log | unpinned, 14:21:42–14:26:26Z | 5b744463357ad04c609d2f0bc55aec214d4137d5367cc17bdf64506763615dd9 |
| e2e2 | e2e2_pf-t8-pin-node0phys.log | taskset 0-15, 14:28:10–14:32:34Z | 42a5b147f242ba98ac6eaf5587edf5d4007fa2ba63b8eba7d3dc90ae53541e77 |
| e2e3 | e2e3_pf-t8-unpinned-rerun.log | unpinned, 14:33:32–14:38:13Z | 38d265e39bef09b2e40ef0c5733227c25163b38d3f3faa9bf2b9c3b74ae435c6 |
| e2e4 | e2e4_pf-t8-pin-node0ht.log | taskset 0-15,32-47, 14:39:33–14:44:18Z | bd7a2f9283ad67b22c23c290a9720a01253dd038bb99f47d1542809ad65890de |

Raw logs remain ONLY on host under R/logs/ (not copied down; curated data.json carries their extracted values).

## 5. Host-side parsing and audit

- `S/parse_wb.py` — log→JSON extractor. Local authored copy sha 175ec507c7316f8c66675ac00a16c79d767e3d61529288629724cf90e4ef67ff; two fixes applied on host via sed (KPARITY capture group; KSTAT off-by-one field indices) plus path-shortening edits to keep posture fields out of the sanitizer's entropy redaction. Final on-host sha d5fd2210a7305664fef8c73c8ca9988f45dd8c4f1f717b2b940c612235b3b686.
- `S/audit_wb.py` — cross-checks JSON against raw logs: output `KERNEL_CELLS_CHECKED 96`, `JSON_BAD 0`; all prefill fit_lines verbatim-present. On-host sha f62b53116f3856db93256f250a275f479571ea74941ac99176e7ececc81495a4.
- Curated `R/data.json` (compact, separators no-space): 30742 B, sha 9673370508c1b1f47966155aa6a22973e454f44b1b45a1a85589545606e3e129.

## 6. Byte-verified download of data.json (host → local report dir)

Direct scp/streaming unusable: local OpenSSH probe `root@192.168.33.53` → Permission denied; MCP streaming SFTP-to-disk tool unavailable on Windows. base64 inbound transfer blocked by sanitizer entropy redaction. Method used: host-side `split -b 4096` into `S/xfer/datajson.part.{aa..ah}` (part shas aa=5411aff…, ab=56289405e2d4fa2167a5b041029058bc9678edb4c6a61c506ac352eeee26f970, ac=222f766b…, ad=d408b725a2f7fa0d0b99239186eac6158b3807cfcad9d94e4537399c0a9f99cb, ae=e13b49136d65181d2b78c9d3220dc50497b8b62a41691064ba04b8fc0291c7a3, af=86cb77431b2f198e65e2fbdaa628ea81f82cb710b86538df4ea3149beca825cc, ag=e7435193f04b578efd9684096ea482ec43db250970881b0d031a525f33beb94f, ah=28051263b940f5ebee0af3fdc55d5eedc493df00544d57aada0ed1434f7f55ef), each part transcribed into local staging files and gated against its own remote hash at every prefix boundary (gates passed at 4096/8192/10240/11264). One transcription defect: ac-q4 chunk lost a trailing `"` — located by single-char brute-force insertion search against the full-part gate da4f977375b6d877cac97406f56a4e551959b310443825f24855e12933cee727 (found at position 12287) and repaired. Final local file: len=30742, sha matches remote 967337… exactly ✓.

## 7. summary.md generation and upload (local → host)

- Generated locally by `gen_summary.ps1` (kept in this directory for provenance) from the byte-exact local data.json — every number pulled programmatically via ConvertFrom-Json, none hand-typed. Generator fixes applied before finalizing: removed traversalBytes reference (field exists only in raw arm logs, not in data.json), replaced a dual-value Where-Object subexpression in the verdict bullet, restored DateTime formatting to verbatim ISO Z timestamps, and fixed backtick-escape garbling in one prose line.
- Final local summary.md: 14368 B, sha 65315c947338aecc8e192f6a49d5b97246d48840ad16925fa821a72eb5a61104.
- Upload method: split into 5 raw chunks (3072×4 + 2080; decoded-shas 63ff7425…, 77bf37f9…, 60de808d…, 42ad04c9…, f25f2d0012ca0f9a2c172b88e470a042f74c666ef2be89184f789f176c439fb5), each base64-folded to ≤1500-char lines, uploaded via sftp-upload as `S/xfer/smx.b64.{1..5}`, then host-side decode gate: all five decoded hashes matched first try ✓. Assembled with cat → R/summary.md.new (14368 B) → sha 65315c94… matches local exactly ✓ → renamed to `R/summary.md`.
- An earlier ah-part transcription during the data.json phase dropped a 157-byte segment mid-block; caught by per-part hash gate, fixed from a fresh fetch.

## 8. Final state verification (host)

`ls -la R`: data.json 30742 B + logs/ + summary.md 14368 B; `sha256sum` re-run confirms both artifacts match the local byte-exact copies:
- data.json → 9673370508c1b1f47966155aa6a22973e454f44b1b45a1a85589545606e3e129
- summary.md → 65315c947338aecc8e192f6a49d5b97246d48840ad16925fa821a72eb5a61104

Nothing deleted on host; staging (S/, S/xfer/) retained. No commits or pushes performed anywhere in this run.

## 9. Local-only artifacts in this directory

- data.json — byte-exact copy of remote (hash above).
- summary.md — source of the uploaded artifact (identical bytes to remote).
- gen_summary.ps1 — deterministic generator, local provenance for summary.md.
- commands.md — this file (authored locally only; not present on host by design).
