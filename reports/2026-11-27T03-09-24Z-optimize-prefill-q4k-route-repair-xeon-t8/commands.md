# commands.md — mvp7 q4k route repair pass (warmup + odd-round rotation), Xeon t8

Local report dir: `.agent-worktrees/mvp7/reports/2026-11-27T03-09-24Z-optimize-prefill-q4k-route-repair-xeon-t8`
Remote report dir: `/root/lair-measurements/qwen25-small-cpu/reports/2026-11-27T03-09-24Z-optimize-prefill-q4k-route-repair-xeon-t8`

## Step 1 — remote checkout sync (SSH MCP profile lair-lab, root@192.168.33.53)

Command:
git -C /root/lair-measurements/qwen25-small-cpu/repo fetch origin phase/mvp7 && git -C /root/lair-measurements/qwen25-small-cpu/repo merge --ff-only origin/phase/mvp7 && git -C /root/lair-measurements/qwen25-small-cpu/repo rev-parse --short HEAD

First attempt output (verbatim):
Updating f5ea30c..860d15c
[stderr] From https://github.com/ScorchedHands/lair
 * branch            phase/mvp7 -> FETCH_HEAD
   f5ea30c..860d15c  phase/mvp7 -> origin/phase/mvp7
error: The following untracked working tree files would be overwritten by merge:
	scripts/run-xeon-prefill-stage-ab.py
Please move or remove them before you merge.
Aborting

ANOMALY (documented, non-destructive resolution): the checkout carried a stale UNTRACKED copy of scripts/run-xeon-prefill-stage-ab.py from an earlier pass that blocked the ff-merge. It was NOT deleted; it was moved aside to /root/lair-measurements/qwen25-small-cpu/attic/run-xeon-prefill-stage-ab.py.stale-untracked-at-f5ea30c (sha256 a5dc93aba6f1952683612edce7fade7b733810e9cfd1f2b9bf712d2b37657544, differs from the committed version b51fe5…, so it was stale/modified). No remote deletions were performed anywhere in this pass.

git status before merge (verbatim): `?? scripts/run-xeon-prefill-stage-ab.py` ; HEAD then f5ea30c.
Pre-merge blob hash of origin/phase/mvp7:scripts/run-xeon-prefill-stage-ab.py = b51fe5aee563ae960b735a95864dd173c548dacde15b5dc7325dc4c00341a72e

Second attempt (after mv), output (verbatim diffstat):
Updating f5ea30c..860d15c
Fast-forward
 NOTES.md                                           |   16 +-
 PLAN.md                                            |    4 +-
 reports/…interleaved-xeon-t8/commands.md           |   72 +
 reports/…interleaved-xeon-t8/data.json             | 3362 ++++++++++++++++++++
 reports/…interleaved-xeon-t8/logs/stage-ab-r{0..3}-{16,off,on}.txt (12 files) | 30 each
 reports/…interleaved-xeon-t8/report.html           |   24 +
 reports/…interleaved-xeon-t8/summary.md            |   52 +
 scripts/run-xeon-prefill-stage-ab.py               |  430 +++
 19 files changed, 4318 insertions(+), 2 deletions(-)

HEAD after ff (short): **860d15c** — as expected.

Binary-reuse proof: git -C …/repo diff --name-only e3f001f..HEAD | grep -E '(^|/)(src|bench)/' || echo none
Output (verbatim): `none` → no engine/bench source changed since the published build; existing app reused, nothing re-published.

Script integrity: post-checkout remote sha256sum scripts/run-xeon-prefill-stage-ab.py = b51fe5aee563ae960b735a95864dd173c548dacde15b5dc7325dc4c00341a72e
Local C:\source\gitlab.scorchedhands.com\personal\LAIR\.agent-worktrees\mvp7\scripts\run-xeon-prefill-stage-ab.py = B51FE5AEE563AE960B735A95864DD173C548DACDE15B5DC7325DC4C00341A72E (18743 bytes) → **match**.

## Step 2 — dry-run schedule verification

Command: python3 scripts/run-xeon-prefill-stage-ab.py --root /root/lair-measurements/qwen25-small-cpu --app …/build/app-q4k-ab/Lair.Bench.dll --model …/qwen2.5-3b-instruct-q4_k_m.gguf --dotnet /usr/local/dotnet/dotnet --threads 8 --widths 64 --lengths 128,256 --kernels gemv,tiled --rounds 5 --warmup-rounds 2 --arms off,on,16 --report-name optimize-prefill-q4k-route-repair-xeon-t8 --dry-run (run from repo dir)

Header printed: `dry-run … warmup_rounds=2 rounds=5 rotate=True`
NOTE: the MCP output sanitizer redacted some path strings in this display ([REDACTED:entropy:N]); counts/order below are taken from the unredacted RUN labels. Each invocation runs one dotnet call covering lengths 128,256 × kernels gemv,tiled with --reps 1; arms map to `--q4k-matmul-route off|on|16` (baseline flags: attention-batch-dots on, matmul-route on).

Schedule as printed — 6 warmup invocations then 15 measured:
warm0: off, on, 16 | warm1: off, on, 16   (fixed arm order per warm round)
round0: off, on, 16 | round1: 16, on, off | round2: off, on, 16 | round3: 16, on, off | round4: off, on, 16   (arm order reversed on odd rounds)
Verified against expectation: 6 warmup + 15 measured = 21 invocations ✓.

## Step 3 — measured run

Exact command (= dry-run minus --dry-run), launched as remote background SSH session `mvp7-repair-ab` via open-session (type=background): requested ttlMs 5400000 was rejected by profile cap; opened with max allowed ttlMs=3600000.

cd /root/lair-measurements/qwen25-small-cpu/repo && python3 scripts/run-xeon-prefill-stage-ab.py --root /root/lair-measurements/qwen25-small-cpu --app /root/lair-measurements/qwen25-small-cpu/build/app-q4k-ab/Lair.Bench.dll --model /root/lair-measurements/qwen25-small-cpu/models/qwen2.5-3b-instruct-q4_k_m.gguf --dotnet /usr/local/dotnet/dotnet --threads 8 --widths 64 --lengths 128,256 --kernels gemv,tiled --rounds 5 --warmup-rounds 2 --arms off,on,16 --report-name optimize-prefill-q4k-route-repair-xeon-t8

Poll timeline (read-session-output; nothing else executed on host during run):
- start ≈03:09Z (report stamp 2026-11-27T03-09-24Z); first poll saw warm arm-off running.
- ~03:18Z: invocation 4/21 started; ~03:28Z: 9/21 (rounds off/on,16 then reversed round1 confirmed live); ~03:39Z: 13/21; ~03:47Z: 19/21 dispatched.
- ~03:59–04:00Z: read-session-output returned `Background session "mvp7-repair-ab" not found.` — ANOMALY: background session buffer was closed/reaped after completion, so the final stdout line (`wrote … logs=15 warmup_logs=6 rows=… charts=2`) and per-invocation exit codes were NOT captured. Completion evidence is instead artifact-based (below). No failure stderr appeared in any polled output; script fails fast on invocation error, and all 21 logs + analysis artifacts exist with consistent timestamps → run completed normally: warmup logs mtime 03:11–03:22Z, measured logs 03:25–03:56Z, data.json/summary.md/report.html generated at 03:56 UTC (report HTML states Generated at 2026-11-27T03:56:39Z). Wall time ≈47 min.
- Remote logs dir contains exactly 15 measured files stage-ab-r{0..4}-{off,on,16}.txt and 6 warmup files stage-ab-warm{0,1}-{off,on,16}.txt; all non-empty (3838–3894 B each).

## Step 4 — download method + verification

Streaming ssh_sftp-download-file/upload-file refused on this host as expected. Base64 transport attempt was blocked by the MCP entropy sanitizer ([REDACTED:entropy:25000]) → content-based transfer used instead; harness auto-saved full tool outputs byte-exact and those were copied/sliced with SHA-256 verification against remote hashes (all 21 log slices verified 21/21 OK). summary.md required appending one trailing LF locally to match the remote byte count.

## Local artifacts — sizes + SHA-256 (verified equal to remote)

| file | bytes | sha256 |
| --- | ---: | --- |
| data.json | 78000 | a08913e48410a7170092ca54c775c15e8ce9ec7d38b7072747a809866e7181e8 |
| summary.md | 2805 | b54b5771aea61cca0c6d8edc98de1b5d25c9c616f8b87b94f8508b40165520a3 |
| report.html | 3046 | b67d05fdf3fbc08ae387e5f894db93bb53aab358e9d3911ccc33a5f6d880d054 |
| logs/stage-ab-r0-16.txt | 3843 | f258d8850aa161c73e56e35b9c834c99d178769b7226e6e1f7a0f2cc0bcc711d |
| logs/stage-ab-r0-off.txt | 3844 | 84a3f300d349afbf063432fb188e7be90f7a861fd122c509f107e7ed8adde518 |
| logs/stage-ab-r0-on.txt | 3889 | 912a3a40ec0931e2229631139d16c56c7f3cf57a9287372ebd2d985d6a2b8113 |
| logs/stage-ab-r1-16.txt | 3843 | af7d4ed88b30a6ad3e65a59dea3ffc61275b17d1ecf3b6a39729629ddda95387 |
| logs/stage-ab-r1-off.txt | 3840 | 9f2a7cee1fb5f321304b2e22e77d0cdff78a776b650b3c2a1e3e2b0998763a9c |
| logs/stage-ab-r1-on.txt | 3890 | 343b468e112c539aa04ae07b65a21d9723a63ce455b29174cf005b8be2fa5faf |
| logs/stage-ab-r2-16.txt | 3843 | b5e237fc705b5d5744e5e6f5db7a5c8f42a91d8d3a513cc00254f2dbdb82e03d |
| logs/stage-ab-r2-off.txt | 3846 | 54a732fe70077f312689e1179c8fd9ac5185adf0ec581d194dec1192f6c9e134 |
| logs/stage-ab-r2-on.txt | 3843 | 72366e39af06cbac59175d27a8bac31c00141a96600bdbe962f96ccb3aa7c70a |
| logs/stage-ab-r3-16.txt | 3848 | 16795503ad5584fe3bb93500fb884240b748457d780721da17da571ed9d24b71 |
| logs/stage-ab-r3-off.txt | 3844 | 339a91a101f8b9f54cbb5b7ce43766dea1fbf377a2f998a7e6b1b43fb13accc9 |
| logs/stage-ab-r3-on.txt | 3894 | 10aec12520861ecf27ac67821517af177a29a03d9562eae0f970a87a9c502aeb |
| logs/stage-ab-r4-16.txt | 3849 | fd59581da2a15973e4e06d7de32c5ef5ef5f314bf8bc51c35aa0d64a1b05d4ee |
| logs/stage-ab-r4-off.txt | 3843 | 804779d3ae9e8901b5b436c05c9c8450aeeec6f005be8243e79966012abdf0af |
| logs/stage-ab-r4-on.txt | 3838 | 220bc74cfa58df94b7c2cc89dcbf724ddd12362034382844806feea02414c14a |
| logs/stage-ab-warm0-16.txt | 3844 | 4e4ee58d496c49a4d07dbbd6160a390377ecd36486046964e0a1027a2fd3464e |
| logs/stage-ab-warm0-off.txt | 3845 | 0abc92c130e77a740e36fe855fef0d622b9eaa4005d26b4dc979901cc5b671c2 |
| logs/stage-ab-warm0-on.txt | 3842 | 78674f3a10e02f2cf0836a573f93e67c90c7aec006dbf13cae80e16550ba3c46 |
| logs/stage-ab-warm1-16.txt | 3848 | 724ad7bd574830a42457e1634df07467bb1e86eb4801a385f428970fc2703ffd |
| logs/stage-ab-warm1-off.txt | 3843 | 048c7eed7d525b25c5a3ffc131f8360a80636424096ffb3ddc44ec0a7da6546a |
| logs/stage-ab-warm1-on.txt | 3842 | 28bc47d06259cb137f4baa3dec0a259e8423e0778dbd984e508f3392a32f00da |

No commits or pushes were made on any repo. No remote files deleted; one file moved (stale untracked script → attic, preserved).
