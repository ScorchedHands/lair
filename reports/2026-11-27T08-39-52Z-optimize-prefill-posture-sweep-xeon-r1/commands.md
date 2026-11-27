# Provenance — 2026-11-27T08-39-52Z-optimize-prefill-posture-sweep-xeon-r1

Host: profile `lair-lab` (root@192.168.33.53), Intel Xeon E5-2683 v4, 2×16 physical cores / 64 logical, AVX2+FMA only (no AVX-512 per `/proc/cpuinfo` flags), NUMA node0 = CPUs 0-15 + siblings 32-47, governor `performance`. Host idle before the run (`ps -eo … | grep dotnet|python3?|Lair` returned nothing).

## Acquisition and build provenance
- llama.cpp: prebuilt release asset for tag `b12032` installed by the measurement worker into disk-backed `/root/lair-tools/llama-b12032/` (dynamic CPU backend set present, e.g. `libggml-cpu-haswell.so`). Prior copy under `/tmp/lair-tools/…` had disappeared with tmpfs.
- LAIR: `/root/lair-measurements/qwen25-small-cpu/repo` fast-forwarded to pushed HEAD `016dd9c`; bench app published fresh as `bench/Lair.Bench` only, into `/root/lair-measurements/qwen25-small-cpu/build/app-posture-2026-11-27T06-53-30Z/` (CPU source had changed since the previously published binary).
- Model: `/root/lair-measurements/qwen25-small-cpu/models/qwen2.5-3b-instruct-q4_k_m.gguf`.

## Runner and command shape
Script `scripts/run-xeon-prefill-posture-sweep.py` (this repo, committed with this report). Remote copy verified byte-equal: 35,432 bytes, sha256 `FBB798…226B` both sides; remote `--dry-run` printed the same schedule that ran.

Warmup arms (×2 rounds) then measured arms (×2 rounds, order rotated on odd rounds), each invocation logged separately:
- LAIR arm template: `/usr/local/dotnet/dotnet /root/lair-measurements/qwen25-small-cpu/build/app-posture-2026-11-27T06-53-30Z/Lair.Bench.dll prefill-profile <model> --threads {8|16|32} --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 1`
- llama arm template: `/root/lair-tools/llama-b12032/llama-bench -m <model> -ngl 0 -t {8|16|32} -p 128,256 -n 0 -r 1`

Run launched as a background SSH session on the host from `/root/lair-measurements/qwen25-small-cpu/repo`; finished cleanly at 09:01 host time with `data.json`, `summary.md`, `report.html` and 24 raw logs (`posture-warm{0,1}-*`, `posture-r{0,1}-*`).

## Artifact hashes (sha256sum on the measurement host; local copies verified identical)
- `data.json` `875c44fbc173119b3f1528cba69ea049735c8a2185f028a0f60a2579d9e2345e` — committed here, byte-identical to the host copy.
- `report.html` `a64481d6d1ab11ae5f0b3af39f541b92be6d20c1db5cf69b2ecbce8392cacc14` — committed here, byte-identical to the host copy.
- `summary.md`: this directory holds a main-authored headline extract; the full 4,948-byte run summary stays on the measurement host with hash `e5e80ddf1e73aba1449329d54497e9a978aa21be099a5af256e632db15c1804e`.
- Raw logs (24 files) remain only on the measurement host under `…/reports/2026-11-27T08-39-52Z-optimize-prefill-posture-sweep-xeon-r1/logs/`; they were not copied into this working tree.

## Anomalies
- Transfer of the runner to the host required 29 marker-chunked parts with per-part hash gates (streaming SFTP file tools are refused on the Windows client); one part drifted once and was retried byte-exact before assembly, and the assembled file hashed equal to local.
- Worker scratch fragments from failed upload attempts remain at `/root/lair-tools/` (`sweep.b64`, `sweep.p1`, `sweep.s1-6`, `sweep.c04_partial_probe`, `sweep.c04_retry1_probe`, `sweep.c04_probe2`) — left in place deliberately, nothing was deleted on the host.
