# NOTES

## Worker verification policy (owner 2026-11-02)

- Per-slice: run only the targeted category tests (`--filter Category=<new>`) + build.
- Full `dotnet test` suite: once at phase close (before ship), not per slice.

## Documentation maintenance (owner 2026-11-16)

- User-facing surface: `docs/README.md` (overview + operation), `docs/reference/CLI.md` (every command, flag, output, exit code), `docs/reference/CONFIG.md` (every `lair.yaml` key, default, constraint, mutation class). Root `README.md` stays the repository entry point and links into `docs/`.
- Update the docs at every phase ship, in the same change as the ship: new commands/flags go to the CLI reference; new or changed keys, defaults, constraints, or mutation classes go to the config reference; new features go to the base page and its feature sections.
- Unshipped features are documented with an explicit status label (`in progress` / `planned` / design tentative) in the base page's Upcoming table and in the two reference pages' Planned sections. When a phase ships, move its content out of those sections into the shipped ones — never leave a shipped feature listed as upcoming.
- Docs describe the branch they sit on. Anything verified against source beats anything copied from another doc: `lair.example.yaml`, root `README.md`, and `DEVELOP.md` are part of the documented surface and must match what the loader and CLI actually accept.

## Active

- Building MVP3 on `phase/mvp3`. Slices: B1 pipeline+shell [done], B2 event schema+stats endpoint, B3 Monitor SPA wiring, B4 Chat tab. Manual Monitor + Manual Chat UI at phase close.

## Xeon Q4_K benchmark continuation (2026-11-27)

```text
Continue LAIR `phase/mvp7` from worktree `C:\source\gitlab.scorchedhands.com\personal\LAIR\.agent-worktrees\mvp7`. Current pushed HEAD is expected to be `e3f001f docs(reports): record xeon cpu prefill evidence`; local source/test/bench commit before it was `ffea76a perf(cpu): tune dense prefill routing seams`. Targeted Release gates passed 742/742 for `Category=Kernel|Backend|DenseWeightFusion|CpuTuning|Profile&Category!=Cuda`, but the full non-CUDA suite previously hung and is a caveat. Next step: verify Xeon host `192.168.33.53` reachability (TCP port 22, ping/ARP). If reachable, delegate one foreground implementation-worker to run Q4_K dense matmul-route A/B stage profiles on profile `lair-lab`: remote root `/root/lair-measurements/qwen25-small-cpu`, model `/root/lair-measurements/qwen25-small-cpu/models/qwen2.5-3b-instruct-q4_k_m.gguf`, dotnet absolute path `/usr/local/dotnet/dotnet`. Prefer remote checkout under `/root/lair-measurements/qwen25-small-cpu/repo` from branch `phase/mvp7`; if anonymous clone/auth fails, transfer a clean local archive via short-lived HTTP. Publish only bench app with `/usr/local/dotnet/dotnet publish bench/Lair.Bench -c Release -o <disk-backed build/app dir>`. Run variants using repo script `scripts/run-xeon-prefill-stage-profile.py` with explicit model/app paths, kernels `gemv,tiled`, lengths `128,256`, threads `8`, widths `64`, rounds 3 or 4: baseline `--q4k-matmul-route off`; candidate width 8 (`on`) and optionally width 16. Download artifacts/raw logs into a new local report dir under `.agent-worktrees\mvp7\reports\<UTC-startstamp>-optimize-prefill-q4k-route-ab-xeon-t8`. Do not fabricate evidence if host/tooling fails.
```

- Host state at 2026-11-27T01:10Z: TCP port 22 reachable, ping ~1ms; remote has git, python3.13, `/root/.nuget` cache, dotnet SDK 10.0.404.
- A/B collected at `reports/2026-11-27T01-16-57Z-optimize-prefill-q4k-route-ab-xeon-t8` from fresh clone of `phase/mvp7@e3f001f` (remote `/root/lair-measurements/qwen25-small-cpu/repo`) + publish to `build/app-q4k-ab`. Kernels gemv,tiled; lengths 128,256; t8 w64; rounds 3.
- total_ms medians vs baseline(off): gemv@128 −9.1% (w8) / −10.0% (w16); gemv@256 −4.7% / −7.7%; tiled@256 −7.3% / −2.5%; tiled@128 +6.0% at w8 but inside its 22% noise band, −0.4% at w16. Mean-of-medians −3.8% (w8), −5.1% (w16).
- Caveat: noisy spreads (>20%) on baseline gemv@128 and route-on-w8 tiled combos; rounds=4 re-pair not run — rerun before any promotion decision. Aborted remote dir `reports/2026-11-27T01-22-48Z-*` left on host, excluded from evidence.
