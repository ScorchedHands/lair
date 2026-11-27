# Provenance — 2026-11-27T19-56-50Z-optimize-prefill-q4k-route-wide-xeon-t8-highpower

Background SSH session `route_ab_highpower` on profile `lair-lab`, launched from `/root/lair-measurements/qwen25-small-cpu/repo` (HEAD verified as `016dd9c` before launch; host repo carries uncommitted bench-seam files for the `q4k-widebatch` verb, which do not affect `prefill-profile`). Host idle before launch (`ps -eo pid,etime,cmd | grep stage-ab` empty).

Command as run:

```
python3 scripts/run-xeon-prefill-stage-ab.py \
  --app /root/lair-measurements/qwen25-small-cpu/build/app-posture-2026-11-27T06-53-30Z/Lair.Bench.dll \
  --model /root/lair-measurements/qwen25-small-cpu/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --dotnet /usr/local/dotnet/dotnet \
  --arms off,32 --lengths 256 --rounds 9 --warmup-rounds 3 \
  --report-name optimize-prefill-q4k-route-wide-xeon-t8-highpower
```

Arm template (runner dry-run verified earlier): `<dotnet> <app> prefill-profile <model> --threads 8 --lengths 256 --widths 64 --kernel gemv,tiled --reps 1 --stage-profile --attention-batch-dots on --matmul-route on --q4k-matmul-route off|32`. Arms ran strictly sequentially with odd-round rotation; run completed normally (~37 minutes wall time for 18 invocations).

## Remote artifact hashes (sha256sum on the measurement host)
- `data.json` `f6ba443395d1e6b45fcb72740cbb88dd0135cf967f582ac6e23eefb06c931598`
- `summary.md` (full version) `97f335f1e3b9aee4656652da499b54aec7b98a93c10b1efda40591455185a278`
- `report.html` `e5f001adf3c5432043284e74488e75c60e058f5d00df54e8e460b9d826305716`
- Raw logs remain only under `…/reports/2026-11-27T19-56-50Z-optimize-prefill-q4k-route-wide-xeon-t8-highpower/logs/`; they were not copied into this working tree, so this directory is a headline extract.

## Anomalies
None: no refusal lines, every arm produced stage-profile tables, runner exited cleanly. Nothing was deleted on the host; earlier scratch (`/root/lair-tools/sweep.*`, prior run logs) remains deliberately.
