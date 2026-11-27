# Provenance — 2026-11-27T15-25-26Z-optimize-prefill-q4k-route-wide-xeon-t8

Background SSH session `route_wide_ab` on profile `lair-lab`, launched from `/root/lair-measurements/qwen25-small-cpu/repo` (HEAD verified earlier this session as `016dd9c`; host repo carries uncommitted bench-seam files for the `q4k-widebatch` verb, which do not affect `prefill-profile`). Host idle before launch (`ps -eo pid,etime,cmd | grep stage-ab` empty).

Command as run:

```
python3 scripts/run-xeon-prefill-stage-ab.py \
  --app /root/lair-measurements/qwen25-small-cpu/build/app-posture-2026-11-27T06-53-30Z/Lair.Bench.dll \
  --model /root/lair-measurements/qwen25-small-cpu/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --dotnet /usr/local/dotnet/dotnet \
  --arms off,32,64 --rounds 5 --warmup-rounds 2 \
  --report-name optimize-prefill-q4k-route-wide-xeon-t8
```

Arm template (from the runner's own dry-run schedule, verified before launching): `<dotnet> <app> prefill-profile <model> --threads 8 --lengths 128,256 --widths 64 --kernel gemv,tiled --reps 1 --stage-profile --attention-batch-dots on --matmul-route on --q4k-matmul-route off|32|64`. Arms ran strictly sequentially; measured order rotated on odd rounds. Run completed with `data.json`, `summary.md`, `report.html` and per-arm raw logs (exit evidence in the runner's own log lines).

## Remote artifact hashes (sha256sum on the measurement host)
- `data.json` `a1d1ecf494fa8cbc22dfd900d3aeec93d0d1f97f422831fb7bc508d669179145`
- `summary.md` (full version) `3df7d51e75039fc95c62e7e55e47a8ee145a17715cb4a941e90f7c64833facff`
- `report.html` `fe362f261e200669242bd75a4b98ec2ef90246200dbcfe35116b448f2b990589`
- Raw logs remain only under `…/reports/2026-11-27T15-25-26Z-optimize-prefill-q4k-route-wide-xeon-t8/logs/`; they were not copied into this working tree, so this directory is a headline extract.

## Anomalies
None observed: no refusal lines, all arms produced stage-profile tables, and the runner completed normally (~39 minutes wall time for 21 invocations). Nothing was deleted on the host; scratch from earlier runs (`/root/lair-tools/sweep.*`, worker helpers inside `…T10-05-02Z-optimize-prefill-rowwidth-xeon`) remains in place deliberately.
