# Provenance — 2026-11-27T09-03-56Z-optimize-prefill-posture-sweep-xeon-r2

Chained repeat of the r1 sweep on the same host/build/model, launched directly by the main session as a background SSH session from `/root/lair-measurements/qwen25-small-cpu/repo` after confirming r1 had completed and no other process held the host. Runner: `scripts/run-xeon-prefill-posture-sweep.py`, invoked with

```
python3 scripts/run-xeon-prefill-posture-sweep.py --root /root/lair-measurements/qwen25-small-cpu \
  --app /root/lair-measurements/qwen25-small-cpu/build/app-posture-2026-11-27T06-53-30Z/Lair.Bench.dll \
  --model /root/lair-measurements/qwen25-small-cpu/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --dotnet /usr/local/dotnet/dotnet --llama /root/lair-tools/llama-b12032/llama-bench \
  --lengths 128,256 --widths 64 --kernels gemv,tiled --rounds 4 --warmup-rounds 2 \
  --arms lair:8,lair:16,lair:32,llama:8,llama:16,llama:32 \
  --report-name optimize-prefill-posture-sweep-xeon-r2
```

Repo HEAD at run time `016dd9c`. Arms ran strictly sequentially; measured arm order rotated on odd rounds. Completed with exit success and wrote `data.json`, `summary.md`, `report.html` plus raw logs (`posture-warm{0,1}-*`, `posture-r{0..3}-*`).

## Remote artifact hashes (sha256sum on the measurement host)
- `data.json` `c705b6b56f32f89478d14c99b7378d4a4b7fc3f99412327c0b413144acf0cf44`
- `summary.md` (full version) `cb3833d64f02194c1a6997c8c2523fa860aa17a39aeef7052aa65e07b987913d`
- Raw logs remain only on the measurement host under `…/reports/2026-11-27T09-03-56Z-optimize-prefill-posture-sweep-xeon-r2/logs/`; not copied into this tree, so this directory is a headline extract.

## Anomalies
None for the run itself: every arm exited zero and no refusal lines appeared. LAIR spread at threads=32 (up to 43.5%) is larger than at lower postures — noted in `summary.md` rather than treated as an error, since llama arms under the same posture stayed within 16%.
