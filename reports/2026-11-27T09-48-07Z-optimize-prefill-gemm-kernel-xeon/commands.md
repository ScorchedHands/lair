# Provenance — 2026-11-27T09-48-07Z-optimize-prefill-gemm-kernel-xeon

Host/build/model identical to the two posture sweeps (profile `lair-lab`, Xeon E5-2683 v4, AVX2-only, repo HEAD `016dd9c`, app `/root/lair-measurements/qwen25-small-cpu/build/app-posture-2026-11-27T06-53-30Z/Lair.Bench.dll`, llama at `/root/lair-tools/llama-b12032/llama-bench`).

## Command as run (background SSH session from the repo dir, after r2 exited)

```
python3 scripts/run-xeon-prefill-posture-sweep.py --root /root/lair-measurements/qwen25-small-cpu \
  --app /root/lair-measurements/qwen25-small-cpu/build/app-posture-2026-11-27T06-53-30Z/Lair.Bench.dll \
  --model /root/lair-measurements/qwen25-small-cpu/models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --dotnet /usr/local/dotnet/dotnet --llama /root/lair-tools/llama-b12032/llama-bench \
  --lengths 128,256 --widths 64 --kernels gemm --rounds 2 --warmup-rounds 1 \
  --arms lair:8,lair:16,lair:32,llama:8,llama:16,llama:32 \
  --report-name optimize-prefill-gemm-kernel-xeon
```

All arms exited zero; no `gemm` capability refusal line appeared in any log.

## Remote artifact hashes (sha256sum on the measurement host)
- `data.json` `13cfd6e3d15de20641591cb561e016f10fc7129f89dde5797f6903fd8bb54718`
- `summary.md` (full version) `a2e55ab7a3a2c843ab98394b9b92581dc44950d693c4559799aefeb11500c43c`
- Raw logs remain only at `…/reports/2026-11-27T09-48-07Z-optimize-prefill-gemm-kernel-xeon/logs/`.

## Anomalies
- A first attempt at this probe was launched as a chained background session whose guard loop used `pgrep -f 'kernels gemv,tiled --rounds 4'`; the pattern matched that session's own command line, so it idled instead of starting. That session was closed before any measurement ran and relaunched directly — no data lost, nothing deleted on the host.
- One fit in this run is engine-flagged `(UNRELIABLE: measured costs are not separable into the two terms on this run)`; its bandwidth figure is quoted verbatim in `summary.md` with that flag intact.
