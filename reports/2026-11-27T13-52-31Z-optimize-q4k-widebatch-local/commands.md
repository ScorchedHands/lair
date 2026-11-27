# Provenance — 2026-11-27T13-52-31Z-optimize-q4k-widebatch-local

Produced on the developer box (local AVX2 host), not on `lair-lab`, by the measurement worker that built the `q4k-widebatch` seam. Directory stamp is taken from the earliest committed log's write time (`2026-11-27T13:52:31Z`); the runs themselves are identified inside each log's own header line rather than by wall-clock files.

## Seam under measurement
`bench/Lair.Bench q4k-widebatch` (new verb in this change) opens three backends over one synthetic pinned Q4_K fixture — auto posture, forced streaming fallback (`MaxScratchCols = MaxScratchColsMin`), and the unpromoted tile executor (`MatmulRouteBatchWidthQ4K = 8`) — plus an optional chain-group-1 diagnostic arm. Arms share two untimed settle passes and rotated-order timed rounds with a true median and `--reps >= 3`, after an inter-arm bit-parity pass; earlier drafts of this seam measured arms sequentially, which produced a JIT-warmth ordering bias (a count=1 artifact) that the rewritten core removed.

## Invocations reconstructed from log headers
The exact shell lines were not preserved by the worker; each arm set's own header line is committed verbatim in the logs below, so these reconstructions are stated as reconstructions:
- `dotnet run --project bench/Lair.Bench -c Release q4k-widebatch --threads 1 --reps <n> --widebatch-colblock 512` → `q4k-wb-clean-t1-c512.log` (candidate arm present in that build)
- same with `--widebatch-colblock 1024` → `q4k-wb-clean-t1-c1024.log`
- `dotnet run --project bench/Lair.Bench -c Release q4k-widebatch --threads 8 --reps 5 --counts 8,16,32,64 --widebatch-colblock 512` → `q4k-wb-clean-t8-c512.log`

## Committed raw logs (sha256)
- `q4k-wb-clean-t1-c512.log` `914cd8110dc14109ee83ecbca97d754b6bdba2c0b0b39164eb1286e917a857d5`
- `q4k-wb-clean-t1-c1024.log` `fb08ac9e8df652eac49d147b8b81fc0ab026a4676eb24c0e1b41812d9ab5a134`
- `q4k-wb-clean-t8-c512.log` `5ac505dbf1160e6aa9fd24cf1b9378ebd0942735eaba472dddf12533fa87998c`

## Candidate disposition and gates
The measured-worse `WidebatchColBlockQ4K` seam (kernel, tuning constants, tests, bench arm) was reverted out of `src/`, `tests/`, and both bench files in the same change that committed this report; a grep across those trees shows no remaining reference. Gates re-run after the revert: `dotnet build Lair.sln` → 0 errors; `dotnet test Lair.sln -c Release --filter "Category=DenseWeightFusion|Kernel|CpuTuning|Profile&Category!=Cuda"` → 479 passed, 0 failed. Unfiltered `dotnet test` was not run (known hanging defect recorded in the plan backlog). No Xeon run was taken from this candidate because local numbers moved only in the negative direction.
