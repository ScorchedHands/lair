# Prefill kernel rollout proof — gemv vs gemm on a live server, 2026-11-14

This run drives two serving servers over one pinned fixture whose dense weight matrices all carry a format the blocked
matmul executes, and changes only `engine.prefill_kernel` between them. The correctness half compares the full returned
text of an identical greedy request on both routes; the speed half measures long-prompt wall time through `/v1/completions`
with `max_tokens: 1`.

## Measurement posture

- Fixture: `models/Qwen3-0.6B-Q8_0.gguf` (all dense attention/FFN matrices are Q8_0).
- Two temporary configs held outside the repository, one server running at a time on its own port: `gemv` on
  `127.0.0.1:58411`, then `gemm` on `127.0.0.1:58412`. Both configs otherwise identical: `max_context_tokens: 4096`,
  `kv_blocks: 256`, `engine.block_size: 16`, `engine.max_batch_tokens: 512`, `engine.threads: 0`, model
  `placement.thread_quota: 8`, no core pinning. The monitor reported the same resolved posture on both servers —
  `metrics_worker_threads = 8`, `metrics_thread_quota = 8`, `metrics_kv_capacity_blocks = 256`,
  `metrics_kv_pool_mb = 896`.
- Which route actually served each request is read from the live feed, not assumed: `metrics_prefill_kernel` reported
  `gemv` on port 58411 and `gemm` on port 58412 across every snapshot that carried model facts, with
  `metrics_weight_formats = Q8_0` on both.
- Long-prompt shape: a fresh GUID nonce, the sentence below repeated 96 times, and one `" 0"` pad unit. Each timed request
  used its own nonce so no prompt was reused through the prefix cache; the captures confirm `metrics_cache_hit_sequences_total = 0`
  and `metrics_cache_skipped_tokens_total = 0`, with `metrics_prefill_tokens_forwarded_total` matching the summed usage of
  all three requests on each server exactly (3547 for gemv, 3545 for gemm).

```text
The quick brown fox jumps over the lazy dog while a patient analyst records every measured step. 
```

- Peak working set is read from `metrics_process_working_set_mb` in `/monitor/stream` snapshots captured live during the
  long requests (curl `-N` attached to each server's stream for the whole measurement window).

## Exact-stream correctness

One identical request on both servers — deterministic prompt, `temperature: 0`, `max_tokens: 32`:

```text
Once upon a time, a careful librarian catalogued every book on a single shelf. She wrote the first three entries as: 1)
```

| Server | Returned text | finish_reason | usage (prompt / completion / total) | SHA-256 of returned text |
|---|---|---|---|---|
| gemv :58411 | ` 1, 2) 2, 3) 3, 4) 4, 5) 5, 6) 6` | `length` | 28 / 32 / 60 | `ED4E24F7…D8246E5` |
| gemm :58412 | ` 1, 2) 2, 3) 3, 4) 4, 5) 5, 6) 6` | `length` | 28 / 32 / 60 | `ED4E24F7…D8246E5` |

Full hashes: `ED4E24F7798615B60EF00F4B1F72611FE3DB0B52A52EC09EF39F1468DD8246E5`, identical on both routes. The 32-token
greedy stream sampled after this prefill is character-identical between the batched-row route and the dense-matrix route,
and the usage tallies match exactly — selecting `gemm` does not move the arithmetic on a file whose formats it runs.

## Long-prompt speed

Each request: fresh nonce prompt, `temperature: 0`, `max_tokens: 1`. Wall seconds are measured around the HTTP call;
prompt tokens come from server-reported usage.

| Kernel | Run | Prompt tokens | Wall seconds | Prefill tok/s | Peak working set (live) |
|---|---|---:|---:|---:|---:|
| gemv | 1 | 1759 | 95.07 | 18.50 | 1748.5 MB |
| gemv | 2 | 1760 | 96.50 | 18.24 | — (same process) |
| gemm | 1 | 1762 | 102.31 | 17.22 | 1738.8 MB |
| gemm | 2 | 1755 | 105.22 | 16.68 | — (same process) |

Combined over each kernel's two timed runs: `gemv` forwarded `(1759 + 1760)` tokens in `191.57 s` = **18.37 tok/s**;
`gemm` forwarded `(1762 + 1755)` tokens in `207.53 s` = **16.95 tok/s** — a `-7.7%` prefill throughput change on this
prompt at this posture. Peak working set moved the other way, from `1748.5 MB` to `1738.8 MB`, a `-0.6%` change within
snapshot noise: both routes hold the same mapped weights and the same 256-block pool (the captures show KV usage climbing
to 220 of 256 blocks on gemv and 221 on gemm as each long window wrote its blocks, so the peak was taken during live long
requests rather than at idle).

## Reading

At this prompt length the serving engine splits a ~1760-token window into `ceil(1760 / 32) = 55` weight sub-batches on
both routes — the resolved row width is unchanged by the kernel selection, so neither run pays extra or fewer weight
traversals. The dense-matrix route amortizes dequantization per tile and still measured slower here: at Q8_0 the blocked
matmul's unpacking saving is small because one byte per element already decodes cheaply, while its fixed 8-row tile shape
adds accumulator work that the batched GEMV pass does not. The bench tables written beside this report on the same fixture
at a fixed row width point slightly the other way — `gemm` ahead of `gemv` at every measured length (`10936 ms` vs
`12081 ms` at 512, `37891 ms` vs `38125 ms` at 1024, `155944 ms` vs `161763 ms` at 2048) — while this live-server run has
it `7.7%` behind. Both sides are single-pass measurements on a shared box under different postures (direct engine call in
one process versus scheduled serving through HTTP), so the speed half reads as "no consistent throughput win for either
route at Q8_0" rather than a settled ordering; the exact-stream equality is the strong result, and it held on every
request pair measured.

## Caveats

- `max_tokens: 1` means each timed request also decodes one generated token after prefill plus HTTP overhead; both kernels
  pay the same accounting cost, so the comparison holds even though absolute tok/s is slightly pessimistic.
- Fresh nonces shift prompt length by a few tokens between runs (1755–1762); rates are computed per request from each
  server's own reported usage rather than assuming equal lengths across kernels.
- Each kernel was measured in its own server process, one at a time; the box ran no other benchmark alongside these two
  servers but is not otherwise isolated, and each length got a single timed pass — treat the speed delta as directional,
  with the exact-stream equality as the strong result.
- The working-set figures are serving-process residency sampled at snapshot cadence (~2 s): they include mapped weights,
  the KV pool reservation, and window scratch together, so they bound what a kernel switch costs in memory rather than
  isolating a per-kernel allocation.
