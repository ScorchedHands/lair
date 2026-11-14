# Prefill kernel speed proof - 2026-11-13

This run compares the existing `gemv` prefill route with the row-tiled attention route on a live serving server. The
fixture, context profile, KV budget, thread quota, block size, and max batch width are held constant; only
`engine.prefill_kernel` changes between runs. Each request is sent to `/v1/completions` with `max_tokens: 1`, so the
server reports exact prompt token usage while keeping HTTP measurement simple.

## Measurement posture

- Fixture: `models/Llama-3.2-1B-Instruct-IQ4_XS.gguf`
- Serve profile: `max_context_tokens: 4096`, `kv_blocks: 256`, `engine.block_size: 16`, `engine.max_batch_tokens: 512`,
  `engine.threads: 0`, and model `placement.thread_quota: 8` with no core pinning.
- Kernel values tested through temporary configs: `gemv` on port `58391`, then `tiled` on port `58392`. One server ran at a time.
- Prompt shape: a random GUID nonce, the sentence below repeated 96 times, and one `" 0"` pad unit; final prompt strings were accepted only when `/v1/completions` usage reported exactly `prompt_tokens = 1751`:

```text
The quick brown fox jumps over the lazy dog while a patient analyst records every measured step. 
```

- Each timed request used a fresh nonce, so the exact long sequence was not reused through prefix cache during measurement.
- Peak working set is read from `metrics_process_working_set_mb` snapshots collected live on `/monitor/stream`.

## Results

| Kernel | Prompt tokens | Wall seconds | Prefill tok/s | Peak working set |
|---|---:|---:|---:|---:|
| gemv run 1 | 1751 | 74.9243887 | 23.37 | 1353.8 MB |
| gemv run 2 | 1751 | 75.6483948 | 23.15 | 1353.8 MB |
| tiled run 1 | 1751 | 69.7240270 | 25.11 | 1222.8 MB |
| tiled run 2 | 1751 | 71.5923544 | 24.46 | 1222.8 MB |

Combined over the two exact-token runs for each kernel: `gemv` averaged `23.26` tok/s and `tiled` averaged `24.78`
tok/s, a `+6.5%` prefill throughput gain with a `6.1%` wall-time reduction. Peak working set dropped from `1353.8 MB`
to `1222.8 MB`.

## Reading

The resolved prompt row width is unchanged between the two runs, so both engines split `1751` tokens into `56` weight
sub-batches (`ceil(1751 / 32)`). The weight-traversal count therefore does not explain the gain. The same engine posture
has a fitted attention crossover floor at length `>= 1081` in `prefill-rowwidth-2026-11-13.md`, so this prompt is already
in the region where quadratic attention dominates measured prefill cost. A faster wall time and lower peak working set on
the tiled route confirm that the attention-side term fell when row-tiled attention became active.

## Caveat

`max_tokens: 1` means each request also decodes one generated token after prefill, so the computed prompt-token rate is
slightly pessimistic by roughly one decode step plus HTTP overhead; both kernels pay the same accounting cost. The peak
working set was captured during the live long-prompt requests, not from idle snapshots only.
