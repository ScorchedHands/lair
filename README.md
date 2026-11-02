# LAIR — Local AI Inference Runtime

A C#/.NET inference runtime that serves GGUF LLMs locally behind an OpenAI-compatible HTTP API, with a built-in chat REPL and a live metrics monitor. Dense models (Llama/Qwen-class), CPU-first AVX2 kernels, continuous batching with chunked prefill, paged KV cache with prefix reuse, config hot-reload, GGUF-embedded tokenizer and chat template.

## Requirements

- .NET SDK 10.0.400+ (x64)
- A GGUF model file (F16, Q8_0, or Q4_K quants), e.g. from Hugging Face

## Quick start

```powershell
# 1. Copy the example config and point it at your model
copy lair.example.yaml lair.yaml
# edit lair.yaml: model path, name, context, kv_blocks, port

# 2. Serve the OpenAI-compatible API (binds 127.0.0.1 by default)
dotnet run --project src/Lair.Host -- serve -c lair.yaml

# 3. Or talk to the model directly in a terminal REPL
dotnet run --project src/Lair.Host -- chat -c lair.yaml
```

## API

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | JSON or SSE streaming (`"stream": true`); supports `tools`/`tool_choice`, `logprobs`/`top_logprobs`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `seed` |
| `POST /v1/completions` | legacy text completions, JSON or SSE |
| `GET /v1/models` | model list including `context_length` and quant metadata |
| `GET /monitor/stream` | SSE stream of engine metrics snapshots |
| `GET /healthz` | liveness + model readiness |

Any OpenAI SDK works by pointing `base_url` at `http://127.0.0.1:8080/v1`. Errors use the OpenAI error shape with codes `invalid_request`, `model_not_found`, `context_length_exceeded`, `engine_overloaded`, `internal`. Every response carries `x-lair-request-id`.

Requests run concurrently through shared batch steps; `503 engine_overloaded` surfaces only when engine capacity is exhausted. Chat requests that declare tools get schema injection and a fenced-JSON tool-call detector; recognized calls appear as `tool_calls` (execution is client-side).

## REPL

`/exit` or `/quit` to leave, `/reset` to clear the conversation. Ctrl+C interrupts the current generation. Overlong turns are reported as a single-line error after an automatic trim-and-retry.

## Monitor

```powershell
dotnet run --project src/Lair.Host -- monitor            # default http://127.0.0.1:8080
dotnet run --project src/Lair.Host -- monitor -u http://127.0.0.1:8080 --interval 1000
```

Live table per model: tok/s, step time, batch occupancy, cache-hit rate, KV utilization, active sequences.

## Configuration

One YAML file (`lair.example.yaml` shows every knob): server bind/port, engine `threads` and `max_batch_tokens`, block size, and exactly one model with path, name, type, `max_context_tokens`, `kv_blocks`, and optional placement policy (`core_set`, `thread_quota`).

A running server watches the config file and applies edits automatically: batch width, thread quota, and KV budget take effect live; context length applies to new requests; block size, model path, and quant geometry are refused with a message telling you to restart. Invalid edits are rejected without touching the running engine.
