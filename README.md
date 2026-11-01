# LAIR — Local AI Inference Runtime

A C#/.NET inference runtime that serves GGUF LLMs locally behind an OpenAI-compatible HTTP API, with a built-in chat REPL. Dense models (Llama/Qwen-class), CPU-first AVX2 kernels, paged KV cache, GGUF-embedded tokenizer and chat template.

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
| `POST /v1/chat/completions` | JSON or SSE streaming (`"stream": true`) |
| `POST /v1/completions` | legacy text completions, JSON or SSE |
| `GET /v1/models` | model list including `context_length` and quant metadata |
| `GET /healthz` | liveness + model readiness |

Any OpenAI SDK works by pointing `base_url` at `http://127.0.0.1:8080/v1`. Errors use the OpenAI error shape with codes `invalid_request`, `model_not_found`, `context_length_exceeded`, `engine_overloaded`, `internal`. Every response carries `x-lair-request-id`.

The engine serves one generation at a time; a second concurrent request receives `503 engine_overloaded`.

## REPL

`/exit` or `/quit` to leave, `/reset` to clear the conversation. Ctrl+C interrupts the current generation. Overlong turns are reported as a single-line error after an automatic trim-and-retry.

## Configuration

One YAML file (`lair.example.yaml` shows every knob): server bind/port, engine threads/block size, and exactly one model with path, name, type, `max_context_tokens`, `kv_blocks`, and optional placement policy. Restart the process to pick up config edits.
