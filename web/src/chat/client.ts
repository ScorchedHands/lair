// Hand-rolled OpenAI-compatible chat client for /v1/chat/completions (stream:true). No SDKs, no fetch wrappers:
// one POST with an AbortController signal, then the byte stream is decoded and split into SSE frames by hand.
//
// Wire shape this parses (lair's serializer emits exactly this):
//   data: {"choices":[{"index":0,"delta":{"role":"assistant"}}], ...}\n\n     role frame first
//   data: {... "delta":{"content":"piece"}}\n\n                                content deltas, in order
//   data: {... "delta":{"tool_calls":[{id,type,function:{name,arguments}}]}}  complete-call frames (lair) or
//         index-keyed argument fragments (OpenAI-style) — both assemble through the same accumulator
//   data: {... "choices":[{"index":0,"delta":{},"finish_reason":"eos"}]}\n\n  terminal chunk
//   data: [DONE]\n\n
// Frames with no choices[] (keepalives/comments) are skipped. A stream that ends without [DONE] is treated as done.

export const CHAT_PATH = "/v1/chat/completions";
export const MODELS_PATH = "/v1/models";

// Sampling knobs the shell's meta line advertises; every request carries them explicitly so what the UI states and
// what the engine runs cannot drift apart.
export const REQUEST_TEMPERATURE = 0.7;
export const REQUEST_MAX_TOKENS = 256;

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface ToolCallView {
  name: string;
  argumentsText: string; // pretty-printed compact JSON, or the raw string when it is not valid JSON
}

// A single assistant turn as rendered by the chat tab.
export interface ChatMessageView {
  kind: "assistant" | "error";
  text: string;
  streaming: boolean;
  stopped: boolean; // mid-stream abort: partial text stays, caret is replaced by a stopped marker
  toolCalls: ToolCallView[];
}

// A model entry the selector may offer. Entries explicitly typed embed are excluded at fetch time.
export interface ChatModelOption {
  id: string;
  quant: string | null;
  contextLength: number;
  type: string | null; // absent on generic OpenAI-shaped listings; treated as chat-compatible
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ---- models ---------------------------------------------------------------------------------------

export async function fetchChatModels(signal?: AbortSignal): Promise<ChatModelOption[]> {
  const response = await fetch(MODELS_PATH, signal ? { signal } : undefined);
  if (!response.ok) {
    throw new ApiError(`model listing failed with HTTP ${response.status}`, "models_unavailable", response.status);
  }
  const body = (await response.json()) as { data?: unknown };
  const entries: ChatModelOption[] = [];
  for (const raw of Array.isArray(body.data) ? body.data : []) {
    if (raw === null || typeof raw !== "object") continue;
    const entry = raw as Record<string, unknown>;
    if (typeof entry.id !== "string" || entry.id.length === 0) continue;
    const type = typeof entry.type === "string" ? entry.type : null;
    // Embed-type entries are explicitly excluded; everything else is chat-compatible on this server.
    if (type === "embed") continue;
    entries.push({
      id: entry.id,
      quant: typeof entry.quant === "string" ? entry.quant : null,
      contextLength: typeof entry.context_length === "number" ? entry.context_length : 0,
      type,
    });
  }
  return entries;
}

// ---- SSE framing ------------------------------------------------------------------------------------

type Frame = { kind: "data"; payload: string } | { kind: "done" };

function findFrameBoundary(buffer: string): { start: number; end: number } | null {
  const lf = buffer.indexOf("\n\n");
  const crlf = buffer.indexOf("\r\n\r\n");
  if (crlf !== -1 && (lf === -1 || crlf < lf)) return { start: crlf, end: crlf + 4 };
  if (lf !== -1) return { start: lf, end: lf + 2 };
  return null;
}

// Pulls one complete frame out of the accumulated buffer. rest === null means no full frame is buffered yet; a
// found boundary always advances rest, even when the frame carries nothing to consume (keepalives/comments).
// A frame ends at a blank line; its payload is built from the non-empty `data:` lines only (OpenAI-style
// keepalives/comments carry none). A data line that fails JSON.parse is dropped, not fatal.
function takeFrame(buffer: string): { rest: string | null; frame: Frame | null } {
  const boundary = findFrameBoundary(buffer);
  if (boundary === null) return { rest: null, frame: null };

  let payload = "";
  for (const line of buffer.slice(0, boundary.start).split(/\r?\n/)) {
    if (!line.startsWith("data:")) continue;
    const value = line.slice(5).replace(/^ /, "");
    payload += payload.length > 0 ? `\n${value}` : value;
  }

  let frame: Frame | null = null;
  if (payload === "[DONE]") {
    frame = { kind: "done" };
  } else if (payload.length > 0) {
    try {
      JSON.parse(payload);
      frame = { kind: "data", payload };
    } catch {
      // malformed data line: skip it rather than break the stream
    }
  }

  return { rest: buffer.slice(boundary.end), frame };
}

// ---- tool_calls delta assembly ----------------------------------------------------------------------

interface ToolCallFragment {
  index?: number;
  id?: string | null;
  type?: string | null;
  function?: { name?: string | null; arguments?: string | null } | null;
}

class ToolCallAccumulator {
  private readonly calls = new Map<number, { name: string; args: string }>();
  // Lair emits complete-call frames with no index: each such frame opens its own call in arrival order.
  // OpenAI-style streams key fragments by an explicit index and only extend the matching call — both shapes
  // land in this map without special-casing, so a future switch to incremental deltas needs no client change.
  private implicitIndex = 0;

  apply(fragments: ToolCallFragment[]): void {
    for (const fragment of fragments) {
      const explicit = typeof fragment.index === "number" ? fragment.index : null;
      if (explicit !== null && this.calls.has(explicit)) {
        const call = this.calls.get(explicit)!;
        if (fragment.function?.name) call.name += fragment.function.name;
        if (fragment.function?.arguments) call.args += fragment.function.arguments;
      } else {
        this.calls.set(explicit ?? this.implicitIndex++, {
          name: fragment.function?.name ?? "",
          args: fragment.function?.arguments ?? "",
        });
      }
    }
  }

  finish(): ToolCallView[] {
    return [...this.calls.values()].map((call) => ({
      name: call.name || "(unnamed)",
      argumentsText: prettyArgs(call.args),
    }));
  }
}

// Compact, mockup-shaped argument rendering: `{ "expression": "12*12" }` on one line. Non-JSON payloads
// (a half-streamed or malformed tool frame) display verbatim rather than being hidden.
export function prettyArgs(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.length === 0) return "{}";
  try {
    return compactJson(JSON.parse(trimmed) as unknown);
  } catch {
    return trimmed;
  }
}

function compactJson(value: unknown): string {
  if (value === null || typeof value !== "object") return stringifyScalar(value);
  if (Array.isArray(value)) return `[${value.map(compactJson).join(", ")}]`;
  const parts = Object.entries(value as Record<string, unknown>).map(([key, item]) => `${JSON.stringify(key)}: ${compactJson(item)}`);
  return `{ ${parts.join(", ")} }`;
}

function stringifyScalar(value: unknown): string {
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (value === null) return "null";
  return JSON.stringify(String(value));
}

// ---- streaming request ------------------------------------------------------------------------------

export interface StreamHandlers {
  onText(piece: string): void; // every content delta, live — the UI appends progressively
  onFinish(reason: string | null): void; // fired when a chunk carries finish_reason (may precede [DONE])
  onComplete(toolCalls: ToolCallView[]): void; // exactly once per non-aborted stream; calls are complete by then
}

// Runs one streamed chat completion. Aborting `signal` stops consumption immediately and resolves normally:
// whatever text arrived so far is already delivered through onText and stays rendered as partial output;
// onComplete/onFinish simply do not fire for an aborted stream (the UI marks it stopped instead).
export async function streamChatCompletion(
  model: string,
  messages: ChatTurn[],
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  // A Stop that lands before response headers rejects the fetch itself with an AbortError; that is flow control,
  // not a failed exchange — it must resolve like any other abort so no error bubble appears for pressing Stop.
  let response: Response;
  try {
    response = await fetch(CHAT_PATH, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model,
        messages,
        stream: true,
        temperature: REQUEST_TEMPERATURE,
        max_tokens: REQUEST_MAX_TOKENS,
      }),
      signal,
    });
  } catch (error: unknown) {
    if (!signal.aborted && !isAbortLike(error)) throw error; // genuine network failure still surfaces as an error bubble
    return;
  }

  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  if (signal.aborted) return; // abort landed before/while headers arrived — nothing to consume

  const body = response.body;
  if (!body) return; // no readable stream: treat as an empty completion, not a failure

  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  const tools = new ToolCallAccumulator();
  let buffer = "";

  try {
    let sawDone = false;
    while (true) {
      if (signal.aborted) break;
      const { done, value } = await reader.read(); // abort rejects this read — caught below
      if (done) break;

      buffer += decoder.decode(value as Uint8Array, { stream: true });

      for (;;) {
        const frameResult = takeFrame(buffer);
        if (frameResult.rest === null) break; // an incomplete tail stays buffered until the next read arrives
        buffer = frameResult.rest;
        const frame = frameResult.frame;
        if (frame !== null && frame.kind === "done") {
          sawDone = true; // [DONE] terminates the completion — stop consuming immediately
          break;
        }
        if (frame !== null) consumeChunk(frame.payload, tools, handlers);
      }

      if (sawDone) break;
    }

    if (!signal.aborted) {
      // Flush the decoder and settle any unterminated trailing line a DONE-less close left in the buffer.
      buffer += decoder.decode();
      const tail = takeFrame(`${buffer}\n\n`);
      if (tail.frame !== null && tail.frame.kind === "data") consumeChunk(tail.frame.payload, tools, handlers);
    }
  } catch (error: unknown) {
    // AbortController.abort() rejects the pending read with an AbortError; that is expected flow control here.
    if (!signal.aborted && !isAbortLike(error)) throw error;
    return; // aborted mid-stream: no completion callback, partial text already delivered
  } finally {
    try {
      await reader.cancel();
    } catch {
      // cancellation errors on a torn stream are irrelevant to rendering
    }
  }

  if (!signal.aborted) handlers.onComplete(tools.finish());
}

interface ChunkShape {
  choices?: Array<{
    index?: number;
    delta?: { role?: string | null; content?: string | null } & Record<string, unknown>;
    finish_reason?: string | null;
  }> | null;
}

function consumeChunk(payload: string, tools: ToolCallAccumulator, handlers: StreamHandlers): void {
  let chunk: ChunkShape;
  try {
    chunk = JSON.parse(payload) as ChunkShape;
  } catch {
    return; // malformed frame: skip it rather than break the conversation
  }

  for (const choice of Array.isArray(chunk.choices) ? chunk.choices : []) {
    const content = choice.delta?.content;
    if (typeof content === "string" && content.length > 0) handlers.onText(content);

    const calls = choice.delta?.["tool_calls"];
    if (Array.isArray(calls)) tools.apply(calls as ToolCallFragment[]);

    // Lair's finish_reason rides on a trailing empty-delta chunk; OpenAI-style streams may also carry it with text.
    if (choice.finish_reason !== undefined && choice.finish_reason !== null) handlers.onFinish(choice.finish_reason);
  }
}

// ---- errors ------------------------------------------------------------------------------------------

async function errorFromResponse(response: Response): Promise<ApiError> {
  let message = `HTTP ${response.status}`;
  let code = "http_error";
  try {
    const body = (await response.json()) as { error?: { message?: string; type?: string; code?: string } };
    if (body.error) {
      if (typeof body.error.message === "string" && body.error.message.length > 0) message = body.error.message;
      if (typeof body.error.code === "string" && body.error.code.length > 0) code = body.error.code;
      else if (typeof body.error.type === "string" && body.error.type.length > 0) code = body.error.type;
    }
  } catch {
    // non-JSON error bodies fall back to the status line
  }
  return new ApiError(message, code, response.status);
}

function isAbortLike(error: unknown): boolean {
  if (typeof error !== "object" || error === null) return false;
  const candidate = error as { name?: string };
  return candidate.name === "AbortError";
}
