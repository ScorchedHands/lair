// Chat-tab live wiring smoke test (dev-only tooling): renders the BUILT dist/app.js inside jsdom with stubbed fetch
// and a hand-fed fake SSE body stream, then asserts the wired behavior end to end: model selector populated from
// /v1/models with embed entries excluded; request payload shape (stream/temperature/max_tokens); progressive delta
// rendering into an assistant bubble with the streaming caret class across split frame boundaries; tool_calls delta
// assembly in BOTH wire shapes — lair's complete-call frames and OpenAI-style index-keyed fragments; multi-turn
// history on every send; Stop mid-stream aborts through AbortSignal leaving partial text + a stopped marker and a
// working next send; API error JSON renders as an inline bubble while the composer stays usable.
// Run with: npm run build && node scripts/smoke-chat.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { TextDecoder, TextEncoder } from "node:util";
import { JSDOM } from "jsdom";

const here = dirname(fileURLToPath(import.meta.url));
const bundlePath = join(here, "..", "dist", "app.js");
const bundle = readFileSync(bundlePath, "utf8");

// ---- stubs ---------------------------------------------------------------------------------------

class FakeEventSource {
  // The monitor tab mounts alongside chat; a no-op stream keeps it inert for this harness.
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 2;
  constructor(url) {
    this.url = String(url);
    this.readyState = FakeEventSource.CONNECTING;
  }
  addEventListener() {}
  close() {
    this.readyState = FakeEventSource.CLOSED;
  }
}

const modelsPayload = {
  object: "list",
  data: [
    { id: "smollm-135m-instruct", object: "model", created: 1760000000, owned_by: "lair", context_length: 1024, quant: "F16", architecture: "llama", type: "chat" },
    { id: "modernbert-embed", object: "model", created: 1760000000, owned_by: "lair", context_length: 512, quant: "Q8_0", architecture: "modernbert", type: "embed" },
  ],
};

// One scripted chat completion. The harness pushes encoded frame pieces; the client consumes them through a fake
// reader that rejects pending reads on AbortSignal — exactly what a browser fetch does when aborted mid-stream.
class ScriptedStream {
  constructor(signal) {
    this.signal = signal ?? null;
    this.abortedObserved = false;
    this._pending = null; // one outstanding read at a time (the client's loop is sequential)
    this._chunks = [];

    if (this.signal) {
      const onAbort = () => {
        this.abortedObserved = true;
        if (this._pending !== null) {
          const error = new Error("The operation was aborted.");
          error.name = "AbortError";
          const pending = this._pending;
          this._pending = null;
          pending.reject(error);
        }
      };
      if (this.signal.aborted) onAbort();
      else this.signal.addEventListener("abort", onAbort, { once: true });
    }
  }

  push(text) {
    const chunk = new TextEncoder().encode(text);
    if (this._pending !== null && this._chunks.length === 0) {
      const pending = this._pending;
      this._pending = null;
      pending.resolve({ done: false, value: chunk });
    } else {
      this._chunks.push(chunk);
    }
  }

  // Convenience for tests: emit one complete SSE frame. Objects are JSON-serialized; strings pass through raw.
  frame(payload) {
    this.push(`data: ${typeof payload === "string" ? payload : JSON.stringify(payload)}\n\n`);
  }

  getReader() {
    return {
      read: () =>
        new Promise((resolve, reject) => {
          if (this.abortedObserved || (this.signal !== null && this.signal.aborted)) {
            const error = new Error("The operation was aborted.");
            error.name = "AbortError";
            reject(error);
            return;
          }
          if (this._chunks.length > 0) {
            resolve({ done: false, value: this._chunks.shift() });
            return;
          }
          this._pending = { resolve, reject };
        }),
      cancel: async () => {},
    };
  }
}

const chatRequests = []; // every POST to /v1/chat/completions, in order — the harness scripts each outcome first

let nextChatResponse = null; // set by tests before clicking Send: { script: [...] } or { status, error }

// ---- DOM ------------------------------------------------------------------------------------------

const dom = new JSDOM(
  `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>lair</title></head><body><div id="root"></div></body></html>`,
  { url: "http://127.0.0.1:8080/ui/index.html", pretendToBeVisual: true, runScripts: "outside-only" },
);
const win = dom.window;

// jsdom lacks MessageChannel (React 19's scheduler uses it) — same shim as the monitor harness so window.close()
// can stop everything React queued. TextDecoder is pinned from Node when jsdom does not expose it.
if (!win.MessageChannel) {
  win.MessageChannel = class MessageChannel {
    constructor() {
      const port1 = { onmessage: null, close() {}, start() {}, postMessage() {} };
      const port2 = {
        onmessage: null,
        close() {},
        start() {},
        postMessage: (data) => {
          win.setTimeout(() => {
            if (typeof port1.onmessage === "function") port1.onmessage({ data, ports: [], source: port2 });
          }, 0);
        },
      };
      this.port1 = port1;
      this.port2 = port2;
    }
  };
}
if (!win.TextDecoder) win.TextDecoder = TextDecoder;

win.EventSource = FakeEventSource;
win.fetch = async (url, options = {}) => {
  const target = String(url);

  if (target === "/v1/models") {
    return { ok: true, status: 200, json: async () => modelsPayload };
  }

  if (target === "/v1/chat/completions") {
    const bodyJson = JSON.parse(options.body ?? "{}");
    const stream = new ScriptedStream(options.signal);
    chatRequests.push({ body: bodyJson, signal: options.signal, stream });

    const planned = nextChatResponse;
    nextChatResponse = null;

    if (planned?.error !== undefined) {
      return { ok: false, status: planned.status, body: null, json: async () => planned.error };
    }

    // Fully scripted turns queue their pieces up front; live-pushed turns start from the keepalive only and get
    // frames through stream.push/frame() as the test observes rendering mid-stream.
    stream.push(": accepted\n\n"); // SSE comment line — must be skipped by the parser, never rendered
    for (const piece of planned?.script ?? []) {
      if (typeof piece === "string") stream.push(piece);
      else stream.frame(piece); // frame() JSON-serializes objects itself
    }

    return { ok: true, status: 200, body: { getReader: () => stream.getReader() }, json: async () => ({}) };
  }

  throw new Error(`unexpected fetch target in smoke harness: ${target}`);
};

await win.eval(bundle); // mounts App against the stubs

// ---- assertion plumbing ---------------------------------------------------------------------------

let passed = 0;
const failures = [];

function check(name, condition) {
  if (condition) {
    passed++;
    console.log(`PASS ${name}`);
  } else {
    failures.push(name);
    console.log(`FAIL ${name}`);
    if (process.env.SMOKE_DEBUG) {
      const b = assistantBubbles().map((x) => JSON.stringify(x.textContent));
      console.log(`   [debug] bubbles=${b.join(" | ")} stopDisabled=${stopButton()?.disabled} sendDisabled=${sendButton()?.disabled}`);
    }
  }
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const doc = win.document;
const $one = (selector) => doc.querySelector(selector);
const assistantBubbles = () => [...doc.querySelectorAll("#tab-chat .msg.assistant")];
const lastAssistant = () => assistantBubbles().at(-1) ?? null;
const sendButton = () => $one('.composer button:not(.stop)');
const stopButton = () => $one(".composer .btn.stop");

function typeMessage(text) {
  const input = $one(".composer input");
  const setter = Object.getOwnPropertyDescriptor(win.HTMLInputElement.prototype, "value").set;
  setter.call(input, text);
  input.dispatchEvent(new win.Event("input", { bubbles: true })); // React's controlled-input update path
}

const clickSend = () => sendButton().click();
const clickStop = () => stopButton().click();

// ---- 1. selector populated from /v1/models, embed excluded ----------------------------------------

await wait(200); // the model fetch resolves long before any retry backoff would engage

const options = [...doc.querySelectorAll("#chat-model option")];
check("selector lists exactly the chat models from /v1/models", options.length === 1 && options[0].value === "smollm-135m-instruct");
check("embed-type model excluded from selector", !options.some((option) => option.value === "modernbert-embed"));
check("selector label carries quant metadata lowercased", (options[0]?.textContent ?? "").includes("(chat · f16)"));

check("Send disabled while the draft is empty", sendButton().disabled === true);
check("Stop disabled while nothing streams", stopButton().disabled === true);

// ---- 2. streamed deltas render progressively into an assistant bubble (caret state) -----------------

typeMessage("Explain what a KV cache is.");
await wait(60);
check("Send enabled once the draft has text", sendButton().disabled === false);

nextChatResponse = { script: [] }; // frames are pushed live below so rendering can be observed mid-stream
clickSend();
await wait(80);

const firstRequest = chatRequests[0];
check(
  "request payload is OpenAI-shaped with the advertised knobs",
  firstRequest.body.model === "smollm-135m-instruct" &&
    firstRequest.body.stream === true &&
    firstRequest.body.temperature === 0.7 &&
    firstRequest.body.max_tokens === 256 &&
    Array.isArray(firstRequest.body.messages) &&
    firstRequest.body.messages.length === 1 &&
    firstRequest.body.messages[0].role === "user" &&
    firstRequest.body.messages[0].content === "Explain what a KV cache is.",
);

firstRequest.stream.frame({ choices: [{ index: 0, delta: { role: "assistant" } }] });
await wait(60);

let bubble = lastAssistant();
check("assistant bubble exists and carries the streaming caret class", bubble !== null && bubble.classList.contains("streaming"));
check("role-only frame renders no text", (bubble?.textContent ?? "") === "");

// One delta delivered split mid-JSON across chunk boundaries to prove buffering.
const deltaOne = { choices: [{ index: 0, delta: { content: "A KV cache stores key/value vectors" } }] };
const deltaOneText = `data: ${JSON.stringify(deltaOne)}\n\n`;
firstRequest.stream.push(deltaOneText.slice(0, 31)); // no frame boundary yet — nothing may render
await wait(60);
bubble = lastAssistant();
check("no partial text before a frame boundary completes", (bubble?.textContent ?? "") === "");

firstRequest.stream.push(deltaOneText.slice(31) + "\n\n");
await wait(60);
bubble = lastAssistant();
check("content delta renders after reassembly across chunk splits", bubble?.textContent === "A KV cache stores key/value vectors");
check("caret still streaming mid-turn", bubble !== null && bubble.classList.contains("streaming"));

// Lair's complete-call tool_calls frame (no index field) plus a second content delta.
firstRequest.stream.frame({ choices: [{ index: 0, delta: { content: " for earlier tokens." } }] });
await wait(60);
bubble = lastAssistant();
check("second delta appends progressively", bubble?.textContent === "A KV cache stores key/value vectors for earlier tokens.");

firstRequest.stream.frame({
  choices: [
    { index: 0, delta: { tool_calls: [{ id: "call_abc123def456", type: "function", function: { name: "calculator", arguments: '{"expression":"12*12"}' } }] } },
  ],
});

// finish chunk (empty delta + finish_reason, as lair emits) then [DONE].
firstRequest.stream.frame({ choices: [{ index: 0, delta: {}, finish_reason: "eos" }] });
firstRequest.stream.push("data: [DONE]\n\n");
await wait(120);

bubble = lastAssistant();
const toolCards = bubble ? [...bubble.querySelectorAll(".tool")] : [];
check("complete-call tool frame assembles a mockup-style tool card", toolCards.length === 1 && toolCards[0].textContent === 'tool call → calculator { "expression": "12*12" }');
check("caret removed after [DONE] completion", bubble !== null && !bubble.classList.contains("streaming"));

// ---- 3. multi-turn history + OpenAI-style index-keyed fragments ------------------------------------

typeMessage("Now search for paged memory.");
await wait(60);

nextChatResponse = {
  script: [
    { choices: [{ index: 0, delta: { role: "assistant" } }] },
    { choices: [{ index: 0, delta: { content: "Searching." } }] },
    // OpenAI-style fragments: explicit index; name arrives first, arguments accumulate across frames.
    { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "search" } }] } }] },
    { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '{"query":"kv' } }] } }] },
    { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: ' cache"}' } }] } }] },
    { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
    "data: [DONE]\n\n",
  ],
};
clickSend();
await wait(200);

const secondRequest = chatRequests[1];
check(
  "multi-turn state: full messages array carries the completed assistant turn verbatim",
  Array.isArray(secondRequest.body.messages) &&
    secondRequest.body.messages.length === 3 &&
    secondRequest.body.messages[1].role === "assistant" &&
    secondRequest.body.messages[1].content === "A KV cache stores key/value vectors for earlier tokens.",
);

bubble = lastAssistant();
const turn2Cards = bubble ? [...bubble.querySelectorAll(".tool")] : [];
check(
  "index-keyed fragments assemble one call with accumulated arguments",
  (bubble?.textContent ?? "").includes("Searching.") &&
    turn2Cards.length === 1 &&
    turn2Cards[0].textContent === 'tool call → search { "query": "kv cache" }',
);

// ---- 4. Stop mid-stream: abort leaves partial text + stopped marker, next send works -----------------

typeMessage("Write a haiku about paged memory.");
await wait(60);

nextChatResponse = { script: [] }; // live-pushed so the stream can be cut mid-turn
clickSend();
await wait(80);

const thirdRequest = chatRequests[2];
thirdRequest.stream.frame({ choices: [{ index: 0, delta: { role: "assistant" } }] });
for (const piece of ["Blocks of sixteen tokens,", " mapped like pages"]) {
  thirdRequest.stream.frame({ choices: [{ index: 0, delta: { content: piece } }] }); // second piece carries the join space
}
await wait(80);

bubble = lastAssistant();
check("partial assistant text is rendered before Stop", bubble?.textContent === "Blocks of sixteen tokens, mapped like pages" && bubble.classList.contains("streaming"));

thirdRequest.stream.push('data: {"choices":[{"index":0,"delta":{"content":" UNRENDERED'); // half a frame on the wire when Stop lands
await wait(60);
check("unterminated partial frame never renders as text", !(lastAssistant()?.textContent ?? "").includes("UNRENDERED"));

clickStop();
await wait(120);

bubble = lastAssistant();
const stoppedMarkers = bubble ? [...bubble.querySelectorAll(".stopped")] : [];
check("aborted turn keeps its partial text with a stopped indicator", (bubble?.textContent ?? "").startsWith("Blocks of sixteen tokens, mapped like pages") && stoppedMarkers.length === 1 && stoppedMarkers[0].textContent.trim() === "· stopped");
check("caret cleared on the aborted turn", bubble !== null && !bubble.classList.contains("streaming"));
check("client abort reached the wire (AbortSignal honored)", thirdRequest.signal.aborted === true && thirdRequest.stream.abortedObserved === true);

// The next send must work after an abort — and history includes the stopped assistant text.
typeMessage("Continue.");
await wait(60);
nextChatResponse = {
  script: [
    { choices: [{ index: 0, delta: { content: "done" } }] },
    { choices: [{ index: 0, delta: {}, finish_reason: "eos" }] },
    "data: [DONE]\n\n",
  ],
};
clickSend();
await wait(160);

const fourthRequest = chatRequests[3];
check("next send after an abort works", chatRequests.length === 4 && (lastAssistant()?.textContent ?? "") === "done");
check("stopped partial turn entered the multi-turn history", JSON.stringify(fourthRequest.body.messages).includes('"Blocks of sixteen tokens, mapped like pages"'));

// ---- 5. API error renders inline without breaking the composer --------------------------------------

typeMessage("Say something enormous beyond the context window.");
await wait(60);
nextChatResponse = {
  status: 400,
  error: { error: { message: "prompt of 18 tokens plus max_tokens 256 exceeds a 32-token context", type: "invalid_request_error", code: "context_length_exceeded" } },
};
clickSend();
await wait(120);

const errorBubble = $one("#tab-chat .msg.error");
check("API error renders as one inline bubble naming message + code", errorBubble !== null && !errorBubble.classList.contains("streaming") && (errorBubble.textContent ?? "").includes("context_length_exceeded") && (errorBubble.textContent ?? "").includes("exceeds a 32-token context"));

check("failed exchange leaves no ghost caret bubble", assistantBubbles().length === 4); // turns 1-4 only; the failed turn's empty bubble was removed
check("failed exchange still issued its POST (the error came from the response)", chatRequests.length === 5 && chatRequests[4].body.messages.at(-1).content.includes("enormous beyond"));

typeMessage("Short question after an error.");
await wait(60);
check("composer stays usable after an inline error", sendButton().disabled === false && stopButton().disabled === true);

nextChatResponse = {
  script: [
    { choices: [{ index: 0, delta: { content: "recovered" } }] },
    { choices: [{ index: 0, delta: {}, finish_reason: "eos" }] },
    "data: [DONE]\n\n",
  ],
};
clickSend();
await wait(160);

const recoveryRequest = chatRequests[5]; // index 4 was the failed POST itself; this is the send after it
check("conversation stays usable: the failed exchange was dropped from sent history", chatRequests.length === 6 && !JSON.stringify(recoveryRequest.body.messages).includes("enormous beyond") && (lastAssistant()?.textContent ?? "") === "recovered");

// ---- teardown ----------------------------------------------------------------------------------------

win.close(); // stops client timers so Node can exit

console.log(`\n${passed} checks passed, ${failures.length} failed`);
if (failures.length > 0) {
  for (const failure of failures) console.error(`failed: ${failure}`);
  process.exitCode = 1;
}
