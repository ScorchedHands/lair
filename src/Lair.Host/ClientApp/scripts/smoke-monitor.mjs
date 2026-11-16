// Monitor-tab live wiring smoke test (dev-only tooling): renders the BUILT dist/app.js inside jsdom with a stub
// EventSource and fetch, feeds synthetic v1 stream frames, and asserts the render logic reacts exactly as wired:
// KPI updates, model rows + expand/collapse, in-flight lifecycle from request_start/update/complete, sparkline
// points derived from samples[], status tag mapping (499 for client abort), resync replacement, connection badge.
// Run with: npm run build && node scripts/smoke-monitor.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { JSDOM } from "jsdom";

const here = dirname(fileURLToPath(import.meta.url));
const bundlePath = join(here, "..", "dist", "app.js");
const bundle = readFileSync(bundlePath, "utf8");

// ---- stubs ---------------------------------------------------------------------------------------

const openedStreams = [];

class FakeEventSource {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 2;

  constructor(url) {
    this.url = String(url);
    this.readyState = FakeEventSource.CONNECTING;
    this.closedExplicitly = false;
    this._handlers = new Map();
    openedStreams.push(this);
  }

  addEventListener(name, fn) {
    if (!this._handlers.has(name)) this._handlers.set(name, []);
    this._handlers.get(name).push(fn);
  }

  close() {
    this.closedExplicitly = true;
    this.readyState = FakeEventSource.CLOSED;
  }

  _fire(name, data) {
    for (const fn of this._handlers.get(name) ?? []) fn({ data });
  }

  emitOpen() {
    this.readyState = FakeEventSource.OPEN;
    this._fire("open", undefined);
  }

  emit(event, payload) {
    this._fire(event, JSON.stringify(payload));
  }

  emitError({ closed = false } = {}) {
    // Per spec a recoverable failure leaves readyState CONNECTING (auto-retry in progress); only a terminal close
    // lands on CLOSED.
    this.readyState = closed ? FakeEventSource.CLOSED : FakeEventSource.CONNECTING;
    this._fire("error", undefined);
  }
}

const modelsPayload = {
  object: "list",
  data: [
    { id: "smollm-135m-instruct", object: "model", created: 1760000000, owned_by: "lair", context_length: 1024, quant: "Q8_0", architecture: "llama" },
  ],
};

// ---- DOM ------------------------------------------------------------------------------------------

const dom = new JSDOM(
  `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>lair</title></head><body><div id="root"></div></body></html>`,
  { url: "http://127.0.0.1:8080/ui/index.html", pretendToBeVisual: true, runScripts: "outside-only" },
);
const win = dom.window;

// jsdom lacks MessageChannel (React 19's scheduler uses it for task scheduling). Node's own implementation would
// pin the event loop past teardown, so shim it in the window realm on top of jsdom timers: window.close() then
// stops everything React has queued.
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

win.EventSource = FakeEventSource;
win.fetch = async (url) => ({
  ok: true,
  status: 200,
  json: async () => modelsPayload,
});

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
  }
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const doc = win.document;
const $text = (selector) => doc.querySelector(selector)?.textContent ?? null;
const rowsOf = (headingText) => {
  const sections = [...doc.querySelectorAll("section.block")];
  const section = sections.find((s) => s.querySelector("h2")?.textContent === headingText);
  return section ? [...section.querySelectorAll("tbody tr")] : [];
};

// ---- 1. mount: stream wiring + optimistic badge ----------------------------------------------------

await wait(150);

check("EventSource constructed against /monitor/stream", openedStreams.length >= 1 && openedStreams[0].url === "/monitor/stream");
const es = openedStreams[0];
check("header meta reads the page origin", ($text("header.app .meta") ?? "").includes("http://127.0.0.1:8080"));
check("badge shows live on first dial", $text("header.app .badge") === "live");

const kpiBefore = { toks: $text("#toks"), stepms: $text("#stepms"), hit: $text("#hit"), active: $text("#active") };
check("KPI cards read em dashes while idle", [kpiBefore.toks, kpiBefore.stepms, kpiBefore.hit].every((v) => v === "—"));

const modelRowsBefore = rowsOf("Configured models"); // model row + its (default-open) detail row
check("model row built from one-time /v1/models identity fetch (name + lowercased quant)",
  modelRowsBefore.length >= 1 &&
  modelRowsBefore[0].textContent.includes("smollm-135m-instruct") &&
  modelRowsBefore[0].textContent.includes("chat · q8_0"));

// ---- 2. snapshot frame: KPI cards + model row facts --------------------------------------------------

es.emitOpen();
await wait(60);

const snapshot = {
  metrics_ts: "2026-11-02T21:00:05.000Z",
  metrics_window_ms: 2000,
  metrics_model: "smollm-135m-instruct",
  metrics_tok_s: 27.4,
  metrics_step_time_avg_ms: 36.2,
  metrics_decode_steps_total: 812,
  metrics_batch_occupancy_seqs_per_step: 2.0,
  metrics_prefill_chunks_total: 9,
  metrics_prefill_tokens_forwarded_total: 500,
  metrics_cache_hit_sequences_total: 6,
  metrics_cache_skipped_tokens_total: 770,
  metrics_kv_used_blocks: 54,
  metrics_kv_capacity_blocks: 96,
  metrics_kv_cached_blocks: 18,
  metrics_active_sequences: 2,
  metrics_requests_total: 20,
  metrics_requests_eos_total: 12,
  metrics_requests_length_total: 5,
  metrics_requests_stop_total: 1,
  metrics_requests_cancelled_total: 1,
  metrics_requests_failed_total: 1,
  metrics_process_working_set_mb: 2457.6,
  metrics_process_cpu_percent: 812.4,
  metrics_events_version: 1,
  metrics_event_seq: 41,
  metrics_dropped_events_total: 0,
  // additive fields as the backend now publishes them
  metrics_model_type: "chat",
  metrics_kv_block_tokens: 16,
  metrics_kv_budget_blocks: 96,
  metrics_kv_pool_mb: 384.0,
  metrics_worker_threads: 24,
  metrics_core_set: "0-23",
  metrics_thread_quota: null,
  metrics_decode_device: "cpu",
  metrics_prefill_device: "cpu",
  metrics_kv_residency: "dram",
};

es.emit("snapshot", snapshot);
await wait(150);

check("decode tok/s card decodes from the snapshot window rate", $text("#toks") === "27.4");
check("step time card decodes the windowed average", $text("#stepms") === "36 ms");
check("cache hit card = skipped / (skipped + forwarded)", $text("#hit") === `${Math.round((770 / 1270) * 100)}%`);
check("active reqs card decodes engine active sequences", $text("#active") === "2");
check("sub-line shows queued derived from lifecycle vs snapshot counts", [...doc.querySelectorAll(".card .sub")].some((el) => el.textContent === "queued 0"));
check("process RAM card live from working set", $text("#ram") === "2.4 GB");

const modelRow = rowsOf("Configured models")[0];
const cells = [...modelRow.querySelectorAll("td")].map((td) => td.textContent);
check("model row device column from snapshot facts", cells[2] === "cpu");
check("decode tok/s per-model cell live", cells[3] === "27.4");
check("seqs/step from batch occupancy", cells[4] === "2.00");
check("KV used bar + text from blocks", modelRow.querySelector(".bar.kv span") !== null && cells[5].includes("54/96"));
check("workers column live from snapshot facts", cells[7] === "24");

const detail = rowsOf("Configured models")[1]; // the detail row follows its model row and is open by default for chat kinds
check("detail row visible (expandable default) with configured identity + additive fields",
  !detail.hidden && [...detail.querySelectorAll(".val")].map((v) => v.textContent).join("\n").includes("1024 / 16"));

const detailVals = () => [...rowsOf("Configured models")[1].querySelectorAll(".val")].map((v) => v.textContent);
check("kv budget soft cap cell", detailVals().some((v) => v === "96 (soft cap)"));
check("core set / quota cell from additive fields", detailVals().some((v) => v === "0-23 / —"));
check("prefill · decode · kv residency cell from placement facts", detailVals().some((v) => v === "cpu · cpu · dram"));
check("cache blocks by location derived (prefix/active/free)", detailVals().some((v) => v.includes("prefix 18") && v.includes("active 54") && v.includes("free 42")));
check("finish tallies from snapshot totals", detailVals().some((v) => v === "stop 1 · eos 12 · length 5 · abort 1 · failed 1"));

// A CPU-only snapshot carries no accelerator facts: the device cards keep their placeholder reading rather than showing zeros.
check("GPU util card stays em dash on a CPU-only serve", $text("#gpuutil") === "—" && doc.querySelector("#gpuutil").closest(".card").classList.contains("placeholder"));
check("PCIe transfer card stays em dash on a CPU-only serve", $text("#pcie") === "—" && doc.querySelector("#pcie").closest(".card").classList.contains("placeholder"));
check("model row VRAM column stays reserved-marked without device residency", rowsOf("Configured models")[0].textContent.includes("reserved —"));

// expand/collapse still works on live rows
modelRow.querySelector("td.expand").click();
await wait(80);
check("expand toggle hides the detail row", rowsOf("Configured models")[1].hidden === true);
modelRow.querySelector("td.expand").click();
await wait(80);
check("expand toggle reopens the detail row", rowsOf("Configured models")[1].hidden === false);

// ---- 3. request lifecycle: in-flight table + sparkline from samples ---------------------------------

const idA = "9f2c7a55b4e04d8fa0aa66c2bb8ea1";
es.emit("request_start", { events_version: 1, request_id: idA, model: "smollm-135m", endpoint: "chat", prompt_tokens: 127, status: "running" });
await wait(100);

let flight = rowsOf("In-flight requests");
check("request_start opens an in-flight row (prefill tag, zero gen tok)",
  flight.length === 1 &&
  $text(`#gen-${idA}`) === "0" &&
  [...flight[0].querySelectorAll(".tag")].some((t) => t.textContent === "prefill"));

const samples = Array.from({ length: 8 }, (_, i) => ({ at_token: i + 1, elapsed_ms: Math.round(((i + 1) * 1000) / 25) }));
es.emit("request_update", { events_version: 1, request_id: idA, model: "smollm-135m", endpoint: "chat", prompt_tokens: 127, status: "running", generated_tokens: 8, elapsed_ms: 400, tok_s: 28.1, samples });
await wait(100);

flight = rowsOf("In-flight requests");
check("request_update advances gen tok and tok/s cells in place", $text(`#gen-${idA}`) === "8" && $text(`#tps-${idA}`) === "28.1");
const sparkPoints = flight[0].querySelector("svg.spark polyline")?.getAttribute("points") ?? "";
check("sparkline points come from the samples series (one point per sample interval)", sparkPoints.split(/\s+/).length === 7);

const elapsedBefore = $text(`#el-${idA}`);
await wait(1_300); // local tick between server frames
const elapsedAfter = $text(`#el-${idA}`);
check("elapsed timer ticks locally between updates", elapsedBefore !== null && elapsedAfter !== null && elapsedBefore !== elapsedAfter);

// ---- 4. completion: recent table + status mapping ---------------------------------------------------

es.emit("request_complete", { events_version: 1, request_id: idA, model: "smollm-135m", endpoint: "chat", prompt_tokens: 127, status: "cancelled", generated_tokens: 7, elapsed_ms: 900, tok_s: 7.8, cached_tokens: 112, finish_reason: "cancelled", samples });
await wait(100);

check("completion removes the in-flight row", rowsOf("In-flight requests").length === 0);
let recent = rowsOf("Recent requests (X = 5, configurable)");
const abortTag = recent[0]?.querySelector(".tag");
check("client abort maps to err tag + 499 status", recent.length === 1 && abortTag?.textContent === "499" && abortTag.classList.contains("err"));
check("finish label reads client abort", [...recent[0].querySelectorAll("td")].some((td) => td.textContent === "client abort"));

const idB = "77aa2c8e1d3f4a5eb6cd9e0f4ab12c31";
es.emit("request_complete", { events_version: 1, request_id: idB, model: "smollm-135m-instruct", endpoint: "completion", prompt_tokens: 407, status: "eos", generated_tokens: 32, elapsed_ms: 4600, tok_s: 7.0, cached_tokens: 0, finish_reason: "eos", samples });
await wait(100);

recent = rowsOf("Recent requests (X = 5, configurable)");
const eosTag = recent[0]?.querySelector(".tag");
check("successful completion maps to done tag + 200 status", recent.length === 2 && eosTag?.textContent === "200" && eosTag.classList.contains("done"));

// ---- 5. resync replaces transient state as delivered -------------------------------------------------

const idC = "4b70d1e9c8f7a6b5d3e2";
const idD = "aa02bb18cc29dd30ee41";
es.emit("resync", {
  events_version: 1,
  in_flight: [
    { events_version: 1, request_id: idC, model: "smollm-135m-instruct", endpoint: "completion", prompt_tokens: 407, status: "running" },
    { events_version: 1, request_id: idD, model: "smollm-135m-instruct", endpoint: "chat", prompt_tokens: 96, status: "running", generated_tokens: 5, elapsed_ms: 200, tok_s: 25.0, samples: [{ at_token: 1, elapsed_ms: 40 }, { at_token: 3, elapsed_ms: 110 }, { at_token: 5, elapsed_ms: 200 }] },
  ],
  recent_requests: [
    { events_version: 1, request_id: idB, model: "smollm-135m-instruct", endpoint: "completion", prompt_tokens: 407, status: "eos", generated_tokens: 32, elapsed_ms: 4600, tok_s: 7.0, cached_tokens: 0, finish_reason: "eos" },
  ],
});
await wait(100);

flight = rowsOf("In-flight requests");
recent = rowsOf("Recent requests (X = 5, configurable)");
check("resync rebuilds the in-flight table exactly as delivered", flight.length === 2 && $text(`#gen-${idC}`) === "0" && $text(`#gen-${idD}`) === "5");
check("resync replaces the recent list with what the server ring holds", recent.length === 1);

// ---- 6. connection badge transitions -----------------------------------------------------------------

es.emitError({ closed: false }); // native auto-reconnect in progress (readyState stays CONNECTING)
await wait(80);
check("error during a native retry shows reconnecting", $text("header.app .badge") === "reconnecting" && doc.querySelector("header.app .badge").classList.contains("reconnect"));

const closedStream = openedStreams[openedStreams.length - 1];
closedStream.emitError({ closed: true }); // socket closed: manual backoff path engages
await wait(80);
check("closed stream shows offline until the retry lands", $text("header.app .badge") === "offline" && doc.querySelector("header.app .badge").classList.contains("down"));

await wait(1_400); // client backoff (starts at 1s) opens a fresh EventSource on its own timer
const retried = openedStreams[openedStreams.length - 1];
check("manual backoff reopens the stream", retried !== closedStream && retried.readyState === FakeEventSource.CONNECTING);

retried.emitOpen();
await wait(80);
check("retry reconnects back to live", $text("header.app .badge") === "live" && !doc.querySelector("header.app .badge").classList.contains("down"));

// ---- 6b. accelerator facts: device cards and residency column come live from a CUDA-backed snapshot --------------

es.emit("snapshot", {
  ...snapshot,
  metrics_model: "hybrid-cuda-model",
  metrics_event_seq: 42,
  metrics_decode_device: "cuda",
  metrics_gpu_utilization_percent: 41.5,
  metrics_pcie_transfer_bytes_total: 3_145_728,
  metrics_pcie_transfers_total: 640,
  metrics_persistent_vram_bytes: 78_643_200,
});
await wait(150);

check("GPU util card reads the windowed kernel busy share", $text("#gpuutil") === "41.5%" && !doc.querySelector("#gpuutil").closest(".card").classList.contains("placeholder"));
check("PCIe transfer card totals runtime copies beside resident weights", $text("#pcie") === "3.0 MiB" && [...doc.querySelectorAll(".card .sub")].some((el) => el.textContent === "640 runtime copies · 75.0 MiB resident"));

const hybridRow = rowsOf("Configured models").find((row) => row.textContent.includes("hybrid-cuda-model"));
check("model row VRAM column reads device residency from the engine's own accounting", hybridRow?.textContent.includes("75.0 MiB resident") ?? false);

const hybridDetail = rowsOf("Configured models")[rowsOf("Configured models").indexOf(hybridRow) + 1];
check(
  "detail row carries kernel busy beside runtime transfer totals",
  [...(hybridDetail?.querySelectorAll(".val") ?? [])].some((v) => v.textContent === "41.5% · 3.0 MiB over 640 copies"),
);

// ---- teardown ----------------------------------------------------------------------------------------

win.close(); // stops the client's interval/reconnect timers so Node can exit

console.log(`\n${passed} checks passed, ${failures.length} failed`);
if (failures.length > 0) {
  for (const failure of failures) console.error(`failed: ${failure}`);
  process.exitCode = 1;
}
