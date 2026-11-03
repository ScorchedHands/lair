// Live monitor wiring: one EventSource client for the v1 /monitor/stream schema, a typed store React
// subscribes to through useSyncExternalStore, and view-model builders that format what each event carries.
// The stream vocabulary (MonitorEventNames) is mirrored exactly; unknown fields are ignored, missing ones read null.

import { useCallback, useSyncExternalStore } from "react";

export const STREAM_PATH = "/monitor/stream";
export const MODELS_PATH = "/v1/models";

export type ConnectionState = "connecting" | "live" | "reconnecting" | "offline";

// ---- wire payloads (schema v1) -------------------------------------------------------------

interface WireSample {
  at_token: number;
  elapsed_ms: number;
}

interface WireSnapshot {
  metrics_ts: string;
  metrics_window_ms: number;
  metrics_model: string;
  metrics_tok_s: number;
  metrics_step_time_avg_ms: number;
  metrics_decode_steps_total: number;
  metrics_batch_occupancy_seqs_per_step: number;
  metrics_prefill_chunks_total: number;
  metrics_prefill_tokens_forwarded_total: number;
  metrics_cache_hit_sequences_total: number;
  metrics_cache_skipped_tokens_total: number;
  metrics_kv_used_blocks: number;
  metrics_kv_capacity_blocks: number;
  metrics_kv_cached_blocks: number;
  metrics_active_sequences: number;
  metrics_requests_total: number;
  metrics_requests_eos_total: number;
  metrics_requests_length_total: number;
  metrics_requests_stop_total: number;
  metrics_requests_cancelled_total: number;
  metrics_requests_failed_total: number;
  metrics_process_working_set_mb: number;
  metrics_process_cpu_percent: number;
  metrics_events_version: number;
  metrics_event_seq: number;
  metrics_dropped_events_total: number;
  // additive operator-view fields (null until the host attaches facts)
  metrics_model_type?: string | null;
  metrics_kv_block_tokens?: number;
  metrics_kv_budget_blocks?: number;
  metrics_kv_pool_mb?: number;
  metrics_worker_threads?: number | null;
  metrics_core_set?: string | null;
  metrics_thread_quota?: number | null;
  metrics_decode_device?: string | null;
  metrics_prefill_device?: string | null;
  metrics_kv_residency?: string | null;
}

interface WireRequest {
  events_version: number;
  request_id: string;
  model: string;
  endpoint: string;
  prompt_tokens: number;
  status: string;
  generated_tokens?: number | null;
  elapsed_ms?: number | null;
  tok_s?: number | null;
  cached_tokens?: number | null;
  finish_reason?: string | null;
  samples?: WireSample[] | null;
}

interface WireResync {
  events_version: number;
  in_flight: WireRequest[];
  recent_requests: WireRequest[];
}

interface ModelIdentity {
  id: string;
  quant: string;
  contextLength: number;
}

// ---- view models ------------------------------------------------------------------------------

export interface KpiView {
  value: string;
  sub: string;
}

export interface DetailField {
  label: string;
  value: string;
}

export interface ModelRowView {
  id: string;
  name: string;
  kind: "chat" | "embed";
  quant: string;
  device: string;
  decodeTps: string;
  seqsPerStep: string;
  kvBarPct: number | null;
  kvText: string;
  cacheHit: string;
  workers: string;
  vram: string;
  expandedByDefault: boolean;
  detail: DetailField[];
}

export interface SparkView {
  points: string;
}

export interface InFlightRowView {
  id: string;
  model: string;
  endpoint: string;
  promptTok: string;
  genTokId: string;
  genTok: string;
  cached: string;
  tpsId: string;
  tps: string;
  spark: SparkView;
  elapsedId: string;
  elapsed: string;
  tagClass: string;
  tagText: string;
}

export interface RecentRowView {
  id: string;
  model: string;
  statusClass: string;
  statusText: string;
  finish: string;
  tokPC: string;
  avgTps: string;
  spark: SparkView;
  duration: string;
}

export interface MonitorView {
  connection: ConnectionState;
  meta: string;
  kpis: Record<string, KpiView>;
  processRam: KpiView | null;
  models: ModelRowView[];
  inFlight: InFlightRowView[];
  recent: RecentRowView[];
}

// ---- formatting helpers ------------------------------------------------------------------------

const DASH = "—";

function fmt1(value: number): string {
  return value.toFixed(1);
}

function shortId(id: string): string {
  // Stream ids are full correlation GUIDs (32 hex chars, no dashes for generated ones; API ids may carry dashes).
  const plain = id.replace(/-/g, "");
  if (plain.length <= 9) return id;
  return `${plain.slice(0, 4)}…${plain.slice(-2)}`;
}

function fmtDuration(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

export function percentOf(part: number, whole: number): string {
  if (whole <= 0) return DASH;
  return `${Math.round((part / whole) * 100)}%`;
}

// Sparkline points from the tok/s sample series: each interval between consecutive samples becomes one segment
// rate; x spreads across the fixed 90-wide viewBox, y inverts and scales rates into [2..16] so faster reads higher.
export function sparkPoints(samples: WireSample[] | null | undefined): string {
  if (!samples || samples.length === 0) return "";
  const rates: number[] = [];
  for (let i = 1; i < samples.length; i++) {
    const dMs = samples[i].elapsed_ms - samples[i - 1].elapsed_ms;
    const dTok = samples[i].at_token - samples[i - 1].at_token;
    if (dMs > 0 && dTok > 0) rates.push((dTok * 1000.0) / dMs);
  }
  if (rates.length === 0) {
    // A single sample has no interval to plot: one mid-height point keeps the cell quiet but present.
    return "45,9";
  }
  const max = Math.max(...rates);
  const min = Math.min(...rates);
  const span = max - min;
  const step = rates.length > 1 ? 90 / (rates.length - 1) : 0;
  return rates
    .map((rate, i) => {
      const x = rates.length === 1 ? 45 : Math.round(i * step);
      const y = span === 0 ? 6 : Math.round(16 - ((rate - min) / span) * 14);
      return `${x},${y}`;
    })
    .join(" ");
}

function endpointLabel(endpoint: string): string {
  return endpoint === "chat" ? "chat/completions" : endpoint === "completion" ? "completions" : endpoint;
}

const FINISH_LABELS: Record<string, string> = {
  eos: "eos",
  length: "length",
  stop: "stop",
  tool_calls: "tool calls",
  cancelled: "client abort",
  failed: "failed",
};

function recentStatus(payload: WireRequest): { statusClass: string; statusText: string } {
  const finish = payload.finish_reason ?? "";
  if (finish === "cancelled") return { statusClass: "err", statusText: "499" };
  if (finish === "failed") return { statusClass: "err", statusText: "500" };
  return { statusClass: "done", statusText: "200" };
}

// ---- store -------------------------------------------------------------------------------------

interface InFlightEntry {
  payload: WireRequest;
  seenAtMs: number; // client clock when the frame arrived — local tick extends its elapsed between updates
}

class MonitorStore {
  private snapshots = new Map<string, WireSnapshot>();
  private identities = new Map<string, ModelIdentity>();
  private inFlight = new Map<string, InFlightEntry>();
  private recent: WireRequest[] = []; // newest-first; resync replaces wholesale (server ring is the bound)
  private lastSnapshotSeq = 0;
  private connection: ConnectionState = "connecting";

  private listeners = new Set<() => void>();
  private viewCache: MonitorView | null = null;

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  getSnapshot(): MonitorView {
    if (this.viewCache === null) {
      this.viewCache = this.buildView(Date.now());
    }
    return this.viewCache;
  }

  private buildView(nowMs: number): MonitorView {
    const snapshots = [...this.snapshots.values()];
    const latestByModel = this.snapshots; // one newest-per-model already, by the seq rule in acceptSnapshot

    const kpis: Record<string, KpiView> = {};
    if (snapshots.length === 0) {
      // Graceful idle state until the first engine frame lands: every measured value reads as an em dash.
      kpis.toks = { value: DASH, sub: "cpu in use" };
      kpis.stepms = { value: DASH, sub: DASH };
      kpis.hit = { value: DASH, sub: "engine-wide prefix reuse" };
      kpis.active = { value: DASH, sub: "queued —" };
    } else {
      const decodeTps = snapshots.reduce((sum, s) => sum + s.metrics_tok_s, 0);
      const stepAvgMs = Math.max(...snapshots.map((s) => s.metrics_step_time_avg_ms));
      const windowSec = Math.max(...snapshots.map((s) => s.metrics_window_ms)) / 1000;
      const skippedTotal = snapshots.reduce((sum, s) => sum + s.metrics_cache_skipped_tokens_total, 0);
      const forwardedTotal = snapshots.reduce((sum, s) => sum + s.metrics_prefill_tokens_forwarded_total, 0);
      const activeSequences = snapshots.reduce((sum, s) => sum + s.metrics_active_sequences, 0);

      // Requests whose lifecycle frames are live but which the last snapshot did not count as active yet — a true
      // micro-queue between admission and the next step boundary; normally zero.
      const queued = Math.max(0, this.inFlight.size - activeSequences);

      const firstSnapshot = snapshots[0];
      const device = (firstSnapshot ? firstSnapshot.metrics_decode_device : null) ?? "cpu";
      kpis.toks = { value: fmt1(decodeTps), sub: `${device} in use` };
      // No server-side p95 today (windowed average only): the sub names what was measured instead.
      kpis.stepms = { value: `${Math.round(stepAvgMs)} ms`, sub: `avg over ${fmt1(windowSec)}s window` };
      const hitWhole = skippedTotal + forwardedTotal;
      kpis.hit = { value: percentOf(skippedTotal, hitWhole), sub: "engine-wide prefix reuse" };
      kpis.active = { value: String(activeSequences), sub: `queued ${queued}` };
    }

    let processRam: KpiView | null = null;
    if (snapshots.length > 0) {
      const workingSetMb = Math.max(...snapshots.map((s) => s.metrics_process_working_set_mb));
      processRam = { value: `${fmt1(workingSetMb / 1024)} GB`, sub: "engine working set" };
    }

    const modelIds = new Set<string>([...this.snapshots.keys(), ...this.identities.keys()]);
    const models: ModelRowView[] = [...modelIds].sort().map((id) => buildModelRow(id, latestByModel.get(id), this.identities.get(id)));

    const inFlight: InFlightRowView[] = [...this.inFlight.values()]
      .sort((a, b) => a.seenAtMs - b.seenAtMs)
      .map((entry) => buildInFlightRow(entry, nowMs));

    const recent: RecentRowView[] = this.recent.map(buildRecentRow);

    const meta = `${window.location.origin} · up ${fmtDuration(nowMs - mountedAtMs)}`;

    return { connection: this.connection, meta, kpis, processRam, models, inFlight, recent };
  }

  private invalidate(): void {
    this.viewCache = null;
    for (const listener of [...this.listeners]) listener();
  }

  setConnection(state: ConnectionState): void {
    if (this.connection !== state) {
      this.connection = state;
      this.invalidate();
    }
  }

  // Keep the newest snapshot per model by its embedded seq: replay (ascending) and the live tail can both deliver a
  // frame for one model, and an older duplicate must never overwrite a newer view.
  acceptSnapshot(snapshot: WireSnapshot): void {
    const existing = this.snapshots.get(snapshot.metrics_model);
    if (existing && existing.metrics_event_seq >= snapshot.metrics_event_seq) return;
    if (snapshot.metrics_event_seq > this.lastSnapshotSeq) this.lastSnapshotSeq = snapshot.metrics_event_seq;
    this.snapshots.set(snapshot.metrics_model, snapshot);
    this.invalidate();
  }

  acceptRequestStart(payload: WireRequest): void {
    this.inFlight.set(payload.request_id, { payload, seenAtMs: Date.now() });
    this.invalidate();
  }

  acceptRequestUpdate(payload: WireRequest): void {
    const existing = this.inFlight.get(payload.request_id);
    if (existing) {
      existing.payload = payload;
      existing.seenAtMs = Date.now();
    } else {
      // A coalesced tail may have dropped the start frame: adopt the update as the row's first view.
      this.inFlight.set(payload.request_id, { payload, seenAtMs: Date.now() });
    }
    this.invalidate();
  }

  acceptRequestComplete(payload: WireRequest): void {
    this.inFlight.delete(payload.request_id);
    if (!this.recent.some((row) => row.request_id === payload.request_id)) {
      this.recent.unshift(payload);
      if (this.recent.length > 50) this.recent.length = 50; // client-side ceiling only — the server ring is authoritative
    }
    this.invalidate();
  }

  acceptResync(resync: WireResync): void {
    this.inFlight.clear();
    for (const payload of resync.in_flight) {
      this.inFlight.set(payload.request_id, { payload, seenAtMs: Date.now() });
    }
    this.recent = [...resync.recent_requests];
    // Resync rows name models too; make sure a row exists even before the first snapshot for them.
    for (const payload of [...resync.in_flight, ...resync.recent_requests]) {
      if (!this.snapshots.has(payload.model) && !this.identities.has(payload.model)) {
        this.identities.set(payload.model, { id: payload.model, quant: DASH, contextLength: 0 });
      }
    }
    this.invalidate();
  }

  acceptModels(list: ModelIdentity[]): void {
    for (const identity of list) this.identities.set(identity.id, identity);
    this.invalidate();
  }

  tick(): void {
    // Rebuild the view so locally ticking elapsed timers advance between server frames.
    if (this.inFlight.size > 0 || this.connection !== "offline") this.invalidate();
  }
}

// ---- view assembly ------------------------------------------------------------------------------

function buildModelRow(id: string, snapshot: WireSnapshot | undefined, identity: ModelIdentity | undefined): ModelRowView {
  const kind = (snapshot?.metrics_model_type === "embed" ? "embed" : "chat") as "chat" | "embed";
  const embed = kind === "embed";

  const name = identity?.id ?? snapshot?.metrics_model ?? id;
  const quant = identity && identity.quant !== DASH ? identity.quant.toLowerCase() : DASH;
  const device = snapshot?.metrics_decode_device ?? DASH;

  const decodeTps = embed || !snapshot ? DASH : fmt1(snapshot.metrics_tok_s);
  const occupancy = snapshot?.metrics_batch_occupancy_seqs_per_step ?? null;
  // No measured window yet reads as an em dash; a lone sequence in flight is exactly one per step.
  const seqsPerStep = embed || occupancy === null ? DASH : occupancy.toFixed(2);

  let kvBarPct: number | null = null;
  let kvText = embed ? "n/a" : DASH;
  if (!embed && snapshot) {
    const used = snapshot.metrics_kv_used_blocks;
    const capacity = snapshot.metrics_kv_capacity_blocks;
    kvBarPct = capacity > 0 ? Math.round((used / capacity) * 100) : null;
    kvText = `${used}/${capacity}`;
  }

  let cacheHit = DASH;
  if (!embed && snapshot) {
    const skipped = snapshot.metrics_cache_skipped_tokens_total;
    const whole = skipped + snapshot.metrics_prefill_tokens_forwarded_total;
    cacheHit = percentOf(skipped, whole);
  }

  const workers = snapshot?.metrics_worker_threads != null ? String(snapshot.metrics_worker_threads) : DASH;

  const contextTokens = identity && identity.contextLength > 0 ? String(identity.contextLength) : DASH;
  const blockTokens = snapshot?.metrics_kv_block_tokens ? String(snapshot.metrics_kv_block_tokens) : DASH;
  const budget = snapshot?.metrics_kv_budget_blocks ? `${snapshot.metrics_kv_budget_blocks} (soft cap)` : DASH;

  const coreSet = snapshot?.metrics_core_set ?? "none (unpinned)";
  const quota = snapshot?.metrics_thread_quota != null ? String(snapshot.metrics_thread_quota) : DASH;

  const prefillDevice = snapshot?.metrics_prefill_device ?? DASH;
  const decodeDevice = snapshot?.metrics_decode_device ?? DASH;
  const residency = snapshot?.metrics_kv_residency ?? DASH;

  let cacheBlocks = DASH;
  if (!embed && snapshot) {
    const prefix = snapshot.metrics_kv_cached_blocks;
    const active = snapshot.metrics_kv_used_blocks;
    const free = Math.max(snapshot.metrics_kv_capacity_blocks - snapshot.metrics_kv_used_blocks, 0);
    cacheBlocks = `prefix ${prefix} · active ${active} · free ${free}`;
  }

  let tallies = DASH;
  if (snapshot) {
    tallies = [
      `stop ${snapshot.metrics_requests_stop_total}`,
      `eos ${snapshot.metrics_requests_eos_total}`,
      `length ${snapshot.metrics_requests_length_total}`,
      `abort ${snapshot.metrics_requests_cancelled_total}`,
      `failed ${snapshot.metrics_requests_failed_total}`,
    ].join(" · ");
  }

  let modelRam = DASH;
  if (snapshot) {
    const workingSetGb = fmt1(snapshot.metrics_process_working_set_mb / 1024);
    const kvPoolMb = snapshot.metrics_kv_pool_mb ?? 0;
    const kvGb = kvPoolMb >= 1024 ? `${fmt1(kvPoolMb / 1024)} GB` : `${Math.round(kvPoolMb)} MB`;
    modelRam = `${workingSetGb} GB working set · ${kvGb} KV reserved`;
  }

  return {
    id,
    name,
    kind,
    quant,
    device,
    decodeTps,
    seqsPerStep,
    kvBarPct,
    kvText,
    cacheHit,
    workers,
    vram: "reserved —", // no GPU surface yet; the mockup's VRAM column stays reserved-marked
    expandedByDefault: !embed,
    detail: [
      { label: "context / block size", value: `${contextTokens} / ${blockTokens}` },
      { label: "kv_blocks budget", value: budget },
      { label: "core set / quota", value: `${coreSet} / ${quota}` },
      { label: "prefill · decode · kv residency", value: `${prefillDevice} · ${decodeDevice} · ${residency}` },
      { label: "cache blocks (DRAM)", value: cacheBlocks },
      { label: "finish tallies", value: tallies },
      { label: "experts hot / cold", value: "n/a — dense model (MoE phase adds tier counts)" },
      { label: "model RAM", value: modelRam },
    ],
  };
}

function buildInFlightRow(entry: InFlightEntry, nowMs: number): InFlightRowView {
  const payload = entry.payload;
  const generated = payload.generated_tokens ?? 0;
  const serverElapsedMs = payload.elapsed_ms ?? 0;
  const elapsedMs = serverElapsedMs + Math.max(nowMs - entry.seenAtMs, 0);

  return {
    id: shortId(payload.request_id),
    model: payload.model,
    endpoint: endpointLabel(payload.endpoint),
    promptTok: String(payload.prompt_tokens),
    genTokId: `gen-${payload.request_id}`,
    genTok: String(generated),
    cached: payload.cached_tokens != null ? String(payload.cached_tokens) : DASH, // known from the completion frame onward
    tpsId: `tps-${payload.request_id}`,
    tps: payload.tok_s != null && payload.tok_s > 0 ? fmt1(payload.tok_s) : DASH,
    spark: { points: sparkPoints(payload.samples) },
    elapsedId: `el-${payload.request_id}`,
    elapsed: fmtDuration(elapsedMs),
    tagClass: "stream",
    tagText: generated === 0 ? "prefill" : "streaming",
  };
}

function buildRecentRow(payload: WireRequest): RecentRowView {
  const status = recentStatus(payload);
  return {
    id: shortId(payload.request_id),
    model: payload.model,
    statusClass: status.statusClass,
    statusText: status.statusText,
    finish: FINISH_LABELS[payload.finish_reason ?? ""] ?? (payload.finish_reason || DASH),
    tokPC: `${payload.prompt_tokens} / ${payload.generated_tokens ?? 0}`,
    avgTps: payload.tok_s != null && payload.tok_s > 0 ? fmt1(payload.tok_s) : DASH,
    spark: { points: sparkPoints(payload.samples) },
    duration: fmtDuration(payload.elapsed_ms ?? 0),
  };
}

// ---- EventSource client --------------------------------------------------------------------------

let mountedAtMs = Date.now();

class LiveClient {
  private source: EventSource | null = null;
  private reconnectDelayMs = 1_000;
  private connectTimer: ReturnType<typeof setTimeout> | null = null;
  private started = false;

  constructor(private readonly store: MonitorStore) {}

  start(): void {
    if (this.started) return; // idempotent across StrictMode double-mounts
    this.started = true;
    mountedAtMs = Date.now();
    void this.fetchModelsOnce();
    setInterval(() => this.store.tick(), 1_000); // locally ticking elapsed timers and the header uptime clock
    this.connect();
  }

  private connect(): void {
    if (this.source) return;
    const source = new EventSource(STREAM_PATH); // Last-Event-ID reconnect semantics are native: the browser replays its last frame id automatically.

    source.addEventListener("open", () => {
      this.reconnectDelayMs = 1_000;
      this.store.setConnection("live");
    });

    source.addEventListener("error", () => {
      if (source.readyState === EventSource.CONNECTING) {
        // Native auto-reconnect is in progress.
        this.store.setConnection("reconnecting");
        return;
      }
      // CLOSED: native reconnect stopped — schedule a manual reopen with capped backoff. A fresh connect without
      // Last-Event-ID would skip history, but the browser keeps sending it for automatic retries only, so the
      // manual path relies on the resync every stale/unknown-id connect already receives server-side.
      this.source = null;
      source.close();
      this.store.setConnection("offline");
      if (this.connectTimer === null) {
        const delay = Math.min(this.reconnectDelayMs, 8_000);
        this.reconnectDelayMs *= 2;
        this.connectTimer = setTimeout(() => {
          this.connectTimer = null;
          this.connect();
        }, delay);
      }
    });

    source.addEventListener("snapshot", (event: MessageEvent<string>) => {
      const snapshot = parseJson<WireSnapshot>(event.data);
      if (snapshot) this.store.acceptSnapshot(snapshot);
    });

    source.addEventListener("request_start", (event: MessageEvent<string>) => {
      const payload = parseJson<WireRequest>(event.data);
      if (payload) this.store.acceptRequestStart(payload);
    });

    source.addEventListener("request_update", (event: MessageEvent<string>) => {
      const payload = parseJson<WireRequest>(event.data);
      if (payload) this.store.acceptRequestUpdate(payload);
    });

    source.addEventListener("request_complete", (event: MessageEvent<string>) => {
      const payload = parseJson<WireRequest>(event.data);
      if (payload) this.store.acceptRequestComplete(payload);
    });

    source.addEventListener("resync", (event: MessageEvent<string>) => {
      const resync = parseJson<WireResync>(event.data);
      if (resync) this.store.acceptResync(resync);
    });

    this.source = source;
  }

  private async fetchModelsOnce(): Promise<void> {
    try {
      const response = await fetch(MODELS_PATH);
      if (!response.ok) return;
      const body = (await response.json()) as { data?: Array<{ id: string; quant?: string | null; context_length?: number }> };
      const identities: ModelIdentity[] = (body.data ?? []).map((entry) => ({
        id: entry.id,
        quant: entry.quant ?? DASH,
        contextLength: entry.context_length ?? 0,
      }));
      this.store.acceptModels(identities);
    } catch {
      // Identity stays snapshot-derived; the stream alone still builds every row.
    }
  }
}

function parseJson<T>(text: string): T | null {
  try {
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}

// ---- React binding -------------------------------------------------------------------------------

const monitorStore = new MonitorStore();
export const liveClient = new LiveClient(monitorStore);

function subscribeMonitor(listener: () => void): () => void {
  liveClient.start(); // starting the subscription is what starts the stream — once, ever
  return monitorStore.subscribe(listener);
}

export function useMonitor(): MonitorView {
  const getSnapshot = useCallback(() => monitorStore.getSnapshot(), []);
  return useSyncExternalStore(subscribeMonitor, getSnapshot, getSnapshot);
}
