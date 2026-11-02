export interface KpiCardData {
  id: string;
  label: string;
  value: string;
  sub: string;
}

export interface DetailField {
  label: string;
  value: string;
}

export interface ModelRowData {
  id: string;
  name: string;
  kind: string;
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

export interface SparklineData {
  points: string;
}

export interface InFlightRowData {
  id: string;
  model: string;
  endpoint: string;
  promptTok: string;
  genTokId: string;
  genTok: string;
  cached: string;
  tpsId: string;
  tps: string;
  spark: SparklineData;
  elapsedId: string;
  elapsed: string;
  tagClass: string;
  tagText: string;
}

export interface RecentRowData {
  id: string;
  model: string;
  statusClass: string;
  statusText: string;
  finish: string;
  tokPC: string;
  avgTps: string;
  spark: SparklineData;
  duration: string;
}

export interface ChatMessageData {
  role: "user" | "assistant";
  streaming?: boolean;
  text: string;
  tool?: string;
}

export const kpiCards: KpiCardData[] = [
  { id: "toks", label: "decode tok/s", value: "27.4", sub: "cpu in use" },
  { id: "stepms", label: "step time", value: "36 ms", sub: "p95 41 ms" },
  { id: "hit", label: "cache hit", value: "62%", sub: "engine-wide prefix reuse" },
  { id: "active", label: "active reqs", value: "2", sub: "queued 0" },
];

export const models: ModelRowData[] = [
  {
    id: "smollm-135m-instruct",
    name: "smollm-135m-instruct",
    kind: "chat",
    quant: "q8_0",
    device: "cpu",
    decodeTps: "27.4",
    seqsPerStep: "2.00",
    kvBarPct: 56,
    kvText: "54/96",
    cacheHit: "62%",
    workers: "24",
    vram: "reserved —",
    expandedByDefault: true,
    detail: [
      { label: "context / block size", value: "1024 / 16" },
      { label: "kv_blocks budget", value: "96 (soft cap)" },
      { label: "core set / quota", value: "0-23 / —" },
      { label: "prefill · decode · kv residency", value: "cpu · cpu · dram" },
      { label: "cache blocks (DRAM)", value: "prefix 18 · active 36 · free 42" },
      { label: "finish tallies", value: "stop 12 · length 5 · abort 1 · tool_calls 2" },
      { label: "experts hot / cold", value: "n/a — dense model (MoE phase adds tier counts)" },
      { label: "model RAM", value: "1.9 GB weights · 0.5 GB KV" },
    ],
  },
  {
    id: "modernbert-embed",
    name: "modernbert-embed",
    kind: "embed",
    quant: "f16",
    device: "cpu",
    decodeTps: "—",
    seqsPerStep: "—",
    kvBarPct: null,
    kvText: "n/a",
    cacheHit: "—",
    workers: "4",
    vram: "reserved —",
    expandedByDefault: false,
    detail: [
      { label: "context / block size", value: "512 / —" },
      { label: "kv_blocks budget", value: "—" },
      { label: "core set / quota", value: "0-23 / —" },
      { label: "prefill · decode · kv residency", value: "cpu · cpu · dram" },
      { label: "cache blocks (DRAM)", value: "prefix — · active — · free —" },
      { label: "finish tallies", value: "stop 3 · length 0 · abort 0 · tool_calls 0" },
      { label: "experts hot / cold", value: "n/a — dense model (MoE phase adds tier counts)" },
      { label: "model RAM", value: "0.6 GB weights · 0.1 GB activations" },
    ],
  },
];

export const inFlightRows: InFlightRowData[] = [
  {
    id: "9f2c…a1",
    model: "smollm-135m",
    endpoint: "chat/completions",
    promptTok: "127",
    genTokId: "gen1",
    genTok: "18",
    cached: "112",
    tpsId: "tps1",
    tps: "28.1",
    spark: { points: "0,14 10,10 20,11 30,8 40,9 50,7 60,8 70,6 80,7 90,5" },
    elapsedId: "el1",
    elapsed: "1.2 s",
    tagClass: "stream",
    tagText: "streaming",
  },
  {
    id: "4b70…0d",
    model: "smollm-135m",
    endpoint: "completions",
    promptTok: "407",
    genTokId: "gen2",
    genTok: "3",
    cached: "0",
    tpsId: "tps2",
    tps: "—",
    spark: { points: "0,16 15,16 30,16 45,16 60,12 75,8 90,6" },
    elapsedId: "el2",
    elapsed: "2.8 s",
    tagClass: "stream",
    tagText: "prefill chunk 24/51",
  },
];

export const recentRows: RecentRowData[] = [
  {
    id: "77aa…31",
    model: "smollm-135m",
    statusClass: "done",
    statusText: "200",
    finish: "stop",
    tokPC: "31 / 24",
    avgTps: "22.7",
    spark: { points: "0,12 15,9 30,10 45,8 60,9 75,7 90,8" },
    duration: "1.1 s",
  },
  {
    id: "e0c4…9f",
    model: "smollm-135m",
    statusClass: "done",
    statusText: "200",
    finish: "length",
    tokPC: "407 / 32",
    avgTps: "7.0",
    spark: { points: "0,16 20,16 40,16 55,10 70,7 90,6" },
    duration: "4.6 s",
  },
  {
    id: "1b8d…55",
    model: "smollm-135m",
    statusClass: "err",
    statusText: "499",
    finish: "client abort",
    tokPC: "18 / 7",
    avgTps: "7.8",
    spark: { points: "0,10 20,8 40,9 60,7 80,8" },
    duration: "0.9 s",
  },
  {
    id: "c3f1…72",
    model: "smollm-135m",
    statusClass: "done",
    statusText: "200",
    finish: "tool_calls",
    tokPC: "96 / 21",
    avgTps: "15.0",
    spark: { points: "0,13 15,10 30,11 45,9 60,10 75,8 90,9" },
    duration: "1.4 s",
  },
  {
    id: "aa02…18",
    model: "smollm-135m",
    statusClass: "done",
    statusText: "200",
    finish: "stop",
    tokPC: "127 / 19",
    avgTps: "23.8",
    spark: { points: "0,11 15,9 30,9 45,8 60,8 75,7 90,7" },
    duration: "0.8 s",
  },
];

export const placeholderCards: KpiCardData[] = [
  { id: "gpuutil", label: "GPU util", value: "—", sub: "awaiting CUDA phase" },
  { id: "pcie", label: "PCIe transfer", value: "—", sub: "awaiting CUDA phase" },
  { id: "ram", label: "process RAM", value: "2.4 GB", sub: "engine working set" },
  { id: "cpupower", label: "CPU power", value: "—", sub: "not exposed on Windows (backlog)" },
];

export const chatModels: string[] = ["smollm-135m-instruct (chat)"];

export const chatMessages: ChatMessageData[] = [
  { role: "user", text: "Explain what a KV cache is in one short paragraph." },
  {
    role: "assistant",
    text: "A KV cache stores the key and value vectors computed for earlier tokens so the model does not recompute them when generating each new token. It grows with the sequence and is the main memory cost of inference.",
  },
  { role: "user", text: "Now use the calculator tool for 12*12." },
  {
    role: "assistant",
    text: "Computing that for you.",
    tool: 'tool call → calculator { "expression": "12*12" } · result 144',
  },
  { role: "user", text: "Write a haiku about paged memory." },
  { role: "assistant", streaming: true, text: "Blocks of sixteen tokens, mapped like pages of a book —" },
];
