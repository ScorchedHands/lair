// Empty-state defaults for the monitor surface (values read as em dashes until live stream frames arrive) and the
// static tiles that stay placeholders by design. All measured monitor data now comes from web/src/monitor/live.ts;
// only the chat tab's mockup content is still fed from here.

export interface KpiCardData {
  id: string;
  label: string;
  value: string;
  sub: string;
}

export const kpiCards: KpiCardData[] = [
  { id: "toks", label: "decode tok/s", value: "—", sub: "cpu in use" },
  { id: "stepms", label: "step time", value: "—", sub: "—" },
  { id: "hit", label: "cache hit", value: "—", sub: "engine-wide prefix reuse" },
  { id: "active", label: "active reqs", value: "—", sub: "queued —" },
];

export const placeholderCards: KpiCardData[] = [
  { id: "gpuutil", label: "GPU util", value: "—", sub: "awaiting CUDA phase" },
  { id: "pcie", label: "PCIe transfer", value: "—", sub: "awaiting CUDA phase" },
  { id: "ram", label: "process RAM", value: "—", sub: "engine working set" },
  { id: "cpupower", label: "CPU power", value: "—", sub: "not exposed on Windows (backlog)" },
];

export interface ChatMessageData {
  role: "user" | "assistant";
  streaming?: boolean;
  text: string;
  tool?: string;
}

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
