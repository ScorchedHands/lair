// Empty-state defaults for the monitor surface (values read as em dashes until live stream frames arrive) and the
// static tiles that stay placeholders by design. All measured monitor data now comes from web/src/monitor/live.ts;
// the chat tab is fully wired to /v1/models + /v1/chat/completions (web/src/chat/client.ts), so nothing here feeds it.

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
