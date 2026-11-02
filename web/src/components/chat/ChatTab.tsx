import { useEffect, useRef, useState } from "react";
import type { ChatModelOption, ChatTurn, ToolCallView } from "../../chat/client";
import { ApiError, fetchChatModels, streamChatCompletion } from "../../chat/client";

interface RowUser {
  kind: "user";
  id: number;
  text: string;
}
interface RowAssistant {
  kind: "assistant";
  id: number;
  text: string;
  streaming: boolean;
  stopped: boolean;
  toolCalls: ToolCallView[];
}
interface RowError {
  kind: "error";
  id: number;
  line: string; // single-line inline error bubble: `error: <message> · code`
}
type ChatRow = RowUser | RowAssistant | RowError;

// The multi-turn conversation is in-memory per tab session and outlives re-renders and tab switches (the section
// stays mounted, only hidden). Assistant turns enter the sent history as their visible text only — tool cards are
// display-only under the client-side tools doctrine.
let nextRowId = 1;

export function ChatTab({ hidden }: { hidden: boolean }) {
  const [models, setModels] = useState<ChatModelOption[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [rows, setRows] = useState<ChatRow[]>([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);

  const historyRef = useRef<ChatTurn[]>([]); // authoritative multi-turn state sent with every request
  const controllerRef = useRef<AbortController | null>(null);
  const activeAssistantIdRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    void loadModels();

    async function loadModels(): Promise<void> {
      // One attempt now, capped backoff retries after — a page that loads before `lair serve` is ready must still
      // fill the selector once the engine comes up.
      for (let delayMs = 500; !cancelled; delayMs = Math.min(delayMs * 2, 8_000)) {
        try {
          const list = await fetchChatModels();
          if (cancelled) return;
          setModels(list);
          setSelectedModel((current) => current || (list[0]?.id ?? ""));
          return;
        } catch {
          await new Promise((resolve) => setTimeout(resolve, delayMs));
          if (cancelled) return;
        }
      }
    }

    return () => {
      cancelled = true; // StrictMode double-mount: the first pass stops retrying; a live stream is not torn down here
    };
  }, []);

  const canSend = !streaming && models.length > 0 && selectedModel !== "" && draft.trim().length > 0;

  function send(): void {
    if (!canSend) return; // mirrors the REPL: blank lines are ignored, never sent to the engine

    const text = draft.trim();
    setDraft("");
    historyRef.current.push({ role: "user", content: text });
    const userRowId = nextRowId++;
    const assistantId = nextRowId++;
    setRows((prev) => [
      ...prev,
      { kind: "user", id: userRowId, text },
      { kind: "assistant", id: assistantId, text: "", streaming: true, stopped: false, toolCalls: [] },
    ]);

    const controller = new AbortController();
    controllerRef.current = controller;
    activeAssistantIdRef.current = assistantId;
    setStreaming(true);

    void runStream(controller.signal).finally(() => {
      if (controllerRef.current === controller) controllerRef.current = null;
      setStreaming(false);
    });
  }

  async function runStream(signal: AbortSignal): Promise<void> {
    const assistantId = activeAssistantIdRef.current;
    try {
      await streamChatCompletion(
        selectedModel,
        [...historyRef.current], // the full messages array on every request
        {
          onText(piece) {
            patchAssistant(assistantId, (row) => ({ ...row, text: row.text + piece }));
          },
          onFinish() {
            // The finish reason is a server-side fact with no dedicated slot in the mockup layout; completion state
            // is expressed through caret removal and tool cards instead.
          },
          onComplete(toolCalls) {
            patchAssistant(assistantId, (row) => ({ ...row, streaming: false, toolCalls }));
            historyRef.current.push({ role: "assistant", content: assistantTextRef.current.get(assistantId) ?? "" });
          },
        },
        signal,
      );
    } catch (error: unknown) {
      // A failed exchange must not poison the conversation: drop it from the sent history exactly like the REPL
      // removes an offending turn so a context overflow cannot wedge every later send. The visible bubbles stay;
      // an assistant bubble that never received any text is removed rather than left as a ghost artifact.
      const last = historyRef.current.at(-1);
      if (last?.role === "user") historyRef.current.pop();

      patchAssistant(assistantId, (row) => ({ ...row, streaming: false })); // no caret on a dead turn
      setRows((prev) => [
        ...prev.filter((row) => !(row.kind === "assistant" && row.id === assistantId && row.text.length === 0)),
        { kind: "error", id: nextRowId++, line: errorLine(error) },
      ]);
    }
  }

  const assistantTextRef = useRef(new Map<number, string>());
  function patchAssistant(id: number, patch: (row: RowAssistant) => RowAssistant): void {
    setRows((prev) =>
      prev.map((row) => {
        if (row.kind !== "assistant" || row.id !== id) return row;
        const next = patch(row);
        assistantTextRef.current.set(id, next.text); // keep the text reachable for history on completion
        return next;
      }),
    );
  }

  function stop(): void {
    if (!streaming || controllerRef.current === null) return;
    const id = activeAssistantIdRef.current;
    patchAssistant(id, (row) => ({ ...row, streaming: false, stopped: true })); // partial text stays rendered
    historyRef.current.push({ role: "assistant", content: assistantTextRef.current.get(id) ?? "" });
    controllerRef.current.abort(); // server-side cancellation already exists; the next send must work after this
  }

  return (
    <section id="tab-chat" hidden={hidden}>
      <div className="chat">
        <div className="meta">
          model:{" "}
          <select
            id="chat-model"
            value={selectedModel}
            disabled={models.length === 0}
            onChange={(event) => setSelectedModel(event.target.value)}
            style={{
              background: "var(--panel-2)",
              color: "var(--text)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              padding: "2px 6px",
            }}
          >
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.quant ? `${model.id} (${model.type ?? "chat"} · ${model.quant.toLowerCase()})` : model.id}
              </option>
            ))}
          </select>{" "}
          · temp 0.7 · max tokens 256 · SSE streaming · selector lists LLM models only
        </div>
        <div className="messages">
          {rows.map((row) => (
            <Message key={row.id} row={row} />
          ))}
        </div>
        <div className="composer">
          <input
            type="text"
            placeholder="Send a message…"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") send();
            }}
          />
          <button className="btn stop" type="button" disabled={!streaming} onClick={stop}>
            Stop
          </button>
          <button className="btn" type="button" disabled={!canSend} onClick={send}>
            Send
          </button>
        </div>
        <div className="meta">abort mid-stream keeps the engine healthy · tool calls rendered from the fenced-JSON detector</div>
      </div>
    </section>
  );
}

function Message({ row }: { row: ChatRow }) {
  if (row.kind === "user") return <div className="msg user">{row.text}</div>;
  if (row.kind === "error") return <div className="msg error">{row.line}</div>;

  const className = row.streaming ? "msg assistant streaming" : "msg assistant";
  return (
    <div className={className}>
      {row.text}
      {row.stopped && <span className="stopped"> · stopped</span>}
      {row.toolCalls.map((call, index) => (
        <div key={index} className="tool">{`tool call → ${call.name} ${call.argumentsText}`}</div>
      ))}
    </div>
  );
}

function errorLine(error: unknown): string {
  if (error instanceof ApiError) return `error: ${error.message} · ${error.code}`;
  const message = error instanceof Error ? error.message : String(error);
  return `error: ${message}`;
}
