import type { ChatMessageData } from "../../data/placeholder";
import { chatMessages, chatModels } from "../../data/placeholder";

export function ChatTab({ hidden }: { hidden: boolean }) {
  return (
    <section id="tab-chat" hidden={hidden}>
      <div className="chat">
        <div className="meta">
          model:{" "}
          <select
            id="chat-model"
            style={{
              background: "var(--panel-2)",
              color: "var(--text)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              padding: "2px 6px",
            }}
          >
            {chatModels.map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>{" "}
          · temp 0.7 · max tokens 256 · SSE streaming · selector lists LLM models only
        </div>
        <div className="messages">
          {chatMessages.map((message, index) => (
            <Message key={index} message={message} />
          ))}
        </div>
        <div className="composer">
          <input type="text" placeholder="Send a message…" defaultValue="" />
          <button className="btn stop" type="button">Stop</button>
          <button className="btn" type="button">Send</button>
        </div>
        <div className="meta">abort mid-stream keeps the engine healthy · tool calls rendered from the fenced-JSON detector</div>
      </div>
    </section>
  );
}

function Message({ message }: { message: ChatMessageData }) {
  const className = message.streaming ? "msg assistant streaming" : `msg ${message.role}`;
  return (
    <div className={className}>
      {message.text}
      {message.tool !== undefined && <div className="tool">{message.tool}</div>}
    </div>
  );
}
