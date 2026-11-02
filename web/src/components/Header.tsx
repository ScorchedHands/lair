import { useMonitor } from "../monitor/live";
import type { ConnectionState } from "../monitor/live";

// Badge copy per connection state: the stream is live once frames flow; a native auto-retry reads as reconnecting;
// a closed/retrying-externally socket reads as offline. The first dial (before any open) shows "live" optimistically —
// it resolves to one of the other two within milliseconds if the host is not there.
function badgeText(connection: ConnectionState): string {
  switch (connection) {
    case "reconnecting": return "reconnecting";
    case "offline": return "offline";
    default: return "live";
  }
}

function badgeClass(connection: ConnectionState): string {
  if (connection === "reconnecting") return "badge reconnect";
  if (connection === "offline") return "badge down";
  return "badge";
}

export function Header() {
  const view = useMonitor();

  return (
    <header className="app">
      <span className="brand">LAIR</span>
      <span className={badgeClass(view.connection)}>{badgeText(view.connection)}</span>
      <span style={{ marginLeft: "auto" }} className="meta">{view.meta}</span>
    </header>
  );
}
