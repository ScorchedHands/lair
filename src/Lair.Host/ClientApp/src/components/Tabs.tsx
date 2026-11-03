import type { TabId } from "../App";

interface TabsProps {
  active: TabId;
  onSelect: (tab: TabId) => void;
}

export function Tabs({ active, onSelect }: TabsProps) {
  return (
    <nav className="tabs" role="tablist">
      <button type="button" role="tab" aria-selected={active === "monitor"} onClick={() => onSelect("monitor")}>
        Monitor
      </button>
      <button type="button" role="tab" aria-selected={active === "chat"} onClick={() => onSelect("chat")}>
        Chat
      </button>
    </nav>
  );
}

