import { useState } from "react";
import { Header } from "./components/Header";
import { Tabs } from "./components/Tabs";
import { MonitorTab } from "./components/monitor/MonitorTab";
import { ChatTab } from "./components/chat/ChatTab";

export type TabId = "monitor" | "chat";

export function App() {
  const [tab, setTab] = useState<TabId>("monitor");

  return (
    <>
      <Header />
      <Tabs active={tab} onSelect={setTab} />
      <main>
        <MonitorTab hidden={tab !== "monitor"} />
        <ChatTab hidden={tab !== "chat"} />
      </main>
    </>
  );
}
