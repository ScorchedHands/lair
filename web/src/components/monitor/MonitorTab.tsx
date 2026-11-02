import { useMonitor } from "../../monitor/live";
import { kpiCards, placeholderCards } from "../../data/placeholder";
import { KpiCard } from "./KpiCard";
import { ModelsTable } from "./ModelsTable";
import { InFlightTable } from "./InFlightTable";
import { RecentTable } from "./RecentTable";

export function MonitorTab({ hidden }: { hidden: boolean }) {
  const view = useMonitor();

  return (
    <section id="tab-monitor" hidden={hidden}>
      <div className="grid">
        {kpiCards.map((card) => {
          const live = view.kpis[card.id];
          return <KpiCard key={card.id} card={live ? { ...card, value: live.value, sub: live.sub } : card} />;
        })}
      </div>

      <ModelsTable models={view.models} />
      <InFlightTable rows={view.inFlight} />
      <RecentTable rows={view.recent} />

      <div className="grid">
        <KpiCard card={placeholderCards[0]!} placeholder />
        <KpiCard card={placeholderCards[1]!} placeholder />
        {view.processRam ? (
          <KpiCard card={{ ...placeholderCards[2]!, value: view.processRam.value, sub: view.processRam.sub }} />
        ) : (
          <KpiCard card={{ ...placeholderCards[2]!, value: "—" }} />
        )}
        <KpiCard card={placeholderCards[3]!} placeholder />
      </div>
    </section>
  );
}
