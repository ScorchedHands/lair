import { kpiCards, placeholderCards } from "../../data/placeholder";
import { KpiCard } from "./KpiCard";
import { ModelsTable } from "./ModelsTable";
import { InFlightTable } from "./InFlightTable";
import { RecentTable } from "./RecentTable";

export function MonitorTab({ hidden }: { hidden: boolean }) {
  return (
    <section id="tab-monitor" hidden={hidden}>
      <div className="grid">
        {kpiCards.map((card) => (
          <KpiCard key={card.id} card={card} />
        ))}
      </div>

      <ModelsTable />
      <InFlightTable />
      <RecentTable />

      <div className="grid">
        <KpiCard card={placeholderCards[0]!} placeholder />
        <KpiCard card={placeholderCards[1]!} placeholder />
        <KpiCard card={placeholderCards[2]!} />
        <KpiCard card={placeholderCards[3]!} />
      </div>
    </section>
  );
}
