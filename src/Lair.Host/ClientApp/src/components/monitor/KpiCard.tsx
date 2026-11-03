import type { KpiCardData } from "../../data/placeholder";

interface KpiCardProps {
  card: KpiCardData;
  placeholder?: boolean;
}

export function KpiCard({ card, placeholder = false }: KpiCardProps) {
  return (
    <div className={placeholder ? "card placeholder" : "card"}>
      <div className="label">{card.label}</div>
      <div className="value" id={card.id}>{card.value}</div>
      <div className="sub">{card.sub}</div>
    </div>
  );
}
