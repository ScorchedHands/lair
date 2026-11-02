import type { SparklineData } from "../../data/placeholder";

export function Sparkline({ spark }: { spark: SparklineData }) {
  return (
    <svg className="spark" width={90} height={18} viewBox="0 0 90 18">
      <polyline points={spark.points} />
    </svg>
  );
}
