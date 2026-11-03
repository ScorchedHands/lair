import type { SparkView } from "../../monitor/live";

export function Sparkline({ spark }: { spark: SparkView }) {
  return (
    <svg className="spark" width={90} height={18} viewBox="0 0 90 18">
      <polyline points={spark.points} />
    </svg>
  );
}
