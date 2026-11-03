import type { RecentRowView } from "../../monitor/live";
import { Sparkline } from "./Sparkline";

export function RecentTable({ rows }: { rows: RecentRowView[] }) {
  return (
    <section className="block">
      <h2>Recent requests (X = 5, configurable)</h2>
      <table>
        <thead>
          <tr>
            <th>id</th>
            <th>model</th>
            <th>status</th>
            <th>finish</th>
            <th>tok p/c</th>
            <th>avg tok/s</th>
            <th></th>
            <th>duration</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>{row.id}</td>
              <td>{row.model}</td>
              <td><span className={`tag ${row.statusClass}`}>{row.statusText}</span></td>
              <td>{row.finish}</td>
              <td>{row.tokPC}</td>
              <td>{row.avgTps}</td>
              <td><Sparkline spark={row.spark} /></td>
              <td>{row.duration}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
