import type { InFlightRowView } from "../../monitor/live";
import { Sparkline } from "./Sparkline";

export function InFlightTable({ rows }: { rows: InFlightRowView[] }) {
  return (
    <section className="block">
      <h2>In-flight requests</h2>
      <table>
        <thead>
          <tr>
            <th>id</th>
            <th>model</th>
            <th>endpoint</th>
            <th>prompt tok</th>
            <th>gen tok</th>
            <th>cached</th>
            <th>tok/s</th>
            <th></th>
            <th>elapsed</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>{row.id}</td>
              <td>{row.model}</td>
              <td>{row.endpoint}</td>
              <td>{row.promptTok}</td>
              <td id={row.genTokId}>{row.genTok}</td>
              <td>{row.cached}</td>
              <td id={row.tpsId}>{row.tps}</td>
              <td><Sparkline spark={row.spark} /></td>
              <td id={row.elapsedId}>{row.elapsed}</td>
              <td><span className={`tag ${row.tagClass}`}>{row.tagText}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
