import { useState } from "react";
import type { ModelRowData } from "../../data/placeholder";
import { models } from "../../data/placeholder";

export function ModelsTable() {
  const [expanded, setExpanded] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(models.map((model) => [model.id, model.expandedByDefault])),
  );

  return (
    <section className="block">
      <h2>Configured models</h2>
      <table>
        <thead>
          <tr>
            <th></th>
            <th>model</th>
            <th>device</th>
            <th>decode tok/s</th>
            <th>seqs/step</th>
            <th>KV used</th>
            <th>cache hit</th>
            <th>workers</th>
            <th>VRAM</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <ModelRows key={model.id} model={model} open={expanded[model.id] === true} onToggle={() => setExpanded((current) => ({ ...current, [model.id]: !(current[model.id] === true) }))} />
          ))}
        </tbody>
      </table>
    </section>
  );
}

interface ModelRowsProps {
  model: ModelRowData;
  open: boolean;
  onToggle: () => void;
}

function ModelRows({ model, open, onToggle }: ModelRowsProps) {
  return (
    <>
      <tr className="model-row">
        <td className="expand" title="expand" aria-expanded={open} onClick={onToggle}>
          {open ? "▾" : "▸"}
        </td>
        <td>
          {model.name}
          <br />
          <span className="meta">{model.kind} · {model.quant}</span>
        </td>
        <td>{model.device}</td>
        <td>{model.decodeTps}</td>
        <td>{model.seqsPerStep}</td>
        <td>
          {model.kvBarPct === null ? (
            <span className="meta">{model.kvText}</span>
          ) : (
            <>
              <div className="bar kv"><span style={{ width: `${model.kvBarPct}%` }} /></div>
              {model.kvText}
            </>
          )}
        </td>
        <td>{model.cacheHit}</td>
        <td>{model.workers}</td>
        <td><span className="meta">{model.vram}</span></td>
      </tr>
      <tr className="detail-row" hidden={!open}>
        <td></td>
        <td colSpan={8}>
          <div className="detail-grid">
            {model.detail.map((field) => (
              <div key={field.label}>
                <div className="label">{field.label}</div>
                <div className="val">{field.value}</div>
              </div>
            ))}
          </div>
        </td>
      </tr>
    </>
  );
}
