import type { Reading } from "../types";

interface Props {
  readings: Reading[];
}

export default function AnomalyLog({ readings }: Props) {
  const anomalies = readings
    .filter((r) => r.anomaly_type !== "Normal")
    .slice()
    .reverse(); // most recent first

  return (
    <div className="bg-panel border border-border rounded-md p-5 mb-6">
      <div className="text-xs uppercase tracking-wider text-text-muted font-mono mb-4">
        Anomaly Log — {anomalies.length} flagged readings
      </div>

      {anomalies.length === 0 ? (
        <div className="text-text-muted text-sm font-mono">No anomalies in range.</div>
      ) : (
        <div className="max-h-64 overflow-y-auto font-mono text-xs">
          <table className="w-full border-collapse">
            <thead>
              <tr className="text-left text-text-muted border-b border-border sticky top-0 bg-panel">
                <th className="py-2 pr-4 font-normal">Time</th>
                <th className="py-2 pr-4 font-normal">Type</th>
                <th className="py-2 pr-4 font-normal">Severity</th>
                <th className="py-2 font-normal">% Above Baseline</th>
              </tr>
            </thead>
            <tbody>
              {anomalies.map((a, i) => (
                <tr key={i} className="border-b border-border/40">
                  <td className="py-1.5 pr-4 text-text-muted whitespace-nowrap">
                    {new Date(a.reading_time).toLocaleString()}
                  </td>
                  <td className="py-1.5 pr-4 text-text">{a.anomaly_type}</td>
                  <td
                    className={`py-1.5 pr-4 ${
                      a.severity === "Severe" ? "text-severe" : "text-mild"
                    }`}
                  >
                    {a.severity}
                  </td>
                  <td className="py-1.5 text-text">
                    {a.pct_above_baseline != null ? `${a.pct_above_baseline.toFixed(1)}%` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}