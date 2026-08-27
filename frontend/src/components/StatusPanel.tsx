import type { StatusResponse, Reading } from "../types";

interface Props {
  status: StatusResponse | null;
  readings: Reading[];
}

function severityColor(label: string): string {
  if (label.includes("Stoppage") || label === "Severe") return "text-severe";
  if (label === "Mild") return "text-mild";
  return "text-ok";
}

function recommendedAction(label: string): string {
  if (label.includes("Stoppage")) {
    return "Inspect immediately — equipment may have stopped or failed.";
  }
  if (label === "Severe") {
    return "Schedule an emergency shutdown for maintenance as soon as possible.";
  }
  if (label === "Mild") {
    return "Monitor closely — schedule maintenance during the next planned downtime.";
  }
  return "No action needed — operating within normal range.";
}

export default function StatusPanel({ status, readings }: Props) {
  if (!status) {
    return (
      <div className="bg-panel border border-border rounded-md p-5 font-mono text-text-muted text-sm">
        Loading status…
      </div>
    );
  }

  const peakRms = readings.length
    ? Math.max(...readings.map((r) => r.rms))
    : status.recent_peak_rms;
  const severeCount = readings.filter((r) => r.severity === "Severe").length;
  const anomalyCount = readings.filter((r) => r.anomaly_type !== "Normal").length;

  const colorClass = severityColor(status.status);

  return (
    <div className="bg-panel border border-border rounded-md p-5 flex flex-col gap-5">
      <div>
        <div className="text-xs uppercase tracking-wider text-text-muted font-mono mb-1">
          Latest RMS
        </div>
        <div className={`font-mono text-4xl font-semibold ${colorClass}`}>
          {status.latest_rms.toFixed(4)}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <span
          className={`inline-block w-2.5 h-2.5 rounded-full ${
            status.requires_attention ? "bg-severe animate-pulse" : "bg-ok"
          }`}
        />
        <span className={`font-mono text-sm ${colorClass}`}>{status.status}</span>
      </div>

      <div className="grid grid-cols-2 gap-4 pt-3 border-t border-border">
        <Stat label="Peak RMS" value={peakRms.toFixed(4)} />
        <Stat label="Threshold" value={status.alert_threshold_rms.toFixed(4)} />
        <Stat label="Anomalies" value={String(anomalyCount)} />
        <Stat label="Severe" value={String(severeCount)} valueClass="text-severe" />
      </div>

      <div className="pt-3 border-t border-border">
        <div className="text-[10px] uppercase tracking-wider text-text-muted font-mono mb-1.5">
          Recommended Action
        </div>
        <div className={`text-sm leading-snug ${colorClass}`}>
          {recommendedAction(status.status)}
        </div>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  valueClass = "text-text",
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-text-muted font-mono mb-0.5">
        {label}
      </div>
      <div className={`font-mono text-lg ${valueClass}`}>{value}</div>
    </div>
  );
}