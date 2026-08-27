import {
  ComposedChart,
  Line,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { Reading } from "../types";

interface Props {
  readings: Reading[];
  threshold: number;
}

interface ChartPoint extends Reading {
  t: number; // epoch ms — numeric X axis avoids Scatter/Line misalignment
}

function formatTick(t: number): string {
  return new Date(t).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null;
  const point: ChartPoint = payload[0].payload;
  return (
    <div className="bg-panel border border-border rounded-md px-3 py-2 font-mono text-xs">
      <div className="text-text-muted mb-1">
        {new Date(point.reading_time).toLocaleString()}
      </div>
      <div className="text-text">RMS: {point.rms.toFixed(4)}</div>
      {point.severity !== "Normal" && (
        <div className={point.severity === "Severe" ? "text-severe" : "text-mild"}>
          {point.anomaly_type} ({point.severity})
        </div>
      )}
    </div>
  );
}

export default function TrendChart({ readings, threshold }: Props) {
  const chartData: ChartPoint[] = readings.map((r) => ({
    ...r,
    t: new Date(r.reading_time).getTime(),
  }));

  const mildPoints = chartData.filter((r) => r.severity === "Mild");
  const severePoints = chartData.filter((r) => r.severity === "Severe");

  return (
    <div className="bg-panel border border-border rounded-md p-5">
      <div className="text-xs uppercase tracking-wider text-text-muted font-mono mb-4">
        Bearing 1 — Vibration Trend (RMS)
      </div>
      <ResponsiveContainer width="100%" height={360}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
          <CartesianGrid stroke="#28323D" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={formatTick}
            stroke="#7E8A97"
            tick={{ fontFamily: "IBM Plex Mono", fontSize: 11 }}
            minTickGap={50}
          />
          <YAxis
            stroke="#7E8A97"
            tick={{ fontFamily: "IBM Plex Mono", fontSize: 11 }}
            width={50}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            y={threshold}
            stroke="#F2A93C"
            strokeDasharray="4 4"
            label={{
              value: `Alert Threshold (${threshold.toFixed(4)})`,
              position: "insideTopRight",
              fill: "#F2A93C",
              fontSize: 10,
              fontFamily: "IBM Plex Mono",
            }}
          />
          <Line
            type="monotone"
            dataKey="rms"
            stroke="#5C7A93"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Scatter data={mildPoints} dataKey="rms" fill="#F2A93C" />
          <Scatter data={severePoints} dataKey="rms" fill="#E5484D" />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="flex gap-5 mt-3 font-mono text-xs text-text-muted">
        <Legend color="#5C7A93" label="RMS" />
        <Legend color="#F2A93C" label="Mild" />
        <Legend color="#E5484D" label="Severe" />
      </div>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </div>
  );
}