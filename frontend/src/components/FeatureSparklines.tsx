import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import type { Reading } from "../types";

interface Props {
  readings: Reading[];
}

interface Point {
  t: number;
  value: number;
}

function toSeries(readings: Reading[], key: "kurtosis" | "bpfo_energy"): Point[] {
  return readings.map((r) => ({ t: new Date(r.reading_time).getTime(), value: r[key] }));
}

function MiniChart({
  title,
  subtitle,
  data,
  color,
}: {
  title: string;
  subtitle: string;
  data: Point[];
  color: string;
}) {
  return (
    <div className="bg-panel border border-border rounded-md p-4 flex-1 min-w-0">
      <div className="text-[10px] uppercase tracking-wider text-text-muted font-mono mb-0.5">
        {title}
      </div>
      <div className="text-[10px] text-text-muted mb-2">{subtitle}</div>
      <ResponsiveContainer width="100%" height={110}>
        <LineChart data={data} margin={{ top: 5, right: 5, left: 5, bottom: 0 }}>
          <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} hide />
          <YAxis hide domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{
              background: "#161D24",
              border: "1px solid #28323D",
              fontFamily: "IBM Plex Mono",
              fontSize: 11,
            }}
            labelFormatter={(t) => new Date(t).toLocaleString()}
            formatter={(v) => [Number(v).toFixed(2), title]}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function FeatureSparklines({ readings }: Props) {
  return (
    <div className="flex flex-col md:flex-row gap-4 mb-6">
      <MiniChart
        title="Kurtosis"
        subtitle="Impulsiveness — spikes when impacts begin"
        data={toSeries(readings, "kurtosis")}
        color="#3fd8b8"
      />
      <MiniChart
        title="BPFO Spectral Energy"
        subtitle="Energy at the outer-race fault frequency (~236 Hz)"
        data={toSeries(readings, "bpfo_energy")}
        color="#5c7a93"
      />
    </div>
  );
}