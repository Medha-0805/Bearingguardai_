import { useEffect, useState } from "react";
import { fetchTrend, fetchStatus, fetchFaultTypePrediction, fetchRulEstimate } from "./api";
import type { Reading, StatusResponse, FaultTypePrediction, RulEstimate } from "./types";
import StatusPanel from "./components/StatusPanel";
import TrendChart from "./components/TrendChart";
import FeatureSparklines from "./components/FeatureSparklines";
import AnomalyLog from "./components/AnomalyLog";
import ChatPanel from "./components/ChatPanel";

export default function App() {
  const [readings, setReadings] = useState<Reading[]>([]);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [faultPrediction, setFaultPrediction] = useState<FaultTypePrediction | null>(null);
  const [rulEstimate, setRulEstimate] = useState<RulEstimate | null>(null);

  useEffect(() => {
    Promise.all([fetchTrend(), fetchStatus()])
      .then(([trendData, statusData]) => {
        setReadings(trendData.readings);
        setStatus(statusData);
      })
      .catch(() => setError("Could not reach the backend. Is it running on port 8000?"));

    // ML predictions are fetched separately and fail silently (leaving the
    // panel section simply hidden) — they're a bonus on top of the core
    // rule-based dashboard above, not something that should block it.
    fetchFaultTypePrediction().then(setFaultPrediction).catch(() => setFaultPrediction(null));
    fetchRulEstimate().then(setRulEstimate).catch(() => setRulEstimate(null));
  }, []);

  const threshold = status?.alert_threshold_rms ?? 0.1546;

  return (
    <div className="min-h-screen px-6 py-6 md:px-10 md:py-8">
      <header className="flex items-center justify-between mb-8 fade-in">
        <div>
          <h1 className="font-mono text-xl font-semibold tracking-tight">
            BEARINGGUARD AI
          </h1>
          <div className="text-text-muted text-sm">bearing_1 · IMS Test Set 2</div>
        </div>
        <div className="flex items-center gap-2 font-mono text-xs text-text-muted">
          <span className="inline-block w-2 h-2 rounded-full bg-ok" />
          MONITORING
        </div>
      </header>

      {error && (
        <div className="bg-panel border border-severe text-severe rounded-md p-4 font-mono text-sm mb-6">
          {error}
        </div>
      )}

      <div
        className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6 mb-6 fade-in"
        style={{ animationDelay: "60ms" }}
      >
        <TrendChart readings={readings} threshold={threshold} />
        <StatusPanel
          status={status}
          readings={readings}
          faultPrediction={faultPrediction}
          rulEstimate={rulEstimate}
        />
      </div>

      <div className="fade-in" style={{ animationDelay: "120ms" }}>
        <FeatureSparklines readings={readings} />
      </div>

      <div className="fade-in" style={{ animationDelay: "180ms" }}>
        <AnomalyLog readings={readings} />
      </div>

      <div className="fade-in" style={{ animationDelay: "240ms" }}>
        <ChatPanel />
      </div>
    </div>
  );
}