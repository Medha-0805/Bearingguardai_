export interface Reading {
  reading_time: string;
  rms: number;
  kurtosis: number;
  bpfo_energy: number;
  anomaly_type: string;
  severity: "Normal" | "Mild" | "Severe";
  pct_above_baseline: number;
}

export interface TrendResponse {
  bearing_id: string;
  start_date: string;
  end_date: string;
  readings: Reading[];
}

export interface StatusResponse {
  bearing_id: string;
  latest_reading_time: string;
  latest_rms: number;
  alert_threshold_rms: number;
  is_above_threshold: boolean;
  margin_pct: number;
  recent_peak_rms: number;
  status: string;
  requires_attention: boolean;
}

export interface ChatMessage {
  role: "user" | "agent";
  content: string;
}

// From GET /predict/fault-type. Despite the endpoint name, the model is a
// binary Normal/Anomalous health-state classifier, not a multi-fault-type
// (inner race / outer race / ball) identifier — see backend/ml_tools.py.
export interface FaultTypePrediction {
  bearing_id: string;
  reading_time: string;
  predicted_health_state: "Normal" | "Anomalous";
  confidence: number;
  class_probabilities: Record<string, number>;
  model_scope_note: string;
}

// From GET /predict/rul.
export interface RulEstimate {
  bearing_id: string;
  reading_time: string;
  estimated_rul_hours: number;
  estimated_rul_days: number;
  model_scope_note: string;
}