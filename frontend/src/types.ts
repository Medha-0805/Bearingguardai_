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