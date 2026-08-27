import axios from "axios";
import type { TrendResponse, StatusResponse } from "./types";

// In production, set VITE_API_BASE_URL (e.g. on Vercel) to the deployed
// backend's URL. Falls back to localhost for local dev.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function fetchTrend(): Promise<TrendResponse> {
  const res = await axios.get<TrendResponse>(`${API_BASE}/trend`);
  return res.data;
}

export async function fetchStatus(): Promise<StatusResponse> {
  const res = await axios.get<StatusResponse>(`${API_BASE}/status`);
  return res.data;
}

export async function askAgent(question: string): Promise<string> {
  const res = await axios.post<{ answer: string }>(`${API_BASE}/chat`, { question });
  return res.data.answer;
}