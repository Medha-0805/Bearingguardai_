# BearingGuard AI

An LLM-powered monitoring dashboard for industrial bearing vibration data
(NASA IMS Bearing Test Set), built around a run-to-failure recording. It
detects and classifies anomalies (gradual degradation, outer race fault
onset) from vibration features, stores them in Supabase, and lets you
ask a chat agent plain-English questions about bearing health — the
agent calls real backend tools to answer, it never invents numbers.

## Architecture

- **Data pipeline** (`backend/explore_data.py`, `detect_anomalies.py`) —
  extracts RMS/kurtosis/BPFO-energy features and flags anomalies against
  a healthy baseline.
- **Time-series store** (`backend/push_to_supabase.py`) — loads the
  computed readings into a Supabase `bearing_readings` table.
- **Backend API** (`backend/main.py`, `db_tools.py`) — FastAPI service
  exposing `/trend`, `/status`, `/summary`, and `/chat`.
- **Agent layer** (`backend/agent.py`) — Groq-hosted LLM with function
  calling; decides which backend tool to call, then explains the real
  result in plain language.
- **Frontend** (`frontend/`) — React + TypeScript + Vite + Tailwind +
  Recharts dashboard with a chat panel.

## Setup

### Backend

```bash
cd backend
python -m venv venv
./venv/Scripts/activate   # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
cp .env.example .env      # fill in your Supabase + Groq credentials
python push_to_supabase.py  # one-time: load computed readings into Supabase
python main.py             # runs on http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev                # runs on http://localhost:5173
```

## Environment variables (`backend/.env`)

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Supabase service-role key (backend-only, bypasses RLS) |
| `GROQ_API_KEY` | Groq API key for the chat agent's LLM |
