"""
FastAPI Backend — exposes the agent as an HTTP API for the frontend.

This is the "Backend API" layer from the architecture document: it sits
between the Presentation Layer (React dashboard + chat UI) and the
Agent Layer (agent.py), which itself talks to the Time-Series Store
(Supabase) via db_tools.py.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agent
import db_tools

app = FastAPI(title="BearingGuard AI API")

# Allow the React dev server (Vite's default port is 5173; Create React
# App's is 3000) plus, in production, whatever origin(s) the deployed
# frontend is served from — set via FRONTEND_ORIGINS (comma-separated)
# on the host, e.g. FRONTEND_ORIGINS=https://your-app.vercel.app
extra_origins = [
    o.strip() for o in os.environ.get("FRONTEND_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", *extra_origins],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/trend")
def trend():
    """Raw readings for the dashboard chart — bypasses the LLM entirely."""
    # end_date is exclusive-ish in effect: a bare date compares at midnight,
    # so we use the 20th to make sure all of the 19th's readings (including
    # the actual failure spike at 05:42) are included.
    return db_tools.get_trend("bearing_1", "2004-02-12", "2004-02-20")


@app.get("/status")
def status():
    """Current threshold status for the dashboard's status panel."""
    return db_tools.get_threshold_status("bearing_1")


@app.get("/summary")
def summary(window_days: int = 3):
    """Health summary over a recent window, for the dashboard's status panel."""
    return db_tools.get_health_summary("bearing_1", window_days)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    The main agent endpoint. Takes a plain-English question, runs it
    through the agent (which decides which backend tool to call, if any),
    and returns the final answer.
    """
    answer = agent.ask_agent(request.question)
    return ChatResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn
    # Render (and most PaaS hosts) assign the port via $PORT at runtime;
    # 8000 stays the default for local dev.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))