"""
Step 5 — The Agent Layer.

Takes a plain-English question, lets Groq's LLM decide which backend
tool to call (and with what arguments), executes that tool against
Supabase, and asks the LLM to explain the real result in plain language.

The LLM never invents numbers here — it only ever calls a function
that reads actual stored data, then describes what came back.
"""

import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

import db_tools
import ml_tools

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
# "llama-3.1-8b-instant" has been retired by Groq and is no longer served
# to this key (confirmed via client.models.list()); openai/gpt-oss-20b is
# the closest fast/cheap replacement that still produces well-formed tool
# calls against our schema.
MODEL = "openai/gpt-oss-20b"

# This dataset is from Feb 2004 — the LLM needs to know that "now" /
# "today" / "last 3 days" should be interpreted relative to the last
# real reading in the data, not the actual calendar date.
SYSTEM_PROMPT = """You are a monitoring assistant for industrial bearing vibration data.

The current reference time ("now") is 2004-02-19 06:22:39 — this is the
timestamp of the most recent sensor reading. Interpret all relative time
expressions (e.g. "last 3 days", "past week", "yesterday") relative to
this reference time, not the real-world calendar date.

The only bearing currently being monitored is "bearing_1".

For trend-related questions ("how has it been trending", "summarize the
last week"), use get_health_summary — it already includes direction,
peak value, and anomaly counts. The full point-by-point chart is shown
separately on the dashboard; you do not need, and should never request,
every individual raw reading.

Always use the available tools to answer questions — never guess or
invent numbers.

Two of the tools, predict_fault_type and estimate_rul, are backed by
trained machine-learning models (Random Forest), not the rule-based
z-score logic the others use. predict_fault_type only ever returns
"Normal" or "Anomalous" — it is a health-state classifier, NOT a
fault-type identifier, so never claim it detected a specific fault like
"inner race" or "ball defect"; if asked which specific fault type is
present, say plainly that this system can only confirm outer-race
defects (the one fault type it has real training examples of) and
cannot distinguish other fault types. estimate_rul gives a rough
remaining-hours estimate with real but limited accuracy (roughly ±28
hours on this dataset) — present it as a planning signal, not a precise
countdown.

CRITICAL RULE: if a tool result contains "requires_attention": true, or
"ended_with_suspected_equipment_stoppage": true, or a "status" field that
is not simply "Normal", you MUST lead your answer with that warning,
explicitly and in plain words. Never summarize a reading as fine or
normal just because one raw number (like RMS) happens to look low — a
sudden drop right after severe readings is itself the warning sign of
equipment failure, not reassurance. Omitting this warning is a critical
failure of your job.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_anomalies",
            "description": "Get only the flagged anomalies for a bearing between two dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bearing_id": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["bearing_id", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_threshold_status",
            "description": "Check whether a bearing's latest reading is above the alert threshold right now.",
            "parameters": {
                "type": "object",
                "properties": {"bearing_id": {"type": "string"}},
                "required": ["bearing_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_health_summary",
            "description": "Get an overall health summary (trend direction, peak severity, anomaly count) for a bearing over a recent window. Use this for any trend or 'how has it been' style question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bearing_id": {"type": "string"},
                    "window_days": {"type": "integer", "description": "How many days back from now to summarize"},
                },
                "required": ["bearing_id", "window_days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_fault_type",
            "description": (
                "Use the trained Random Forest ML model to classify a bearing's current "
                "health state (Normal vs Anomalous) from its latest reading. This is a "
                "binary health-state classifier, NOT a specific fault-type (inner race / "
                "outer race / ball) identifier — use it for 'is there a fault right now' "
                "style questions, not 'what type of fault'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"bearing_id": {"type": "string"}},
                "required": ["bearing_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_rul",
            "description": (
                "Use the trained Random Forest ML model to estimate Remaining Useful Life "
                "(hours until likely failure) from a bearing's latest reading. Real but "
                "approximate (~28 hour MAE on this dataset) — treat as a planning signal."
            ),
            "parameters": {
                "type": "object",
                "properties": {"bearing_id": {"type": "string"}},
                "required": ["bearing_id"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "get_anomalies": db_tools.get_anomalies,
    "get_threshold_status": db_tools.get_threshold_status,
    "get_health_summary": db_tools.get_health_summary,
    "predict_fault_type": ml_tools.predict_fault_type,
    "estimate_rul": ml_tools.estimate_rul,
}

# Generic safety net: no matter which tool ran or how it was called, never
# hand the LLM a result bigger than this many characters. If a tool result
# turns out to be unexpectedly large (e.g. a wide date range), strip any
# list-valued fields and keep only the summary scalars instead of crashing
# with a token-limit error.
MAX_TOOL_RESULT_CHARS = 4000


def _safe_tool_content(result: dict) -> str:
    content = json.dumps(result, default=str)
    if len(content) <= MAX_TOOL_RESULT_CHARS:
        return content

    summary = {k: v for k, v in result.items() if not isinstance(v, list)}
    summary["_note"] = "Full result was too large to include; showing summary fields only."
    return json.dumps(summary, default=str)


# A question can legitimately need more than one tool now (e.g. "what's
# the predicted fault type AND the remaining useful life" pulls from two
# separate ML tools) -- cap the back-and-forth so a confused model can't
# loop forever, but allow more than one round.
MAX_TOOL_ROUNDS = 4


def ask_agent(user_question: str, max_retries: int = 2) -> str:
    """
    Sends the question to the LLM with tools available, executes whichever
    tool(s) it calls, feeds the real results back, and repeats until the
    model answers in plain text instead of calling another tool (or the
    round cap is hit).

    Small/fast models occasionally generate a malformed function call
    instead of a properly structured one — this is a known reliability
    quirk, not a bug in our code. We retry a couple of times (a fresh
    generation usually succeeds) and use a lower temperature, which makes
    structured tool-call output noticeably more consistent.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    for round_num in range(MAX_TOOL_ROUNDS):
        message = None
        for attempt in range(max_retries + 1):
            try:
                # tools/tool_choice stay on every round -- Groq rejects a
                # response with tool_calls if the request that produced it
                # didn't offer tools, so once tools are on they need to stay
                # on until the model actually stops calling them.
                response = client.chat.completions.create(
                    model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
                    temperature=0.2,
                )
                message = response.choices[0].message
                break
            except Exception as e:
                print(f"  [tool-call generation failed, attempt {attempt + 1}/{max_retries + 1}: {e}]")
                if attempt == max_retries:
                    return "Sorry — I had trouble generating a valid tool call for that question after a few attempts. Please try rephrasing it."
                time.sleep(1)

        if not message.tool_calls:
            # The model has what it needs and answered in plain text.
            return message.content

        messages.append(message)

        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"  [agent called {func_name}({args})]")

            func = TOOL_REGISTRY[func_name]
            result = func(**args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": _safe_tool_content(result),
            })

    # Hit the round cap while the model still wanted another tool call --
    # ask once more without tools available so it's forced to answer in
    # words from whatever real results it already has.
    final_response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.2,
    )
    return final_response.choices[0].message.content


if __name__ == "__main__":
    import sys
    # Windows terminals often default stdout to cp1252, which can't encode
    # the warning emoji/em-dashes the LLM sometimes uses — force UTF-8 so
    # running this file directly doesn't crash on a plain print().
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_questions = [
        "Is bearing 1 currently above the alert threshold?",
        "Summarize the health of bearing 1 for the last 3 days.",
        "What anomalies were detected between Feb 16 and Feb 19?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        answer = ask_agent(q)
        print(f"A: {answer}")