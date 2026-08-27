import { useState, useRef, useEffect } from "react";
import { askAgent } from "../api";
import type { ChatMessage } from "../types";

const SUGGESTIONS = [
  "Is bearing 1 currently above the alert threshold?",
  "Summarize the health of bearing 1 for the last 3 days.",
  "What anomalies were detected between Feb 16 and Feb 19?",
];

export default function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(question: string) {
    if (!question.trim() || loading) return;
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    try {
      const answer = await askAgent(question);
      setMessages((prev) => [...prev, { role: "agent", content: answer }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "agent", content: "Error reaching the agent. Is the backend running?" },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-panel border border-border rounded-md flex flex-col h-[420px]">
      <div className="text-xs uppercase tracking-wider text-text-muted font-mono px-5 pt-5 pb-3 border-b border-border">
        Ask the Agent
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-3 font-mono text-sm space-y-3">
        {messages.length === 0 && (
          <div className="space-y-2">
            <div className="text-text-muted">Try asking:</div>
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => send(s)}
                className="block text-left text-text-muted hover:text-ok transition-colors"
              >
                &gt; {s}
              </button>
            ))}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i}>
            {m.role === "user" ? (
              <div className="text-text">
                <span className="text-ok">&gt; </span>
                {m.content}
              </div>
            ) : (
              <div
                className={
                  m.content.toUpperCase().includes("WARNING")
                    ? "text-severe whitespace-pre-wrap"
                    : "text-text-muted whitespace-pre-wrap"
                }
              >
                {m.content}
              </div>
            )}
          </div>
        ))}

        {loading && <div className="text-text-muted animate-pulse">thinking…</div>}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex border-t border-border"
      >
        <span className="px-3 py-3 font-mono text-ok">&gt;</span>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about bearing_1…"
          className="flex-1 bg-transparent font-mono text-sm text-text placeholder:text-text-muted py-3 pr-4 outline-none"
        />
      </form>
    </div>
  );
}