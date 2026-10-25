import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, FormEvent, KeyboardEvent } from "react";

import { askAgent, fetchAgentStatus } from "../api.ts";
import type { AgentReply, AgentStatus, AgentToolCall, Question } from "../api.ts";

const EXAMPLES = [
  "For each lead time, how much more stock does LightGBM need than MA28 for a 90% fill rate?",
  "Can LightGBM reach a 95% fill rate with a 7-day lead time?",
  "With 35,000 units on hand and a 7-day lead time, what fill rate does each method reach?",
  "If holding costs 30% a year and a lost sale costs 0.75 times the price, which method is cheaper at a 3-day lead time?",
];

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : "Something went wrong.";
}

function formatArguments(call: AgentToolCall): string {
  return Object.entries(call.arguments)
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(", ");
}

function ToolCalls({ calls }: { calls: AgentToolCall[] }) {
  if (calls.length === 0) {
    return <p className="agent-note">No tools were called for this answer.</p>;
  }
  return (
    <details className="agent-tools">
      <summary>
        {calls.length} tool {calls.length === 1 ? "call" : "calls"}
      </summary>
      <ol>
        {calls.map((call, index) => (
          <li key={index}>
            <code>
              {call.name}({formatArguments(call)})
            </code>
            {call.error ? (
              <p className="agent-error">Error: {call.error}</p>
            ) : (
              <pre>{JSON.stringify(call.result, null, 2)}</pre>
            )}
          </li>
        ))}
      </ol>
    </details>
  );
}

interface AnswerProps {
  question: string;
  reply: AgentReply;
  onApply: (patch: Partial<Question>) => void;
}

function Answer({ question, reply, onApply }: AnswerProps) {
  const canApply = Object.keys(reply.suggested_question).length > 0;
  return (
    <article className="agent-answer" aria-live="polite">
      <p className="agent-asked">{question}</p>
      <p className="agent-text">{reply.answer}</p>
      <div className="agent-meta">
        {reply.grounded ? (
          <span className="badge badge-ok">Every number checked against tool results</span>
        ) : (
          <span className="badge badge-warn">
            Unverified numbers: {reply.unverified_numbers.join(", ")}
          </span>
        )}
        {reply.revised ? <span className="agent-note">Revised once after a failed check.</span> : null}
        {canApply ? (
          <button type="button" className="button-secondary" onClick={() => onApply(reply.suggested_question)}>
            Show this on the charts
          </button>
        ) : null}
      </div>
      <ToolCalls calls={reply.tool_calls} />
    </article>
  );
}

export function AgentPanel({ onApply }: { onApply: (patch: Partial<Question>) => void }) {
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [draft, setDraft] = useState("");
  const [asked, setAsked] = useState("");
  const [reply, setReply] = useState<AgentReply | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const pending = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchAgentStatus(controller.signal)
      .then(setStatus)
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setStatus({ configured: false, provider: null, model: null, problem: describeError(reason) });
        }
      });
    return () => {
      controller.abort();
      pending.current?.abort();
    };
  }, []);

  const ready = status?.configured === true && !busy;

  function ask(question: string) {
    const text = question.trim();
    if (!text || !ready) {
      return;
    }
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setBusy(true);
    setError(null);
    askAgent(text, controller.signal)
      .then((result) => {
        setAsked(text);
        setReply(result);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(describeError(reason));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setBusy(false);
        }
      });
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    ask(draft);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      ask(draft);
    }
  }

  return (
    <section id="assistant" className="panel assistant" aria-labelledby="assistant-title">
      <div className="chart-head">
        <h2 id="assistant-title">Ask the planning assistant</h2>
        {status?.configured ? (
          <span className="badge badge-neutral">
            {status.provider} · {status.model}
          </span>
        ) : null}
      </div>
      <p className="chart-note">
        It answers from the same published results by calling the what-if tools, and every number in
        an answer is checked against what the tools returned. It covers this 2016 simulation only.
      </p>

      {status !== null && !status.configured ? (
        <p className="assistant-off">
          The assistant is off. Put <code>ANTHROPIC_API_KEY</code> (or <code>OPENAI_API_KEY</code>) in a{" "}
          <code>.env</code> file at the project root and restart the server.
          {status.problem ? ` (${status.problem})` : ""}
        </p>
      ) : null}

      <form className="assistant-form" onSubmit={onSubmit}>
        <label htmlFor="agent-question" className="visually-hidden">
          Your question
        </label>
        <textarea
          id="agent-question"
          rows={2}
          maxLength={500}
          value={draft}
          placeholder="Ask about stock, fill rates or costs for a lead time…"
          disabled={!status?.configured}
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setDraft(event.currentTarget.value)}
          onKeyDown={onKeyDown}
        />
        <button type="submit" className="button-primary" disabled={!ready || draft.trim() === ""}>
          {busy ? "Thinking…" : "Ask"}
        </button>
      </form>

      <div className="examples" role="group" aria-label="Example questions">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            className="chip chip-text"
            disabled={!ready}
            onClick={() => {
              setDraft(example);
              ask(example);
            }}
          >
            {example}
          </button>
        ))}
      </div>

      {error ? (
        <p className="alert" role="alert">
          {error}
        </p>
      ) : null}
      {reply ? <Answer question={asked} reply={reply} onApply={onApply} /> : null}
    </section>
  );
}
