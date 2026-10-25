import { useEffect, useState } from "react";

import { askWhatIf, fetchScenarios } from "./api.ts";
import type { Answer, Question, Scenarios } from "./api.ts";
import { AgentPanel } from "./components/AgentPanel.tsx";
import { CostChart, TradeoffChart } from "./components/Charts.tsx";
import { Controls } from "./components/Controls.tsx";
import { SettingsNote, StatCards } from "./components/Stats.tsx";
import { formatDateRange, formatUnits } from "./format.ts";

const REPOSITORY = "https://github.com/alennchik/supplypilot-ai";
const DEFAULT_BUDGET = 35_000;
const DEBOUNCE_MS = 150;

function startingQuestion(scenarios: Scenarios): Question {
  return { ...scenarios.defaults, stock_budget: DEFAULT_BUDGET };
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : "Something went wrong.";
}

function Logo() {
  return (
    <svg className="logo" width="30" height="30" viewBox="0 0 30 30" aria-hidden="true">
      <rect width="30" height="30" rx="8" />
      <path d="M7 19 L12 13 L16.5 16 L23 9" />
    </svg>
  );
}

export function App() {
  const [scenarios, setScenarios] = useState<Scenarios | null>(null);
  const [question, setQuestion] = useState<Question | null>(null);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetchScenarios(controller.signal)
      .then((loaded) => {
        setScenarios(loaded);
        setQuestion(startingQuestion(loaded));
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(`Could not load the published results: ${describeError(reason)}`);
        }
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (question === null) {
      return undefined;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setBusy(true);
      askWhatIf(question, controller.signal)
        .then((result) => {
          setAnswer(result);
          setError(null);
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
    }, DEBOUNCE_MS);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [question]);

  const provenance = scenarios?.provenance;

  function applyAgentScenario(patch: Partial<Question>) {
    setQuestion((current) => (current === null ? current : { ...current, ...patch }));
    document.querySelector(".controls")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="./">
            <Logo />
            <span>SupplyPilot</span>
          </a>
          <nav aria-label="Main">
            <a className="nav-link is-current" href="./" aria-current="page">
              What-if
            </a>
            <a className="nav-link" href="#assistant">
              Assistant
            </a>
            <a className="nav-link" href={`${REPOSITORY}#phase-3-inventory-simulation-frozen-protocol-v10`}>
              Method
            </a>
            <a className="nav-link" href={REPOSITORY}>
              GitHub
            </a>
          </nav>
        </div>
      </header>

      <main className="page">
        <section className="intro">
          <p className="eyebrow">Inventory what-if · replay of the preregistered Phase 3 simulation</p>
          <h1>How much stock does each forecast need?</h1>
          <p className="lede">
            {provenance
              ? `Pick a lead time, a service target and your costs. Every number comes from one registered simulation run: ${formatUnits(provenance.series)} Walmart M5 product-store series, weekly replenishment with lost sales, ${formatDateRange(provenance.evaluation_start, provenance.evaluation_end)}.`
              : "Loading the published results…"}
          </p>
        </section>

        {scenarios && question ? (
          <Controls
            scenarios={scenarios}
            question={question}
            onChange={setQuestion}
            onReset={() => setQuestion(startingQuestion(scenarios))}
          />
        ) : null}

        {error ? (
          <p className="alert" role="alert">
            {error}
          </p>
        ) : null}

        {answer ? (
          <div className={busy ? "results is-busy" : "results"} aria-busy={busy}>
            <SettingsNote settings={answer.settings} />
            <StatCards answer={answer} />
            <div className="charts">
              <TradeoffChart answer={answer} />
              <CostChart answer={answer} />
            </div>
          </div>
        ) : null}

        <AgentPanel onApply={applyAgentScenario} />
      </main>

      <footer className="footer">
        <p>
          {provenance
            ? `Protocol v${provenance.protocol_version} · one registered run (Issue #${provenance.issue}) at commit ${provenance.source_commit.slice(0, 7)} · ${formatUnits(provenance.demand_units)} demanded units over ${provenance.evaluated_days} days. `
            : ""}
          Interpolated values are estimates between two simulated policies. This replays a 2016 simulation;
          it is not a planning tool for a real business.
        </p>
        <p>
          <a href={REPOSITORY}>Source on GitHub</a>
        </p>
      </footer>
    </div>
  );
}
