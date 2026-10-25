// Types and calls for the FastAPI service in api/main.py.

export type MethodId = "lightgbm" | "ma28";
export const METHOD_IDS: readonly MethodId[] = ["lightgbm", "ma28"];

export interface Question {
  variant: string;
  lead_time: number;
  target_fill_rate: number;
  holding_rate: number;
  penalty_fraction: number;
  stock_budget: number | null;
}

export interface Scenarios {
  methods: { id: MethodId; label: string }[];
  variants: { id: string; label: string }[];
  lead_times: number[];
  z_grid: number[];
  defaults: Omit<Question, "stock_budget">;
  preregistered: {
    targets: number[];
    holding_rate: number;
    penalty_fractions: number[];
  };
  limits: {
    holding_rate: [number, number];
    penalty_fraction: [number, number];
    stock_budget_max: number;
  };
  provenance: {
    protocol_version: string;
    issue: number;
    source_commit: string;
    series: number;
    evaluated_days: number;
    demand_units: number;
    evaluation_start: string;
    evaluation_end: string;
  };
}

export interface Policy {
  z: number;
  fill_rate: number;
  average_on_hand_units: number;
  holding_cost: number;
  penalty_cost: number;
  total_cost: number;
}

export type TargetStatus = "exact" | "interpolated" | "above_target_at_min_z" | "not_reached";

export interface StockForTarget {
  status: TargetStatus;
  units: number | null;
  lower_z: number | null;
  upper_z: number | null;
  first_reaching_z: number | null;
  max_fill_rate: number;
}

export type BudgetStatus = "exact" | "interpolated" | "below_grid" | "above_grid";

export interface FillForBudget {
  status: BudgetStatus;
  fill_rate: number | null;
  lower_z: number | null;
  upper_z: number | null;
  min_units: number;
  max_units: number;
}

export interface Cheapest {
  z: number;
  total_cost: number;
  holding_cost: number;
  penalty_cost: number;
  fill_rate: number;
  average_on_hand_units: number;
  grid_edge: "lowest_z" | "highest_z" | null;
}

export interface MethodAnswer {
  label: string;
  policies: Policy[];
  stock_for_target: StockForTarget;
  fill_for_budget: FillForBudget | null;
  cheapest: Cheapest;
}

export interface Answer {
  scenario: Question;
  methods: Record<MethodId, MethodAnswer>;
  comparison: {
    stock_difference_lightgbm_vs_ma28: number | null;
    stock_comparable: boolean;
    cheapest_cost_difference_lightgbm_vs_ma28: number | null;
  };
  reference: { label: string; fill_rate: number; average_on_hand_units: number };
  settings: {
    costs_preregistered: boolean;
    target_preregistered: boolean;
    primary_scenario: boolean;
    notes: string[];
  };
}

interface ValidationIssue {
  loc?: (string | number)[];
  msg?: string;
}

/** Turns FastAPI's error body (a string or a list of issues) into one sentence. */
export function errorMessage(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    const issues = (detail as ValidationIssue[]).map((issue) => {
      const field = issue.loc?.filter((part) => part !== "body").join(".");
      return field ? `${field}: ${issue.msg ?? "invalid"}` : (issue.msg ?? "invalid");
    });
    if (issues.length > 0) {
      return issues.join("; ");
    }
  }
  return `The server answered with status ${status}.`;
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { Accept: "application/json", ...init.headers },
  });
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(errorMessage(response.status, body));
  }
  return body as T;
}

export function fetchScenarios(signal?: AbortSignal): Promise<Scenarios> {
  return request<Scenarios>("api/scenarios", { signal });
}

export function askWhatIf(question: Question, signal?: AbortSignal): Promise<Answer> {
  return request<Answer>("api/what-if", {
    method: "POST",
    body: JSON.stringify(question),
    headers: { "Content-Type": "application/json" },
    signal,
  });
}

export interface AgentStatus {
  configured: boolean;
  provider: string | null;
  model: string | null;
  problem: string | null;
}

export interface AgentToolCall {
  name: string;
  arguments: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface AgentReply {
  answer: string;
  tool_calls: AgentToolCall[];
  grounded: boolean;
  unverified_numbers: string[];
  revised: boolean;
  stop_reason: "answered" | "step_limit";
  provider: string;
  model: string;
  suggested_question: Partial<Question>;
}

export function fetchAgentStatus(signal?: AbortSignal): Promise<AgentStatus> {
  return request<AgentStatus>("api/agent/status", { signal });
}

export function askAgent(question: string, signal?: AbortSignal): Promise<AgentReply> {
  return request<AgentReply>("api/agent", {
    method: "POST",
    body: JSON.stringify({ question }),
    headers: { "Content-Type": "application/json" },
    signal,
  });
}
