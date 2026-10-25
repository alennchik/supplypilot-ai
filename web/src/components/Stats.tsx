import { METHOD_IDS } from "../api.ts";
import type { Answer, Cheapest, FillForBudget, MethodId, StockForTarget } from "../api.ts";
import { formatMoney, formatPercent, formatPercentAuto, formatUnits, formatZ } from "../format.ts";

interface Row {
  id: MethodId;
  label: string;
  value: string;
  detail: string;
}

function MethodRows({ rows }: { rows: Row[] }) {
  return (
    <dl className="method-rows">
      {rows.map((row) => (
        <div key={row.id} className="method-row">
          <dt>
            <span className={`legend-swatch legend-dot tone-${row.id}`} aria-hidden="true" />
            {row.label}
          </dt>
          <dd>
            <span className="stat-value">{row.value}</span>
            <span className="stat-detail">{row.detail}</span>
          </dd>
        </div>
      ))}
    </dl>
  );
}

function stockRow(stock: StockForTarget, highestZ: number): Pick<Row, "value" | "detail"> {
  switch (stock.status) {
    case "exact":
      return { value: `${formatUnits(stock.units ?? 0)} units`, detail: `simulated at z = ${formatZ(stock.lower_z ?? 0)}` };
    case "interpolated":
      return {
        value: `${formatUnits(stock.units ?? 0)} units`,
        detail: `between z = ${formatZ(stock.lower_z ?? 0)} and ${formatZ(stock.upper_z ?? 0)}`,
      };
    case "above_target_at_min_z":
      return { value: "Above target at z = 0", detail: "no stock estimate below the grid" };
    case "not_reached":
      return {
        value: "Not reached",
        detail: `best ${formatPercent(stock.max_fill_rate)} at z = ${formatZ(highestZ)}`,
      };
  }
}

function budgetRow(fill: FillForBudget | null): Pick<Row, "value" | "detail"> {
  if (fill === null) {
    return { value: "—", detail: "no budget set" };
  }
  switch (fill.status) {
    case "exact":
    case "interpolated":
      return {
        value: formatPercent(fill.fill_rate ?? 0),
        detail:
          fill.status === "exact"
            ? `simulated at z = ${formatZ(fill.lower_z ?? 0)}`
            : `between z = ${formatZ(fill.lower_z ?? 0)} and ${formatZ(fill.upper_z ?? 0)}`,
      };
    case "below_grid":
      return { value: "Below tested range", detail: `z = 0 already holds ${formatUnits(fill.min_units)}` };
    case "above_grid":
      return { value: "Above tested range", detail: `z = 3.0 holds ${formatUnits(fill.max_units)}` };
  }
}

function cheapestRow(cheapest: Cheapest): Pick<Row, "value" | "detail"> {
  return {
    value: formatMoney(cheapest.total_cost),
    detail: `z = ${formatZ(cheapest.z)} · ${formatPercent(cheapest.fill_rate)} fill`,
  };
}

function moreOrLess(value: number): string {
  return `${formatPercent(Math.abs(value))} ${value >= 0 ? "more" : "less"}`;
}

export function StatCards({ answer }: { answer: Answer }) {
  const { scenario, methods, comparison, reference } = answer;
  const target = formatPercentAuto(scenario.target_fill_rate);
  const highestZ = Math.max(...methods.lightgbm.policies.map((policy) => policy.z));
  const rows = (pick: (id: MethodId) => Pick<Row, "value" | "detail">): Row[] =>
    METHOD_IDS.map((id) => ({ id, label: methods[id].label, ...pick(id) }));

  const stockDifference = comparison.stock_difference_lightgbm_vs_ma28;
  const missing = METHOD_IDS.filter((id) => methods[id].stock_for_target.units === null).map(
    (id) => methods[id].label,
  );
  const stockFoot =
    stockDifference !== null
      ? `LightGBM needs ${moreOrLess(stockDifference)} stock than MA28.`
      : `No comparison: ${missing.join(" and ")} ${missing.length === 1 ? "does" : "do"} not reach ${target} on the tested range.`;

  const budgetFills = METHOD_IDS.map((id) => methods[id].fill_for_budget?.fill_rate ?? null);
  const [lightgbmFill, ma28Fill] = budgetFills;
  const budgetFoot =
    lightgbmFill != null && ma28Fill != null
      ? `MA28 serves ${Math.abs((ma28Fill - lightgbmFill) * 100).toFixed(1)} percentage points ${
          ma28Fill >= lightgbmFill ? "more" : "less"
        } demand with the same stock.`
      : "Outside the simulated range for at least one method.";

  const costDifference = comparison.cheapest_cost_difference_lightgbm_vs_ma28;
  const atEdge = METHOD_IDS.filter((id) => methods[id].cheapest.grid_edge !== null).map(
    (id) => methods[id].label,
  );
  const costFoot = [
    costDifference === null ? "" : `LightGBM's cheapest policy costs ${moreOrLess(costDifference)} than MA28's.`,
    atEdge.length === 0
      ? ""
      : `${atEdge.join(" and ")} ${atEdge.length === 1 ? "is" : "are"} cheapest at the edge of the tested range, so the true minimum may lie outside it.`,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <section className="stats" aria-label="Answers">
      <article className="panel stat">
        <h2>Stock for {target} fill</h2>
        <MethodRows rows={rows((id) => stockRow(methods[id].stock_for_target, highestZ))} />
        <p className="stat-foot">{stockFoot}</p>
      </article>
      <article className="panel stat">
        <h2>
          Fill rate with {scenario.stock_budget === null ? "a budget" : `${formatUnits(scenario.stock_budget)} units`}
        </h2>
        <MethodRows rows={rows((id) => budgetRow(methods[id].fill_for_budget))} />
        <p className="stat-foot">{budgetFoot}</p>
      </article>
      <article className="panel stat">
        <h2>Cheapest policy at these costs</h2>
        <MethodRows rows={rows((id) => cheapestRow(methods[id].cheapest))} />
        <p className="stat-foot">{costFoot}</p>
      </article>
      <article className="panel stat stat-reference">
        <h2>Perfect foresight</h2>
        <p className="stat-value">{formatUnits(reference.average_on_hand_units)} units</p>
        <p className="stat-detail">serves {formatPercent(reference.fill_rate, 0)} of demand</p>
        <p className="stat-foot">
          A reference that knows future demand: the least stock that loses no sales on this delivery
          schedule. Not a forecasting method.
        </p>
      </article>
    </section>
  );
}

export function SettingsNote({ settings }: { settings: Answer["settings"] }) {
  const exploratory = !settings.costs_preregistered || !settings.target_preregistered;
  const label = exploratory
    ? "Exploratory setting"
    : settings.primary_scenario
      ? "Preregistered primary scenario"
      : "Preregistered secondary analysis";

  return (
    <div className={`settings-note ${exploratory ? "is-exploratory" : "is-preregistered"}`}>
      <span className="badge">{label}</span>
      <ul>
        {settings.notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
    </div>
  );
}
