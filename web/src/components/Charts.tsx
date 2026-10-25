import { METHOD_IDS } from "../api.ts";
import type { Answer } from "../api.ts";
import {
  formatCompactUnits,
  formatMoney,
  formatPercent,
  formatPercentAuto,
  formatUnits,
  formatZ,
} from "../format.ts";
import { interpolateAt, nearestIndex, niceTicks, roundOut } from "../scale.ts";
import { DataTable } from "./DataTable.tsx";
import { LineChart } from "./LineChart.tsx";
import type { ChartMarker, ChartRule, ChartSeries, HoverDetail } from "./LineChart.tsx";

interface LegendItem {
  id: string;
  tone: string;
  label: string;
  shape: "line" | "dot" | "ring";
}

function Legend({ items }: { items: LegendItem[] }) {
  return (
    <ul className="legend" aria-label="Legend">
      {items.map((item) => (
        <li key={item.id}>
          <span className={`legend-swatch legend-${item.shape} tone-${item.tone}`} aria-hidden="true" />
          {item.label}
        </li>
      ))}
    </ul>
  );
}

function methodLegend(answer: Answer): LegendItem[] {
  return METHOD_IDS.map((id) => ({
    id,
    tone: id,
    label: answer.methods[id].label,
    shape: "line" as const,
  }));
}

function evenStops(low: number, high: number, count: number): number[] {
  return Array.from({ length: count + 1 }, (_, index) => low + ((high - low) * index) / count);
}

export function TradeoffChart({ answer }: { answer: Answer }) {
  const { scenario, methods, reference } = answer;
  const target = formatPercentAuto(scenario.target_fill_rate);
  const series: ChartSeries[] = METHOD_IDS.map((id) => ({
    id,
    label: methods[id].label,
    tone: id,
    points: methods[id].policies.map((policy) => ({
      x: policy.average_on_hand_units,
      y: policy.fill_rate,
    })),
  }));

  const stocks = series.flatMap((line) => line.points.map((point) => point.x));
  const fills = series.flatMap((line) => line.points.map((point) => point.y));
  const xDomain = roundOut(
    Math.min(...stocks, reference.average_on_hand_units) * 0.92,
    Math.max(...stocks) * 1.03,
    5000,
  );
  const yDomain: [number, number] = [roundOut(Math.min(...fills), 1, 0.05)[0], 1.0];

  const markers: ChartMarker[] = [
    {
      id: "reference",
      x: reference.average_on_hand_units,
      y: reference.fill_rate,
      tone: "reference",
      shape: "dot",
      label: "Perfect foresight",
    },
  ];
  for (const id of METHOD_IDS) {
    const units = methods[id].stock_for_target.units;
    if (units !== null) {
      markers.push({
        id: `target-${id}`,
        x: units,
        y: scenario.target_fill_rate,
        tone: id,
        shape: "ring",
      });
    }
  }

  const rules: ChartRule[] = [
    { id: "target", axis: "y", value: scenario.target_fill_rate, label: `${target} target` },
  ];
  if (scenario.stock_budget !== null) {
    rules.push({
      id: "budget",
      axis: "x",
      value: scenario.stock_budget,
      label: `${formatCompactUnits(scenario.stock_budget)} budget`,
    });
  }

  function describe(stock: number): HoverDetail {
    return {
      x: stock,
      title: `${formatUnits(stock)} units on hand`,
      rows: series.map((line) => {
        const fill = interpolateAt(line.points, stock);
        return {
          id: line.id,
          tone: line.tone,
          label: line.label,
          value: fill === null ? "not simulated" : formatPercent(fill),
          y: fill,
        };
      }),
    };
  }

  return (
    <section className="panel chart-card" aria-labelledby="tradeoff-title">
      <div className="chart-head">
        <h2 id="tradeoff-title">Fill rate against average stock</h2>
        <Legend
          items={[
            ...methodLegend(answer),
            { id: "reference", tone: "reference", label: "Perfect foresight", shape: "dot" },
          ]}
        />
      </div>
      <p className="chart-note">
        Each line joins 31 simulated policies, from z = 0 on the left to z = 3.0 on the right; higher and
        further left is better.
        {markers.length > 1 ? ` Rings mark where a line meets the ${target} target.` : ""}
      </p>
      <LineChart
        label={`Fill rate against average units on hand for LightGBM and MA28 at a ${scenario.lead_time}-day lead time`}
        series={series}
        markers={markers}
        rules={rules}
        xDomain={xDomain}
        yDomain={yDomain}
        formatX={formatCompactUnits}
        formatY={(value) => formatPercent(value, 0)}
        xLabel="Average units on hand, all series combined"
        yLabel="Fill rate"
        stops={evenStops(xDomain[0], xDomain[1], 40)}
        describe={describe}
      />
      <DataTable
        caption="Simulated policies behind the chart"
        columns={[
          { key: "z", label: "z" },
          { key: "lightgbmFill", label: "LightGBM fill", numeric: true },
          { key: "lightgbmStock", label: "LightGBM stock", numeric: true },
          { key: "ma28Fill", label: "MA28 fill", numeric: true },
          { key: "ma28Stock", label: "MA28 stock", numeric: true },
        ]}
        rows={methods.lightgbm.policies.map((policy, index) => {
          const other = methods.ma28.policies[index];
          return {
            z: formatZ(policy.z),
            lightgbmFill: formatPercent(policy.fill_rate),
            lightgbmStock: formatUnits(policy.average_on_hand_units),
            ma28Fill: other ? formatPercent(other.fill_rate) : "",
            ma28Stock: other ? formatUnits(other.average_on_hand_units) : "",
          };
        })}
      />
    </section>
  );
}

export function CostChart({ answer }: { answer: Answer }) {
  const { methods } = answer;
  const series: ChartSeries[] = METHOD_IDS.map((id) => ({
    id,
    label: methods[id].label,
    tone: id,
    points: methods[id].policies.map((policy) => ({ x: policy.z, y: policy.total_cost })),
  }));
  const zValues = methods.lightgbm.policies.map((policy) => policy.z);
  const xDomain: [number, number] = [Math.min(...zValues), Math.max(...zValues)];
  const highest = Math.max(...series.flatMap((line) => line.points.map((point) => point.y)));
  const yTicks = niceTicks(0, highest, 5);
  const yDomain: [number, number] = [0, Math.max(highest, yTicks[yTicks.length - 1] ?? highest)];

  const markers: ChartMarker[] = METHOD_IDS.map((id) => ({
    id: `cheapest-${id}`,
    x: methods[id].cheapest.z,
    y: methods[id].cheapest.total_cost,
    tone: id,
    shape: "ring" as const,
  }));

  function describe(z: number): HoverDetail | null {
    const index = nearestIndex(zValues, z);
    const step = zValues[index];
    if (step === undefined) {
      return null;
    }
    return {
      x: step,
      title: `z = ${formatZ(step)}`,
      rows: METHOD_IDS.map((id) => {
        const policy = methods[id].policies[index];
        return {
          id,
          tone: id,
          label: methods[id].label,
          value: policy ? formatMoney(policy.total_cost) : "",
          y: policy ? policy.total_cost : null,
        };
      }),
    };
  }

  return (
    <section className="panel chart-card" aria-labelledby="cost-title">
      <div className="chart-head">
        <h2 id="cost-title">Total cost by safety factor</h2>
        <Legend
          items={[
            ...methodLegend(answer),
            { id: "cheapest", tone: "neutral", label: "Cheapest policy", shape: "ring" },
          ]}
        />
      </div>
      <p className="chart-note">
        Holding plus lost-sale cost over the 84 evaluation days, in dollars at sell price. These are
        illustrative costs, not Walmart&apos;s.
      </p>
      <LineChart
        label="Total cost against the safety factor z for LightGBM and MA28"
        series={series}
        markers={markers}
        xDomain={xDomain}
        yDomain={yDomain}
        formatX={formatZ}
        formatY={(value) => (value === 0 ? "$0" : `$${formatCompactUnits(value)}`)}
        xLabel="Safety factor z"
        yLabel="Total cost"
        stops={zValues}
        describe={describe}
      />
      <DataTable
        caption="Cost of every simulated policy"
        columns={[
          { key: "z", label: "z" },
          { key: "lightgbmHolding", label: "LightGBM holding", numeric: true },
          { key: "lightgbmPenalty", label: "LightGBM lost sales", numeric: true },
          { key: "lightgbmTotal", label: "LightGBM total", numeric: true },
          { key: "ma28Holding", label: "MA28 holding", numeric: true },
          { key: "ma28Penalty", label: "MA28 lost sales", numeric: true },
          { key: "ma28Total", label: "MA28 total", numeric: true },
        ]}
        rows={methods.lightgbm.policies.map((policy, index) => {
          const other = methods.ma28.policies[index];
          return {
            z: formatZ(policy.z),
            lightgbmHolding: formatMoney(policy.holding_cost),
            lightgbmPenalty: formatMoney(policy.penalty_cost),
            lightgbmTotal: formatMoney(policy.total_cost),
            ma28Holding: other ? formatMoney(other.holding_cost) : "",
            ma28Penalty: other ? formatMoney(other.penalty_cost) : "",
            ma28Total: other ? formatMoney(other.total_cost) : "",
          };
        })}
      />
    </section>
  );
}
