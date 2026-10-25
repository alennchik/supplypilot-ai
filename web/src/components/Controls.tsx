import type { ChangeEvent } from "react";

import type { Question, Scenarios } from "../api.ts";
import { formatDays, formatPercent, formatUnits } from "../format.ts";

/** Range inputs report decimals like 0.9450000001; keep six places. */
function snap(value: number): number {
  return Number(value.toFixed(6));
}

interface RangeControlProps {
  id: string;
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
  onChange: (value: number) => void;
  presets?: number[];
  presetLabel?: string;
  formatPreset?: (value: number) => string;
}

function RangeControl(props: RangeControlProps) {
  const hintId = props.hint ? `${props.id}-hint` : undefined;
  const formatPreset = props.formatPreset ?? props.format;

  return (
    <div className="control">
      <div className="control-head">
        <label htmlFor={props.id}>{props.label}</label>
        <output htmlFor={props.id} className="control-value">
          {props.format(props.value)}
        </output>
      </div>
      <input
        id={props.id}
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        aria-describedby={hintId}
        aria-valuetext={props.format(props.value)}
        onChange={(event: ChangeEvent<HTMLInputElement>) =>
          props.onChange(snap(Number(event.currentTarget.value)))
        }
      />
      {props.hint ? (
        <p id={hintId} className="control-hint">
          {props.hint}
        </p>
      ) : null}
      {props.presets && props.presets.length > 0 ? (
        <div className="presets" role="group" aria-label={props.presetLabel}>
          <span className="presets-label">{props.presetLabel}</span>
          {props.presets.map((preset) => (
            <button
              key={preset}
              type="button"
              className="chip"
              aria-pressed={Math.abs(preset - props.value) < 1e-9}
              onClick={() => props.onChange(preset)}
            >
              {formatPreset(preset)}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

interface ControlsProps {
  scenarios: Scenarios;
  question: Question;
  onChange: (question: Question) => void;
  onReset: () => void;
}

export function Controls({ scenarios, question, onChange, onReset }: ControlsProps) {
  const update = (patch: Partial<Question>) => onChange({ ...question, ...patch });
  const leadTimes = [...scenarios.lead_times].sort((a, b) => a - b);
  const { preregistered } = scenarios;

  return (
    <section className="panel controls" aria-label="Your question">
      <fieldset className="control">
        <legend>Supplier lead time</legend>
        <div className="segmented">
          {leadTimes.map((days) => (
            <button
              key={days}
              type="button"
              aria-pressed={question.lead_time === days}
              onClick={() => update({ lead_time: days })}
            >
              {formatDays(days)}
            </button>
          ))}
        </div>
        <p className="control-hint">Orders are placed every Sunday.</p>
      </fieldset>

      <RangeControl
        id="target"
        label="Target fill rate"
        hint="Share of demanded units served from stock."
        value={question.target_fill_rate}
        min={0.8}
        max={0.98}
        step={0.005}
        format={(value) => formatPercent(value, 1)}
        onChange={(value) => update({ target_fill_rate: value })}
        presets={preregistered.targets}
        presetLabel="Protocol"
        formatPreset={(value) => formatPercent(value, 0)}
      />

      <RangeControl
        id="holding"
        label="Holding cost"
        hint="Per year, as a share of the sell price."
        value={question.holding_rate}
        min={0}
        max={0.6}
        step={0.01}
        format={(value) => `${formatPercent(value, 0)} a year`}
        onChange={(value) => update({ holding_rate: value })}
        presets={[preregistered.holding_rate]}
        presetLabel="Protocol"
        formatPreset={(value) => formatPercent(value, 0)}
      />

      <RangeControl
        id="penalty"
        label="Lost-sale penalty"
        hint="Cost of each unit not sold, times its sell price."
        value={question.penalty_fraction}
        min={0}
        max={2}
        step={0.05}
        format={(value) => `${value.toFixed(2)} × price`}
        onChange={(value) => update({ penalty_fraction: value })}
        presets={preregistered.penalty_fractions}
        presetLabel="Protocol"
        formatPreset={(value) => `${value}×`}
      />

      <RangeControl
        id="budget"
        label="Stock budget"
        hint="Average units on hand across all series."
        value={question.stock_budget ?? 35_000}
        min={10_000}
        max={60_000}
        step={500}
        format={(value) => `${formatUnits(value)} units`}
        onChange={(value) => update({ stock_budget: value })}
      />

      <div className="control">
        <label htmlFor="variant">Starting condition</label>
        <select
          id="variant"
          value={question.variant}
          onChange={(event: ChangeEvent<HTMLSelectElement>) =>
            update({ variant: event.currentTarget.value })
          }
        >
          {scenarios.variants.map((variant) => (
            <option key={variant.id} value={variant.id}>
              {variant.label}
            </option>
          ))}
        </select>
        <p className="control-hint">The two alternatives are the protocol&apos;s robustness checks.</p>
      </div>

      <div className="control control-actions">
        <button type="button" className="button-secondary" onClick={onReset}>
          Reset to protocol defaults
        </button>
      </div>
    </section>
  );
}
