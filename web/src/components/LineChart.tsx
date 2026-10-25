import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent } from "react";

import { clamp, linePath, linearScale, niceTicks } from "../scale.ts";

export interface ChartPoint {
  x: number;
  y: number;
}

/** tone selects the series color through a CSS class: tone-lightgbm, tone-ma28, ... */
export interface ChartSeries {
  id: string;
  label: string;
  tone: string;
  points: ChartPoint[];
}

/** A dot is a reference point; a ring marks a value read off a line for the question. */
export interface ChartMarker {
  id: string;
  x: number;
  y: number;
  tone: string;
  shape: "dot" | "ring";
  label?: string;
}

export interface ChartRule {
  id: string;
  axis: "x" | "y";
  value: number;
  label: string;
}

export interface HoverRow {
  id: string;
  tone: string;
  label: string;
  value: string;
  /** Where to mark the line at the hovered x; null when the line does not reach it. */
  y: number | null;
}

export interface HoverDetail {
  x: number;
  title: string;
  rows: HoverRow[];
}

interface LineChartProps {
  label: string;
  series: ChartSeries[];
  markers?: ChartMarker[];
  rules?: ChartRule[];
  xDomain: [number, number];
  yDomain: [number, number];
  formatX: (value: number) => string;
  formatY: (value: number) => string;
  xLabel: string;
  yLabel: string;
  /** x values the arrow keys step through when the chart has focus. */
  stops: number[];
  describe: (x: number) => HoverDetail | null;
}

const TOOLTIP_WIDTH = 188;
const ROW_HEIGHT = 20;

/** Width of an element in CSS pixels, kept current as the layout changes. */
function useWidth(fallback: number) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(fallback);

  useEffect(() => {
    const element = ref.current;
    if (!element || typeof ResizeObserver === "undefined") {
      return undefined;
    }
    const observer = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (box && box.width > 0) {
        setWidth(Math.round(box.width));
      }
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return { ref, width };
}

/**
 * A line chart drawn at the size it is shown, so text stays at its CSS size on
 * every screen. Hover, or focus and use the arrow keys, to read values.
 */
export function LineChart(props: LineChartProps) {
  const { ref, width: measured } = useWidth(640);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hover, setHover] = useState<HoverDetail | null>(null);
  const [stopIndex, setStopIndex] = useState(-1);

  const width = Math.max(300, measured);
  const narrow = width < 520;
  const height = narrow ? 300 : 340;
  const xTicks = niceTicks(props.xDomain[0], props.xDomain[1], narrow ? 4 : 6);
  const yTicks = niceTicks(props.yDomain[0], props.yDomain[1], 5);
  // Room for the rotated axis title plus the widest tick label (12px monospace).
  const widestTick = Math.max(...yTicks.map((tick) => props.formatY(tick).length));
  const margin = {
    top: 22,
    right: narrow ? 14 : 24,
    bottom: 52,
    left: Math.round(36 + widestTick * 7.4),
  };
  const plotRight = width - margin.right;
  const plotBottom = height - margin.bottom;
  const x = linearScale(props.xDomain, [margin.left, plotRight]);
  const y = linearScale(props.yDomain, [plotBottom, margin.top]);
  const inX = (value: number) => value >= props.xDomain[0] && value <= props.xDomain[1];
  const inY = (value: number) => value >= props.yDomain[0] && value <= props.yDomain[1];

  function showAt(value: number) {
    setHover(props.describe(clamp(value, props.xDomain[0], props.xDomain[1])));
  }

  function showStop(index: number) {
    const stop = props.stops[index];
    if (stop !== undefined) {
      setStopIndex(index);
      showAt(stop);
    }
  }

  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    const box = svgRef.current?.getBoundingClientRect();
    if (!box || box.width === 0) {
      return;
    }
    const position = ((event.clientX - box.left) / box.width) * width;
    if (position < margin.left || position > plotRight) {
      setHover(null);
      return;
    }
    showAt(x.invert(position));
  }

  function onKeyDown(event: KeyboardEvent<SVGSVGElement>) {
    const last = props.stops.length - 1;
    const steps: Record<string, number> = {
      ArrowRight: Math.min(last, stopIndex + 1),
      ArrowLeft: Math.max(0, stopIndex - 1),
      Home: 0,
      End: last,
    };
    if (event.key === "Escape") {
      setHover(null);
      return;
    }
    const next = steps[event.key];
    if (next === undefined || last < 0) {
      return;
    }
    event.preventDefault();
    showStop(next);
  }

  function onFocus() {
    if (!hover && props.stops.length > 0) {
      showStop(stopIndex >= 0 ? stopIndex : Math.floor(props.stops.length / 2));
    }
  }

  const hoverX = hover ? x(hover.x) : 0;
  const tooltipHeight = hover ? 34 + hover.rows.length * ROW_HEIGHT : 0;
  const tooltipLeft =
    hoverX + 14 + TOOLTIP_WIDTH > plotRight ? Math.max(0, hoverX - 14 - TOOLTIP_WIDTH) : hoverX + 14;

  return (
    <div ref={ref} className="chart-frame">
      <svg
        ref={svgRef}
        className="chart"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={props.label}
        tabIndex={0}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setHover(null)}
        onFocus={onFocus}
        onBlur={() => setHover(null)}
        onKeyDown={onKeyDown}
      >
        <g className="chart-grid" aria-hidden="true">
          {yTicks.filter(inY).map((tick) => (
            <line key={tick} x1={margin.left} x2={plotRight} y1={y(tick)} y2={y(tick)} />
          ))}
        </g>
        <line className="chart-axis" x1={margin.left} x2={plotRight} y1={plotBottom} y2={plotBottom} />

        <g className="chart-ticks" aria-hidden="true">
          {yTicks.filter(inY).map((tick) => (
            <text key={tick} x={margin.left - 10} y={y(tick)} textAnchor="end" dominantBaseline="middle">
              {props.formatY(tick)}
            </text>
          ))}
          {xTicks.filter(inX).map((tick) => (
            <text key={tick} x={x(tick)} y={plotBottom + 20} textAnchor="middle">
              {props.formatX(tick)}
            </text>
          ))}
        </g>
        <text
          className="chart-axis-label"
          x={(margin.left + plotRight) / 2}
          y={height - 10}
          textAnchor="middle"
        >
          {props.xLabel}
        </text>
        <text
          className="chart-axis-label"
          transform={`translate(14 ${(margin.top + plotBottom) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {props.yLabel}
        </text>

        {(props.rules ?? []).map((rule) => {
          if (!(rule.axis === "y" ? inY(rule.value) : inX(rule.value))) {
            return null;
          }
          return rule.axis === "y" ? (
            <g key={rule.id} className="chart-rule">
              <line x1={margin.left} x2={plotRight} y1={y(rule.value)} y2={y(rule.value)} />
              <text x={margin.left + 8} y={y(rule.value) - 7}>
                {rule.label}
              </text>
            </g>
          ) : (
            <g key={rule.id} className="chart-rule">
              <line x1={x(rule.value)} x2={x(rule.value)} y1={margin.top} y2={plotBottom} />
              <text x={x(rule.value) + 7} y={plotBottom - 8}>
                {rule.label}
              </text>
            </g>
          );
        })}

        {props.series.map((line) => (
          <g key={line.id} className={`chart-series tone-${line.tone}`}>
            <path className="chart-line" d={linePath(line.points.map((p) => [x(p.x), y(p.y)]))} />
            {[line.points[0], line.points[line.points.length - 1]].map((point, index) =>
              point ? (
                <circle key={index} className="chart-dot" cx={x(point.x)} cy={y(point.y)} r={4} />
              ) : null,
            )}
          </g>
        ))}

        {(props.markers ?? []).map((marker) => (
          <g key={marker.id} className={`chart-marker tone-${marker.tone}`}>
            <circle
              className={marker.shape === "ring" ? "chart-ring" : "chart-dot"}
              cx={x(marker.x)}
              cy={y(marker.y)}
              r={marker.shape === "ring" ? 7 : 5}
            />
            {marker.label ? (
              <text x={x(marker.x) + 11} y={y(marker.y)} dominantBaseline="middle">
                {marker.label}
              </text>
            ) : null}
          </g>
        ))}

        {hover ? (
          <g className="chart-hover" aria-hidden="true">
            <line className="chart-crosshair" x1={hoverX} x2={hoverX} y1={margin.top} y2={plotBottom} />
            {hover.rows.map((row) =>
              row.y !== null && inY(row.y) ? (
                <circle
                  key={row.id}
                  className={`chart-dot tone-${row.tone}`}
                  cx={hoverX}
                  cy={y(row.y)}
                  r={4.5}
                />
              ) : null,
            )}
            <g transform={`translate(${tooltipLeft} ${margin.top + 4})`}>
              <rect className="chart-tooltip" width={TOOLTIP_WIDTH} height={tooltipHeight} rx={8} />
              <text className="chart-tooltip-title" x={12} y={21}>
                {hover.title}
              </text>
              {hover.rows.map((row, index) => (
                <g key={row.id} transform={`translate(12 ${42 + index * ROW_HEIGHT})`}>
                  <circle className={`chart-key tone-${row.tone}`} cx={4} cy={-4} r={4} />
                  <text className="chart-tooltip-label" x={14} y={0}>
                    {row.label}
                  </text>
                  <text className="chart-tooltip-value" x={TOOLTIP_WIDTH - 24} y={0} textAnchor="end">
                    {row.value}
                  </text>
                </g>
              ))}
            </g>
          </g>
        ) : null}
      </svg>
    </div>
  );
}
