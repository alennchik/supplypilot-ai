// Small chart helpers: linear scales, tick values, SVG paths and interpolation.

export interface LinearScale {
  (value: number): number;
  invert(position: number): number;
}

export function linearScale(
  domain: readonly [number, number],
  range: readonly [number, number],
): LinearScale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const scale = ((value: number) => r0 + ((value - d0) / span) * (r1 - r0)) as LinearScale;
  scale.invert = (position: number) => d0 + ((position - r0) / (r1 - r0 || 1)) * span;
  return scale;
}

/** About `count` round tick values covering [min, max], in steps of 1, 2 or 5 x 10^k. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!(max > min) || count < 1) {
    return [min];
  }
  const raw = (max - min) / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const fraction = raw / power;
  // Same thresholds as d3: pick the round step closest to the raw step.
  const multiple =
    fraction >= Math.sqrt(50) ? 10 : fraction >= Math.sqrt(10) ? 5 : fraction >= Math.SQRT2 ? 2 : 1;
  const step = multiple * power;
  const first = Math.ceil(min / step - 1e-9) * step;
  const ticks: number[] = [];
  for (let value = first; value <= max + step * 1e-9; value += step) {
    ticks.push(Number(value.toPrecision(12)));
  }
  return ticks;
}

/** Rounds a range outwards to whole steps: (0.78, 0.95, 0.05) -> [0.75, 0.95]. */
export function roundOut(min: number, max: number, step: number): [number, number] {
  return [
    Number((Math.floor(min / step + 1e-9) * step).toPrecision(12)),
    Number((Math.ceil(max / step - 1e-9) * step).toPrecision(12)),
  ];
}

export function linePath(points: ReadonlyArray<readonly [number, number]>): string {
  return points
    .map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`)
    .join(" ");
}

/** Linear interpolation of y at x on points sorted by x; null outside their range. */
export function interpolateAt(
  points: ReadonlyArray<{ readonly x: number; readonly y: number }>,
  x: number,
): number | null {
  const first = points[0];
  const last = points[points.length - 1];
  if (!first || !last || x < first.x || x > last.x) {
    return null;
  }
  for (let index = 1; index < points.length; index += 1) {
    const left = points[index - 1];
    const right = points[index];
    if (left && right && x <= right.x) {
      const width = right.x - left.x;
      return width === 0 ? right.y : left.y + ((x - left.x) / width) * (right.y - left.y);
    }
  }
  return last.y;
}

/** Index of the value closest to target (values in any order); -1 when empty. */
export function nearestIndex(values: readonly number[], target: number): number {
  let best = -1;
  let distance = Number.POSITIVE_INFINITY;
  values.forEach((value, index) => {
    const gap = Math.abs(value - target);
    if (gap < distance) {
      best = index;
      distance = gap;
    }
  });
  return best;
}

export function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}
