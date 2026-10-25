// Number formatting for the page. Kept free of React so Node can test it directly.

const units = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const MINUS = "−";

export function formatUnits(value: number): string {
  return units.format(Math.round(value));
}

/** 39157 -> "39.2K", 40000 -> "40K", 950 -> "950". */
export function formatCompactUnits(value: number): string {
  if (Math.abs(value) < 1000) {
    return formatUnits(value);
  }
  const thousands = value / 1000;
  const rounded = Math.round(thousands * 10) / 10;
  return `${Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1)}K`;
}

/** 0.92243 -> "92.2%". */
export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** Whole percentages without decimals: 0.95 -> "95%", 0.945 -> "94.5%". */
export function formatPercentAuto(value: number): string {
  const percent = value * 100;
  return formatPercent(value, Math.abs(percent - Math.round(percent)) < 1e-9 ? 0 : 1);
}

/** 0.185 -> "+18.5%", -0.05 -> "−5.0%" (a true minus sign). */
export function formatSignedPercent(value: number, digits = 1): string {
  const text = (Math.abs(value) * 100).toFixed(digits);
  if (Number(text) === 0) {
    return `${text}%`;
  }
  return `${value > 0 ? "+" : MINUS}${text}%`;
}

export function formatMoney(value: number): string {
  return money.format(Math.round(value)).replace("-", MINUS);
}

/** Safety factors always show one decimal: 3 -> "3.0". */
export function formatZ(value: number): string {
  return value.toFixed(1);
}

/** 1 -> "1 day", 7 -> "7 days". */
export function formatDays(value: number): string {
  return `${value} ${value === 1 ? "day" : "days"}`;
}

const shortDate = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});

/** ("2016-02-01", "2016-04-24") -> "Feb 1 – Apr 24, 2016". */
export function formatDateRange(start: string, end: string): string {
  const first = new Date(`${start}T00:00:00Z`);
  const last = new Date(`${end}T00:00:00Z`);
  const firstYear = first.getUTCFullYear();
  const lastYear = last.getUTCFullYear();
  const left = shortDate.format(first) + (firstYear === lastYear ? "" : `, ${firstYear}`);
  return `${left} – ${shortDate.format(last)}, ${lastYear}`;
}
