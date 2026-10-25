import assert from "node:assert/strict";
import { test } from "node:test";

import {
  formatCompactUnits,
  formatDateRange,
  formatDays,
  formatMoney,
  formatPercent,
  formatPercentAuto,
  formatSignedPercent,
  formatUnits,
  formatZ,
} from "./format.ts";

test("units use thousands separators", () => {
  assert.equal(formatUnits(39156.74), "39,157");
  assert.equal(formatUnits(0), "0");
});

test("compact units drop a trailing .0", () => {
  assert.equal(formatCompactUnits(39157), "39.2K");
  assert.equal(formatCompactUnits(40000), "40K");
  assert.equal(formatCompactUnits(950), "950");
});

test("percentages", () => {
  assert.equal(formatPercent(0.922427), "92.2%");
  assert.equal(formatPercent(0.95, 0), "95%");
  assert.equal(formatPercentAuto(0.95), "95%");
  assert.equal(formatPercentAuto(0.945), "94.5%");
  assert.equal(formatSignedPercent(0.18500263), "+18.5%");
  assert.equal(formatSignedPercent(-0.05), "−5.0%");
  assert.equal(formatSignedPercent(0.00001), "0.0%");
});

test("money, safety factors and days", () => {
  assert.equal(formatMoney(27467.4), "$27,467");
  assert.equal(formatMoney(-1200), "−$1,200");
  assert.equal(formatZ(3), "3.0");
  assert.equal(formatZ(2.9), "2.9");
  assert.equal(formatDays(1), "1 day");
  assert.equal(formatDays(7), "7 days");
});

test("date ranges", () => {
  assert.equal(formatDateRange("2016-02-01", "2016-04-24"), "Feb 1 – Apr 24, 2016");
  assert.equal(formatDateRange("2015-12-28", "2016-01-03"), "Dec 28, 2015 – Jan 3, 2016");
});
