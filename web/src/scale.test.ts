import assert from "node:assert/strict";
import { test } from "node:test";

import {
  clamp,
  interpolateAt,
  linePath,
  linearScale,
  nearestIndex,
  niceTicks,
  roundOut,
} from "./scale.ts";

test("a linear scale maps and inverts", () => {
  const scale = linearScale([0, 100], [50, 250]);
  assert.equal(scale(0), 50);
  assert.equal(scale(50), 150);
  assert.equal(scale.invert(250), 100);

  const flipped = linearScale([0.75, 1], [300, 20]);
  assert.equal(flipped(1), 20);
  assert.ok(Math.abs(flipped.invert(160) - 0.875) < 1e-12);
});

test("nice ticks use round steps", () => {
  assert.deepEqual(niceTicks(0, 52000, 5), [0, 10000, 20000, 30000, 40000, 50000]);
  assert.deepEqual(niceTicks(0.75, 1, 5), [0.75, 0.8, 0.85, 0.9, 0.95, 1]);
  assert.deepEqual(niceTicks(0, 3, 6), [0, 0.5, 1, 1.5, 2, 2.5, 3]);
  assert.deepEqual(niceTicks(5, 5), [5]);
});

test("ranges round outwards to whole steps", () => {
  assert.deepEqual(roundOut(0.7801, 1, 0.05), [0.75, 1]);
  assert.deepEqual(roundOut(12314, 51705, 5000), [10000, 55000]);
});

test("line paths", () => {
  assert.equal(
    linePath([
      [0, 1],
      [2.345, 3],
    ]),
    "M0.00 1.00 L2.35 3.00",
  );
  assert.equal(linePath([]), "");
});

test("interpolation stays inside the points", () => {
  const points = [
    { x: 10, y: 0.8 },
    { x: 20, y: 0.9 },
    { x: 40, y: 0.95 },
  ];
  assert.ok(Math.abs((interpolateAt(points, 15) ?? 0) - 0.85) < 1e-12);
  assert.equal(interpolateAt(points, 40), 0.95);
  assert.equal(interpolateAt(points, 5), null);
  assert.equal(interpolateAt(points, 41), null);
  assert.equal(interpolateAt([], 1), null);
});

test("nearest index and clamp", () => {
  assert.equal(nearestIndex([0, 0.1, 0.2, 0.3], 0.16), 2);
  assert.equal(nearestIndex([], 1), -1);
  assert.equal(clamp(5, 0, 3), 3);
  assert.equal(clamp(-1, 0, 3), 0);
});
