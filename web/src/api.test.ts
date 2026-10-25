import assert from "node:assert/strict";
import { test } from "node:test";

import { errorMessage } from "./api.ts";

test("a plain error detail is shown as is", () => {
  assert.equal(
    errorMessage(422, { detail: "Lead time must be one of [7, 3, 1] days" }),
    "Lead time must be one of [7, 3, 1] days",
  );
});

test("validation issues name their field", () => {
  const body = {
    detail: [
      { loc: ["body", "target_fill_rate"], msg: "Input should be less than or equal to 1" },
      { loc: ["body"], msg: "Extra inputs are not permitted" },
    ],
  };
  assert.equal(
    errorMessage(422, body),
    "target_fill_rate: Input should be less than or equal to 1; Extra inputs are not permitted",
  );
});

test("an unreadable body falls back to the status", () => {
  assert.equal(errorMessage(502, null), "The server answered with status 502.");
  assert.equal(errorMessage(500, { detail: [] }), "The server answered with status 500.");
});
