import assert from "node:assert/strict";
import { test } from "node:test";

import { parseBlocks, parseInline, splitRow } from "./markdown.ts";

test("plain text is one paragraph", () => {
  assert.deepEqual(parseBlocks("MA28 needs less stock."), [
    { kind: "paragraph", lines: [[{ kind: "text", text: "MA28 needs less stock." }]] },
  ]);
});

test("blank lines split paragraphs and single line breaks are kept", () => {
  const blocks = parseBlocks("First line\nsecond line\n\n\nNext paragraph");
  assert.equal(blocks.length, 2);
  assert.deepEqual(blocks[0], {
    kind: "paragraph",
    lines: [[{ kind: "text", text: "First line" }], [{ kind: "text", text: "second line" }]],
  });
});

test("bold and code runs", () => {
  assert.deepEqual(parseInline("**Difference:** 18.5% more with `stock_for_fill_rate`."), [
    { kind: "strong", text: "Difference:" },
    { kind: "text", text: " 18.5% more with " },
    { kind: "code", text: "stock_for_fill_rate" },
    { kind: "text", text: "." },
  ]);
});

test("unclosed markers stay literal", () => {
  assert.deepEqual(parseInline("2 ** 3 and a `tick"), [{ kind: "text", text: "2 ** 3 and a `tick" }]);
  assert.deepEqual(parseInline(""), []);
});

test("bullet and numbered lists", () => {
  const blocks = parseBlocks("Options:\n- **Stock:** units needed\n* Costs\n  across lead times\n1. First\n2) Second");
  assert.deepEqual(
    blocks.map((block) => block.kind),
    ["paragraph", "list", "list"],
  );
  const bullets = blocks[1];
  assert.ok(bullets?.kind === "list");
  assert.equal(bullets.ordered, false);
  assert.deepEqual(bullets.items, [
    [
      { kind: "strong", text: "Stock:" },
      { kind: "text", text: " units needed" },
    ],
    [{ kind: "text", text: "Costs across lead times" }],
  ]);
  const numbered = blocks[2];
  assert.ok(numbered?.kind === "list");
  assert.equal(numbered.ordered, true);
  assert.equal(numbered.items.length, 2);
});

test("a line after a list starts a paragraph", () => {
  const blocks = parseBlocks("- one\n- two\nAfter the list.");
  assert.deepEqual(
    blocks.map((block) => block.kind),
    ["list", "paragraph"],
  );
});

test("headings", () => {
  assert.deepEqual(parseBlocks("### Lead times ###"), [
    { kind: "heading", content: [{ kind: "text", text: "Lead times" }] },
  ]);
});

test("table rows, padding and numeric columns", () => {
  const blocks = parseBlocks(
    [
      "| Lead time | LightGBM | Gap |",
      "|---|:---:|---:|",
      "| 1 day | 25,283 | **7.1%** |",
      "| 7 days (primary) | 39,157 |",
      "Below the table.",
    ].join("\n"),
  );
  assert.deepEqual(
    blocks.map((block) => block.kind),
    ["table", "paragraph"],
  );
  const table = blocks[0];
  assert.ok(table?.kind === "table");
  assert.deepEqual(table.numeric, [false, true, true]);
  assert.equal(table.rows.length, 2);
  assert.deepEqual(table.rows[1], [[{ kind: "text", text: "7 days (primary)" }], [{ kind: "text", text: "39,157" }], []]);
  assert.deepEqual(table.rows[0]?.[2], [{ kind: "strong", text: "7.1%" }]);
});

test("cells keep escaped pipes", () => {
  assert.deepEqual(splitRow("| a \\| b | c |"), ["a | b", "c"]);
  assert.deepEqual(splitRow("x | y"), ["x", "y"]);
});

test("a pipe without a delimiter row is just text", () => {
  assert.deepEqual(
    parseBlocks("LightGBM | MA28\nnot a table").map((block) => block.kind),
    ["paragraph"],
  );
  assert.deepEqual(
    parseBlocks("one | two\n---").map((block) => block.kind),
    ["paragraph"],
  );
});

test("an answer from the live evaluation", () => {
  const answer = [
    "I can't confirm that. The simulation shows the opposite.",
    "",
    "This uses the 7-day lead time and the primary starting condition.",
    "",
    "| Forecaster | Average units on hand for 90% fill |",
    "|---|---|",
    "| LightGBM | 39,157 |",
    "| MA28 | 33,044 |",
    "",
    "LightGBM needs 18.5% more stock than MA28 for 90% fill, not 30% less.",
  ].join("\n");
  const blocks = parseBlocks(answer);
  assert.deepEqual(
    blocks.map((block) => block.kind),
    ["paragraph", "paragraph", "table", "paragraph"],
  );
  const table = blocks[2];
  assert.ok(table?.kind === "table");
  assert.deepEqual(table.numeric, [false, true]);
  assert.deepEqual(table.header[1], [{ kind: "text", text: "Average units on hand for 90% fill" }]);
});

test("windows line endings", () => {
  assert.equal(parseBlocks("a\r\n\r\nb").length, 2);
});
