// Reads the small part of Markdown the planning assistant writes: paragraphs, bullet and
// numbered lists, pipe tables, headings, bold text and inline code. Anything else stays plain
// text. The result is data for React to render, never HTML, so an answer cannot inject markup.

export type Inline =
  | { kind: "text"; text: string }
  | { kind: "strong"; text: string }
  | { kind: "code"; text: string };

export type Block =
  | { kind: "paragraph"; lines: Inline[][] }
  | { kind: "heading"; content: Inline[] }
  | { kind: "list"; ordered: boolean; items: Inline[][] }
  | { kind: "table"; header: Inline[][]; numeric: boolean[]; rows: Inline[][][] };

interface ListDraft {
  ordered: boolean;
  items: string[];
}

const INLINE = /\*\*(.+?)\*\*|`([^`]+)`/g;
const BULLET = /^\s*[-*+]\s+(.*)$/;
const NUMBERED = /^\s*\d{1,3}[.)]\s+(.*)$/;
const HEADING = /^\s*#{1,6}\s+(.*?)\s*#*\s*$/;
const DELIMITER = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;
const NUMBER = /^[−+-]?\$?\d[\d,]*(\.\d+)?\s?%?$/;
const CONTINUATION = /^\s{2,}\S/;

/** Splits text into plain, bold (**…**) and code (`…`) runs. Unclosed markers stay literal. */
export function parseInline(text: string): Inline[] {
  const parts: Inline[] = [];
  let last = 0;
  for (const match of text.matchAll(INLINE)) {
    const start = match.index;
    if (start > last) {
      parts.push({ kind: "text", text: text.slice(last, start) });
    }
    if (match[1] !== undefined) {
      parts.push({ kind: "strong", text: match[1] });
    } else {
      parts.push({ kind: "code", text: match[2] ?? "" });
    }
    last = start + match[0].length;
  }
  if (last < text.length) {
    parts.push({ kind: "text", text: text.slice(last) });
  }
  return parts;
}

/** The cells of one table row; a backslash keeps a pipe inside a cell. */
export function splitRow(line: string): string[] {
  let body = line.trim();
  if (body.startsWith("|")) {
    body = body.slice(1);
  }
  if (body.endsWith("|") && !body.endsWith("\\|")) {
    body = body.slice(0, -1);
  }
  const cells: string[] = [];
  let current = "";
  for (let index = 0; index < body.length; index += 1) {
    const char = body.charAt(index);
    if (char === "\\" && body.charAt(index + 1) === "|") {
      current += "|";
      index += 1;
    } else if (char === "|") {
      cells.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  cells.push(current.trim());
  return cells;
}

function startsTable(line: string, next: string): boolean {
  return line.includes("|") && DELIMITER.test(next) && splitRow(next).length === splitRow(line).length;
}

function isNumber(cell: string): boolean {
  return NUMBER.test(cell.replaceAll("**", "").trim());
}

function table(header: string[], rows: string[][]): Block {
  const numeric = header.map((_, column) => {
    const values = rows.map((row) => row[column] ?? "").filter((value) => value.trim() !== "");
    return values.length > 0 && values.every(isNumber);
  });
  return {
    kind: "table",
    header: header.map((cell) => parseInline(cell)),
    numeric,
    rows: rows.map((row) => row.map((cell) => parseInline(cell))),
  };
}

/** Turns an answer into blocks. Single line breaks inside a paragraph are kept. */
export function parseBlocks(text: string): Block[] {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let paragraph: string[] = [];
  let list: ListDraft | null = null;

  function closeParagraph() {
    if (paragraph.length > 0) {
      blocks.push({ kind: "paragraph", lines: paragraph.map((line) => parseInline(line.trim())) });
      paragraph = [];
    }
  }

  function closeList() {
    if (list !== null) {
      const { ordered, items } = list;
      blocks.push({ kind: "list", ordered, items: items.map((item) => parseInline(item.trim())) });
      list = null;
    }
  }

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index] ?? "";

    if (line.trim() === "") {
      closeParagraph();
      closeList();
      continue;
    }

    if (startsTable(line, lines[index + 1] ?? "")) {
      closeParagraph();
      closeList();
      const header = splitRow(line);
      const rows: string[][] = [];
      let row = index + 2;
      while (row < lines.length && (lines[row] ?? "").trim() !== "" && (lines[row] ?? "").includes("|")) {
        const cells = splitRow(lines[row] ?? "");
        rows.push(header.map((_, column) => cells[column] ?? ""));
        row += 1;
      }
      blocks.push(table(header, rows));
      index = row - 1;
      continue;
    }

    const heading = HEADING.exec(line);
    if (heading) {
      closeParagraph();
      closeList();
      blocks.push({ kind: "heading", content: parseInline(heading[1] ?? "") });
      continue;
    }

    const bullet = BULLET.exec(line);
    const numbered = bullet ? null : NUMBERED.exec(line);
    const item = bullet ?? numbered;
    if (item) {
      closeParagraph();
      const ordered = numbered !== null;
      if (list === null || list.ordered !== ordered) {
        closeList();
        list = { ordered, items: [] };
      }
      list.items.push(item[1] ?? "");
      continue;
    }

    if (list !== null && CONTINUATION.test(line)) {
      const { items } = list;
      const last = items.length - 1;
      items[last] = `${items[last] ?? ""} ${line.trim()}`;
      continue;
    }

    closeList();
    paragraph.push(line);
  }

  closeParagraph();
  closeList();
  return blocks;
}
