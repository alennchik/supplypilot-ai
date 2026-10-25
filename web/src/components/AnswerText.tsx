import { Fragment } from "react";
import type { ReactNode } from "react";

import { parseBlocks } from "../markdown.ts";
import type { Block, Inline } from "../markdown.ts";

function renderInline(parts: Inline[]): ReactNode[] {
  return parts.map((part, index) => {
    if (part.kind === "strong") {
      return <strong key={index}>{part.text}</strong>;
    }
    if (part.kind === "code") {
      return <code key={index}>{part.text}</code>;
    }
    return <Fragment key={index}>{part.text}</Fragment>;
  });
}

function renderLines(lines: Inline[][]): ReactNode[] {
  return lines.map((line, index) => (
    <Fragment key={index}>
      {index > 0 ? <br /> : null}
      {renderInline(line)}
    </Fragment>
  ));
}

function renderBlock(block: Block, key: number): ReactNode {
  switch (block.kind) {
    case "paragraph":
      return <p key={key}>{renderLines(block.lines)}</p>;
    case "heading":
      return (
        <p key={key} className="answer-heading">
          {renderInline(block.content)}
        </p>
      );
    case "list": {
      const items = block.items.map((item, index) => <li key={index}>{renderInline(item)}</li>);
      return block.ordered ? <ol key={key}>{items}</ol> : <ul key={key}>{items}</ul>;
    }
    case "table":
      return (
        <div key={key} className="answer-table">
          <table>
            <thead>
              <tr>
                {block.header.map((cell, index) => (
                  <th key={index} scope="col" className={block.numeric[index] ? "numeric" : undefined}>
                    {renderInline(cell)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, index) => (
                    <td key={index} className={block.numeric[index] ? "numeric" : undefined}>
                      {renderInline(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
  }
}

/** The assistant's answer with its tables, lists and bold text laid out. */
export function AnswerText({ text }: { text: string }) {
  return <div className="agent-text">{parseBlocks(text).map((block, index) => renderBlock(block, index))}</div>;
}
