export interface Column {
  key: string;
  label: string;
  numeric?: boolean;
}

interface DataTableProps {
  caption: string;
  columns: Column[];
  rows: Record<string, string>[];
}

/** The chart's numbers as a table, collapsed until asked for. */
export function DataTable({ caption, columns, rows }: DataTableProps) {
  return (
    <details className="table-view">
      <summary>Show the numbers</summary>
      <div className="table-scroll">
        <table>
          <caption>{caption}</caption>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column.key} scope="col" className={column.numeric ? "numeric" : undefined}>
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={index}>
                {columns.map((column) => (
                  <td key={column.key} className={column.numeric ? "numeric" : undefined}>
                    {row[column.key] ?? ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
