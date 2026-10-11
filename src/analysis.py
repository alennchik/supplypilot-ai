import pandas as pd

from src.data_pipeline import process_sales_in_chunks


def calculate_sales_statistics(df: pd.DataFrame) -> dict:
    """Calculate basic statistics for sales data."""
    return {
        "total_sales": int(df["sales"].sum()),
        "average_sales": float(df["sales"].mean()),
        "max_sales": int(df["sales"].max()),
        "zero_sales_days": int((df["sales"] == 0).sum()),
    }

def analyze_sales_dataset(path: str, chunk_size: int = 100) -> dict:
    """Calculate sales statistics across the entire dataset."""
    total_sales = 0
    total_rows = 0
    zero_sales_days = 0
    max_sales = 0

    for chunk in process_sales_in_chunks(path, chunk_size):
        stats = calculate_sales_statistics(chunk)

        total_sales += stats["total_sales"]
        total_rows += len(chunk)
        zero_sales_days += stats["zero_sales_days"]
        max_sales = max(max_sales, stats["max_sales"])

    return {
        "total_sales": total_sales,
        "average_sales": total_sales / total_rows,
        "max_sales": max_sales,
        "zero_sales_days": zero_sales_days,
        "total_rows": total_rows,
    }


def get_top_products(path: str, top_n: int = 10, chunk_size: int = 100) -> pd.Series:
    """Find the products with the highest total sales."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    product_sales = {}

    for chunk in process_sales_in_chunks(path, chunk_size):
        totals = chunk.groupby("item_id")["sales"].sum()

        for item_id, sales in totals.items():
            product_sales[item_id] = product_sales.get(item_id, 0) + int(sales)

    return pd.Series(product_sales, dtype="int64").nlargest(top_n)



def get_daily_sales(path: str, chunk_size: int = 100) -> pd.Series:
    """Calculate total sales for each day across all products and stores."""
    daily_totals = {}

    for chunk in process_sales_in_chunks(path, chunk_size):
        totals = chunk.groupby("day")["sales"].sum()

        for day, sales in totals.items():
            daily_totals[day] = daily_totals.get(day, 0) + int(sales)

    result = pd.Series(daily_totals, dtype="int64")

    return result.sort_index(
        key=lambda days: days.str.replace("d_", "", regex=False).astype(int)
    )


def get_products_with_most_zero_sales(
    path: str,
    top_n: int = 10,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """Find products with the highest percentage of zero-sales days."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    product_stats = {}

    for chunk in process_sales_in_chunks(path, chunk_size):
        stats = chunk.groupby("item_id")["sales"].agg(
            zero_days=lambda sales: (sales == 0).sum(),
            total_days="count",
        )

        for item_id, row in stats.iterrows():
            if item_id not in product_stats:
                product_stats[item_id] = [0, 0]

            product_stats[item_id][0] += int(row["zero_days"])
            product_stats[item_id][1] += int(row["total_days"])

    results = pd.DataFrame.from_dict(
        product_stats,
        orient="index",
        columns=["zero_days", "total_days"],
    )

    results.index.name = "item_id"
    results["zero_percentage"] = (
        results["zero_days"] / results["total_days"] * 100
    )

    return results.sort_values(
        ["zero_percentage", "zero_days"],
        ascending=False,
    ).head(top_n)
