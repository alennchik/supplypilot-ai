from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_top_products(
    product_sales: pd.Series,
    output_path: str | Path,
) -> None:
    """Save a bar chart of the top-selling products."""
    if product_sales.empty:
        raise ValueError("Product sales cannot be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    product_sales.sort_values().plot.barh(ax=ax)

    ax.set_title("Top-Selling Walmart Products")
    ax.set_xlabel("Total Units Sold")
    ax.set_ylabel("Product ID")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_daily_sales(
    daily_sales: pd.Series,
    output_path: str | Path,
) -> None:
    """Save a line chart showing daily sales trends."""
    if daily_sales.empty:
        raise ValueError("Daily sales cannot be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(range(1, len(daily_sales) + 1), daily_sales.values)

    ax.set_title("Walmart M5 Daily Sales Trend")
    ax.set_xlabel("Day")
    ax.set_ylabel("Total Units Sold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
