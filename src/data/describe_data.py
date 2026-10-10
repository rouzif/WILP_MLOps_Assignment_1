import pandas as pd
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def describe_data(df: pd.DataFrame) -> dict[str, list[str]]:
    """Logs data types, column categories, summary stats, and percentage distributions,

    returning a dictionary of categorized columns.
    """
    logger.info("\n\n=== Starting Data Description ===")

    # 1. Log data type for each column
    logger.info("\n--- Column Data Types ---")
    for col, dtype in df.dtypes.items():
        logger.info(f"Column: '{col}' | DataType: {dtype}")

    # Initialize categories dict
    column_categories: dict[str, list[str]] = {
        "numeric-low-cardinality": [],
        "numeric-distributed": [],
        "categorical": [],
        "text": [],
    }

    # 2. Categorise columns and log category for each
    logger.info("\n--- Column Categorization ---")
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        if is_numeric:
            if n_unique <= 5:
                cat = "numeric-low-cardinality"
            else:
                cat = "numeric-distributed"
        else:
            if n_unique <= 5:
                cat = "categorical"
            else:
                cat = "text"

        column_categories[cat].append(col)
        logger.info(
            f"Column: '{col}' | Unique Values: {n_unique} | Category: {cat}"
        )

    # 3. Log min, 25%, median, 75%, max for numeric columns
    logger.info("\n--- Summary Statistics for Numeric Columns ---")
    numeric_cols = (
        column_categories["numeric-low-cardinality"]
        + column_categories["numeric-distributed"]
    )

    if numeric_cols:
        quantiles = df[numeric_cols].quantile([0.0, 0.25, 0.50, 0.75, 1.0])
        for col in numeric_cols:
            stats_str = (
                f"Min: {quantiles.loc[0.00, col]:.2f} | "
                f"25%: {quantiles.loc[0.25, col]:.2f} | "
                f"Median: {quantiles.loc[0.50, col]:.2f} | "
                f"75%: {quantiles.loc[0.75, col]:.2f} | "
                f"Max: {quantiles.loc[1.00, col]:.2f}"
            )
            logger.info(f"Column: '{col}' -> {stats_str}")
    else:
        logger.info("No numeric columns found.")

    # 4. Log % distribution across categories for columns with < 5 unique entries
    logger.info(
        "\n--- Percentage Distribution for Low-Cardinality Columns (< 5 Unique"
        " Entries) ---"
    )
    low_cardinality_cols = [
        col for col in df.columns if df[col].nunique(dropna=True) < 5
    ]

    if low_cardinality_cols:
        for col in low_cardinality_cols:
            dist = df[col].value_counts(normalize=True, dropna=False) * 100
            dist_str = ", ".join(
                [f"'{val}': {pct:.2f}%" for val, pct in dist.items()]
            )
            logger.info(f"Column: '{col}' -> [{dist_str}]")
    else:
        logger.info("No columns with fewer than 5 unique values found.")

    logger.info("\n=== Finished Data Description ===")

    # Return column category dict
    return column_categories


if __name__ == "__main__":
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["input_data_path"])
    categories = describe_data(df)
    logger.info(f"Categorized columns mapping: {categories}")
