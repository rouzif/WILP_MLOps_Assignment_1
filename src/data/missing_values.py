import json
import numpy as np
import pandas as pd
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def impute_missing_values(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """Imputes missing values in a dataframe according to column category and missing percentage rules."""
    df = df.copy()

    logger.info("=== Starting Missing Values Imputation ===")

    # Invert the dictionary to map each column to its category
    col_to_category = {}
    for category, cols in column_types.items():
        for col in cols:
            col_to_category[col] = category

    total_rows = len(df)
    missing_imputation = {}

    # For each column in data
    for col in df.columns:
        missing_count = df[col].isna().sum()

        if missing_count == 0:
            logger.info(f"Column '{col}': No missing values found.")
            continue

        missing_pct = (missing_count / total_rows) * 100
        category = col_to_category.get(col, "unknown")

        logger.info(
            f"Processing Column '{col}' | Category: {category} | Missing: {missing_count} ({missing_pct:.2f}%)"
        )

        # 1. For numeric distributed columns impute median
        if category == "numeric-distributed":
            median_val = float(df[col].median())
            df[col] = df[col].fillna(median_val)
            missing_imputation[col] = median_val
            logger.info(f"Imputed column '{col}' missing values with median: {median_val}")

        # 2. For numeric low cardinality columns
        # Impute max(col) + 1 if missing % is > 10% else impute mode
        elif category == "numeric-low-cardinality":
            if missing_pct > 10:
                fill_val = float(df[col].max() + 1)
                df[col] = df[col].fillna(fill_val)
                missing_imputation[col] = fill_val
                logger.info(
                    f"Imputed column '{col}' missing values with max + 1 value: {fill_val} (Missing > 10%)"
                )
            else:
                mode_val = float(df[col].mode()[0])
                df[col] = df[col].fillna(mode_val)
                missing_imputation[col] = mode_val
                logger.info(
                    f"Imputed column '{col}' missing values with mode: {mode_val} (Missing <= 10%)"
                )

        # 3. For categorical columns
        # Impute "Missing" if missing % is > 10% else impute mode
        elif category in ("categorical", "text"):
            if missing_pct > 10:
                fill_val = "Missing"
                df[col] = df[col].fillna(fill_val)
                missing_imputation[col] = fill_val
                logger.info(
                    f"Imputed column '{col}' missing values with category: '{fill_val}' (Missing > 10%)"
                )
            else:
                mode_val = df[col].mode()[0]
                fill_val = str(mode_val) if not isinstance(mode_val, (int, float)) else float(mode_val)
                df[col] = df[col].fillna(fill_val)
                missing_imputation[col] = fill_val
                logger.info(
                    f"Imputed column '{col}' missing values with mode: '{fill_val}' (Missing <= 10%)"
                )

        else:
            logger.warning(
                f"Column '{col}' was not found in column_types mapping. Skipping imputation."
            )

    logger.info("=== Finished Missing Values Imputation ===")
    return df, missing_imputation


def run_missing_value_pipeline(cfg: dict, df: pd.DataFrame, column_types: dict) -> tuple[pd.DataFrame, dict[str, list[str]]]:

    logger.info("\n\n")
    logger.info("=== Running Missing Value Imputation Pipeline ===")

    # Imputing missing values
    df, missing_imputation = impute_missing_values(df, column_types)

    # Saving data
    df.to_csv(cfg["data"]["missing_imputed_data_path"], index=False)
    with open(cfg['data']['missing_imputation_path'], "w", encoding="utf-8") as f:
        json.dump(missing_imputation, f, indent=4)

    logger.info("=== Finished Missing Value Pipeline Execution ===")
    return df, missing_imputation


if __name__ == "__main__":

    # Loading data and config
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["input_data_path"])

    # Loading column types
    with open(cfg['data']['column_category_path'], "r", encoding="utf-8") as f:
        column_types = json.load(f)

    # Running missing value pipeline
    run_missing_value_pipeline(cfg, df, column_types)
