import json
import pandas as pd
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def encode_data(
    df: pd.DataFrame, column_types: dict
) -> tuple[pd.DataFrame, dict, dict[str, list]]:
    """One-hot encodes numerical-low-cardinality and categorical columns (dropping the last category),

    updates the column_types dictionary with encoded column names, and returns
    the encoding scheme mapping.
    """
    df = df.copy()

    logger.info("=== Starting Data Encoding ===")

    # Identify target columns to encode
    cols_to_encode = column_types.get(
        "numeric-low-cardinality", []
    ) + column_types.get("categorical", [])

    logger.info(
        f"Columns selected for one-hot encoding ({len(cols_to_encode)}): {cols_to_encode}"
    )

    encoded_columns_list = []
    encoding_scheme = {}

    for col in cols_to_encode:
        if col in df.columns:
            # Generate one-hot encoded columns
            dummies = pd.get_dummies(
                df[col], prefix=col, drop_first=False, dtype=int
            )

            # Record categories being encoded before dropping the last category
            categories = [
                cat
                for cat in df[col].dropna().unique().tolist()
            ]

            # Drop the last column to satisfy 'drop last' rule
            if dummies.shape[1] > 1:
                dropped_col = dummies.columns[-1]
                dummies = dummies.iloc[:, :-1]
                encoded_cats = categories[:-1]
                logger.info(
                    f"Encoded column '{col}' -> Created {dummies.shape[1]} dummy column(s), dropped last category '{dropped_col}'"
                )
            else:
                encoded_cats = categories
                logger.info(
                    f"Encoded column '{col}' -> Created 1 dummy column"
                )

            encoding_scheme[col] = encoded_cats
            encoded_columns_list.extend(dummies.columns.tolist())

            # Concatenate encoded dummy columns and drop original column
            df = pd.concat([df, dummies], axis=1).drop(columns=[col])
        else:
            logger.warning(
                f"Column '{col}' specified for encoding not found in DataFrame. Skipping."
            )

    # Create new key 'encoded' in the column_types dictionary
    column_types["encoded"] = encoded_columns_list

    logger.info(
        f"Total generated encoded features ({len(encoded_columns_list)}): {encoded_columns_list}"
    )
    logger.info(f"Updated DataFrame shape post-encoding: {df.shape}")
    logger.info("=== Finished Data Encoding ===")

    return df, column_types, encoding_scheme


def run_encoding_pipeline(
    cfg: dict, df: pd.DataFrame, column_types: dict
) -> tuple[pd.DataFrame, dict, dict]:
    """Executes the encoding step, saves outputs to configured paths, and returns the results."""
    logger.info("\n\n")
    logger.info("=== Running Feature Encoding Pipeline ===")

    # Call encode_data function
    df_encoded, column_types_updated, encoding_scheme = encode_data(
        df, column_types
    )

    # Save the column_types dictionary returned to config['data']['column_category_encoded_path']
    encoded_cat_path = cfg["data"]["column_category_encoded_path"]
    with open(encoded_cat_path, "w", encoding="utf-8") as f:
        json.dump(column_types_updated, f, indent=4)
    logger.info(
        f"Saved updated column categories JSON to: '{encoded_cat_path}'"
    )

    # Optionally save encoding scheme if configured
    encoding_scheme_path = cfg["data"].get("encoding_scheme_path")
    if encoding_scheme_path:
        with open(encoding_scheme_path, "w", encoding="utf-8") as f:
            json.dump(encoding_scheme, f, indent=4)
        logger.info(
            f"Saved encoding scheme mapping JSON to: '{encoding_scheme_path}'"
        )

    # Save dataframe returned to config['data']['encoded_data_path']
    encoded_data_path = cfg["data"]["encoded_data_path"]
    df_encoded.to_csv(encoded_data_path, index=False)
    logger.info(f"Saved encoded DataFrame CSV to: '{encoded_data_path}'")

    logger.info("=== Finished Encoding Pipeline Execution ===")
    return df_encoded, column_types_updated, encoding_scheme


if __name__ == "__main__":
    logger.info("Initializing Encoding Pipeline Script Execution...")

    # Load config
    cfg = load_config()

    # Load dataframe from config['data']['missing_imputed_data_path']
    imputed_path = cfg["data"]["missing_imputed_data_path"]
    logger.info(f"Loading imputed data from: '{imputed_path}'")
    df_imputed = pd.read_csv(imputed_path)

    # Load column types dictionary from config['data']['column_category_path']
    cat_path = cfg["data"]["column_category_path"]
    logger.info(f"Loading column category metadata from: '{cat_path}'")
    with open(cat_path, "r", encoding="utf-8") as f:
        col_types = json.load(f)

    # Execute encoding pipeline
    df_encoded, updated_col_types, enc_scheme = run_encoding_pipeline(
        cfg, df_imputed, col_types
    )

    logger.info("Script execution completed successfully.")