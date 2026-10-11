import json
import pandas as pd
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def scale_data(
    df: pd.DataFrame, column_types: dict
) -> tuple[pd.DataFrame, dict, dict]:
    """Scales numeric distributed columns using mean and standard deviation,

    adds scaled feature columns to the DataFrame, updates the column_types
    dictionary with the 'scaled_numeric' key, and returns the scaling parameters.
    """
    df = df.copy()

    logger.info("=== Starting Data Scaling ===")

    # Identify numeric distributed columns to scale
    numeric_cols = column_types.get("numeric-distributed", [])

    logger.info(
        f"Columns selected for z-score scaling ({len(numeric_cols)}): {numeric_cols}"
    )

    scaled_columns_list = []
    scaling_params = {}

    for col in numeric_cols:
        if col in df.columns:
            mean_val = float(df[col].mean())
            std_val = float(df[col].std())

            # Handle edge case where standard deviation is zero
            if std_val == 0:
                logger.warning(
                    f"Column '{col}' has standard deviation of 0. Standardized values set to 0.0."
                )
                scaled_col_name = f"{col}_scaled"
                df[scaled_col_name] = 0.0
            else:
                scaled_col_name = f"{col}_scaled"
                df[scaled_col_name] = (df[col] - mean_val) / std_val

            scaled_columns_list.append(scaled_col_name)
            scaling_params[col] = {"mean": mean_val, "std": std_val}

            logger.info(
                f"Scaled column '{col}' -> Mean: {mean_val:.4f}, Std: {std_val:.4f} | Created: '{scaled_col_name}'"
            )
        else:
            logger.warning(
                f"Column '{col}' specified for scaling not found in DataFrame. Skipping."
            )

    # Add 'scaled_numeric' key to column_types dictionary
    column_types["scaled_numeric"] = scaled_columns_list

    logger.info(
        f"Total generated scaled features ({len(scaled_columns_list)}): {scaled_columns_list}"
    )
    logger.info(f"Updated DataFrame shape post-scaling: {df.shape}")
    logger.info("=== Finished Data Scaling ===")

    return df, column_types, scaling_params


def run_scaling_pipeline(
    cfg: dict, df: pd.DataFrame, column_types: dict
) -> tuple[pd.DataFrame, dict, dict]:
    """Executes the scaling step, saves outputs to configured paths, and returns the results."""
    logger.info("\n\n")
    logger.info("=== Running Feature Scaling Pipeline ===")

    # Call scale_data function
    df_scaled, column_types_updated, scaling_params = scale_data(
        df, column_types
    )

    # Save the column_types dictionary returned to config['data']['column_category_scaled_path']
    scaled_cat_path = cfg["data"].get("column_category_scaled_path")
    if scaled_cat_path:
        with open(scaled_cat_path, "w", encoding="utf-8") as f:
            json.dump(column_types_updated, f, indent=4)
        logger.info(
            f"Saved updated column categories JSON to: '{scaled_cat_path}'"
        )

    # Save scaling parameters (mean and std) to config['data']['scaling_params_path']
    scaling_params_path = cfg["data"].get("scaling_params_path")
    if scaling_params_path:
        with open(scaling_params_path, "w", encoding="utf-8") as f:
            json.dump(scaling_params, f, indent=4)
        logger.info(f"Saved scaling parameters JSON to: '{scaling_params_path}'")

    # Save dataframe returned to config['data']['scaled_data_path']
    scaled_data_path = cfg["data"].get("scaled_data_path")
    if scaled_data_path:
        df_scaled.to_csv(scaled_data_path, index=False)
        logger.info(f"Saved scaled DataFrame CSV to: '{scaled_data_path}'")

    logger.info("=== Finished Scaling Pipeline Execution ===")
    return df_scaled, column_types_updated, scaling_params


if __name__ == "__main__":
    logger.info("Initializing Scaling Pipeline Script Execution...")

    # Load config
    cfg = load_config()

    # Load dataframe from config['data']['encoded_data_path']
    encoded_path = cfg["data"]["encoded_data_path"]
    logger.info(f"Loading encoded data from: '{encoded_path}'")
    df_encoded = pd.read_csv(encoded_path)

    # Load column types dictionary from config['data']['column_category_encoded_path']
    encoded_cat_path = cfg["data"]["column_category_encoded_path"]
    logger.info(
        f"Loading encoded column category metadata from: '{encoded_cat_path}'"
    )
    with open(encoded_cat_path, "r", encoding="utf-8") as f:
        col_types = json.load(f)

    # Execute scaling pipeline
    df_scaled, updated_col_types, scaling_parameters = run_scaling_pipeline(
        cfg, df_encoded, col_types
    )

    logger.info("Script execution completed successfully.")