import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def partition_data(
    df: pd.DataFrame, test_ratio: float = 0.3, strata: pd.Series = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into train and test sets, optionally stratified on the provided series."""
    logger.info("=== Starting Data Partitioning ===")
    logger.info(
        f"Input DataFrame shape: {df.shape} | Test ratio: {test_ratio}"
    )

    df_train, df_test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=42,
        stratify=strata,
    )

    logger.info(
        f"Partitioning complete -> Train shape: {df_train.shape}, Test shape: {df_test.shape}"
    )
    logger.info("=== Finished Data Partitioning ===")

    return df_train, df_test


def run_data_partitioning(
    cfg: dict, df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executes the data partitioning step, saves train/test outputs to configured paths, and returns the DataFrames."""
    logger.info("\n\n")
    logger.info("=== Running Data Partitioning Pipeline ===")

    # Retrieve target variable name and test ratio from config
    target_var = cfg["variables"]["target"]
    test_ratio = cfg.get("partition", {}).get("test_ratio", 0.3)

    if target_var in df.columns:
        strata_series = df[target_var]
        logger.info(f"Stratifying partition split on target variable: '{target_var}'")
    else:
        strata_series = None
        logger.warning(
            f"Target variable '{target_var}' not found in DataFrame. Performing unstratified split."
        )

    # Call partition_data
    df_train, df_test = partition_data(
        df=df, test_ratio=test_ratio, strata=strata_series
    )

    # Save df_train and df_test
    train_path = cfg["data"]["train_data_path"]
    test_path = cfg["data"]["test_data_path"]

    df_train.to_csv(train_path, index=False)
    logger.info(f"Saved train dataset CSV to: '{train_path}'")

    df_test.to_csv(test_path, index=False)
    logger.info(f"Saved test dataset CSV to: '{test_path}'")

    logger.info("=== Finished Data Partitioning Pipeline Execution ===")
    return df_train, df_test


if __name__ == "__main__":
    logger.info("Initializing Data Partitioning Pipeline Script Execution...")

    # Load config
    cfg = load_config()

    # Load input processed data (e.g., scaled data or imputed/encoded data)
    data_path = cfg["data"]['input_data_path']
    logger.info(f"Loading data for partitioning from: '{data_path}'")
    df_input = pd.read_csv(data_path)

    # Execute data partitioning pipeline
    df_train, df_test = run_data_partitioning(cfg, df_input)

    logger.info("Script execution completed successfully.")