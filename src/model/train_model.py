import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_model_training(df: pd.DataFrame, feature_list: list, target_var: str, model_algo: str):
    """Selects target features, fits the specified classification algorithm,

    and returns the trained model object.
    """
    logger.info("=== Starting Model Training ===")

    # Validate feature presence
    missing_features = [col for col in feature_list if col not in df.columns]
    if missing_features:
        logger.warning(
            f"The following requested features were not found in the DataFrame: {missing_features}"
        )
        feature_list = [col for col in feature_list if col in df.columns]

    X_train = df[feature_list]
    y_train = df[target_var]

    logger.info(
        f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape} | Algorithm: '{model_algo}'"
    )

    # Model instantiation and fitting
    if model_algo == "rf":
        logger.info("Fitting Random Forest Classifier...")
        # TODO: Set random forest params using cross validation
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

    elif model_algo == "xgb":
        logger.info("Fitting Gradient Boosting Classifier...")
        # TODO: Set gradient boosting params using cross validation
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)

    else:
        raise ValueError(
            f"Unsupported algorithm '{model_algo}' specified in configuration. Expected 'rf' or 'xgb'."
        )

    logger.info("=== Model Training Completed Successfully ===")
    return model


def run_training_pipeline(cfg: dict, df: pd.DataFrame, column_types: dict):
    """Executes the training pipeline, saves the model artifact to disk, and returns the model."""
    logger.info("\n\n")
    logger.info("=== Running Model Training Pipeline ===")

    # Retrieve target variable and model algorithm
    target_var = cfg["variables"]["target"]
    model_algo = cfg["model"]["algo"]

    # Select feature list based on custom fit flag or column types
    if cfg["model"].get("custom_feature_fit", False):
        feature_list = cfg["model"]["custom_feature_list"]
        logger.info(f"Using custom feature list ({len(feature_list)}): {feature_list}")
    else:
        encoded_cols = column_types.get("encoded", [])
        scaled_cols = column_types.get("scaled_numeric", [])
        feature_list = encoded_cols + scaled_cols
        logger.info(
            f"Automatically extracted feature list ({len(feature_list)}) from column_types: {feature_list}"
        )

    # Call run_model_training function
    model = run_model_training(df, feature_list, target_var, model_algo)

    # Save trained model object to configured path
    model_path = cfg["model"]["model_path"]
    joblib.dump(model, model_path)
    logger.info(f"Saved trained model artifact to: '{model_path}'")

    logger.info("=== Finished Model Training Pipeline Execution ===")
    return model


if __name__ == "__main__":
    logger.info("Initializing Model Training Pipeline Script Execution...")

    # Load config
    cfg = load_config()

    # Load train dataset from train_data_path
    train_data_path = cfg["data"]["scaled_data_path"]
    logger.info(f"Loading training data from: '{train_data_path}'")
    df_train = pd.read_csv(train_data_path)

    # Load scaled/encoded column types metadata
    col_types_path = cfg["data"].get(
        "column_category_scaled_path", cfg["data"]["column_category_encoded_path"]
    )
    logger.info(f"Loading column metadata from: '{col_types_path}'")
    with open(col_types_path, "r", encoding="utf-8") as f:
        col_types = json.load(f)

    # Execute training pipeline
    trained_model = run_training_pipeline(cfg, df_train, col_types)

    logger.info("Script execution completed successfully.")