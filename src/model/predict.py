import json
import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_predictions(
    df: pd.DataFrame, model: ClassifierMixin
) -> pd.DataFrame:
    """Generates predicted probabilities using the trained model and adds them

    as a new column 'predicted_probability' to the DataFrame.
    """
    df = df.copy()

    # Resolve feature list used during model training
    feature_list = list(model.feature_names_in_)

    logger.info("=== Starting Inference & Probability Prediction ===")

    # Select features required by the model
    if feature_list:
        X = df[feature_list]
        logger.info(
            f"Using specified feature list ({len(feature_list)} features) for prediction."
        )
    else:
        # Fallback: drop target column if present
        X = df.drop(columns=["target"], errors="ignore")
        logger.info(
            f"No feature list provided. Predicting using all available features ({X.shape[1]} columns)."
        )

    # Get predicted probabilities for positive class (class 1)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        logger.warning(
            "Model does not support 'predict_proba'. Falling back to 'predict'."
        )
        probabilities = model.predict(X)

    # Add predicted probabilities as a new column
    df["predicted_probability"] = probabilities

    logger.info(
        f"Generated predictions for {len(df)} records. Added 'predicted_probability' column."
    )
    logger.info("=== Finished Prediction Execution ===")

    return df


def run_prediction_pipeline(cfg: dict, df: pd.DataFrame, model: ClassifierMixin, tag: str = '') -> pd.DataFrame:
    """Executes the prediction step, saves output predictions to configured path, and returns the DataFrame."""
    logger.info("\n\n")
    logger.info("=== Running Prediction Pipeline ===")

    # Call get_predictions function
    df_predictions = get_predictions(df, model)

    # Save dataset to configured path with suffix tag
    output_path = cfg['data']['predicted_path_suffix'] + f'{tag}.csv'
    df_predictions.to_csv(output_path, index=False)
    logger.info(f"Saved prediction results CSV to: '{output_path}'")

    logger.info("=== Finished Prediction Pipeline Execution ===")
    return df_predictions


if __name__ == "__main__":
    logger.info("Initializing Prediction Pipeline Script Execution...")

    # Load config
    cfg = load_config()

    # Load test dataset (or new incoming data)
    test_data_path = cfg["data"]["scaled_data_path"]
    logger.info(f"Loading test dataset from: '{test_data_path}'")
    df_test = pd.read_csv(test_data_path)

    # Load trained model object
    model_path = cfg["model"]["model_path"]
    logger.info(f"Loading trained model artifact from: '{model_path}'")
    model = joblib.load(model_path)

    # Execute prediction pipeline
    df_results = run_prediction_pipeline(cfg, df_test, model)

    logger.info("Script execution completed successfully.")