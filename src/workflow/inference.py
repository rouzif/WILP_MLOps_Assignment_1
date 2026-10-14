import warnings
import joblib
import pandas as pd
import mlflow
from src.model.predict import run_prediction_pipeline
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


def inference_pipeline():

    logger.info("\n\n")
    logger.info("Starting Machine Learning Inference pipeline!\n\n")
    warnings.filterwarnings("ignore")

    # Loading config
    cfg = load_config()

    # Loading model and preprocessor
    mlflow.set_tracking_uri(cfg['ml_flow']['tracking_uri'])
    model = mlflow.sklearn.load_model("models:/heart-disease-classifier@champion")
    preprocessor = mlflow.sklearn.load_model("models:/heart-disease-preprocessor@champion")

    # Read input and converting to dataframe
    input_jsonl_path = cfg["data"].get(
        "inference_input_path", "data/prediction_input/sample_pred_input.jsonl"
    )
    logger.info(f"Loading inference input data from: '{input_jsonl_path}'")
    df = pd.read_json(input_jsonl_path, lines=True)

    # Running preprocessing
    df_processed = preprocessor.transform(df)

    # Getting predictions
    df_test_predictions = run_prediction_pipeline(
        cfg, df_processed, model, "inference"
    )

    # Log predictions for each ID
    logger.info("--- Inference Predictions Summary ---")
    id_col = cfg["variables"].get("id_column", "id") if "variables" in cfg else "id"

    for idx, row in df_test_predictions.iterrows():
        record_id = row[id_col] if id_col in row else idx
        pred_prob = row.get("predicted_probability", None)
        logger.info(f"Record ID: {record_id} | Predicted Probability: {round(pred_prob, 3)}")

    logger.info("Machine Learning Inference pipeline completed successfully!")

    return None


if __name__ == "__main__":
    inference_pipeline()