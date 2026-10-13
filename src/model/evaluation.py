import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(target: pd.Series, predicted_prob: pd.Series) -> dict[str, float]:
    """Computes Accuracy, Recall, Precision (at 0.5 threshold), and ROC-AUC.

    Returns a dictionary of evaluation metrics.
    """
    logger.info("=== Starting Model Evaluation ===")

    # Convert probabilities to binary predictions using a 0.5 threshold
    predicted_class = (predicted_prob >= 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(target, predicted_class)
    precision = precision_score(target, predicted_class, zero_division=0)
    recall = recall_score(target, predicted_class, zero_division=0)

    try:
        auc = roc_auc_score(target, predicted_prob)
    except ValueError as e:
        logger.warning(f"Could not compute ROC-AUC score: {e}")
        auc = 0.0

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(auc),
    }

    logger.info(
        f"Evaluation Metrics -> Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | ROC-AUC: {auc:.4f}"
    )
    logger.info("=== Finished Model Evaluation ===")

    return metrics


def run_evaluation_pipeline(cfg: dict, df: pd.DataFrame, tag: str = "") -> dict[str, float]:
    """Executes evaluation pipeline, saves metrics JSON to configured path, and returns metrics dict."""
    logger.info("\n\n")
    logger.info(f"=== Running Evaluation Pipeline for {tag.capitalize()} data ===")

    target_var = cfg["variables"]["target"]

    if target_var not in df.columns:
        raise KeyError(f"Target column '{target_var}' not found in the DataFrame.")
    if "predicted_probability" not in df.columns:
        raise KeyError("Column 'predicted_probability' not found in the DataFrame.")

    target = df[target_var]
    predicted_prob = df["predicted_probability"]

    # Call evaluate function
    metrics_dict = evaluate(target, predicted_prob)

    # Save evaluation metrics JSON to configured path
    prefix = cfg["model"]["eval_path_prefix"]
    output_path = f"{prefix}{tag}.json" if tag else f"{prefix}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)

    logger.info(f"Saved evaluation metrics JSON to: '{output_path}'")

    logger.info("=== Finished Evaluation Pipeline Execution ===")
    return metrics_dict


if __name__ == "__main__":
    logger.info("Initializing Evaluation Pipeline Script Execution...")

    # Load config
    cfg = load_config()

    # Load prediction dataset (containing target and predicted_probability)
    pred_prefix = cfg["data"].get("predicted_path_prefix", "data/predictions_")
    pred_data_path = f"{pred_prefix}test.csv"

    logger.info(f"Loading predicted data from: '{pred_data_path}'")
    df_predictions = pd.read_csv(pred_data_path)

    # Execute evaluation pipeline
    metrics = run_evaluation_pipeline(cfg, df_predictions, tag="test")

    logger.info("Script execution completed successfully.")