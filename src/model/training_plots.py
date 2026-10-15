from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_training_plots(
    y_pred_prob: pd.Series,
    y_true: pd.Series,
    model,
    output_dir: str = "data/training_plot/",
) -> None:
    """Generates evaluation plots (ROC curve, PR curve, Confusion Matrix, and Feature Importances)

    and saves them as PNG files in the configured output directory.
    """
    logger.info("=== Starting Evaluation Plots Generation ===")

    sns.set_theme(style="whitegrid")

    # Ensure output directory exists
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC Curve
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
        RocCurveDisplay.from_predictions(
            y_true, y_pred_prob, ax=ax_roc, name="Model"
        )
        ax_roc.set_title("ROC Curve", fontsize=12, fontweight="bold")
        fig_roc.tight_layout()

        roc_path = plot_dir / "roc_curve.png"
        fig_roc.savefig(roc_path, dpi=300)
        plt.close(fig_roc)
        logger.info(f"Saved ROC Curve plot -> '{roc_path}'")
    except Exception as e:
        logger.error(f"Failed to generate ROC Curve plot: {e}", exc_info=True)

    # 2. Precision-Recall Curve
    try:
        fig_pr, ax_pr = plt.subplots(figsize=(7, 5))
        PrecisionRecallDisplay.from_predictions(
            y_true, y_pred_prob, ax=ax_pr, name="Model"
        )
        ax_pr.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
        fig_pr.tight_layout()

        pr_path = plot_dir / "precision_recall_curve.png"
        fig_pr.savefig(pr_path, dpi=300)
        plt.close(fig_pr)
        logger.info(f"Saved Precision-Recall Curve plot -> '{pr_path}'")
    except Exception as e:
        logger.error(f"Failed to generate Precision-Recall Curve plot: {e}", exc_info=True)

    # 3. Confusion Matrix
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        y_pred = (y_pred_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm).plot(ax=ax_cm, cmap="Blues")
        ax_cm.set_title("Confusion Matrix (Threshold = 0.5)", fontsize=12, fontweight="bold")
        fig_cm.tight_layout()

        cm_path = plot_dir / "confusion_matrix.png"
        fig_cm.savefig(cm_path, dpi=300)
        plt.close(fig_cm)
        logger.info(f"Saved Confusion Matrix plot -> '{cm_path}'")
    except Exception as e:
        logger.error(f"Failed to generate Confusion Matrix plot: {e}", exc_info=True)

    # 4. Feature Importance Plot (if available on the estimator)
    if hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
        try:
            fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
            importances = pd.Series(
                model.feature_importances_, index=model.feature_names_in_
            ).sort_values(ascending=False)

            sns.barplot(
                x=importances.values,
                y=importances.index,
                palette="viridis",
                ax=ax_fi,
            )
            ax_fi.set_title("Feature Importances", fontsize=12, fontweight="bold")
            ax_fi.set_xlabel("Importance Score")
            fig_fi.tight_layout()

            fi_path = plot_dir / "feature_importance.png"
            fig_fi.savefig(fi_path, dpi=300)
            plt.close(fig_fi)
            logger.info(f"Saved Feature Importance plot -> '{fi_path}'")
        except Exception as e:
            logger.error(f"Failed to generate Feature Importance plot: {e}", exc_info=True)

    logger.info("=== Finished Evaluation Plots Generation ===")