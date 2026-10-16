import os
from pathlib import Path
import joblib
import mlflow
from mlflow import MlflowClient
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def register_model(cfg):

    logger.info("Registering model...")

    mlflow.set_tracking_uri(cfg["ml_flow"]["tracking_uri"])

    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(cfg["ml_flow"]["experiment"])

    # Getting best experiment
    best = client.search_runs(
        [exp.experiment_id],
        order_by=["metrics.test_roc_auc DESC"],
        max_results=1,
    )[0]
    run_id = best.info.run_id

    # Logging best model details
    logger.info(
        f"Best run {best.info.run_id[:8]} auc={best.data.metrics['test_roc_auc']:.4f}"
    )
    logger.info(f"Best params={best.data.params}")

    # Register Trained Classifier Model
    model_name = "heart-disease-classifier"
    model_mv = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model", name=model_name
    )
    client.set_registered_model_alias(model_name, "champion", model_mv.version)

    # Register Preprocessor
    prep_name = "heart-disease-preprocessor"
    prep_mv = mlflow.register_model(
        model_uri=f"runs:/{run_id}/preprocessor", name=prep_name
    )
    client.set_registered_model_alias(prep_name, "champion", prep_mv.version)

    logger.info(
        f"Registered {model_name} v{model_mv.version} and {prep_name} v{prep_mv.version} as @champion"
    )

    # --- Export Champion Artifacts to local disk (model/champion) ---
    export_dir = Path(cfg['model']['champion_directory'])
    export_dir.mkdir(parents=True, exist_ok=True)

    # Method 1: Download registered champion artifacts using MLflow
    model_export_path = client.download_artifacts(
        run_id=run_id, path="model", dst_path=str(export_dir)
    )
    prep_export_path = client.download_artifacts(
        run_id=run_id, path="preprocessor", dst_path=str(export_dir)
    )

    # 2. Save standalone .joblib files inside model/champion
    # Load objects from MLflow download and dump direct .joblib binaries
    loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    loaded_prep = mlflow.sklearn.load_model(f"runs:/{run_id}/preprocessor")

    joblib.dump(loaded_model, export_dir / "model.joblib")
    joblib.dump(loaded_prep, export_dir / "preprocessor.joblib")

    logger.info(
        f"Exported champion model and preprocessor artifacts to '{export_dir.resolve()}'"
    )
