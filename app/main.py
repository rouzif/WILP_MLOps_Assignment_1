import joblib
import mlflow
import mlflow.sklearn
import uvicorn
import warnings
import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Request
from prometheus_fastapi_instrumentator import Instrumentator
import pandas as pd

from app.schemas import HeartDataInput, PredictionResponse
from src.data.preprocessor import DataPreprocessor
from src.utils.config import load_config
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# Global model state
ML_MODELS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load ML artifacts at startup."""
    logger.info("=== Initializing API Server & Loading Artifacts ===")
    cfg = load_config()

    try:

        mlflow.set_tracking_uri(cfg["ml_flow"]["tracking_uri"])

        try:
            logger.info("Attempting to load Champion artifacts from MLflow...")
            ML_MODELS["preprocessor"] = mlflow.sklearn.load_model(
                "models:/heart-disease-preprocessor@champion",
                skops_trusted_types=[DataPreprocessor],
            )
            ML_MODELS["model"] = mlflow.sklearn.load_model(
                "models:/heart-disease-classifier@champion"
            )
            logger.info("Successfully loaded Champion artifacts from MLflow.")
        except Exception as e:
            logger.warning(
                f"Failed to load from MLflow Model Registry ({e}). Falling back to local disk artifacts."
            )
            prep_path = Path(cfg["model"]["champion_directory"] + 'preprocessor/')
            model_path = Path(cfg["model"]["champion_directory"] + 'model/')

            # Option A: Load using mlflow.sklearn from local directory (Recommended for MLflow formats)
            if prep_path.exists() and model_path.exists():
                ML_MODELS["preprocessor"] = mlflow.sklearn.load_model(
                    str(prep_path)
                )
                ML_MODELS["model"] = mlflow.sklearn.load_model(str(model_path))
                logger.info("Successfully loaded artifacts from 'model/champion/'.")

            # Option B: Direct joblib load if you saved single .joblib files inside model/champion/
            else:
                prep_joblib = Path(cfg["model"]["champion_directory"] / "model.joblib")
                model_joblib = Path(cfg["model"]["champion_directory"] / "preprocessor.joblib")

                ML_MODELS["preprocessor"] = joblib.load(prep_joblib)
                ML_MODELS["model"] = joblib.load(model_joblib)
                logger.info("Successfully loaded .joblib files from 'model/champion/'.")

    except Exception as err:
        logger.error(f"Critical error loading model artifacts: {err}", exc_info=True)
        raise RuntimeError(f"Could not initialize ML artifacts: {err}")

    yield

    # Clean up state on shutdown
    ML_MODELS.clear()
    logger.info("=== Shutting Down API Server ===")


# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production-grade MLOps API serving Heart Disease classification model.",
    version="1.0.0",
    lifespan=lifespan,
)

# Instrument Prometheus metrics endpoint (/metrics) for Task 8
Instrumentator().instrument(app).expose(app)



@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Intercepts incoming HTTP requests, measures execution time,

    and logs operational request metadata.
    """
    start_time = time.time()

    # Process the incoming request
    response = await call_next(request)

    # Calculate execution duration in milliseconds
    process_time_ms = (time.time() - start_time) * 1000

    # Retrieve request details
    client_host = request.client.host if request.client else "unknown"
    method = request.method
    url_path = request.url.path
    status_code = response.status_code

    # Format log message
    log_msg = (
        f"Client: {client_host} | "
        f"Method: {method} | "
        f"Path: {url_path} | "
        f"Status: {status_code} | "
        f"Latency: {process_time_ms:.2f}ms"
    )

    if status_code >= 500:
        logger.error(log_msg)
    elif status_code >= 400:
        logger.warning(log_msg)
    else:
        logger.info(log_msg)

    return response


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    if "model" not in ML_MODELS or "preprocessor" not in ML_MODELS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded.",
        )
    return {"status": "healthy", "service": "heart-disease-prediction-api"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict_heart_disease(payload: HeartDataInput):
    """Accepts raw patient health metrics in JSON, transforms features,

    and returns predicted risk class and confidence score.
    """
    if "model" not in ML_MODELS or "preprocessor" not in ML_MODELS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model server is not fully initialized.",
        )

    try:
        # 1. Convert incoming Pydantic payload to DataFrame
        raw_dict = payload.model_dump()
        df_raw = pd.DataFrame([raw_dict])

        # 2. Apply fitted preprocessor
        preprocessor = ML_MODELS["preprocessor"]
        df_processed = preprocessor.transform(df_raw)

        # 3. Predict class probability & label
        model = ML_MODELS["model"]

        # Selecting only required features
        df_processed = df_processed[model.feature_names_in_]

        if hasattr(model, "predict_proba"):
            # Probabilities for positive class (Class 1)
            prob_class_1 = float(model.predict_proba(df_processed)[0, 1])
        else:
            prob_class_1 = float(model.predict(df_processed)[0])

        pred_class = int(prob_class_1 >= 0.5)
        label = "High Risk" if pred_class == 1 else "No Risk"

        return PredictionResponse(
            prediction=pred_class,
            prediction_label=label,
            confidence=round(prob_class_1, 4),
        )

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing prediction request: {str(e)}",
        )


if __name__ == "__main__":

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)