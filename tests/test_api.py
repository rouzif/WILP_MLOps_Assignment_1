import pytest
from fastapi.testclient import TestClient
from app.main import app
import warnings
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.data.preprocessor import DataPreprocessor

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module", autouse=True)
def setup_dummy_model_artifacts():
    """Generates dummy model and preprocessor artifacts in model/champion/

    to satisfy lifespan startup requirements during unit tests.
    """
    champion_dir = Path("model/champion")
    champion_dir.mkdir(parents=True, exist_ok=True)

    model_file = champion_dir / "model.joblib"
    prep_file = champion_dir / "preprocessor.joblib"

    # Also ensure config artifacts directory exists as a fallback
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create dummy fitted model
    clf = RandomForestClassifier(n_estimators=2, random_state=42)
    # Fit with dummy sample data matching expected input schema
    import pandas as pd

    X_dummy = pd.DataFrame({"age": [50, 60], "trestbps": [120, 140]})
    y_dummy = [0, 1]
    clf.fit(X_dummy, y_dummy)

    # 2. Save dummy artifacts
    joblib.dump(clf, model_file)
    joblib.dump(clf, artifacts_dir / "model.joblib")

    preprocessor = DataPreprocessor()
    joblib.dump(preprocessor, prep_file)
    joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")

    yield  # Run tests

    # Clean up after test module completes (optional)


@pytest.fixture(scope="module")
def client():
    """Pytest fixture to initialize TestClient within the lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Tests if the GET /health endpoint returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_metrics_endpoint(client):
    """Tests if Prometheus /metrics endpoint is exposed."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text or "process_cpu_seconds_total" in response.text


def test_predict_endpoint_valid_payload(client):
    """Tests POST /predict with a valid patient input payload."""
    payload = {
        "age": 63.0,
        "sex": 1,
        "cp": 1,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150.0,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0.0,
        "thal": 6.0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "prediction_label" in data
    assert "confidence" in data
    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in ["High Risk", "No Risk"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_endpoint_invalid_payload(client):
    """Tests POST /predict validation error handling on bad data types."""
    invalid_payload = {
        "age": "invalid_age_string",
        "sex": 1,
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422