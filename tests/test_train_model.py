"""Tests model training with GridSearchCV, model instantiation, probability predictions, and metric calculation."""

import pandas as pd
from sklearn.base import ClassifierMixin
from src.model.evaluation import evaluate
from src.model.predict import get_predictions
from src.model.train_model import run_model_training


def test_run_model_training_rf(processed_sample_data):
    """Tests Random Forest cross-validation and fitting."""
    feature_list = ["age_scaled", "chol_scaled", "cp_1", "cp_2"]
    target_var = "heart_disease_flag"

    model, _ = run_model_training(
        df=processed_sample_data,
        feature_list=feature_list,
        target_var=target_var,
        model_algo="rf",
    )

    assert isinstance(model, ClassifierMixin)
    assert hasattr(model, "predict_proba")


def test_get_predictions(processed_sample_data):
    """Tests prediction output structure and probability ranges."""
    feature_list = ["age_scaled", "chol_scaled", "cp_1", "cp_2"]
    target_var = "heart_disease_flag"

    model, _ = run_model_training(
        df=processed_sample_data,
        feature_list=feature_list,
        target_var=target_var,
        model_algo="rf",
    )

    df_preds = get_predictions(df=processed_sample_data, model=model)

    # 1. Assert predicted_probability column is present
    assert "predicted_probability" in df_preds.columns

    # 2. Assert probabilities are between 0.0 and 1.0
    probs = df_preds["predicted_probability"]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_evaluate():
    """Tests evaluation metrics calculation dictionary output."""
    y_true = pd.Series([0, 1, 1, 0, 1])
    y_prob = pd.Series([0.1, 0.8, 0.9, 0.3, 0.7])

    metrics = evaluate(y_true, y_prob)

    # Assert expected metric keys exist and are numeric floats
    for metric_name in ["accuracy", "precision", "recall", "roc_auc"]:
        assert metric_name in metrics
        assert isinstance(metrics[metric_name], float)