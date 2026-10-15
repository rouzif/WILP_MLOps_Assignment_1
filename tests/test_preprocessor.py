"""Verifies state persistence across .fit() and .transform() for inference readiness."""

import pandas as pd
from src.data.preprocessor import DataPreprocessor
from src.utils.config import load_config


def test_data_preprocessor_fit_transform(raw_sample_data):
    """Tests if Custom DataPreprocessor fits state and transforms raw data."""
    preprocessor = DataPreprocessor()
    cfg = load_config()
    df_transformed = preprocessor.fit_transform(raw_sample_data, cfg)

    # 1. Assert transformer state attributes are set
    assert preprocessor.is_fitted_ is True
    assert len(preprocessor.imputation_params_) > 0
    assert len(preprocessor.scaling_params_) > 0

    # 2. Assert transformed DataFrame has no NaNs
    assert df_transformed.isna().sum().sum() == 0


def test_data_preprocessor_transform_inference(raw_sample_data):
    """Tests transform on unseen inference samples using pre-learned states."""
    preprocessor = DataPreprocessor()
    cfg = load_config()
    preprocessor.fit(raw_sample_data, cfg)

    # Raw inference input without target column
    df_inference_input = raw_sample_data.drop(columns=["heart_disease_flag"]).iloc[:2].copy()

    # Transform using pre-calculated state
    df_transformed = preprocessor.transform(df_inference_input)

    assert isinstance(df_transformed, pd.DataFrame)
    assert df_transformed.isna().sum().sum() == 0
