import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.describe_data import run_data_description_pipeline
from src.data.encode import run_encoding_pipeline
from src.data.missing_values import run_missing_value_pipeline
from src.data.scaling import run_scaling_pipeline
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn Transformer wrapping modular data processing functions.

    Stores learned states (imputation maps, encodings, scaling params) during
    fit() and applies them during transform().
    """

    def __init__(self):
        # Learned state attributes (populated during fit)
        self.column_types = {}
        self.imputation_params_ = {}
        self.scaling_params_ = {}
        self.column_types_updated_ = {}
        self.encoding_scheme_ = {}
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, cfg: dict):
        """Learns preprocessing parameters strictly from the training data."""
        logger.info("=== Fitting DataPreprocessor ===")
        X_copy = X.copy()

        # Get column types
        logger.info("Describing dataset column types...")
        self.column_types = run_data_description_pipeline(cfg, X_copy)

        # Impute missing values & learn imputation state
        logger.info("Fitting missing value imputation logic...")
        X_copy, self.imputation_params_ = run_missing_value_pipeline(
            cfg, X_copy, self.column_types
        )

        # Fit Encoding logic
        logger.info("Fitting categorical encoding logic...")
        X_copy, self.column_types_updated_, self.encoding_scheme_ = run_encoding_pipeline(
            cfg, X_copy, self.column_types
        )

        # Fit Scaling logic & learn scaling parameters (mean, std)
        logger.info("Fitting numerical feature scaling logic...")
        X_copy, self.column_types_updated_, self.scaling_params_ = run_scaling_pipeline(
            cfg, X_copy, self.column_types_updated_
        )

        self.is_fitted_ = True
        logger.info("=== DataPreprocessor Successfully Fitted ===")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies learned parameters from fit() to new incoming data (test/inference)."""
        if not self.is_fitted_:
            raise RuntimeError(
                "CustomPreprocessor is not fitted yet. Call 'fit' before calling 'transform'."
            )

        logger.info("=== Transforming Data using Fitted DataPreprocessor ===")
        X_trans = X.copy()

        # Apply learned Imputation
        logger.info("Imputing missing values...")
        for col, fill_val in self.imputation_params_.items():
            if col in X_trans.columns:
                X_trans[col] = X_trans[col].fillna(fill_val)
        logger.info("Missing value imputation complete!")

        # Apply Encoding based on self.encoding_scheme_
        logger.info("Encoding categorical columns...")
        for col, encoded_cats in self.encoding_scheme_.items():
            if col in X_trans.columns:
                for cat in encoded_cats:
                    dummy_col_name = f"{col}_{cat}"
                    X_trans[dummy_col_name] = (X_trans[col] == cat).astype(int)

                X_trans = X_trans.drop(columns=[col])
        logger.info("Encoding categorical columns complete!")

        # Apply learned Scaling using pre-calculated mean and std
        logger.info("Scaling columns...")
        for col, params in self.scaling_params_.items():
            scaled_col_name = f"{col}_scaled"
            mean_val = params["mean"]
            std_val = params["std"]

            if col in X_trans.columns:
                if std_val == 0:
                    X_trans[scaled_col_name] = 0.0
                else:
                    X_trans[scaled_col_name] = (
                        X_trans[col] - mean_val
                    ) / std_val
        logger.info("Scaling columns complete!")

        logger.info("=== Data Transformation Completed ===")
        return X_trans


def run_preprocessor(cfg: dict, df: pd.DataFrame) -> tuple([pd.DataFrame, DataPreprocessor]):
    """Fits preprocessor on data, serializes preprocessor object to disk, and returns processed data."""
    logger.info("\n\n")
    logger.info("=== Running Data Preprocessor Pipeline ===")

    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(df, cfg)

    # Save fitted preprocessor object
    preprocessor_path = cfg["data"]["preprocessor_path"]
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved fitted preprocessor artifact to: '{preprocessor_path}'")

    # Optionally save fully preprocessed dataset
    preprocessed_data_path = cfg["data"].get("preprocessed_data_path")
    if preprocessed_data_path:
        df_processed.to_csv(preprocessed_data_path, index=False)
        logger.info(
            f"Saved preprocessed DataFrame CSV to: '{preprocessed_data_path}'"
        )

    logger.info("=== Finished Data Preprocessor Pipeline Execution ===")
    return df_processed, preprocessor


if __name__ == "__main__":
    logger.info("Initializing Preprocessor Pipeline Script Execution...")

    # Load config
    cfg = load_config()

    # Load raw input dataset
    input_path = cfg["data"]["train_data_path"]
    logger.info(f"Loading raw input training data from: '{input_path}'")
    df_raw = pd.read_csv(input_path)

    # Execute preprocessor pipeline
    df_transformed, preprocessor = run_preprocessor(cfg, df_raw)

    logger.info("Script execution completed successfully.")
