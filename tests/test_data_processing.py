"""Tests column categorization, target column exclusion, missing value imputation, one-hot encoding, and data
partitioning."""

import pandas as pd
from src.data.describe_data import describe_data
from src.data.encode import encode_data
from src.data.missing_values import impute_missing_values
from src.data.partition import partition_data


def test_describe_data_target_exclusion(raw_sample_data):
    """Tests that describe_data categorizes columns and excludes the target column."""
    target_var = "heart_disease_flag"
    col_categories = describe_data(raw_sample_data, target_var=target_var)

    # 1. Assert categories keys exist
    for key in ["numeric-low-cardinality", "numeric-distributed", "categorical", "text"]:
        assert key in col_categories

    # 2. Assert target variable is excluded from all feature categories
    all_categorized_cols = [
        col for cols in col_categories.values() for col in cols
    ]
    assert target_var not in all_categorized_cols


def test_impute_missing_values(raw_sample_data, column_types_dict):
    """Tests missing value imputation and state parameter dictionary output."""
    df_imputed, imputation_params = impute_missing_values(raw_sample_data, column_types_dict)

    # Assert no NaNs remain in imputed dataset
    assert df_imputed.isna().sum().sum() == 0

    # Assert imputation params were computed for column with NaNs ('chol')
    assert "chol" in imputation_params


def test_encode_data(raw_sample_data, column_types_dict):
    """Tests encoding logic, dropped category rule, and encoding scheme generation."""
    df_imputed, _ = impute_missing_values(raw_sample_data, column_types_dict)
    df_encoded, col_types_updated, enc_scheme = encode_data(df_imputed, column_types_dict)

    # 1. Assert original categorical column ('cp') is replaced
    assert "cp" not in df_encoded.columns

    # 2. Assert 'encoded' key is added to column_types
    assert "encoded" in col_types_updated

    # 3. Assert encoding_scheme dictionary mapping exists
    assert "cp" in enc_scheme


def test_partition_data(raw_sample_data):
    """Tests stratified data partitioning split ratio."""
    target_var = "heart_disease_flag"
    df_train, df_test = partition_data(
        df=raw_sample_data, test_ratio=0.4, strata=raw_sample_data[target_var]
    )

    # Assert train/test row counts match 60/40 split on 5 rows
    assert len(df_train) == 9
    assert len(df_test) == 6
