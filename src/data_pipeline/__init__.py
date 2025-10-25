"""
Data pipeline module for Prefect-based workflows.
Contains tasks for data ingestion, preprocessing, and transformation.
"""

from .ingestion import (
    load_csv_data,
    load_from_api,
    validate_data_schema
)

from .preprocessing import (
    clean_data,
    handle_missing_values,
    encode_categorical_features,
    scale_numerical_features
)

from .transformation import (
    feature_engineering,
    split_train_test,
    save_processed_data
)

__all__ = [
    "load_csv_data",
    "load_from_api",
    "validate_data_schema",
    "clean_data",
    "handle_missing_values",
    "encode_categorical_features",
    "scale_numerical_features",
    "feature_engineering",
    "split_train_test",
    "save_processed_data"
]