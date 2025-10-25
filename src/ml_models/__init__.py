"""
Machine learning models module for Prefect workflows.
"""

from .training import (
    train_sklearn_model,
    evaluate_model,
    save_model,
    load_model
)

from .prediction import (
    make_predictions,
    batch_predict,
    model_inference
)

from .validation import (
    cross_validate_model,
    calculate_metrics,
    generate_classification_report,
    plot_confusion_matrix
)

__all__ = [
    "train_sklearn_model",
    "evaluate_model",
    "save_model",
    "load_model",
    "make_predictions",
    "batch_predict",
    "model_inference",
    "cross_validate_model",
    "calculate_metrics",
    "generate_classification_report",
    "plot_confusion_matrix"
]