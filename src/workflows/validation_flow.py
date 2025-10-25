"""
Validation workflows for model assessment.
"""

from prefect import flow, task
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ml_models.validation import (
    cross_validate_model, calculate_metrics, generate_classification_report,
    plot_confusion_matrix, model_comparison
)
from ml_models.training import load_model


@flow(name="model-validation-flow", log_prints=True)
def model_validation_flow(
    model_path: str,
    test_data_path: str,
    target_column: str,
    output_dir: str = "validation_results"
):
    """
    Comprehensive model validation flow.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model validation flow")
    
    try:
        # Load model and test data
        model, metadata = load_model(model_path)
        test_data = pd.read_csv(test_data_path)
        
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Determine problem type
        problem_type = metadata.get('problem_type', 'classification')
        
        # Calculate comprehensive metrics
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        
        metrics = calculate_metrics(
            y_test, y_pred, problem_type, y_proba
        )
        
        # Generate classification report (if classification)
        report_results = None
        if problem_type == "classification":
            report_results = generate_classification_report(
                y_test, y_pred,
                output_path=f"{output_dir}/classification_report.txt"
            )
            
            # Plot confusion matrix
            confusion_matrix_path = plot_confusion_matrix(
                y_test, y_pred,
                output_path=f"{output_dir}/confusion_matrix.png"
            )
        
        validation_results = {
            'metrics': metrics,
            'classification_report': report_results,
            'model_metadata': metadata,
            'validation_time': datetime.now().isoformat(),
            'test_data_shape': test_data.shape
        }
        
        logger.info("Model validation completed successfully")
        return validation_results
        
    except Exception as e:
        logger.error(f"Error in model validation: {str(e)}")
        raise


@flow(name="cross-validation-flow", log_prints=True)
def cross_validation_flow(
    model_path: str,
    training_data_path: str,
    target_column: str,
    cv_folds: int = 5
):
    """
    Cross-validation flow for model assessment.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting cross-validation flow")
    
    try:
        # Load model and training data
        model, metadata = load_model(model_path)
        training_data = pd.read_csv(training_data_path)
        
        X = training_data.drop(columns=[target_column])
        y = training_data[target_column]
        
        # Perform cross-validation
        problem_type = metadata.get('problem_type', 'classification')
        cv_results = cross_validate_model(
            model, X, y,
            cv_folds=cv_folds,
            problem_type=problem_type
        )
        
        logger.info("Cross-validation completed successfully")
        
        return {
            'cv_results': cv_results,
            'model_metadata': metadata,
            'validation_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        raise