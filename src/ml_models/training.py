"""
Machine learning model training tasks using Prefect.
"""

import pandas as pd
import numpy as np
from prefect import task
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from typing import Dict, Any, Optional, Tuple
import logging
import joblib
from pathlib import Path
import json


@task(name="train_sklearn_model", retries=2)
def train_sklearn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    problem_type: str = "classification",
    hyperparameters: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Train a scikit-learn model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train
        problem_type: "classification" or "regression"
        hyperparameters: Model hyperparameters
    
    Returns:
        Trained model
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Training {model_type} model for {problem_type}")
        
        if hyperparameters is None:
            hyperparameters = {}
        
        # Initialize model based on type and problem
        if problem_type == "classification":
            if model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=hyperparameters.get('n_estimators', 100),
                    max_depth=hyperparameters.get('max_depth', None),
                    random_state=hyperparameters.get('random_state', 42),
                    **{k: v for k, v in hyperparameters.items() 
                       if k not in ['n_estimators', 'max_depth', 'random_state']}
                )
            elif model_type == "logistic_regression":
                model = LogisticRegression(
                    C=hyperparameters.get('C', 1.0),
                    random_state=hyperparameters.get('random_state', 42),
                    max_iter=hyperparameters.get('max_iter', 1000),
                    **{k: v for k, v in hyperparameters.items() 
                       if k not in ['C', 'random_state', 'max_iter']}
                )
            elif model_type == "svm":
                model = SVC(
                    C=hyperparameters.get('C', 1.0),
                    kernel=hyperparameters.get('kernel', 'rbf'),
                    random_state=hyperparameters.get('random_state', 42),
                    **{k: v for k, v in hyperparameters.items() 
                       if k not in ['C', 'kernel', 'random_state']}
                )
            else:
                raise ValueError(f"Unsupported classification model type: {model_type}")
                
        elif problem_type == "regression":
            if model_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=hyperparameters.get('n_estimators', 100),
                    max_depth=hyperparameters.get('max_depth', None),
                    random_state=hyperparameters.get('random_state', 42),
                    **{k: v for k, v in hyperparameters.items() 
                       if k not in ['n_estimators', 'max_depth', 'random_state']}
                )
            elif model_type == "linear_regression":
                model = LinearRegression(
                    **hyperparameters
                )
            elif model_type == "svm":
                model = SVR(
                    C=hyperparameters.get('C', 1.0),
                    kernel=hyperparameters.get('kernel', 'rbf'),
                    **{k: v for k, v in hyperparameters.items() 
                       if k not in ['C', 'kernel']}
                )
            else:
                raise ValueError(f"Unsupported regression model type: {model_type}")
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        logger.info(f"Model training completed successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise


@task(name="evaluate_model")
def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str = "classification"
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        problem_type: "classification" or "regression"
    
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Evaluating model performance")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        metrics = {}
        
        if problem_type == "classification":
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            
            # Add probability predictions if available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                metrics['predicted_probabilities'] = y_proba.tolist()
                
                # Calculate AUC-ROC for binary classification
                if len(np.unique(y_test)) == 2:
                    from sklearn.metrics import roc_auc_score
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            
        elif problem_type == "regression":
            from sklearn.metrics import mean_absolute_error, r2_score
            
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2_score'] = r2_score(y_test, y_pred)
        
        metrics['predictions'] = y_pred.tolist()
        metrics['test_size'] = len(y_test)
        
        logger.info("Model evaluation completed")
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise


@task(name="save_model")
def save_model(
    model: Any,
    model_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model to save
        model_path: Path to save the model
        metadata: Optional metadata to save with the model
    
    Returns:
        str: Path where model was saved
    """
    try:
        logger = logging.getLogger(__name__)
        
        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        joblib.dump(model, model_path)
        
        # Save metadata if provided
        if metadata:
            metadata_path = str(Path(model_path).with_suffix('.json'))
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Model metadata saved to {metadata_path}")
        
        logger.info(f"Model saved to {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


@task(name="load_model")
def load_model(model_path: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Tuple of (loaded model, metadata if available)
    """
    try:
        logger = logging.getLogger(__name__)
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model
        model = joblib.load(model_path)
        
        # Try to load metadata
        metadata = None
        metadata_path = str(Path(model_path).with_suffix('.json'))
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Model metadata loaded from {metadata_path}")
        
        logger.info(f"Model loaded from {model_path}")
        return model, metadata
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


@task(name="hyperparameter_tuning")
def hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    problem_type: str,
    param_grid: Dict[str, list],
    cv_folds: int = 5,
    scoring: Optional[str] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to tune
        problem_type: "classification" or "regression"
        param_grid: Grid of parameters to search
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric for optimization
    
    Returns:
        Tuple of (best model, tuning results)
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Starting hyperparameter tuning for {model_type}")
        
        from sklearn.model_selection import GridSearchCV
        
        # Get base model
        base_model = train_sklearn_model(
            X_train, y_train, model_type, problem_type, {}
        )
        
        # Set default scoring if not provided
        if scoring is None:
            scoring = 'accuracy' if problem_type == 'classification' else 'neg_mean_squared_error'
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Hyperparameter tuning completed. Best score: {grid_search.best_score_}")
        return grid_search.best_estimator_, tuning_results
        
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        raise