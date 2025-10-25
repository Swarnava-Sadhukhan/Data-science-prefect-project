"""
Model validation tasks using Prefect.
"""

import pandas as pd
import numpy as np
from prefect import task
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path


@task(name="cross_validate_model")
def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    scoring: Optional[List[str]] = None,
    problem_type: str = "classification"
) -> Dict[str, Any]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model to validate
        X: Features
        y: Target
        cv_folds: Number of cross-validation folds
        scoring: List of scoring metrics
        problem_type: "classification" or "regression"
    
    Returns:
        Dictionary containing cross-validation results
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        if scoring is None:
            if problem_type == "classification":
                scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            else:
                scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Process results
        results = {
            'cv_folds': cv_folds,
            'scoring_metrics': scoring
        }
        
        for metric in scoring:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results[f'{metric}_test_mean'] = np.mean(test_scores)
            results[f'{metric}_test_std'] = np.std(test_scores)
            results[f'{metric}_train_mean'] = np.mean(train_scores)
            results[f'{metric}_train_std'] = np.std(train_scores)
            results[f'{metric}_test_scores'] = test_scores.tolist()
            results[f'{metric}_train_scores'] = train_scores.tolist()
        
        # Calculate fit times
        results['fit_time_mean'] = np.mean(cv_results['fit_time'])
        results['fit_time_std'] = np.std(cv_results['fit_time'])
        results['score_time_mean'] = np.mean(cv_results['score_time'])
        results['score_time_std'] = np.std(cv_results['score_time'])
        
        logger.info("Cross-validation completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        raise


@task(name="calculate_metrics")
def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    problem_type: str = "classification",
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        problem_type: "classification" or "regression"
        y_proba: Predicted probabilities (for classification)
    
    Returns:
        Dictionary containing calculated metrics
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Calculating evaluation metrics")
        
        metrics = {}
        
        if problem_type == "classification":
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            # Weighted averages
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Classification report
            metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
            
            # ROC AUC for binary classification
            if len(np.unique(y_true)) == 2 and y_proba is not None:
                if y_proba.ndim == 2:
                    y_proba_binary = y_proba[:, 1]
                else:
                    y_proba_binary = y_proba
                
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_binary)
                
                # ROC curve data
                fpr, tpr, thresholds = roc_curve(y_true, y_proba_binary)
                metrics['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
            
        elif problem_type == "regression":
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2_score'] = r2_score(y_true, y_pred)
            
            # Additional regression metrics
            metrics['mean_absolute_percentage_error'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)
            
            # Residuals statistics
            residuals = y_true - y_pred
            metrics['residuals_mean'] = np.mean(residuals)
            metrics['residuals_std'] = np.std(residuals)
            metrics['residuals_min'] = np.min(residuals)
            metrics['residuals_max'] = np.max(residuals)
        
        metrics['sample_size'] = len(y_true)
        metrics['unique_labels'] = len(np.unique(y_true))
        
        logger.info("Metrics calculation completed")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise


@task(name="generate_classification_report")
def generate_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        output_path: Path to save the report
    
    Returns:
        Dictionary containing the classification report
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Generating classification report")
        
        # Get classification report as dictionary
        report_dict = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Create a formatted text report
        report_text = classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0
        )
        
        results = {
            'report_dict': report_dict,
            'report_text': report_text,
            'accuracy': report_dict['accuracy'],
            'macro_avg': report_dict['macro avg'],
            'weighted_avg': report_dict['weighted avg']
        }
        
        # Save report if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Classification report saved to {output_path}")
            results['report_path'] = output_path
        
        logger.info("Classification report generated successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error generating classification report: {str(e)}")
        raise


@task(name="plot_confusion_matrix")
def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> str:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        output_path: Path to save the plot
        figsize: Figure size
    
    Returns:
        str: Path where plot was saved
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Plotting confusion matrix")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names or range(cm.shape[1]),
            yticklabels=class_names or range(cm.shape[0])
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            output_path = 'confusion_matrix.png'
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        raise


@task(name="plot_learning_curves")
def plot_learning_curves(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Optional[str] = None,
    cv_folds: int = 5,
    train_sizes: Optional[np.ndarray] = None,
    scoring: str = 'accuracy'
) -> str:
    """
    Plot learning curves to analyze model performance vs training set size.
    
    Args:
        model: Model to analyze
        X: Features
        y: Target
        output_path: Path to save the plot
        cv_folds: Number of cross-validation folds
        train_sizes: Training set sizes to evaluate
        scoring: Scoring metric
    
    Returns:
        str: Path where plot was saved
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Generating learning curves")
        
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Generate learning curves
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel(f'{scoring.title()} Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            output_path = 'learning_curves.png'
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Learning curves plot saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error plotting learning curves: {str(e)}")
        raise


@task(name="model_comparison")
def model_comparison(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str = "classification"
) -> Dict[str, Any]:
    """
    Compare multiple models on the same test set.
    
    Args:
        models: Dictionary of model names to model objects
        X_test: Test features
        y_test: Test target
        problem_type: "classification" or "regression"
    
    Returns:
        Dictionary containing comparison results
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Comparing {len(models)} models")
        
        comparison_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics based on problem type
            if problem_type == "classification":
                results = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                # Add ROC AUC for binary classification
                if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    results['roc_auc'] = roc_auc_score(y_test, y_proba)
                    
            else:  # regression
                results = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2_score': r2_score(y_test, y_pred)
                }
            
            comparison_results[model_name] = results
        
        # Find best model for each metric
        best_models = {}
        if comparison_results:
            metrics = list(next(iter(comparison_results.values())).keys())
            
            for metric in metrics:
                if problem_type == "classification":
                    # Higher is better for most classification metrics
                    if metric in ['mse', 'rmse', 'mae']:
                        best_model = min(comparison_results.items(), key=lambda x: x[1][metric])
                    else:
                        best_model = max(comparison_results.items(), key=lambda x: x[1][metric])
                else:
                    # For regression, lower is better for error metrics, higher for R²
                    if metric in ['mse', 'rmse', 'mae']:
                        best_model = min(comparison_results.items(), key=lambda x: x[1][metric])
                    else:
                        best_model = max(comparison_results.items(), key=lambda x: x[1][metric])
                
                best_models[metric] = {
                    'model_name': best_model[0],
                    'score': best_model[1][metric]
                }
        
        results = {
            'model_results': comparison_results,
            'best_models': best_models,
            'num_models_compared': len(models)
        }
        
        logger.info("Model comparison completed")
        return results
        
    except Exception as e:
        logger.error(f"Error in model comparison: {str(e)}")
        raise