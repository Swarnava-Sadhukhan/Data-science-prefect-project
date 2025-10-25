"""
Model prediction tasks using Prefect.
"""

import pandas as pd
import numpy as np
from prefect import task
from typing import Dict, Any, Optional, List, Union
import logging
import joblib
from pathlib import Path


@task(name="make_predictions")
def make_predictions(
    model: Any,
    X_data: pd.DataFrame,
    return_probabilities: bool = False
) -> Dict[str, Any]:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model
        X_data: Features for prediction
        return_probabilities: Whether to return prediction probabilities (if available)
    
    Returns:
        Dictionary containing predictions and optionally probabilities
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Making predictions for {len(X_data)} samples")
        
        # Make predictions
        predictions = model.predict(X_data)
        
        results = {
            'predictions': predictions.tolist(),
            'num_predictions': len(predictions)
        }
        
        # Add probabilities if requested and available
        if return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_data)
            results['probabilities'] = probabilities.tolist()
            results['classes'] = model.classes_.tolist() if hasattr(model, 'classes_') else None
        
        logger.info("Predictions completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise


@task(name="batch_predict")
def batch_predict(
    model_path: str,
    data_path: str,
    output_path: str,
    batch_size: int = 1000,
    return_probabilities: bool = False
) -> str:
    """
    Make predictions in batches for large datasets.
    
    Args:
        model_path: Path to the saved model
        data_path: Path to the data file
        output_path: Path to save predictions
        batch_size: Size of each batch
        return_probabilities: Whether to include probabilities
    
    Returns:
        str: Path where predictions were saved
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Starting batch prediction with batch size {batch_size}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load data
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data with {len(data)} rows")
        
        predictions_list = []
        probabilities_list = []
        
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch_end = min(i + batch_size, len(data))
            batch_data = data.iloc[i:batch_end]
            
            # Make predictions for batch
            batch_predictions = model.predict(batch_data)
            predictions_list.extend(batch_predictions)
            
            if return_probabilities and hasattr(model, 'predict_proba'):
                batch_probabilities = model.predict_proba(batch_data)
                probabilities_list.extend(batch_probabilities)
            
            logger.info(f"Processed batch {i//batch_size + 1}, rows {i}-{batch_end}")
        
        # Create results dataframe
        results_df = pd.DataFrame({'predictions': predictions_list})
        
        if probabilities_list:
            # Add probability columns
            prob_array = np.array(probabilities_list)
            classes = model.classes_ if hasattr(model, 'classes_') else range(prob_array.shape[1])
            
            for idx, class_name in enumerate(classes):
                results_df[f'prob_class_{class_name}'] = prob_array[:, idx]
        
        # Save results
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Batch predictions saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise


@task(name="model_inference")
def model_inference(
    model_path: str,
    input_data: Union[Dict[str, Any], pd.DataFrame],
    preprocess_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform model inference with optional preprocessing.
    
    Args:
        model_path: Path to the saved model
        input_data: Input data for inference (dict or DataFrame)
        preprocess_config: Optional preprocessing configuration
    
    Returns:
        Dictionary containing inference results
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Starting model inference")
        
        # Load model
        model = joblib.load(model_path)
        
        # Convert input to DataFrame if needed
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Apply preprocessing if config provided
        if preprocess_config:
            df = apply_preprocessing(df, preprocess_config)
        
        # Make predictions
        predictions = model.predict(df)
        
        results = {
            'predictions': predictions.tolist(),
            'input_shape': df.shape,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Add probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)
            results['probabilities'] = probabilities.tolist()
            
            if hasattr(model, 'classes_'):
                results['classes'] = model.classes_.tolist()
        
        logger.info("Model inference completed")
        return results
        
    except Exception as e:
        logger.error(f"Error in model inference: {str(e)}")
        raise


@task(name="apply_preprocessing")
def apply_preprocessing(
    df: pd.DataFrame,
    preprocess_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply preprocessing steps to input data.
    
    Args:
        df: Input dataframe
        preprocess_config: Preprocessing configuration
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Applying preprocessing steps")
        
        processed_df = df.copy()
        
        # Load and apply saved preprocessors
        if 'preprocessors_path' in preprocess_config:
            preprocessors_path = preprocess_config['preprocessors_path']
            
            # Apply scaler if available
            scaler_path = Path(preprocessors_path) / 'scaler_preprocessor.joblib'
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    processed_df[numeric_cols] = scaler.transform(processed_df[numeric_cols])
                    logger.info("Applied scaling transformation")
            
            # Apply encoders if available
            for col in processed_df.select_dtypes(include=['object']).columns:
                encoder_path = Path(preprocessors_path) / f'{col}_encoder.joblib'
                if encoder_path.exists():
                    encoder = joblib.load(encoder_path)
                    processed_df[col] = encoder.transform(processed_df[col])
                    logger.info(f"Applied encoding transformation to {col}")
        
        # Apply feature engineering if specified
        if 'feature_engineering' in preprocess_config:
            processed_df = apply_feature_engineering(processed_df, preprocess_config['feature_engineering'])
        
        logger.info("Preprocessing completed")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error applying preprocessing: {str(e)}")
        raise


@task(name="apply_feature_engineering")
def apply_feature_engineering(
    df: pd.DataFrame,
    feature_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply feature engineering transformations during inference.
    
    Args:
        df: Input dataframe
        feature_config: Feature engineering configuration
    
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    try:
        logger = logging.getLogger(__name__)
        processed_df = df.copy()
        
        # Apply same feature engineering as training
        # (This should match the feature_engineering task in transformation.py)
        
        # Polynomial features
        for column in feature_config.get('polynomial_features', []):
            if column in processed_df.columns:
                processed_df[f"{column}_squared"] = processed_df[column] ** 2
                processed_df[f"{column}_cubed"] = processed_df[column] ** 3
        
        # Interaction features
        for feature_pair in feature_config.get('interaction_features', []):
            if len(feature_pair) == 2 and all(col in processed_df.columns for col in feature_pair):
                col1, col2 = feature_pair
                processed_df[f"{col1}_{col2}_interaction"] = processed_df[col1] * processed_df[col2]
        
        # Date features
        for column in feature_config.get('date_features', []):
            if column in processed_df.columns:
                try:
                    processed_df[column] = pd.to_datetime(processed_df[column])
                    processed_df[f"{column}_year"] = processed_df[column].dt.year
                    processed_df[f"{column}_month"] = processed_df[column].dt.month
                    processed_df[f"{column}_day"] = processed_df[column].dt.day
                    processed_df[f"{column}_dayofweek"] = processed_df[column].dt.dayofweek
                    processed_df[f"{column}_quarter"] = processed_df[column].dt.quarter
                except Exception as e:
                    logger.warning(f"Could not create date features for {column}: {e}")
        
        # Log transformation
        for column in feature_config.get('log_transform', []):
            if column in processed_df.columns:
                processed_df[f"{column}_log"] = np.log1p(processed_df[column])
        
        logger.info("Feature engineering applied")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error applying feature engineering: {str(e)}")
        raise


@task(name="save_predictions")
def save_predictions(
    predictions: Dict[str, Any],
    output_path: str,
    include_metadata: bool = True
) -> str:
    """
    Save predictions to a file.
    
    Args:
        predictions: Predictions dictionary
        output_path: Path to save predictions
        include_metadata: Whether to include metadata
    
    Returns:
        str: Path where predictions were saved
    """
    try:
        logger = logging.getLogger(__name__)
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame from predictions
        df = pd.DataFrame({'predictions': predictions['predictions']})
        
        # Add probabilities if available
        if 'probabilities' in predictions and predictions['probabilities']:
            prob_array = np.array(predictions['probabilities'])
            classes = predictions.get('classes', range(prob_array.shape[1]))
            
            for idx, class_name in enumerate(classes):
                df[f'prob_class_{class_name}'] = prob_array[:, idx]
        
        # Add metadata if requested
        if include_metadata:
            df['prediction_timestamp'] = pd.Timestamp.now()
            if 'num_predictions' in predictions:
                df['total_predictions'] = predictions['num_predictions']
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        logger.info(f"Predictions saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        raise