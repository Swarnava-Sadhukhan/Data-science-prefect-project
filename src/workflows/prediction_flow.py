"""
Prediction workflows for the data science application.
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

from ml_models.prediction import make_predictions, batch_predict, model_inference
from ml_models.training import load_model


@flow(name="prediction-flow", log_prints=True)
def prediction_flow(
    model_path: str,
    input_data_path: str,
    output_path: str = None
):
    """
    Flow for making predictions on new data.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting prediction flow")
    
    try:
        # Load model
        model, metadata = load_model(model_path)
        
        # Load input data
        input_data = pd.read_csv(input_data_path)
        
        # Make predictions
        predictions_result = make_predictions(
            model,
            input_data,
            return_probabilities=True
        )
        
        # Save predictions if output path provided
        if output_path:
            predictions_df = pd.DataFrame({
                'predictions': predictions_result['predictions']
            })
            
            if 'probabilities' in predictions_result:
                prob_array = np.array(predictions_result['probabilities'])
                for i in range(prob_array.shape[1]):
                    predictions_df[f'prob_class_{i}'] = prob_array[:, i]
            
            predictions_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        
        return {
            'predictions': predictions_result,
            'model_metadata': metadata,
            'input_shape': input_data.shape,
            'prediction_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in prediction flow: {str(e)}")
        raise


@flow(name="batch-prediction-flow", log_prints=True)
def batch_prediction_flow(
    model_path: str,
    data_path: str,
    output_path: str,
    batch_size: int = 1000
):
    """
    Flow for batch predictions on large datasets.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting batch prediction flow")
    
    try:
        # Perform batch prediction
        result_path = batch_predict(
            model_path=model_path,
            data_path=data_path,
            output_path=output_path,
            batch_size=batch_size,
            return_probabilities=True
        )
        
        logger.info(f"Batch predictions completed and saved to {result_path}")
        
        return {
            'output_path': result_path,
            'batch_size': batch_size,
            'completion_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction flow: {str(e)}")
        raise