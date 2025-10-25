"""
Data ingestion tasks using Prefect.
"""

import pandas as pd
import requests
from prefect import task
from typing import Dict, Any, Optional
import logging
import os
from pathlib import Path


@task(name="load_csv_data", retries=3, retry_delay_seconds=5)
def load_csv_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Loading CSV data from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        raise


@task(name="load_from_api", retries=3, retry_delay_seconds=10)
def load_from_api(
    url: str, 
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Load data from an API endpoint.
    
    Args:
        url: API endpoint URL
        headers: Optional headers for the request
        params: Optional query parameters
    
    Returns:
        pd.DataFrame: Data from API response
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Loading data from API: {url}")
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data)
        
        logger.info(f"Successfully loaded {len(df)} rows from API")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from API: {str(e)}")
        raise


@task(name="validate_data_schema")
def validate_data_schema(
    df: pd.DataFrame, 
    required_columns: list,
    schema_config: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Validate dataframe schema and data types.
    
    Args:
        df: Input dataframe
        required_columns: List of required column names
        schema_config: Optional mapping of column names to expected data types
    
    Returns:
        pd.DataFrame: Validated dataframe
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Validating data schema")
        
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types if schema config provided
        if schema_config:
            for column, expected_type in schema_config.items():
                if column in df.columns:
                    try:
                        df[column] = df[column].astype(expected_type)
                    except Exception as e:
                        logger.warning(f"Could not convert {column} to {expected_type}: {e}")
        
        logger.info("Data schema validation completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Schema validation failed: {str(e)}")
        raise


@task(name="save_raw_data")
def save_raw_data(df: pd.DataFrame, output_path: str) -> str:
    """
    Save raw data to specified path.
    
    Args:
        df: Dataframe to save
        output_path: Output file path
    
    Returns:
        str: Path where data was saved
    """
    try:
        logger = logging.getLogger(__name__)
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Raw data saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving raw data: {str(e)}")
        raise