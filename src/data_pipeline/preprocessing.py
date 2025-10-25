"""
Data preprocessing tasks using Prefect.
"""

import pandas as pd
import numpy as np
from prefect import task
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Dict, List, Optional, Tuple, Any
import logging
import joblib
from pathlib import Path


@task(name="preprocess_telco_dataset")
def preprocess_telco_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Special preprocessing for Telco Customer Churn dataset.
    
    Args:
        df: Input dataframe
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Applying Telco dataset specific preprocessing")
        
        processed_df = df.copy()
        
        # Handle TotalCharges column - convert empty strings to NaN and then to numeric
        if 'TotalCharges' in processed_df.columns:
            logger.info("Processing TotalCharges column")
            # Replace empty strings with NaN
            processed_df['TotalCharges'] = processed_df['TotalCharges'].replace(' ', np.nan)
            # Convert to numeric
            processed_df['TotalCharges'] = pd.to_numeric(processed_df['TotalCharges'], errors='coerce')
            empty_charges_count = processed_df['TotalCharges'].isna().sum()
            logger.info(f"Found {empty_charges_count} empty TotalCharges values, converted to NaN")
        
        # Convert SeniorCitizen to categorical for consistency
        if 'SeniorCitizen' in processed_df.columns:
            processed_df['SeniorCitizen'] = processed_df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
            logger.info("Converted SeniorCitizen to categorical (Yes/No)")
        
        # Convert target variable to binary (for ML algorithms)
        if 'Churn' in processed_df.columns:
            processed_df['Churn_binary'] = processed_df['Churn'].map({'Yes': 1, 'No': 0})
            logger.info("Created binary Churn_binary column (1=Yes, 0=No)")
        
        # Log basic statistics
        logger.info(f"Dataset shape after preprocessing: {processed_df.shape}")
        if 'Churn' in processed_df.columns:
            churn_dist = processed_df['Churn'].value_counts()
            logger.info(f"Churn distribution: {churn_dist.to_dict()}")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error in Telco dataset preprocessing: {str(e)}")
        raise


@task(name="clean_data")
def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    drop_columns: Optional[List[str]] = None,
    filter_conditions: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Clean the input dataframe by removing duplicates, dropping columns, and applying filters.
    
    Args:
        df: Input dataframe
        drop_duplicates: Whether to drop duplicate rows
        drop_columns: List of columns to drop
        filter_conditions: Dictionary of column names and conditions to filter by
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Starting data cleaning process")
        
        cleaned_df = df.copy()
        original_shape = cleaned_df.shape
        
        # Drop duplicates
        if drop_duplicates:
            cleaned_df = cleaned_df.drop_duplicates()
            logger.info(f"Dropped {original_shape[0] - cleaned_df.shape[0]} duplicate rows")
        
        # Drop specified columns
        if drop_columns:
            existing_cols = [col for col in drop_columns if col in cleaned_df.columns]
            cleaned_df = cleaned_df.drop(columns=existing_cols)
            logger.info(f"Dropped columns: {existing_cols}")
        
        # Apply filter conditions
        if filter_conditions:
            for column, condition in filter_conditions.items():
                if column in cleaned_df.columns:
                    if isinstance(condition, dict):
                        if 'min' in condition:
                            cleaned_df = cleaned_df[cleaned_df[column] >= condition['min']]
                        if 'max' in condition:
                            cleaned_df = cleaned_df[cleaned_df[column] <= condition['max']]
                        if 'values' in condition:
                            cleaned_df = cleaned_df[cleaned_df[column].isin(condition['values'])]
        
        logger.info(f"Data cleaning completed. Shape: {original_shape} -> {cleaned_df.shape}")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error in data cleaning: {str(e)}")
        raise


@task(name="handle_missing_values")
def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "simple",
    numeric_strategy: str = "mean",
    categorical_strategy: str = "most_frequent",
    threshold: float = 0.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle missing values in the dataframe.
    
    Args:
        df: Input dataframe
        strategy: Imputation strategy ("simple", "knn", "drop")
        numeric_strategy: Strategy for numeric columns ("mean", "median", "constant")
        categorical_strategy: Strategy for categorical columns ("most_frequent", "constant")
        threshold: Threshold for dropping columns with too many missing values
    
    Returns:
        Tuple of (processed dataframe, imputer objects)
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Handling missing values")
        
        processed_df = df.copy()
        imputers = {}
        
        # Drop columns with too many missing values
        missing_ratios = processed_df.isnull().sum() / len(processed_df)
        cols_to_drop = missing_ratios[missing_ratios > threshold].index.tolist()
        if cols_to_drop:
            processed_df = processed_df.drop(columns=cols_to_drop)
            logger.info(f"Dropped columns with >={threshold*100}% missing values: {cols_to_drop}")
        
        # Separate numeric and categorical columns
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
        
        if strategy == "simple":
            # Handle numeric columns
            if numeric_cols:
                numeric_imputer = SimpleImputer(strategy=numeric_strategy)
                processed_df[numeric_cols] = numeric_imputer.fit_transform(processed_df[numeric_cols])
                imputers['numeric'] = numeric_imputer
            
            # Handle categorical columns
            if categorical_cols:
                categorical_imputer = SimpleImputer(strategy=categorical_strategy)
                processed_df[categorical_cols] = categorical_imputer.fit_transform(processed_df[categorical_cols])
                imputers['categorical'] = categorical_imputer
                
        elif strategy == "knn":
            # Use KNN imputer for all columns
            knn_imputer = KNNImputer(n_neighbors=5)
            # Only apply to numeric columns for KNN
            if numeric_cols:
                processed_df[numeric_cols] = knn_imputer.fit_transform(processed_df[numeric_cols])
                imputers['knn'] = knn_imputer
            
            # Still use simple imputer for categorical
            if categorical_cols:
                categorical_imputer = SimpleImputer(strategy=categorical_strategy)
                processed_df[categorical_cols] = categorical_imputer.fit_transform(processed_df[categorical_cols])
                imputers['categorical'] = categorical_imputer
                
        elif strategy == "drop":
            # Drop rows with any missing values
            processed_df = processed_df.dropna()
            logger.info(f"Dropped rows with missing values. Remaining rows: {len(processed_df)}")
        
        logger.info("Missing value handling completed")
        return processed_df, imputers
        
    except Exception as e:
        logger.error(f"Error handling missing values: {str(e)}")
        raise


@task(name="encode_categorical_features")
def encode_categorical_features(
    df: pd.DataFrame,
    encoding_strategy: str = "onehot",
    categorical_columns: Optional[List[str]] = None,
    max_categories: int = 10
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical features.
    
    Args:
        df: Input dataframe
        encoding_strategy: Encoding method ("onehot", "label", "target")
        categorical_columns: Specific columns to encode (if None, auto-detect)
        max_categories: Maximum number of categories for one-hot encoding
    
    Returns:
        Tuple of (encoded dataframe, encoder objects)
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Encoding categorical features")
        
        processed_df = df.copy()
        encoders = {}
        
        # Auto-detect categorical columns if not specified
        if categorical_columns is None:
            categorical_columns = processed_df.select_dtypes(include=['object']).columns.tolist()
        
        for column in categorical_columns:
            if column not in processed_df.columns:
                continue
                
            unique_values = processed_df[column].nunique()
            
            if encoding_strategy == "onehot" and unique_values <= max_categories:
                # One-hot encoding for low cardinality
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_values = encoder.fit_transform(processed_df[[column]])
                
                # Create column names for one-hot encoded features
                feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_values, columns=feature_names, index=processed_df.index)
                
                # Drop original column and add encoded columns
                processed_df = processed_df.drop(columns=[column])
                processed_df = pd.concat([processed_df, encoded_df], axis=1)
                
                encoders[column] = encoder
                
            elif encoding_strategy == "label" or unique_values > max_categories:
                # Label encoding for high cardinality or when specified
                encoder = LabelEncoder()
                processed_df[column] = encoder.fit_transform(processed_df[column].astype(str))
                encoders[column] = encoder
        
        logger.info(f"Categorical encoding completed. Final shape: {processed_df.shape}")
        return processed_df, encoders
        
    except Exception as e:
        logger.error(f"Error encoding categorical features: {str(e)}")
        raise


@task(name="scale_numerical_features")
def scale_numerical_features(
    df: pd.DataFrame,
    scaling_method: str = "standard",
    columns_to_scale: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Any]:
    """
    Scale numerical features.
    
    Args:
        df: Input dataframe
        scaling_method: Scaling method ("standard", "minmax", "robust")
        columns_to_scale: Specific columns to scale (if None, scale all numeric)
    
    Returns:
        Tuple of (scaled dataframe, scaler object)
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Scaling numerical features")
        
        processed_df = df.copy()
        
        # Auto-detect numeric columns if not specified
        if columns_to_scale is None:
            columns_to_scale = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter columns that actually exist in the dataframe
        columns_to_scale = [col for col in columns_to_scale if col in processed_df.columns]
        
        if not columns_to_scale:
            logger.warning("No numeric columns found to scale")
            return processed_df, None
        
        # Initialize scaler based on method
        if scaling_method == "standard":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif scaling_method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        # Fit and transform the specified columns
        processed_df[columns_to_scale] = scaler.fit_transform(processed_df[columns_to_scale])
        
        logger.info(f"Scaling completed for columns: {columns_to_scale}")
        return processed_df, scaler
        
    except Exception as e:
        logger.error(f"Error scaling numerical features: {str(e)}")
        raise


@task(name="save_preprocessors")
def save_preprocessors(
    preprocessors: Dict[str, Any],
    output_dir: str
) -> str:
    """
    Save preprocessing objects (imputers, encoders, scalers) to disk.
    
    Args:
        preprocessors: Dictionary of preprocessing objects
        output_dir: Directory to save objects
    
    Returns:
        str: Path where objects were saved
    """
    try:
        logger = logging.getLogger(__name__)
        
        # Create directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for name, preprocessor in preprocessors.items():
            if preprocessor is not None:
                file_path = Path(output_dir) / f"{name}_preprocessor.joblib"
                joblib.dump(preprocessor, file_path)
                logger.info(f"Saved {name} preprocessor to {file_path}")
        
        logger.info(f"All preprocessors saved to {output_dir}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Error saving preprocessors: {str(e)}")
        raise


# ===================== ENHANCED PREPROCESSING FUNCTIONS =====================

@task(name="telco_feature_engineering")
def telco_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Telco domain-specific feature engineering to improve model performance.
    
    Args:
        df: Input dataframe
    
    Returns:
        pd.DataFrame: Feature-engineered dataframe
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Starting Telco-specific feature engineering")
        
        processed_df = df.copy()
        
        # 1. Service bundling features - count total services
        service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        processed_df['total_services'] = 0
        for col in service_cols:
            if col in processed_df.columns:
                # Count services that are not 'No' or 'No internet service' or 'No phone service'
                service_mask = ~processed_df[col].isin(['No', 'No internet service', 'No phone service'])
                processed_df['total_services'] += service_mask.astype(int)
        
        logger.info(f"Created total_services feature (mean: {processed_df['total_services'].mean():.2f})")
        
        # 2. Customer Lifetime Value (CLV) features
        if 'MonthlyCharges' in processed_df.columns and 'tenure' in processed_df.columns:
            # Average monthly charges (handling division by zero)
            processed_df['AvgMonthlyCharges'] = processed_df['TotalCharges'] / (processed_df['tenure'] + 1)
            
            # Charges per service (avoid division by zero)
            processed_df['ChargesPerService'] = processed_df['MonthlyCharges'] / (processed_df['total_services'] + 1)
            
            logger.info("Created CLV features: AvgMonthlyCharges, ChargesPerService")
        
        # 3. Contract risk features
        if 'Contract' in processed_df.columns:
            processed_df['is_month_to_month'] = (processed_df['Contract'] == 'Month-to-month').astype(int)
            processed_df['has_long_contract'] = (processed_df['Contract'].isin(['One year', 'Two year'])).astype(int)
            logger.info("Created contract risk features")
        
        # 4. Payment risk features
        if 'PaymentMethod' in processed_df.columns:
            processed_df['risky_payment'] = (processed_df['PaymentMethod'] == 'Electronic check').astype(int)
            processed_df['auto_payment'] = (processed_df['PaymentMethod'].str.contains('automatic', case=False, na=False)).astype(int)
            logger.info("Created payment risk features")
        
        # 5. Customer lifecycle features
        if 'tenure' in processed_df.columns:
            # Tenure groups
            processed_df['tenure_group_new'] = (processed_df['tenure'] <= 12).astype(int)
            processed_df['tenure_group_growing'] = ((processed_df['tenure'] > 12) & (processed_df['tenure'] <= 24)).astype(int)
            processed_df['tenure_group_mature'] = ((processed_df['tenure'] > 24) & (processed_df['tenure'] <= 48)).astype(int)
            processed_df['tenure_group_loyal'] = (processed_df['tenure'] > 48).astype(int)
            
            processed_df['is_new_customer'] = (processed_df['tenure'] <= 6).astype(int)
            processed_df['is_loyal_customer'] = (processed_df['tenure'] >= 48).astype(int)
            logger.info("Created customer lifecycle features")
        
        # 6. Internet service quality features
        if 'InternetService' in processed_df.columns:
            processed_df['has_fiber'] = (processed_df['InternetService'] == 'Fiber optic').astype(int)
            processed_df['has_internet'] = (processed_df['InternetService'] != 'No').astype(int)
            processed_df['has_dsl'] = (processed_df['InternetService'] == 'DSL').astype(int)
            logger.info("Created internet service features")
        
        # 7. Family status features
        if 'Partner' in processed_df.columns and 'Dependents' in processed_df.columns:
            processed_df['has_family'] = ((processed_df['Partner'] == 'Yes') | (processed_df['Dependents'] == 'Yes')).astype(int)
            processed_df['family_size'] = (processed_df['Partner'] == 'Yes').astype(int) + (processed_df['Dependents'] == 'Yes').astype(int)
            logger.info("Created family status features")
        
        # 8. Premium service features
        premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        processed_df['premium_services_count'] = 0
        for col in premium_services:
            if col in processed_df.columns:
                processed_df['premium_services_count'] += (processed_df[col] == 'Yes').astype(int)
        
        logger.info(f"Feature engineering completed. New shape: {processed_df.shape}")
        logger.info(f"Added {processed_df.shape[1] - df.shape[1]} new features")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error in Telco feature engineering: {str(e)}")
        raise


@task(name="advanced_categorical_encoding") 
def advanced_categorical_encoding(
    df: pd.DataFrame, 
    target_col: str = 'Churn_binary'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply advanced categorical encoding techniques including target encoding and frequency encoding.
    
    Args:
        df: Input dataframe
        target_col: Target column for target encoding
    
    Returns:
        Tuple[pd.DataFrame, Dict]: Encoded dataframe and encoders dictionary
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Starting advanced categorical encoding")
        
        processed_df = df.copy()
        encoders = {}
        
        # Target encoding for high-cardinality features (only if target exists)
        high_cardinality_cols = ['Contract', 'PaymentMethod', 'InternetService']
        
        if target_col in processed_df.columns:
            for col in high_cardinality_cols:
                if col in processed_df.columns:
                    # Calculate target mean for each category with smoothing
                    global_mean = processed_df[target_col].mean()
                    category_stats = processed_df.groupby(col)[target_col].agg(['mean', 'count'])
                    
                    # Add smoothing to prevent overfitting (minimum 10 samples for reliability)
                    min_samples = 10
                    smoothing = 1.0
                    
                    category_stats['target_encoded'] = (
                        (category_stats['count'] * category_stats['mean'] + smoothing * global_mean) / 
                        (category_stats['count'] + smoothing)
                    )
                    
                    # Only apply target encoding if category has sufficient samples
                    reliable_categories = category_stats[category_stats['count'] >= min_samples].index
                    target_map = category_stats.loc[reliable_categories, 'target_encoded'].to_dict()
                    
                    # Fill others with global mean
                    processed_df[f'{col}_target_encoded'] = processed_df[col].map(target_map).fillna(global_mean)
                    encoders[f'{col}_target'] = target_map
                    
                    logger.info(f"Applied target encoding to {col} with {len(target_map)} categories")
        
        # Frequency encoding for categorical variables
        categorical_cols = processed_df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in [target_col, 'Churn']]
        
        for col in categorical_cols:
            if processed_df[col].nunique() > 1:  # Only if there's variation
                freq_map = processed_df[col].value_counts(normalize=True).to_dict()
                processed_df[f'{col}_frequency'] = processed_df[col].map(freq_map)
                encoders[f'{col}_frequency'] = freq_map
                logger.info(f"Applied frequency encoding to {col}")
        
        logger.info("Advanced categorical encoding completed")
        return processed_df, encoders
        
    except Exception as e:
        logger.error(f"Error in advanced categorical encoding: {str(e)}")
        raise


@task(name="handle_outliers")
def handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Detect and handle outliers in numerical features using IQR or Z-score method.
    
    Args:
        df: Input dataframe
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (1.5 for IQR, 3 for Z-score)
        columns: Specific columns to process (if None, process all numerical)
    
    Returns:
        pd.DataFrame: Dataframe with outliers handled
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Starting outlier handling using {method} method")
        
        processed_df = df.copy()
        
        # Determine columns to process
        if columns is None:
            numerical_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude binary and categorical encoded columns
            numerical_cols = [col for col in numerical_cols if not col.endswith('_binary') 
                             and not col.endswith('_encoded') and col not in ['Churn_binary']]
        else:
            numerical_cols = columns
        
        outlier_counts = {}
        
        for col in numerical_cols:
            if col in processed_df.columns:
                original_values = processed_df[col].copy()
                
                if method == "iqr":
                    Q1 = processed_df[col].quantile(0.25)
                    Q3 = processed_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    # Count outliers
                    outliers = (processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)
                    outlier_count = outliers.sum()
                    
                    # Cap outliers instead of removing them (preserves data)
                    processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
                    
                elif method == "zscore":
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(processed_df[col].dropna()))
                    outlier_mask = z_scores > threshold
                    outlier_count = outlier_mask.sum()
                    
                    # Replace extreme values with median
                    median_val = processed_df[col].median()
                    processed_df.loc[outlier_mask, col] = median_val
                
                if outlier_count > 0:
                    outlier_counts[col] = outlier_count
                    logger.info(f"Handled {outlier_count} outliers in {col}")
        
        total_outliers = sum(outlier_counts.values())
        logger.info(f"Outlier handling completed. Total outliers handled: {total_outliers}")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error handling outliers: {str(e)}")
        raise


@task(name="feature_selection")
def feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "mutual_info",
    k_features: int = 20
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select most important features using various selection methods.
    
    Args:
        X: Feature dataframe
        y: Target series
        method: Selection method ('mutual_info', 'chi2', 'rfe', 'variance')
        k_features: Number of features to select
    
    Returns:
        Tuple[pd.DataFrame, List[str]]: Selected features dataframe and feature names
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Starting feature selection using {method} method")
        
        # Ensure k_features doesn't exceed available features
        k_features = min(k_features, X.shape[1])
        
        if method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif, SelectKBest
            selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == "chi2":
            from sklearn.feature_selection import chi2, SelectKBest
            # Ensure non-negative values for chi2
            X_positive = X - X.min() + 1
            selector = SelectKBest(score_func=chi2, k=k_features)
            X_selected = selector.fit_transform(X_positive, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == "rfe":
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            selector = RFE(estimator, n_features_to_select=k_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == "variance":
            from sklearn.feature_selection import VarianceThreshold
            # Remove features with low variance
            selector = VarianceThreshold(threshold=0.01)
            X_variance = selector.fit_transform(X)
            remaining_features = X.columns[selector.get_support()].tolist()
            
            # If still too many features, use mutual info on remaining
            if len(remaining_features) > k_features:
                from sklearn.feature_selection import mutual_info_classif, SelectKBest
                X_remaining = X[remaining_features]
                selector2 = SelectKBest(score_func=mutual_info_classif, k=k_features)
                X_selected = selector2.fit_transform(X_remaining, y)
                selected_features = [remaining_features[i] for i in range(len(remaining_features)) 
                                   if selector2.get_support()[i]]
            else:
                selected_features = remaining_features
                X_selected = X[selected_features].values
        
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        logger.info(f"Feature selection completed. Selected {len(selected_features)} features from {X.shape[1]}")
        logger.info(f"Selected features: {selected_features[:10]}..." if len(selected_features) > 10 else f"Selected features: {selected_features}")
        
        return X_selected_df, selected_features
        
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        raise