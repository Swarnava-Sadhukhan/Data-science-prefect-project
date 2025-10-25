import logging
from datetime import timedelta
from pathlib import Path
import sys
from typing import Dict, Any, Tuple

# Import Prefect components
import pandas as pd
import numpy as np
from prefect import flow, task

# Import data pipeline tasks
from data_pipeline.ingestion import load_csv_data, validate_data_schema
from data_pipeline.preprocessing import (
    preprocess_telco_dataset, clean_data, handle_missing_values, 
    encode_categorical_features, scale_numerical_features,
    # Enhanced preprocessing functions
    telco_feature_engineering, advanced_categorical_encoding, 
    handle_outliers, feature_selection
)
from data_pipeline.transformation import split_train_test, exploratory_data_analysis

# Import ML pipeline tasks
from ml_models.training import train_sklearn_model, evaluate_model, save_model
from ml_models.validation import cross_validate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

@flow(name="ml-training-flow", description="Complete ML pipeline for Telco Customer Churn prediction")
def ml_training_flow(
    input_path: str = "data/train/telco_customer_churn.csv",
    output_dir: str = "data/processed",
    models_dir: str = "models",
    target_column: str = "Churn",
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Complete ML training pipeline with all preprocessing and training as Prefect tasks.
    
    This flow includes:
    1. Data ingestion and validation
    2. Telco-specific preprocessing  
    3. Data cleaning and transformation
    4. Feature engineering
    5. Model training (Random Forest & Logistic Regression)
    6. Model evaluation and saving
    
    All steps run as orchestrated Prefect tasks with full logging.
    """
    logger.info("🔄 Starting complete ML training pipeline")
    
    # Ensure directories exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Data Ingestion (Prefect tasks)
        logger.info("📊 Step 1: Data ingestion and validation")
        raw_data = load_csv_data(input_path)
        
        # Define required columns for Telco Customer Churn dataset
        required_columns = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ]
        
        validated_data = validate_data_schema(raw_data, required_columns)
        
        # Step 2: Telco-specific preprocessing (Prefect task)
        logger.info("🧹 Step 2: Telco-specific preprocessing")
        telco_preprocessed = preprocess_telco_dataset(validated_data)
        
        # Step 3: General preprocessing (Prefect tasks)
        logger.info("⚙️ Step 3: Data cleaning and preprocessing")
        cleaned_data = clean_data(
            telco_preprocessed,
            drop_duplicates=True,
            drop_columns=['customerID']  # Remove ID column as it's not useful for ML
        )
        data_with_handled_missing, imputers = handle_missing_values(cleaned_data)
        
        # Step 3.5: Enhanced preprocessing (NEW)
        logger.info("🧠 Step 3.5: Enhanced feature engineering and preprocessing")
        
        # Apply Telco-specific feature engineering
        feature_engineered = telco_feature_engineering(data_with_handled_missing)
        
        # Apply advanced categorical encoding
        advanced_encoded, advanced_encoders = advanced_categorical_encoding(
            feature_engineered, target_col='Churn_binary'
        )
        
        # Standard categorical encoding for remaining features
        encoded_data, encoders = encode_categorical_features(advanced_encoded)
        
        # Handle outliers in numerical features
        outlier_handled = handle_outliers(encoded_data, method="iqr", threshold=1.5)
        
        # Scale features but exclude target columns and binary features
        feature_cols_to_scale = [col for col in outlier_handled.select_dtypes(include=[np.number]).columns 
                                if col not in ['Churn_binary', 'Churn_No', 'Churn_Yes'] 
                                and not col.endswith('_binary')
                                and not col.startswith('is_')
                                and not col.startswith('has_')
                                and not col.startswith('risky_')
                                and not col.startswith('auto_')]
        scaled_data, scaler = scale_numerical_features(
            outlier_handled, 
            columns_to_scale=feature_cols_to_scale
        )
        
        # Step 3.75: Exploratory Data Analysis (NEW - Assignment Sub-Objective 1.4)
        logger.info("🔍 Step 3.75: Comprehensive Exploratory Data Analysis")
        eda_results = exploratory_data_analysis(
            scaled_data, 
            target_column="Churn_binary",
            output_dir=f"{output_dir}/eda_outputs"
        )
        logger.info(f"📊 EDA completed. Results saved to {output_dir}/eda_outputs/")
        
        # Step 4: Train-test split (Prefect task)
        logger.info("📊 Step 4: Train-test split")
        X_train, X_test, y_train, y_test = split_train_test(
            scaled_data, 
            target_column="Churn_binary",  # Use binary target for ML
            test_size=test_size, 
            random_state=random_state
        )
        
        # Step 4.5: Feature selection (NEW)
        logger.info("🎯 Step 4.5: Feature selection")
        X_train_selected, selected_features = feature_selection(
            X_train, y_train, 
            method="mutual_info",  # Use mutual information for feature selection
            k_features=min(25, X_train.shape[1])  # Select top 25 features or all if less
        )
        X_test_selected = X_test[selected_features]
        
        # Step 5: Model training (Prefect tasks)
        logger.info("🤖 Step 5: Model training with selected features")
        
        # Train Random Forest with enhanced features
        rf_model = train_sklearn_model(
            X_train_selected, y_train, 
            model_type="random_forest",
            problem_type="classification",
            hyperparameters={
                "n_estimators": 150,  # Increased for better performance
                "max_depth": 10,      # Prevent overfitting
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": random_state
            }
        )
        
        # Train Logistic Regression with enhanced features
        lr_model = train_sklearn_model(
            X_train_selected, y_train,
            model_type="logistic_regression",
            problem_type="classification", 
            hyperparameters={
                "random_state": random_state, 
                "max_iter": 2000,     # Increased for convergence
                "C": 0.1,             # Regularization for better generalization
                "class_weight": "balanced"  # Handle class imbalance
            }
        )
        
        # Step 6: Model evaluation (Prefect tasks)
        logger.info("📈 Step 6: Model evaluation with enhanced features")
        rf_metrics = evaluate_model(
            rf_model, X_test_selected, y_test, 
            problem_type="classification"
        )
        lr_metrics = evaluate_model(
            lr_model, X_test_selected, y_test, 
            problem_type="classification"
        )
        
        # Cross-validation with selected features
        rf_cv_scores = cross_validate_model(
            rf_model, X_train_selected, y_train, 
            cv_folds=5, 
            problem_type="classification"
        )
        lr_cv_scores = cross_validate_model(
            lr_model, X_train_selected, y_train, 
            cv_folds=5, 
            problem_type="classification"
        )
        
        # Step 7: Save models (Prefect tasks)
        logger.info("💾 Step 7: Saving models")
        rf_model_path = save_model(rf_model, f"{models_dir}/random_forest_model.joblib")
        lr_model_path = save_model(lr_model, f"{models_dir}/logistic_regression_model.joblib")
        
        # Compile enhanced results
        results = {
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "original_features": list(X_train.columns),
            "selected_features": selected_features,
            "num_original_features": X_train.shape[1],
            "num_selected_features": len(selected_features),
            "exploratory_data_analysis": {
                "eda_completed": True,
                "output_directory": f"{output_dir}/eda_outputs",
                "correlation_analysis": eda_results.get('correlation_analysis', {}),
                "dataset_overview": eda_results.get('dataset_overview', {}),
                "feature_importance": eda_results.get('feature_importance', {}),
                "report_files": ["eda_report.json", "eda_summary.md"]
            },
            "feature_engineering": {
                "telco_features_added": True,
                "advanced_encoding_applied": True,
                "outliers_handled": True,
                "feature_selection_method": "mutual_info"
            },
            "random_forest": {
                "metrics": rf_metrics,
                "cv_scores": rf_cv_scores,
                "model_path": str(rf_model_path)
            },
            "logistic_regression": {
                "metrics": lr_metrics, 
                "cv_scores": lr_cv_scores,
                "model_path": str(lr_model_path)
            }
        }
        
        logger.info("✅ ML training pipeline completed successfully!")
        logger.info(f"📊 EDA Report: {output_dir}/eda_outputs/eda_summary.md")
        logger.info(f"🧠 Features: {X_train.shape[1]} → {len(selected_features)} (selected)")
        logger.info(f"🎯 Random Forest Accuracy: {rf_metrics['accuracy']:.3f}")
        logger.info(f"🎯 Logistic Regression Accuracy: {lr_metrics['accuracy']:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Import required tasks
    from src.data_pipeline.ingestion import load_csv_data, validate_data_schema
    from src.data_pipeline.preprocessing import (
        preprocess_telco_dataset, clean_data, handle_missing_values, 
        encode_categorical_features, scale_numerical_features,
        # Enhanced preprocessing functions
        telco_feature_engineering, advanced_categorical_encoding,
        handle_outliers, feature_selection
    )
    from src.data_pipeline.transformation import split_train_test
    from src.ml_models.training import train_sklearn_model, evaluate_model, save_model
    from src.ml_models.validation import cross_validate_model
    
    try:
        logger.info("🚀 Creating Prefect deployment for ML training pipeline")
        
        # Use the new Prefect 3.x serve() method for local deployment with scheduling
        print("📦 Deploying Telco Customer Churn ML Pipeline...")
        print("🔄 Using Prefect 3.x flow.serve() for automatic scheduling")
        
        # Deploy with 2-minute interval scheduling
        ml_training_flow.serve(
            name="telco-churn-pipeline-2min",
            description="Telco Customer Churn ML Pipeline - Runs every 2 minutes with all preprocessing and training tasks",
            interval=timedelta(minutes=2),
            tags=["ml", "telco", "churn", "preprocessing", "training"]
        )
        
        logger.info("✅ Deployment successful!")
        logger.info("📅 Schedule: Every 2 minutes")
        logger.info("🎯 All preprocessing and training tasks will run as Prefect tasks")
        
        print("\n" + "="*60)
        print("🎉 PREFECT DEPLOYMENT SUCCESSFUL!")
        print("="*60)
        print("📋 WHAT HAPPENS NEXT:")
        print("   • Pipeline serves locally and runs every 2 minutes")
        print("   • All preprocessing steps run as Prefect tasks")
        print("   • All training steps run as Prefect tasks") 
        print("   • Full task logging and monitoring")
        print("\n🚀 FLOW IS NOW SERVING:")
        print("   • Automatic execution every 2 minutes")
        print("   • Manual trigger: Call ml_training_flow() directly")
        print("\n📊 MONITOR AT:")
        print("   • Prefect UI: http://localhost:4200")
        print("   • View all task executions and logs")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        print("💡 Make sure Prefect server is running: prefect server start")
        sys.exit(1)