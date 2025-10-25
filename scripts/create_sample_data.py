"""
Create sample data for testing the data science pipeline.
This script creates a sample dataset based on the Telco Customer Churn dataset structure
from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import requests
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_telco_churn_dataset(n_samples=7043):
    """
    Create a sample dataset matching the Telco Customer Churn dataset structure.
    
    The original dataset from Kaggle has the following features:
    - customerID: Customer ID
    - gender: Whether the customer is male or female
    - SeniorCitizen: Whether the customer is a senior citizen or not (1, 0)
    - Partner: Whether the customer has a partner or not (Yes, No)
    - Dependents: Whether the customer has dependents or not (Yes, No)
    - tenure: Number of months the customer has stayed with the company
    - PhoneService: Whether the customer has a phone service or not (Yes, No)
    - MultipleLines: Whether the customer has multiple lines or not (Yes, No, No phone service)
    - InternetService: Customer's internet service provider (DSL, Fiber optic, No)
    - OnlineSecurity: Whether the customer has online security or not (Yes, No, No internet service)
    - OnlineBackup: Whether the customer has online backup or not (Yes, No, No internet service)
    - DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)
    - TechSupport: Whether the customer has tech support or not (Yes, No, No internet service)
    - StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)
    - StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service)
    - Contract: The contract term of the customer (Month-to-month, One year, Two year)
    - PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)
    - PaymentMethod: The customer's payment method (Electronic check, Mailed check, Bank transfer, Credit card)
    - MonthlyCharges: The amount charged to the customer monthly
    - TotalCharges: The total amount charged to the customer
    - Churn: Whether the customer churned or not (Yes, No)
    """
    logger.info(f"Creating Telco Customer Churn dataset with {n_samples} samples")
    
    np.random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f"{i:04d}-ABCDE" for i in range(1, n_samples + 1)]
    
    # Generate demographic features
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.5, 0.5])
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])  # ~16% senior citizens
    partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.52, 0.48])
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
    
    # Generate tenure (months with company)
    tenure = np.random.exponential(scale=20, size=n_samples).astype(int)
    tenure = np.clip(tenure, 0, 72)  # Cap at 72 months
    
    # Generate service features
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
    
    # Multiple lines depends on phone service
    multiple_lines = []
    for phone in phone_service:
        if phone == 'No':
            multiple_lines.append('No phone service')
        else:
            multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.42, 0.58]))
    multiple_lines = np.array(multiple_lines)
    
    # Internet service
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22])
    
    # Services that depend on internet
    def generate_internet_dependent_service(base_prob=0.5):
        service = []
        for internet in internet_service:
            if internet == 'No':
                service.append('No internet service')
            else:
                service.append(np.random.choice(['Yes', 'No'], p=[base_prob, 1-base_prob]))
        return np.array(service)
    
    online_security = generate_internet_dependent_service(0.29)
    online_backup = generate_internet_dependent_service(0.34)
    device_protection = generate_internet_dependent_service(0.34)
    tech_support = generate_internet_dependent_service(0.29)
    streaming_tv = generate_internet_dependent_service(0.38)
    streaming_movies = generate_internet_dependent_service(0.38)
    
    # Contract and billing
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                               n_samples, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
    payment_method = np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], n_samples, p=[0.33, 0.19, 0.22, 0.26])
    
    # Generate charges
    # Monthly charges based on services
    base_monthly_charge = 20
    monthly_charges = base_monthly_charge + np.random.normal(0, 5, n_samples)
    
    # Adjust based on services
    monthly_charges += (phone_service == 'Yes') * np.random.normal(10, 2, n_samples)
    monthly_charges += (internet_service == 'DSL') * np.random.normal(15, 3, n_samples)
    monthly_charges += (internet_service == 'Fiber optic') * np.random.normal(25, 5, n_samples)
    monthly_charges += (multiple_lines == 'Yes') * np.random.normal(5, 1, n_samples)
    monthly_charges += (streaming_tv == 'Yes') * np.random.normal(8, 2, n_samples)
    monthly_charges += (streaming_movies == 'Yes') * np.random.normal(8, 2, n_samples)
    
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)  # Realistic range
    monthly_charges = np.round(monthly_charges, 2)
    
    # Total charges based on tenure and monthly charges
    total_charges = monthly_charges * tenure + np.random.normal(0, tenure * 2, n_samples)
    total_charges = np.maximum(total_charges, monthly_charges)  # At least one month
    total_charges = np.round(total_charges, 2)
    
    # Generate churn with realistic business logic
    churn_probability = np.zeros(n_samples)
    
    # Base churn probability
    churn_probability += 0.15
    
    # Factors that increase churn
    churn_probability += (contract == 'Month-to-month') * 0.35
    churn_probability += (tenure < 6) * 0.25
    churn_probability += (senior_citizen == 1) * 0.15
    churn_probability += (payment_method == 'Electronic check') * 0.20
    churn_probability += (monthly_charges > 70) * 0.15
    churn_probability += (tech_support == 'No') * 0.10
    churn_probability += (online_security == 'No') * 0.10
    
    # Factors that decrease churn
    churn_probability -= (contract == 'Two year') * 0.25
    churn_probability -= (tenure > 24) * 0.20
    churn_probability -= (partner == 'Yes') * 0.10
    churn_probability -= (dependents == 'Yes') * 0.15
    
    # Add some randomness
    churn_probability += np.random.normal(0, 0.1, n_samples)
    churn_probability = np.clip(churn_probability, 0.05, 0.95)
    
    # Convert to binary churn
    churn_binary = (np.random.random(n_samples) < churn_probability).astype(int)
    churn = ['Yes' if c == 1 else 'No' for c in churn_binary]
    
    # Handle total charges for new customers (some have empty total charges in original)
    total_charges_str = []
    for i, (tenure_val, total_val) in enumerate(zip(tenure, total_charges)):
        if tenure_val == 0 and np.random.random() < 0.02:  # 2% have missing total charges
            total_charges_str.append(' ')  # Empty string like in original dataset
        else:
            total_charges_str.append(str(total_val))
    
    # Create DataFrame
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges_str,
        'Churn': churn
    })
    
    logger.info(f"Created Telco dataset with churn rate: {(np.array(churn) == 'Yes').mean():.3f}")
    return df


def create_regression_dataset(n_samples=1000, n_features=8, random_state=42):
    """Create a sample regression dataset"""
    np.random.seed(random_state)
    
    # Generate feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Add some polynomial features
    X[:, 1] = X[:, 0] ** 2 + np.random.randn(n_samples) * 0.1
    X[:, 2] = np.sin(X[:, 0]) + np.random.randn(n_samples) * 0.1
    
    # Generate target variable
    y = (
        2.5 * X[:, 0] + 
        -1.2 * X[:, 1] + 
        3.0 * X[:, 2] + 
        0.8 * X[:, 3] + 
        np.random.randn(n_samples) * 0.5
    )
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add categorical features
    df['category_A'] = np.random.choice(['level_1', 'level_2', 'level_3', 'level_4'], n_samples)
    df['category_B'] = np.random.choice(['status_active', 'status_inactive'], n_samples)
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=int(0.04 * n_samples), replace=False)
    df.loc[missing_indices, 'feature_3'] = np.nan
    
    return df


def create_business_case_dataset(dataset_type='classification', n_samples=1500):
    """
    Create a business case dataset for demonstration.
    This addresses Sub-Objective 1.1: Business Understanding
    """
    
    if dataset_type == 'classification':
        # Customer Churn Prediction Dataset
        logger.info("Creating Customer Churn Prediction dataset")
        
        np.random.seed(42)
        
        # Customer demographics and behavior
        data = {
            'age': np.random.normal(40, 12, n_samples).astype(int),
            'tenure_months': np.random.exponential(24, n_samples).astype(int),
            'monthly_charges': np.random.normal(65, 20, n_samples),
            'total_charges': np.random.normal(1800, 800, n_samples),
            'contract_length': np.random.choice(['month-to-month', '1-year', '2-year'], n_samples, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['electronic_check', 'mailed_check', 'bank_transfer', 'credit_card'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber_optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'phone_service': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
            'multiple_lines': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'online_security': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'tech_support': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'streaming_tv': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'paperless_billing': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        }
        
        df = pd.DataFrame(data)
        
        # Create churn target with business logic
        churn_probability = (
            0.3 * (df['contract_length'] == 'month-to-month') +
            0.2 * (df['monthly_charges'] > 80) +
            0.15 * (df['tenure_months'] < 12) +
            0.1 * (df['payment_method'] == 'electronic_check') +
            0.1 * (df['tech_support'] == 0) +
            0.1 * (df['online_security'] == 0) +
            np.random.normal(0, 0.1, n_samples)
        )
        
        df['churn'] = (churn_probability > 0.4).astype(int)
        df = df.rename(columns={'churn': 'target'})
        
    else:
        # House Price Prediction Dataset
        logger.info("Creating House Price Prediction dataset")
        
        np.random.seed(42)
        
        data = {
            'bedrooms': np.random.poisson(3, n_samples),
            'bathrooms': np.random.gamma(2, 0.8, n_samples),
            'sqft_living': np.random.normal(2100, 600, n_samples).astype(int),
            'sqft_lot': np.random.exponential(8000, n_samples).astype(int),
            'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.1, 0.1, 0.6, 0.1, 0.1]),
            'waterfront': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'view': np.random.choice(range(5), n_samples, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
            'condition': np.random.choice(range(1, 6), n_samples, p=[0.05, 0.1, 0.65, 0.15, 0.05]),
            'grade': np.random.choice(range(1, 14), n_samples),
            'yr_built': np.random.randint(1900, 2021, n_samples),
            'zipcode': np.random.choice(range(98001, 98200), n_samples),
            'neighborhood': np.random.choice(['downtown', 'suburban', 'rural', 'waterfront'], n_samples, p=[0.3, 0.5, 0.15, 0.05])
        }
        
        df = pd.DataFrame(data)
        
        # Create price target with realistic logic
        price = (
            150 * df['sqft_living'] +
            2 * df['sqft_lot'] +
            15000 * df['bedrooms'] +
            20000 * df['bathrooms'] +
            10000 * df['floors'] +
            100000 * df['waterfront'] +
            5000 * df['view'] +
            8000 * df['condition'] +
            3000 * df['grade'] +
            (2021 - df['yr_built']) * -500 +
            np.random.normal(50000, 20000, n_samples)
        )
        
        df['price'] = np.maximum(price, 100000)  # Minimum price
        df = df.rename(columns={'price': 'target'})
    
    # Add some missing values for realistic data processing challenges
    missing_cols = df.select_dtypes(include=[np.number]).columns[:3]
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df


def main():
    """Create and save Telco Customer Churn dataset"""
    logger.info("Starting Telco Customer Churn dataset creation...")
    
    # Create directories if they don't exist
    train_dir = Path("../data/train")
    test_dir = Path("../data/test")
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Telco Customer Churn dataset
    logger.info("Creating Telco Customer Churn dataset...")
    df = create_telco_churn_dataset(n_samples=7043)  # Original dataset size
    
    # Split into train and test
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    # Save datasets
    train_path = train_dir / "telco_customer_churn.csv"
    test_path = test_dir / "telco_customer_churn.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved training data to: {train_path}")
    logger.info(f"Saved test data to: {test_path}")
    logger.info(f"Training dataset shape: {train_df.shape}")
    logger.info(f"Test dataset shape: {test_df.shape}")
    logger.info(f"Churn distribution in training: {train_df['Churn'].value_counts().to_dict()}")
    
    # Display sample
    logger.info("\nSample of the training data:")
    logger.info(train_df.head().to_string())
    
    # Display dataset info
    logger.info(f"\nDataset Summary:")
    logger.info(f"Total customers: {len(df)}")
    logger.info(f"Features: {len(df.columns) - 1}")  # Excluding target
    logger.info(f"Churn rate: {(df['Churn'] == 'Yes').mean():.3f}")
    logger.info(f"Missing values in TotalCharges: {(df['TotalCharges'] == ' ').sum()}")
    
    # Display feature types
    categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                           'PaperlessBilling', 'PaymentMethod', 'Churn']
    numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges']
    logger.info(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")
    logger.info(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    logger.info(f"Special handling needed: TotalCharges (contains empty strings)")
    
    return train_df, test_df


if __name__ == "__main__":
    main()