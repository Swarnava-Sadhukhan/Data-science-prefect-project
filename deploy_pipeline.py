#!/usr/bin/env python3
"""
Quick deployment script for the Telco Customer Churn pipeline.
Shows exactly when preprocessing and training tasks execute.
"""

import os
import sys
from pathlib import Path

def main():
    print("🔄 TELCO CUSTOMER CHURN PIPELINE DEPLOYMENT")
    print("="*50)
    
    # Step 1: Create dataset if it doesn't exist
    dataset_path = Path("data/train/telco_customer_churn.csv")
    if not dataset_path.exists():
        print("📊 Step 1: Creating Telco dataset...")
        result = os.system("python scripts/create_sample_data.py")
        if result == 0:
            print("✅ Dataset created!")
        else:
            print("❌ Failed to create dataset!")
            return
    else:
        print("✅ Step 1: Dataset already exists")
    
    # Step 2: Configure Prefect
    print("\n⚙️ Step 2: Configuring Prefect client...")
    os.system("prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api")
    print("✅ Prefect configured!")
    
    # Step 3: Test Prefect connection
    print("\n🔗 Step 3: Testing Prefect server connection...")
    result = os.system("prefect version")
    if result != 0:
        print("❌ Cannot connect to Prefect server!")
        print("💡 Make sure Prefect server is running: prefect server start")
        return
    
    # Step 4: Deploy the workflow
    print("\n🚀 Step 4: Deploying complete pipeline with all Prefect tasks...")
    print("This includes:")
    print("   • Data ingestion tasks")
    print("   • Telco preprocessing tasks") 
    print("   • Missing value handling tasks")
    print("   • Feature engineering tasks")
    print("   • ML training tasks")
    print("   • Model evaluation tasks")
    
    # Change to project directory to ensure proper imports
    os.chdir(Path(__file__).parent)
    
    result = os.system("python src/workflows/training_flow.py")
    
    if result == 0:
        print("\n✅ DEPLOYMENT SUCCESSFUL!")
        print("\n🎯 WHEN PREPROCESSING & TRAINING RUN:")
        print("   • Automatically every 2 minutes (scheduled)")
        print("   • All steps run as connected Prefect tasks")
        print("   • Each execution processes the full pipeline:")
        print("     1. Load Telco dataset")
        print("     2. Telco-specific preprocessing")
        print("     3. Handle missing values")
        print("     4. Encode categorical features")
        print("     5. Scale numerical features") 
        print("     6. Split train/test")
        print("     7. Train Random Forest")
        print("     8. Train Logistic Regression")
        print("     9. Evaluate models")
        print("    10. Save results")
        
        print("\n📋 TO START EXECUTION:")
        print("   Option 1 (Scheduled): prefect worker start --pool default-agent-pool")
        print("   Option 2 (Manual): prefect deployment run telco-churn-pipeline-2min")
        
        print("\n📊 MONITOR EXECUTION:")
        print("   • Prefect UI: http://localhost:4200")
        print("   • View real-time task execution")
        print("   • See logs for each preprocessing step")
        print("   • Monitor model training progress")
        
    else:
        print("\n❌ DEPLOYMENT FAILED!")
        print("Check the error messages above.")
        print("Common issues:")
        print("   • Prefect server not running: prefect server start")
        print("   • Missing dependencies: pip install -r requirements.txt")
        print("   • Import path issues: run from project root directory")

if __name__ == "__main__":
    main()