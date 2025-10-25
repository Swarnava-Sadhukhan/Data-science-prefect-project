#!/usr/bin/env python3
"""
Script to deploy and run the complete Telco Customer Churn pipeline with Prefect.
This includes all preprocessing and training steps as scheduled Prefect tasks.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import logging

# Add src to Python path for absolute imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_prefect_server():
    """Check if Prefect server is running"""
    try:
        import requests
        response = requests.get("http://localhost:4200/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def wait_for_server(timeout=30):
    """Wait for Prefect server to be ready"""
    logger.info("Waiting for Prefect server to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if check_prefect_server():
            logger.info("✅ Prefect server is ready!")
            return True
        time.sleep(2)
    
    logger.error("❌ Prefect server did not start within timeout")
    return False

def setup_pipeline():
    """Setup and deploy the complete pipeline"""
    logger.info("🚀 Setting up Telco Customer Churn Pipeline...")
    
    # Check if dataset exists
    dataset_path = Path("data/train/telco_customer_churn.csv")
    if not dataset_path.exists():
        logger.info("📊 Creating Telco dataset...")
        os.system("python scripts/create_sample_data.py")
    
    # Configure Prefect to use local server
    logger.info("⚙️ Configuring Prefect client...")
    os.system("prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api")
    
    # Deploy the workflow by importing and running directly
    logger.info("🔧 Deploying the complete pipeline workflow...")
    try:
        from workflows.training_flow import ml_training_flow
        # Deploy with serve()
        ml_training_flow.serve(
            name="telco-churn-pipeline-2min",
            interval=120  # 2 minutes in seconds
        )
        logger.info("✅ Pipeline deployed successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to deploy pipeline: {e}")
        return False

def start_worker():
    """Start Prefect worker to execute scheduled tasks"""
    logger.info("👷 Starting Prefect worker for task execution...")
    logger.info("📝 The worker will execute all preprocessing and training tasks")
    
    # Start worker in background
    worker_cmd = "prefect worker start --pool default-agent-pool"
    logger.info(f"Running: {worker_cmd}")
    
    return subprocess.Popen(
        worker_cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def show_pipeline_info():
    """Show information about the pipeline tasks"""
    print("\n" + "="*70)
    print("🔄 TELCO CUSTOMER CHURN PIPELINE - PREFECT TASKS")
    print("="*70)
    print()
    print("📋 PREPROCESSING TASKS (Sub-Objective 1):")
    print("   ├── 🔍 load_csv_data")
    print("   ├── ✅ validate_data_schema") 
    print("   ├── 🧹 preprocess_telco_dataset (handles TotalCharges, SeniorCitizen)")
    print("   ├── 🚿 clean_data (remove duplicates, drop customerID)")
    print("   ├── 🔧 handle_missing_values (imputation)")
    print("   ├── 🏷️  encode_categorical_features (one-hot encoding)")
    print("   ├── 📊 scale_numerical_features (standardization)")
    print("   └── 🎯 split_train_test (70/30 split with stratification)")
    print()
    print("🤖 ML TRAINING TASKS (Sub-Objective 2):")
    print("   ├── 🌳 train_sklearn_model (Random Forest)")
    print("   ├── 🔄 train_sklearn_model (Logistic Regression)")
    print("   ├── 📈 evaluate_model (4+ metrics: accuracy, precision, recall, F1)")
    print("   ├── ✅ cross_validate_model (5-fold CV)")
    print("   └── 💾 save_model (model artifacts)")
    print()
    print("⏰ SCHEDULE: Every 2 minutes (Sub-Objective 1.5 - DataOps)")
    print("🔗 WORKFLOW: All tasks run as connected Prefect tasks")
    print("📊 MONITORING: View at http://localhost:4200")
    print()
    print("="*70)

def manual_trigger():
    """Manually trigger the pipeline"""
    logger.info("🎯 Manually triggering the pipeline...")
    result = os.system("prefect deployment run telco-churn-pipeline-2min")
    
    if result == 0:
        logger.info("✅ Pipeline triggered successfully!")
        logger.info("🔍 Check http://localhost:4200 to monitor execution")
    else:
        logger.error("❌ Failed to trigger pipeline")

def main():
    """Main function to set up and run the complete pipeline"""
    
    show_pipeline_info()
    
    print("\n🚀 STARTING TELCO CUSTOMER CHURN PIPELINE SETUP")
    print("="*50)
    
    # Check if Prefect server is running
    if not check_prefect_server():
        print("\n❌ Prefect server is not running!")
        print("Please start it first with: prefect server start")
        print("Then run this script again.")
        return
    
    # Setup pipeline
    if not setup_pipeline():
        logger.error("Failed to setup pipeline")
        return
    
    print("\n✅ PIPELINE SETUP COMPLETE!")
    print("\n📋 WHAT HAPPENS NEXT:")
    print("   1. All preprocessing and training steps are now Prefect tasks")
    print("   2. The pipeline runs automatically every 2 minutes")
    print("   3. Each run includes: data loading → preprocessing → training → evaluation")
    print("   4. All tasks are logged and monitored in Prefect UI")
    
    print("\n🎯 OPTIONS:")
    print("   [1] Start worker to enable scheduled execution")
    print("   [2] Trigger pipeline manually once")
    print("   [3] Show deployment info")
    print("   [q] Quit")
    
    while True:
        choice = input("\nEnter your choice (1/2/3/q): ").strip().lower()
        
        if choice == '1':
            logger.info("Starting worker for scheduled execution...")
            print("Worker starting... Press Ctrl+C to stop")
            worker_process = start_worker()
            try:
                worker_process.wait()
            except KeyboardInterrupt:
                logger.info("Stopping worker...")
                worker_process.terminate()
                break
                
        elif choice == '2':
            manual_trigger()
            
        elif choice == '3':
            os.system("prefect deployment ls")
            print("\n📊 Monitor at: http://localhost:4200")
            
        elif choice == 'q':
            print("👋 Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, 3, or q")

if __name__ == "__main__":
    main()