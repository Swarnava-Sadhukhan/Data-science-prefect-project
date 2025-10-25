"""
API layer for accessing application details.
Implements Sub-Objective 3: API Access to retrieve and display application information.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import json
from pathlib import Path
import os
import sys
import asyncio
from contextlib import asynccontextmanager

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    logger.info("Starting Data Science Pipeline API")
    yield
    # Shutdown
    logger.info("Shutting down Data Science Pipeline API")


app = FastAPI(
    title="Simplified Data Science Pipeline API",
    description="API for accessing four key application details without Prefect dependencies",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Data Science Pipeline API",
        "version": "1.0.0",
        "description": "API for accessing four key application details",
        "endpoints": [
            "/model-details",
            "/all-model-metrics",
            "/model-recommendation",
            "/latest-pipeline-status"
        ],
        "key_details_provided": [
            "model_details (information about both models)",
            "latest_pipeline_status (separate endpoint)",
            "model_metrics_comparison",
            "model_recommendation"
        ]
    }

@app.get("/model-details")
async def get_model_details_endpoint():
    """
    Get detailed information about both trained models
    Returns comprehensive model details and metadata
    """
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent
        
        # Application Detail: Model Details (both models)
        model_details = get_model_details(project_root)
        
        return {
            "application_name": "Data Science Pipeline API",
            "version": "1.0.0",
            "retrieved_at": datetime.now().isoformat(),
            
            # Core Application Details
            "model_details": model_details
        }
        
    except Exception as e:
        logger.error(f"Error retrieving application details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model details: {str(e)}")


@app.get("/all-model-metrics")
async def get_all_model_metrics():
    """
    Get metrics for all models to compare performance
    Shows accuracy, precision, recall, F1-score for each model
    """
    try:
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "models"
        
        if not models_dir.exists():
            return {"error": "No models directory found"}
        
        all_metrics = {}
        json_files = list(models_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    metrics_data = json.load(f)
                
                model_name = json_file.stem.replace('_model', '')
                eval_metrics = metrics_data.get("evaluation_metrics", {})
                
                all_metrics[model_name] = {
                    "accuracy": eval_metrics.get("accuracy"),
                    "precision": eval_metrics.get("precision"), 
                    "recall": eval_metrics.get("recall"),
                    "f1_score": eval_metrics.get("f1_score"),
                    "training_date": metrics_data.get("training_date"),
                    "model_type": metrics_data.get("model_type", model_name)
                }
                
            except Exception as e:
                logger.warning(f"Could not read {json_file}: {e}")
                all_metrics[json_file.stem] = {"error": str(e)}
        
        return {
            "all_model_metrics": all_metrics,
            "total_models": len(all_metrics),
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving all model metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving all model metrics: {str(e)}")


@app.get("/model-recommendation")
async def get_model_recommendation():
    """
    Get model recommendation based on classification performance
    Compares both models and recommends the best one based on accuracy
    """
    try:
        project_root = Path(__file__).parent.parent.parent
        result = get_all_model_metrics_with_recommendation(project_root)
        
        return {
            "model_comparison": result,
            "summary": {
                "total_models_compared": result.get("total_models", 0),
                "recommended_model": result.get("recommendation", {}).get("recommended_model", "None"),
                "recommendation_reason": result.get("recommendation", {}).get("reason", "No recommendation available"),
                "best_accuracy": result.get("recommendation", {}).get("accuracy", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting model recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model recommendation: {str(e)}")


@app.get("/latest-pipeline-status")
async def get_latest_pipeline_status_endpoint():
    """
    Get latest pipeline status and deployment information
    """
    try:
        project_root = Path(__file__).parent.parent.parent
        pipeline_status = await get_latest_pipeline_status(project_root)
        
        return {
            "latest_pipeline_status": pipeline_status,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving pipeline status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving pipeline status: {str(e)}")
    """Get the latest flow run name from Prefect"""
    try:
        from prefect.client.orchestration import get_client
        
        async with get_client() as client:
            recent_runs = await client.read_flow_runs(limit=10)
            
            if recent_runs:
                # Get the most recent run
                latest_run = recent_runs[0]
                return latest_run.name
            else:
                return f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                
    except Exception as e:
        logger.warning(f"Could not get latest flow run name from Prefect: {e}")
        # Fallback to timestamp-based name
        return f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


async def get_latest_pipeline_status(project_root):
    """Get latest pipeline execution status with deployment information"""
    try:
        models_dir = project_root / "models"
        data_dir = project_root / "data"
        
        # Check if pipeline components exist
        has_models = models_dir.exists() and len(list(models_dir.glob("*.joblib"))) > 0
        has_metrics = models_dir.exists() and len(list(models_dir.glob("*.json"))) > 0
        has_data = data_dir.exists()
        
        # Determine pipeline status
        if has_models and has_metrics:
            status = "COMPLETED"
            health = "healthy"
        elif has_models:
            status = "PARTIAL"
            health = "needs_metrics"
        else:
            status = "FAILED"
            health = "needs_training"
        
        # Get last activity timestamp
        all_files = []
        if models_dir.exists():
            all_files.extend(list(models_dir.glob("*.*")))
        if data_dir.exists():
            all_files.extend(list((data_dir / "processed").glob("*.csv")) if (data_dir / "processed").exists() else [])
        
        last_activity = None
        if all_files:
            latest_file = max(all_files, key=lambda x: x.stat().st_mtime)
            last_activity = datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
        
        # Get deployment information
        deployment_info = get_deployment_timestamp(project_root)
        
        return {
            "deployment_name": "telco-churn-ml-pipeline",
            "status": status,
            "health": health,
            "last_execution": last_activity or datetime.now().isoformat(),
            "components": {
                "models_available": has_models,
                "metrics_available": has_metrics,
                "data_available": has_data
            },
            "latest_deployment_timestamp": {
                "deployment_date": deployment_info.get("deployment_date") or datetime.now().isoformat(),
                "deployment_type": deployment_info.get("deployment_type", "file_based"),
                "last_deployed_component": deployment_info.get("last_deployed_component", "unknown"),
                "deployment_status": deployment_info.get("deployment_status", "model_ready"),
                "days_since_deployment": deployment_info.get("days_since_deployment", 0)
            }
        }
        
    except Exception as e:
        logger.warning(f"Could not get pipeline status: {e}")
        return {"status": "UNKNOWN", "error": str(e)}


def get_current_model_version(project_root):
    """Application Detail 2: Get current deployed model version information"""
    try:
        models_dir = project_root / "models"
        
        if not models_dir.exists():
            return {"version": "none", "error": "No models directory found"}
        
        model_files = list(models_dir.glob("*.joblib"))
        
        if not model_files:
            return {"version": "none", "error": "No model files found"}
        
        # Find the most recent model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        model_stats = latest_model.stat()
        
        # Create version identifier based on model name and timestamp
        model_name = latest_model.stem.replace('_model', '')
        timestamp = datetime.fromtimestamp(model_stats.st_mtime)
        version_id = f"{model_name}_v{timestamp.strftime('%Y%m%d_%H%M')}"
        
        return {
            "version": version_id,
            "model_name": model_name,
            "file_name": latest_model.name,
            "created_date": timestamp.isoformat(),
            "file_size_mb": round(model_stats.st_size / (1024*1024), 2),
            "total_models": len(model_files)
        }
        
    except Exception as e:
        logger.warning(f"Could not get model version: {e}")
        return {"version": "unknown", "error": str(e)}


def get_latest_model_metrics(project_root):
    """Application Detail 3: Get latest model performance metrics (accuracy, precision, recall, F1-score)"""
    try:
        models_dir = project_root / "models"
        
        if not models_dir.exists():
            return {"accuracy": None, "precision": None, "recall": None, "f1_score": None, "error": "No models directory"}
        
        # Get the current model version first
        current_model_info = get_current_model_version(project_root)
        current_model_name = current_model_info.get("model_name", "")
        
        json_files = list(models_dir.glob("*.json"))
        
        if not json_files:
            return {"accuracy": None, "precision": None, "recall": None, "f1_score": None, "error": "No metrics files found"}
        
        # Try to find metrics file matching current model
        target_metrics_file = None
        if current_model_name:
            target_file_name = f"{current_model_name}_model.json"
            target_metrics_file = models_dir / target_file_name
        
        # If target file exists, use it; otherwise use most recent
        if target_metrics_file and target_metrics_file.exists():
            latest_metrics_file = target_metrics_file
        else:
            # Fallback to most recent metrics file
            latest_metrics_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        eval_metrics = metrics_data.get("evaluation_metrics", {})
        
        # Extract the four required metrics
        return {
            "accuracy": eval_metrics.get("accuracy"),
            "precision": eval_metrics.get("precision"),
            "recall": eval_metrics.get("recall"),
            "f1_score": eval_metrics.get("f1_score"),
            "model_name": latest_metrics_file.stem.replace('_model', ''),
            "metrics_date": metrics_data.get("training_date"),
            "source_file": latest_metrics_file.name,
            "matches_current_model": latest_metrics_file.stem.replace('_model', '') == current_model_name
        }
        
    except Exception as e:
        logger.warning(f"Could not get model metrics: {e}")
        return {"accuracy": None, "precision": None, "recall": None, "f1_score": None, "error": str(e)}


def get_deployment_timestamp(project_root):
    """Application Detail 4: Get deployment timestamp information"""
    try:
        # Check for deployment-related files
        deployment_files = []
        
        # Check for model files (indicating deployment readiness)
        models_dir = project_root / "models"
        if models_dir.exists():
            deployment_files.extend(list(models_dir.glob("*.joblib")))
        
        # Check for API server file (current deployment)
        api_file = project_root / "src" / "api" / "api_server.py"
        if api_file.exists():
            deployment_files.append(api_file)
        
        # Check for configuration files
        config_file = project_root / "config" / "config.yaml"
        if config_file.exists():
            deployment_files.append(config_file)
        
        if not deployment_files:
            return {"deployment_date": None, "error": "No deployment files found"}
        
        # Find the most recent deployment activity
        latest_deployment = max(deployment_files, key=lambda x: x.stat().st_mtime)
        deployment_stats = latest_deployment.stat()
        deployment_time = datetime.fromtimestamp(deployment_stats.st_mtime)
        
        return {
            "deployment_date": deployment_time.isoformat(),
            "deployment_type": "file_based",
            "last_deployed_component": latest_deployment.name,
            "deployment_status": "active" if latest_deployment.name.endswith('.py') else "model_ready",
            "days_since_deployment": (datetime.now() - deployment_time).days
        }
        
    except Exception as e:
        logger.warning(f"Could not get deployment timestamp: {e}")
        return {"deployment_date": None, "error": str(e)}


def get_model_details(project_root):
    """Application Detail 2: Get details about both models (not just the latest)"""
    try:
        models_dir = project_root / "models"
        
        if not models_dir.exists():
            return {"error": "No models directory found", "models": {}, "total_models": 0}
        
        model_files = list(models_dir.glob("*.joblib"))
        
        if not model_files:
            return {"error": "No model files found", "models": {}, "total_models": 0}
        
        all_models = {}
        
        # Process each model file
        for model_file in model_files:
            model_stats = model_file.stat()
            model_name = model_file.stem.replace('_model', '')
            timestamp = datetime.fromtimestamp(model_stats.st_mtime)
            version_id = f"{model_name}_v{timestamp.strftime('%Y%m%d_%H%M')}"
            
            # Check for corresponding JSON metrics file
            json_file = model_file.with_suffix('.json')
            has_metrics = json_file.exists()
            
            model_info = {
                "version": version_id,
                "model_name": model_name,
                "file_name": model_file.name,
                "created_date": timestamp.isoformat(),
                "file_size_mb": round(model_stats.st_size / (1024*1024), 2),
                "has_metrics": has_metrics,
                "model_type": model_name.replace('_', ' ').title()
            }
            
            # Add basic metrics info if available
            if has_metrics:
                try:
                    with open(json_file, 'r') as f:
                        metrics_data = json.load(f)
                    
                    eval_metrics = metrics_data.get("evaluation_metrics", {})
                    model_info["accuracy"] = eval_metrics.get("accuracy")
                    model_info["training_date"] = metrics_data.get("training_date")
                    
                except Exception as e:
                    logger.warning(f"Could not read metrics from {json_file}: {e}")
                    model_info["metrics_error"] = str(e)
            
            all_models[model_name] = model_info
        
        # Find the most recent model (current active model)
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        current_model_name = latest_model_file.stem.replace('_model', '')
        
        return {
            "models": all_models,
            "total_models": len(model_files),
            "current_active_model": current_model_name,
            "models_with_metrics": len([m for m in all_models.values() if m.get("has_metrics", False)]),
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.warning(f"Could not get model details: {e}")
        return {"error": str(e), "models": {}, "total_models": 0}


def get_all_model_metrics_with_recommendation(project_root):
    """
    Application Detail 3: Get all model metrics and provide recommendation
    Shows accuracy, precision, recall, F1-score for both models
    Recommends best model based on accuracy for classification problems
    """
    try:
        models_dir = project_root / "models"
        
        if not models_dir.exists():
            return {
                "error": "No models directory found",
                "models": {},
                "recommendation": None
            }
        
        json_files = list(models_dir.glob("*.json"))
        
        if not json_files:
            return {
                "error": "No metrics files found", 
                "models": {},
                "recommendation": None
            }
        
        all_models = {}
        best_model = None
        best_accuracy = 0.0
        
        # Process each model's metrics
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    metrics_data = json.load(f)
                
                model_name = json_file.stem.replace('_model', '')
                eval_metrics = metrics_data.get("evaluation_metrics", {})
                
                # Extract the four required metrics
                model_metrics = {
                    "accuracy": eval_metrics.get("accuracy", 0),
                    "precision": eval_metrics.get("precision", 0),
                    "recall": eval_metrics.get("recall", 0),
                    "f1_score": eval_metrics.get("f1_score", 0),
                    "training_date": metrics_data.get("training_date"),
                    "model_type": metrics_data.get("model_type", model_name.title()),
                    "problem_type": metrics_data.get("problem_type", "classification")
                }
                
                all_models[model_name] = model_metrics
                
                # Track best model based on accuracy (for classification)
                current_accuracy = model_metrics["accuracy"] or 0
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_model = model_name
                    
            except Exception as e:
                logger.warning(f"Could not read {json_file}: {e}")
                all_models[json_file.stem] = {"error": str(e)}
        
        # Generate recommendation
        recommendation = None
        if best_model and best_accuracy > 0:
            recommendation = {
                "recommended_model": best_model,
                "reason": f"Highest accuracy for classification problem",
                "accuracy": best_accuracy,
                "criteria": "accuracy",
                "problem_type": "classification",
                "confidence": "high" if best_accuracy > 0.8 else "medium" if best_accuracy > 0.7 else "low"
            }
            
            # Add comparison details
            if len(all_models) > 1:
                other_models = [name for name in all_models.keys() if name != best_model and "error" not in all_models[name]]
                if other_models:
                    other_accuracies = [all_models[name].get("accuracy", 0) for name in other_models]
                    max_other_accuracy = max(other_accuracies) if other_accuracies else 0
                    improvement = ((best_accuracy - max_other_accuracy) / max_other_accuracy * 100) if max_other_accuracy > 0 else 0
                    
                    recommendation["improvement_over_others"] = f"{improvement:.2f}%" if improvement > 0 else "0%"
                    recommendation["comparison"] = {
                        name: {"accuracy": all_models[name].get("accuracy", 0)} 
                        for name in other_models
                    }
        
        return {
            "models": all_models,
            "recommendation": recommendation,
            "total_models": len([m for m in all_models.values() if "error" not in m]),
            "problem_type": "classification",
            "evaluation_criteria": "accuracy (primary metric for classification)",
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.warning(f"Could not get model metrics with recommendation: {e}")
        return {
            "error": str(e),
            "models": {},
            "recommendation": None
        }


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )