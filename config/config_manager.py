"""
Configuration management for the data science application.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
from dataclasses import dataclass, asdict


@dataclass
class DataConfig:
    """Data processing configuration"""
    input_path: str = "data/train/telco_customer_churn.csv"
    raw_data_dir: str = "data"
    processed_data_dir: str = "data/processed"
    target_column: str = "Churn"
    test_size: float = 0.3
    random_state: int = 42
    
    # Data preprocessing
    drop_duplicates: bool = True
    missing_value_strategy: str = "simple"
    numeric_imputation: str = "mean"
    categorical_imputation: str = "most_frequent"
    scaling_method: str = "standard"
    encoding_strategy: str = "onehot"
    max_categories: int = 10


@dataclass
class MLConfig:
    """Machine learning configuration"""
    algorithms: list = None
    problem_type: str = "auto"  # auto, classification, regression
    models_dir: str = "models"
    cv_folds: int = 5
    
    # Model hyperparameters
    random_forest_params: Dict[str, Any] = None
    logistic_regression_params: Dict[str, Any] = None
    svm_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = ["random_forest", "logistic_regression"]
        
        if self.random_forest_params is None:
            self.random_forest_params = {
                "n_estimators": 100,
                "max_depth": None,
                "random_state": 42
            }
        
        if self.logistic_regression_params is None:
            self.logistic_regression_params = {
                "C": 1.0,
                "random_state": 42,
                "max_iter": 1000
            }
        
        if self.svm_params is None:
            self.svm_params = {
                "C": 1.0,
                "kernel": "rbf",
                "random_state": 42
            }


@dataclass
class PrefectConfig:
    """Prefect workflow configuration"""
    schedule_interval_minutes: int = 2
    work_queue_name: str = "default"
    log_level: str = "INFO"
    logs_dir: str = "logs"
    
    # Flow configurations
    enable_data_pipeline: bool = True
    enable_ml_pipeline: bool = True
    enable_monitoring: bool = True


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"
    title: str = "Data Science Pipeline API"
    version: str = "1.0.0"


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    enable_logging: bool = True
    log_file: str = "logs/pipeline.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_rotation: bool = True
    max_log_size: str = "10MB"
    backup_count: int = 5
    
    # Metrics
    enable_metrics: bool = True
    metrics_file: str = "logs/metrics.json"
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "accuracy_threshold": 0.8,
                "pipeline_failure_rate": 0.2,
                "data_quality_threshold": 0.9
            }


@dataclass
class AppConfig:
    """Main application configuration"""
    environment: str = "development"  # development, staging, production
    debug: bool = True
    
    data: DataConfig = None
    ml: MLConfig = None
    prefect: PrefectConfig = None
    api: APIConfig = None
    monitoring: MonitoringConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.ml is None:
            self.ml = MLConfig()
        if self.prefect is None:
            self.prefect = PrefectConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()


class ConfigManager:
    """Configuration manager for the application"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/config.yaml"
        self.config = self.load_config()
    
    def load_config(self) -> AppConfig:
        """Load configuration from file or environment variables"""
        config_dict = {}
        
        # Load from file if exists
        if Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    config_dict = yaml.safe_load(f) or {}
                elif self.config_file.endswith('.json'):
                    config_dict = json.load(f)
        
        # Override with environment variables
        config_dict = self._load_from_env(config_dict)
        
        # Create config object
        return self._dict_to_config(config_dict)
    
    def _load_from_env(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_mappings = {
            'ENVIRONMENT': ['environment'],
            'DEBUG': ['debug'],
            'DATA_PATH': ['data', 'input_path'],
            'TARGET_COLUMN': ['data', 'target_column'],
            'MODELS_DIR': ['ml', 'models_dir'],
            'API_HOST': ['api', 'host'],
            'API_PORT': ['api', 'port'],
            'SCHEDULE_INTERVAL': ['prefect', 'schedule_interval_minutes'],
            'LOG_LEVEL': ['prefect', 'log_level']
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert types
                if env_var in ['DEBUG']:
                    value = value.lower() in ('true', '1', 'yes')
                elif env_var in ['API_PORT', 'SCHEDULE_INTERVAL']:
                    value = int(value)
                
                # Set nested config value
                current = config_dict
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = value
        
        return config_dict
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to configuration objects"""
        # Create nested config objects
        data_config = DataConfig(**config_dict.get('data', {}))
        ml_config = MLConfig(**config_dict.get('ml', {}))
        prefect_config = PrefectConfig(**config_dict.get('prefect', {}))
        api_config = APIConfig(**config_dict.get('api', {}))
        monitoring_config = MonitoringConfig(**config_dict.get('monitoring', {}))
        
        # Create main config
        main_config_dict = {k: v for k, v in config_dict.items() 
                           if k not in ['data', 'ml', 'prefect', 'api', 'monitoring']}
        
        return AppConfig(
            data=data_config,
            ml=ml_config,
            prefect=prefect_config,
            api=api_config,
            monitoring=monitoring_config,
            **main_config_dict
        )
    
    def save_config(self, config_file: Optional[str] = None):
        """Save current configuration to file"""
        file_path = config_file or self.config_file
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self.config)
        
        with open(file_path, 'w') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif file_path.endswith('.json'):
                json.dump(config_dict, f, indent=2)
    
    def get_config(self) -> AppConfig:
        """Get the current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        env = self.config.environment
        
        env_configs = {
            'development': {
                'debug': True,
                'api_reload': True,
                'log_level': 'DEBUG'
            },
            'staging': {
                'debug': False,
                'api_reload': False,
                'log_level': 'INFO'
            },
            'production': {
                'debug': False,
                'api_reload': False,
                'log_level': 'WARNING'
            }
        }
        
        return env_configs.get(env, env_configs['development'])


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the global configuration"""
    return config_manager.get_config()


def setup_directories(config: AppConfig):
    """Setup required directories based on configuration"""
    directories = [
        config.data.raw_data_dir,
        config.data.processed_data_dir,
        config.ml.models_dir,
        config.prefect.logs_dir,
        Path(config.monitoring.log_file).parent,
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)