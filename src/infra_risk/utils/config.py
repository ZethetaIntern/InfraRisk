"""
InfraRisk AI - Configuration Management
"""
import os
from pathlib import Path
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.infra_risk.schemas.project import AppConfig, DataSourceConfig, MLflowConfig


class Settings(BaseSettings):
    """Application settings from environment variables"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # Dashboard
    DASHBOARD_HOST: str = "0.0.0.0"
    DASHBOARD_PORT: int = 8501
    
    # Paths
    DATA_DIR: str = "./data"
    MODELS_DIR: str = "./models"
    OUTPUTS_DIR: str = "./outputs"
    CONFIG_DIR: str = "./config"
    
    # Google Earth Engine
    GEE_PROJECT_ID: Optional[str] = None
    GEE_CREDENTIALS_PATH: Optional[str] = None
    
    # Macro data APIs
    IMF_API_KEY: Optional[str] = None
    WORLD_BANK_API_KEY: Optional[str] = None
    OECD_API_KEY: Optional[str] = None
    BIS_API_KEY: Optional[str] = None
    
    # Financial data APIs
    BLOOMBERG_API_KEY: Optional[str] = None
    REFINITIV_API_KEY: Optional[str] = None
    SP_API_KEY: Optional[str] = None
    MOODYS_API_KEY: Optional[str] = None
    FITCH_API_KEY: Optional[str] = None
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "infra-risk"
    MLFLOW_ARTIFACT_LOCATION: Optional[str] = None
    
    # Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "infra_risk"
    POSTGRES_PASSWORD: str = "infra_risk"
    POSTGRES_DB: str = "infra_risk"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    
    # MinIO/S3
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "infra-risk-artifacts"
    MINIO_SECURE: bool = False
    
    # Model versions
    GEOSPATIAL_MODEL_VERSION: str = "siamese-resnet50-v1.0"
    DEMAND_MODEL_VERSION: str = "tft-v1.0"
    GNN_MODEL_VERSION: str = "gnn-contagion-v1.0"
    PINN_MODEL_VERSION: str = "pinn-carul-v1.0"
    LEGAL_MODEL_VERSION: str = "legal-bert-v1.0"
    CREDIT_MODEL_VERSION: str = "credit-ensemble-v1.0"
    SIMULATION_MODEL_VERSION: str = "rl-opponent-v1.0"
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}"
    
    def to_app_config(self) -> AppConfig:
        """Convert to AppConfig schema"""
        return AppConfig(
            environment=self.ENVIRONMENT,
            debug=self.DEBUG,
            log_level=self.LOG_LEVEL,
            api_host=self.API_HOST,
            api_port=self.API_PORT,
            api_workers=self.API_WORKERS,
            dashboard_host=self.DASHBOARD_HOST,
            dashboard_port=self.DASHBOARD_PORT,
            data_dir=self.DATA_DIR,
            models_dir=self.MODELS_DIR,
            outputs_dir=self.OUTPUTS_DIR,
            data_sources=DataSourceConfig(
                gee_project_id=self.GEE_PROJECT_ID,
                gee_credentials_path=self.GEE_CREDENTIALS_PATH,
                imf_api_key=self.IMF_API_KEY,
                world_bank_api_key=self.WORLD_BANK_API_KEY,
                oecd_api_key=self.OECD_API_KEY,
                bis_api_key=self.BIS_API_KEY,
                bloomberg_api_key=self.BLOOMBERG_API_KEY,
                refinitiv_api_key=self.REFINITIV_API_KEY,
                s_and_p_api_key=self.SP_API_KEY,
                moodys_api_key=self.MOODYS_API_KEY,
                fitch_api_key=self.FITCH_API_KEY,
            ),
            mlflow=MLflowConfig(
                tracking_uri=self.MLFLOW_TRACKING_URI,
                experiment_name=self.MLFLOW_EXPERIMENT_NAME,
                artifact_location=self.MLFLOW_ARTIFACT_LOCATION,
            ),
            geospatial_model_version=self.GEOSPATIAL_MODEL_VERSION,
            demand_model_version=self.DEMAND_MODEL_VERSION,
            gnn_model_version=self.GNN_MODEL_VERSION,
            pinn_model_version=self.PINN_MODEL_VERSION,
            legal_model_version=self.LEGAL_MODEL_VERSION,
            credit_model_version=self.CREDIT_MODEL_VERSION,
            simulation_model_version=self.SIMULATION_MODEL_VERSION,
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def get_app_config() -> AppConfig:
    """Get application configuration"""
    return get_settings().to_app_config()